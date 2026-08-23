"""
duplicates_web.py

FastAPI application for the browser-based front end: the scan/group/score/
apply pipeline lives in duplicates_core.py, exposed here as a token-gated
HTTP API instead of a CLI-driven loop.

Importable module (unlike find_duplicates.py, whose CLI-focused top level
isn't meant to be imported) so tests can drive it directly via FastAPI's
TestClient. find_duplicates.py is the thin CLI entry point that calls
create_app() and hands the result to uvicorn.

Session model: one in-memory Session per process, seeded by the CLI args
and replaceable via POST /api/scan. A rescan builds a brand new set of
groups in a background thread and only swaps them into the live Session
once the scan completes successfully -- concurrent requests keep seeing the
previous (still consistent) state while a scan is in flight, and a failed
rescan leaves the previous state in place rather than half-updating it.
"""

import asyncio
import io
import json
import secrets
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image as PILImage

from duplicates_core import (
    DEFAULT_HASH_THRESHOLD,
    Group,
    METRIC_DESCRIPTIONS,
    METRIC_ROWS,
    METRIC_WEIGHTS,
    PREVIEW_MAX_SIDE,
    THUMBNAIL_FAILURE_COLOR,
    apply_pick,
    build_groups,
    humansize,
    pick_needs_reapply,
    unapply,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
COOKIE_NAME = "fd_token"

# The only formats a browser can't render natively -- everything else in
# duplicates_core.IMAGE_EXTS (jpg/png/webp/bmp/tiff) is served as-is via
# /api/full; these need transcoding to JPEG on the fly.
HEIC_EXTS = {".heic", ".heif"}

# The browser UI shows one candidate at a time at display scale (see the
# direction contract in static/index.html), so it needs a render well above
# PREVIEW_MAX_SIDE's 800px -- upscaling a 800px preview to fill a 1400px
# stage blurs away the exact sharpness difference the stage exists to show.
# PREVIEW_MAX_SIDE is used elsewhere (thumbnail size) and deliberately
# tuned, so this is a separate size rather than a bump of it.
STAGE_MAX_SIDE = 1600


@dataclass
class ScanParams:
    directory: Path
    threshold: int
    recursive: bool
    dest_dir: Path
    dry_run: bool


@dataclass
class Session:
    """All server-side state for one scan session. Guarded by `lock` for
    every read/write of groups/manifest/status/thumb_cache -- route handlers
    run on the asyncio event loop thread, but a scan's progress_callback and
    completion callback run from a background executor thread (see
    _launch_scan), so this can't rely on the single-threaded-event-loop
    assumption a plain asyncio app would get for free."""

    params: ScanParams
    groups: list[Group] = field(default_factory=list)
    manifest: list[dict] = field(default_factory=list)
    status: str = "idle"  # idle | scanning | ready | error
    error: str | None = None
    # Keyed (group, file, max_side): the switcher strip's 800px previews and
    # the stage's 1600px renders of the same file are both cached here, and a
    # (group, file) key alone would serve whichever was rendered first for
    # both sizes.
    image_cache: dict[tuple[int, int, int], bytes] = field(default_factory=dict)
    # The scan caches, keyed on path + mtime + size (see duplicates_core's
    # cached_hash/cached_result). Unlike image_cache above, these deliberately
    # SURVIVE a rescan and are never reset in on_done -- reusing them is the
    # entire reason the control panel's rescan button is fast on a large
    # library, since a rescan after confirming a few groups re-encounters
    # thousands of unchanged files. Don't "fix" the missing reset. They hold
    # no file handles and nothing on disk: process exit is the only eviction,
    # which is also what keeps them from ever going stale across a code
    # change. Growth is bounded by distinct paths seen this process (a few
    # tens of bytes each for hashes; the analyze cache only ever sees files
    # that landed in a group, not the whole library).
    hash_cache: dict[str, dict] = field(default_factory=dict)
    analyze_cache: dict[str, dict] = field(default_factory=dict)
    # Bumped on every successful scan swap. (i, j) indices get reused across
    # rescans for different images, and the browser's own HTTP cache doesn't
    # know that -- the frontend appends ?g=<generation> to thumb/full URLs
    # so a rescan's new photo at group 3 slot 0 doesn't render as the old
    # one still sitting in the browser's image cache under the same URL.
    generation: int = 0
    lock: Lock = field(default_factory=Lock)
    # Separate lock for progress: updated frequently from the scan's worker
    # thread and polled frequently by the SSE endpoint -- keeping it apart
    # from the main lock means a long-running /api/group/confirm move can't
    # stall progress-bar updates for an unrelated in-flight scan, and vice
    # versa.
    progress_lock: Lock = field(default_factory=Lock)
    progress: dict = field(default_factory=lambda: {"label": "", "done": 0, "total": 0})
    progress_seq: int = 0


# A scan runs here rather than in the loop's default executor: asyncio's
# shutdown joins the default executor's threads, so a Ctrl-C mid-scan would
# hang until the whole scan finished. Sized 1 -- only one scan runs at a time.
_scan_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="scan")


# Set from the signal handler in find_duplicates.main(). An open
# /api/progress stream would otherwise keep uvicorn's graceful shutdown
# waiting on it for the rest of an in-flight scan; polling this lets the
# stream end itself instead of being cancelled on a timeout.
shutting_down = Event()


def _launch_scan(session: Session, params: ScanParams, loop: asyncio.AbstractEventLoop) -> None:
    """Runs build_groups() in the default executor (a thread pool) so the
    event loop stays responsive to other requests while a scan is in
    flight, then swaps the result into *session* -- but only on success;
    a failed scan leaves the previous groups/manifest/status untouched
    except for the error message, so a bad rescan (e.g. a typo'd directory)
    doesn't wipe out a review already in progress."""

    def progress_cb(label: str, done: int, total: int) -> None:
        with session.progress_lock:
            session.progress = {"label": label, "done": done, "total": total}
            session.progress_seq += 1

    def run_scan() -> list[Group]:
        # Safe to hand these to the executor thread unlocked: /api/scan
        # rejects a second scan while one is running, so this is the only
        # writer, and no route handler reads them.
        return build_groups(
            params.directory, params.threshold, recursive=params.recursive,
            dest_dir=params.dest_dir, progress_callback=progress_cb,
            hash_cache=session.hash_cache, analyze_cache=session.analyze_cache,
        )

    def on_done(fut: Future) -> None:
        with session.lock:
            try:
                groups = fut.result()
            except Exception as exc:  # noqa: BLE001 -- surface any scan failure as session.error
                session.status = "error"
                session.error = str(exc)
                return
            session.params = params
            session.groups = groups
            session.manifest = []
            session.image_cache = {}
            session.generation += 1
            session.status = "ready"
            session.error = None

    with session.progress_lock:
        session.progress = {"label": "", "done": 0, "total": 0}
        session.progress_seq += 1
    future = loop.run_in_executor(_scan_executor, run_scan)
    future.add_done_callback(on_done)


def _display_path(session: Session, path: Path) -> str:
    """Filename alone is ambiguous under --recursive (two subdirectories
    can each hold an IMG_1234.jpg), so show the path relative to the scan
    root instead."""
    if session.params.recursive:
        try:
            return str(path.relative_to(session.params.directory))
        except ValueError:
            return path.name
    return path.name


def _group_summary(i: int, g: Group) -> dict:
    return {
        "index": i,
        "status": g.status,
        "file_count": len(g.paths),
        "current_pick": g.current_pick,
        "suggested_idx": g.suggested_idx,
        "is_close_call": g.is_close_call,
    }


def _require_not_scanning(session: Session) -> None:
    """A rescan's on_done swaps session.groups/manifest wholesale on
    success. A pick/confirm/skip that lands mid-scan (multi-tab, or a
    control-panel rescan fired while reviewing) would mutate/move files
    against groups that are about to be replaced -- the move itself would
    still be real and non-destructive (files land in dest_dir, never
    deleted), but the manifest entry recording it would be wiped by the
    swap, breaking the "manifest reflects filesystem state for the
    session" invariant unapply relies on. Caller must hold session.lock."""
    if session.status == "scanning":
        raise HTTPException(409, "a scan is in progress; try again once it finishes")


def _render_scaled_jpeg(path: Path, max_side: int, quality: int) -> bytes:
    """JPEG bytes of *path* fitted inside a *max_side* box, or a neutral gray
    placeholder of that size if the file can't be decoded -- same contract as
    duplicates_core.make_thumbnail, with the size as a parameter so the
    switcher strip (PREVIEW_MAX_SIDE) and the stage (STAGE_MAX_SIDE) share
    one code path instead of drifting apart."""
    try:
        img = PILImage.open(path)
        img = img.convert("RGB")
        img.thumbnail((max_side, max_side))
    except Exception:  # noqa: BLE001 -- an undecodable file must not 500 the review
        img = PILImage.new("RGB", (max_side, max_side), THUMBNAIL_FAILURE_COLOR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _group_detail(session: Session, i: int, g: Group) -> dict:
    return {
        **_group_summary(i, g),
        "paths": [_display_path(session, p) for p in g.paths],
        # Numeric siblings of the formatted METRIC_ROWS strings below. The
        # stage needs real numbers -- pixel dimensions to size its 1:1 zoom
        # layer, byte counts and scores for the candidate strip -- and
        # re-parsing the display strings would couple the frontend to their
        # formatting. Explicit int()/float(): these come off numpy-derived
        # analyze() results, and a numpy scalar in a JSON response is the bug
        # is_close_call's bool() cast exists to prevent.
        "dimensions": [[int(r["dimensions"][0]), int(r["dimensions"][1])] for r in g.results],
        "sizes": [int(r["file_size"]) for r in g.results],
        "size_labels": [humansize(r["file_size"]) for r in g.results],
        "scores": [float(r["quality_score"]) for r in g.results],
        # Only meaningful for a confirmed group: True once its current_pick
        # has diverged from what the manifest says is actually on disk, which
        # is what the frontend's "confirm again to apply it" line keys off.
        # (For a pending group there's no manifest entry at all, so this is
        # trivially True and the frontend ignores it -- same condition
        # confirm_group re-checks server-side before moving anything.)
        "needs_reapply": bool(pick_needs_reapply(session.manifest, i, g)),
        # direction/kind are structured alongside the display label so the
        # frontend never has to re-derive a row's meaning by pattern-matching
        # `label` text -- see MetricRow's docstring in duplicates_core.py.
        "metrics": [
            {"label": row.label, "values": [row.fn(r) for r in g.results], "direction": row.direction, "kind": row.kind}
            for row in METRIC_ROWS
        ],
    }


class ScanRequest(BaseModel):
    directory: str
    # 0-64 matches the CLI's _threshold_arg validation (max Hamming distance
    # over a 64-bit hash) -- Pydantic rejects out-of-range values with a 422
    # before this ever reaches build_groups.
    threshold: int = Field(default=DEFAULT_HASH_THRESHOLD, ge=0, le=64)
    recursive: bool = False
    dest: str | None = None
    dry_run: bool = False


class PickRequest(BaseModel):
    idx: int


def create_app(initial_params: ScanParams, token: str) -> FastAPI:
    session = Session(initial_params)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # /api/scan sets this before calling _launch_scan; the initial scan
        # needs the same, for two reasons. A client loading the page mid-scan
        # must see "scanning" (not the "idle" default) or it never opens the
        # SSE connection that tells it when the scan finishes. And the guards
        # keyed off this status are what stop /api/scan from launching a
        # second concurrent build_groups over the same caches while the
        # startup scan is still running, with both on_done callbacks racing
        # to swap groups in.
        session.status = "scanning"
        _launch_scan(session, session.params, asyncio.get_running_loop())
        yield

    app = FastAPI(lifespan=lifespan)
    app.state.token = token
    app.state.session = session  # exposed for tests; route handlers close over `session` directly

    def require_token(request: Request) -> str:
        """Accepts the token either as a query param (the initial tokened
        URL) or the fd_token cookie (set by GET / on first load) -- an <img
        src> can't carry an Authorization header, so image endpoints need
        the cookie path to work at all. secrets.compare_digest for a
        constant-time check even though the stakes here (LAN, single user)
        are low."""
        supplied = request.query_params.get("token") or request.cookies.get(COOKIE_NAME)
        if supplied is None or not secrets.compare_digest(supplied, app.state.token):
            raise HTTPException(status_code=401, detail="missing or invalid token")
        return supplied

    @app.get("/")
    async def index(token: str = Depends(require_token)) -> Response:
        resp = FileResponse(STATIC_DIR / "index.html")
        resp.set_cookie(COOKIE_NAME, token, httponly=True, samesite="lax")
        return resp

    @app.get("/api/state")
    async def get_state(_: str = Depends(require_token)) -> JSONResponse:
        with session.lock:
            return JSONResponse({
                "status": session.status,
                "error": session.error,
                "generation": session.generation,
                "params": {
                    "directory": str(session.params.directory),
                    "threshold": session.params.threshold,
                    "recursive": session.params.recursive,
                    "dest": str(session.params.dest_dir),
                    "dry_run": session.params.dry_run,
                },
                "groups": [_group_summary(i, g) for i, g in enumerate(session.groups)],
            })

    @app.get("/api/metrics-info")
    async def get_metrics_info(_: str = Depends(require_token)) -> JSONResponse:
        return JSONResponse({"weights": METRIC_WEIGHTS, "descriptions": METRIC_DESCRIPTIONS})

    @app.get("/api/group/{i}")
    async def get_group(i: int, _: str = Depends(require_token)) -> JSONResponse:
        with session.lock:
            if not (0 <= i < len(session.groups)):
                raise HTTPException(404, "no such group")
            return JSONResponse(_group_detail(session, i, session.groups[i]))

    def _cached_render(i: int, j: int, max_side: int, quality: int) -> Response:
        with session.lock:
            if not (0 <= i < len(session.groups)):
                raise HTTPException(404, "no such group")
            g = session.groups[i]
            if not (0 <= j < len(g.paths)):
                raise HTTPException(404, "no such file in group")
            path = g.paths[j]
            cached = session.image_cache.get((i, j, max_side))
        if cached is None:
            cached = _render_scaled_jpeg(path, max_side, quality)
            with session.lock:
                session.image_cache[(i, j, max_side)] = cached
        return Response(content=cached, media_type="image/jpeg")

    @app.get("/api/thumb/{i}/{j}")
    async def get_thumb(i: int, j: int, _: str = Depends(require_token)) -> Response:
        return _cached_render(i, j, PREVIEW_MAX_SIDE, 85)

    @app.get("/api/stage/{i}/{j}")
    async def get_stage(i: int, j: int, _: str = Depends(require_token)) -> Response:
        """The review stage's image: the same file as /api/thumb rendered
        large enough to fill it without upscaling (see STAGE_MAX_SIDE), at a
        higher JPEG quality since this is the render the keeper decision is
        actually made on."""
        return _cached_render(i, j, STAGE_MAX_SIDE, 92)

    @app.get("/api/full/{i}/{j}")
    async def get_full(i: int, j: int, _: str = Depends(require_token)) -> Response:
        with session.lock:
            if not (0 <= i < len(session.groups)):
                raise HTTPException(404, "no such group")
            g = session.groups[i]
            if not (0 <= j < len(g.paths)):
                raise HTTPException(404, "no such file in group")
            path = g.paths[j]
        if not path.exists():
            raise HTTPException(404, "file no longer exists on disk")
        if path.suffix.lower() in HEIC_EXTS:
            img = PILImage.open(path).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=92)
            return Response(content=buf.getvalue(), media_type="image/jpeg")
        return FileResponse(path)

    @app.post("/api/group/{i}/pick")
    async def pick_group(i: int, body: PickRequest, _: str = Depends(require_token)) -> JSONResponse:
        with session.lock:
            _require_not_scanning(session)
            if not (0 <= i < len(session.groups)):
                raise HTTPException(404, "no such group")
            g = session.groups[i]
            if not (0 <= body.idx < len(g.paths)):
                raise HTTPException(400, "idx out of range")
            g.current_pick = body.idx
            return JSONResponse(_group_detail(session, i, g))

    @app.post("/api/group/{i}/confirm")
    async def confirm_group(i: int, _: str = Depends(require_token)) -> JSONResponse:
        """Retry-safe confirm sequence built on the duplicates_core
        primitives: only re-move files if the pick actually diverged from
        what's on disk when already confirmed; unapply-then-reapply (a
        no-op unless a prior attempt partially failed) for pending/skipped
        groups."""
        with session.lock:
            _require_not_scanning(session)
            if not (0 <= i < len(session.groups)):
                raise HTTPException(404, "no such group")
            g = session.groups[i]
            p = session.params
            try:
                if g.status == "confirmed":
                    if pick_needs_reapply(session.manifest, i, g):
                        unapply(session.manifest, i)
                        apply_pick(
                            g, i, g.current_pick, p.dest_dir, p.dry_run, session.manifest,
                            recursive=p.recursive, scan_root=p.directory,
                        )
                elif g.status in ("pending", "skipped"):
                    unapply(session.manifest, i)
                    apply_pick(
                        g, i, g.current_pick, p.dest_dir, p.dry_run, session.manifest,
                        recursive=p.recursive, scan_root=p.directory,
                    )
            except Exception as exc:
                # apply_pick's own apply_group leaves group.status alone on
                # a raise (stays whatever it was) -- surface the error
                # rather than swallowing it so the client can show it.
                raise HTTPException(500, f"failed to apply group {i}: {exc}") from exc
            return JSONResponse(_group_detail(session, i, g))

    @app.post("/api/group/{i}/skip")
    async def skip_group(i: int, _: str = Depends(require_token)) -> JSONResponse:
        """pending/confirmed -> skipped (unapplying first if confirmed),
        skipped -> pending (toggle back)."""
        with session.lock:
            _require_not_scanning(session)
            if not (0 <= i < len(session.groups)):
                raise HTTPException(404, "no such group")
            g = session.groups[i]
            if g.status in ("pending", "confirmed"):
                unapply(session.manifest, i)
                g.status = "skipped"
            elif g.status == "skipped":
                g.status = "pending"
            return JSONResponse(_group_detail(session, i, g))

    @app.post("/api/scan")
    async def start_scan(req: ScanRequest, _: str = Depends(require_token)) -> JSONResponse:
        directory = Path(req.directory)
        if not directory.exists() or not directory.is_dir():
            raise HTTPException(400, f"'{req.directory}' is not a valid, existing directory")
        directory = directory.resolve()
        dest_dir = (Path(req.dest).resolve() if req.dest else directory / "_duplicates")

        with session.lock:
            if session.status == "scanning":
                raise HTTPException(409, "a scan is already running")
            session.status = "scanning"
            session.error = None

        new_params = ScanParams(
            directory=directory, threshold=req.threshold, recursive=req.recursive,
            dest_dir=dest_dir, dry_run=req.dry_run,
        )
        _launch_scan(session, new_params, asyncio.get_running_loop())
        return JSONResponse({"status": "scanning"})

    @app.get("/api/progress")
    async def progress_stream(request: Request, _: str = Depends(require_token)) -> StreamingResponse:
        """Polling-based SSE: simpler than wiring cross-thread asyncio
        notification for what's fundamentally a low-frequency progress
        counter (see _launch_scan's comment on why progress updates happen
        on a worker thread, not the event loop). Closes the stream once the
        scan reaches a terminal status so a client doesn't hold a
        connection open forever after a scan finishes."""

        async def event_gen():
            last_seq = -1
            while True:
                if shutting_down.is_set() or await request.is_disconnected():
                    break
                with session.progress_lock:
                    seq = session.progress_seq
                    progress = dict(session.progress)
                status = session.status
                if seq != last_seq or status != "scanning":
                    last_seq = seq
                    yield f"data: {json.dumps({'status': status, **progress})}\n\n"
                    if status != "scanning":
                        break
                await asyncio.sleep(0.2)

        return StreamingResponse(event_gen(), media_type="text/event-stream")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
