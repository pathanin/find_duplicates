"""
duplicates_web.py

FastAPI application for the browser-based front end: same scan/group/score/
apply pipeline as the Textual TUI (find_duplicates.py), all imported from
duplicates_core.py, with a token-gated HTTP API instead of a terminal UI.

Importable module (unlike find_duplicates-web.py, whose hyphenated filename
can't be `import`ed) so tests can drive it directly via FastAPI's
TestClient. find_duplicates-web.py is the thin CLI entry point that calls
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
from concurrent.futures import Future
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image as PILImage

from duplicates_core import (
    DEFAULT_HASH_THRESHOLD,
    Group,
    METRIC_DESCRIPTIONS,
    METRIC_ROWS,
    METRIC_WEIGHTS,
    apply_pick,
    build_groups,
    humansize,
    make_thumbnail,
    pick_needs_reapply,
    unapply,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
COOKIE_NAME = "fd_token"

# The only formats a browser can't render natively -- everything else in
# duplicates_core.IMAGE_EXTS (jpg/png/webp/bmp/tiff) is served as-is via
# /api/full; these need transcoding to JPEG on the fly.
HEIC_EXTS = {".heic", ".heif"}


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
    thumb_cache: dict[tuple[int, int], bytes] = field(default_factory=dict)
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
        return build_groups(
            params.directory, params.threshold, recursive=params.recursive,
            dest_dir=params.dest_dir, progress_callback=progress_cb,
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
            session.thumb_cache = {}
            session.generation += 1
            session.status = "ready"
            session.error = None

    with session.progress_lock:
        session.progress = {"label": "", "done": 0, "total": 0}
        session.progress_seq += 1
    future = loop.run_in_executor(None, run_scan)
    future.add_done_callback(on_done)


def _display_path(session: Session, path: Path) -> str:
    """Filename alone is ambiguous under --recursive (two subdirectories
    can each hold an IMG_1234.jpg) -- mirrors DuplicateReviewApp._display_path."""
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


def _group_detail(session: Session, i: int, g: Group) -> dict:
    return {
        **_group_summary(i, g),
        "paths": [_display_path(session, p) for p in g.paths],
        "metrics": [{"label": label, "values": [fn(r) for r in g.results]} for label, fn in METRIC_ROWS],
    }


class ScanRequest(BaseModel):
    directory: str
    threshold: int = DEFAULT_HASH_THRESHOLD
    recursive: bool = False
    dest: str | None = None
    dry_run: bool = False


class PickRequest(BaseModel):
    idx: int


def create_app(initial_params: ScanParams, token: str) -> FastAPI:
    session = Session(initial_params)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
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

    @app.get("/api/thumb/{i}/{j}")
    async def get_thumb(i: int, j: int, _: str = Depends(require_token)) -> Response:
        with session.lock:
            if not (0 <= i < len(session.groups)):
                raise HTTPException(404, "no such group")
            g = session.groups[i]
            if not (0 <= j < len(g.paths)):
                raise HTTPException(404, "no such file in group")
            path = g.paths[j]
            cached = session.thumb_cache.get((i, j))
        if cached is None:
            img = make_thumbnail(path)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            cached = buf.getvalue()
            with session.lock:
                session.thumb_cache[(i, j)] = cached
        return Response(content=cached, media_type="image/jpeg")

    @app.get("/api/full/{i}/{j}")
    async def get_full(i: int, j: int, _: str = Depends(require_token)) -> Response:
        with session.lock:
            if not (0 <= i < len(session.groups)):
                raise HTTPException(404, "no such group")
            g = session.groups[i]
            if not (0 <= j < len(g.paths)):
                raise HTTPException(404, "no such file in group")
            path = g.paths[j]
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
        """Mirrors DuplicateReviewApp.action_confirm's retry-safe sequence,
        via the same duplicates_core primitives: only re-move files if the
        pick actually diverged from what's on disk when already confirmed;
        unapply-then-reapply (a no-op unless a prior attempt partially
        failed) for pending/skipped groups."""
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
                # a raise (stays whatever it was), matching the TUI's
                # "stays pending on failure" invariant -- surface the error
                # rather than swallowing it so the client can show it.
                raise HTTPException(500, f"failed to apply group {i}: {exc}") from exc
            return JSONResponse(_group_detail(session, i, g))

    @app.post("/api/group/{i}/skip")
    async def skip_group(i: int, _: str = Depends(require_token)) -> JSONResponse:
        """Mirrors DuplicateReviewApp.action_skip: pending/confirmed -> skipped
        (unapplying first if confirmed), skipped -> pending (toggle back)."""
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
                if await request.is_disconnected():
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
