# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single-purpose tool: scan a directory for near-duplicate images (the same photo at different sizes/quality), then pick which one to keep from a browser page — LAN-capable, so a library on a NAS or headless box can be reviewed from another machine. One front end (the web UI), plus a plain-pip installer. No package, no `requirements.txt`, no test framework.

## Running

```bash
python3 find_duplicates.py [directory] [--threshold N] [--dest DIR] [--recursive] [--auto] [--dry-run] [--host H] [--port N] [--no-browser]
python3 find_duplicates.py --set-typesafe-key             # prompt for the optional filename-hint key, save it, exit
python3 compare_image_quality.py imageA.jpg imageB.jpg   # standalone 2-image comparison
```

Without `--auto` this prints a tokened URL and runs until Ctrl-C — it does *not* exit when a review finishes. Rescans happen from the page's own control panel (`POST /api/scan`), not by restarting the process. `--auto` keeps each group's top-scored file and never starts the web server.

Runtime deps: `numpy opencv-python-headless pillow pillow-heif fastapi uvicorn`. Install via `./install.sh`; **add any new dependency to that script's `pip install` block**, since there is no other manifest.

## Tests

Each test file is a standalone script with its own `main()` that asserts, prints `ok` lines, and exits non-zero on failure. Run them individually:

```bash
python3 tests/test_auto_mode.py
python3 tests/test_claude_md_test_list_sync.py
python3 tests/test_confirm_hash.py
python3 tests/test_effective_resolution_downsampling.py
python3 tests/test_fast_scan.py
python3 tests/test_group_ordering.py
python3 tests/test_heic_support.py
python3 tests/test_help_and_labels.py
python3 tests/test_name_hint.py
python3 tests/test_optional_metrics.py
python3 tests/test_quit.py
python3 tests/test_recursive_scan.py
python3 tests/test_scan_progress.py
python3 tests/test_score_group.py
python3 tests/test_shutdown.py
python3 tests/test_streaming_scan.py
python3 tests/test_unapply_crash_safety.py
python3 tests/test_vectorized_sweep.py
python3 tests/test_web_api.py
python3 tests/test_web_progress.py
```

`test_claude_md_test_list_sync.py` machine-checks that list against `tests/*.py` in both directions. It finds the list by regex — literal `## Tests` heading, **first** ```` ```bash ```` fence after it, `tests/`-prefixed paths. Adding a test file without a line here fails that check; so does putting another bash fence between the heading and the list.

Tests reach the modules via `sys.path.insert(0, ...parent.parent)` — there is no install step. `test_web_api.py` additionally needs `httpx < 0.28` (test-only; `install.sh` does not install it): FastAPI's `TestClient` constructs its client with the `app=` shortcut httpx 0.28 removed, and the resulting `TypeError: unexpected keyword argument 'app'` is a version mismatch, not a bug.

Many tests exist to lock in one specific past bug. **Read a test's docstring before changing the code it covers** — it usually names a failure mode the assertion alone doesn't reveal.

## Architecture

Five modules, layered so the bottom two never know about the web:

- **`compare_image_quality.py`** — per-image quality metrics (`analyze`): laplacian sharpness, FFT-based `effective_resolution` (resists fake upscaling), noise, blockiness. Also runs standalone on two files. `brisque`/`niqe` are optional imports that stay unresolved by design, and each **latches after its first failure** (`_brisque_unavailable`/`_niqe_unavailable`) — a missing package is cheap to retry, but `brisque` 0.2.0 computes its whole feature set before dying on modern numpy, which measured 418 ms per image (analyze 25 ms → 443 ms) to return `None` every time. Don't remove the latch, and don't add either package to `install.sh`: `brisque` costs ~40 MB across 7 packages for a metric that currently never returns a number, and `pyiqa` pulls 61 packages including `transformers`, `tensorboard` and an `opencv-python` that conflicts with the project's `opencv-python-headless`. `tests/test_optional_metrics.py` covers the latch.
- **`duplicates_core.py`** — the whole scan/score/move pipeline. `find_images` → `group_duplicates` (perceptual hash + `UnionFind`) → `analyze_paths` → `score_group` → `build_groups`, returning `list[Group]`. Applying decisions: `apply_group`, `apply_pick`, `unapply`, `auto_apply_groups`.
- **`duplicates_web.py`** — FastAPI app (`create_app`) plus the `Session` dataclass holding all server-side state. Routes: `/api/state`, `/api/group/{i}` and its `pick`/`confirm`/`skip` posts, `/api/thumb|stage|full/{i}/{j}`, `/api/scan`, `/api/progress` (SSE), `/api/metrics-info`, `/api/group/{i}/name-hint`, `/api/quit`, and a token-gated `/static/{path}`.
- **`find_duplicates.py`** — CLI entry point, `--auto` path, signal handling, uvicorn startup.
- **`name_hint.py`** — asks TypeSafe's Jev which duplicate's *filename* reads as the original, and whether the group is one photo at all. Stdlib-only HTTP, advisory, served by `/api/group/{i}/name-hint` and shown in the ledger note. See "Name hint" below.

Front end is vanilla JS in `static/app.js` (no build step, no framework), organized in commented sections: state/API, queue sidebar, stage, switcher strip, ledger, decision bar.

### Scan lifecycle

`_launch_scan` runs `build_groups` on `_scan_executor` (a dedicated 1-worker pool). Progress, group and completion callbacks fire on that worker thread and marshal back into `Session` under its locks. `session.status` moves `idle → scanning → ready|error`, and the `scanning` status is what stops a second concurrent scan — `/api/scan`'s own check, which is separate from `_require_not_scanning` and must stay that way.

The **startup scan streams** (`_launch_scan(..., stream=True)`, `Session.streaming`): `build_groups`' `group_callback` appends each finished group into the live session, so review begins on group 1 while the rest of the library is still being analyzed. `params` and `generation` are set up front and `on_done` swaps nothing, so an index handed out mid-scan keeps addressing the same group and a mid-scan confirm's manifest entry survives. `_require_not_scanning` is exempt while `streaming` (the pick/confirm/skip routes only). A **rescan never streams** — its `on_done` replaces groups and manifest wholesale, which is exactly what the guard exists to protect. `/api/state`'s `streaming` flag is how the frontend knows a "scanning" status still means the groups below are reviewable.

Publication order is `raw_groups` order, not completion order, and a group is scored, permuted best-first and filtered (`< 2` valid members) *before* it is ever handed out. Both matter: `tests/test_streaming_scan.py` locks them.

### Name hint

`name_hint.py` reads what the pixel metrics structurally cannot: the filenames. Two questions per group, one TypeSafe call — a Choice over the group's paths ("which reads as the original") and a Noul ("one photo stored twice, or separate shots"). The Noul is the counterweight to `CONFIRM_HASH_THRESHOLD`'s recall tuning: a burst series shares a filename stem but isn't a duplicate.

It is **advisory and must stay that way**. Nothing it returns feeds `quality_score`, `suggested_idx`, `score_group` or the file moves, and `--auto` never calls it — a wrong hint has to cost one glance in the ledger, never a moved file. It is also optional in the brisque/niqe sense: missing `TYPESAFE_API_KEY`, an unreachable API or a malformed answer all return `None` and the app behaves exactly as before (`tests/test_name_hint.py`).

The ledger shows the pick hint only when it disagrees with the current pick *and* `confidence >= 0.6`. A group merged from two name families (`tests/Test-image` group 1) answers near 0.5 — the names offer no winner, and a flat sentence there would read as certainty the model didn't claim.

The key comes from `$TYPESAFE_API_KEY`, else `~/.config/find_duplicates/typesafe-key` (`load_key`, env var wins). `--set-typesafe-key` prompts via `getpass` and writes that file 0600 — prompted, not a flag value, since an argument is visible in `ps` and lands in shell history, and this runs on boxes reached over SSH where an env var doesn't survive the next login. `tests/test_name_hint.py` redirects `KEY_PATH` before the no-key cases, or the suite's result would depend on whether whoever runs it has saved a real key.

Its own route, not a field on `/api/group/{i}`: the call is blocking third-party HTTP and the detail response is what every keypress waits on. The frontend fetches it after the group is on screen and `asyncio.to_thread` keeps it off the event loop, with `session.lock` released first. The cache is keyed by the paths tuple, so it survives a rescan like `hash_cache`, and only successes are stored — caching a `None` would pin one transient failure for the life of the process.

Deliberately no `METRIC_WEIGHTS` entry. That would drag `METRIC_DESCRIPTIONS`/`METRIC_ROWS` and the help sheet along and route the judgment straight into `suggested_idx`, which is the destructive path.

### Grouping is two-stage

`phash_pair` returns a 64-bit DCT hash and a 256-bit confirmation hash. The 64-bit hash (`DEFAULT_HASH_THRESHOLD`, default 10/64) *proposes* a pair; the 256-bit half (`CONFIRM_HASH_THRESHOLD`) *confirms* it. The 64-bit hash alone cannot tell a re-export from a different frame of the same scene.

`CONFIRM_HASH_THRESHOLD` is tuned for **recall**: a false positive costs one keypress in the review UI, a false negative is never surfaced at all. Some near-identical frames group on purpose. Tightening it drops real duplicates before it stops those. `tests/Test-image/` will not tell you if you went too far — every known pair there sits at distance 0, nowhere near the tail the constant is set against. The tail cases are aspect recrops (one artwork exported for two screen sizes) and heavy downscales; `tests/test_confirm_hash.py` is what actually covers them.

### Group ordering

`build_groups` reorders each group best-first so the suggested file is always index 0 — the reviewer's eye shouldn't hunt for a ★ in a different position each group. Both `paths` and `results` are permuted together, so every downstream index (`current_pick`, the manifest, `/api/thumb`'s `j`, digit-key shortcuts) stays consistent; there is no "original index" to translate back to. The sort key is `(-quality_score, len(name), name)`. The tie-break matters more than it looks: byte-identical copies score *exactly* equal, so on the commonest duplicate of all it is the whole decision. Filename order used to lose it — `" 2.jpg"` and `" copy.jpg"` both sort ahead of `".jpg"` (space is 0x20, period is 0x2e), so the tool suggested keeping the copy and `--auto` moved the original into `_duplicates/`. Shorter name wins now, because every convention for naming a derived copy *appends* to the original (`" copy"`, `" 2"`, `" (1)"`, `"-edited"`, `"_v2"`). The full name comes last so the order stays total and identical copies don't shuffle between scans. The length key only ever runs on an exact tie; when the metrics differ at all they still decide alone. `tests/test_group_ordering.py` locks both halves.

## Traps

- `duplicates_core.py` must stay importable without the web stack — never import FastAPI/uvicorn into it.
- **`tests/Test-image/` cannot validate a *scoring* change.** All 42 files are 21 byte-identical pairs, so every group's top two entries score exactly equal; `suggested_idx` is pinned to 0 by the name tie-break over that tie, and `is_close_call` is unconditionally `True` in all 11 groups. "No suggested pick changed on Test-image" is therefore true of *any* edit to `analyze`/`load_gray`/`METRIC_WEIGHTS` and proves nothing. To test a scoring change you have to build groups of genuinely *different* files (re-encode at another quality, downscale, fake-upscale) and check `suggested_idx` across them. `score_group` min-max normalizes within a group, so a sub-1% metric drift that flips the sign of a near-tie becomes a full 0→1 swing — and `--auto` moves files on `suggested_idx`. Same blindness the `CONFIRM_HASH_THRESHOLD` note above describes, for a different reason.
- The hash/scoring constants are empirically tuned, not arbitrary: `DEFAULT_HASH_THRESHOLD`, `CONFIRM_HASH_THRESHOLD`, `CLOSE_CALL_MARGIN`, `MIN_REDUCED_DECODE_SIDE`, `METRIC_WEIGHTS`. Re-verify changes against the real photos in `tests/Test-image/`, not just unit tests.
- `load_hash_gray`'s reduced-decode and full-decode paths must agree on the **64-bit** hash bits — check `MIN_REDUCED_DECODE_SIDE` before touching either. They do *not* agree on the 256-bit half, which reaches into mid frequencies where the decode paths genuinely differ. That drift is content-dependent (single digits on real photos, tens of bits on synthetic noise), expected, and absorbed by `CONFIRM_HASH_THRESHOLD`. Don't chase it as a bug.
- `METRIC_WEIGHTS`, `METRIC_DESCRIPTIONS`, and `METRIC_ROWS` get entries added and removed together; the UI's help sheet renders off the first.
- Moving files is the only genuinely destructive path (`apply_group`, `_compute_dest`, `apply_pick`, `unapply`, `auto_apply_groups`). Non-kept files are **moved to `_duplicates/`, never deleted** — preserve that invariant. The manifest is in-memory only; recovery after process exit is a manual move back out.
- `apply_group` never sets `group.status` — the caller owns that, including the "stays pending on failure" invariant.
- `auto_apply_groups` must read `file_size` *before* the move; the source path is gone once `apply_group` returns.
- Any numpy-derived value reaching the JSON API needs an explicit `bool()`/`float()`/`int()` cast **where it's computed**. `numpy.bool_` doesn't subclass `bool` and isn't JSON-serializable (this bit `build_groups`'s `is_close_call`).
- `compare_image_quality.load_gray` builds one shared **float32** buffer, and the metrics fuse multi-pass numpy into single cv2 calls (`cv2.absdiff`, `cv2.norm(..., NORM_L1)`); `effective_resolution` uses `cv2.dft`, not `np.fft.fft2`. Analyze is the per-image bottleneck — "simplifying" these back to plain numpy halves throughput.
- `duplicates_web.py` imports core functions **by name**, so a test patching one for a route handler must patch `duplicates_web.X`, not `duplicates_core.X` — the latter silently does nothing. Names called bare inside `duplicates_core.py` (`load_hash_gray`, `ThreadPoolExecutor`) are patched there instead.
- `Session` is guarded by a plain `threading.Lock`, not `asyncio.Lock`: scan callbacks run on an executor thread, not the event loop. `progress` has its own separate lock so a slow file move can't stall progress updates. Keep the `scanning` guards (`_require_not_scanning`) on the mutating endpoints — its `streaming` exemption is for the append-only startup scan only, and `/api/scan` must keep its own separate check or a rescan fired mid-stream would run a second `build_groups` over the same caches.
- `analyze_paths` attaches `file_size` to each result as that result lands, not in one pass at the end — a group published mid-scan has to be complete. The cache-hit branch needs it too, or `/api/group`'s `sizes` KeyErrors on the second scan of any file (`tests/test_streaming_scan.py` covers this).
- `image_cache` is keyed `(generation, group, file, max_side)`. `max_side` is in the key because the 800px switcher preview and 1600px stage render of the same file would otherwise collide; `generation` is in it because a render runs outside the lock and a rescan finishing mid-render must not drop stale bytes into the fresh cache.
- `hash_cache`/`analyze_cache` deliberately **survive** a rescan and are never reset — that's what makes the control panel's rescan fast on a large library. Don't "fix" the missing reset.
- The frontend appends `?g=<generation>` to image URLs because `(i, j)` indices get reused across rescans and the browser's HTTP cache doesn't know that.
- Every request needs the token, supplied as `?token=` or the `fd_token` cookie that `GET /` sets (an `<img src>` can't carry a header). `/static` is a token-gated route rather than a `StaticFiles` mount for that reason — and not a `BaseHTTPMiddleware` guard, which would wrap `StreamingResponse` and disturb `/api/progress`'s disconnect handling.
- Read the design-direction comment at the top of `static/index.html` before changing layout. The stage swap is deliberately transition-free: a cross-fade hides the very difference being judged.
- Bind keyboard shortcuts on `KeyboardEvent.code`, not `.key` — an alternate layout remaps `.key` before the browser sees it.
- `install.sh` is POSIX sh, not bash (the curl-piped invocation ignores the shebang): no arrays, no `[[ ]]`, no `pipefail`.
- **`POST /api/quit` deliberately reuses the Ctrl-C teardown**: `request_exit` sends this process SIGINT rather than growing a second shutdown path, so everything the next bullet describes covers it too. `os.kill`, not `signal.raise_signal` — the route's exit rides a `BackgroundTask`, which runs on a worker thread, and `raise_signal` would fire the handler there instead of the main thread. The background task is also what gets the 200 to the browser before the process dies; without it the page draws a lost-connection error instead of its goodbye (`tests/test_quit.py` asserts on the received body, which is the only part a patched-`request_exit` test can't prove).
- Ctrl-C shutdown has two moving parts, both regression-tested in `tests/test_shutdown.py`. Scans run in `duplicates_web._scan_executor`, not the loop's default executor (asyncio's teardown joins the default one, so Ctrl-C mid-scan would hang until the scan finished). And `main()` drives `server.serve()` on a bare loop then calls `os._exit(0)` (`asyncio.run`'s SIGINT handler turns a quick second Ctrl-C into a lifespan-cancel traceback). `main()` also sets `duplicates_web.shutting_down` from the signal handler so an open `/api/progress` stream ends itself, and flushes stdout/stderr, which `os._exit` skips (block-buffered under a redirect, so the tokened URL would otherwise be lost).
- Don't `pkill -f find_duplicates.py` while manually testing in a browser — it kills the server under test and the connection failure reads as a product bug.

## LSP

Code intelligence (the `LSP` tool: hover, goToDefinition, findReferences) comes from two user-scope plugins, not from anything in this repo: `pyright-lsp@claude-plugins-official` for `.py`, and `web-lsp` (`~/.claude/skills/web-lsp`) for `.html`/`.css`/`.json`. Both need their servers on PATH — `pyright-langserver`, and the `vscode-*-language-server` binaries from `vscode-langservers-extracted`.

`pyrightconfig.json` is gitignored and machine-local: it points pyright at a dev venv holding cv2/numpy/fastapi. `brisque` and `pyiqa` stay unresolved on purpose — they are optional imports. Diagnostics in `tests/` are mostly stub noise (`cv2.imread` typed Optional, `PIL.Image.LANCZOS` missing from stubs, duck-typed fakes); `duplicates_core.py` and `duplicates_web.py` sit at zero, so a new error there is real.
