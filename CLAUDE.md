# CLAUDE.md

Single-purpose duplicate-image tool: scan a directory for near-duplicate photos (the same image at different sizes/quality), then pick which one to keep from a browser page — including over the LAN, for a library sitting on a NAS or headless box. One front end (the web UI), plus a plain-pip installer.

Before searching for a file or symbol, read `.claude/graph.md` — it maps modules, key symbols, and where to add new code.

## Running

```bash
python3 find_duplicates.py [directory] [--threshold N] [--dest DIR] [--recursive] [--auto] [--dry-run] [--host H] [--port N]
python3 compare_image_quality.py imageA.jpg imageB.jpg   # standalone 2-image comparison
```

Without `--auto`, this prints a tokened URL and runs until Ctrl-C — it does *not* exit when a review finishes, and rescans happen from the page's own control panel rather than by restarting the process. `--auto` skips the review UI and never starts the web server.

Runtime deps: `opencv-python-headless numpy pillow pillow-heif fastapi uvicorn`. There is no requirements.txt — install via `./install.sh`, and add any new dependency to that script.

## Tests

No pytest/unittest — each file is a standalone script with its own `main()` that asserts, prints `ok` lines, and exits non-zero on failure. Run individually (keep the block below the **first** fenced bash block after this heading — `test_claude_md_test_list_sync.py` finds the list by regex and would read an earlier fence instead, reporting all 13 files as missing):

```bash
python3 tests/test_auto_mode.py
python3 tests/test_claude_md_test_list_sync.py
python3 tests/test_confirm_hash.py
python3 tests/test_effective_resolution_downsampling.py
python3 tests/test_fast_scan.py
python3 tests/test_group_ordering.py
python3 tests/test_heic_support.py
python3 tests/test_help_and_labels.py
python3 tests/test_recursive_scan.py
python3 tests/test_scan_progress.py
python3 tests/test_score_group.py
python3 tests/test_shutdown.py
python3 tests/test_unapply_crash_safety.py
python3 tests/test_web_api.py
python3 tests/test_web_progress.py
```

Tests reach the modules via `sys.path.insert(0, ...parent.parent)` — there is no package install step. `test_web_api.py` additionally needs `httpx < 0.28` (test-only; `install.sh` does not install it), since FastAPI's `TestClient` constructs its client with the `app=` shortcut httpx 0.28 removed.

Many tests exist to lock in one specific past bug. Read a test's docstring before changing the code it covers — it usually names a failure mode the assertion alone doesn't reveal.

## Traps

- `duplicates_core.py` must stay importable without the web stack — never import FastAPI/uvicorn into it.
- The hash/scoring constants are empirically tuned, not arbitrary (`DEFAULT_HASH_THRESHOLD`, `CONFIRM_HASH_THRESHOLD`, `CLOSE_CALL_MARGIN`, `MIN_REDUCED_DECODE_SIDE`, `METRIC_WEIGHTS`). Re-verify any change against the real photos in `tests/Test-image/`, not just unit tests.
- Grouping is two-stage: the 64-bit `phash` proposes a pair, the 256-bit half of `phash_pair` confirms it. The 64-bit hash alone cannot distinguish a re-export from a different frame of the same scene. `CONFIRM_HASH_THRESHOLD` is tuned for **recall** — a false positive costs one keypress in the review UI, a false negative is never surfaced — so some near-identical frames still group on purpose. Tightening it drops real duplicates before it stops those. Note `tests/Test-image/` will **not** tell you if you tightened it too far: every known pair there sits at distance 0, nowhere near the tail the constant is set against. The tail cases are aspect recrops (one artwork exported for two screen sizes) and heavy downscales — `tests/test_confirm_hash.py` is what actually covers them.
- `load_hash_gray`'s reduced-decode and full-decode paths must agree on the **64-bit** hash bits — check `MIN_REDUCED_DECODE_SIDE` before touching either one. They do *not* agree on the 256-bit confirmation half — it reaches into mid frequencies, where the two decode paths genuinely differ. That drift is content-dependent (single digits on real photos, tens of bits on synthetic noise), expected, and already absorbed by `CONFIRM_HASH_THRESHOLD`. Don't chase it as a bug.
- `METRIC_WEIGHTS`, `METRIC_DESCRIPTIONS`, and `METRIC_ROWS` get entries added and removed together; the UI's help sheet renders straight off the first.
- Moving files is the only genuinely destructive path (`apply_group`, `_compute_dest`, `apply_pick`, `unapply`, `auto_apply_groups`). Non-kept files are **moved, never deleted** — preserve that invariant. The manifest is in-memory only; recovery after the process exits is a manual move back out of `_duplicates/`.
- `apply_group` never sets `group.status` — the caller owns that decision, including the "stays pending on failure" invariant.
- `auto_apply_groups` must read `file_size` *before* the move; the source path is gone once `apply_group` returns.
- Any numpy-derived value that reaches the JSON API needs an explicit `bool()`/`float()`/`int()` cast where it's computed — `numpy.bool_` doesn't subclass `bool` and isn't JSON-serializable (this bit `build_groups`'s `is_close_call`).
- `compare_image_quality.load_gray` builds one shared **float32** (not float64) buffer, and the metrics fuse multi-pass numpy into single cv2 calls (`cv2.absdiff`, `cv2.norm(..., NORM_L1)`); effective resolution uses `cv2.dft`, not `np.fft.fft2`. Analyze is the per-image bottleneck — "simplifying" any of these back to plain numpy halves its throughput.
- `duplicates_web.py` imports core functions **by name**, so a test patching one for a route handler must patch `duplicates_web.X`, not `duplicates_core.X` — the latter silently has no effect. Names called bare inside `duplicates_core.py` (`load_hash_gray`, `ThreadPoolExecutor`) are patched there instead.
- `Session` is guarded by a plain `threading.Lock`, not `asyncio.Lock`: scan progress/completion callbacks run on an executor thread, not the event loop. Keep the `scanning` guards on the mutating endpoints.
- `image_cache`'s key includes `max_side` — a `(group, file)` key would serve whichever size was requested first for both.
- Read the design-direction comment at the top of `static/index.html` before changing layout. The stage swap is deliberately transition-free: a cross-fade would hide the very difference being judged.
- Bind keyboard shortcuts on `KeyboardEvent.code`, not `.key` — an alternate layout remaps `.key` before the browser sees it.
- `install.sh` is POSIX sh, not bash (the curl-piped invocation ignores the shebang): no arrays, no `[[ ]]`, no `pipefail`.
- Ctrl-C shutdown has two moving parts, both regression-tested in `tests/test_shutdown.py`: scans run in `duplicates_web._scan_executor`, not the loop's default executor (asyncio's teardown joins the default one, so a Ctrl-C mid-scan would hang until the scan finished), and `main()` drives `server.serve()` on a bare loop then calls `os._exit(0)` (`asyncio.run`'s SIGINT handler turns a quick second Ctrl-C into a lifespan-cancel traceback). `main()` also sets `duplicates_web.shutting_down` from the signal handler so an open `/api/progress` stream ends itself -- uvicorn's graceful shutdown otherwise waits on it for the whole scan -- and flushes stdout/stderr, which `os._exit` skips (block-buffered under a redirect, so the tokened URL would be lost).
- Don't `pkill -f find_duplicates.py` while manually testing in a browser — it kills the server under test and the connection failure reads as a product bug.
