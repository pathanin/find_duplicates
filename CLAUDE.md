# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single-purpose CLI/TUI tool: scan a directory for near-duplicate images (same photo at different sizes/quality) and interactively pick which one to keep. Two Python scripts do the real work; the rest of the repo is a plain-pip installer.

## Running the tool

```bash
python3 find_duplicates.py [directory] [--threshold N] [--dest DIR] [--dry-run]
python3 compare_image_quality.py imageA.jpg imageB.jpg   # standalone 2-image comparison, no grouping/TUI
```

Runtime deps: `opencv-python-headless numpy pillow pillow-heif textual textual-image`. There's no requirements.txt — install via `./contrib/install.sh` (creates an isolated venv with prebuilt wheels).

## Tests

No pytest/unittest — each test file is a standalone script with its own `main()` that runs a list of test functions and asserts, printing `ok` lines and exiting non-zero on `AssertionError`. Run individually:

```bash
python3 tests/test_claude_md_test_list_sync.py
python3 tests/test_effective_resolution_downsampling.py
python3 tests/test_fast_scan.py
python3 tests/test_footer_click_safety.py
python3 tests/test_heic_support.py
python3 tests/test_help_and_labels.py
python3 tests/test_keyboard_aliases.py
python3 tests/test_manifest_crash_safety.py
python3 tests/test_preview_render.py
python3 tests/test_scan_progress.py
python3 tests/test_score_group.py
python3 tests/test_unapply_crash_safety.py
```

Tests import the app module via `sys.path.insert(0, str(Path(__file__).resolve().parent.parent)); import find_duplicates as fd` — there's no package install step. To add a test, follow the existing pattern (module-level `test_*` functions, a `main()` list, `if __name__ == "__main__"` trailer) rather than introducing a new test runner.

Many tests exist to lock in a specific bug fix — read the docstring/comment above a test before changing the code it covers; it usually explains a failure mode that isn't obvious from the assertion alone (e.g. `test_keyboard_aliases.py`'s note on alternate keyboard layouts remapping letter keys, or `test_fast_scan.py`'s note on asymmetric resampling drifting the perceptual hash).

## Architecture

**`find_duplicates.py`** — the whole app, in three stages that run in this order:

1. **Scan + group** (`find_images`, `load_hash_gray`, `phash`, `group_duplicates`): top-level-only directory scan, 64-bit DCT perceptual hash per image, union-find grouping by Hamming distance ≤ `--threshold` (default 10/64). `load_hash_gray` prefers a fast 1/8-scale decode and only falls back to a full decode when the image is small enough that the reduced decode would need to upsample — the two paths must agree on hash bits, so don't change one without checking `MIN_REDUCED_DECODE_SIDE`. Hashes are cached on disk (`.find_duplicates_hash_cache.json`, keyed by path + mtime + size, kept separate from the analyze cache below since it stores a bare int rather than an analyze() result dict) so a re-scan only re-decodes new/changed files; the uncached subset always hashes through a `ThreadPoolExecutor` (not a process pool) since `load_hash_gray`/`phash`'s cv2 calls release the GIL, giving real parallelism without a process-spawn tax — benchmarked faster than the old process-pool-above-a-threshold design at every batch size tried, see the comment at `THREAD_POOL_WORKERS`.
2. **Analyze + score** (`analyze_paths`, `score_group`, `build_groups`): each grouped file goes through `compare_image_quality.analyze()`, either from an on-disk cache (`.find_duplicates_cache.json`, keyed by path + mtime + size) or a `ThreadPoolExecutor` — like the hash phase, `analyze()`'s cv2/numpy calls release the GIL, so threads parallelize it without a process-spawn cost (this also sidesteps the fork-after-cv2-threads macOS crash that used to require a spawned, not forked, `ProcessPoolExecutor` here). `score_group` min-max normalizes each metric *within its group only* and combines via `METRIC_WEIGHTS` into `quality_score`; metric weights and their one-line glosses in `METRIC_DESCRIPTIONS` must be added/removed together (the help screen renders straight off `METRIC_WEIGHTS`, so an undocumented metric is a hard error waiting to happen if you rely on the two staying in sync).
3. **Interactive review** (`DuplicateReviewApp`, a Textual `App`): sidebar `ListView` of groups, per-group image previews + `DataTable` of metrics, keybindings for pick/confirm/skip. Confirm/skip actually move files (via `_apply`/`_unapply`), tracked in an in-memory `self.manifest` (not persisted to disk) so re-picking after a confirm, or un-confirming back to skipped, can find what to reverse within the same session. Two non-obvious constraints baked into the bindings:
   - Every destructive action (confirm, skip) has a control-key alias (Enter, Delete/Backspace, Escape) alongside its letter key, because alternate keyboard layouts remap letter keys to different Unicode characters before the terminal sees them, silently breaking letter-only bindings.
   - `simulate_key` is overridden to block Footer's clickable buttons from firing confirm/skip — Footer renders one of a compound binding's keys as a button, and a stray click there must never move files.

**`compare_image_quality.py`** — pure image-metrics module, no Textual/app dependency. `analyze(path) -> dict` is the single entry point `find_duplicates.py` calls per image (also runnable standalone for a two-image side-by-side report). Metrics: Laplacian sharpness (scale-normalized to 512×512 first), FFT-based effective resolution (radially-averaged power spectrum cutoff via `cv2.dft`, not `np.fft.fft2` — resistant to fake upscaling, weighted heaviest in scoring), Immerkaer noise sigma, JPEG blockiness, and optional BRISQUE/NIQE (each wrapped in try/except returning `None` if the optional package isn't installed — never let a missing optional dependency crash a scan). `load_gray` decodes into a shared **float32** (not float64) grayscale buffer that every metric operates on — analyze() is the actual per-image bottleneck (an order of magnitude slower than hashing at realistic photo resolutions), and this buffer's memory-bandwidth cost dominates that; float32 plus fusing multi-pass numpy ops into single cv2 calls (`cv2.absdiff`, `cv2.norm(..., NORM_L1)`) roughly doubled analyze phase throughput, verified against `tests/Test-image`'s real photos at ~1e-7 relative drift on every metric and zero `quality_score` keeper-choice flips across all real duplicate pairs there.

**Packaging**: `contrib/install.sh` creates an isolated venv and installs runtime deps as prebuilt wheels via plain pip, then puts a `find-duplicates` wrapper on PATH.

## Working with this code

- Money/math-sensitive code: the hash/scoring pipeline is tuned by empirical observation (see comments citing specific verified drift, e.g. the 3/64-bit hash drift note in `load_hash_gray`, and the float32/`cv2.dft` notes in `compare_image_quality.py`). Treat existing constants (`DEFAULT_HASH_THRESHOLD`, `CLOSE_CALL_MARGIN`, `MIN_REDUCED_DECODE_SIDE`, `METRIC_WEIGHTS`) as deliberate, not arbitrary — changing one changes duplicate-detection or ranking behavior and should be re-verified against real images, not just unit tests. `tests/Test-image/` holds real duplicate-pair photos for exactly this kind of re-verification (not part of the automated test suite — no `test_*.py` runner touches it, it's a manual/ad hoc validation fixture).
- File moves are the one genuinely destructive path in this codebase (`_apply` in `DuplicateReviewApp`). Non-kept files are moved, never deleted — preserve that invariant in any change touching `_apply`/`_unapply`/`_dest_for`. There's no on-disk manifest; recovery after the app exits is a manual move back out of `_duplicates/` (the scan is top-level-only, so the original location is always the scanned directory). `self.manifest` is in-memory only, used within a session so re-picking a confirmed group or un-confirming to skip can reverse the specific files that were moved.
