# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A duplicate-image-finder CLI/TUI: scans a directory for near-duplicate images (same photo exported at
different sizes/quality), groups them by perceptual hash, scores each candidate for *actual* quality
(resistant to fake upscaling, not just pixel count), and lets you interactively confirm which one to keep.

## Commands

- Run the tool: `python3 find_duplicates.py [directory] [--threshold N] [--dest DIR] [--dry-run]`
  `directory` defaults to `.`; `--threshold` is max Hamming distance out of 64 (default 10, lower = stricter);
  `--dest` defaults to `<directory>/_duplicates`; `--dry-run` shows what would happen without moving files.
- Run the standalone quality comparator on two images: `python3 compare_image_quality.py imageA.jpg imageB.jpg`
- Run tests (plain assert-based scripts, not pytest — run directly, exit non-zero on failure):
  - `python3 test_preview_render.py` — headless Textual preview-rendering regression test
  - `python3 test_fast_scan.py` — reduced-decode hashing + analyze()-cache regression tests
  - `python3 test_footer_click_safety.py` — footer key-hints are legends, not buttons, for confirm/skip
  - `python3 test_help_and_labels.py` — metric rows state their direction; `?` help screen covers all weights
  - `python3 test_keyboard_aliases.py` — Enter/Delete/Backspace/Escape/F1 work as layout-independent aliases
    for confirm/skip/finish/help, and every alias (not just whichever one Footer renders) is footer-click-blocked
- No build step, no lint/format config in this repo — plain Python 3 scripts.
- No requirements.txt/pyproject.toml; install manually:
  `pip install opencv-python-headless numpy textual textual-image pillow`
  Optional, for extra no-reference quality metrics in `compare_image_quality.analyze()`:
  `pip install brisque[opencv-python-headless]` and/or `pip install pyiqa torch` — both degrade to
  `None` silently if not installed.

## Architecture

### Two-module split
- `compare_image_quality.py` — no-reference image quality metrics, pre-existing and treated as a black
  box by `find_duplicates.py`. `analyze(path)` returns sharpness (Laplacian variance, scale-normalized),
  effective resolution (FFT radial power spectrum cutoff — detects fake upscaling/softness beyond the
  true detail cutoff), noise sigma, JPEG blockiness, and optional BRISQUE/NIQE. Also runnable standalone
  for a two-image comparison report.
- `find_duplicates.py` — the tool itself. Pipeline: scan (top-level only) → perceptual-hash grouping →
  parallel quality analysis (cached) → per-group scoring → Textual TUI for confirmation → move losers to
  `_duplicates/` + log every decision to `decisions.json`.

### Grouping
- `phash()` is a classic 64-bit DCT hash (low 8x8 DCT coefficients thresholded against their mean),
  robust to resize/recompression — this is what makes "same photo, different export" detectable.
- `load_hash_gray()` decodes at `cv2.IMREAD_REDUCED_GRAYSCALE_8` (1/8 scale) instead of full resolution
  for speed, but falls back to a full decode whenever the reduced decode would land below
  `MIN_REDUCED_DECODE_SIDE` (64px) on the short side. Below that, phash's own 32x32 resize step would
  upsample instead of downsample, asymmetrically drifting the hash relative to a full-decoded sibling of
  a different size — a real large/small duplicate pair drifted 3/64 bits without this guard. This is the
  actual failure mode duplicate detection cares about (inter-pair drift on heterogeneous sizes), not
  same-file drift; verified in `test_fast_scan.py`.
- Grouping is `UnionFind` over all pairwise Hamming distances ≤ `--threshold` (default 10); O(n²), fine
  for small directories, not yet for large libraries.

### Quality scoring
- `score_group()` min-max normalizes each metric *within a group only* — raw metric ranges aren't
  comparable across unrelated photos, only within a set of duplicates of the same photo — then combines
  them via `METRIC_WEIGHTS`, weighted most heavily toward `effective_resolution_px_equiv` since it's the
  metric most resistant to fake upscaling. This weighting is a hand-tuned heuristic, only validated
  empirically ("picked the objectively higher-resolution original in this directory's real duplicate
  pairs"), not derived analytically — treat changes to it as needing re-validation against real photos,
  not just unit tests.
- A "close call ⚠" flag fires when the top two scores in a group are within `CLOSE_CALL_MARGIN` (0.08) —
  these currently can't be resolved in-app since thumbnails are too small to show the actual difference.

### Caching + parallelism
- `analyze()` (the expensive step — FFT, optional BRISQUE/NIQE) runs through a `ProcessPoolExecutor`
  across every group member in one batch, backed by a cache at `.find_duplicates_cache.json` in the
  scanned directory, keyed by `(path, mtime, size)`. If every requested path is a cache hit, the pool is
  never even constructed.
- Deliberately does **not** force a `fork` multiprocessing context, even though it starts faster:
  `group_duplicates()` already does real cv2 decode work in-process before the pool spins up, and forking
  after cv2/numpy have touched internal threads reliably crashes the pool (`BrokenProcessPool`) on macOS —
  reproduced, not hypothetical. Default `spawn` is the safe choice despite its one-time re-import cost.

### Textual TUI
- The `Group` dataclass holds `paths`, `results` (from `analyze()`), `thumbnails` (PIL images),
  `suggested_idx` (highest quality score), `current_pick`, and `status` (`pending`/`confirmed`/`skipped`).
  It's currently built fully — every `analyze()` call completes — before the app launches; there's no
  lazy/background loading yet.
- Two gotchas, both silent-failure (no exception, no visible error) if violated:
  - Use the package's auto-resolved `Image` alias from `textual_image.widget`, not `AutoImage` directly —
    `AutoImage` silently renders nothing when the active protocol is Sixel.
  - The library only letterboxes (preserves aspect ratio) when a widget's CSS width *and* height are both
    literally `auto`; any concrete size forces a stretch-to-fill. Guarded by `test_preview_render.py`,
    which also proves its own check can fail (re-runs with the old buggy CSS and asserts distortion IS
    detected, so the assertion isn't vacuous).
- Confirmed picks move non-kept files to `_duplicates/` (collision-suffixed, e.g. `_dup1`) and log every
  decision to `decisions.json`; `--dry-run` skips the actual move but still logs.
- Core actions (confirm/skip/finish/help) each have both a mnemonic letter binding (c/s/q/?) and a
  control-key alias (Enter/Delete-or-Backspace/Escape/F1), since alternate keyboard layouts remap letters
  to different characters before the terminal ever sees them but leave control keys alone. The Enter alias
  needs `priority=True` to win over ListView's and DataTable's own built-in enter-to-select_cursor binding
  — and doing that changes which key Textual's `Footer` renders as confirm's clickable button (it picked
  "enter" over "c"), which silently reopened the footer-click-confirms hole below if the click guard is
  keyed off a literal string instead of derived from `BINDINGS` (`test_keyboard_aliases.py` guards this).
- `DuplicateReviewApp.simulate_key()` blocks Footer clicks (which route through this method) for every key
  bound to confirm/skip (`_DESTRUCTIVE_KEYS`, computed from `BINDINGS` — don't hardcode it, see above) —
  otherwise Footer's key hints are real clickable buttons, and an incidental click (e.g. clicking the
  terminal to refocus it right as the scan finishes) silently confirms/skips the active group with no
  keypress at all. Guarded by `test_footer_click_safety.py`.

### Known gaps (not yet built)
- No lazy per-group analysis — the TUI waits for all `analyze()` calls before opening. Any future work
  here changes `Group` from "fully populated before launch" to "populated over time," which
  `test_preview_render.py` currently assumes — treat that as a compatibility constraint, not just an
  implementation detail.
- No 1:1 zoom-crop compare — thumbnails are too small to resolve "close call" groups in-app (`o` opens
  full images in Preview.app but aligns nothing for comparison).
- No recursive scan, and grouping is O(n²) pairwise — fine for ~20 images, not a real photo library.
- `o` (open full-res in Preview.app) is only verified to compile, not exercised in automated tests
  (avoids popping GUI windows during test runs); the Linux `xdg-open` fallback is unverified.
- Terminal graphics support varies by emulator; tuned against iTerm2 3.6.11, other terminals fall back to
  the Halfcell renderer (the one actually verified visually here).

## Working directory contents

`.gitignore` excludes tool output (`decisions.json`, `_duplicates/`, `.find_duplicates_cache.json`) and
also excludes all image/video extensions plus `.DS_Store` — this working directory has personal
photos/videos sitting in it that are not part of the program and not test fixtures. Don't assume image
files present here should be committed, read as sample data, or treated as part of the codebase.
