# Handoff: duplicate image finder

## What this is

Two scripts:

- `compare_image_quality.py` — pre-existing, unmodified. Pairwise no-reference
  image quality comparison (sharpness, effective resolution via FFT cutoff,
  noise, blockiness, optional BRISQUE/NIQE).
- `find_duplicates.py` — new. Scans a directory (top level only) for
  near-duplicate images (same photo, different export size/quality), groups
  them with a perceptual hash, scores each candidate in a group using
  `compare_image_quality.analyze()`, and gives you a Textual TUI to confirm
  which one to keep. Non-kept files move to `_duplicates/`; every decision is
  logged to `decisions.json` in the scanned directory.

Usage: `python3 find_duplicates.py [directory] [--threshold N] [--dest DIR] [--dry-run]`

## Status: complete and verified

- Grouping found all 7 real duplicate pairs in this directory; quality
  scoring consistently suggested the higher-resolution, sharper original.
- Confirm / skip / override (`1`-`9`, `←`/`→`) / quit, move-to-`_duplicates`
  with collision suffixing, `--dry-run`, and re-deciding an already-decided
  group (no-op, no crash) all verified end-to-end.
- Preview rendering verified both visually and mechanically without screen
  capture: run the app under Textual's test `Pilot` with the `HalfcellImage`
  renderer, `App.export_screenshot()` to SVG (faithful for character-cell
  renderers), `qlmanage -t` to PNG, inspect. Aspect ratios correct within
  0–2.5% in 2- and 3-image rows. `test_preview_render.py` locks this in and
  proves its own check catches the old stretch bug. The result carries over
  to iTerm2 because `SixelImage` inherits the same sizing methods verified
  headlessly.

Gotchas worth remembering (both produced silent blank/distorted output, no
errors):

- Inside Textual, Sixel needs the package's auto-resolved `Image` alias from
  `textual_image.widget`, not `AutoImage` directly — `AutoImage` silently
  renders nothing when the active protocol is Sixel.
- The library only letterboxes (preserves aspect) when the widget's CSS
  width *and* height are both literally `auto`; any concrete size forces a
  stretch-to-fill. Guarded by `test_preview_render.py`.

## Next features (planned)

1. **Fast scanning — perf trio landed, lazy-UI still open.** Done:
   `load_hash_gray()` decodes at `cv2.IMREAD_REDUCED_GRAYSCALE_8` (1/8 scale)
   for hashing instead of full resolution, falling back to a full decode
   whenever the reduced decode would land below 32px on the short side (the
   phash's own resize target) — verified this matters: a real large/small
   duplicate pair matched exactly (0 bits) at full decode but drifted 3/64
   bits when the small side used a naive reduced-8 decode with no fallback.
   `analyze()` calls are parallelized via `ProcessPoolExecutor` across every
   group member in one batch, backed by a metrics cache keyed by
   `(path, mtime, size)` at `.find_duplicates_cache.json` in the scanned
   directory (gitignored) so unchanged files skip recomputation and the pool
   isn't even constructed on an all-cache-hit re-run. Deliberately **not**
   forcing a `fork` multiprocessing context despite it measuring ~6x faster
   pool startup than the default `spawn`: `group_duplicates()` has already
   done real cv2 decode work by the time the pool spins up, and forking
   after cv2/numpy have used internal threads reliably crashed the pool
   (`BrokenProcessPool`) in testing — a macOS fork-after-threads hazard, not
   a fluke. Spawn's one-time ~0.3s tax per invocation is the safe tradeoff.
   Covered by `test_fast_scan.py`. Still open: lazy per-group analysis in
   Textual background workers so the TUI opens right after grouping instead
   of waiting for all `analyze()` calls to finish — deferred as its own
   checkpoint since it changes the `Group` lifecycle from "fully populated
   before app launch" to "populated over time," which is exactly the
   invariant `test_preview_render.py` currently assumes.
2. **1:1 zoom-crop compare.** Thumbnails are too small to show sharpness or
   compression differences, so "close call ⚠" groups can't actually be
   resolved in-app (`o`/Preview.app opens full images but aligns nothing).
   Add a `z` binding showing the *same* center region of each candidate at
   100% pixel scale, side by side.
3. **Recursive scan + scalable grouping.** Scanning is top-level only and
   grouping is O(n²) pairwise Hamming — fine for ~20 images, not for a real
   library. Add `--recursive` and bucket hashes (prefix buckets or a
   BK-tree) before pairwise comparison.

## Known limitations / not deeply tested

- `o` (open full-res in Preview.app) only verified to compile; not exercised
  in automated tests to avoid popping GUI windows. The Linux `xdg-open`
  fallback is unverified.
- Terminal graphics support varies by emulator (the `textual-image` source
  notes Kitty's compositing trick is broken on Konsole/WezTerm even when
  advertised). Tuned against iTerm2 3.6.11; other terminals fall back to the
  Halfcell renderer, which is the one verified visually here.
- `METRIC_WEIGHTS` in `find_duplicates.py` is a designed heuristic, only
  validated by "it picked the objectively higher-resolution original in this
  directory's 7 real duplicate pairs."

## Files

- `find_duplicates.py` — main script
- `compare_image_quality.py` — unmodified, imported for `analyze()`
- `test_preview_render.py` — headless preview-rendering regression test
  (aspect ratio + non-blank widgets + stretch-detection self-check)
- `test_fast_scan.py` — regression test for reduced-decode hashing (fast
  path + small-image fallback) and the analyze() cache (round-trip,
  invalidation, corrupt-file handling, pool skipped on all-cache-hit)
- `.gitignore` — excludes tool output (`decisions.json`, `_duplicates/`,
  `.find_duplicates_cache.json`) and the personal photos/videos that happen
  to live in this working directory, since none of it is part of "the
  program"
