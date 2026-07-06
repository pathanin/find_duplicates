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

1. **Fast scanning.** Measured on this directory's 12 photos: the phash pass
   decodes every JPEG at full resolution just to shrink it to 32×32 (0.78s;
   `cv2.IMREAD_REDUCED_GRAYSCALE_8` does DCT-domain 1/8 decode in 0.41s with
   at most 1 bit of hash drift out of 64 — harmless against the threshold of
   10). The dominant cost is `analyze()`: 0.6s per ~3MB photo, run serially
   on every group member before the UI opens. Plan: reduced-resolution
   decode for hashing, `analyze()` parallelized with a `ProcessPoolExecutor`,
   a metrics cache keyed by `(path, mtime, size)` so re-runs are near-free,
   and optionally lazy per-group analysis in Textual background workers so
   the TUI opens right after grouping.
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
- `.gitignore` — excludes tool output (`decisions.json`, `_duplicates/`) and
  the personal photos/videos that happen to live in this working directory,
  since neither is part of "the program"
