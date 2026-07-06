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

## Status: complete — rendering now verified headlessly (see below)

### Verified end-to-end (tmux automation + this directory's real photos)

- Perceptual-hash grouping correctly found all 7 real duplicate pairs here
  (each `uuid.jpg` paired with its higher-resolution `ellekorea_...jpg`
  original).
- Quality scoring consistently suggested keeping the higher-resolution,
  sharper, cleaner original.
- Confirm / skip / override (`1`-`9` or `←`/`→`) / quit flow all work.
- Move-to-`_duplicates` with filename-collision suffixing (`_dup1`, `_dup2`,
  ...) works.
- `--dry-run` leaves every file untouched.
- Re-confirming or re-skipping an already-decided group is now a no-op
  instead of crashing (`shutil.move` on an already-moved file) — caught by
  an advisor review pass, not by my own testing.
- `decisions.json` writes into the scanned directory, not the process's cwd
  (also caught by advisor review, not my own testing).

### Previously "struggling with": visual verification — now resolved

The earlier blocker was that the sandbox has no Screen Recording permission,
so there was no way to *see* whether the preview rendered proportioned and
un-stretched. Workaround that closed the gap without any screen capture:

- Run the real `DuplicateReviewApp` headlessly under Textual's test `Pilot`,
  forcing the `HalfcellImage` renderer (plain colored half-block cells).
- Export the screen with `App.export_screenshot()` — for character-cell
  renderers this SVG is a faithful picture of what the terminal would draw
  (unlike Sixel escapes, which SVG export can't capture).
- Convert SVG → PNG with macOS `qlmanage -t` and inspect the pixels.

Result, verified visually on this directory's real photos: photos render as
recognizable images (no blank boxes), portraits stay portrait, a landscape
photo stays landscape with its in-photo white border intact, everything is
letterboxed and centered in its box, and the green suggested / heavy orange
picked borders move correctly when changing the pick with `→`.

The aspect fix is also verified mechanically and renderer-independently:
`SixelImage` (what iTerm2 uses) inherits `get_content_width/height`
unchanged from the same base class as `HalfcellImage`, so the letterboxing
geometry checked headlessly is exactly what runs under Sixel. The
`container.height or 2**32` first-measure concern flagged below never
materialized — regions settle at aspect-correct sizes in a `Horizontal` of
2 and of 3 image boxes (measured error 0–2.5%).

`test_preview_render.py` (new) locks this in: it asserts non-empty,
aspect-correct regions for portrait/landscape/square plus 4:1 and 1:4
boundary shapes using synthetic images (no personal photos needed), and
re-runs with the old buggy CSS to prove the check actually detects the
stretch regression (321% error, caught). Run: `python3 test_preview_render.py`.

History, for context — the two rounds of bugs that originally only surfaced
from live-terminal screenshots:

1. **Blank black boxes.** I imported `textual_image.widget.AutoImage`
   directly instead of the package's dynamically-resolved `Image` alias.
   iTerm2 supports the Sixel protocol, and Sixel rendering inside Textual
   needs a structurally different widget (`SixelImage`, which overrides
   low-level `render_lines()` to inject raw sixel escapes) — the generic
   `AutoImage`/`BaseImage` widget silently no-ops when the resolved renderer
   is Sixel. It works fine outside Textual (plain `rich.console.Console`)
   and works fine for the Halfcell/Unicode/TGP renderers; only
   Sixel-inside-Textual needs the special widget, and getting it wrong
   produces no error — just an empty box. Fixed by importing `Image` (the
   package's own auto-resolving alias) instead of `AutoImage`.

2. **Stretched/skewed photos** (current fix, not yet confirmed by you). CSS
   on the image widget set `width: 100%; height: 1fr` — both concrete
   sizes. The library only preserves the source aspect ratio when *both*
   width and height are the literal Textual `auto` keyword, which triggers
   its own letterboxing math against the available container size; any
   concrete width/height forces a stretch-to-fill with no aspect
   correction. Fixed by setting `.preview-image { width: auto; height: auto; }`
   and centering the result with `align: center middle` on the parent box.
   This fix is now confirmed correct by the headless verification above.

### Known limitations / not deeply tested

- `o` (open full-res in Preview.app) only verified to compile; not exercised
  in automated tests to avoid popping GUI windows during test runs. Should
  work via `open -a Preview <files>` on macOS; the Linux `xdg-open` fallback
  is unverified.
- Terminal graphics support is inherently inconsistent across emulators —
  the `textual-image` source itself notes the Kitty protocol's
  "unicode placeholder" compositing trick is broken on Konsole/WezTerm even
  when they report support. This has only been tuned against iTerm2 3.6.11.
  Other terminals (Terminal.app, VS Code's integrated terminal) will fall
  back to the Halfcell (colored block character) renderer, which is the
  renderer now verified visually via the headless SVG-screenshot workaround
  above.
- Directory scan is intentionally non-recursive, and grouping is O(n²)
  pairwise Hamming-distance comparisons — fine for the ~20 images here,
  would need an optimization (e.g. bucket by hash prefix) for a directory
  with thousands of images.
- `METRIC_WEIGHTS` in `find_duplicates.py` is a heuristic I designed, not
  validated against ground truth beyond "it picked the objectively
  higher-resolution original in this directory's 7 real duplicate pairs."

## Files

- `find_duplicates.py` — main script
- `compare_image_quality.py` — unmodified, imported for `analyze()`
- `test_preview_render.py` — headless preview-rendering regression test
  (aspect ratio + non-blank widgets + stretch-detection self-check)
- `.gitignore` — excludes tool output (`decisions.json`, `_duplicates/`) and
  the personal photos/videos that happen to live in this working directory,
  since neither is part of "the program"
