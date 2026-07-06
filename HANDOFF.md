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

## Status: functionally verified, one visual fix awaiting your confirmation

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

### What I'm struggling with, and why

The one thing I cannot verify myself is whether the inline image preview
**actually looks right on screen**. This sandbox has no Screen Recording
permission, so `screencapture` fails outright, and there's no other way for
me to see rendered pixels. All I can check mechanically is: does the app
crash, does it emit real per-pixel escape-sequence data (verified this at
the byte level — decoding a real photo through the fallback renderer
produced 361 distinct genuine RGB values, not blank), and does it run
without exceptions. Whether it looks *correct* — proportioned, not
stretched, actually showing the photo — requires a human looking at a live
terminal window. That's you.

This gap produced two rounds of bugs that only surfaced from your
screenshots, not my own testing:

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
   I relaunched a live iTerm2 window with this fix right before this commit
   but haven't heard back on whether it looks right.

If the aspect ratio is still off after this fix, the next thing I'd check is
whether `get_content_width`/`get_content_height` are being called with a
sane, non-zero `container` size the first time the image widget measures
itself in our horizontally-packed multi-image row — Textual computes
intrinsic sizes before layout fully settles, and the library has a
compensating hack for a zero-height container (`2**32` fallback) that I
haven't fully audited in the context of a `Horizontal` of several image
boxes rather than the library's own (presumably simpler) demo layout.

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
  back to the Halfcell (colored block character) renderer, which I verified
  at the byte level produces real per-pixel color data, but haven't seen
  visually.
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
- `.gitignore` — excludes tool output (`decisions.json`, `_duplicates/`) and
  the personal photos/videos that happen to live in this working directory,
  since neither is part of "the program"
