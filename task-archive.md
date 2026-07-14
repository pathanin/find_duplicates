# Daily Improvement Log — Archive

Cold storage for rows moved out of `task.md`. Oldest-first.

Note: the entry below predates the compact single-table log format; kept
verbatim as the historical record of that run rather than reformatted.

### 2026-07-14

**Goal chosen:** Add test coverage for the FFT memory-cap downsampling logic in
`compare_image_quality.effective_resolution()` (introduced in `9b6a6bd`, the
most recent commit on `main`).

**Why over alternatives:** Surveyed the repo per the CLAUDE.md architecture notes
and `improvement-status.md`, which already tracks two prior improvement passes
(40 items, all but a handful resolved or deliberately deferred with stated
reasons — e.g. O(n²) grouping bucketing, NIQE/BRISQUE model reconstruction,
`decisions.json` rewrite cost). No TODO/FIXME/XXX markers exist in the tree, and
the deferred items are all either explicitly low-priority, structurally
unavoidable, or require re-verification against real images that this sandbox
can't do responsibly. The most recent commit (`9b6a6bd`) added new,
correctness-sensitive logic — downsampling large images before FFT to cap
memory, while deliberately computing `equivalent_pixels` off the *original*
dimensions rather than the downsampled ones so the absolute estimate doesn't
collapse for huge images — and shipped with zero test coverage. That is exactly
the kind of subtle, easy-to-silently-break invariant (the inline comment even
flags the scale-invariance assumption) that a future refactor could break
without any existing test catching it. Effort was low (pure test addition, no
production code touched) and risk was minimal (no behavior change), so it beat
touching the scoring/hash constants or other deferred items, which explicitly
call for re-verification against real images rather than a sandbox pass.

**Change made:** Added `tests/test_effective_resolution_downsampling.py` with
5 tests:
- image at/under `EFFECTIVE_RES_MAX_PX` never triggers the downsample resize
  (verified by patching `cv2.resize` to record calls, not just by comparing
  numbers)
- image over the cap triggers exactly one downsample resize
- the internal downsample produces a `cutoff_fraction` identical to manually
  pre-downsampling with the same scale/interpolation before calling the
  function — the property that makes the memory cap safe
- `equivalent_pixels` is verified to equal `cutoff_fraction * original min(h, w)`,
  not a value derived from the downsampled shape
- the exact boundary (`max(h, w) == EFFECTIVE_RES_MAX_PX`) is confirmed not to
  downsample, since the guard condition is a strict `>`

No production code was changed.

**Test results:** All 8 test scripts pass (7 pre-existing + 1 new), each run
individually and exiting 0:
```
tests/test_fast_scan.py                        ok (all fast-scan tests passed)
tests/test_footer_click_safety.py               ok (all footer click-safety tests passed)
tests/test_help_and_labels.py                   ok (all help/label tests passed)
tests/test_keyboard_aliases.py                  ok (all keyboard-alias tests passed)
tests/test_manifest_crash_safety.py             ok (all manifest crash-safety tests passed)
tests/test_preview_render.py                    ok (all preview render tests passed)
tests/test_score_group.py                       ok (12/12 tests passed)
tests/test_effective_resolution_downsampling.py ok (5/5 tests passed, new)
```
(Sandbox note: `rich`/`textual`/`textual-image` had to be installed via pip
first — not present by default in this sandbox — before the Textual-dependent
tests could import `find_duplicates`.)

**Branch / commit:** `test/effective-resolution-downsampling` — see
`git log -1 --format="%H %s"` on that branch for the exact commit hash.
