"""The 64-bit perceptual hash cannot tell "the same photo exported twice"
from "two different frames of one scene", so group_duplicates confirms every
candidate pair against a wider 256-bit hash before grouping it.

The bug this locks in: scanning a photo dump grouped photos that merely
*looked* alike at thumbnail scale -- four screenshots of one chat UI with
different text in the panel, two frames of one photoshoot seconds apart.
The 64-bit hash keeps only an 8x8 low-frequency block, which is little more
than a thumbnail's gross layout, and those images share that layout entirely;
measured distances were 1-5, well inside the default threshold of 10.

The direction of the tuning matters as much as the fix, and
test_aspect_recrop_still_groups is the real point of this file:
CONFIRM_HASH_THRESHOLD is set above the whole measured true-duplicate tail,
because a false positive costs one keypress to skip in the review UI while a
false negative is never surfaced at all. The two distributions overlap, so
lowering the cut to catch the near-identical frames that still slip through
drops genuine duplicates first -- a cut of 56 was tried, looked clean against
re-exports, and silently lost six real duplicates from a real library.

Run: python3 test_confirm_hash.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import duplicates_core as dc


# Seeds picked so the pair lands in the narrow regime this fix is about: the
# 64-bit hash sees a distance of 4 (well inside the default threshold of 10)
# while the 256-bit hash sees 110 (well past CONFIRM_HASH_THRESHOLD). That
# band is narrow on purpose -- it is exactly where an 8x8 low-frequency block
# is fooled and a 16x16 one is not, so an arbitrary seed pair usually misses
# it. test_detector_without_the_fix_would_have_grouped_them asserts both
# halves, so this stays honest if the hashes ever change.
UI_FRAME_SEEDS = (0, 13)


def make_ui_frame(seed: int, path: Path) -> None:
    """One frame of a chat-style screenshot: a mostly-flat dark field with a
    light panel in a fixed place, and different content inside it each time.
    Reproduces the real failure -- everything the 8x8 block can resolve is
    identical between frames, since the panel and the dark surround dominate
    it, and the differences live at a finer scale."""
    im = Image.new("RGB", (720, 1280), (14, 14, 16))
    draw = ImageDraw.Draw(im)
    draw.rectangle([90, 150, 630, 1130], fill=(236, 236, 238))
    rng = np.random.default_rng(seed)
    for _ in range(30):
        x = 100 + int(rng.integers(0, 530 - 52))
        y = 160 + int(rng.integers(0, 960 - 52))
        draw.rectangle([x, y, x + 52, y + 28], fill=(26, 26, 30))
    im.save(path, quality=92)


def make_photo(seed: int, path: Path, size: tuple[int, int] = (1200, 900)) -> Image.Image:
    """Blurred noise: enough real structure across the whole frequency range
    for a perceptual hash to bite on, unlike a flat or synthetic gradient."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, size=(size[1], size[0], 3), dtype=np.uint8)
    im = Image.fromarray(arr).filter(ImageFilter.GaussianBlur(3))
    im.save(path, quality=95)
    return im


def distances(a: Path, b: Path) -> tuple[int, int]:
    """(64-bit grouping distance, 256-bit confirmation distance)."""
    ha = dc.phash_pair(dc.load_hash_gray(a))
    hb = dc.phash_pair(dc.load_hash_gray(b))
    return dc.hamming(ha[0], hb[0]), dc.hamming(ha[1], hb[1])


def test_same_scene_frames_do_not_group() -> None:
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        for i in UI_FRAME_SEEDS:
            make_ui_frame(i, d / f"frame{i}.jpg")
        groups = dc.group_duplicates(sorted(d.glob("*.jpg")), dc.DEFAULT_HASH_THRESHOLD, {})
        assert groups == [], f"different frames of one scene must not group, got {groups}"
    print("  ok  two frames of one scene are not grouped as duplicates")


def test_detector_without_the_fix_would_have_grouped_them() -> None:
    """Proves the case above is a real regression guard: the 64-bit hash on
    its own puts those two frames comfortably inside the default threshold,
    so this test fails if the confirmation step is ever removed."""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        for i in UI_FRAME_SEEDS:
            make_ui_frame(i, d / f"frame{i}.jpg")
        d64, d256 = distances(*(d / f"frame{i}.jpg" for i in UI_FRAME_SEEDS))
        assert d64 <= dc.DEFAULT_HASH_THRESHOLD, (
            f"the 64-bit hash must still propose this pair or the test proves nothing (got {d64})")
        assert d256 > dc.CONFIRM_HASH_THRESHOLD, (
            f"confirmation must be what rejects it (got {d256} <= {dc.CONFIRM_HASH_THRESHOLD})")
    print("  ok  the 64-bit hash alone would have grouped them; confirmation is what rejects")


def test_reexport_still_groups() -> None:
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        im = make_photo(7, d / "orig.jpg")
        im.resize((600, 450), Image.LANCZOS).save(d / "small.jpg", quality=55)
        groups = dc.group_duplicates(sorted(d.glob("*.jpg")), dc.DEFAULT_HASH_THRESHOLD, {})
        assert len(groups) == 1 and len(groups[0]) == 2, (
            f"a downscaled, recompressed export is the duplicate we exist to find, got {groups}")
    print("  ok  a resized and recompressed re-export still groups")


def test_extreme_reexport_still_groups() -> None:
    """Boundary case, and the reason CONFIRM_HASH_THRESHOLD sits where it
    does: a 20% NEAREST downscale at quality 35 is the worst re-export
    measured, and it must still survive confirmation. A cut tightened to
    reject the near-identical frames that currently slip through would lose
    exactly this -- a real duplicate the user would never be shown."""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        im = make_photo(11, d / "orig.jpg")
        im.resize((240, 180), Image.NEAREST).save(d / "tiny.jpg", quality=35)
        d64, d256 = distances(d / "orig.jpg", d / "tiny.jpg")
        assert d64 <= dc.DEFAULT_HASH_THRESHOLD, f"prefilter dropped it first (got {d64})"
        assert d256 <= dc.CONFIRM_HASH_THRESHOLD, (
            f"confirmation must not reject a genuine re-export (got {d256} > "
            f"{dc.CONFIRM_HASH_THRESHOLD}) -- this is a false negative, the failure "
            f"mode this threshold is explicitly tuned to avoid")
    print("  ok  even a 20% NEAREST/q35 re-export survives confirmation")


def test_aspect_recrop_still_groups() -> None:
    """The case that actually set CONFIRM_HASH_THRESHOLD: one artwork exported
    for two phone screens (1440x3200 and 1170x2532) is the same image, but the
    aspect-ratio crop moves far more bits than a plain rescale -- real pairs of
    this kind reached 80. An earlier cut of 56 looked clean on re-exports alone
    and silently dropped six of them from a real library."""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        rng = np.random.default_rng(9)
        art = Image.fromarray(
            rng.integers(0, 255, size=(2400, 1200, 3), dtype=np.uint8)
        ).filter(ImageFilter.GaussianBlur(4))

        def export(w: int, h: int, path: Path) -> None:
            """Centre-crop to the target aspect, then scale -- what a wallpaper
            pack does to fit one artwork to several devices."""
            aw, ah = art.size
            if aw / ah > w / h:
                nw = int(ah * w / h)
                box = ((aw - nw) // 2, 0, (aw - nw) // 2 + nw, ah)
            else:
                nh = int(aw * h / w)
                box = (0, (ah - nh) // 2, aw, (ah - nh) // 2 + nh)
            art.crop(box).resize((w, h), Image.LANCZOS).save(path, quality=88)

        export(1440, 3200, d / "galaxy.jpg")
        export(1170, 2532, d / "iphone.jpg")
        d64, d256 = distances(d / "galaxy.jpg", d / "iphone.jpg")
        assert d64 <= dc.DEFAULT_HASH_THRESHOLD, f"prefilter dropped it first (got {d64})"
        assert d256 <= dc.CONFIRM_HASH_THRESHOLD, (
            f"confirmation rejected the same artwork recropped for another screen "
            f"(got {d256} > {dc.CONFIRM_HASH_THRESHOLD}) -- a false negative, the "
            f"failure mode this threshold is explicitly tuned to avoid")
    print("  ok  one artwork exported for two screen shapes survives confirmation")


def test_flat_screenshot_reexport_still_groups() -> None:
    """The recall direction for the image class that motivated all this. The
    negative cases above are screenshots, and every other positive case here
    is full-spectrum blurred noise -- so without this, the suite would prove
    only that screenshots which differ get rejected, and never that
    screenshots which are genuinely the same still group.

    Flat frames are the plausible place for confirmation to misfire: almost
    all their energy sits in the DC and the lowest coefficients, so the
    mid-frequency bits the 16x16 block adds could in principle be thresholding
    noise. They are not -- the panel edges carry real structure -- and this
    pins that down."""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        make_ui_frame(UI_FRAME_SEEDS[0], d / "orig.jpg")
        im = Image.open(d / "orig.jpg")
        im.resize((int(im.width * 0.35), int(im.height * 0.35)), Image.LANCZOS).save(
            d / "small.jpg", quality=40)
        groups = dc.group_duplicates(sorted(d.glob("*.jpg")), dc.DEFAULT_HASH_THRESHOLD, {})
        assert len(groups) == 1 and len(groups[0]) == 2, (
            f"a re-exported screenshot is still the same image, got {groups}")
    print("  ok  a downscaled re-export of a flat screenshot still groups")


def test_raising_the_threshold_switches_confirmation_off() -> None:
    """--threshold is the documented remedy when a scan finds too little, and
    a fixed confirmation gate would quietly defeat it: the two hashes are
    correlated, so the pairs a wider threshold is meant to recover are the
    same ones confirmation rejects. Measured on a real library, --threshold 30
    recovered 51 of 51 known duplicates unconfirmed but only 45 confirmed.

    The frames here are a genuine non-duplicate -- grouping them above the
    default is the point, not a bug: widening the threshold is a request for
    looser matching, and the review UI is where the extra pairs get skipped."""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        for i in UI_FRAME_SEEDS:
            make_ui_frame(i, d / f"frame{i}.jpg")
        files = sorted(d.glob("*.jpg"))
        assert dc.group_duplicates(files, dc.DEFAULT_HASH_THRESHOLD, {}) == [], (
            "confirmation must still apply at the default threshold")
        widened = dc.group_duplicates(files, dc.DEFAULT_HASH_THRESHOLD + 1, {})
        assert len(widened) == 1 and len(widened[0]) == 2, (
            f"above the default threshold the user has asked for looser matching, so "
            f"confirmation must step aside, got {widened}")
    print("  ok  a widened --threshold turns confirmation off instead of overriding it")


def test_phash_is_the_grouping_half_of_phash_pair() -> None:
    """phash() and phash_pair() must not drift apart: the cache stores the
    pair, while phash() is what other callers (and tests/test_heic_support)
    still use."""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        make_photo(3, d / "a.jpg")
        gray = dc.load_hash_gray(d / "a.jpg")
        assert dc.phash(gray) == dc.phash_pair(gray)[0], "phash must be phash_pair's first half"
        # The two halves must be genuinely different hashes, not the same block
        # twice: a 16x16 low-frequency block strictly contains the 8x8 one, so
        # wiring both to dct[:8,:8] would still pass the assertion above while
        # making confirmation a no-op.
        h64, h256 = dc.phash_pair(gray)
        assert h256 != h64, "the confirmation hash must not just repeat the 64-bit one"
        assert h256.bit_count() > h64.bit_count(), (
            "the 256-bit half must carry more set bits than the 8x8 block alone")
    print("  ok  phash() is phash_pair()'s 64-bit half")


def test_hash_cache_round_trips_the_pair() -> None:
    """The cache stores both hashes now; a cache hit must be usable by the
    pair loop without re-decoding."""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        make_photo(5, d / "a.jpg")
        p = d / "a.jpg"
        cache: dict = {}
        assert dc.cached_hash(cache, p, p.stat()) is None, "empty cache must miss"
        dc.store_hash(cache, p, p.stat(), dc._hash_one(p))
        hit = dc.cached_hash(cache, p, p.stat())
        assert hit == dc._hash_one(p), "cache must round-trip both hashes"
        assert len(hit) == 2, f"a cache hit must carry the pair, got {hit!r}"
    print("  ok  the hash cache round-trips both hashes")


def main() -> None:
    for fn in (
        test_same_scene_frames_do_not_group,
        test_detector_without_the_fix_would_have_grouped_them,
        test_reexport_still_groups,
        test_extreme_reexport_still_groups,
        test_aspect_recrop_still_groups,
        test_flat_screenshot_reexport_still_groups,
        test_raising_the_threshold_switches_confirmation_off,
        test_phash_is_the_grouping_half_of_phash_pair,
        test_hash_cache_round_trips_the_pair,
    ):
        print(f"{fn.__name__}:")
        fn()
    print("all confirmation-hash tests passed")


if __name__ == "__main__":
    main()
