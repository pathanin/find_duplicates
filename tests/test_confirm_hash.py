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
test_extreme_reexport_still_groups is the real point of this file:
CONFIRM_HASH_THRESHOLD is set to cover the whole measured true-duplicate
tail, because a false positive costs one keypress to skip in the review UI
while a false negative is never surfaced at all. Lowering it to catch the
near-identical frames that still slip through would start dropping genuine
duplicates -- see the constant's comment for the measured distributions.

Run: python3 test_confirm_hash.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import duplicates_core as dc


def make_ui_frame(seed: int, path: Path) -> None:
    """One frame of a chat-style screenshot: a mostly-flat dark field with a
    light panel in a fixed place, and different text inside it each time.
    Reproduces the real failure -- everything the 8x8 block can see is
    identical between frames, and the only difference is small and central."""
    im = Image.new("RGB", (720, 1280), (12, 12, 14))
    draw = ImageDraw.Draw(im)
    draw.rectangle([120, 180, 600, 300], fill=(238, 238, 240))
    rng = np.random.default_rng(seed)
    for _ in range(14):
        x = 140 + int(rng.integers(0, 420))
        y = 205 + int(rng.integers(0, 70))
        draw.rectangle([x, y, x + int(rng.integers(8, 34)), y + 9], fill=(20, 20, 24))
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
        for i in (1, 2):
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
        for i in (1, 2):
            make_ui_frame(i, d / f"frame{i}.jpg")
        d64, d256 = distances(d / "frame1.jpg", d / "frame2.jpg")
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


def test_phash_is_the_grouping_half_of_phash_pair() -> None:
    """phash() and phash_pair() must not drift apart: the cache stores the
    pair, while phash() is what other callers (and tests/test_heic_support)
    still use."""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        make_photo(3, d / "a.jpg")
        gray = dc.load_hash_gray(d / "a.jpg")
        assert dc.phash(gray) == dc.phash_pair(gray)[0], "phash must be phash_pair's first half"
        assert dc.phash_pair(gray)[1].bit_length() <= 256, "confirmation hash must fit in 256 bits"
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
        test_phash_is_the_grouping_half_of_phash_pair,
        test_hash_cache_round_trips_the_pair,
    ):
        print(f"{fn.__name__}:")
        fn()
    print("all confirmation-hash tests passed")


if __name__ == "__main__":
    main()
