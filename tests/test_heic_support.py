"""Regression tests for HEIC/HEIF support in find_duplicates.py and
compare_image_quality.py: the .heic/.heif extensions must be scanned, and
files in that format must decode through both the perceptual-hashing path
(load_hash_gray) and the quality-analysis path (compare_image_quality's
load_gray/analyze), via a PIL + pillow-heif fallback -- cv2.imread cannot
decode HEIC/HEIF at all without OS-level codecs that aren't reliably
present.

pillow-heif is an optional dependency (see contrib/install.sh). If it isn't
installed, or can't encode a real HEIC file in this environment, the
round-trip test prints a skip note and passes rather than failing -- the
same convention this repo already uses for brisque/niqe in
compare_image_quality.py (a missing optional package must never break the
suite).

Run: python3 test_heic_support.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import find_duplicates as fd
import compare_image_quality as ciq


def test_image_exts_include_heic_and_heif() -> None:
    assert ".heic" in fd.IMAGE_EXTS, (
        "IMAGE_EXTS must include .heic, or find_images() silently skips iPhone/Apple Photos exports"
    )
    assert ".heif" in fd.IMAGE_EXTS, "IMAGE_EXTS must include .heif"
    print("  ok  IMAGE_EXTS contains .heic and .heif")


def _make_test_array(h: int, w: int, seed: int) -> np.ndarray:
    """Deterministic-but-non-trivial RGB texture, same shape as
    test_fast_scan.py's make_texture but built via PIL (not cv2.imwrite),
    since this file needs to encode real HEIC bytes, not a cv2-writable
    format."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(h // 8 + 1, w // 8 + 1, 3), dtype=np.uint8)
    img = PILImage.fromarray(base, mode="RGB").resize((w, h), PILImage.BICUBIC)
    return np.array(img)


def _try_make_real_heic(path: Path) -> bool:
    """Attempt to save a real HEIC file at `path`. Returns False (rather than
    raising) if pillow_heif isn't installed or the registered plugin can't
    encode HEIF in this environment -- callers must treat that as a skip,
    not a failure."""
    try:
        import pillow_heif  # noqa: F401
    except ImportError:
        return False

    try:
        arr = _make_test_array(120, 160, seed=99)
        PILImage.fromarray(arr, mode="RGB").save(path, format="HEIF")
        return path.exists() and path.stat().st_size > 0
    except Exception:
        return False


def test_heic_round_trips_through_hash_and_analysis() -> None:
    """Failure case this guards against: before this feature, cv2.imread
    returned None for HEIC and both load_hash_gray/load_gray had no
    fallback, so a real HEIC file would silently fail to hash and fail to
    analyze -- exactly the "iPhone exports get ignored" bug this feature
    fixes. This proves the whole pipeline (hash, phash, quality metrics,
    and the TUI thumbnail) actually decodes a real HEIC file end to end."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "photo.heic"
        if not _try_make_real_heic(p):
            print(
                "  skip  pillow_heif not installed / can't encode HEIC in this "
                "environment -- treating as pass per this repo's optional-dependency convention"
            )
            return

        # load_hash_gray: must decode via the PIL fallback (cv2 can't read
        # HEIC at all) and return a 2D grayscale array, not None.
        gray = fd.load_hash_gray(p)
        assert gray is not None, "load_hash_gray must decode a real HEIC file, not return None"
        assert gray.ndim == 2, f"expected a 2D grayscale array, got shape {gray.shape}"
        assert gray.shape[0] > 0 and gray.shape[1] > 0

        # The perceptual hash itself must be computable from that array.
        hash_value = fd.phash(gray)
        assert isinstance(hash_value, int)

        # compare_image_quality.load_gray: must decode via its own PIL
        # fallback and return BGR uint8 (matching cv2's channel convention)
        # plus a matching-shape grayscale float32.
        img, ciq_gray = ciq.load_gray(str(p))
        assert img is not None and img.ndim == 3 and img.shape[2] == 3, (
            f"expected a BGR (H, W, 3) array from load_gray, got shape {getattr(img, 'shape', None)}"
        )
        assert ciq_gray.shape == img.shape[:2], "grayscale array must match the color array's (H, W)"
        assert ciq_gray.dtype == np.float32

        # analyze() end to end: the full metrics pipeline must not raise and
        # must return sane, finite values keyed off the real HEIC dimensions.
        result = ciq.analyze(str(p))
        assert result["dimensions"] == (img.shape[1], img.shape[0]), "analyze() dimensions must match the decoded HEIC"
        assert isinstance(result["sharpness_normalized"], float)
        assert result["effective_resolution_fraction"] >= 0.0

        # make_thumbnail: verify (not assume) that it picks up HEIC decoding
        # for free once the HEIF opener is registered with PIL -- it must
        # return the real decoded thumbnail, not the gray failure placeholder.
        thumb = fd.make_thumbnail(p)
        assert thumb.size != (0, 0)
        thumb_arr = np.array(thumb)
        placeholder = np.array(fd.THUMBNAIL_FAILURE_COLOR, dtype=thumb_arr.dtype)
        assert not np.all(thumb_arr == placeholder), (
            "make_thumbnail returned the gray failure placeholder instead of decoding the real HEIC file"
        )

        print(
            f"  ok  real HEIC file ({img.shape[1]}x{img.shape[0]}) decoded through load_hash_gray, "
            f"phash, ciq.load_gray, ciq.analyze(), and make_thumbnail"
        )


def main() -> None:
    tests = [
        test_image_exts_include_heic_and_heif,
        test_heic_round_trips_through_hash_and_analysis,
    ]
    for test in tests:
        print(f"{test.__name__}:")
        test()
    print("all HEIC support tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)
