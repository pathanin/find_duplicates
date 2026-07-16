"""Regression tests for the FFT memory-cap downsampling in
compare_image_quality.effective_resolution() (added in 9b6a6bd).

That change downsamples images larger than EFFECTIVE_RES_MAX_PX before
running the FFT, to bound peak memory (~1.5 GB -> ~35 MB on a 24MP image).
The commit's inline comment asserts two properties that make this safe:
  1. cutoff_fraction is a fraction of Nyquist, so it is unaffected by the
     internal downsample -- it should come out identical to what you'd get
     by pre-downsampling the image yourself with the same resize params
     before calling effective_resolution().
  2. equivalent_pixels is computed from the *original* min(h, w), not the
     downsampled shape, so the absolute pixel estimate doesn't collapse
     toward EFFECTIVE_RES_MAX_PX for huge images.

Neither property had test coverage, and both are easy to silently break
in a later refactor (e.g. moving the orig_min_side capture after the
resize, or reusing h/w post-resize for the pixel estimate) without any
existing test catching it.

Run: python3 test_effective_resolution_downsampling.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import compare_image_quality as ciq


def make_gray(h: int, w: int, seed: int) -> np.ndarray:
    """Deterministic-but-structured grayscale array (float64, like load_gray
    produces) so the FFT has real content rather than pure white noise."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(h // 8 + 1, w // 8 + 1), dtype=np.uint8)
    return cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float64)


def test_small_image_never_invokes_resize() -> None:
    """At or under EFFECTIVE_RES_MAX_PX, the downsample branch must not
    trigger at all -- confirmed by patching cv2.resize to record calls
    rather than by comparing numbers (a numeric-only check could pass by
    coincidence)."""
    gray = make_gray(300, 400, seed=1)
    assert max(gray.shape) <= ciq.EFFECTIVE_RES_MAX_PX

    calls = []
    real_resize = cv2.resize

    def spy(*args, **kwargs):
        calls.append(1)
        return real_resize(*args, **kwargs)

    cv2.resize = spy
    try:
        ciq.effective_resolution(gray)
    finally:
        cv2.resize = real_resize

    assert not calls, "effective_resolution() resized an image at/under the cap"
    print("  ok  image at/under EFFECTIVE_RES_MAX_PX skips the downsample branch")


def test_large_image_triggers_resize() -> None:
    """Sanity check for the opposite branch: an image over the cap must
    hit cv2.resize exactly once inside effective_resolution()."""
    gray = make_gray(1200, 3000, seed=2)  # max dim 3000 > 2048
    assert max(gray.shape) > ciq.EFFECTIVE_RES_MAX_PX

    calls = []
    real_resize = cv2.resize

    def spy(*args, **kwargs):
        calls.append(1)
        return real_resize(*args, **kwargs)

    cv2.resize = spy
    try:
        ciq.effective_resolution(gray)
    finally:
        cv2.resize = real_resize

    assert len(calls) == 1, f"expected exactly one downsample resize, got {len(calls)}"
    print("  ok  image over EFFECTIVE_RES_MAX_PX triggers exactly one downsample resize")


def test_internal_downsample_matches_manual_predownsample() -> None:
    """The internal downsample must be equivalent to the caller having
    pre-downsampled with the same scale/interpolation before calling --
    that equivalence is exactly what makes cutoff_fraction scale-invariant
    here. cutoff_fraction should match exactly; equivalent_pixels should
    NOT (it's scaled by different min(h, w) values by design -- see next
    test)."""
    large = make_gray(2250, 3000, seed=3)  # max dim 3000 > 2048
    h, w = large.shape

    scale = ciq.EFFECTIVE_RES_MAX_PX / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    manually_downsampled = cv2.resize(large, (new_w, new_h), interpolation=cv2.INTER_AREA)

    cutoff_internal, eq_internal = ciq.effective_resolution(large.copy())
    cutoff_manual, eq_manual = ciq.effective_resolution(manually_downsampled.copy())

    assert cutoff_internal == cutoff_manual, (
        f"cutoff_fraction should be identical whether downsampling happens "
        f"inside effective_resolution() or before calling it: "
        f"{cutoff_internal} vs {cutoff_manual}"
    )
    print(f"  ok  cutoff_fraction matches manual pre-downsample ({cutoff_internal})")


def test_equivalent_pixels_uses_original_dimensions() -> None:
    """equivalent_pixels must scale off the ORIGINAL min(h, w), not the
    post-downsample shape -- otherwise a 6000px image and a 2048px image
    with the same real detail would report wildly different absolute
    resolutions."""
    large = make_gray(2250, 3000, seed=4)
    h, w = large.shape
    orig_min_side = min(h, w)

    cutoff, eq_px = ciq.effective_resolution(large.copy())

    assert eq_px == cutoff * orig_min_side, (
        f"equivalent_pixels ({eq_px}) should equal cutoff_fraction * original "
        f"min(h, w) ({cutoff * orig_min_side}), not a downsampled-shape estimate"
    )
    print(f"  ok  equivalent_pixels ({eq_px:.1f}) derived from original min(h, w)={orig_min_side}, not downsampled shape")


def test_boundary_dimension_is_not_downsampled() -> None:
    """max(h, w) == EFFECTIVE_RES_MAX_PX exactly should NOT trigger the
    resize -- the condition is a strict '>', so the boundary itself is the
    smallest image the cap applies to only when exceeded."""
    side = ciq.EFFECTIVE_RES_MAX_PX
    gray = make_gray(side, side, seed=5)

    calls = []
    real_resize = cv2.resize

    def spy(*args, **kwargs):
        calls.append(1)
        return real_resize(*args, **kwargs)

    cv2.resize = spy
    try:
        ciq.effective_resolution(gray)
    finally:
        cv2.resize = real_resize

    assert not calls, "image exactly at EFFECTIVE_RES_MAX_PX should not be downsampled"
    print(f"  ok  image exactly at EFFECTIVE_RES_MAX_PX ({side}px) is not downsampled")


def main() -> None:
    tests = [
        ("small image never invokes resize", test_small_image_never_invokes_resize),
        ("large image triggers resize", test_large_image_triggers_resize),
        ("internal downsample matches manual pre-downsample", test_internal_downsample_matches_manual_predownsample),
        ("equivalent_pixels uses original dimensions", test_equivalent_pixels_uses_original_dimensions),
        ("boundary dimension is not downsampled", test_boundary_dimension_is_not_downsampled),
    ]
    for name, fn in tests:
        try:
            fn()
            print(f"ok {name}")
        except Exception as e:
            print(f"FAIL {name}: {e}")
            sys.exit(1)
    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    main()
