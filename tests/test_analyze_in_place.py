"""analyze() decodes a JPEG straight into its gray buffer, and must stay
bit-identical to the plain decode while doing it.

On a 100 MP photo, memory -- not cores -- decides how many analyses run at
once: duplicates_core admits them against a byte budget priced per pixel.
analyze() used to peak at ~21 B/px. cv2.imread allocates its result twice
over, the color image lived to the end for an optional metric that is almost
never installed, and both blockiness diffs were held together. Now one
float32 gray and one float32 scratch are all that scale with the image
(~9 B/px), so twice as many 100 MP files analyze at once.

What gets it there is cv2.imread's dst overload, which decodes in place but
says nothing on failure: a missing, corrupt or unsupported file hands back
the untouched buffer, and scoring that would rank photos on garbage.
_imread_into presets a marker in the last row to tell. And the scoring trap
in CLAUDE.md applies in full -- a metric that drifts at all can flip a
suggested pick -- so these tests compare against the plain decode bit for
bit rather than within a tolerance.

Run: python3 tests/test_analyze_in_place.py
"""

import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import compare_image_quality as ciq
import duplicates_core as dc


def textured(h: int, w: int, seed: int = 0) -> np.ndarray:
    """BGR with structure at every scale, so no metric is degenerate."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(h // 8 + 1, w // 8 + 1, 3), dtype=np.uint8)
    img = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)
    return cv2.add(img, rng.integers(0, 12, size=img.shape, dtype=np.uint8))


def save_with_orientation(img_bgr: np.ndarray, path: Path, orientation: int) -> None:
    exif = PILImage.Exif()
    exif[0x0112] = orientation
    PILImage.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).save(
        path, exif=exif.tobytes(), quality=90)


def test_every_orientation_decodes_like_the_plain_call(tmp: Path) -> None:
    """cv2 applies EXIF orientation itself on both paths; the in-place one
    must agree with it for the flips (2-4) it handles, and hand the
    transposing ones (5-8) to the plain call rather than guess."""
    img = textured(48, 64)
    used_in_place = {}
    real = ciq._imread_into

    def spy(path, buf):
        out = real(path, buf)
        used_in_place[Path(path).name] = out is not None
        return out

    ciq._imread_into = spy
    try:
        for o in range(1, 9):
            p = tmp / f"o{o}.jpg"
            save_with_orientation(img, p, o)
            got, _ = ciq._gray_and_brisque(str(p))
            _, want = ciq.load_gray(str(p))
            assert got.dtype == np.float32 and got.shape == want.shape, (o, got.dtype, got.shape, want.shape)
            assert np.array_equal(got, want), f"orientation {o}: in-place gray differs from the plain decode"
    finally:
        ciq._imread_into = real
    assert used_in_place == {f"o{o}.jpg": True for o in range(1, 5)}, used_in_place
    print("  ok  all 8 EXIF orientations match the plain decode; 1-4 decode in place, 5-8 don't try")


class _SilentFailureCv2:
    """cv2 whose dst overload does what the real one does on a bad file --
    returns the buffer as handed in -- after optionally writing every row but
    the last, as a decode dying partway would. Everything else is real cv2."""

    def __init__(self, partial: bool):
        self.partial = partial

    def __getattr__(self, name):
        return getattr(cv2, name)

    def imread(self, path, *args):
        if args and isinstance(args[0], np.ndarray):
            buf = args[0]
            if self.partial:
                buf[:-1] = 99
            return buf
        return cv2.imread(path, *args)


def test_a_silent_decode_failure_falls_back(tmp: Path) -> None:
    """The dst overload reports nothing when it fails. A buffer that comes
    back still ending in the marker must never be scored."""
    p = tmp / "photo.jpg"
    cv2.imwrite(str(p), textured(40, 56, seed=3))
    _, want = ciq.load_gray(str(p))
    for partial in (False, True):
        ciq.cv2 = _SilentFailureCv2(partial)
        try:
            got, _ = ciq._gray_and_brisque(str(p))
        finally:
            ciq.cv2 = cv2
        assert np.array_equal(got, want), f"partial={partial}: scored the undecoded buffer"

    buf = np.zeros((40, 56, 3), np.uint8)
    assert ciq._imread_into(str(tmp / "missing.jpg"), buf) is None
    (tmp / "junk.jpg").write_bytes(b"not a jpeg at all")
    assert ciq._imread_into(str(tmp / "junk.jpg"), buf) is None
    assert ciq._imread_into(str(p), np.zeros((56, 40, 3), np.uint8)) is None, "wrong shape must not raise"
    assert ciq._imread_into(str(p), buf) is buf
    assert np.array_equal(buf, cv2.imread(str(p), cv2.IMREAD_COLOR))
    print("  ok  an untouched or half-written buffer falls back; bad files and shapes return None")


def test_shared_scratch_matches_fresh_arrays() -> None:
    """noise_estimate and blockiness_score now write into one scratch array
    in turn. Reference values here come from the formulation they replaced:
    a fresh contiguous array per step. A stale scratch (NaN-filled, as the
    noise filter leaves it for blockiness) must not leak into either."""
    gray = cv2.cvtColor(textured(203, 157, seed=5), cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape

    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    l1 = cv2.norm(cv2.filter2D(gray, -1, kernel), cv2.NORM_L1)
    want_noise = l1 * np.sqrt(0.5 * np.pi) / (6 * (w - 2) * (h - 2))
    dh = cv2.absdiff(gray[:, 1:], gray[:, :-1])
    dv = cv2.absdiff(gray[1:, :], gray[:-1, :])
    cols, rows = np.arange(7, w - 1, 8), np.arange(7, h - 1, 8)
    want_block = float(max(((dh[:, cols].mean() - dh.mean()) + (dv[rows, :].mean() - dv.mean())) / 2, 0))

    scratch = np.full_like(gray, np.nan)
    assert ciq.noise_estimate(gray, out=scratch) == want_noise
    assert ciq.blockiness_score(gray, out=scratch) == want_block
    assert ciq.noise_estimate(gray) == want_noise and ciq.blockiness_score(gray) == want_block
    print("  ok  noise and blockiness through a shared, stale scratch equal fresh-array values exactly")


_MEASURE = textwrap.dedent("""
    import resource, subprocess, sys, os
    sys.path.insert(0, sys.argv[1])
    import cv2
    import compare_image_quality as ciq
    cv2.setNumThreads(1)  # as the scan runs it
    ciq.analyze(sys.argv[3])  # warm-up: lazy imports, the optional-metric latches
    rss = int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())])) * 1024
    ciq.analyze(sys.argv[2])
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    print(peak - rss)
""")


def test_peak_memory_fits_its_price(tmp: Path) -> None:
    """The budget is only a guard if its price is an upper bound: under-
    pricing is what swaps a 16 GB machine. Measured in a fresh process per
    file, since a process's peak RSS only ever goes up. 24 MP is big enough
    that the per-pixel part dominates -- and the old analyze() peaked at
    ~570 MB there, over the in-place price, so this also holds the saving."""
    img = textured(4000, 6000, seed=7)
    tiny = tmp / "tiny.jpg"
    cv2.imwrite(str(tiny), img[:64, :96])
    flat = tmp / "flat.jpg"
    cv2.imwrite(str(flat), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    rotated = tmp / "rotated.jpg"
    save_with_orientation(img, rotated, 6)
    repo = str(Path(__file__).resolve().parent.parent)
    for p, route in ((flat, "in_place"), (rotated, "cv2")):
        plan = ciq.decode_plan(str(p))
        assert plan is not None and plan[2] == route, (p.name, plan)
        out = subprocess.run([sys.executable, "-c", _MEASURE, repo, str(p), str(tiny)],
                             capture_output=True, text=True, check=True)
        peak, price = int(out.stdout.split()[-1]), dc._analyze_cost(str(p))
        assert peak <= price, f"{route}: analyze() peaked at {peak / 1e6:.0f} MB, priced at {price / 1e6:.0f} MB"
        print(f"  ok  {route}: 24 MP peak {peak / 1e6:.0f} MB <= price {price / 1e6:.0f} MB")


def main() -> None:
    tests = [
        test_every_orientation_decodes_like_the_plain_call,
        test_a_silent_decode_failure_falls_back,
        test_shared_scratch_matches_fresh_arrays,
        test_peak_memory_fits_its_price,
    ]
    for test in tests:
        print(f"{test.__name__}:")
        if test.__code__.co_argcount:
            with tempfile.TemporaryDirectory() as tmp:
                test(Path(tmp))
        else:
            test()
    print("all in-place analyze tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)
