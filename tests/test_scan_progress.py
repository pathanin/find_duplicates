"""Tests for the progress reporting added to group_duplicates()/analyze_paths()
in find_duplicates.py: prints "\rHashing: n/N" / "\rAnalyzing: n/N" as each
uncached item completes, falling back to occasional plain lines when stdout
isn't a TTY (see _print_progress).

This is purely additive over the existing cached/to_compute split and pool
execution (a ThreadPoolExecutor for hashing, a ProcessPoolExecutor for
analyze) -- these tests exist because threading progress output through a
loop that zips against a pool's .map() result is exactly the kind of change
that could silently break the order-pairing that store_hash()/store_result()
rely on (e.g. enumerating one iterator but zipping a different one), without
visibly breaking anything until results get attached to the wrong path.

Run: python3 tests/test_scan_progress.py
"""

import contextlib
import io
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import find_duplicates as fd


def make_texture(h: int, w: int, seed: int) -> np.ndarray:
    """Deterministic-but-non-trivial texture so phash/analyze have real
    structure to work on (a flat image would hash/measure degenerately)."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(h // 8 + 1, w // 8 + 1, 3), dtype=np.uint8)
    return cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)


def save_jpeg(img: np.ndarray, path: Path) -> None:
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])


def make_distinguishable_images(tmp: str, n: int) -> list[Path]:
    """n images with distinct dimensions and distinct content, so a
    mis-pairing between `to_compute` and a process-pool's results would
    attach the wrong hash/analysis to the wrong path instead of silently
    looking fine."""
    paths = []
    for i in range(n):
        h, w = 100 + i * 20, 120 + i * 17
        p = Path(tmp) / f"img_{i}.jpg"
        save_jpeg(make_texture(h, w, seed=1000 + i), p)
        paths.append(p)
    return paths


class _FakeTTYBuffer(io.StringIO):
    """StringIO that reports itself as a TTY, so we can exercise the
    \\r-overwrite branch of _print_progress without a real terminal."""

    def isatty(self) -> bool:
        return True


def test_group_duplicates_progress_preserves_order_and_correctness():
    """The pool path (executor.map() zipped against to_compute) is where an
    off-by-one or wrong-iterator bug in progress printing would silently
    misattribute a hash to the wrong file. group_duplicates always routes
    uncached files through a thread pool now, so no threshold-forcing is
    needed here -- verify every cached hash matches an independent direct
    recomputation for that exact path."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = make_distinguishable_images(tmp, 5)
        cache: dict = {}

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            fd.group_duplicates(paths, fd.DEFAULT_HASH_THRESHOLD, cache)

        for p in paths:
            cached = fd.cached_hash(cache, p, p.stat())
            assert cached is not None, f"{p} missing from cache after group_duplicates"
            assert cached == fd._hash_one(p), f"{p} got the wrong hash -- order-pairing broke"
        print("  ok  pool-path hashes are correctly paired to their own file, not shuffled")


def test_analyze_paths_progress_preserves_order_and_correctness():
    """Same order-pairing concern as above, for analyze_paths' pool path
    (always taken for any uncached batch). Each file has distinct real
    dimensions, so a mis-pairing shows up as a dimensions mismatch instead
    of silently passing."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = make_distinguishable_images(tmp, 5)
        expected_dims = {}
        for p in paths:
            img = cv2.imread(str(p))
            expected_dims[p] = (img.shape[1], img.shape[0])  # (width, height), matches analyze()'s convention

        cache: dict = {}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            analyzed = fd.analyze_paths(paths, cache)

        for p in paths:
            assert p in analyzed, f"{p} missing from analyze_paths result"
            assert analyzed[p]["dimensions"] == expected_dims[p], (
                f"{p} got another file's dimensions -- order-pairing broke"
            )
        print("  ok  pool-path analyze results are correctly paired to their own file")


def test_progress_emitted_and_final_count_matches_non_tty():
    """Boundary: stdout isn't a TTY (exactly what redirect_stdout's StringIO
    gives us), so _print_progress must take the plain-line fallback -- no
    bare \\r in the output -- and still report a final line whose count
    equals len(to_compute)."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = make_distinguishable_images(tmp, 3)
        cache: dict = {}

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            fd.group_duplicates(paths, fd.DEFAULT_HASH_THRESHOLD, cache)
        output = buf.getvalue()

        assert "\r" not in output, "non-TTY stdout must not get carriage-return-overwrite output"
        assert "Hashing: 3/3" in output, f"expected a final done-count line, got: {output!r}"
        print("  ok  non-TTY fallback prints plain progress lines with a correct final count")


def test_progress_emitted_with_carriage_return_when_tty():
    """The TTY branch must actually use \\r to overwrite the line, and reach
    the final n/n count -- proof the isatty()-true branch isn't dead code."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = make_distinguishable_images(tmp, 4)
        cache: dict = {}

        buf = _FakeTTYBuffer()
        with contextlib.redirect_stdout(buf):
            fd.group_duplicates(paths, fd.DEFAULT_HASH_THRESHOLD, cache)
        output = buf.getvalue()

        assert "\r" in output, "TTY stdout should get carriage-return-overwrite progress output"
        assert "Hashing: 4/4" in output, f"expected the final count in the output, got: {output!r}"
        print("  ok  TTY path overwrites via \\r and reaches the final count")


def test_no_progress_output_when_nothing_to_compute():
    """All-cache-hit run: to_compute is empty, so no progress line at all
    should print -- nothing to report, and no stray leading \\r or blank
    line to clutter the surrounding "Scanning..."/"Found N groups" prints."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = make_distinguishable_images(tmp, 2)
        cache: dict = {}
        fd.group_duplicates(paths, fd.DEFAULT_HASH_THRESHOLD, cache)  # warm the cache

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            fd.group_duplicates(paths, fd.DEFAULT_HASH_THRESHOLD, cache)
        assert buf.getvalue() == "", f"expected no progress output on an all-cache-hit run, got: {buf.getvalue()!r}"
        print("  ok  all-cache-hit run prints no progress output")


def test_analyze_paths_results_identical_with_and_without_capturing_progress():
    """The progress code must be purely additive: analyze_paths' returned
    results shouldn't differ depending on whether stdout is captured
    (non-TTY) or left alone -- content, not just presence, must match."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = make_distinguishable_images(tmp, 3)

        cache_a: dict = {}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            analyzed_a = fd.analyze_paths(paths, cache_a)

        cache_b: dict = {}
        analyzed_b = fd.analyze_paths(paths, cache_b)  # stdout left alone

        for p in paths:
            assert analyzed_a[p]["dimensions"] == analyzed_b[p]["dimensions"]
            assert analyzed_a[p]["sharpness_normalized"] == analyzed_b[p]["sharpness_normalized"]
        print("  ok  analyze_paths results are identical whether or not stdout is captured")


def main() -> None:
    tests = [
        test_group_duplicates_progress_preserves_order_and_correctness,
        test_analyze_paths_progress_preserves_order_and_correctness,
        test_progress_emitted_and_final_count_matches_non_tty,
        test_progress_emitted_with_carriage_return_when_tty,
        test_no_progress_output_when_nothing_to_compute,
        test_analyze_paths_results_identical_with_and_without_capturing_progress,
    ]
    for test in tests:
        print(f"{test.__name__}:")
        test()
    print("all scan-progress tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)
