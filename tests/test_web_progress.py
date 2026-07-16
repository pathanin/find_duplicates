"""Tests for the progress_callback parameter added to group_duplicates()/
analyze_paths()/build_groups() in duplicates_core.py, ahead of the browser
front end (find_duplicates-web.py) that needs scan progress delivered as
data (for an SSE stream) rather than printed to stdout.

Purely additive: progress_callback defaults to None, which must reproduce
the exact stdout behavior test_scan_progress.py already locks in. When a
callback is given, no stdout printing should happen at all -- a web session
has no terminal to write \r-lines into.

Run: python3 tests/test_web_progress.py
"""

import contextlib
import io
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import duplicates_core as dc


def make_texture(h: int, w: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(h // 8 + 1, w // 8 + 1, 3), dtype=np.uint8)
    return cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)


def save_jpeg(img: np.ndarray, path: Path) -> None:
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])


def make_images(tmp: str, n: int) -> list[Path]:
    paths = []
    for i in range(n):
        h, w = 100 + i * 20, 120 + i * 17
        p = Path(tmp) / f"img_{i}.jpg"
        save_jpeg(make_texture(h, w, seed=2000 + i), p)
        paths.append(p)
    return paths


def test_group_duplicates_calls_progress_callback_with_label_done_total() -> None:
    """The callback must fire once per completed item, in order, with the
    exact (label, done, total) shape a caller threading this into an SSE
    stream would need."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = make_images(tmp, 4)
        events = []
        dc.group_duplicates(
            paths, dc.DEFAULT_HASH_THRESHOLD, {}, progress_callback=lambda *a: events.append(a)
        )
        assert events, "expected at least one progress_callback call for an uncached batch"
        assert all(e[0] == "Hashing" for e in events), f"expected label 'Hashing' throughout, got {events}"
        assert [e[1] for e in events] == list(range(1, len(paths) + 1)), (
            f"expected done to count up 1..N in order, got {[e[1] for e in events]}"
        )
        assert all(e[2] == len(paths) for e in events), f"expected total == {len(paths)} on every call, got {events}"
        print("  ok  group_duplicates calls progress_callback with (label, done, total) in order")


def test_analyze_paths_calls_progress_callback_with_label_done_total() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        paths = make_images(tmp, 3)
        events = []
        dc.analyze_paths(paths, {}, progress_callback=lambda *a: events.append(a))
        assert events, "expected at least one progress_callback call for an uncached batch"
        assert all(e[0] == "Analyzing" for e in events), f"expected label 'Analyzing' throughout, got {events}"
        assert [e[1] for e in events] == list(range(1, len(paths) + 1))
        assert all(e[2] == len(paths) for e in events)
        print("  ok  analyze_paths calls progress_callback with (label, done, total) in order")


def test_progress_callback_suppresses_stdout_printing() -> None:
    """A web session has no terminal -- when a callback is given, nothing
    should go to stdout at all, not even the non-TTY fallback lines."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = make_images(tmp, 3)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            dc.group_duplicates(paths, dc.DEFAULT_HASH_THRESHOLD, {}, progress_callback=lambda *a: None)
        assert buf.getvalue() == "", f"expected no stdout output when progress_callback is given, got: {buf.getvalue()!r}"
        print("  ok  passing progress_callback suppresses stdout printing entirely")


def test_no_progress_callback_preserves_default_print_behavior() -> None:
    """Boundary: progress_callback=None (the default) must reproduce the
    exact print-based behavior from before this parameter existed --
    test_scan_progress.py already locks in the fine detail of that stdout
    format, this just proves the new parameter doesn't disturb the
    default path at all."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = make_images(tmp, 3)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            dc.group_duplicates(paths, dc.DEFAULT_HASH_THRESHOLD, {})
        output = buf.getvalue()
        assert f"Hashing: {len(paths)}/{len(paths)}" in output, (
            f"expected default (no callback) run to still print a final done-count line, got: {output!r}"
        )
        print("  ok  omitting progress_callback preserves the original print-based default")


def test_build_groups_threads_progress_callback_through_both_phases() -> None:
    """build_groups must pass its progress_callback down to both
    group_duplicates (Hashing) and analyze_paths (Analyzing) -- a caller
    driving a single SSE stream off one callback needs both phases to
    report through it, not just the first."""
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        # Two near-duplicate photos (same base texture, two sizes) so
        # build_groups actually reaches the analyze phase -- a lone
        # unmatched file never gets analyzed at all.
        rng = np.random.default_rng(99)
        base = rng.integers(0, 255, size=(150, 200, 3), dtype=np.uint8)
        big = cv2.resize(base, (1600, 1200), interpolation=cv2.INTER_CUBIC)
        small = cv2.resize(base, (400, 300), interpolation=cv2.INTER_CUBIC)
        save_jpeg(big, directory / "big.jpg")
        save_jpeg(small, directory / "small.jpg")

        labels_seen = set()
        dc.build_groups(
            directory, dc.DEFAULT_HASH_THRESHOLD,
            progress_callback=lambda label, done, total: labels_seen.add(label),
        )
        assert labels_seen == {"Hashing", "Analyzing"}, (
            f"expected both phases to report through progress_callback, got labels: {labels_seen}"
        )
        print("  ok  build_groups threads progress_callback through both the hash and analyze phases")


def main() -> None:
    tests = [
        test_group_duplicates_calls_progress_callback_with_label_done_total,
        test_analyze_paths_calls_progress_callback_with_label_done_total,
        test_progress_callback_suppresses_stdout_printing,
        test_no_progress_callback_preserves_default_print_behavior,
        test_build_groups_threads_progress_callback_through_both_phases,
    ]
    for test in tests:
        print(f"{test.__name__}:")
        test()
    print("all progress-callback tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)
