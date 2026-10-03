"""Regression test for Pillow's decompression-bomb guard on large photos.

Pillow warns above MAX_IMAGE_PIXELS (~89 MP by default) and raises
DecompressionBombError above twice that (~179 MP). Ordinary camera and
cosplay-shoot JPEGs reach 100 MP (8742x11656), so the default spammed the
console on every thumbnail. Worse, above 2x the error was swallowed by
make_thumbnail/_render_scaled_jpeg's `except Exception` and the photo showed
up as a gray placeholder -- unjudgeable in review -- while a HEIC that size
(PIL is its only decoder) was silently dropped from the scan.

The check runs in Image.open on the header size, before any decode, so this
calls it directly instead of allocating a 180 MP image.

Run: python3 tests/test_large_image.py
"""

import io
import sys
import tempfile
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import duplicates_core as dc  # importing it must raise the limit


def test_large_photos_pass_the_bomb_check() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for size in [(8742, 11656), (13500, 13500)]:  # 102 MP, 182 MP (> 2x old default)
            PILImage._decompression_bomb_check(size)
    print("  ok  102 MP and 182 MP images open without warning or error")


def test_analyze_cost_reads_the_header() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "a.png"
        PILImage.new("L", (300, 200)).save(p)
        assert dc._analyze_cost(str(p)) == 300 * 200 * dc.ANALYZE_BYTES_PER_PIXEL
        assert dc._analyze_cost(str(Path(tmp) / "missing.jpg")) == 0
    print("  ok  analyze cost comes from the header; unreadable costs 0")


def test_analyze_stays_under_memory_budget() -> None:
    """analyze() peaks at ~2.7 GB on a 100 MP JPEG. Ten of those at once (one
    per core) swapped a 24 GB Mac to a crawl -- browser included -- so
    _analyze_one admits work against a byte budget. A file bigger than the
    whole budget must still run, alone, or the scan would hang on it."""
    costs = {"a": 60, "b": 60, "c": 60, "d": 60, "huge": 250, "e": 30, "f": 30}
    lock = threading.Lock()
    in_flight: dict[str, int] = {}
    violations: list[dict] = []

    def fake_analyze(p: str) -> dict:
        with lock:
            in_flight[p] = costs[p]
            if sum(in_flight.values()) > 100 and list(in_flight) != ["huge"]:
                violations.append(dict(in_flight))
        time.sleep(0.03)
        with lock:
            del in_flight[p]
        return {"path": p}

    saved = dc.analyze, dc._analyze_cost, dc.ANALYZE_MEMORY_BUDGET
    dc.analyze, dc._analyze_cost, dc.ANALYZE_MEMORY_BUDGET = fake_analyze, costs.__getitem__, 100
    try:
        with ThreadPoolExecutor(max_workers=8) as ex:
            out = list(ex.map(dc._analyze_one, costs))
    finally:
        dc.analyze, dc._analyze_cost, dc.ANALYZE_MEMORY_BUDGET = saved
    assert [r["path"] for r in out] == list(costs), "every file must be analyzed, in order"
    assert not violations, f"over budget: {violations}"
    print("  ok  concurrent analyze stays under the byte budget; oversized file runs alone")


def test_preview_render_uses_reduced_jpeg_decode() -> None:
    """A stage/thumb render of a 100 MP JPEG used to decode every pixel and
    convert to RGB first: ~1.5 GB peak and 0.7 s for an 800-1600 px preview.
    draft() lets libjpeg decode at 1/2-1/8 scale instead. It must keep at
    least 2x the target, the same margin Pillow's own reducing_gap keeps, or
    the preview the keep decision is made on loses detail."""
    import duplicates_web
    from PIL import JpegImagePlugin

    calls = []
    real_draft = JpegImagePlugin.JpegImageFile.draft

    def spy(self, mode, size):
        calls.append(size)
        return real_draft(self, mode, size)

    JpegImagePlugin.JpegImageFile.draft = spy
    try:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "big.jpg"
            PILImage.new("RGB", (4000, 3000), (200, 120, 40)).save(p)
            out = PILImage.open(io.BytesIO(duplicates_web._render_scaled_jpeg(p, 800, 85)))
    finally:
        JpegImagePlugin.JpegImageFile.draft = real_draft
    assert calls == [(1600, 1600)], f"expected one draft at 2x the box, got {calls}"
    assert out.size == (800, 600), out.size
    print("  ok  preview render drafts the JPEG at 2x the target box")


def test_preview_renders_are_capped() -> None:
    """Renders moved off the event loop onto to_thread's pool (~14 threads),
    and opening a group requests a stage render per candidate plus a thumb
    each plus the next group's prefetch -- a dozen 100 MP decodes at ~0.8 GB
    apiece, on top of a streaming scan's analyze budget. On the loop they had
    run one at a time by accident; RENDER_SLOTS keeps that bound on purpose."""
    import duplicates_web

    lock = threading.Lock()
    live = [0]
    peak = [0]
    real_open = PILImage.open

    def slow_open(path, *a, **k):
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        time.sleep(0.03)
        with lock:
            live[0] -= 1
        raise OSError("fake undecodable")  # falls through to the placeholder

    PILImage.open = slow_open
    try:
        with ThreadPoolExecutor(max_workers=12) as ex:
            list(ex.map(lambda _: duplicates_web._render_scaled_jpeg(Path("x.jpg"), 64, 85), range(12)))
    finally:
        PILImage.open = real_open
    assert peak[0] <= 2, f"{peak[0]} renders decoded at once"
    print(f"  ok  at most 2 preview renders decode at once (peak {peak[0]})")


def main() -> None:
    tests = [
        test_large_photos_pass_the_bomb_check,
        test_analyze_cost_reads_the_header,
        test_analyze_stays_under_memory_budget,
        test_preview_render_uses_reduced_jpeg_decode,
        test_preview_renders_are_capped,
    ]
    for test in tests:
        print(f"{test.__name__}:")
        test()
    print("all large image tests passed")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, Warning, PILImage.DecompressionBombError) as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)
