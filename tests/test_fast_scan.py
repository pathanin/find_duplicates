"""Regression tests for the fast-scanning path in find_duplicates.py:
reduced-resolution perceptual hashing and the (mtime, size)-keyed analyze()
cache backing the parallel process pool.

Run: python3 test_fast_scan.py
"""

import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import find_duplicates as fd


def make_texture(h: int, w: int, seed: int) -> np.ndarray:
    """Deterministic-but-non-trivial texture so phash has real structure to hash."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(h // 8 + 1, w // 8 + 1, 3), dtype=np.uint8)
    img = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)
    return img


def save_jpeg(img: np.ndarray, path: Path) -> None:
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 85])


def test_load_hash_gray_uses_reduced_decode_for_normal_size() -> None:
    """A normal-size image should take the fast ~1/8-scale decode path, not
    pay for a full decode it doesn't need."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "large.jpg"
        save_jpeg(make_texture(1200, 1600, seed=1), p)

        result = fd.load_hash_gray(p)
        reduced = cv2.imread(str(p), cv2.IMREAD_REDUCED_GRAYSCALE_8)
        assert result.shape == reduced.shape, "expected the fast reduced-decode path for a large image"
        assert min(result.shape) >= fd.MIN_REDUCED_DECODE_SIDE
        print(f"  ok  large image {reduced.shape}: took the reduced-decode fast path")


def test_load_hash_gray_falls_back_to_full_for_small_export() -> None:
    """Failure case this guards against: phash resizes to 32x32. A 1/8 decode
    of a small duplicate export (short side small enough to drop the reduced
    decode below 32px) upsamples on that axis, while a full decode of a large
    sibling image downsamples -- asymmetric resampling paths that drift the
    hash exactly where duplicate detection needs the two to agree. (Verified
    empirically on a real photo: a genuine large/small duplicate pair matched
    exactly at full decode but drifted 3/64 bits when the small side used a
    naive reduced-8 decode.) load_hash_gray must detect that the reduced
    decode landed too small and fall back to a full decode instead."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "small_export.jpg"
        save_jpeg(make_texture(160, 200, seed=2), p)

        reduced = cv2.imread(str(p), cv2.IMREAD_REDUCED_GRAYSCALE_8)
        assert min(reduced.shape) < fd.MIN_REDUCED_DECODE_SIDE, (
            "test file must be small enough to trigger the fallback; adjust dimensions if this fails"
        )

        result = fd.load_hash_gray(p)
        full = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        assert result.shape == full.shape, "expected fallback to full decode for a small export"
        print(f"  ok  small export {full.shape} (reduced would be {reduced.shape}): fell back to full decode")


def _stat(p: Path) -> os.stat_result:
    return p.stat()


def test_cache_round_trip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "photo.jpg"
        save_jpeg(make_texture(400, 400, seed=3), p)
        cache: dict = {}

        assert fd.cached_result(cache, p, _stat(p)) is None, "empty cache must miss"

        result = {"path": str(p), "dimensions": (400, 400), "sharpness_normalized": 12.3}
        fd.store_result(cache, p, _stat(p), result)

        hit = fd.cached_result(cache, p, _stat(p))
        assert hit is not None, "expected a cache hit right after storing"
        assert hit["dimensions"] == (400, 400), "dimensions must round-trip as a tuple, not a list"
        assert hit["sharpness_normalized"] == 12.3
        print("  ok  cache hit returns stored result with dimensions restored as tuple")


def test_cache_miss_after_modification() -> None:
    """Boundary: any change to mtime or size must invalidate, even with a
    stale entry still present under the same path key."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "photo.jpg"
        save_jpeg(make_texture(400, 400, seed=4), p)
        cache: dict = {}
        fd.store_result(cache, p, _stat(p), {"path": str(p), "dimensions": (400, 400)})
        assert fd.cached_result(cache, p, _stat(p)) is not None

        save_jpeg(make_texture(400, 400, seed=5), p)  # different content, same path
        assert fd.cached_result(cache, p, _stat(p)) is None, "modified file must miss the cache"
        print("  ok  modifying the file invalidates its cache entry")


def test_load_cache_handles_corrupt_file() -> None:
    """Failure case: a truncated/corrupt cache file must not crash a scan,
    just be treated as empty."""
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        (directory / fd.CACHE_FILENAME).write_text("{not valid json")
        cache = fd.load_cache(directory)
        assert cache == {}, "corrupt cache file should load as empty, not raise"
        print("  ok  corrupt cache file loads as {} instead of raising")


def test_analyze_paths_skips_pool_on_all_cache_hits() -> None:
    """When every path is already cached, analyze_paths must not construct a
    ProcessPoolExecutor at all (no subprocess spawn cost for a warm re-run)."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "photo.jpg"
        save_jpeg(make_texture(300, 300, seed=6), p)
        cache: dict = {}
        fd.store_result(
            cache, p, _stat(p),
            {"path": str(p), "dimensions": (300, 300), "sharpness_normalized": 1.0,
             "effective_resolution_fraction": 0.9, "effective_resolution_px_equiv": 270.0,
             "noise_sigma": 0.1, "blockiness": 0.01, "brisque": None, "niqe": None},
        )

        class ExplodingPool:
            def __init__(self, *a, **k):
                raise AssertionError("ProcessPoolExecutor must not be constructed on an all-cache-hit run")

        real_pool = fd.ProcessPoolExecutor
        fd.ProcessPoolExecutor = ExplodingPool
        try:
            analyzed = fd.analyze_paths([p], cache)
        finally:
            fd.ProcessPoolExecutor = real_pool

        assert analyzed[p]["file_size"] == p.stat().st_size
        assert analyzed[p]["dimensions"] == (300, 300)
        print("  ok  all-cache-hit run never constructs a process pool")


def test_analyze_paths_computes_and_caches_on_miss() -> None:
    """Smoke test: an actual cache miss goes through the real process pool
    and produces a usable, cacheable result."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "photo.jpg"
        save_jpeg(make_texture(300, 300, seed=7), p)
        cache: dict = {}

        analyzed = fd.analyze_paths([p], cache)
        r = analyzed[p]
        assert r["dimensions"] == (300, 300)
        assert r["file_size"] == p.stat().st_size
        assert isinstance(r["sharpness_normalized"], float)

        assert str(p.resolve()) in cache, "a computed result must be written back into the cache"
        cached = fd.cached_result(cache, p, p.stat())
        assert cached is not None and cached["dimensions"] == (300, 300)
        print("  ok  cache miss computes via the process pool and is written back to cache")


def test_real_analyze_result_round_trips_through_json_cache_file() -> None:
    """analyze() emits np.float64 for several metrics, which only serializes
    today because np.float64 subclasses Python float. Prove the *actual*
    analyze() -> cache dict -> JSON file -> reload -> hit path works end to
    end, not just an in-memory dict of hand-picked plain floats -- so a
    future metric that isn't JSON-safe (e.g. a bare np.int64) fails loudly
    here instead of silently crashing a real scan's save_cache() call."""
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        p = directory / "photo.jpg"
        save_jpeg(make_texture(300, 300, seed=8), p)

        cache = fd.load_cache(directory)
        analyzed = fd.analyze_paths([p], cache)
        fd.save_cache(directory, cache)

        reloaded_cache = fd.load_cache(directory)
        hit = fd.cached_result(reloaded_cache, p, p.stat())
        assert hit is not None, "real analyze() output failed to round-trip through the JSON cache file"
        assert hit["dimensions"] == (300, 300)
        assert hit["sharpness_normalized"] == analyzed[p]["sharpness_normalized"]
        print("  ok  real analyze() output round-trips through save_cache/load_cache on disk")


def main() -> None:
    tests = [
        test_load_hash_gray_uses_reduced_decode_for_normal_size,
        test_load_hash_gray_falls_back_to_full_for_small_export,
        test_cache_round_trip,
        test_cache_miss_after_modification,
        test_load_cache_handles_corrupt_file,
        test_analyze_paths_skips_pool_on_all_cache_hits,
        test_analyze_paths_computes_and_caches_on_miss,
        test_real_analyze_result_round_trips_through_json_cache_file,
    ]
    for test in tests:
        print(f"{test.__name__}:")
        test()
    print("all fast-scan tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)
