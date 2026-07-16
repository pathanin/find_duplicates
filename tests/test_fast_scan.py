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
import duplicates_core as dc


def make_texture(h: int, w: int, seed: int) -> np.ndarray:
    """Deterministic-but-non-trivial texture so phash has real structure to hash."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(h // 8 + 1, w // 8 + 1, 3), dtype=np.uint8)
    img = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)
    return img


def save_jpeg(img: np.ndarray, path: Path) -> None:
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 85])


def make_duplicate_pair(tmp: str, seed: int) -> list[Path]:
    """Same source texture resized to two different sizes and re-exported at
    different JPEG quality -- perceptually close enough that group_duplicates
    should group them at DEFAULT_HASH_THRESHOLD, the same shape of "real"
    duplicate the tool is meant to catch."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(150, 200, 3), dtype=np.uint8)
    big = cv2.resize(base, (1600, 1200), interpolation=cv2.INTER_CUBIC)
    small = cv2.resize(base, (400, 300), interpolation=cv2.INTER_CUBIC)
    p1 = Path(tmp) / "big.jpg"
    p2 = Path(tmp) / "small.jpg"
    cv2.imwrite(str(p1), big, [cv2.IMWRITE_JPEG_QUALITY, 90])
    cv2.imwrite(str(p2), small, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return [p1, p2]


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


def test_hash_cache_round_trip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "photo.jpg"
        save_jpeg(make_texture(400, 400, seed=50), p)
        cache: dict = {}

        assert fd.cached_hash(cache, p, _stat(p)) is None, "empty cache must miss"

        fd.store_hash(cache, p, _stat(p), 12345)

        hit = fd.cached_hash(cache, p, _stat(p))
        assert hit == 12345, "expected a hash cache hit right after storing"
        print("  ok  hash cache hit returns the stored hash")


def test_hash_cache_miss_after_modification() -> None:
    """Boundary: any change to mtime or size must invalidate, even with a
    stale entry still present under the same path key."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "photo.jpg"
        save_jpeg(make_texture(400, 400, seed=51), p)
        cache: dict = {}
        fd.store_hash(cache, p, _stat(p), 999)
        assert fd.cached_hash(cache, p, _stat(p)) == 999

        save_jpeg(make_texture(400, 400, seed=52), p)  # different content, same path
        assert fd.cached_hash(cache, p, _stat(p)) is None, "modified file must miss the hash cache"
        print("  ok  modifying the file invalidates its hash cache entry")


def test_load_hash_cache_handles_corrupt_file() -> None:
    """Failure case: a truncated/corrupt hash cache file must not crash a
    scan, just be treated as empty."""
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        (directory / fd.HASH_CACHE_FILENAME).write_text("{not valid json")
        cache = fd.load_hash_cache(directory)
        assert cache == {}, "corrupt hash cache file should load as empty, not raise"
        print("  ok  corrupt hash cache file loads as {} instead of raising")


def test_group_duplicates_skips_decode_on_all_cache_hits() -> None:
    """When every path's hash is already cached, group_duplicates must not
    call load_hash_gray again -- the actual benefit a hash cache is for:
    re-scanning an already-hashed directory shouldn't re-decode old files."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = make_duplicate_pair(tmp, seed=40)
        cache: dict = {}
        fd.group_duplicates(paths, fd.DEFAULT_HASH_THRESHOLD, cache)  # warm the cache

        def exploding_load(_p):
            raise AssertionError("load_hash_gray must not be called on an all-cache-hit run")

        # Patched on duplicates_core, not fd: group_duplicates/_hash_one live
        # there now and resolve the bare name `load_hash_gray` in that
        # module's own globals -- reassigning fd.load_hash_gray only rebinds
        # find_duplicates's re-exported alias, which _hash_one never looks at.
        real_load = dc.load_hash_gray
        dc.load_hash_gray = exploding_load
        try:
            groups = fd.group_duplicates(paths, fd.DEFAULT_HASH_THRESHOLD, cache)
        finally:
            dc.load_hash_gray = real_load

        assert len(groups) == 1 and len(groups[0]) == 2, "expected the pair to still be grouped from cached hashes"
        print("  ok  all-cache-hit run never re-decodes for hashing")


def test_group_duplicates_computes_and_caches_on_miss() -> None:
    """Smoke test: an actual hash-cache miss computes real hashes, groups
    the near-duplicate pair correctly, and writes the hashes back to cache."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = make_duplicate_pair(tmp, seed=41)
        cache: dict = {}
        groups = fd.group_duplicates(paths, fd.DEFAULT_HASH_THRESHOLD, cache)
        assert len(groups) == 1 and len(groups[0]) == 2, "expected the near-duplicate pair to be grouped"
        for p in paths:
            assert str(p.resolve()) in cache, "a computed hash must be written back into the cache"
            assert fd.cached_hash(cache, p, p.stat()) is not None
        print("  ok  cache miss computes hashes via the real pipeline and writes them back to cache")


def test_group_duplicates_hashes_small_batch_via_thread_pool() -> None:
    """Even a tiny uncached batch must still hash through a real
    ThreadPoolExecutor, never a ProcessPoolExecutor -- see the comment at
    THREAD_POOL_WORKERS: cv2's decode/resize/dct calls release the GIL, so
    threads always win over both serial execution and a process pool's
    spawn cost, with no threshold to gate on."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i in range(3):
            p = Path(tmp) / f"photo_{i}.jpg"
            save_jpeg(make_texture(200, 200, seed=60 + i), p)
            paths.append(p)

        class ExplodingProcessPool:
            def __init__(self, *a, **k):
                raise AssertionError("group_duplicates must never construct a ProcessPoolExecutor")

        real_thread_cls = dc.ThreadPoolExecutor
        constructed = []

        class RecordingThreadPool(real_thread_cls):
            def __init__(self, *a, **k):
                constructed.append(True)
                super().__init__(*a, **k)

        # Patched on duplicates_core: group_duplicates is defined there now
        # and resolves ThreadPoolExecutor/ProcessPoolExecutor in that
        # module's own globals -- see the load_hash_gray patch above for why
        # patching fd's re-exported alias wouldn't be seen by the function.
        real_process_pool = dc.ProcessPoolExecutor
        dc.ProcessPoolExecutor = ExplodingProcessPool
        dc.ThreadPoolExecutor = RecordingThreadPool
        try:
            groups = fd.group_duplicates(paths, fd.DEFAULT_HASH_THRESHOLD, {})
        finally:
            dc.ProcessPoolExecutor = real_process_pool
            dc.ThreadPoolExecutor = real_thread_cls
        assert constructed, "expected a real ThreadPoolExecutor to be constructed for the small batch"
        assert isinstance(groups, list)
        print("  ok  a small uncached batch hashes via a real thread pool, no process pool constructed")


def test_group_duplicates_uses_thread_pool_and_groups_correctly() -> None:
    """Proof the thread pool path produces the *correct* grouping (a
    near-duplicate pair plus one unrelated filler file), not just that a
    pool object got created."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = make_duplicate_pair(tmp, seed=70)
        filler = Path(tmp) / "unrelated.jpg"
        save_jpeg(make_texture(200, 200, seed=72), filler)
        paths.append(filler)

        groups = fd.group_duplicates(paths, fd.DEFAULT_HASH_THRESHOLD, {})

        assert len(groups) == 1 and set(groups[0]) == set(paths[:2]), (
            f"expected the thread-pool path to group exactly the near-duplicate pair, got {groups}"
        )
        print("  ok  the thread-pool path still produces the correct grouping")


def test_analyze_paths_skips_pool_on_all_cache_hits() -> None:
    """When every path is already cached, analyze_paths must not construct
    any pool at all (no thread/process spawn cost for a warm re-run)."""
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
                raise AssertionError("no pool should be constructed on an all-cache-hit run")

        # Patched on duplicates_core -- see the group_duplicates test above.
        real_process_pool = dc.ProcessPoolExecutor
        real_thread_pool = dc.ThreadPoolExecutor
        dc.ProcessPoolExecutor = ExplodingPool
        dc.ThreadPoolExecutor = ExplodingPool
        try:
            analyzed = fd.analyze_paths([p], cache)
        finally:
            dc.ProcessPoolExecutor = real_process_pool
            dc.ThreadPoolExecutor = real_thread_pool

        assert analyzed[p]["file_size"] == p.stat().st_size
        assert analyzed[p]["dimensions"] == (300, 300)
        print("  ok  all-cache-hit run never constructs a pool")


def test_analyze_paths_computes_and_caches_on_miss() -> None:
    """Smoke test: an actual cache miss goes through the real thread pool
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
        print("  ok  cache miss computes via the thread pool and is written back to cache")


def test_analyze_paths_analyzes_small_batch_via_thread_pool() -> None:
    """Even a tiny uncached batch must analyze through a real
    ThreadPoolExecutor, never a ProcessPoolExecutor -- see the comments at
    THREAD_POOL_WORKERS's definition in find_duplicates.py: analyze()'s
    cv2/numpy calls release the GIL, so threads always win, with no
    threshold to gate on."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i in range(2):
            p = Path(tmp) / f"photo_{i}.jpg"
            save_jpeg(make_texture(200, 200, seed=80 + i), p)
            paths.append(p)

        class ExplodingProcessPool:
            def __init__(self, *a, **k):
                raise AssertionError("analyze_paths must never construct a ProcessPoolExecutor")

        real_thread_cls = dc.ThreadPoolExecutor
        constructed = []

        class RecordingThreadPool(real_thread_cls):
            def __init__(self, *a, **k):
                constructed.append(True)
                super().__init__(*a, **k)

        # Patched on duplicates_core -- see the group_duplicates test above.
        real_process_pool = dc.ProcessPoolExecutor
        dc.ProcessPoolExecutor = ExplodingProcessPool
        dc.ThreadPoolExecutor = RecordingThreadPool
        try:
            analyzed = fd.analyze_paths(paths, {})
        finally:
            dc.ProcessPoolExecutor = real_process_pool
            dc.ThreadPoolExecutor = real_thread_cls

        assert constructed, "expected a real ThreadPoolExecutor to be constructed for the small batch"
        for p in paths:
            assert analyzed[p]["dimensions"] == (200, 200)
        print("  ok  a small uncached batch analyzes via a real thread pool, no process pool constructed")


def test_analyze_paths_uses_thread_pool_for_larger_batch() -> None:
    """Proof the thread pool path produces correct results for every file
    at a larger batch size too, not just prove a pool object got created."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i in range(6):
            p = Path(tmp) / f"photo_{i}.jpg"
            save_jpeg(make_texture(200, 200, seed=90 + i), p)
            paths.append(p)

        analyzed = fd.analyze_paths(paths, {})

        for p in paths:
            assert analyzed[p]["dimensions"] == (200, 200)
            assert analyzed[p]["file_size"] == p.stat().st_size
        print("  ok  a larger batch routes through the thread pool and produces correct results for every file")


class _FakeStat:
    """Minimal stand-in for os.stat_result exposing only the two fields
    analyze_paths actually reads (st_size, st_mtime_ns)."""
    def __init__(self, st_size: int, st_mtime_ns: int) -> None:
        self.st_size = st_size
        self.st_mtime_ns = st_mtime_ns


def test_analyze_paths_honors_precomputed_stats() -> None:
    """build_groups() passes precomputed_stats to avoid re-stat()'ing files
    already stat()'d during the hash phase (9b6a6bd). Prove analyze_paths
    actually *uses* the passed-in stats -- rather than silently ignoring the
    parameter and deriving file_size from a fresh real stat() -- by handing
    it a deliberately wrong size and confirming that wrong value comes back
    out. (A blanket "stat() must never be called again" check doesn't work
    here: cached_result()'s str(p.resolve()) call itself invokes Path.stat()
    internally in this pathlib version, which is unrelated to the
    optimization being tested.)"""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "photo.jpg"
        save_jpeg(make_texture(300, 300, seed=95), p)
        real_st = p.stat()
        fake_size = real_st.st_size + 999_999
        precomputed = {p: _FakeStat(st_size=fake_size, st_mtime_ns=real_st.st_mtime_ns)}

        analyzed = fd.analyze_paths([p], {}, precomputed_stats=precomputed)

        assert analyzed[p]["dimensions"] == (300, 300)
        assert analyzed[p]["file_size"] == fake_size, (
            f"expected file_size from precomputed_stats ({fake_size}), "
            f"got {analyzed[p]['file_size']} (real size is {real_st.st_size}) "
            "-- precomputed_stats appears to be ignored"
        )
        print("  ok  precomputed_stats values (not a fresh stat()) determine the result")


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
        test_hash_cache_round_trip,
        test_hash_cache_miss_after_modification,
        test_load_hash_cache_handles_corrupt_file,
        test_group_duplicates_skips_decode_on_all_cache_hits,
        test_group_duplicates_computes_and_caches_on_miss,
        test_group_duplicates_hashes_small_batch_via_thread_pool,
        test_group_duplicates_uses_thread_pool_and_groups_correctly,
        test_analyze_paths_skips_pool_on_all_cache_hits,
        test_analyze_paths_computes_and_caches_on_miss,
        test_analyze_paths_analyzes_small_batch_via_thread_pool,
        test_analyze_paths_uses_thread_pool_for_larger_batch,
        test_analyze_paths_honors_precomputed_stats,
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
