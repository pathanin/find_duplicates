"""Regression tests for the fast-scanning path in duplicates_core.py:
reduced-resolution perceptual hashing and the (mtime, size)-keyed hash and
analyze() caches backing the parallel thread pool. Those caches are in-memory
and caller-owned -- a scan writes nothing to the scanned directory -- so the
tests here cover both the speedup and that absence of on-disk residue.

Run: python3 test_fast_scan.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
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

        result = dc.load_hash_gray(p)
        reduced = cv2.imread(str(p), cv2.IMREAD_REDUCED_GRAYSCALE_8)
        assert result.shape == reduced.shape, "expected the fast reduced-decode path for a large image"
        assert min(result.shape) >= dc.MIN_REDUCED_DECODE_SIDE
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
        assert min(reduced.shape) < dc.MIN_REDUCED_DECODE_SIDE, (
            "test file must be small enough to trigger the fallback; adjust dimensions if this fails"
        )

        result = dc.load_hash_gray(p)
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

        assert dc.cached_result(cache, p, _stat(p)) is None, "empty cache must miss"

        result = {"path": str(p), "dimensions": (400, 400), "sharpness_normalized": 12.3}
        dc.store_result(cache, p, _stat(p), result)

        hit = dc.cached_result(cache, p, _stat(p))
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
        dc.store_result(cache, p, _stat(p), {"path": str(p), "dimensions": (400, 400)})
        assert dc.cached_result(cache, p, _stat(p)) is not None

        save_jpeg(make_texture(400, 400, seed=5), p)  # different content, same path
        assert dc.cached_result(cache, p, _stat(p)) is None, "modified file must miss the cache"
        print("  ok  modifying the file invalidates its cache entry")


def test_hash_cache_round_trip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "photo.jpg"
        save_jpeg(make_texture(400, 400, seed=50), p)
        cache: dict = {}

        assert dc.cached_hash(cache, p, _stat(p)) is None, "empty cache must miss"

        dc.store_hash(cache, p, _stat(p), (12345, 67890))

        hit = dc.cached_hash(cache, p, _stat(p))
        assert hit == (12345, 67890), "expected a hash cache hit right after storing"
        print("  ok  hash cache hit returns the stored hash")


def test_hash_cache_miss_after_modification() -> None:
    """Boundary: any change to mtime or size must invalidate, even with a
    stale entry still present under the same path key."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "photo.jpg"
        save_jpeg(make_texture(400, 400, seed=51), p)
        cache: dict = {}
        dc.store_hash(cache, p, _stat(p), (999, 111))
        assert dc.cached_hash(cache, p, _stat(p)) == (999, 111)

        save_jpeg(make_texture(400, 400, seed=52), p)  # different content, same path
        assert dc.cached_hash(cache, p, _stat(p)) is None, "modified file must miss the hash cache"
        print("  ok  modifying the file invalidates its hash cache entry")


def test_group_duplicates_skips_decode_on_all_cache_hits() -> None:
    """When every path's hash is already cached, group_duplicates must not
    call load_hash_gray again -- the actual benefit a hash cache is for:
    re-scanning an already-hashed directory shouldn't re-decode old files."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = make_duplicate_pair(tmp, seed=40)
        cache: dict = {}
        dc.group_duplicates(paths, dc.DEFAULT_HASH_THRESHOLD, cache)  # warm the cache

        def exploding_load(_p):
            raise AssertionError("load_hash_gray must not be called on an all-cache-hit run")

        real_load = dc.load_hash_gray
        dc.load_hash_gray = exploding_load
        try:
            groups = dc.group_duplicates(paths, dc.DEFAULT_HASH_THRESHOLD, cache)
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
        groups = dc.group_duplicates(paths, dc.DEFAULT_HASH_THRESHOLD, cache)
        assert len(groups) == 1 and len(groups[0]) == 2, "expected the near-duplicate pair to be grouped"
        for p in paths:
            assert str(p.resolve()) in cache, "a computed hash must be written back into the cache"
            assert dc.cached_hash(cache, p, p.stat()) is not None
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

        real_process_pool = dc.ProcessPoolExecutor
        dc.ProcessPoolExecutor = ExplodingProcessPool
        dc.ThreadPoolExecutor = RecordingThreadPool
        try:
            groups = dc.group_duplicates(paths, dc.DEFAULT_HASH_THRESHOLD, {})
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

        groups = dc.group_duplicates(paths, dc.DEFAULT_HASH_THRESHOLD, {})

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
        dc.store_result(
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
            analyzed = dc.analyze_paths([p], cache)
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

        analyzed = dc.analyze_paths([p], cache)
        r = analyzed[p]
        assert r["dimensions"] == (300, 300)
        assert r["file_size"] == p.stat().st_size
        assert isinstance(r["sharpness_normalized"], float)

        assert str(p.resolve()) in cache, "a computed result must be written back into the cache"
        cached = dc.cached_result(cache, p, p.stat())
        assert cached is not None and cached["dimensions"] == (300, 300)
        print("  ok  cache miss computes via the thread pool and is written back to cache")


def test_analyze_paths_analyzes_small_batch_via_thread_pool() -> None:
    """Even a tiny uncached batch must analyze through a real
    ThreadPoolExecutor, never a ProcessPoolExecutor -- see the comments at
    THREAD_POOL_WORKERS's definition in duplicates_core.py: analyze()'s
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
            analyzed = dc.analyze_paths(paths, {})
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

        analyzed = dc.analyze_paths(paths, {})

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

        analyzed = dc.analyze_paths([p], {}, precomputed_stats=precomputed)

        assert analyzed[p]["dimensions"] == (300, 300)
        assert analyzed[p]["file_size"] == fake_size, (
            f"expected file_size from precomputed_stats ({fake_size}), "
            f"got {analyzed[p]['file_size']} (real size is {real_st.st_size}) "
            "-- precomputed_stats appears to be ignored"
        )
        print("  ok  precomputed_stats values (not a fresh stat()) determine the result")


def test_real_analyze_result_is_json_serializable() -> None:
    """analyze() emits np.float64 for several metrics, which only serializes
    today because np.float64 subclasses Python float. These values reach the
    browser through _group_detail's JSON payload, so a future metric that
    isn't JSON-safe (e.g. a bare np.int64, or np.bool_ -- which doesn't
    subclass bool and already broke is_close_call once) would 500 the
    /api/group route. Serialize the *actual* analyze() output rather than an
    in-memory dict of hand-picked plain floats, so that fails loudly here.

    This used to assert the same property via a save_cache/load_cache round
    trip through a JSON file on disk; the caches are in-memory now, but the
    JSON constraint they incidentally enforced is still real."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "photo.jpg"
        save_jpeg(make_texture(300, 300, seed=8), p)

        analyzed = dc.analyze_paths([p], {})

        encoded = json.dumps(analyzed[p])  # raises TypeError on a non-JSON-safe metric
        decoded = json.loads(encoded)
        assert tuple(decoded["dimensions"]) == (300, 300)
        assert decoded["sharpness_normalized"] == analyzed[p]["sharpness_normalized"]
        print("  ok  real analyze() output is JSON-serializable for the API payload")


def test_scan_writes_nothing_into_the_scanned_directory() -> None:
    """The scan caches are in-memory only: build_groups must leave the user's
    photo directory byte-for-byte as it found it. Compares the full directory
    listing rather than probing for specific filenames, so a stray atomic-write
    temp file would fail here too. Uses a real duplicate pair on purpose -- a
    directory with no duplicates never reaches the analyze phase, so it would
    only exercise half the scan."""
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        make_duplicate_pair(tmp, seed=70)
        before = set(os.listdir(directory))

        groups = dc.build_groups(directory, dc.DEFAULT_HASH_THRESHOLD, dest_dir=directory / "_duplicates")

        assert len(groups) == 1, f"expected the pair to group, got {len(groups)} group(s)"
        after = set(os.listdir(directory))
        assert after == before, f"scan left files behind in the scanned directory: {sorted(after - before)}"
        print("  ok  a full scan writes nothing into the scanned directory")


def test_build_groups_reuses_caller_supplied_caches_across_scans() -> None:
    """The rescan speedup the on-disk caches used to provide, now carried by
    caller-owned dicts (duplicates_web.Session holds them for the control
    panel's rescan button). A second build_groups over unchanged files must
    hit both caches: no re-decode for hashing, and no re-analyze."""
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        paths = make_duplicate_pair(tmp, seed=71)
        hash_cache: dict = {}
        analyze_cache: dict = {}
        dest = directory / "_duplicates"

        dc.build_groups(directory, dc.DEFAULT_HASH_THRESHOLD, dest_dir=dest,
                        hash_cache=hash_cache, analyze_cache=analyze_cache)

        for p in paths:
            assert dc.cached_hash(hash_cache, p, p.stat()) is not None, "first scan must fill the hash cache"
            assert dc.cached_result(analyze_cache, p, p.stat()) is not None, "first scan must fill the analyze cache"

        def exploding_load(_p):
            raise AssertionError("load_hash_gray must not be called on a warm-cache rescan")

        def exploding_analyze(_p):
            raise AssertionError("analyze must not be called on a warm-cache rescan")

        real_load, real_analyze = dc.load_hash_gray, dc.analyze
        dc.load_hash_gray, dc.analyze = exploding_load, exploding_analyze
        try:
            groups = dc.build_groups(directory, dc.DEFAULT_HASH_THRESHOLD, dest_dir=dest,
                                     hash_cache=hash_cache, analyze_cache=analyze_cache)
        finally:
            dc.load_hash_gray, dc.analyze = real_load, real_analyze

        assert len(groups) == 1, f"rescan must rebuild the group from cache alone, got {len(groups)} group(s)"
        assert set(groups[0].paths) == set(paths), "rescan's group must hold the same files as the cold scan's"
        print("  ok  a rescan with warm caller-owned caches re-decodes and re-analyzes nothing")


def test_omitted_caches_default_to_a_cold_scan() -> None:
    """Boundary: the no-cache-argument call (find_duplicates.py's --auto path)
    must still work and must not leak state between calls through a shared
    module-level default -- two independent scans, both fully cold."""
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        make_duplicate_pair(tmp, seed=72)
        dest = directory / "_duplicates"

        first = dc.build_groups(directory, dc.DEFAULT_HASH_THRESHOLD, dest_dir=dest)
        assert len(first) == 1

        def exploding_load(_p):
            raise AssertionError("a cache-less scan must recompute, not reuse hidden global state")

        real_load = dc.load_hash_gray
        dc.load_hash_gray = exploding_load
        try:
            dc.build_groups(directory, dc.DEFAULT_HASH_THRESHOLD, dest_dir=dest)
        except AssertionError as exc:
            assert "must recompute" in str(exc)
        else:
            raise AssertionError("second cache-less scan reused state it should not have had")
        finally:
            dc.load_hash_gray = real_load
        print("  ok  omitting the cache arguments gives each scan a fresh, unshared cache")


def test_group_duplicates_tolerates_a_path_that_vanished_before_stat() -> None:
    """A file deleted between find_images() and the hash phase's stat sweep
    used to raise FileNotFoundError out of `{p: p.stat() for p in paths}` and
    abort the entire scan. Everywhere else in the module an unreadable file is
    simply dropped; the stat sweep must behave the same way."""
    with tempfile.TemporaryDirectory() as tmp:
        p1, p2 = make_duplicate_pair(tmp, seed=7)
        ghost = Path(tmp) / "vanished.jpg"

        groups = dc.group_duplicates([p1, ghost, p2], dc.DEFAULT_HASH_THRESHOLD, {})

        assert len(groups) == 1, f"expected the real pair to still group, got {groups}"
        assert sorted(groups[0]) == sorted([p1, p2])
        print("  ok  a missing path is dropped instead of aborting the hash phase")


def test_build_groups_tolerates_a_file_vanishing_between_the_two_phases() -> None:
    """Boundary case for the second unguarded stat sweep: the file survives
    the hash phase, then disappears before build_groups() stats the grouped
    paths for analyze_paths(). That sweep used to abort the whole scan; the
    remaining members must still come back as a group."""
    with tempfile.TemporaryDirectory() as tmp:
        rng = np.random.default_rng(31)
        base = rng.integers(0, 255, size=(150, 200, 3), dtype=np.uint8)
        paths = []
        for name, side in (("a.jpg", 1600), ("b.jpg", 800), ("c.jpg", 400)):
            q = Path(tmp) / name
            cv2.imwrite(str(q), cv2.resize(base, (side, side * 3 // 4), interpolation=cv2.INTER_CUBIC),
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            paths.append(q)

        real_group_duplicates = dc.group_duplicates

        def vanishing_group_duplicates(*args, **kwargs):
            raw = real_group_duplicates(*args, **kwargs)
            paths[2].unlink()  # gone after hashing, before the analyze-phase stat
            return raw

        dc.group_duplicates = vanishing_group_duplicates
        try:
            groups = dc.build_groups(Path(tmp), dc.DEFAULT_HASH_THRESHOLD)
        finally:
            dc.group_duplicates = real_group_duplicates

        assert len(groups) == 1, f"expected one surviving group, got {len(groups)}"
        assert sorted(groups[0].paths) == sorted(paths[:2]), (
            f"expected the two surviving files, got {groups[0].paths}"
        )
        print("  ok  a file vanishing between phases costs that file, not the scan")


def main() -> None:
    tests = [
        test_load_hash_gray_uses_reduced_decode_for_normal_size,
        test_load_hash_gray_falls_back_to_full_for_small_export,
        test_cache_round_trip,
        test_cache_miss_after_modification,
        test_hash_cache_round_trip,
        test_hash_cache_miss_after_modification,
        test_group_duplicates_skips_decode_on_all_cache_hits,
        test_group_duplicates_computes_and_caches_on_miss,
        test_group_duplicates_hashes_small_batch_via_thread_pool,
        test_group_duplicates_uses_thread_pool_and_groups_correctly,
        test_analyze_paths_skips_pool_on_all_cache_hits,
        test_analyze_paths_computes_and_caches_on_miss,
        test_analyze_paths_analyzes_small_batch_via_thread_pool,
        test_analyze_paths_uses_thread_pool_for_larger_batch,
        test_analyze_paths_honors_precomputed_stats,
        test_real_analyze_result_is_json_serializable,
        test_scan_writes_nothing_into_the_scanned_directory,
        test_build_groups_reuses_caller_supplied_caches_across_scans,
        test_omitted_caches_default_to_a_cold_scan,
        test_group_duplicates_tolerates_a_path_that_vanished_before_stat,
        test_build_groups_tolerates_a_file_vanishing_between_the_two_phases,
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
