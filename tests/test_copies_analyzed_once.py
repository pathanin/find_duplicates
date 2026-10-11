"""A byte-identical copy is decoded once for hashing and analyzed once, not
once per copy.

Byte-identical copies are the commonest duplicate there is (" copy.jpg",
" 2.jpg", a second export), and analyze() is the per-image scan bottleneck:
~0.3 s single-threaded on a 24 MP JPEG, ~1.3 s on a 100 MP one -- and a HEIC
spends ~0.9 s just decoding for its hash. The same bytes always hash and
score the same, so every copy after the first was pure waste: half the
analyze phase on tests/Test-image, which is nothing but such pairs.

Only files that share a size with another are looked at, and those get a
glance at three windows before anyone reads a file whole -- a folder of one
camera's uncompressed TIFFs is all one size. Then one file per content is
hashed and analyzed, and each copy is handed results of its own. Each part
of that is load-bearing:

- every copy still resolves through done_callback, or build_groups' per-group
  countdown never reaches zero and that group -- and, since publication is
  in order, every group after it -- is never published;
- each copy's result is its own dict, with its own "path": score_group
  writes quality_score into each result, and a shared dict would carry one
  file's score onto another;
- the digests are kept for _identical_to, which compares the very same
  files and used to read them all over again.

Run: python3 tests/test_copies_analyzed_once.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import duplicates_core as dc


def textured(h: int, w: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(h // 8 + 1, w // 8 + 1, 3), dtype=np.uint8)
    return cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)


class CountingAnalyze:
    """Stands in for duplicates_core.analyze (patched by name -- _analyze_one
    calls it bare), counting calls per path."""

    def __init__(self, fail: tuple[str, ...] = ()):
        self.real = dc.analyze
        self.calls: dict[str, int] = {}
        self.fail = fail

    def __call__(self, path_str: str) -> dict:
        self.calls[Path(path_str).name] = self.calls.get(Path(path_str).name, 0) + 1
        if Path(path_str).name in self.fail:
            raise OSError("simulated decode failure")
        return self.real(path_str)

    def __enter__(self):
        dc.analyze = self
        return self

    def __exit__(self, *exc):
        dc.analyze = self.real


def test_a_copy_is_analyzed_once(tmp: Path) -> None:
    a, b = tmp / "a.jpg", tmp / "b.jpg"
    cv2.imwrite(str(a), textured(120, 160, 1))
    cv2.imwrite(str(b), textured(120, 160, 2))
    copy = tmp / "a copy.jpg"
    shutil.copyfile(a, copy)
    paths = [a, copy, b]
    cache: dict = {}
    resolved: list[tuple[str, bool]] = []
    progress: list[tuple[int, int]] = []
    with CountingAnalyze() as counted:
        results = dc.analyze_paths(
            paths, cache,
            progress_callback=lambda label, done, total: progress.append((done, total)),
            done_callback=lambda p, r: resolved.append((p.name, r is not None)))

    assert counted.calls == {"a.jpg": 1, "b.jpg": 1}, f"analyze calls: {counted.calls}"
    assert sorted(resolved) == [("a copy.jpg", True), ("a.jpg", True), ("b.jpg", True)], resolved
    assert progress == [(1, 3), (2, 3), (3, 3)], progress
    assert set(results) == set(paths)
    ra, rc = results[a], results[copy]
    assert rc["path"] == str(copy) and ra["path"] == str(a)
    assert {k: v for k, v in rc.items() if k != "path"} == {k: v for k, v in ra.items() if k != "path"}
    assert rc is not ra
    rc["quality_score"] = 0.5
    assert "quality_score" not in ra, "the copy shares its result dict with the original"
    for p in paths:
        entry = cache[dc._cache_key(p)]
        assert entry["result"]["path"] == str(p) and "file_size" not in entry["result"], entry
    print("  ok  the copy is analyzed once and still gets its own result, cache entry, callback and progress tick")


class CountingDigest:
    """Stands in for duplicates_core._file_digest, recording each file read
    whole for its digest."""

    def __init__(self):
        self.real = dc._file_digest
        self.calls: dict[str, int] = {}

    def __call__(self, p: Path):
        self.calls[p.name] = self.calls.get(p.name, 0) + 1
        return self.real(p)

    def __enter__(self):
        dc._file_digest = self
        return self

    def __exit__(self, *exc):
        dc._file_digest = self.real


def test_same_size_different_bytes_are_both_analyzed(tmp: Path) -> None:
    """A size match is only the trigger to look. BMP is uncompressed, so two
    different photos of one size are the same length -- a whole folder of
    one camera's TIFFs is -- and a glance at a few windows of each must tell
    them apart without reading either whole."""
    a, b = tmp / "a.bmp", tmp / "b.bmp"
    cv2.imwrite(str(a), textured(256, 320, 3))
    cv2.imwrite(str(b), textured(256, 320, 4))
    assert a.stat().st_size == b.stat().st_size > 3 * dc._SAMPLE_BYTES
    with CountingAnalyze() as counted, CountingDigest() as digested:
        results = dc.analyze_paths([a, b], {})
    assert counted.calls == {"a.bmp": 1, "b.bmp": 1}, counted.calls
    assert digested.calls == {}, f"read whole to tell apart: {digested.calls}"
    assert results[a]["blockiness"] != results[b]["blockiness"] or results[a]["noise_sigma"] != results[b]["noise_sigma"]
    print("  ok  same size, different bytes: both analyzed, neither read whole")


def test_matching_windows_still_need_the_whole_file(tmp: Path) -> None:
    """The windows only rule copies *out*. Two files that agree on all of
    them but differ elsewhere are different files."""
    size = 1 << 20
    rng = np.random.default_rng(9)
    data = bytearray(rng.integers(0, 256, size, dtype=np.uint8).tobytes())
    (tmp / "a.bin").write_bytes(data)
    (tmp / "b.bin").write_bytes(data)
    data[300_000] ^= 0xFF  # outside the start, middle and end windows
    (tmp / "c.bin").write_bytes(data)
    (tmp / "d.bin").write_bytes(data[:-1])
    paths = [tmp / n for n in ("a.bin", "b.bin", "c.bin", "d.bin")]
    stats = {p: p.stat() for p in paths}
    digests: dict = {}
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(4) as pool, CountingDigest() as digested:
        copies = dc._copies_by_original(paths, stats, digests, pool)
    assert copies == {paths[0]: [paths[1]]}, copies
    assert digested.calls == {"a.bin": 1, "b.bin": 1, "c.bin": 1}, digested.calls
    print("  ok  files agreeing on every window are still compared whole; a lone size is never read")


def test_a_failed_original_fails_its_copies(tmp: Path) -> None:
    """The copies hold the same bytes, so they fail the same way -- and each
    must still resolve, as None, for build_groups' bookkeeping."""
    a = tmp / "a.jpg"
    cv2.imwrite(str(a), textured(120, 160, 5))
    copy = tmp / "a 2.jpg"
    shutil.copyfile(a, copy)
    resolved = []
    with CountingAnalyze(fail=("a.jpg",)) as counted:
        results = dc.analyze_paths([a, copy], {}, done_callback=lambda p, r: resolved.append((p.name, r)))
    assert counted.calls == {"a.jpg": 1}, counted.calls
    assert sorted(resolved) == [("a 2.jpg", None), ("a.jpg", None)], resolved
    assert results == {}
    print("  ok  an original that fails to analyze resolves each copy as None")


def make_library(root: Path) -> None:
    """Pairs of byte copies, a pair at two qualities, and a triple mixing
    both -- so a group can hold copies and a genuinely different file."""
    for g in range(3):
        cv2.imwrite(str(root / f"photo{g}.jpg"), textured(240, 320, 10 + g), [cv2.IMWRITE_JPEG_QUALITY, 92])
        shutil.copyfile(root / f"photo{g}.jpg", root / f"photo{g} copy.jpg")
    img = textured(240, 320, 20)
    cv2.imwrite(str(root / "pair_hi.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(root / "pair_lo.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 60])
    img = textured(240, 320, 30)
    cv2.imwrite(str(root / "trio.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    shutil.copyfile(root / "trio.jpg", root / "trio 2.jpg")
    cv2.imwrite(str(root / "trio_lo.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 60])


def test_streaming_scan_publishes_every_group(tmp: Path) -> None:
    """End to end through build_groups: every group arrives through the
    callback, in returned order, each copy is marked identical, and no file
    is read for its digest twice (analyze_paths and _identical_to share)."""
    make_library(tmp)
    published = []
    decoded: dict[str, int] = {}
    real_hash_gray = dc.load_hash_gray

    def counting_hash_gray(p):
        decoded[p.name] = decoded.get(p.name, 0) + 1
        return real_hash_gray(p)

    hashing = []
    dc.load_hash_gray = counting_hash_gray
    try:
        with CountingAnalyze() as counted, CountingDigest() as digests_read:
            groups = dc.build_groups(
                tmp, dc.DEFAULT_HASH_THRESHOLD, group_callback=published.append,
                progress_callback=lambda label, done, total: label == "Hashing" and hashing.append((done, total)))
    finally:
        dc.load_hash_gray = real_hash_gray
    digested = digests_read.calls

    assert [id(g) for g in published] == [id(g) for g in groups], "published out of order, or not at all"
    by_name = {tuple(sorted(p.name for p in g.paths)): g for g in groups}
    assert set(by_name) == {
        ("photo0 copy.jpg", "photo0.jpg"), ("photo1 copy.jpg", "photo1.jpg"),
        ("photo2 copy.jpg", "photo2.jpg"), ("pair_hi.jpg", "pair_lo.jpg"),
        ("trio 2.jpg", "trio.jpg", "trio_lo.jpg"),
    }, sorted(by_name)
    for names, g in by_name.items():
        copies = {g.paths[j].name for j, k in enumerate(g.identical_to) if k is not None}
        assert copies == {n for n in names if " copy" in n or " 2" in n}, (names, g.identical_to)
    # Which of two identical files gets analyzed is scan order's business
    # (" copy.jpg" sorts first); exactly one per content is the point.
    contents = [{"photo0.jpg", "photo0 copy.jpg"}, {"photo1.jpg", "photo1 copy.jpg"},
                {"photo2.jpg", "photo2 copy.jpg"}, {"trio.jpg", "trio 2.jpg"},
                {"pair_hi.jpg"}, {"pair_lo.jpg"}, {"trio_lo.jpg"}]
    assert all(v == 1 for v in counted.calls.values()), counted.calls
    assert [len(c & set(counted.calls)) for c in contents] == [1] * len(contents), sorted(counted.calls)
    # Hashing decodes once per content too -- a HEIC decode is ~0.9 s.
    assert all(v == 1 for v in decoded.values()), decoded
    assert [len(c & set(decoded)) for c in contents] == [1] * len(contents), sorted(decoded)
    assert hashing == [(i, 11) for i in range(1, 12)], hashing
    assert all(v == 1 for v in digested.values()), f"read twice for a digest: {digested}"
    print(f"  ok  {len(groups)} groups published in order; {len(decoded)} hash decodes and "
          f"{len(counted.calls)} analyses for 11 files; each digest read once")


def test_no_copies_no_digests(tmp: Path) -> None:
    """A library without copies pays nothing: no size collides, so no file
    is read beyond what hashing and analysis already read."""
    for g in range(3):
        img = textured(240, 320, 40 + g)
        cv2.imwrite(str(tmp / f"p{g}_hi.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(tmp / f"p{g}_lo.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 60])
    calls = []
    real_digest, real_sample = dc._file_digest, dc._sample_digest
    dc._file_digest = lambda p: calls.append(("whole", p.name)) or real_digest(p)
    dc._sample_digest = lambda p: calls.append(("windows", p.name)) or real_sample(p)
    try:
        groups = dc.build_groups(tmp, dc.DEFAULT_HASH_THRESHOLD, progress_callback=lambda *a: None)
    finally:
        dc._file_digest, dc._sample_digest = real_digest, real_sample
    assert len(groups) == 3, len(groups)
    assert calls == [], f"read for copies without a size collision: {calls}"
    print("  ok  no size collisions, nothing read to look for copies")


def main() -> None:
    tests = [
        test_a_copy_is_analyzed_once,
        test_same_size_different_bytes_are_both_analyzed,
        test_matching_windows_still_need_the_whole_file,
        test_a_failed_original_fails_its_copies,
        test_streaming_scan_publishes_every_group,
        test_no_copies_no_digests,
    ]
    for test in tests:
        print(f"{test.__name__}:")
        with tempfile.TemporaryDirectory() as tmp:
            test(Path(tmp))
    print("all copies-analyzed-once tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)
