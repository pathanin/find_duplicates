"""
duplicates_core.py

Shared, UI-agnostic core of the duplicate-image tool: directory scanning,
perceptual hashing + grouping, quality scoring, thumbnailing, and the one
destructive path (moving non-kept files out of the way). No web framework
dependency here -- this module must stay importable standalone (e.g. for
tests) without pulling in FastAPI/uvicorn.

find_duplicates.py and duplicates_web.py both import from this module
rather than duplicating any of it.
"""

import csv
import hashlib
import math
import os
import shutil
import sys
import threading
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Callable, NamedTuple

import cv2
import numpy as np
from PIL import Image as PILImage

from compare_image_quality import analyze, decode_plan

# HEIC/HEIF (the default format Apple Photos/iPhone exports) has no reliable
# OS-level decoder behind cv2.imread, so PIL needs this optional plugin
# registered before PIL.Image.open can read those files. A missing package
# must never crash a scan -- HEIC files just fail to decode and get silently
# skipped like any other corrupt/unreadable file already does today.
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".heic", ".heif"}
DEFAULT_HASH_THRESHOLD = 10  # max Hamming distance out of 64 bits to call two images duplicates
# Max Hamming distance out of 256 bits for phash_pair's confirmation hash: a
# pair the 64-bit hash proposes is grouped only if the wider hash agrees too.
# Applied only at or below DEFAULT_HASH_THRESHOLD (see group_duplicates).
#
# Deliberately loose. The true-duplicate and same-scene distance ranges
# overlap, so no cut separates them: this sits *above* the worst duplicate
# observed rather than between the two, because a false positive costs one
# keypress in the review UI and a false negative is never surfaced at all.
# The binding case is an aspect recrop (one artwork exported for two phone
# screens), which moves far more bits than a rescale. Lowering this drops
# real duplicates well before it stops the near-identical frames that still
# get through -- 56 looked clean against re-exports alone and lost six real
# duplicates. tests/test_confirm_hash.py locks both directions.
CONFIRM_HASH_THRESHOLD = 88
PREVIEW_MAX_SIDE = 800
CLOSE_CALL_MARGIN = 0.08  # quality_score gap below which we flag "close call"
# phash resizes to 32x32; a reduced-scale decode smaller than this on either side
# would upsample instead of downsample there, drifting the hash. 64 gives margin,
# and images this small are cheap to fully decode anyway.
MIN_REDUCED_DECODE_SIDE = 64
# load_hash_gray/phash's cv2 calls (imread/resize/dct) release the GIL, so a
# thread pool gets real parallelism without ProcessPoolExecutor's process-spawn
# cost (benchmarked at ~0.6-3s on Apple M4/10 cores -- a serial loop used to
# beat a process pool below a few hundred files purely because of that spawn
# tax). Benchmarked on the same machine hashing synthetic 1600x1200 JPEGs: a
# thread pool beat both serial execution and a process pool at every batch
# size tried (30, 150, 3000 files) -- ~5-9x faster than serial at 30-150
# files (no spawn tax to pay), and matching or slightly beating a process
# pool's throughput at 3000. So hashing always parallelizes now; there's no
# serial fallback threshold to tune. cv2.setNumThreads(1) is set for the
# duration of the pool (see group_duplicates) so cv2's own internal thread
# pool doesn't oversubscription-fight these worker threads -- ~20% faster at
# 3000 files than leaving cv2's default thread count in place.
THREAD_POOL_WORKERS = os.cpu_count() or 1
# analyze()'s cv2 calls (resize/Laplacian/filter2D) and numpy's FFT release
# the GIL the same way hashing's do, so analyze_paths routes through a
# thread pool too -- same THREAD_POOL_WORKERS, no threshold. Benchmarked on
# the same machine: at a typical single-group batch (6 files), threads hit
# ~19.6 img/s against a process pool's ~8.7 img/s (indistinguishable from
# serial -- spawn overhead ate the whole benefit); at 300 files, ~36.3 img/s
# against ~23.1 img/s. Threads also sidestep the fork-after-cv2-threads
# macOS crash that motivated spawn-only ProcessPoolExecutor here before,
# since there's no forking or spawning at all.

# Weight > 0 means higher raw value is better; weight < 0 means lower raw value is better.
# effective_resolution_px_equiv is weighted heaviest since it's the metric most resistant
# to fake upscaling (true detail amount rather than just stored pixel count).
METRIC_WEIGHTS = {
    "effective_resolution_px_equiv": 0.35,
    "sharpness_normalized": 0.20,
    "effective_resolution_fraction": 0.15,
    "noise_sigma": -0.10,
    "blockiness": -0.10,
    "brisque": -0.10,
    "niqe": -0.10,
}

# One-line plain-English gloss per metric, shown in the in-app help (`?`).
# Keyed off METRIC_WEIGHTS so the help text can't drift out of sync with
# what's actually scored -- add a metric to the weights and its description
# is required here too, or the help screen would silently omit it.
METRIC_DESCRIPTIONS = {
    "effective_resolution_px_equiv": "true detail amount; resistant to fake upscaling",
    "sharpness_normalized": "edge/detail sharpness, compared at a common scale",
    "effective_resolution_fraction": "fraction of native resolution that's real detail, not just interpolated pixels",
    "noise_sigma": "sensor/compression noise",
    "blockiness": "JPEG block-edge artifacts",
    "brisque": "no-reference perceptual quality score; optional, and counts only where it produces a value",
    "niqe": "no-reference perceptual quality score; optional, and counts only where it produces a value",
}

# Weighted metrics that normally produce no value at all: their packages are
# deliberately not installed (see CLAUDE.md), so the help sheet lists them as
# not counting unless a scan actually produced one.
OPTIONAL_METRICS = {"brisque", "niqe"}

class MetricRow(NamedTuple):
    """One row of the metrics table rendered by the web UI's /api/group/{i}
    response, structured so the frontend never has to re-derive a row's
    meaning by parsing `label` -- label text is for human display only (and
    free to be reworded), never a machine-readable contract. That used to be
    exactly what the web frontend did (matching a "(higher better)"/"(lower
    better)" suffix and a "Quality score" prefix out of the label string),
    which a purely cosmetic label rewording could silently break with no
    error and no test catching it.

    key: the METRIC_WEIGHTS key this row's raw value comes from, or None for
        rows that don't derive from a single weighted metric (Dimensions/File
        size, and quality_score itself -- kind="score" already says
        everything about that row).
    kind: "reference" (Dimensions/File size -- shown but never scored),
        "metric" (an individual weighted input -- comparable/highlightable
        against the other files in the group), or "score" (the single
        combined quality_score row).
    """
    label: str
    fn: Callable[[dict], str]
    key: str | None
    kind: str

    @property
    def direction(self) -> int:
        """+1 if a higher raw value is better, -1 if lower is better, 0 if
        this row isn't scored at all (kind == "reference"). Derived from
        METRIC_WEIGHTS' sign rather than hand-duplicated, so an edit to a
        weight's sign can't silently disagree with what this row claims."""
        if self.kind == "score":
            return 1  # a normalized composite of already-direction-corrected metrics: always higher-is-better
        if self.key is None:
            return 0
        return 1 if METRIC_WEIGHTS[self.key] > 0 else -1


# Every scored row is labeled with what a bigger/smaller number means, since
# a raw number is meaningless without knowing which direction is "better".
# Dimensions/file size carry no such label since they aren't part of the
# score at all -- that's explained once, in the '?' help screen, rather than
# on every row. Defined here (not in the web module) so it stays reusable
# by anything else that wants the same metrics-JSON shape without keeping a
# second label list in sync by hand.
METRIC_ROWS = [
    MetricRow("Dimensions", lambda r: f"{r['dimensions'][0]}x{r['dimensions'][1]}", None, "reference"),
    MetricRow("File size", lambda r: humansize(r["file_size"]), None, "reference"),
    MetricRow("Sharpness (higher better)", lambda r: f"{r['sharpness_normalized']:.1f}", "sharpness_normalized", "metric"),
    MetricRow("Eff. res. fraction (higher better)", lambda r: f"{r['effective_resolution_fraction']:.3f}", "effective_resolution_fraction", "metric"),
    MetricRow("Eff. res. px equiv (higher better)", lambda r: f"{r['effective_resolution_px_equiv']:.0f}", "effective_resolution_px_equiv", "metric"),
    MetricRow("Noise sigma (lower better)", lambda r: f"{r['noise_sigma']:.3f}", "noise_sigma", "metric"),
    MetricRow("Blockiness (lower better)", lambda r: f"{r['blockiness']:.3f}", "blockiness", "metric"),
    MetricRow("BRISQUE (lower better)", lambda r: f"{r['brisque']:.2f}" if r.get("brisque") is not None else "n/a", "brisque", "metric"),
    MetricRow("NIQE (lower better)", lambda r: f"{r['niqe']:.2f}" if r.get("niqe") is not None else "n/a", "niqe", "metric"),
    MetricRow("Quality score (higher better)", lambda r: f"{r['quality_score']:.3f}", None, "score"),
]


# ---------------------------------------------------------------------------
# Scanning + perceptual hashing + grouping
# ---------------------------------------------------------------------------

# Folders a recursive scan never walks into: each belongs to another program
# that keeps a smaller copy of every photo beside it -- a Photos/Aperture
# library's derivatives, Synology's @eaDir thumbnails, a recycle bin -- so
# every original would come back paired with its own preview as a
# "duplicate", and moving one out of a Photos library damages the library.
# Dot-folders (.Trashes, .thumbnails, QNAP's .@__thumb) are skipped for the
# same reason. Only folders met *during* the walk: a scan rooted at one of
# them on purpose still runs.
SKIP_DIR_NAMES = {
    "@eaDir", "#recycle", "#snapshot", "@Recycle", "@Recently-Snapshot",
    "$RECYCLE.BIN", "System Volume Information", "lost+found",
}
SKIP_DIR_SUFFIXES = (".photoslibrary", ".photolibrary", ".migratedphotolibrary", ".aplibrary", ".lrdata")


def _skipped_dir(name: str) -> bool:
    return name.startswith(".") or name in SKIP_DIR_NAMES or name.lower().endswith(SKIP_DIR_SUFFIXES)


def _is_image_name(name: str) -> bool:
    # "._" is an AppleDouble file (macOS metadata written beside each file
    # on a NAS): it keeps the image's extension but holds no image.
    return not name.startswith("._") and os.path.splitext(name)[1].lower() in IMAGE_EXTS


def find_images(directory: Path, recursive: bool = False, exclude_dir: Path | None = None) -> list[Path]:
    """Top-level-only scan by default. With *recursive*, walks subdirectories
    too, minus the folders _skipped_dir names -- *exclude_dir* (typically the
    move destination) is then required to keep a re-scan from picking up
    files already moved out by a prior run; it's meaningless (and ignored) in
    non-recursive mode since the default destination (<directory>/_duplicates)
    already sits below the top level iterdir() looks at."""
    if not recursive:
        return sorted(p for p in directory.iterdir() if _is_image_name(p.name) and p.is_file())
    exclude_resolved = exclude_dir.resolve() if exclude_dir is not None else None
    found = []
    # os.walk rather than rglob so a skipped folder is pruned, not walked and
    # filtered: a Photos library alone can hold a hundred thousand files.
    for root, dirnames, filenames in os.walk(directory):
        dirnames[:] = [d for d in dirnames if not _skipped_dir(d)]
        for name in filenames:
            if not _is_image_name(name):
                continue
            p = Path(root) / name
            if not p.is_file():
                continue
            if exclude_resolved is not None and p.resolve().is_relative_to(exclude_resolved):
                continue
            found.append(p)
    return sorted(found)


def _load_gray_via_pil(p: Path) -> np.ndarray | None:
    """Fallback decode for formats cv2 can't read at all (currently just
    HEIC/HEIF), via PIL + the registered pillow-heif opener. Always a full
    decode -- no reduced-scale trick like the cv2 path above, since
    correctness matters more than that specific optimization for this
    format. Returns None (rather than raising) on any decode failure so a
    HEIC file with no HEIF plugin installed, or a genuinely corrupt file,
    is silently skipped exactly like any other unreadable file today."""
    try:
        with PILImage.open(p) as pil_img:
            return np.array(pil_img.convert("L"))
    except Exception:
        return None


def load_hash_gray(p: Path) -> np.ndarray | None:
    """Grayscale decode for perceptual hashing. Uses a 1/8-scale DCT decode
    for speed (skips full-resolution JPEG decode just to shrink it to 32x32
    afterwards); falls back to a full decode when the image is small enough
    that the reduced decode would land below what the hash needs. Formats
    cv2 can't decode at all (e.g. HEIC/HEIF) fall through both cv2 attempts
    as None and get a full PIL-based decode instead."""
    img = cv2.imread(str(p), cv2.IMREAD_REDUCED_GRAYSCALE_8)
    if img is None or min(img.shape) < MIN_REDUCED_DECODE_SIDE:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = _load_gray_via_pil(p)
    return img


def _low_freq_bits(block: np.ndarray) -> int:
    """Pack a square low-frequency DCT block into one big int, one bit per
    coefficient, thresholded against the block mean with DC excluded (DC is
    overall brightness and would swamp the mean). Row-major, first
    coefficient in the most significant bit."""
    avg = (block.sum() - block[0, 0]) / (block.size - 1)
    return int.from_bytes(np.packbits(block > avg).tobytes(), "big")


def phash_pair(gray: np.ndarray) -> tuple[int, int]:
    """Both perceptual hashes off a single DCT: the classic 64-bit one that
    proposes candidate pairs, and a 256-bit one that confirms them.

    The 64-bit hash keeps only an 8x8 low-frequency block, so it describes
    little more than a thumbnail's gross layout -- robust to resizing and
    recompression (the "same photo, different export" duplicate we're after),
    but by the same token nearly blind to the difference between two separate
    frames of one scene. The 16x16 block reaches into the mid frequencies
    where those frames actually diverge; see CONFIRM_HASH_THRESHOLD.

    Computing both here rather than in two passes is deliberate: the resize
    and DCT are shared, so the wider hash costs essentially nothing on top of
    a decode that dominates either way.

    The hash cache keys on path + mtime + size, not on this function's code,
    so a rescan within one process would serve old hashes after an edit here.
    That cache lives in memory only and dies with the process -- restart the
    tool after changing this or load_hash_gray and you're testing the new
    code, no cache file to hunt down and delete."""
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(resized)
    return _low_freq_bits(dct[:8, :8]), _low_freq_bits(dct[:16, :16])


def phash(gray: np.ndarray) -> int:
    """The 64-bit hash alone, for callers that only need the grouping key."""
    return phash_pair(gray)[0]


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


# Byte -> set-bit count, fallback for numpy < 2.0 (no np.bitwise_count).
_POPCOUNT8 = np.unpackbits(
    np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1).astype(np.uint8)

# Byte budget for the sweep's XOR temp; shrink in tests to force >1 block.
_SWEEP_XOR_BYTE_BUDGET = 32_000_000


def _popcount(arr: np.ndarray) -> np.ndarray:
    """Per-element set-bit count of a uint64 array."""
    try:
        return np.bitwise_count(arr)
    except AttributeError:
        return _POPCOUNT8[arr.view(np.uint8).reshape(*arr.shape, -1)].sum(-1)


def _pack_hashes(values, nlanes: int) -> tuple[np.ndarray, np.ndarray]:
    """Pack a list of int hashes (or None) into an (n, nlanes) uint64 array
    (big-endian bytes viewed as uint64 -- lane order and endianness don't
    matter to XOR or popcount). None rows are zeroed and flagged in `valid`."""
    n = len(values)
    arr = np.zeros((n, nlanes * 8), dtype=np.uint8)
    valid = np.zeros(n, dtype=bool)
    for i, v in enumerate(values):
        if v is not None:
            arr[i] = np.frombuffer(v.to_bytes(nlanes * 8, "big"), dtype=np.uint8)
            valid[i] = True
    return arr.view(np.uint64), valid


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


@lru_cache(maxsize=None)
def _cache_key(p: Path) -> str:
    """Resolved path string, the key both scan caches use. Memoized because
    resolve() runs 2-4 times per file per scan (a lookup and a store in each
    of the hash and analyze caches) and is a per-path-component readlink
    round-trip on the NAS/SMB shares this tool is built for -- ~10 us on
    local APFS, unmeasured but plausibly far worse there. Safe to memoize
    because it only ever produces cache keys: a symlink repointed mid-session
    would at worst cost that file a re-hash.
    ponytail: unbounded, one string per path seen this process -- bound it if
    a single process ever scans enough distinct paths for that to matter."""
    return str(p.resolve())


def cached_hash(cache: dict, p: Path, st: os.stat_result) -> tuple[int, int] | None:
    """Returns the (grouping, confirmation) hash pair. None both when there's
    no entry and when the cached entry itself is None (the file failed to
    decode/hash last time) -- a permanently-corrupt file is simply re-attempted
    every run, no worse than today's uncached behavior for that one file."""
    entry = cache.get(_cache_key(p))
    if entry is None or entry.get("mtime") != st.st_mtime_ns or entry.get("size") != st.st_size:
        return None
    return entry["hash"]


def store_hash(cache: dict, p: Path, st: os.stat_result,
               hash_value: tuple[int, int] | None) -> None:
    cache[_cache_key(p)] = {"mtime": st.st_mtime_ns, "size": st.st_size, "hash": hash_value}


def _stat_paths(paths: list[Path]) -> tuple[list[Path], dict[Path, os.stat_result]]:
    """stat() every path, dropping the ones that vanished or turned
    unreadable between listing and now (routine on a NAS or a syncing
    folder). A single missing file costs that file, not the whole scan."""
    kept: list[Path] = []
    stats: dict[Path, os.stat_result] = {}
    for p in paths:
        try:
            stats[p] = p.stat()
        except OSError:
            continue
        kept.append(p)
    return kept, stats


def _hash_one(p: Path) -> tuple[int, int] | None:
    img = load_hash_gray(p)
    return phash_pair(img) if img is not None else None


def _print_progress(label: str, done: int, total: int, tty: bool) -> None:
    """Incremental progress for a long-running scan phase. On a TTY,
    overwrites the same terminal line via carriage return so it doesn't spam
    scrollback; when stdout isn't a TTY (redirected to a file, running under
    test), falls back to occasional plain lines instead of \r-laden output."""
    if tty:
        print(f"\r{label}: {done}/{total}", end="", flush=True)
    elif done == total or done % 100 == 0:
        print(f"{label}: {done}/{total}")


def group_duplicates(
    paths: list[Path], threshold: int, cache: dict, progress_callback=None,
    digests: dict[Path, bytes | None] | None = None,
) -> list[list[Path]]:
    """Groups `paths` by perceptual-hash Hamming distance, reusing `cache`
    for files whose (mtime, size) haven't changed (see cached_hash/
    store_hash above) so a re-scan of an already-hashed directory doesn't
    re-decode every old file. The uncached subset always hashes through a
    thread pool -- see THREAD_POOL_WORKERS for why threads (not a process
    pool) win here.

    *progress_callback*, if given, is called as progress_callback(label,
    done, total) as each uncached item completes, instead of the default
    TTY-aware print via _print_progress -- lets a caller (e.g. the web
    front end's SSE progress stream) route progress to something other than
    stdout without touching the CLI's default behavior.

    A byte-identical copy is decoded once and shares its original's hash --
    the decode is most of a hash, and ~0.9 s on a 24 MP HEIC. *digests*, if
    given, keeps what was read to find copies (see analyze_paths)."""
    paths, stats = _stat_paths(paths)
    hashes: dict[Path, tuple[int, int] | None] = {}
    to_compute = []
    for p in paths:
        cached = cached_hash(cache, p, stats[p])
        if cached is not None:
            hashes[p] = cached
        else:
            to_compute.append(p)

    if to_compute:
        total = len(to_compute)
        tty = sys.stdout.isatty()
        original_cv2_threads = cv2.getNumThreads()
        cv2.setNumThreads(1)
        try:
            with ThreadPoolExecutor(max_workers=THREAD_POOL_WORKERS) as executor:
                copies = _copies_by_original(to_compute, stats, {} if digests is None else digests, executor)
                is_copy = {q for qs in copies.values() for q in qs}
                originals = [p for p in to_compute if p not in is_copy]
                done = 0
                for p, h in zip(originals, executor.map(_hash_one, originals)):
                    for q in (p, *copies.get(p, ())):
                        store_hash(cache, q, stats[q], h)
                        hashes[q] = h
                        done += 1
                        if progress_callback is not None:
                            progress_callback("Hashing", done, total)
                        else:
                            _print_progress("Hashing", done, total, tty)
            if progress_callback is None and tty:
                print()
        finally:
            cv2.setNumThreads(original_cv2_threads)

    hash_list = [hashes[p] for p in paths]

    # Raising *threshold* is the documented way to ask for looser matching
    # ("Raise the threshold" is what the UI suggests when a scan finds too
    # little), and the two hashes are correlated enough that a fixed gate
    # would reject much of what the wider threshold just admitted -- turning
    # the knob into a partial no-op. Someone who widened it has already
    # accepted the false positives that come with it.
    confirm = threshold <= DEFAULT_HASH_THRESHOLD

    n = len(paths)
    uf = UnionFind(n)
    h64, valid = _pack_hashes([h[0] if h is not None else None for h in hash_list], 1)
    h64 = h64.reshape(-1)
    h256, _ = _pack_hashes([h[1] if h is not None else None for h in hash_list], 4)

    # ponytail: O(n^2) vectorized-popcount sweep -- bit-identical to the old
    # double loop, ~7-30x faster. Add pigeonhole-LSH banding if n routinely
    # exceeds ~100k. Blocked over rows to bound the XOR temp to the byte budget.
    block = max(1, _SWEEP_XOR_BYTE_BUDGET // (8 * max(n, 1)))
    for start in range(0, n, block):
        d64 = _popcount(h64[start:start + block, None] ^ h64[None, :])
        ii, jj = np.nonzero(d64 <= threshold)
        ii = ii + start
        sel = (jj > ii) & valid[ii] & valid[jj]
        ii, jj = ii[sel], jj[sel]
        if confirm and ii.size:
            d256 = _popcount(h256[ii] ^ h256[jj]).sum(axis=1, dtype=np.int16)
            keep = d256 <= CONFIRM_HASH_THRESHOLD
            ii, jj = ii[keep], jj[keep]
        for a, b in zip(ii.tolist(), jj.tolist()):
            uf.union(a, b)

    clusters: dict[int, list[Path]] = {}
    for i, p in enumerate(paths):
        if hash_list[i] is None:
            continue
        clusters.setdefault(uf.find(i), []).append(p)

    return [members for members in clusters.values() if len(members) > 1]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_group(results: list[dict]) -> None:
    """Attach a 0-1 'quality_score' to each result dict, min-max normalized
    within this group only (raw metric ranges aren't comparable across
    unrelated images, but are meaningful when comparing duplicates of the
    same photo)."""
    if not results:
        return
    keys = [
        k for k in METRIC_WEIGHTS
        if all(r.get(k) is not None and math.isfinite(r[k]) for r in results)
    ]
    total_weight = sum(abs(METRIC_WEIGHTS[k]) for k in keys) or 1.0

    ranges = {}
    for k in keys:
        vals = [r[k] for r in results]
        lo, hi = min(vals), max(vals)
        ranges[k] = (lo, hi if hi > lo else lo + 1e-9)

    for r in results:
        score = 0.0
        for k in keys:
            lo, hi = ranges[k]
            norm = (r[k] - lo) / (hi - lo)
            weight = METRIC_WEIGHTS[k]
            score += norm * weight if weight > 0 else (1 - norm) * abs(weight)
        r["quality_score"] = score / total_weight


def humansize(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if unit == "B":
            if n < 1024:
                return f"{n:.0f}{unit}"
        elif n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}GB"


THUMBNAIL_FAILURE_COLOR = (60, 60, 60)  # neutral gray placeholder, visually distinct from real photos


def make_thumbnail(path: Path) -> PILImage.Image:
    try:
        img = PILImage.open(path)
        img = img.convert("RGB")
        img.thumbnail((PREVIEW_MAX_SIDE, PREVIEW_MAX_SIDE))
        return img
    except Exception:
        return PILImage.new("RGB", (PREVIEW_MAX_SIDE, PREVIEW_MAX_SIDE), THUMBNAIL_FAILURE_COLOR)


@dataclass
class Group:
    paths: list[Path]
    results: list[dict]
    thumbnails: list[PILImage.Image] | None  # lazily generated on first view, see refresh_detail
    suggested_idx: int
    current_pick: int
    is_close_call: bool
    status: str = "pending"  # pending | confirmed | skipped
    # Indices kept *alongside* current_pick -- for a group that caught a
    # similar-but-different photo (a false positive) next to its real
    # duplicates. Independent of the pick: the kept set is always
    # {current_pick} | also_keep, so flipping the stage never drops a pin.
    also_keep: list[int] = field(default_factory=list)
    # For each file, the first earlier file in the group with exactly the
    # same bytes, else None -- set by _build_group, so empty on a Group built
    # by hand. all_identical: every file is the same bytes as the first, so
    # keeping any one of them loses nothing.
    identical_to: list[int | None] = field(default_factory=list)
    all_identical: bool = False

    def kept_indices(self, keep_idx: int | None = None) -> set[int]:
        return {self.current_pick if keep_idx is None else keep_idx, *self.also_keep}


# Written beside the moved files by every real (non-dry-run) move and
# restore. The manifest lives in memory and dies with the process, and by
# then a collision rename (name_dup1.jpg) has already lost the original
# name -- so after an exit this file is what "move it back by hand" reads.
MOVE_LOG_NAME = "moves.log"
_MOVE_LOG_HEADER = "# find-duplicates move log: time, action, from, to -- tab-separated, one row per file\n"


def _append_move_log(log_path: Path, action: str, src: str, dst: str) -> None:
    """A record, not part of the move: failing to write it warns on stderr
    and never fails the move or restore it describes."""
    try:
        new = not log_path.exists()
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            if new:
                f.write(_MOVE_LOG_HEADER)
            csv.writer(f, delimiter="\t", lineterminator="\n").writerow(
                [datetime.now().isoformat(timespec="seconds"), action, src, dst])
    except OSError as exc:
        print(f"warning: couldn't record a move in {log_path}: {exc}", file=sys.stderr)


def _compute_dest(
    path: Path,
    dest_dir: Path,
    dry_run: bool,
    recursive: bool = False,
    scan_root: Path | None = None,
) -> Path:
    """Where *path* should land under *dest_dir* if moved as a non-kept
    duplicate. Preserves *path*'s position relative to *scan_root* when
    *recursive* (so two same-named files from different subdirectories don't
    collide into one flat name); just the filename otherwise. A collision
    suffix keeps the same relative parent directory -- dropping it would
    silently flatten the file into dest_dir's root, defeating the point of
    preserving structure in the first place."""
    rel = path.relative_to(scan_root) if (recursive and scan_root is not None) else Path(path.name)
    dest = dest_dir / rel
    if not dry_run:
        dest.parent.mkdir(parents=True, exist_ok=True)
    n = 1
    while dest.exists():
        dest = dest_dir / rel.parent / f"{rel.stem}_dup{n}{rel.suffix}"
        n += 1
    return dest


def apply_group(
    group: Group,
    group_index: int,
    keep_idx: int,
    dest_dir: Path,
    dry_run: bool = False,
    manifest: list[dict] | None = None,
    recursive: bool = False,
    scan_root: Path | None = None,
) -> dict:
    """Move every non-kept file in *group* to *dest_dir* -- kept being
    *keep_idx* plus any pinned group.also_keep indices. Shared by the
    interactive confirm path (apply_pick) and --auto mode. Files whose real
    target is a kept symlink's target are left alone (moving the target
    would leave the kept path dangling). Records whatever was
    actually moved into a manifest entry from a `finally` -- even a failure
    partway through the loop (disk full, permission error) leaves an
    accurate, reversible record instead of silently losing track of files
    already relocated to disk. Does not set group.status; callers decide
    (kept "pending" on a raised exception is load-bearing for the retry
    path, see test_web_api.py's test_confirm_partial_failure_leaves_group_pending).
    Appends the entry to *manifest* if given, and always returns it."""
    group.current_pick = keep_idx
    kept_path = group.paths[keep_idx]
    keep = group.kept_indices(keep_idx)
    also_kept = [str(group.paths[j]) for j in sorted(keep - {keep_idx})]
    log_path = dest_dir / MOVE_LOG_NAME
    moved = []
    try:
        # is_symlink()/resolve() can themselves raise (e.g. EACCES on a
        # parent directory) -- kept inside the try so a failure here still
        # goes through the finally below instead of leaving *manifest*
        # without an entry for this group at all.
        kept_reals = {group.paths[j].resolve() for j in keep if group.paths[j].is_symlink()}
        for idx, path in enumerate(group.paths):
            if idx in keep:
                continue
            if kept_reals and path.resolve() in kept_reals:
                continue
            dest = _compute_dest(path, dest_dir, dry_run, recursive=recursive, scan_root=scan_root)
            if not dry_run:
                shutil.move(str(path), str(dest))
                _append_move_log(log_path, "moved", str(path), str(dest))
            moved.append({"from": str(path), "to": str(dest)})
    finally:
        entry = {"group": group_index, "kept": str(kept_path), "also_kept": also_kept,
                 "moved": moved, "dry_run": dry_run, "log": str(log_path)}
        if manifest is not None:
            manifest.append(entry)
    return entry


def auto_apply_groups(
    groups: list[Group],
    dest_dir: Path,
    dry_run: bool = False,
    recursive: bool = False,
    scan_root: Path | None = None,
) -> dict:
    """Apply every pending group's suggested (top-scored) pick, no review UI
    -- used by --auto. Doesn't second-guess close calls: the suggested pick
    is applied exactly as it would default to in interactive review.

    A group whose apply_group() raises partway through is recorded as a
    failure (its status stays "pending", the same state a partial move
    failure leaves it in during interactive review) and the run continues
    with the remaining groups -- one bad group (disk full, permission
    error) must not kill an unattended run and leave earlier groups'
    successfully-moved files unreported.

    bytes_reclaimed sums each moved file's size from group.results[idx]
    ["file_size"] (already populated by analyze_paths) *before* the move --
    the source path is gone from disk by the time apply_group returns, so
    re-stat()'ing it afterward would silently always read 0.

    bytes_moved is the same sum taken under dry run too -- what the run
    would free -- since bytes_reclaimed stays 0 when nothing moved. entries
    are the manifest entries apply_group recorded, one per group attempted,
    so the CLI can name every file kept and moved.

    Returns {"confirmed": int, "failed": int, "files_moved": int,
    "bytes_reclaimed": int, "bytes_moved": int, "entries": [manifest entry, ...],
    "failures": [{"group", "error", "files_moved", "bytes_moved"}, ...]}."""
    confirmed = 0
    failed = 0
    files_moved = 0
    bytes_reclaimed = 0
    bytes_moved = 0
    failures = []
    manifest: list[dict] = []

    for i, group in enumerate(groups):
        if group.status != "pending":
            continue
        keep_idx = group.suggested_idx
        size_by_path = {str(p): r.get("file_size", 0) for p, r in zip(group.paths, group.results)}

        error = None
        pre_len = len(manifest)
        try:
            apply_group(group, i, keep_idx, dest_dir, dry_run, manifest, recursive=recursive, scan_root=scan_root)
        except Exception as exc:  # noqa: BLE001 -- one group's failure must not abort the rest
            error = exc

        # apply_group's finally appends an entry in the normal case, but
        # defend against a future change reintroducing a raise that happens
        # before that finally is reached (it must not be possible today --
        # see the comment in apply_group -- but indexing manifest[-1]
        # unconditionally would otherwise misattribute a previous group's
        # entry, or IndexError on group 0).
        moved = manifest[-1]["moved"] if len(manifest) > pre_len else []
        moved_bytes = sum(size_by_path.get(m["from"], 0) for m in moved)
        n_moved = len(moved)
        files_moved += n_moved
        bytes_moved += moved_bytes

        if error is not None:
            failed += 1
            failures.append({"group": i, "error": str(error), "files_moved": n_moved, "bytes_moved": moved_bytes})
            continue

        group.status = "confirmed"
        confirmed += 1
        if not dry_run:
            bytes_reclaimed += moved_bytes

    return {
        "confirmed": confirmed,
        "failed": failed,
        "files_moved": files_moved,
        "bytes_reclaimed": bytes_reclaimed,
        "bytes_moved": bytes_moved,
        "entries": manifest,
        "failures": failures,
    }


# ---------------------------------------------------------------------------
# Interactive confirm/skip/re-pick primitives
# ---------------------------------------------------------------------------
# Take the manifest/group explicitly rather than reading state off a caller
# object, so the same manifest invariants (stays-pending-on-partial-failure,
# unapply-on-skip, clear-stale-entries-before-reapply, re-pick-after-confirm)
# live in exactly one place -- the one genuinely destructive, stateful path
# in the app.

def unapply(manifest: list[dict], group_index: int) -> None:
    """Reverse file moves for group *group_index* using *manifest* (the
    record apply_pick/apply_group left behind). Does NOT change the group's
    status -- the caller decides. Purely data-driven: doesn't need the Group
    object itself, only the manifest entry apply_group recorded for it."""
    entry = next((m for m in manifest if m["group"] == group_index), None)
    if not entry:
        return
    if entry["dry_run"]:
        # Dry-run moves never touch the filesystem (see apply_group), so
        # there's nothing to check for on disk -- the manifest entry itself
        # is the only state a dry-run "move" left behind, and reversing it
        # is just dropping that entry.
        manifest.remove(entry)
        return
    restored = []
    try:
        for moved in entry["moved"]:
            src = Path(moved["from"])
            dst = Path(moved["to"])
            if dst.exists() and not src.exists():
                # shutil.move, not Path.rename: dest_dir may point at a
                # different filesystem than the scanned directory, and a
                # plain rename raises OSError (EXDEV) cross-device where
                # shutil.move falls back to copy+remove -- the same reason
                # apply_group uses shutil.move rather than rename for the
                # forward move.
                shutil.move(str(dst), str(src))
                restored.append(moved)
                if entry.get("log"):
                    _append_move_log(Path(entry["log"]), "restored", str(dst), str(src))
    finally:
        # Keep tracking whatever wasn't restored (never just the fact that
        # *something* was restored) even if a move raised partway through,
        # so *manifest* always reflects real filesystem state for the rest
        # of this session -- the same invariant apply_group preserves on
        # the forward move (see test_manifest_crash_safety.py). Dropping
        # the whole entry here regardless of partial failure would
        # silently lose track of files still sitting in dest_dir/.
        remaining = [m for m in entry["moved"] if m not in restored]
        if remaining:
            entry["moved"] = remaining
        else:
            manifest.remove(entry)


def pick_needs_reapply(manifest: list[dict], group_index: int, group: Group) -> bool:
    """True for a confirmed group whose kept set (current_pick plus pins)
    has since diverged from what's actually on disk (manifest's record of
    what got applied) -- i.e. re-applying would really re-move files, rather
    than being a no-op. Compares the whole set: making an already pinned
    file the primary keeps the same files and moves nothing. Re-picking or
    re-pinning on an already confirmed group only stages the change;
    nothing moves until the caller explicitly re-applies."""
    entry = next((m for m in manifest if m["group"] == group_index), None)
    if entry is None:
        return True
    on_disk = {entry["kept"], *entry.get("also_kept", [])}
    return on_disk != {str(group.paths[j]) for j in group.kept_indices()}


def apply_pick(
    group: Group,
    group_index: int,
    keep_idx: int,
    dest_dir: Path,
    dry_run: bool,
    manifest: list[dict],
    recursive: bool = False,
    scan_root: Path | None = None,
) -> None:
    """Confirm *keep_idx* for *group*: clear any stale manifest entries left
    over from a previous partial failure for this group (unapply only
    processes the first match, so a sequence of partial failures can orphan
    entries), then apply_group the move, then mark the group confirmed.
    Unlike apply_group (which deliberately leaves group.status alone -- see
    its docstring), this DOES set it, since it's the caller apply_group
    expects to make that decision: only reached after every move in
    apply_group completed without raising, so the group is genuinely
    confirmed. If apply_group raises, group.status is left untouched
    (stays "pending"), same invariant apply_group's own callers rely on."""
    manifest[:] = [m for m in manifest if m["group"] != group_index]
    apply_group(
        group, group_index, keep_idx, dest_dir, dry_run, manifest,
        recursive=recursive, scan_root=scan_root,
    )
    group.status = "confirmed"


def cached_result(cache: dict, p: Path, st: os.stat_result) -> dict | None:
    entry = cache.get(_cache_key(p))
    if entry is None or entry.get("mtime") != st.st_mtime_ns or entry.get("size") != st.st_size:
        return None
    try:
        result = dict(entry["result"])
        result["dimensions"] = tuple(result["dimensions"])
        return result
    except (KeyError, TypeError):
        return None


def store_result(cache: dict, p: Path, st: os.stat_result, result: dict) -> None:
    cache[_cache_key(p)] = {"mtime": st.st_mtime_ns, "size": st.st_size, "result": dict(result)}


# analyze() works on the full-resolution image (the metrics must; see the
# scoring trap in CLAUDE.md), so its memory scales with pixel count. One
# thread per core of 100 MP photos swapped a 24 GB Mac to a crawl, so
# _analyze_one admits work against a byte budget instead: a third of RAM.
# Ordinary 12-24 MP photos still run one per core; only huge ones queue.
#
# The price is peak RSS measured per decode route (decode_plan), rounded up
# so it stays an upper bound -- under-pricing is what swaps a 16 GB machine.
# Measured on a 24 GB / 10-core M4, 24 to 102 MP: an in-place JPEG peaks at
# ~9.3 B/px plus ~150 MB that doesn't scale (the FFT at its 2048 px cap);
# the plain cv2 decode (PNG, TIFF, a JPEG EXIF rotates sideways) at ~17;
# HEIC through PIL at ~29. The in-place route used to be ~21 like the rest:
# halving it is what lets twice as many 100 MP files analyze at once --
# 28 of them went from 12.4 s to 6.7 s. Concurrency itself is not the
# lever: forcing 6-10 of the old 2 GB analyses past the budget spent 11-25 s
# of system time in page faults and finished slower than 3.
ANALYZE_BYTES_PER_PIXEL = {"in_place": 10, "cv2": 18, "pil": 30}
ANALYZE_FIXED_BYTES = 192 * 1024 * 1024
ANALYZE_MEMORY_BUDGET = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // 3
_analyze_slots = threading.Condition()
_analyze_in_flight = 0
# Admitted first come, first served, by ticket. Groups publish in scan
# order, so a file that loses the race for the budget holds up its own group
# and every group after it -- and Condition wakeups are in no order at all:
# on 28 x 100 MP photos the very first file got in behind seven later ones,
# and the first group showed up at 4.1 s instead of 2.5.
_analyze_next_ticket = 0
_analyze_now_serving = 0


def _analyze_cost(path_str: str) -> int:
    """Estimated peak bytes for analyze(), from the header alone (PIL's open
    is lazy). Unreadable means 0: analyze() will fail on it cheaply too."""
    plan = decode_plan(path_str)
    if plan is None:
        return 0
    w, h, route = plan
    return w * h * ANALYZE_BYTES_PER_PIXEL[route] + ANALYZE_FIXED_BYTES


def _analyze_one(path_str: str) -> dict | None:
    global _analyze_in_flight, _analyze_next_ticket, _analyze_now_serving
    # The ticket comes first: pricing reads the header, and the first ten
    # files reading theirs at once would otherwise queue in whatever order
    # those reads happened to finish.
    with _analyze_slots:
        ticket = _analyze_next_ticket
        _analyze_next_ticket += 1
    # Nothing between taking a ticket and being served may raise: a ticket
    # never served would stall every analysis behind it. Capped at the
    # budget so a file bigger than all of it still runs, alone -- which is
    # also where an unpriceable one goes.
    try:
        cost = min(_analyze_cost(path_str), ANALYZE_MEMORY_BUDGET)
    except Exception:  # noqa: BLE001
        cost = ANALYZE_MEMORY_BUDGET
    with _analyze_slots:
        _analyze_slots.wait_for(lambda: _analyze_now_serving == ticket
                                and _analyze_in_flight + cost <= ANALYZE_MEMORY_BUDGET)
        _analyze_now_serving += 1
        _analyze_in_flight += cost
        _analyze_slots.notify_all()  # the next ticket may fit beside this one
    try:
        return analyze(path_str)
    except Exception:
        return None
    finally:
        with _analyze_slots:
            _analyze_in_flight -= cost
            _analyze_slots.notify_all()


_SAMPLE_BYTES = 1 << 16


def _sample_digest(path: Path) -> bytes | None:
    """blake2b of three windows -- start, middle, end -- of the file: enough
    to tell two different files of one size apart without reading either
    whole. Only ever rules a copy *out*; a match still needs _file_digest."""
    h = hashlib.blake2b(digest_size=16)
    try:
        with open(path, "rb") as f:
            size = os.fstat(f.fileno()).st_size
            for offset in sorted({0, max(0, (size - _SAMPLE_BYTES) // 2), max(0, size - _SAMPLE_BYTES)}):
                f.seek(offset)
                h.update(f.read(_SAMPLE_BYTES))
    except OSError:
        return None
    return h.digest()


def _copies_by_original(paths: list[Path], stats: dict[Path, os.stat_result],
                        digests: dict[Path, bytes | None], executor) -> dict[Path, list[Path]]:
    """{first file: [later files with exactly its bytes]} among *paths*.

    Only a file that shares its size with another is looked at, so a library
    without copies reads nothing extra. Even then a glance comes first: an
    uncompressed format makes every photo from one camera the same size, and
    reading a folder of 144 MB TIFFs whole to learn that would cost more than
    the copies it finds. Full digests land in *digests*, for the next phase
    and for _identical_to to reuse instead of reading the files again."""
    by_size: dict[int, list[Path]] = {}
    for p in paths:
        by_size.setdefault(stats[p].st_size, []).append(p)
    candidates = [p for same in by_size.values() if len(same) > 1 for p in same]
    unread = [p for p in candidates if p not in digests]
    samples = dict(zip(unread, executor.map(_sample_digest, unread)))
    seen = Counter((stats[p].st_size, s) for p, s in samples.items() if s is not None)
    alike = [p for p, s in samples.items() if s is not None and seen[stats[p].st_size, s] > 1]
    digests.update(zip(alike, executor.map(_file_digest, alike)))
    first: dict[bytes, Path] = {}
    copies: dict[Path, list[Path]] = {}
    for p in candidates:
        d = digests.get(p)
        if d is None:
            continue
        if d in first:
            copies.setdefault(first[d], []).append(p)
        else:
            first[d] = p
    return copies


def analyze_paths(paths: list[Path], cache: dict,
                  precomputed_stats: dict[Path, os.stat_result] | None = None,
                  progress_callback=None, done_callback=None,
                  digests: dict[Path, bytes | None] | None = None) -> dict[Path, dict]:
    """analyze() every path, reusing `cache` for files whose (mtime, size)
    haven't changed and running the rest through a thread pool (analyze()'s
    cv2/numpy calls release the GIL -- see the comments at
    THREAD_POOL_WORKERS's definition).

    A byte-identical copy is analyzed once, not once per copy: the same
    bytes always score the same, and copies are the commonest duplicate
    there is. Each copy still gets a result dict of its own (score_group
    writes into each), its own cache entry, callback and progress tick.
    *digests*, if given, memoizes the blake2b digests read to find them
    (build_groups shares it with _identical_to).

    If *precomputed_stats* is provided it is used instead of calling stat()
    again, and any path it doesn't cover is dropped (an unreadable file is
    tolerated here, as everywhere else in the scan).

    *progress_callback*, if given, is called as progress_callback(label,
    done, total) as each uncached item completes, instead of the default
    TTY-aware print via _print_progress -- see group_duplicates's matching
    parameter.

    *done_callback*, if given, is called as done_callback(path, result) the
    moment each path resolves -- with the result dict, or None if it failed
    to analyze -- so a caller can act on finished files without waiting for
    the whole batch (build_groups uses it to publish groups incrementally).
    It fires on this thread, not a worker's, and for cached hits too."""
    results: dict[Path, dict] = {}
    if precomputed_stats is not None:
        stats = precomputed_stats
        paths = [p for p in paths if p in stats]
    else:
        paths, stats = _stat_paths(paths)
    to_compute = []
    for p in paths:
        hit = cached_result(cache, p, stats[p])
        if hit is not None:
            # file_size is attached per result rather than in one pass at the
            # end, so a result handed to done_callback mid-batch is already
            # complete -- _group_detail and auto_apply_groups both read it.
            hit["file_size"] = stats[p].st_size
            results[p] = hit
            if done_callback is not None:
                done_callback(p, hit)
        else:
            to_compute.append(p)

    if to_compute:
        total = len(to_compute)
        tty = sys.stdout.isatty()
        original_cv2_threads = cv2.getNumThreads()
        cv2.setNumThreads(1)
        try:
            # Digests get a pool of their own: threads left idle in the
            # analysis pool would take up the analyses in no particular
            # order, and scan order is the order groups publish in.
            with ThreadPoolExecutor(max_workers=THREAD_POOL_WORKERS) as digest_pool:
                copies = _copies_by_original(to_compute, stats, {} if digests is None else digests, digest_pool)
            with ThreadPoolExecutor(max_workers=THREAD_POOL_WORKERS) as executor:
                is_copy = {q for qs in copies.values() for q in qs}
                originals = [p for p in to_compute if p not in is_copy]
                done = 0
                for p, r in zip(originals, executor.map(_analyze_one, [str(p) for p in originals])):
                    # Copies' dicts are taken before anything is added to r.
                    resolved = [(p, r)] + [(q, None if r is None else {**r, "path": str(q)})
                                           for q in copies.get(p, ())]
                    for q, rq in resolved:
                        if rq is not None:
                            # After store_result (which copies): file_size
                            # tracks the file, not the analysis, and has no
                            # business in a cache entry keyed on it.
                            store_result(cache, q, stats[q], rq)
                            rq["file_size"] = stats[q].st_size
                            results[q] = rq
                        if done_callback is not None:
                            done_callback(q, rq)
                        done += 1
                        if progress_callback is not None:
                            progress_callback("Analyzing", done, total)
                        else:
                            _print_progress("Analyzing", done, total, tty)
        finally:
            cv2.setNumThreads(original_cv2_threads)
        if progress_callback is None and tty:
            print()

    return results


def _file_digest(path: Path) -> bytes | None:
    h = hashlib.blake2b(digest_size=32)
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return None
    return h.digest()


def _identical_to(paths: list[Path], results: list[dict],
                  digests: dict[Path, bytes | None] | None = None) -> list[int | None]:
    """For each file, the first earlier file with exactly the same bytes, or
    None. Byte-identical copies are the commonest duplicate there is, and
    they tie on every metric, so a reviewer comparing them by eye is
    comparing a file with itself. Only a pair that already matches on size
    and score is read and hashed; on a fresh scan analyze_paths has usually
    hashed both already, and *digests* hands those over."""
    if digests is None:
        digests = {}

    def digest(j: int) -> bytes | None:
        if paths[j] not in digests:
            digests[paths[j]] = _file_digest(paths[j])
        return digests[paths[j]]

    out: list[int | None] = []
    for j in range(len(paths)):
        match = None
        for k in range(j):
            if out[k] is not None:
                continue  # k's own original was already compared
            if (results[k].get("file_size") != results[j].get("file_size")
                    or results[k]["quality_score"] != results[j]["quality_score"]):
                continue
            dj = digest(j)
            if dj is not None and dj == digest(k):
                match = k
                break
        out.append(match)
    return out


def _build_group(members: list[Path], analyzed: dict[Path, dict],
                 digests: dict[Path, bytes | None] | None = None) -> Group | None:
    """Assemble one finished Group from a raw cluster, or None if fewer than
    two of its files survived analysis (it's no longer a duplicate group).

    Scoring, the best-first permutation and the close-call flag all happen
    here, so a Group is complete and correctly ordered the first time anyone
    sees it -- see build_groups' *group_callback*, which hands groups out
    while the scan is still running. A published group must never reorder
    afterwards; that's what makes its indices safe to act on."""
    valid = [(p, analyzed[p]) for p in members if p in analyzed]
    if len(valid) < 2:
        return None
    paths, results = zip(*valid)
    paths, results = list(paths), list(results)
    score_group(results)

    # Reorder the group best-first rather than just recording which index
    # won, so the suggested file is always [1] -- leftmost column in the
    # web table, first preview on the stage. Reviewing is a high-volume
    # loop and the eye shouldn't have to hunt for the ★ in a different
    # position every group. Both lists are permuted by the same order, so
    # every index downstream (current_pick, the manifest, /api/thumb's j,
    # the digit-key shortcuts) stays consistent -- there is no "original
    # index" left to translate back to.
    #
    # Byte-identical copies score *exactly* equal -- min-max normalizing a
    # group whose metrics are all the same gives every file the same
    # number -- so on the commonest duplicate of all the tie-break is the
    # whole decision. Filename order lost it: " 2.jpg" and " copy.jpg"
    # both sort before ".jpg" (space < period), so the tool suggested
    # keeping the copy. Shorter name wins a tie instead, because every
    # convention for naming a derived copy *appends* to the original --
    # " copy", " 2", " (1)", "-edited", "_v2". Full name last keeps the
    # order total, so identical copies don't shuffle between scans.
    #
    # ponytail: name length is a heuristic, and it only ever runs on an
    # exact tie -- where the pixel metrics have said the files are
    # indistinguishable and something has to break it. When they differ at
    # all, the metrics still decide alone.
    order = sorted(
        range(len(results)),
        key=lambda i: (-results[i]["quality_score"], len(paths[i].name), paths[i].name),
    )
    paths = [paths[i] for i in order]
    results = [results[i] for i in order]

    # bool(...): quality_score can be a numpy float64 (propagated from
    # analyze()'s metrics), and `<` against one produces numpy.bool_, not
    # Python bool -- `and` returns its second operand as-is rather than
    # coercing it, so close_call would silently end up numpy.bool_ too.
    # That's fine for a plain truthy "if g.is_close_call" check, but
    # numpy.bool_ (unlike numpy.float64) isn't a subclass of its Python
    # equivalent and isn't JSON-serializable -- the web front end's
    # /api/state was the first consumer to actually hit this.
    #
    # The close call measures the top file against the best one that actually
    # differs from it: an identical copy at index 1 ties it exactly, and used
    # to flag a group as close when the real choice wasn't (tests/
    # test_group_ordering.py). A group of nothing but copies is no call.
    identical_to = _identical_to(paths, results, digests)
    differing = [j for j in range(1, len(results)) if identical_to[j] != 0]
    close_call = bool(
        differing
        and results[0]["quality_score"] - results[differing[0]]["quality_score"] < CLOSE_CALL_MARGIN
    )
    return Group(
        paths=paths,
        results=results,
        # Not generated here: decoding+downscaling every group's images up
        # front stalls scan completion on large scans, and groups the user
        # never navigates to (e.g. quits early) would pay that cost for
        # nothing. Generated lazily on first request instead (see
        # duplicates_web._cached_render).
        thumbnails=None,
        suggested_idx=0,
        current_pick=0,
        is_close_call=close_call,
        identical_to=identical_to,
        all_identical=not differing,
    )


def build_groups(
    directory: Path, threshold: int, recursive: bool = False, dest_dir: Path | None = None,
    progress_callback=None, hash_cache: dict | None = None, analyze_cache: dict | None = None,
    group_callback=None, stats: dict | None = None,
) -> list[Group]:
    """*progress_callback*, if given, is passed straight through to
    group_duplicates/analyze_paths -- see their matching parameter. None
    (the default) preserves the CLI's existing TTY-aware stdout printing
    unchanged.

    *group_callback*, if given, is called with each Group as it is finished,
    in returned-list order, while the analyze phase is still running. Every
    grouped file still goes through one thread pool, so total throughput is
    unchanged -- what changes is that a caller (the web front end) can start
    showing groups without waiting for the last one. Groups are only ever
    appended and a published group's contents never change afterwards, which
    is what makes its index safe to hand out mid-scan.

    *hash_cache*/*analyze_cache* are caller-owned and in-memory only: a scan
    writes nothing to *directory*. Passing dicts that outlive this call is
    what makes a rescan of an already-scanned library cheap (the web UI's
    Session does exactly that); omitting them means every path is recomputed
    from scratch, which is the right default for one-shot --auto runs. Both
    key on path + mtime + size, so entries stay valid across rescans with a
    different threshold or recursive setting -- neither affects a file's hash
    or its quality metrics.

    *stats*, if given, is filled in as the scan goes: "images" (files found
    with an image extension) once listing is done, then "unreadable" (files
    that failed to decode for hashing or analysis, and so were left out).
    An empty result means nothing without them -- no images at all (usually
    a top-level scan of a library kept in subfolders) wants --recursive,
    not a looser threshold."""
    paths = find_images(directory, recursive=recursive, exclude_dir=dest_dir)
    if stats is not None:
        stats.update(images=len(paths), unreadable=0)

    if hash_cache is None:
        hash_cache = {}
    if analyze_cache is None:
        analyze_cache = {}

    # Shared by every phase: hashing, analysis and _identical_to all want to
    # know which files are byte-identical, and none should read one twice.
    digests: dict[Path, bytes | None] = {}
    raw_groups = group_duplicates(paths, threshold, hash_cache, progress_callback=progress_callback,
                                  digests=digests)
    if stats is not None:
        # A failed decode is cached as hash None (see cached_hash).
        entries = (hash_cache.get(_cache_key(p)) for p in paths)
        stats["unreadable"] = sum(1 for e in entries if e is not None and e["hash"] is None)

    # Compute stats for the grouped files once and pass to analyze_paths,
    # rather than letting it call stat() again on files already stat()'d
    # during the hash phase (the same Path objects are reused).
    grouped_paths = [p for members in raw_groups for p in members]
    grouped_paths, grouped_stats = _stat_paths(grouped_paths)

    # Publish each group as soon as its own files finish analyzing, instead
    # of after the whole analyze phase. Every grouped file is still handed to
    # one thread pool up front, so total throughput is identical -- but a
    # group whose files are done need not wait behind the rest of the
    # library. analyze() costs ~8-19x what hashing one file does, so on a
    # library with duplicates in it that wait is most of the scan.
    #
    # Groups are published in raw_groups order (via *cursor*), never in
    # completion order: the index a caller acts on has to mean the same group
    # for the life of the scan, and analyze_paths resolves files in
    # submission order anyway, so waiting for group N before publishing N+1
    # costs nothing in practice.
    groups: list[Group] = []
    kept = set(grouped_paths)
    remaining = [sum(1 for p in members if p in kept) for members in raw_groups]
    owner = {p: gi for gi, members in enumerate(raw_groups) for p in members if p in kept}
    analyzed: dict[Path, dict] = {}
    cursor = 0

    def publish_ready() -> None:
        nonlocal cursor
        while cursor < len(raw_groups) and remaining[cursor] == 0:
            group = _build_group(raw_groups[cursor], analyzed, digests)
            cursor += 1
            if group is None:
                continue
            groups.append(group)
            if group_callback is not None:
                group_callback(group)

    def on_analyzed(p: Path, result: dict | None) -> None:
        if result is not None:
            analyzed[p] = result
        elif stats is not None:
            stats["unreadable"] += 1
        remaining[owner[p]] -= 1
        publish_ready()

    analyze_paths(
        grouped_paths, analyze_cache, precomputed_stats=grouped_stats,
        progress_callback=progress_callback, done_callback=on_analyzed, digests=digests,
    )
    # Nothing calls back for a group whose every file vanished between
    # find_images and _stat_paths, so those are flushed (and dropped) here.
    publish_ready()
    return groups
