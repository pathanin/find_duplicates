"""
find_duplicates.py

Scans the current directory (top level only) for images that look like the
same photo saved at different sizes/qualities, groups them with a perceptual
hash, scores each candidate using compare_image_quality.analyze(), and lets
you confirm which one to keep in a Textual TUI. Non-kept files are moved to
./_duplicates/, never deleted -- restoring one is a manual move back out of
that folder (the scan is top-level only, so the original location is always
the scanned directory).

Usage:
    python find_duplicates.py [directory] [--threshold N] [--dest DIR] [--dry-run]

Requires:
    pip install opencv-python-headless numpy textual textual-image pillow
"""

import argparse
import functools
import json
import math
import os
import shutil
import tempfile
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PILImage
from rich.markup import escape as rich_escape
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import DataTable, Footer, Header, Label, ListItem, ListView, Static
from textual_image.widget import Image as PreviewImage

from compare_image_quality import analyze

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
PREVIEW_MAX_SIDE = 800
CLOSE_CALL_MARGIN = 0.08  # quality_score gap below which we flag "close call"
CACHE_FILENAME = ".find_duplicates_cache.json"
HASH_CACHE_FILENAME = ".find_duplicates_hash_cache.json"
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
    "brisque": "no-reference perceptual quality score (needs optional `brisque` package)",
    "niqe": "no-reference perceptual quality score (needs optional `pyiqa` package)",
}


# ---------------------------------------------------------------------------
# Scanning + perceptual hashing + grouping
# ---------------------------------------------------------------------------

def find_images(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


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


def phash(gray: np.ndarray) -> int:
    """Classic 64-bit DCT perceptual hash: resize small, keep low frequencies,
    threshold against their mean. Robust to resizing/recompression, which is
    exactly the kind of "same photo, different export" duplicate we're after.

    The on-disk hash cache (.find_duplicates_hash_cache.json) keys on path +
    mtime + size, not on this function's code -- if you change phash or
    load_hash_gray while testing against real images, delete that cache file
    first, or you'll be silently served old hashes and wrongly conclude your
    change had no effect on grouping."""
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(resized)
    low = dct[:8, :8]
    avg = (low.sum() - low[0, 0]) / 63.0
    bits = low > avg
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


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


def load_hash_cache(directory: Path) -> dict:
    path = directory / HASH_CACHE_FILENAME
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _write_json_atomic(path: str, data: dict) -> None:
    """Write a JSON file atomically: write to a temp file in the same
    directory, then rename over the target.  Prevents concurrent or
    interrupted writes from leaving a truncated JSON file."""

    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp", prefix=".find_duplicates_")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def save_hash_cache(directory: Path, cache: dict) -> None:
    _write_json_atomic(str(directory / HASH_CACHE_FILENAME), cache)


def cached_hash(cache: dict, p: Path, st: os.stat_result) -> int | None:
    """Returns None both when there's no entry and when the cached entry
    itself is None (the file failed to decode/hash last time) -- a
    permanently-corrupt file is simply re-attempted every run, no worse
    than today's uncached behavior for that one file."""
    entry = cache.get(str(p.resolve()))
    if entry is None or entry.get("mtime") != st.st_mtime_ns or entry.get("size") != st.st_size:
        return None
    return entry["hash"]


def store_hash(cache: dict, p: Path, st: os.stat_result, hash_value: int | None) -> None:
    cache[str(p.resolve())] = {"mtime": st.st_mtime_ns, "size": st.st_size, "hash": hash_value}


def _hash_one(p: Path) -> int | None:
    img = load_hash_gray(p)
    return phash(img) if img is not None else None


def _print_progress(label: str, done: int, total: int, tty: bool) -> None:
    """Incremental progress for a long-running scan phase. On a TTY,
    overwrites the same terminal line via carriage return so it doesn't spam
    scrollback; when stdout isn't a TTY (redirected to a file, running under
    test), falls back to occasional plain lines instead of \r-laden output."""
    if tty:
        print(f"\r{label}: {done}/{total}", end="", flush=True)
    elif done == total or done % 100 == 0:
        print(f"{label}: {done}/{total}")


def group_duplicates(paths: list[Path], threshold: int, cache: dict) -> list[list[Path]]:
    """Groups `paths` by perceptual-hash Hamming distance, reusing `cache`
    for files whose (mtime, size) haven't changed (see cached_hash/
    store_hash above) so a re-scan of an already-hashed directory doesn't
    re-decode every old file. The uncached subset always hashes through a
    thread pool -- see THREAD_POOL_WORKERS for why threads (not a process
    pool) win here."""
    stats = {p: p.stat() for p in paths}
    hashes: dict[Path, int | None] = {}
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
                computed = executor.map(_hash_one, to_compute)
                for done, (p, h) in enumerate(zip(to_compute, computed), start=1):
                    store_hash(cache, p, stats[p], h)
                    hashes[p] = h
                    _print_progress("Hashing", done, total, tty)
            if tty:
                print()
        finally:
            cv2.setNumThreads(original_cv2_threads)

    hash_list = [hashes[p] for p in paths]

    uf = UnionFind(len(paths))
    for i in range(len(paths)):
        if hash_list[i] is None:
            continue
        for j in range(i + 1, len(paths)):
            if hash_list[j] is None:
                continue
            if hamming(hash_list[i], hash_list[j]) <= threshold:
                uf.union(i, j)

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


def load_cache(directory: Path) -> dict:
    path = directory / CACHE_FILENAME
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_cache(directory: Path, cache: dict) -> None:
    _write_json_atomic(str(directory / CACHE_FILENAME), cache)


def cached_result(cache: dict, p: Path, st: os.stat_result) -> dict | None:
    entry = cache.get(str(p.resolve()))
    if entry is None or entry.get("mtime") != st.st_mtime_ns or entry.get("size") != st.st_size:
        return None
    try:
        result = dict(entry["result"])
        result["dimensions"] = tuple(result["dimensions"])
        return result
    except (KeyError, TypeError):
        return None


def store_result(cache: dict, p: Path, st: os.stat_result, result: dict) -> None:
    cache[str(p.resolve())] = {"mtime": st.st_mtime_ns, "size": st.st_size, "result": dict(result)}


def _analyze_one(path_str: str) -> dict | None:
    try:
        return analyze(path_str)
    except Exception:
        return None


def analyze_paths(paths: list[Path], cache: dict,
                  precomputed_stats: dict[Path, os.stat_result] | None = None) -> dict[Path, dict]:
    """analyze() every path, reusing `cache` for files whose (mtime, size)
    haven't changed and running the rest through a thread pool (analyze()'s
    cv2/numpy calls release the GIL -- see the comments at
    THREAD_POOL_WORKERS's definition).

    If *precomputed_stats* is provided, it must cover every path in *paths*
    and will be used instead of calling stat() again."""
    results: dict[Path, dict] = {}
    if precomputed_stats is not None:
        stats = precomputed_stats
    else:
        stats = {p: p.stat() for p in paths}
    to_compute = []
    for p in paths:
        hit = cached_result(cache, p, stats[p])
        if hit is not None:
            results[p] = hit
        else:
            to_compute.append(p)

    if to_compute:
        total = len(to_compute)
        tty = sys.stdout.isatty()
        original_cv2_threads = cv2.getNumThreads()
        cv2.setNumThreads(1)
        try:
            with ThreadPoolExecutor(max_workers=THREAD_POOL_WORKERS) as executor:
                for done, (p, r) in enumerate(
                    zip(to_compute, executor.map(_analyze_one, [str(p) for p in to_compute])), start=1
                ):
                    if r is not None:
                        store_result(cache, p, stats[p], r)
                        results[p] = r
                    _print_progress("Analyzing", done, total, tty)
        finally:
            cv2.setNumThreads(original_cv2_threads)
        if tty:
            print()

    for p in results:
        results[p]["file_size"] = stats[p].st_size
    return results


def build_groups(directory: Path, threshold: int) -> list[Group]:
    paths = find_images(directory)

    hash_cache = load_hash_cache(directory)
    # A shallow copy suffices to detect changes: store_hash always assigns a
    # brand-new dict literal to cache[key] rather than mutating an existing
    # entry in place, so a stale key's value in this snapshot keeps pointing
    # at the old dict even after group_duplicates rewrites cache[key]. A
    # plain len() comparison misses that case -- rehashing a modified file
    # replaces its entry without changing the key count, so the refreshed
    # value would never get persisted and the file would be recomputed on
    # every subsequent scan.
    hash_cache_snapshot = dict(hash_cache)
    raw_groups = group_duplicates(paths, threshold, hash_cache)
    if hash_cache != hash_cache_snapshot:
        save_hash_cache(directory, hash_cache)

    cache = load_cache(directory)
    cache_snapshot = dict(cache)
    # Compute stats for the grouped files once and pass to analyze_paths,
    # rather than letting it call stat() again on files already stat()'d
    # during the hash phase (the same Path objects are reused).
    grouped_paths = [p for members in raw_groups for p in members]
    grouped_stats = {p: p.stat() for p in grouped_paths}
    analyzed = analyze_paths(grouped_paths, cache, precomputed_stats=grouped_stats)
    if cache != cache_snapshot:
        save_cache(directory, cache)

    groups = []
    for members in raw_groups:
        # Skip files that failed analysis (not in analyzed dict).
        valid = [(p, analyzed[p]) for p in members if p in analyzed]
        if len(valid) < 2:
            continue  # no longer a duplicate group
        members, results = zip(*valid)
        members = list(members)
        results = list(results)
        score_group(results)

        order = sorted(range(len(results)), key=lambda i: -results[i]["quality_score"])
        suggested_idx = order[0]
        close_call = len(order) > 1 and (
            results[order[0]]["quality_score"] - results[order[1]]["quality_score"] < CLOSE_CALL_MARGIN
        )
        groups.append(
            Group(
                paths=members,
                results=results,
                # Not generated here: decoding+downscaling every group's images
                # up front stalls TUI startup on large scans, and groups the
                # user never navigates to (e.g. quits early) would pay that
                # cost for nothing. refresh_detail() generates on first view.
                thumbnails=None,
                suggested_idx=suggested_idx,
                current_pick=suggested_idx,
                is_close_call=close_call,
            )
        )
    return groups


# ---------------------------------------------------------------------------
# TUI
# ---------------------------------------------------------------------------

# Every scored row is labeled with what a bigger/smaller number means, since
# a raw number is meaningless without knowing which direction is "better".
# Dimensions/file size carry no such label since they aren't part of the
# score at all -- that's explained once, in the '?' help screen, rather than
# on every row.
METRIC_ROWS = [
    ("Dimensions", lambda r: f"{r['dimensions'][0]}x{r['dimensions'][1]}"),
    ("File size", lambda r: humansize(r["file_size"])),
    ("Sharpness (higher better)", lambda r: f"{r['sharpness_normalized']:.1f}"),
    ("Eff. res. fraction (higher better)", lambda r: f"{r['effective_resolution_fraction']:.3f}"),
    ("Eff. res. px equiv (higher better)", lambda r: f"{r['effective_resolution_px_equiv']:.0f}"),
    ("Noise sigma (lower better)", lambda r: f"{r['noise_sigma']:.3f}"),
    ("Blockiness (lower better)", lambda r: f"{r['blockiness']:.3f}"),
    ("BRISQUE (lower better)", lambda r: f"{r['brisque']:.2f}" if r.get("brisque") is not None else "n/a"),
    ("NIQE (lower better)", lambda r: f"{r['niqe']:.2f}" if r.get("niqe") is not None else "n/a"),
    ("Quality score (higher better)", lambda r: f"{r['quality_score']:.3f}"),
]


@functools.cache
def _help_body() -> str:
    lines = [
        "QUALITY SCORE",
        "A weighted composite of the metrics below, normalized 0-1 within",
        "this group only (min-max against the other files here -- not",
        "comparable across different photos). It's a hand-tuned heuristic,",
        "not a lab measurement: treat it as a strong hint, not a verdict,",
        "especially on a close call.",
        "",
        "Dimensions and file size are shown for reference only and do NOT",
        "factor into the score. A smaller or larger file is not, by itself,",
        "a quality signal -- e.g. a noisier image can outweigh a cleaner one",
        "in stored bytes without containing any more real detail.",
        "",
        "WEIGHTED METRICS, sorted by influence:",
    ]
    for name, weight in sorted(METRIC_WEIGHTS.items(), key=lambda kv: -abs(kv[1])):
        direction = "higher better" if weight > 0 else "lower better"
        lines.append(f"  {abs(weight):.2f}  {name} ({direction})")
        lines.append(f"        {METRIC_DESCRIPTIONS[name]}")
    lines += [
        "",
        "KEYBOARD LAYOUTS",
        "If typed letters seem to do nothing, an alternate keyboard layout is",
        "probably remapping them to different characters before the terminal",
        "ever sees them. Control keys aren't remapped that way, so each core",
        "action also has a layout-independent alias: Enter = confirm,",
        "Delete/Backspace = skip, Escape = finish, F1 = this help screen.",
    ]
    return "\n".join(lines)


class HelpScreen(ModalScreen):
    CSS = """
    HelpScreen { align: center middle; }
    #help-box {
        width: 78; height: auto; max-height: 90%; border: solid $surface;
        padding: 1 2; background: $panel;
    }
    """
    BINDINGS = [Binding("escape,q,question_mark", "close_help", "Close")]

    def compose(self) -> ComposeResult:
        with Vertical(id="help-box"):
            yield Static(_help_body())
            yield Static("\n[Esc / q / ? to close]")

    def action_close_help(self) -> None:
        self.dismiss()


class DuplicateReviewApp(App):
    TITLE = "Duplicate image review"

    # confirm's Enter alias is a priority binding (see BINDINGS below), which
    # is checked against the *full* screen chain and so pierces modals like
    # HelpScreen and the command palette's input -- unlike every other
    # binding here, it does not respect modal boundaries on its own. This app
    # has no use for the palette, and disabling it removes that surface
    # entirely rather than trying to make "c"/"enter" behave inside its Input.
    ENABLE_COMMAND_PALETTE = False

    CSS = """
    #body { height: 1fr; }
    #sidebar { width: 40; border-right: solid $surface; }
    #detail { width: 1fr; height: 1fr; }
    #images-row { height: 24; overflow-x: auto; }
    .preview-box {
        width: 1fr; min-width: 30; height: 22; border: solid $surface; padding: 0 1;
        align: center middle;
    }

    .preview-box.picked { border: heavy $success; }
    .preview-label { width: 100%; text-wrap: nowrap; text-overflow: ellipsis; }
    .preview-image { width: auto; height: auto; }
    #metrics-table { height: 1fr; }
    #status { height: 3; background: $surface; content-align: left top; padding: 0 1; }
    """

    # Non-latin/alternate keyboard layouts remap letter keys to different
    # Unicode characters entirely (the OS translates the keystroke before the
    # terminal ever sees it), so a 'c'/'s'/'q' binding can silently stop
    # responding the moment the active input source isn't English. Control
    # keys aren't part of that character remapping, so each core action also
    # has a layout-independent alias.
    BINDINGS = [
        Binding("left", "pick_relative(-1)", "Prev pick"),
        Binding("right", "pick_relative(1)", "Next pick"),
        # priority=True: ListView and DataTable both bind "enter" to their own
        # select_cursor, which would otherwise swallow it before it reaches
        # this binding whenever either has focus (ListView is the default).
        Binding("c,enter", "confirm", "Confirm keep", priority=True),
        Binding("s,delete,backspace", "skip", "Skip group"),
        Binding("o", "open_fullres", "Open full-res"),
        Binding("question_mark,f1", "show_help", "Help"),
        Binding("1", "pick(1)", "Pick 1", show=False),
        Binding("2", "pick(2)", "Pick 2", show=False),
        Binding("3", "pick(3)", "Pick 3", show=False),
        Binding("4", "pick(4)", "Pick 4", show=False),
        Binding("5", "pick(5)", "Pick 5", show=False),
        Binding("6", "pick(6)", "Pick 6", show=False),
        Binding("7", "pick(7)", "Pick 7", show=False),
        Binding("8", "pick(8)", "Pick 8", show=False),
        Binding("9", "pick(9)", "Pick 9", show=False),
        Binding("q,escape", "quit_and_apply", "Finish"),
    ]

    # Every key bound to a state-mutating action (confirm/skip), including
    # their layout-independent aliases -- Footer picks *some* one of a
    # compound binding's keys to render as its clickable button, and which
    # one it picks depends on internal Binding ordering/priority, not on
    # source order here. Blocking by literal key alone (e.g. just "c") broke
    # the instant "enter" became the one Footer chose to show for confirm.
    _DESTRUCTIVE_KEYS = frozenset(
        key.strip()
        for binding in BINDINGS
        if isinstance(binding, Binding) and binding.action in ("confirm", "skip")
        for key in binding.key.split(",")
    )

    def __init__(self, groups: list[Group], dest_dir: Path, dry_run: bool):
        super().__init__()
        self.groups = groups
        self.dest_dir = dest_dir
        self.dry_run = dry_run
        # In-memory only -- tracks moves within this session so a re-pick or
        # un-confirm can find what to reverse (see _apply/_unapply). Not
        # persisted to disk: moved files are never deleted, so restoring one
        # after the app exits is just a manual move back out of dest_dir.
        self.manifest: list[dict] = []
        self.active_index = 0

    def simulate_key(self, key: str) -> None:
        """Textual's Footer renders key bindings as clickable buttons, whose
        click handler routes through this exact method. That turns "Confirm
        keep" into a real button one stray click away from silently moving
        files -- e.g. a click meant to focus/scroll the terminal after the
        scan finishes. Confirm/skip mutate group state, so they must only
        fire from a deliberate keypress, never a footer click."""
        if key not in self._DESTRUCTIVE_KEYS:
            super().simulate_key(key)

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="body"):
            with Vertical(id="sidebar"):
                yield ListView(
                    *[ListItem(Label(self._group_label(i))) for i in range(len(self.groups))], id="group-list"
                )
            with Vertical(id="detail"):
                yield Horizontal(id="images-row")
                yield DataTable(id="metrics-table")
        yield Static(id="status")
        yield Footer()

    async def on_mount(self) -> None:
        self.query_one(DataTable).cursor_type = "column"
        await self.refresh_detail(0)
        self.set_focus(self.query_one("#group-list", ListView))

    def _group_label(self, i: int) -> str:
        g = self.groups[i]
        marker = {"pending": "◻", "confirmed": "✔", "skipped": "—"}[g.status]
        close = " ⚠" if g.is_close_call else ""
        pick = ""
        if g.status == "confirmed":
            pick = f" → [{g.current_pick + 1}]"
        return f"{marker} Group {i + 1} ({len(g.paths)} files){close}{pick}"

    async def _relabel(self, i: int) -> None:
        item = self.query_one("#group-list", ListView).children[i]
        item.query_one(Label).update(self._group_label(i))

    async def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        if event.list_view.id == "group-list" and event.list_view.index is not None:
            self.active_index = event.list_view.index
            await self.refresh_detail(self.active_index)

    def _pick_box_classes(self, group: Group, idx: int) -> str:
        # Only the picked box gets a distinguishing (green) border. A
        # colored border on the suggested-but-not-picked box, too, read as a
        # second selection state and confused users about which file was
        # actually kept -- so the suggestion is marked by the "★ suggested"
        # label tag alone (see _pick_label_text) and the box otherwise falls
        # back to .preview-box's plain default border, same as any other
        # non-picked box.
        classes = "preview-box"
        if idx == group.current_pick:
            classes += " picked"
        return classes

    def _pick_label_text(self, group: Group, idx: int) -> str:
        tag = ""
        if idx == group.current_pick:
            tag = "[bold green]✔ KEEP[/]  "
        elif idx == group.suggested_idx:
            tag = "[italic]★ suggested[/]  "
        # Tag (and the pick number) come before the filename, not after, so a
        # narrow terminal's ellipsis truncates the recoverable filename tail
        # rather than the keep/suggested indicator itself.
        return f"{tag}[{idx + 1}] {rich_escape(group.paths[idx].name)}"

    async def refresh_detail(self, i: int) -> None:
        """Full re-render: switching which group is displayed. Only this path
        needs to touch images/metrics at all -- moving the pick *within* the
        same group goes through _update_pick_ui instead, since none of that
        content depends on current_pick."""
        group = self.groups[i]
        if group.thumbnails is None:
            group.thumbnails = [make_thumbnail(p) for p in group.paths]

        row = self.query_one("#images-row", Horizontal)
        await row.remove_children()
        boxes = []
        for idx, thumb in enumerate(group.thumbnails):
            classes = self._pick_box_classes(group, idx)
            label_text = self._pick_label_text(group, idx)
            image = PreviewImage(thumb, classes="preview-image")
            boxes.append(Vertical(Label(label_text, classes="preview-label"), image, classes=classes))
        await row.mount(*boxes)

        table = self.query_one(DataTable)
        table.clear(columns=True)
        table.add_column("Metric")
        for idx in range(len(group.paths)):
            header = f"[{idx + 1}]" + (" ★" if idx == group.suggested_idx else "")
            table.add_column(header)
        for label, fn in METRIC_ROWS:
            table.add_row(label, *[fn(r) for r in group.results])

        self.query_one("#status", Static).update(self._status_text())

    async def _update_pick_ui(self, old_pick: int, new_pick: int) -> None:
        """Lightweight counterpart to refresh_detail for moving the pick
        within the same group: only the old/new picked boxes' CSS class and
        label text can have changed (nothing in METRIC_ROWS or the table
        headers depends on current_pick), so this skips rebuilding the
        PreviewImage widgets (avoids re-encoding/re-transmitting every
        terminal image) and the DataTable entirely."""
        group = self.groups[self.active_index]
        boxes = self.query_one("#images-row", Horizontal).children
        for idx in {old_pick, new_pick}:
            if not (0 <= idx < len(boxes)):
                continue
            box = boxes[idx]
            box.set_class(idx == group.current_pick, "picked")
            box.query_one(Label).update(self._pick_label_text(group, idx))
        self.query_one("#status", Static).update(self._status_text())

    def _status_text(self) -> str:
        confirmed = sum(1 for g in self.groups if g.status == "confirmed")
        skipped = sum(1 for g in self.groups if g.status == "skipped")
        pending = len(self.groups) - confirmed - skipped
        mode = "  [DRY RUN]" if self.dry_run else ""

        group = self.groups[self.active_index]
        n_removed = len(group.paths) - 1
        if group.status == "pending":
            plural = "s" if n_removed != 1 else ""
            action = f"keep [{group.current_pick + 1}] {rich_escape(group.paths[group.current_pick].name)}"
            if n_removed > 0:
                action += f", move {n_removed} other file{plural}"
            if group.current_pick != group.suggested_idx:
                line3 = (
                    f"your pick [{group.current_pick + 1}]  ·  "
                    f"★ suggested [{group.suggested_idx + 1}] {rich_escape(group.paths[group.suggested_idx].name)}"
                )
            else:
                line3 = ""
        else:
            action = f"already {group.status}"
            line3 = ""

        return (
            f"Groups: {len(self.groups)}  confirmed={confirmed}  skipped={skipped}  pending={pending}{mode}\n"
            f"{action}\n{line3}"
        )

    async def action_pick(self, n: int) -> None:
        group = self.groups[self.active_index]
        if group.status == "confirmed":
            return
        idx = n - 1
        if 0 <= idx < len(group.paths) and idx != group.current_pick:
            old_pick = group.current_pick
            group.current_pick = idx
            await self._update_pick_ui(old_pick, idx)

    async def action_pick_relative(self, delta: int) -> None:
        group = self.groups[self.active_index]
        if group.status == "confirmed":
            return
        old_pick = group.current_pick
        group.current_pick = (group.current_pick + delta) % len(group.paths)
        await self._update_pick_ui(old_pick, group.current_pick)

    async def action_confirm(self) -> None:
        # confirm's Enter alias is a priority binding, which pierces modals
        # (see ENABLE_COMMAND_PALETTE above) -- without this guard, Enter
        # pressed just to read the help screen silently confirms the group
        # underneath it.
        if len(self.screen_stack) > 1:
            return
        i = self.active_index
        group = self.groups[i]

        if group.status == "confirmed":
            # Skip if pick hasn't changed
            entry = next((m for m in self.manifest if m["group"] == i), None)
            if entry is not None and entry["kept"] == str(group.paths[group.current_pick]):
                return
            self._unapply(i)
            self._apply(i, group.current_pick)
        elif group.status == "pending" or group.status == "skipped":
            # A prior _apply() may have failed partway through (disk full,
            # permission error) and left the group "pending" with a manifest
            # entry recording whatever it did manage to move (see _apply's
            # comment on that invariant). _unapply is a no-op when there's
            # no such entry, so this is safe to call unconditionally -- it
            # reverses that leftover state before retrying rather than
            # trying to move files that are already gone from their source.
            self._unapply(i)
            self._apply(i, group.current_pick)
        await self._relabel(i)
        await self._advance()

    async def action_skip(self) -> None:
        if len(self.screen_stack) > 1:
            return
        i = self.active_index
        group = self.groups[i]

        if group.status == "pending":
            # As in action_confirm: a prior failed _apply() may have left a
            # partial move recorded against this still-"pending" group.
            # Reverse it before skipping, or the moved files would be
            # stranded in dest_dir/ while the group reads "skipped".
            self._unapply(i)
            group.status = "skipped"
            await self._relabel(i)
            await self._advance()
        elif group.status == "confirmed":
            self._unapply(i)
            group.status = "skipped"
            await self._relabel(i)
            await self._advance()
        elif group.status == "skipped":
            group.status = "pending"
            await self._relabel(i)

    def action_open_fullres(self) -> None:
        group = self.groups[self.active_index]
        paths = [str(p) for p in group.paths]
        try:
            if sys.platform == "darwin":
                subprocess.run(["open", "-a", "Preview", *paths], check=False)
            elif sys.platform.startswith("linux"):
                subprocess.run(["xdg-open", paths[group.current_pick]], check=False)
            else:
                self.notify("Full-resolution open isn't supported on this OS.", severity="warning")
        except FileNotFoundError:
            self.notify("Couldn't find an image viewer to open the file with.", severity="error")

    def action_show_help(self) -> None:
        self.push_screen(HelpScreen())

    def _unapply(self, i: int) -> None:
        """Reverse file moves for a confirmed group using the in-memory
        manifest. Does NOT change the group status — the caller decides."""
        entry = next((m for m in self.manifest if m["group"] == i), None)
        if not entry:
            return
        if entry["dry_run"]:
            # Dry-run moves never touch the filesystem (see _apply), so
            # there's nothing to check for on disk -- the manifest entry
            # itself is the only state a dry-run "move" left behind, and
            # reversing it is just dropping that entry.
            self.manifest.remove(entry)
            return
        restored = []
        try:
            for moved in entry["moved"]:
                src = Path(moved["from"])
                dst = Path(moved["to"])
                if dst.exists() and not src.exists():
                    # shutil.move, not Path.rename: --dest may point at a
                    # different filesystem than the scanned directory, and
                    # a plain rename raises OSError (EXDEV) cross-device
                    # where shutil.move falls back to copy+remove -- the
                    # same reason _apply below uses shutil.move rather than
                    # rename for the forward move.
                    shutil.move(str(dst), str(src))
                    restored.append(moved)
        finally:
            # Keep tracking whatever wasn't restored (never just the fact
            # that *something* was restored) even if a move raised partway
            # through, so self.manifest always reflects real filesystem
            # state for the rest of this session -- the same invariant
            # _apply preserves on the forward move (see
            # test_manifest_crash_safety.py). Dropping the whole entry here
            # regardless of partial failure would silently lose track of
            # files still sitting in dest_dir/.
            remaining = [m for m in entry["moved"] if m not in restored]
            if remaining:
                entry["moved"] = remaining
            else:
                self.manifest.remove(entry)

    def _apply(self, i: int, keep_idx: int) -> None:
        group = self.groups[i]
        group.current_pick = keep_idx
        kept_path = group.paths[keep_idx]
        # If the kept file is a symlink, resolve to the real path so we
        # avoid moving the real target out from under it (leaving a dangling
        # link). The symlink itself is kept.
        kept_real = kept_path.resolve() if kept_path.is_symlink() else None
        moved = []
        try:
            for idx, path in enumerate(group.paths):
                if idx == keep_idx:
                    continue
                # Skip files that are symlinks to the same real path as the
                # kept file (they share content on disk -- moving the target
                # would leave the kept path dangling).
                if kept_real is not None and path.resolve() == kept_real:
                    continue
                dest = self._dest_for(path)
                if not self.dry_run:
                    shutil.move(str(path), str(dest))
                moved.append({"from": str(path), "to": str(dest)})
        finally:
            # Record whatever was actually moved even if a move raised partway
            # through (disk full, permission error) -- otherwise files already
            # moved to disk would have no manifest entry and become unrecoverable
            # by _unapply within this session.
            self.manifest.append(
                {
                    "group": i,
                    "kept": str(group.paths[keep_idx]),
                    "moved": moved,
                    "dry_run": self.dry_run,
                }
            )
        # Only mark confirmed after all moves completed; if an exception
        # propagates from the try block, the group stays "pending" so the
        # user can see it's in an inconsistent state.
        group.status = "confirmed"

    def _dest_for(self, path: Path) -> Path:
        if not self.dry_run:
            self.dest_dir.mkdir(parents=True, exist_ok=True)
        dest = self.dest_dir / path.name
        n = 1
        while dest.exists():
            dest = self.dest_dir / f"{path.stem}_dup{n}{path.suffix}"
            n += 1
        return dest

    async def _advance(self) -> None:
        list_view = self.query_one("#group-list", ListView)
        n = len(self.groups)
        for offset in range(1, n + 1):
            j = (self.active_index + offset) % n
            if self.groups[j].status == "pending":
                list_view.index = j
                return
        self.notify("All groups reviewed. Press q to finish.", severity="information")

    def action_quit_and_apply(self) -> None:
        self.exit()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _threshold_arg(s: str) -> int:
    v = int(s)
    if not 0 <= v <= 64:
        raise argparse.ArgumentTypeError(f"threshold must be 0-64, got {v}")
    return v


def main() -> None:
    parser = argparse.ArgumentParser(description="Find and review potential duplicate images by quality.")
    parser.add_argument("directory", nargs="?", default=".", type=Path)
    parser.add_argument(
        "--threshold",
        type=_threshold_arg,
        default=DEFAULT_HASH_THRESHOLD,
        help="Max Hamming distance (0-64) to consider two images duplicates. Lower = stricter. Default: %(default)s",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Folder to move non-kept duplicates into (default: <directory>/_duplicates)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Don't move any files, just show what would happen.")
    args = parser.parse_args()

    directory = args.directory
    if not directory.exists():
        print(f"Error: directory '{directory}' does not exist.", file=sys.stderr)
        sys.exit(1)
    if not directory.is_dir():
        print(f"Error: '{directory}' is not a directory.", file=sys.stderr)
        sys.exit(1)
    directory = directory.resolve()
    dest_dir = (args.dest or (directory / "_duplicates")).resolve()

    print(f"Scanning {directory} ...")
    groups = build_groups(directory, args.threshold)
    if not groups:
        print("No potential duplicate groups found.")
        return

    print(f"Found {len(groups)} potential duplicate group(s). Launching review UI...")
    app = DuplicateReviewApp(groups, dest_dir, args.dry_run)
    app.run()

    confirmed = sum(1 for g in groups if g.status == "confirmed")
    skipped = sum(1 for g in groups if g.status == "skipped")
    moved_total = sum(len(m["moved"]) for m in app.manifest)
    print(
        f"\nDone. {confirmed} group(s) confirmed, {skipped} skipped, {moved_total} file(s) "
        f"{'would be moved' if args.dry_run else 'moved'} to {dest_dir}"
    )


if __name__ == "__main__":
    main()
