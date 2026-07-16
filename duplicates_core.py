"""
duplicates_core.py

Shared, UI-agnostic core of the duplicate-image tool: directory scanning,
perceptual hashing + grouping, quality scoring, thumbnailing, and the one
destructive path (moving non-kept files out of the way). No Textual/rich
dependency here -- this module must stay importable on a headless box that
only wants the web front end, or none at all.

find_duplicates.py (Textual TUI) and find_duplicates-web.py (browser UI) both
import from this module rather than duplicating any of it.
"""

import json
import math
import os
import shutil
import tempfile
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PILImage

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

# Every scored row is labeled with what a bigger/smaller number means, since
# a raw number is meaningless without knowing which direction is "better".
# Dimensions/file size carry no such label since they aren't part of the
# score at all -- that's explained once, in the '?' help screen, rather than
# on every row. UI-agnostic (plain strings + a dict-in/str-out lambda per
# row), so both the TUI's DataTable and the web UI's metrics JSON reuse it
# rather than keeping two label lists in sync by hand.
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


# ---------------------------------------------------------------------------
# Scanning + perceptual hashing + grouping
# ---------------------------------------------------------------------------

def find_images(directory: Path, recursive: bool = False, exclude_dir: Path | None = None) -> list[Path]:
    """Top-level-only scan by default. With *recursive*, walks subdirectories
    too -- *exclude_dir* (typically the move destination) is then required to
    keep a re-scan from picking up files already moved out by a prior run;
    it's meaningless (and ignored) in non-recursive mode since the default
    destination (<directory>/_duplicates) already sits below the top level
    iterdir() looks at."""
    if not recursive:
        return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    exclude_resolved = exclude_dir.resolve() if exclude_dir is not None else None
    found = []
    for p in directory.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
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


def group_duplicates(
    paths: list[Path], threshold: int, cache: dict, progress_callback=None
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
    front end) route progress to something other than stdout without
    touching the CLI/TUI's default behavior."""
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
                    if progress_callback is not None:
                        progress_callback("Hashing", done, total)
                    else:
                        _print_progress("Hashing", done, total, tty)
            if progress_callback is None and tty:
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
    """Move every non-kept file in *group* to *dest_dir*. Shared by the TUI's
    _apply and --auto mode. Files that are symlinks to the kept file's real
    target are left alone (moving the target would leave the kept path
    dangling). Records whatever was actually moved into a manifest entry
    from a `finally` -- even a failure partway through the loop (disk full,
    permission error) leaves an accurate, reversible record instead of
    silently losing track of files already relocated to disk. Does not set
    group.status; callers decide (kept "pending" on a raised exception is
    load-bearing for the TUI's retry path, see test_manifest_crash_safety.py).
    Appends the entry to *manifest* if given, and always returns it."""
    group.current_pick = keep_idx
    kept_path = group.paths[keep_idx]
    moved = []
    try:
        # is_symlink()/resolve() can themselves raise (e.g. EACCES on a
        # parent directory) -- kept inside the try so a failure here still
        # goes through the finally below instead of leaving *manifest*
        # without an entry for this group at all.
        kept_real = kept_path.resolve() if kept_path.is_symlink() else None
        for idx, path in enumerate(group.paths):
            if idx == keep_idx:
                continue
            if kept_real is not None and path.resolve() == kept_real:
                continue
            dest = _compute_dest(path, dest_dir, dry_run, recursive=recursive, scan_root=scan_root)
            if not dry_run:
                shutil.move(str(path), str(dest))
            moved.append({"from": str(path), "to": str(dest)})
    finally:
        entry = {"group": group_index, "kept": str(kept_path), "moved": moved, "dry_run": dry_run}
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
    """Apply every pending group's suggested (top-scored) pick, no TUI --
    used by --auto. Doesn't second-guess close calls: the suggested pick is
    applied exactly as it would default to in the TUI.

    A group whose apply_group() raises partway through is recorded as a
    failure (its status stays "pending", the same state the TUI leaves it in
    after a partial move failure) and the run continues with the remaining
    groups -- one bad group (disk full, permission error) must not kill an
    unattended run and leave earlier groups' successfully-moved files
    unreported.

    bytes_reclaimed sums each moved file's size from group.results[idx]
    ["file_size"] (already populated by analyze_paths) *before* the move --
    the source path is gone from disk by the time apply_group returns, so
    re-stat()'ing it afterward would silently always read 0.

    Returns {"confirmed": int, "failed": int, "files_moved": int,
    "bytes_reclaimed": int, "failures": [{"group", "error", "files_moved",
    "bytes_moved"}, ...]}."""
    confirmed = 0
    failed = 0
    files_moved = 0
    bytes_reclaimed = 0
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
        "failures": failures,
    }


# ---------------------------------------------------------------------------
# Interactive confirm/skip/re-pick primitives
# ---------------------------------------------------------------------------
# Shared by the TUI (DuplicateReviewApp._apply/_unapply/_pick_needs_reapply,
# thin delegations to these) and the web front end, so both keep the same
# manifest invariants (stays-pending-on-partial-failure, unapply-on-skip,
# clear-stale-entries-before-reapply, re-pick-after-confirm) rather than
# each reimplementing this -- the one genuinely destructive, stateful path
# in the app -- separately.

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
    """True for a confirmed group whose current_pick has since diverged from
    what's actually on disk (manifest's record of what got applied) -- i.e.
    re-applying would really re-move files, rather than being a no-op.
    Re-picking on an already confirmed group only stages current_pick;
    nothing moves until the caller explicitly re-applies."""
    entry = next((m for m in manifest if m["group"] == group_index), None)
    return entry is None or entry["kept"] != str(group.paths[group.current_pick])


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
                  precomputed_stats: dict[Path, os.stat_result] | None = None,
                  progress_callback=None) -> dict[Path, dict]:
    """analyze() every path, reusing `cache` for files whose (mtime, size)
    haven't changed and running the rest through a thread pool (analyze()'s
    cv2/numpy calls release the GIL -- see the comments at
    THREAD_POOL_WORKERS's definition).

    If *precomputed_stats* is provided, it must cover every path in *paths*
    and will be used instead of calling stat() again.

    *progress_callback*, if given, is called as progress_callback(label,
    done, total) as each uncached item completes, instead of the default
    TTY-aware print via _print_progress -- see group_duplicates's matching
    parameter."""
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
                    if progress_callback is not None:
                        progress_callback("Analyzing", done, total)
                    else:
                        _print_progress("Analyzing", done, total, tty)
        finally:
            cv2.setNumThreads(original_cv2_threads)
        if progress_callback is None and tty:
            print()

    for p in results:
        results[p]["file_size"] = stats[p].st_size
    return results


def build_groups(
    directory: Path, threshold: int, recursive: bool = False, dest_dir: Path | None = None,
    progress_callback=None,
) -> list[Group]:
    """*progress_callback*, if given, is passed straight through to
    group_duplicates/analyze_paths -- see their matching parameter. None
    (the default) preserves the CLI/TUI's existing TTY-aware stdout
    printing unchanged."""
    paths = find_images(directory, recursive=recursive, exclude_dir=dest_dir)

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
    raw_groups = group_duplicates(paths, threshold, hash_cache, progress_callback=progress_callback)
    if hash_cache != hash_cache_snapshot:
        save_hash_cache(directory, hash_cache)

    cache = load_cache(directory)
    cache_snapshot = dict(cache)
    # Compute stats for the grouped files once and pass to analyze_paths,
    # rather than letting it call stat() again on files already stat()'d
    # during the hash phase (the same Path objects are reused).
    grouped_paths = [p for members in raw_groups for p in members]
    grouped_stats = {p: p.stat() for p in grouped_paths}
    analyzed = analyze_paths(
        grouped_paths, cache, precomputed_stats=grouped_stats, progress_callback=progress_callback
    )
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
        # bool(...): quality_score can be a numpy float64 (propagated from
        # analyze()'s metrics), and `<` against one produces numpy.bool_,
        # not Python bool -- `and` returns its second operand as-is rather
        # than coercing it, so close_call would silently end up numpy.bool_
        # too. That's fine for the TUI's truthy "if g.is_close_call" check,
        # but numpy.bool_ (unlike numpy.float64) isn't a subclass of its
        # Python equivalent and isn't JSON-serializable -- the web front
        # end's /api/state was the first consumer to actually hit this.
        close_call = bool(
            len(order) > 1
            and results[order[0]]["quality_score"] - results[order[1]]["quality_score"] < CLOSE_CALL_MARGIN
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
