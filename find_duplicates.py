"""
find_duplicates.py

Scans the current directory (top level only) for images that look like the
same photo saved at different sizes/qualities, groups them with a perceptual
hash, scores each candidate using compare_image_quality.analyze(), and lets
you confirm which one to keep in a Textual TUI. Non-kept files are moved to
./_duplicates/ (nothing is deleted) and every decision is logged to
decisions.json.

Usage:
    python find_duplicates.py [directory] [--threshold N] [--dest DIR] [--dry-run]

Requires:
    pip install opencv-python-headless numpy textual textual-image pillow
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PILImage
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Footer, Header, Label, ListItem, ListView, Static
from textual_image.widget import Image as PreviewImage

from compare_image_quality import analyze

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
DEFAULT_HASH_THRESHOLD = 10  # max Hamming distance out of 64 bits to call two images duplicates
PREVIEW_MAX_SIDE = 800
CLOSE_CALL_MARGIN = 0.08  # quality_score gap below which we flag "close call"
CACHE_FILENAME = ".find_duplicates_cache.json"
# phash resizes to 32x32; a reduced-scale decode smaller than this on either side
# would upsample instead of downsample there, drifting the hash. 64 gives margin,
# and images this small are cheap to fully decode anyway.
MIN_REDUCED_DECODE_SIDE = 64

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


# ---------------------------------------------------------------------------
# Scanning + perceptual hashing + grouping
# ---------------------------------------------------------------------------

def find_images(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def load_hash_gray(p: Path) -> np.ndarray | None:
    """Grayscale decode for perceptual hashing. Uses a 1/8-scale DCT decode
    for speed (skips full-resolution JPEG decode just to shrink it to 32x32
    afterwards); falls back to a full decode when the image is small enough
    that the reduced decode would land below what the hash needs."""
    img = cv2.imread(str(p), cv2.IMREAD_REDUCED_GRAYSCALE_8)
    if img is None or min(img.shape) < MIN_REDUCED_DECODE_SIDE:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return img


def phash(gray: np.ndarray) -> int:
    """Classic 64-bit DCT perceptual hash: resize small, keep low frequencies,
    threshold against their mean. Robust to resizing/recompression, which is
    exactly the kind of "same photo, different export" duplicate we're after."""
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
    return bin(a ^ b).count("1")


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def group_duplicates(paths: list[Path], threshold: int) -> list[list[Path]]:
    hashes: list[int | None] = []
    for p in paths:
        img = load_hash_gray(p)
        hashes.append(phash(img) if img is not None else None)

    uf = UnionFind(len(paths))
    for i in range(len(paths)):
        if hashes[i] is None:
            continue
        for j in range(i + 1, len(paths)):
            if hashes[j] is None:
                continue
            if hamming(hashes[i], hashes[j]) <= threshold:
                uf.union(i, j)

    clusters: dict[int, list[Path]] = {}
    for i, p in enumerate(paths):
        if hashes[i] is None:
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
    keys = [k for k in METRIC_WEIGHTS if all(r.get(k) is not None for r in results)]
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


def make_thumbnail(path: Path) -> PILImage.Image:
    img = PILImage.open(path)
    img = img.convert("RGB")
    img.thumbnail((PREVIEW_MAX_SIDE, PREVIEW_MAX_SIDE))
    return img


@dataclass
class Group:
    paths: list[Path]
    results: list[dict]
    thumbnails: list[PILImage.Image]
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
    with open(directory / CACHE_FILENAME, "w") as f:
        json.dump(cache, f)


def cached_result(cache: dict, p: Path, st: os.stat_result) -> dict | None:
    entry = cache.get(str(p.resolve()))
    if entry is None or entry.get("mtime") != st.st_mtime_ns or entry.get("size") != st.st_size:
        return None
    result = dict(entry["result"])
    result["dimensions"] = tuple(result["dimensions"])
    return result


def store_result(cache: dict, p: Path, st: os.stat_result, result: dict) -> None:
    cache[str(p.resolve())] = {"mtime": st.st_mtime_ns, "size": st.st_size, "result": result}


def analyze_paths(paths: list[Path], cache: dict) -> dict[Path, dict]:
    """analyze() every path, reusing `cache` for files whose (mtime, size)
    haven't changed and running the rest through a process pool (analyze()
    is CPU-bound and independent per file)."""
    results: dict[Path, dict] = {}
    stats = {p: p.stat() for p in paths}
    to_compute = []
    for p in paths:
        hit = cached_result(cache, p, stats[p])
        if hit is not None:
            results[p] = hit
        else:
            to_compute.append(p)

    if to_compute:
        # Deliberately not forcing a "fork" context here: by this point
        # group_duplicates() has already done real cv2 decode work in this
        # process, and forking after cv2/numpy have used internal threads
        # reliably crashes the pool (BrokenProcessPool) on macOS -- confirmed
        # empirically. Default spawn pays a one-time ~0.3s re-import tax per
        # pool but is actually safe.
        with ProcessPoolExecutor() as executor:
            for p, r in zip(to_compute, executor.map(analyze, [str(p) for p in to_compute])):
                store_result(cache, p, stats[p], r)
                results[p] = r

    for p in paths:
        results[p]["file_size"] = stats[p].st_size
    return results


def build_groups(directory: Path, threshold: int) -> list[Group]:
    paths = find_images(directory)
    raw_groups = group_duplicates(paths, threshold)

    cache = load_cache(directory)
    analyzed = analyze_paths([p for members in raw_groups for p in members], cache)
    save_cache(directory, cache)

    groups = []
    for members in raw_groups:
        results = [analyzed[p] for p in members]
        score_group(results)

        order = sorted(range(len(results)), key=lambda i: -results[i]["quality_score"])
        suggested_idx = order[0]
        close_call = len(order) > 1 and (
            results[order[0]]["quality_score"] - results[order[1]]["quality_score"] < CLOSE_CALL_MARGIN
        )
        thumbnails = [make_thumbnail(p) for p in members]

        groups.append(
            Group(
                paths=members,
                results=results,
                thumbnails=thumbnails,
                suggested_idx=suggested_idx,
                current_pick=suggested_idx,
                is_close_call=close_call,
            )
        )
    return groups


# ---------------------------------------------------------------------------
# TUI
# ---------------------------------------------------------------------------

METRIC_ROWS = [
    ("Dimensions", lambda r: f"{r['dimensions'][0]}x{r['dimensions'][1]}"),
    ("File size", lambda r: humansize(r["file_size"])),
    ("Sharpness (norm.)", lambda r: f"{r['sharpness_normalized']:.1f}"),
    ("Eff. res. fraction", lambda r: f"{r['effective_resolution_fraction']:.3f}"),
    ("Eff. res. (px equiv)", lambda r: f"{r['effective_resolution_px_equiv']:.0f}"),
    ("Noise sigma", lambda r: f"{r['noise_sigma']:.3f}"),
    ("Blockiness", lambda r: f"{r['blockiness']:.3f}"),
    ("BRISQUE", lambda r: f"{r['brisque']:.2f}" if r.get("brisque") is not None else "n/a"),
    ("NIQE", lambda r: f"{r['niqe']:.2f}" if r.get("niqe") is not None else "n/a"),
    ("Quality score", lambda r: f"{r['quality_score']:.3f}"),
]


class DuplicateReviewApp(App):
    TITLE = "Duplicate image review"

    CSS = """
    #body { height: 1fr; }
    #sidebar { width: 40; border-right: solid $accent; }
    #detail { width: 1fr; height: 1fr; }
    #images-row { height: 24; overflow-x: auto; }
    .preview-box {
        width: 1fr; min-width: 30; height: 22; border: round $panel; padding: 0 1;
        align: center middle;
    }
    .preview-box.suggested { border: round $accent; }
    .preview-box.picked { border: heavy $success; }
    .preview-image { width: auto; height: auto; }
    #metrics-table { height: 1fr; }
    #status { height: 2; background: $panel; content-align: left top; padding: 0 1; }
    """

    BINDINGS = [
        Binding("left", "pick_relative(-1)", "Prev pick"),
        Binding("right", "pick_relative(1)", "Next pick"),
        Binding("c", "confirm", "Confirm keep"),
        Binding("s", "skip", "Skip group"),
        Binding("o", "open_fullres", "Open full-res"),
        Binding("1", "pick(1)", "Pick 1", show=False),
        Binding("2", "pick(2)", "Pick 2", show=False),
        Binding("3", "pick(3)", "Pick 3", show=False),
        Binding("4", "pick(4)", "Pick 4", show=False),
        Binding("5", "pick(5)", "Pick 5", show=False),
        Binding("6", "pick(6)", "Pick 6", show=False),
        Binding("7", "pick(7)", "Pick 7", show=False),
        Binding("8", "pick(8)", "Pick 8", show=False),
        Binding("9", "pick(9)", "Pick 9", show=False),
        Binding("q", "quit_and_apply", "Finish"),
    ]

    def __init__(self, groups: list[Group], dest_dir: Path, dry_run: bool, manifest_path: Path):
        super().__init__()
        self.groups = groups
        self.dest_dir = dest_dir
        self.dry_run = dry_run
        self.manifest_path = manifest_path
        self.manifest: list[dict] = []
        self.active_index = 0

    def simulate_key(self, key: str) -> None:
        """Textual's Footer renders key bindings as clickable buttons, whose
        click handler routes through this exact method. That turns "c Confirm
        keep" into a real button one stray click away from silently moving
        files -- e.g. a click meant to focus/scroll the terminal after the
        scan finishes. Confirm/skip mutate group state, so they must only
        fire from a deliberate keypress, never a footer click."""
        if key not in ("c", "s"):
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
        return f"{marker} Group {i + 1} ({len(g.paths)} files){close}"

    async def _relabel(self, i: int) -> None:
        item = self.query_one("#group-list", ListView).children[i]
        item.query_one(Label).update(self._group_label(i))

    async def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        if event.list_view.id == "group-list" and event.list_view.index is not None:
            self.active_index = event.list_view.index
            await self.refresh_detail(self.active_index)

    async def refresh_detail(self, i: int) -> None:
        group = self.groups[i]

        row = self.query_one("#images-row", Horizontal)
        await row.remove_children()
        boxes = []
        for idx, (path, thumb) in enumerate(zip(group.paths, group.thumbnails)):
            classes = "preview-box"
            tag = ""
            if idx == group.current_pick:
                classes += " picked"
                tag = "  ✔ WILL KEEP"
            elif idx == group.suggested_idx:
                tag = "  ★ suggested"
            if idx == group.suggested_idx:
                classes += " suggested"
            label_text = f"[{idx + 1}] {path.name}{tag}"
            image = PreviewImage(thumb, classes="preview-image")
            boxes.append(Vertical(Label(label_text), image, classes=classes))
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

    def _status_text(self) -> str:
        confirmed = sum(1 for g in self.groups if g.status == "confirmed")
        skipped = sum(1 for g in self.groups if g.status == "skipped")
        pending = len(self.groups) - confirmed - skipped
        mode = "  [DRY RUN]" if self.dry_run else ""

        group = self.groups[self.active_index]
        n_removed = len(group.paths) - 1
        if group.status == "pending":
            plural = "s" if n_removed != 1 else ""
            action = (
                f"c confirms: KEEP {group.paths[group.current_pick].name}"
                f", move {n_removed} other file{plural} to _duplicates"
            )
        else:
            action = f"group already {group.status}"

        return (
            f"Groups: {len(self.groups)}  confirmed={confirmed}  skipped={skipped}  pending={pending}{mode}\n"
            f"{action}   |   ↑↓ groups · ←→ / 1-9 pick keep · c confirm · s skip · o open full-res · q finish"
        )

    async def action_pick(self, n: int) -> None:
        group = self.groups[self.active_index]
        idx = n - 1
        if 0 <= idx < len(group.paths):
            group.current_pick = idx
            await self.refresh_detail(self.active_index)

    async def action_pick_relative(self, delta: int) -> None:
        group = self.groups[self.active_index]
        group.current_pick = (group.current_pick + delta) % len(group.paths)
        await self.refresh_detail(self.active_index)

    async def action_confirm(self) -> None:
        i = self.active_index
        if self.groups[i].status != "pending":
            return
        self._apply(i, self.groups[i].current_pick)
        await self._relabel(i)
        await self._advance()

    async def action_skip(self) -> None:
        i = self.active_index
        if self.groups[i].status != "pending":
            return
        self.groups[i].status = "skipped"
        await self._relabel(i)
        await self._advance()

    def action_open_fullres(self) -> None:
        group = self.groups[self.active_index]
        paths = [str(p) for p in group.paths]
        try:
            if sys.platform == "darwin":
                subprocess.run(["open", "-a", "Preview", *paths], check=False)
            elif sys.platform.startswith("linux"):
                subprocess.run(["xdg-open", paths[0]], check=False)
            else:
                self.notify("Full-resolution open isn't supported on this OS.", severity="warning")
        except FileNotFoundError:
            self.notify("Couldn't find an image viewer to open the file with.", severity="error")

    def _apply(self, i: int, keep_idx: int) -> None:
        group = self.groups[i]
        group.status = "confirmed"
        group.current_pick = keep_idx
        moved = []
        for idx, path in enumerate(group.paths):
            if idx == keep_idx:
                continue
            dest = self._dest_for(path)
            if not self.dry_run:
                shutil.move(str(path), str(dest))
            moved.append({"from": str(path), "to": str(dest)})
        self.manifest.append(
            {
                "group": i,
                "kept": str(group.paths[keep_idx]),
                "moved": moved,
                "dry_run": self.dry_run,
            }
        )
        self._write_manifest()

    def _dest_for(self, path: Path) -> Path:
        if not self.dry_run:
            self.dest_dir.mkdir(parents=True, exist_ok=True)
        dest = self.dest_dir / path.name
        n = 1
        while not self.dry_run and dest.exists():
            dest = self.dest_dir / f"{path.stem}_dup{n}{path.suffix}"
            n += 1
        return dest

    def _write_manifest(self) -> None:
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)

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

def main() -> None:
    parser = argparse.ArgumentParser(description="Find and review potential duplicate images by quality.")
    parser.add_argument("directory", nargs="?", default=".", type=Path)
    parser.add_argument(
        "--threshold",
        type=int,
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

    directory = args.directory.resolve()
    dest_dir = (args.dest or (directory / "_duplicates")).resolve()
    manifest_path = directory / "decisions.json"

    print(f"Scanning {directory} ...")
    groups = build_groups(directory, args.threshold)
    if not groups:
        print("No potential duplicate groups found.")
        return

    print(f"Found {len(groups)} potential duplicate group(s). Launching review UI...")
    app = DuplicateReviewApp(groups, dest_dir, args.dry_run, manifest_path)
    app.run()

    confirmed = sum(1 for g in groups if g.status == "confirmed")
    skipped = sum(1 for g in groups if g.status == "skipped")
    moved_total = sum(len(m["moved"]) for m in app.manifest)
    print(
        f"\nDone. {confirmed} group(s) confirmed, {skipped} skipped, {moved_total} file(s) "
        f"{'would be moved' if args.dry_run else 'moved'} to {dest_dir}"
    )
    print(f"Decisions logged to {manifest_path}")


if __name__ == "__main__":
    main()
