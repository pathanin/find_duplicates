"""
find_duplicates.py

Scans the current directory (top level only, or subdirectories too with
--recursive) for images that look like the same photo saved at different
sizes/qualities, groups them with a perceptual hash, scores each candidate
using compare_image_quality.analyze(), and lets you confirm which one to
keep in a Textual TUI (or automatically, with --auto). Non-kept files are
moved to ./_duplicates/, never deleted -- restoring one is a manual move
back out of that folder (with --recursive, a moved file's subdirectory
structure is mirrored under _duplicates/, so the original relative location
is still recoverable from the path alone).

The scan/group/score/move pipeline lives in duplicates_core.py, shared with
the browser-based front end (find_duplicates-web.py) -- this module holds
only the Textual TUI and the CLI entry point.

Usage:
    python find_duplicates.py [directory] [--threshold N] [--dest DIR] [--recursive] [--auto] [--dry-run]

Requires:
    pip install opencv-python-headless numpy textual textual-image pillow
"""

import argparse
import functools
import shutil  # noqa: F401 -- unused directly; tests patch fd.shutil.move (see duplicates_core.unapply)
import subprocess
import sys
from pathlib import Path

from rich.markup import escape as rich_escape
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import DataTable, Footer, Header, Label, ListItem, ListView, Static
from textual_image.widget import Image as PreviewImage

from duplicates_core import (
    CACHE_FILENAME,
    CLOSE_CALL_MARGIN,
    DEFAULT_HASH_THRESHOLD,
    Group,
    HASH_CACHE_FILENAME,
    IMAGE_EXTS,
    METRIC_DESCRIPTIONS,
    METRIC_ROWS,
    METRIC_WEIGHTS,
    MIN_REDUCED_DECODE_SIDE,
    PREVIEW_MAX_SIDE,
    ProcessPoolExecutor,
    THUMBNAIL_FAILURE_COLOR,
    ThreadPoolExecutor,
    _compute_dest,
    _hash_one,
    analyze_paths,
    apply_group,
    apply_pick,
    auto_apply_groups,
    build_groups,
    cached_hash,
    cached_result,
    find_images,
    group_duplicates,
    humansize,
    load_cache,
    load_hash_cache,
    load_hash_gray,
    make_thumbnail,
    phash,
    pick_needs_reapply,
    save_cache,
    save_hash_cache,
    score_group,
    store_hash,
    store_result,
    unapply,
)

# ---------------------------------------------------------------------------
# TUI
# ---------------------------------------------------------------------------

# Content width (no padding) of the metrics table's "Metric" column -- fixed
# for a given METRIC_ROWS, since that first column's content never varies
# per-group. Used both to width that column explicitly (DataTable.add_column
# ..., width=N is content width; render width adds 2*cell_padding, default
# cell_padding=1 per side) and to size a matching blank spacer at the start
# of the image-preview row, so the two independently-laid-out widgets'
# column boundaries line up -- see refresh_detail/_sync_metric_column_widths.
METRIC_LABEL_COL_WIDTH = max(len("Metric"), max(len(label) for label, _ in METRIC_ROWS))


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


class _PreviewBox(Vertical):
    """A single image's preview box in #images-row. Only exists (rather than
    a plain Vertical) to catch its own resize -- a terminal resize changes
    this box's actual rendered width ('1fr' of the row), and by the time
    *this* widget's own on_resize fires, self.size already reflects the new
    value (unlike DuplicateReviewApp.on_resize, which fires before the row's
    children have been re-arranged to the new terminal size -- reading
    box.size there is still stale). See _sync_metric_column_widths."""

    def on_resize(self, event) -> None:
        app = self.app
        app.call_after_refresh(app._sync_metric_column_widths, app.active_index)


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

    def __init__(
        self,
        groups: list[Group],
        dest_dir: Path,
        dry_run: bool,
        recursive: bool = False,
        scan_root: Path | None = None,
    ):
        super().__init__()
        self.groups = groups
        self.dest_dir = dest_dir
        self.dry_run = dry_run
        self.recursive = recursive
        self.scan_root = scan_root  # only used (for relative-path display/dest) when recursive
        # In-memory only -- tracks moves within this session so a re-pick or
        # un-confirm can find what to reverse (see _apply/_unapply). Not
        # persisted to disk: moved files are never deleted, so restoring one
        # after the app exits is just a manual move back out of dest_dir.
        self.manifest: list[dict] = []
        self.active_index = 0

    def _display_path(self, path: Path) -> str:
        """Filename alone is ambiguous under --recursive (two subdirectories
        can each hold an IMG_1234.jpg), so show the path relative to
        scan_root there instead."""
        if self.recursive and self.scan_root is not None:
            return str(path.relative_to(self.scan_root))
        return path.name

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
        return f"{tag}[{idx + 1}] {rich_escape(self._display_path(group.paths[idx]))}"

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
        # A blank leading spacer sized to match the metrics table's "Metric"
        # column, so preview box [idx] lines up under metrics column [idx]
        # below once _sync_metric_column_widths sets each image column's
        # width to match its box's actual rendered width (only known after
        # layout -- see that method).
        spacer = Static(id="images-spacer")
        spacer.styles.width = METRIC_LABEL_COL_WIDTH + 2
        boxes = [spacer]
        for idx, thumb in enumerate(group.thumbnails):
            classes = self._pick_box_classes(group, idx)
            label_text = self._pick_label_text(group, idx)
            image = PreviewImage(thumb, classes="preview-image")
            boxes.append(_PreviewBox(Label(label_text, classes="preview-label"), image, classes=classes))
        await row.mount(*boxes)

        self._build_metrics_table(group, image_col_widths=None)
        self.call_after_refresh(self._sync_metric_column_widths, i)

        self.query_one("#status", Static).update(self._status_text())

    def _build_metrics_table(self, group: Group, image_col_widths: list[int] | None) -> None:
        """*image_col_widths*, when given, must have one entry per
        group.paths -- each metric column's explicit width, in cells
        (content width, i.e. not including DataTable's own cell padding).
        None leaves every image column auto-width (fits its own content),
        which is what a fresh refresh_detail renders immediately; the actual
        pixel-aligned widths come from a follow-up call once the preview
        boxes' real rendered sizes are known (see _sync_metric_column_widths)."""
        table = self.query_one(DataTable)
        table.clear(columns=True)
        table.add_column("Metric", width=METRIC_LABEL_COL_WIDTH)
        for idx in range(len(group.paths)):
            header = f"[{idx + 1}]" + (" ★" if idx == group.suggested_idx else "")
            width = image_col_widths[idx] if image_col_widths is not None else None
            table.add_column(header, width=width)
        for label, fn in METRIC_ROWS:
            table.add_row(label, *[fn(r) for r in group.results])

    def _sync_metric_column_widths(self, i: int) -> None:
        """Runs once the preview boxes mounted in refresh_detail have
        actually been laid out (scheduled via call_after_refresh, since a
        box's real on-screen width -- it's `1fr` of #images-row, clamped to
        min-width, possibly overflow-scrolled -- isn't known any earlier).
        Rebuilds the metrics table with each image column's width pinned to
        match its corresponding box's rendered width, so the two
        independently-laid-out widgets' column boundaries actually line up.
        """
        if i != self.active_index:
            return  # stale callback from a group navigated away from before layout settled
        group = self.groups[i]
        boxes = [c for c in self.query_one("#images-row", Horizontal).children if c.id != "images-spacer"]
        if len(boxes) != len(group.paths):
            return  # stale callback racing a newer refresh_detail for a different-sized group
        # outer_size, not size: .size is a widget's *content* area (excludes
        # its own border+padding), but what needs to match the table column
        # is the box's total on-screen footprint -- its border(2)+padding(2)
        # are exactly the overhead .size already strips out, so using .size
        # here made every column 4 cells narrower than its box, an error
        # that compounds column-over-column since each column's x-offset is
        # the sum of every render width before it (caught visually: columns
        # drifted further left of their box the further right they were).
        # -2 for the -2*cell_padding: DataTable's own cell_padding (default 1
        # cell each side) is added on top of the width passed to add_column,
        # which is content width -- see METRIC_LABEL_COL_WIDTH's comment.
        widths = [max(box.outer_size.width - 2, 1) for box in boxes]
        self._build_metrics_table(group, image_col_widths=widths)

    async def _update_pick_ui(self, old_pick: int, new_pick: int) -> None:
        """Lightweight counterpart to refresh_detail for moving the pick
        within the same group: only the old/new picked boxes' CSS class and
        label text can have changed (nothing in METRIC_ROWS or the table
        headers depends on current_pick), so this skips rebuilding the
        PreviewImage widgets (avoids re-encoding/re-transmitting every
        terminal image) and the DataTable entirely."""
        group = self.groups[self.active_index]
        # [0] is the alignment spacer (see refresh_detail), not a preview box.
        boxes = [c for c in self.query_one("#images-row", Horizontal).children if c.id != "images-spacer"]
        for idx in {old_pick, new_pick}:
            if not (0 <= idx < len(boxes)):
                continue
            box = boxes[idx]
            box.set_class(idx == group.current_pick, "picked")
            box.query_one(Label).update(self._pick_label_text(group, idx))
        # Re-picking on an already-confirmed group (see action_pick /
        # action_pick_relative -- both allow this) leaves the sidebar's
        # "-> [N]" arrow tracking current_pick, same as it always has; the
        # status line below is what actually flags that a re-confirm is
        # needed to make the new pick take effect on disk.
        await self._relabel(self.active_index)
        self.query_one("#status", Static).update(self._status_text())

    def _pending_pick_text(self, group: Group) -> tuple[str, str]:
        """(action, line3) describing a not-yet-applied current_pick --
        shared by a genuinely "pending" group and a "confirmed" group whose
        pick has since diverged from what's on disk (_pick_needs_reapply)."""
        n_removed = len(group.paths) - 1
        plural = "s" if n_removed != 1 else ""
        action = (
            f"keep [{group.current_pick + 1}] "
            f"{rich_escape(self._display_path(group.paths[group.current_pick]))}"
        )
        if n_removed > 0:
            action += f", move {n_removed} other file{plural}"
        if group.current_pick != group.suggested_idx:
            line3 = (
                f"your pick [{group.current_pick + 1}]  ·  "
                f"★ suggested [{group.suggested_idx + 1}] "
                f"{rich_escape(self._display_path(group.paths[group.suggested_idx]))}"
            )
        else:
            line3 = ""
        return action, line3

    def _status_text(self) -> str:
        confirmed = sum(1 for g in self.groups if g.status == "confirmed")
        skipped = sum(1 for g in self.groups if g.status == "skipped")
        pending = len(self.groups) - confirmed - skipped
        mode = "  [DRY RUN]" if self.dry_run else ""

        group = self.groups[self.active_index]
        if group.status == "pending":
            action, line3 = self._pending_pick_text(group)
        elif group.status == "confirmed" and self._pick_needs_reapply(self.active_index, group):
            action, line3 = self._pending_pick_text(group)
            action = "change " + action + "  (press c/Enter to confirm)"
        else:
            action = f"already {group.status}"
            line3 = ""

        return (
            f"Groups: {len(self.groups)}  confirmed={confirmed}  skipped={skipped}  pending={pending}{mode}\n"
            f"{action}\n{line3}"
        )

    def _pick_needs_reapply(self, i: int, group: Group) -> bool:
        """True for a confirmed group whose current_pick has since diverged
        from what's actually on disk (self.manifest's record of what got
        applied) -- i.e. action_confirm would really re-move files if
        pressed again, rather than being a no-op. Re-picking on an already
        confirmed group (action_pick/action_pick_relative below) only stages
        current_pick; nothing moves until the user explicitly re-confirms.
        Thin delegation to duplicates_core.pick_needs_reapply, shared with
        the web front end."""
        return pick_needs_reapply(self.manifest, i, group)

    async def action_pick(self, n: int) -> None:
        group = self.groups[self.active_index]
        idx = n - 1
        if 0 <= idx < len(group.paths) and idx != group.current_pick:
            old_pick = group.current_pick
            group.current_pick = idx
            await self._update_pick_ui(old_pick, idx)

    async def action_pick_relative(self, delta: int) -> None:
        group = self.groups[self.active_index]
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
            # Only re-move files if the pick actually diverged from what's
            # applied (_pick_needs_reapply) -- but even when it didn't,
            # confirm still means "done with this group," so fall through to
            # _relabel/_advance below rather than returning early. Pressing
            # confirm on a group you only looked at again, without changing
            # anything, should move on to the next pending group, not sit
            # there as if the keypress had no effect.
            if self._pick_needs_reapply(i, group):
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
        manifest. Does NOT change the group status — the caller decides.
        Thin delegation to duplicates_core.unapply, shared with the web
        front end -- see that function for the invariants this preserves."""
        unapply(self.manifest, i)

    def _apply(self, i: int, keep_idx: int) -> None:
        """Thin delegation to duplicates_core.apply_pick, shared with the
        web front end. If apply_group raises inside it, the group stays
        "pending" so the user can see it's in an inconsistent state."""
        apply_pick(
            self.groups[i], i, keep_idx, self.dest_dir, self.dry_run, self.manifest,
            recursive=self.recursive, scan_root=self.scan_root,
        )

    def _dest_for(self, path: Path) -> Path:
        return _compute_dest(
            path, self.dest_dir, self.dry_run, recursive=self.recursive, scan_root=self.scan_root
        )

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
    parser.add_argument(
        "--recursive", "-r", action="store_true", help="Scan subdirectories too, not just the top level."
    )
    parser.add_argument("--dry-run", action="store_true", help="Don't move any files, just show what would happen.")
    parser.add_argument(
        "--auto",
        "--yes",
        action="store_true",
        help="Non-interactive: skip the review UI and keep each group's suggested (top-scored) file automatically.",
    )
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
    groups = build_groups(directory, args.threshold, recursive=args.recursive, dest_dir=dest_dir)
    if not groups:
        print("No potential duplicate groups found.")
        return

    if args.auto:
        print(f"Found {len(groups)} potential duplicate group(s). Auto-applying suggested picks...")
        summary = auto_apply_groups(groups, dest_dir, args.dry_run, recursive=args.recursive, scan_root=directory)
        reclaimed = "(dry run)" if args.dry_run else humansize(summary["bytes_reclaimed"])
        print(
            f"\nDone. {summary['confirmed']} group(s) confirmed, {summary['files_moved']} file(s) "
            f"{'would be moved' if args.dry_run else 'moved'} to {dest_dir}. Reclaimed: {reclaimed}"
        )
        if summary["failed"]:
            print(f"\n{summary['failed']} group(s) FAILED and were left pending:", file=sys.stderr)
            for f in summary["failures"]:
                print(
                    f"  group {f['group']}: {f['error']} "
                    f"({f['files_moved']} file(s)/{humansize(f['bytes_moved'])} moved before the failure)",
                    file=sys.stderr,
                )
            sys.exit(1)
        return

    print(f"Found {len(groups)} potential duplicate group(s). Launching review UI...")
    app = DuplicateReviewApp(groups, dest_dir, args.dry_run, recursive=args.recursive, scan_root=directory)
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
