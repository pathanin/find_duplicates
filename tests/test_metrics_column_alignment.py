"""Regression test: the metrics DataTable's per-image columns must visually
line up with the preview boxes above them.

Before this fix, the DataTable auto-sized each image column to fit only its
own (narrow) numeric content, while the preview boxes above stretched to
fill the full available width -- two independently-laid-out widgets using
completely different sizing rules, so column [N] rarely landed under box
[N]. refresh_detail now mounts a blank spacer at the start of #images-row
sized to match the table's "Metric" column, and _sync_metric_column_widths
(scheduled via call_after_refresh, once real box sizes are known) pins each
image column's width to match its corresponding box's actual rendered
width.

Run: python3 test_metrics_column_alignment.py
"""

import asyncio
import sys
import tempfile
from pathlib import Path

from PIL import Image as PILImage
from textual.widgets import DataTable
from textual_image.widget import HalfcellImage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import find_duplicates as fd


def fake_result(quality: float = 0.5) -> dict:
    return {
        "dimensions": (100, 100),
        "file_size": 12345,
        "sharpness_normalized": 100.0,
        "effective_resolution_fraction": 0.9,
        "effective_resolution_px_equiv": 9000.0,
        "noise_sigma": 1.0,
        "blockiness": 0.1,
        "brisque": None,
        "niqe": None,
        "quality_score": quality,
    }


def make_group(n: int) -> fd.Group:
    thumbs = [PILImage.new("RGB", (100, 100)) for _ in range(n)]
    return fd.Group(
        paths=[Path(f"synthetic_{i}.png") for i in range(n)],
        results=[fake_result() for _ in range(n)],
        thumbnails=thumbs,
        suggested_idx=0,
        current_pick=0,
        is_close_call=False,
    )


def new_app(n: int) -> fd.DuplicateReviewApp:
    scratch = Path(tempfile.mkdtemp())
    return fd.DuplicateReviewApp([make_group(n)], scratch / "_dup", dry_run=True)


def preview_box_widths(app: fd.DuplicateReviewApp) -> list[int]:
    # outer_size, not size: .size is a widget's *content* area (excludes its
    # own border+padding); outer_size is the box's actual on-screen
    # footprint, which is what needs to match the table column below it.
    # Using .size here previously made this whole test file vacuous -- it
    # compared against the same (wrong) metric the implementation used
    # internally, so a real drift between the two widgets went undetected.
    boxes = [c for c in app.query_one("#images-row").children if c.id != "images-spacer"]
    return [b.outer_size.width for b in boxes]


def table_column_render_widths(app: fd.DuplicateReviewApp) -> list[int]:
    table = app.query_one(DataTable)
    return [c.get_render_width(table) for c in table.ordered_columns]


def preview_box_screen_regions(app: fd.DuplicateReviewApp) -> list[tuple[int, int]]:
    """(x, width) in absolute screen coordinates for every preview box."""
    boxes = [c for c in app.query_one("#images-row").children if c.id != "images-spacer"]
    return [(b.region.x, b.region.width) for b in boxes]


def table_column_screen_regions(app: fd.DuplicateReviewApp) -> list[tuple[int, int]]:
    """(x, width) in absolute screen coordinates for every image column
    (index 0, the "Metric" column, excluded) -- computed the same way
    DataTable itself positions columns on screen (_get_column_region),
    not re-derived from column widths alone. This is what actually proves
    on-screen alignment, as opposed to merely equal widths: two rows of
    boxes with matching widths can still drift apart if their starting
    x-offsets diverge, which is exactly the bug this test exists to catch
    (see preview_box_widths' docstring)."""
    table = app.query_one(DataTable)
    regions = []
    for i in range(1, len(table.ordered_columns)):
        col_region = table._get_column_region(i)
        regions.append((table.region.x + col_region.x, col_region.width))
    return regions


async def test_image_columns_match_preview_box_widths() -> None:
    app = new_app(n=3)
    async with app.run_test(size=(170, 50)) as pilot:
        await pilot.pause()
        await asyncio.sleep(0.1)  # let the call_after_refresh sync callback run
        await pilot.pause()

        box_widths = preview_box_widths(app)
        col_widths = table_column_render_widths(app)

        assert len(col_widths) == len(box_widths) + 1, "expected one Metric column plus one per preview box"
        assert col_widths[1:] == box_widths, (
            f"image column widths {col_widths[1:]} must match preview box widths {box_widths} "
            "for the two rows to visually line up"
        )
    print(f"  ok  image column widths {col_widths[1:]} match preview box widths {box_widths}")


async def test_image_columns_align_on_screen_with_preview_boxes() -> None:
    """The definitive alignment check: each column's absolute screen (x,
    width) must exactly match its corresponding box's, not just have equal
    width. Widths alone can match while positions still drift apart --
    that's precisely the bug that shipped and was reported as "still
    slightly not align[ed]": each column landed 4 cells narrower than its
    box (border+padding accounted for on the box side but not matched on
    the column side), an error that compounds column over column since
    each column's x-offset is the sum of every render width before it."""
    app = new_app(n=4)
    async with app.run_test(size=(170, 50)) as pilot:
        await pilot.pause()
        await asyncio.sleep(0.1)
        await pilot.pause()

        box_regions = preview_box_screen_regions(app)
        col_regions = table_column_screen_regions(app)
        assert col_regions == box_regions, (
            f"column regions {col_regions} must exactly match box regions {box_regions} -- "
            "any mismatch, even growing gradually left-to-right, means the columns don't "
            "actually line up under their images"
        )
    print(f"  ok  every column's on-screen (x, width) exactly matches its box: {box_regions}")


async def test_metric_column_matches_spacer() -> None:
    """The alignment spacer at the start of #images-row and the table's own
    "Metric" column must agree on width, or column [1] wouldn't start under
    box [1] even with the image columns individually correctly sized."""
    app = new_app(n=2)
    async with app.run_test(size=(170, 50)) as pilot:
        await pilot.pause()
        spacer = next(c for c in app.query_one("#images-row").children if c.id == "images-spacer")
        col_widths = table_column_render_widths(app)
        assert spacer.styles.width.value == col_widths[0], (
            f"spacer width {spacer.styles.width.value} must equal the Metric column's "
            f"render width {col_widths[0]}"
        )
    print("  ok  the leading spacer's width matches the Metric column's render width")


async def test_resize_resyncs_column_widths() -> None:
    """A terminal resize changes each box's rendered width ('1fr'); the
    table's columns must be recomputed to match, not left stale from the
    original size."""
    app = new_app(n=2)
    async with app.run_test(size=(170, 50)) as pilot:
        await pilot.pause()
        await asyncio.sleep(0.1)
        await pilot.pause()
        narrow_widths = table_column_render_widths(app)[1:]

        await pilot.resize_terminal(240, 50)
        await pilot.pause()
        await asyncio.sleep(0.1)
        await pilot.pause()
        wide_box_widths = preview_box_widths(app)
        wide_col_widths = table_column_render_widths(app)[1:]

        assert wide_col_widths != narrow_widths, "columns should have grown after the terminal widened"
        assert wide_col_widths == wide_box_widths, "columns must still match the (now wider) boxes after a resize"
    print(f"  ok  resizing the terminal re-syncs column widths ({narrow_widths} -> {wide_col_widths})")


async def main() -> None:
    fd.PreviewImage = HalfcellImage  # deterministic headless renderer, no real terminal needed
    for test in (
        test_image_columns_match_preview_box_widths,
        test_image_columns_align_on_screen_with_preview_boxes,
        test_metric_column_matches_spacer,
        test_resize_resyncs_column_widths,
    ):
        print(f"{test.__name__}:")
        await test()
    print("all metrics-column-alignment tests passed")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except AssertionError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)
