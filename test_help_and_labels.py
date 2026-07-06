"""Regression tests for the in-app help screen and metric-row labeling in
find_duplicates.py.

A raw number is meaningless without knowing which direction is "better", so
every METRIC_ROWS label must say so (or say it isn't scored at all, for
dimensions/file size). The '?' help screen expands on that with the actual
weights, and must never mutate group state just by opening/closing.

Run: python3 test_help_and_labels.py
"""

import asyncio
import sys
import tempfile
from pathlib import Path

from PIL import Image as PILImage
from textual.widgets import DataTable, Static
from textual_image.widget import HalfcellImage

import find_duplicates as fd


def fake_result() -> dict:
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
        "quality_score": 0.5,
    }


def make_group() -> fd.Group:
    thumbs = [PILImage.new("RGB", (100, 100)) for _ in range(2)]
    return fd.Group(
        paths=[Path(f"synthetic_{i}.png") for i in range(2)],
        results=[fake_result() for _ in range(2)],
        thumbnails=thumbs,
        suggested_idx=0,
        current_pick=0,
        is_close_call=False,
    )


def new_app() -> fd.DuplicateReviewApp:
    scratch = Path(tempfile.mkdtemp())
    return fd.DuplicateReviewApp(
        [make_group()], scratch / "_dup", dry_run=True, manifest_path=scratch / "decisions.json"
    )


def test_every_metric_row_states_its_direction() -> None:
    for label, _ in fd.METRIC_ROWS:
        has_direction = "better" in label
        has_not_scored = "not scored" in label
        assert has_direction or has_not_scored, (
            f"metric row {label!r} states neither a direction nor that it's unscored -- "
            "a bare number here is meaningless without one"
        )
    print("  ok  every METRIC_ROWS label states a direction or 'not scored'")


def test_help_body_covers_every_weighted_metric() -> None:
    body = fd._help_body()
    for name in fd.METRIC_WEIGHTS:
        assert name in body, f"help text is missing weighted metric {name!r}"
        assert fd.METRIC_DESCRIPTIONS[name] in body, f"help text is missing the description for {name!r}"
    print("  ok  help text covers every weighted metric and its description")


async def test_help_screen_opens_and_closes_without_side_effects() -> None:
    app = new_app()
    async with app.run_test(size=(170, 50)) as pilot:
        await pilot.pause()
        await pilot.press("question_mark")
        await pilot.pause()
        assert any(type(s).__name__ == "HelpScreen" for s in app.screen_stack), "'?' should push the help screen"
        shown = str(app.screen.query(Static).first().render())
        assert "QUALITY SCORE" in shown, "help screen didn't render the expected content"

        await pilot.press("escape")
        await pilot.pause()
        assert not any(type(s).__name__ == "HelpScreen" for s in app.screen_stack), "escape should close the help screen"
        assert app.groups[0].status == "pending", "opening/closing help must not touch group state"
    print("  ok  '?' opens help, escape closes it, group state untouched")


async def test_metric_labels_reach_the_table() -> None:
    app = new_app()
    async with app.run_test(size=(170, 50)) as pilot:
        await pilot.pause()
        table = app.query_one(DataTable)
        rendered_labels = [str(table.get_cell_at((r, 0))) for r in range(table.row_count)]
        assert rendered_labels == [label for label, _ in fd.METRIC_ROWS]
    print("  ok  annotated labels render in the metrics table")


async def main() -> None:
    fd.PreviewImage = HalfcellImage  # deterministic headless renderer, no real terminal needed
    test_every_metric_row_states_its_direction()
    test_help_body_covers_every_weighted_metric()
    for test in (test_help_screen_opens_and_closes_without_side_effects, test_metric_labels_reach_the_table):
        print(f"{test.__name__}:")
        await test()
    print("all help/label tests passed")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except AssertionError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)
