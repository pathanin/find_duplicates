"""Regression tests for re-picking a confirmed group directly (arrow/number
keys), then re-confirming to apply the change -- without first having to
skip and re-select from scratch.

action_confirm already had a branch for "confirmed, pick changed since
applied -> re-apply the new pick" (it compares self.manifest's recorded
"kept" path against group.current_pick), but action_pick/action_pick_relative
both refused to change current_pick at all once a group's status was
"confirmed" -- making that branch unreachable through the UI. The fix drops
that guard and adds a status-line hint ("press c/Enter to confirm") so it's
clear a re-pick on a confirmed group is staged, not yet applied.

Run: python3 test_repick_confirmed_group.py
"""

import asyncio
import sys
import tempfile
from pathlib import Path

from PIL import Image as PILImage
from textual_image.widget import HalfcellImage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
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


def make_real_group(directory: Path, n: int = 3) -> fd.Group:
    """Files that actually exist on disk -- dry_run=False, so re-picking and
    re-confirming really moves files, which is exactly the path this bug
    fix affects."""
    paths = []
    for i in range(n):
        p = directory / f"file_{i}.png"
        PILImage.new("RGB", (20, 20), (10 * i, 0, 0)).save(p)
        paths.append(p)
    thumbs = [PILImage.new("RGB", (20, 20)) for _ in range(n)]
    return fd.Group(
        paths=paths,
        results=[fake_result() for _ in range(n)],
        thumbnails=thumbs,
        suggested_idx=0,
        current_pick=0,
        is_close_call=False,
    )


def new_app(directory: Path) -> fd.DuplicateReviewApp:
    group = make_real_group(directory)
    return fd.DuplicateReviewApp([group], directory / "_duplicates", dry_run=False)


async def test_repick_then_reconfirm_moves_the_new_pick() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        app = new_app(directory)
        p0, p1, p2 = app.groups[0].paths
        async with app.run_test(size=(170, 50)) as pilot:
            await pilot.pause()
            await pilot.press("c")  # confirm keeping [1] (p0)
            await pilot.pause()
            group = app.groups[0]
            assert group.status == "confirmed" and group.current_pick == 0
            assert p0.exists() and not p1.exists() and not p2.exists()

            await pilot.press("right")  # re-pick [2] (p1) -- must work while confirmed
            await pilot.pause()
            assert group.current_pick == 1, "arrow keys must be able to change the pick on a confirmed group"
            assert group.status == "confirmed", "changing the pick alone must not move anything yet"
            assert p0.exists() and not p1.exists(), "no file should move until re-confirmed"

            await pilot.press("c")  # re-confirm with the new pick
            await pilot.pause()
            assert group.status == "confirmed"
            assert p1.exists(), "the newly-picked file must be restored to its original location"
            assert not p0.exists(), "the previously-kept file must now be moved out"
            assert not p2.exists()
        print("  ok  re-picking a confirmed group via arrow key, then re-confirming, swaps the kept file")


async def test_number_key_repick_works_on_confirmed_group() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        app = new_app(directory)
        async with app.run_test(size=(170, 50)) as pilot:
            await pilot.pause()
            await pilot.press("c")
            await pilot.pause()
            group = app.groups[0]
            assert group.status == "confirmed"

            await pilot.press("3")  # pick(3) -> index 2
            await pilot.pause()
            assert group.current_pick == 2, "number keys must also be able to re-pick a confirmed group"
        print("  ok  number-key re-pick works on a confirmed group")


async def test_reconfirm_with_unchanged_pick_is_a_noop() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        app = new_app(directory)
        async with app.run_test(size=(170, 50)) as pilot:
            await pilot.pause()
            await pilot.press("c")
            await pilot.pause()
            manifest_after_first_confirm = list(app.manifest)

            def exploding_apply(i, keep_idx):
                raise AssertionError("_apply must not be called when the pick hasn't changed")

            app._apply = exploding_apply
            await pilot.press("c")  # same pick again -- must be a no-op
            await pilot.pause()
            assert app.manifest == manifest_after_first_confirm, "manifest must be untouched by a no-op re-confirm"
        print("  ok  re-confirming with an unchanged pick does not re-apply (no-op, not vacuous)")


async def test_reconfirm_noop_still_advances_to_next_pending_group() -> None:
    """Regression: action_confirm used to `return` immediately for a
    confirmed group whose pick hadn't changed, skipping _relabel/_advance
    entirely -- so navigating back to an already-confirmed group and
    pressing confirm again (with nothing to actually change) silently did
    nothing, leaving you stuck on that group instead of moving on to the
    next pending one like a normal confirm does."""
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        dir0, dir1 = base / "g0", base / "g1"
        dir0.mkdir()
        dir1.mkdir()
        app = fd.DuplicateReviewApp(
            [make_real_group(dir0), make_real_group(dir1)], base / "_duplicates", dry_run=False
        )
        async with app.run_test(size=(170, 50)) as pilot:
            await pilot.pause()
            await pilot.press("c")  # confirm group 0 -> auto-advances to group 1, the only pending one
            await pilot.pause()
            assert app.groups[0].status == "confirmed"
            assert app.active_index == 1, "confirming group 0 should auto-advance to the only pending group"

            await pilot.press("up")  # navigate back to group 0
            await pilot.pause()
            assert app.active_index == 0

            await pilot.press("c")  # re-confirm with an unchanged pick -- a no-op re-apply
            await pilot.pause()
            assert app.active_index == 1, (
                "a no-op re-confirm (pick unchanged) must still advance to the next pending group, "
                "not leave the cursor sitting on the group that was just (redundantly) confirmed"
            )
        print("  ok  re-confirming an unchanged pick on a confirmed group still advances to the next pending group")


async def test_status_text_hints_at_unconfirmed_pick_change() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        app = new_app(directory)
        async with app.run_test(size=(170, 50)) as pilot:
            await pilot.pause()
            await pilot.press("c")
            await pilot.pause()
            assert "press c/Enter to confirm" not in app._status_text(), (
                "no hint expected right after confirming with no further pick change"
            )

            await pilot.press("right")
            await pilot.pause()
            status = app._status_text()
            assert "press c/Enter to confirm" in status, "status line must hint that a re-confirm is needed"
            assert "[2]" in status, "status line must name the newly staged pick"
        print("  ok  status line hints when a confirmed group's pick has diverged from what's applied")


async def test_sidebar_label_tracks_staged_pick_on_confirmed_group() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        app = new_app(directory)
        async with app.run_test(size=(170, 50)) as pilot:
            await pilot.pause()
            await pilot.press("c")
            await pilot.pause()
            assert "[1]" in app._group_label(0)

            await pilot.press("right")
            await pilot.pause()
            assert "[2]" in app._group_label(0), "sidebar arrow must follow the newly staged pick"
        print("  ok  sidebar label updates live as the pick changes on a confirmed group")


async def main() -> None:
    fd.PreviewImage = HalfcellImage  # deterministic headless renderer, no real terminal needed
    for test in (
        test_repick_then_reconfirm_moves_the_new_pick,
        test_number_key_repick_works_on_confirmed_group,
        test_reconfirm_with_unchanged_pick_is_a_noop,
        test_reconfirm_noop_still_advances_to_next_pending_group,
        test_status_text_hints_at_unconfirmed_pick_change,
        test_sidebar_label_tracks_staged_pick_on_confirmed_group,
    ):
        print(f"{test.__name__}:")
        await test()
    print("all repick-confirmed-group tests passed")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except AssertionError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)
