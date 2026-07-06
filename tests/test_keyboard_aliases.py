"""Regression tests for find_duplicates.py's layout-independent key aliases.

Letter keys (c/s/q/?) are remapped to different Unicode characters entirely
on non-Latin or alternate keyboard layouts, since the OS translates the
keystroke before the terminal ever sees it -- so a user on such a layout can
be unable to interact with the TUI at all without realizing why. Control
keys (Enter, Backspace, Delete, Escape, function keys) aren't part of that
character remapping, so each core action also has a control-key alias.

This also guards a real regression hit while adding those aliases: Enter
needed priority=True to win over ListView/DataTable's own built-in
enter-to-select_cursor binding, and doing that changed which key Textual's
Footer chooses to render as the clickable button for "Confirm keep" (it
picked "enter" over "c") -- silently reopening the footer-click-confirms
hole that test_footer_click_safety.py was written to close, since that test
only clicks whichever key the Footer actually shows. This file makes sure
*every* key in DuplicateReviewApp._DESTRUCTIVE_KEYS is blocked, not just
whichever one happens to be visible.

Run: python3 test_keyboard_aliases.py
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


async def test_enter_confirms_regardless_of_focus() -> None:
    for focus_widget_type in (None, DataTable):
        app = new_app()
        async with app.run_test(size=(170, 50)) as pilot:
            await pilot.pause()
            if focus_widget_type is not None:
                app.set_focus(app.query_one(focus_widget_type))
                await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert app.groups[0].status == "confirmed", (
                f"Enter should confirm even with {focus_widget_type} focused "
                "(both bind their own enter->select_cursor that would otherwise swallow it)"
            )
    print("  ok  Enter confirms regardless of which widget has focus")


async def test_delete_and_backspace_skip() -> None:
    for key in ("delete", "backspace"):
        app = new_app()
        async with app.run_test(size=(170, 50)) as pilot:
            await pilot.pause()
            await pilot.press(key)
            await pilot.pause()
            assert app.groups[0].status == "skipped", f"{key!r} should skip the active group"
    print("  ok  Delete and Backspace both skip")


async def test_escape_quits() -> None:
    app = new_app()
    async with app.run_test(size=(170, 50)) as pilot:
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert not app.is_running, "Escape should quit like 'q' does"
    print("  ok  Escape quits")


async def test_f1_opens_help() -> None:
    app = new_app()
    async with app.run_test(size=(170, 50)) as pilot:
        await pilot.pause()
        await pilot.press("f1")
        await pilot.pause()
        assert any(type(s).__name__ == "HelpScreen" for s in app.screen_stack), "F1 should open the help screen"
    print("  ok  F1 opens help")


async def test_enter_does_not_pierce_the_help_modal() -> None:
    """confirm's Enter alias is a priority binding (needed to beat
    ListView/DataTable's own enter->select_cursor), which is checked against
    the full screen chain and so, unlike every other binding here, does not
    respect modal boundaries by itself. Caught by hand while adding the
    alias: pressing Enter (or even 'c') just to read the help screen was
    silently confirming the group underneath it."""
    app = new_app()
    async with app.run_test(size=(170, 50)) as pilot:
        await pilot.pause()
        await pilot.press("question_mark")
        await pilot.pause()
        assert len(app.screen_stack) > 1, "expected the help screen to be open"

        await pilot.press("enter")
        await pilot.pause()
        assert app.groups[0].status == "pending", "Enter must not confirm through the help modal"

        await pilot.press("c")
        await pilot.pause()
        assert app.groups[0].status == "pending", "'c' must not confirm through the help modal"
    print("  ok  Enter/'c' do not confirm while the help modal is open")


async def test_every_destructive_key_alias_is_click_blocked() -> None:
    """Not just whichever key the Footer happens to render -- every key in
    _DESTRUCTIVE_KEYS must be a no-op through simulate_key(), since that's
    the only path a Footer click can take."""
    app = new_app()
    assert app._DESTRUCTIVE_KEYS >= {"c", "enter", "s", "delete", "backspace"}, (
        "expected the confirm/skip aliases to be present in _DESTRUCTIVE_KEYS; "
        "if this fails, the alias list below and the BINDINGS above have drifted apart"
    )
    async with app.run_test(size=(170, 50)) as pilot:
        await pilot.pause()
        for key in sorted(app._DESTRUCTIVE_KEYS):
            app.simulate_key(key)
            await pilot.pause()
        assert app.groups[0].status == "pending", (
            "a click-simulated destructive key alias slipped past the guard"
        )
    print("  ok  every _DESTRUCTIVE_KEYS entry is blocked via simulate_key")


async def main() -> None:
    fd.PreviewImage = HalfcellImage  # deterministic headless renderer, no real terminal needed
    for test in (
        test_enter_confirms_regardless_of_focus,
        test_delete_and_backspace_skip,
        test_escape_quits,
        test_f1_opens_help,
        test_enter_does_not_pierce_the_help_modal,
        test_every_destructive_key_alias_is_click_blocked,
    ):
        print(f"{test.__name__}:")
        await test()
    print("all keyboard-alias tests passed")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except AssertionError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)
