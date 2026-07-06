"""Regression test: a stray mouse click on the Footer must never confirm or
skip a group in find_duplicates.py's review TUI.

Textual's Footer renders each keybinding as a real clickable button --
clicking one calls App.simulate_key(), posting the same event a keypress
would. Left unguarded, that turns "c Confirm keep" into a button one
incidental click away from silently moving files (e.g. clicking the terminal
to refocus it right as the scan finishes and the TUI takes over the mouse).
DuplicateReviewApp.simulate_key() overrides this to drop click-triggered
"c"/"s", while leaving real keypresses untouched.

Run: python3 test_footer_click_safety.py
"""

import asyncio
import sys
import tempfile
from pathlib import Path

from PIL import Image as PILImage
from textual_image.widget import HalfcellImage
from textual.widgets._footer import FooterKey

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


def new_app(app_cls: type | None = None) -> fd.DuplicateReviewApp:
    cls = app_cls or fd.DuplicateReviewApp
    scratch = Path(tempfile.mkdtemp())
    return cls([make_group()], scratch / "_dup", dry_run=True, manifest_path=scratch / "decisions.json")


async def click_footer_action(app: fd.DuplicateReviewApp, action: str, pilot) -> None:
    for key_widget in app.query(FooterKey):
        if key_widget.action == action:
            region = key_widget.region
            await pilot.click(offset=(region.x + region.width // 2, region.y))
            return
    raise AssertionError(f"no footer key bound to action {action!r}")


async def test_footer_click_does_not_confirm_or_skip() -> None:
    app = new_app()
    async with app.run_test(size=(170, 50)) as pilot:
        await pilot.pause()
        await click_footer_action(app, "confirm", pilot)
        await pilot.pause()
        assert app.groups[0].status == "pending", "footer click on 'c' must not confirm the group"
        assert app.manifest == [], "footer click on 'c' must not move any files"

        await click_footer_action(app, "skip", pilot)
        await pilot.pause()
        assert app.groups[0].status == "pending", "footer click on 's' must not skip the group"
    print("  ok  footer clicks on Confirm/Skip are no-ops")


async def test_keyboard_confirm_still_works() -> None:
    """Proves the click guard doesn't just disable confirm outright."""
    app = new_app()
    async with app.run_test(size=(170, 50)) as pilot:
        await pilot.pause()
        await pilot.press("c")
        await pilot.pause()
        assert app.groups[0].status == "confirmed", "a real keypress must still confirm"
    print("  ok  a real 'c' keypress still confirms")


async def test_click_guard_is_not_vacuous() -> None:
    """Re-runs the click with the guard removed and asserts it WOULD have
    confirmed, proving the assertions above are actually exercising the
    guard rather than passing for an unrelated reason."""

    class UnguardedApp(fd.DuplicateReviewApp):
        def simulate_key(self, key: str) -> None:
            super(fd.DuplicateReviewApp, self).simulate_key(key)

    app = new_app(UnguardedApp)
    async with app.run_test(size=(170, 50)) as pilot:
        await pilot.pause()
        await click_footer_action(app, "confirm", pilot)
        await pilot.pause()
        assert app.groups[0].status == "confirmed", (
            "expected the unguarded app to confirm on a footer click; "
            "if it didn't, the guard test above may be vacuous"
        )
    print("  ok  without the guard, the same click does confirm (guard is load-bearing)")


async def main() -> None:
    fd.PreviewImage = HalfcellImage  # deterministic headless renderer, no real terminal needed
    for test in (
        test_footer_click_does_not_confirm_or_skip,
        test_keyboard_confirm_still_works,
        test_click_guard_is_not_vacuous,
    ):
        print(f"{test.__name__}:")
        await test()
    print("all footer click-safety tests passed")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except AssertionError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)
