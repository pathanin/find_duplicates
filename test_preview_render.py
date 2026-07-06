"""Headless regression test for the image preview in find_duplicates.py.

The preview widget historically had two silent failure modes: blank boxes
(wrong widget class for the active graphics protocol) and stretched photos
(concrete CSS width/height bypassing the library's aspect-ratio letterboxing).
Neither raises an exception, so this test verifies geometry mechanically:

- runs DuplicateReviewApp under Textual's test Pilot with the HalfcellImage
  renderer (deterministic character-cell output, no terminal graphics needed);
- asserts every rendered image region is non-empty and preserves its source
  aspect ratio within tolerance, including boundary shapes (4:1 wide, 1:4
  tall, square, portrait, landscape);
- re-runs with the old buggy CSS (width: 100%; height: 1fr) and asserts the
  distortion IS detected, proving the aspect check can actually fail.

SixelImage (used in iTerm2) inherits get_content_width/height from the same
base class, so the geometry verified here is renderer-independent.

Run: python3 test_preview_render.py
"""

import asyncio
import sys
import tempfile
from pathlib import Path

from PIL import Image as PILImage

import find_duplicates as fd
from textual_image.widget import HalfcellImage
from textual_image._terminal import get_cell_size

ASPECT_TOLERANCE = 0.15  # rendered vs. source ratio, after cell quantization


def fake_result(width: int, height: int) -> dict:
    return {
        "dimensions": (width, height),
        "file_size": 123456,
        "sharpness_normalized": 123.4,
        "effective_resolution_fraction": 0.912,
        "effective_resolution_px_equiv": width * height * 0.8,
        "noise_sigma": 2.345,
        "blockiness": 0.123,
        "brisque": None,
        "niqe": None,
        "quality_score": 0.777,
    }


def make_group(shapes: list[tuple[int, int]]) -> fd.Group:
    thumbs = []
    for i, (w, h) in enumerate(shapes):
        img = PILImage.new("RGB", (w, h), ((60 * i) % 255, 120, 200))
        thumbs.append(img)
    return fd.Group(
        paths=[Path(f"synthetic_{i}_{w}x{h}.png") for i, (w, h) in enumerate(shapes)],
        results=[fake_result(w, h) for w, h in shapes],
        thumbnails=thumbs,
        suggested_idx=0,
        current_pick=0,
        is_close_call=False,
    )


def measure(app: fd.DuplicateReviewApp) -> list[dict]:
    """Return per-widget geometry: source ratio vs. rendered screen-pixel ratio."""
    cell = get_cell_size()
    out = []
    for w in app.query(HalfcellImage):
        reg = w.content_region
        rendered = (
            (reg.width * cell.width) / (reg.height * cell.height) if reg.height else 0.0
        )
        src = w._image_width / w._image_height if w._image_height else 0.0
        err = abs(rendered - src) / src if src else 1.0
        out.append(
            {
                "src": f"{w._image_width}x{w._image_height}",
                "region": f"{reg.width}x{reg.height}",
                "src_ratio": src,
                "rendered_ratio": rendered,
                "err": err,
                "empty": reg.width == 0 or reg.height == 0,
            }
        )
    return out


async def render_and_measure(app: fd.DuplicateReviewApp, n_groups: int) -> list[dict]:
    measurements = []
    async with app.run_test(size=(170, 50)) as pilot:
        for g in range(n_groups):
            if g:
                await pilot.press("down")
            await pilot.pause()
            await asyncio.sleep(0.3)
            await pilot.pause()
            measurements += measure(app)
    return measurements


def new_app(groups: list[fd.Group], app_cls: type | None = None) -> fd.DuplicateReviewApp:
    cls = app_cls or fd.DuplicateReviewApp
    scratch = Path(tempfile.mkdtemp())
    return cls(groups, scratch / "_dup_test", dry_run=True, manifest_path=scratch / "decisions.json")


async def test_aspect_preserved() -> None:
    groups = [
        make_group([(640, 800), (1600, 1200)]),  # portrait + landscape
        make_group([(1200, 300), (300, 1200), (500, 500)]),  # 4:1, 1:4, square
    ]
    for m in await render_and_measure(new_app(groups), len(groups)):
        assert not m["empty"], f"blank image widget: {m}"
        assert m["err"] < ASPECT_TOLERANCE, f"aspect distorted: {m}"
        print(f"  ok  {m['src']:>10} -> {m['region']:>6} cells  err={m['err']:.1%}")


async def test_detector_catches_stretch() -> None:
    """With the old buggy CSS the same check must flag distortion, proving the
    aspect assertion above is capable of failing."""

    class BuggyCSSApp(fd.DuplicateReviewApp):
        CSS = fd.DuplicateReviewApp.CSS.replace(
            ".preview-image { width: auto; height: auto; }",
            ".preview-image { width: 100%; height: 1fr; }",
        )

    assert "width: 100%" in BuggyCSSApp.CSS, "CSS replacement did not apply; update the test"
    groups = [make_group([(300, 1200), (300, 1200), (300, 1200)])]
    measurements = await render_and_measure(new_app(groups, BuggyCSSApp), 1)
    worst = max(m["err"] for m in measurements)
    assert worst >= ASPECT_TOLERANCE, (
        f"expected buggy CSS to distort a 1:4 image but max err was {worst:.1%}; "
        "the aspect check may be vacuous"
    )
    print(f"  ok  buggy CSS correctly detected as distorted (max err {worst:.1%})")


async def main() -> None:
    fd.PreviewImage = HalfcellImage  # force deterministic headless renderer
    for test in (test_aspect_preserved, test_detector_catches_stretch):
        print(f"{test.__name__}:")
        await test()
    print("all preview render tests passed")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except AssertionError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)
