"""Regression test for Pillow's decompression-bomb guard on large photos.

Pillow warns above MAX_IMAGE_PIXELS (~89 MP by default) and raises
DecompressionBombError above twice that (~179 MP). Ordinary camera and
cosplay-shoot JPEGs reach 100 MP (8742x11656), so the default spammed the
console on every thumbnail. Worse, above 2x the error was swallowed by
make_thumbnail/_render_scaled_jpeg's `except Exception` and the photo showed
up as a gray placeholder -- unjudgeable in review -- while a HEIC that size
(PIL is its only decoder) was silently dropped from the scan.

The check runs in Image.open on the header size, before any decode, so this
calls it directly instead of allocating a 180 MP image.

Run: python3 tests/test_large_image.py
"""

import sys
import warnings
from pathlib import Path

from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import duplicates_core  # noqa: F401 -- importing it must raise the limit


def test_large_photos_pass_the_bomb_check() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for size in [(8742, 11656), (13500, 13500)]:  # 102 MP, 182 MP (> 2x old default)
            PILImage._decompression_bomb_check(size)
    print("  ok  102 MP and 182 MP images open without warning or error")


def main() -> None:
    tests = [test_large_photos_pass_the_bomb_check]
    for test in tests:
        print(f"{test.__name__}:")
        test()
    print("all large image tests passed")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, Warning, PILImage.DecompressionBombError) as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)
