"""Regression tests for --recursive: subdirectory scanning, dest_dir
exclusion on re-scan, and relative-path preservation in move destinations
and TUI display.

Run: python3 test_recursive_scan.py
"""

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import find_duplicates as fd


def make_texture(h: int, w: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(h // 8 + 1, w // 8 + 1, 3), dtype=np.uint8)
    return cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)


def save_jpeg(img: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 85])


def make_duplicate_pair(seed: int, p1: Path, p2: Path) -> None:
    """Same source texture resized to two different sizes -- perceptually
    close enough that group_duplicates should group them at
    DEFAULT_HASH_THRESHOLD, saved directly to the given (possibly nested)
    paths."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(150, 200, 3), dtype=np.uint8)
    big = cv2.resize(base, (1600, 1200), interpolation=cv2.INTER_CUBIC)
    small = cv2.resize(base, (400, 300), interpolation=cv2.INTER_CUBIC)
    save_jpeg(big, p1)
    save_jpeg(small, p2)


def test_find_images_flat_default_ignores_subdirs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        save_jpeg(make_texture(100, 100, 1), tmp / "top.jpg")
        save_jpeg(make_texture(100, 100, 2), tmp / "sub" / "nested.jpg")

        found = fd.find_images(tmp)
        assert found == [tmp / "top.jpg"], f"default scan must stay top-level only, got {found}"
        print("  ok  default (non-recursive) scan ignores files in subdirectories")


def test_find_images_recursive_finds_nested() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        save_jpeg(make_texture(100, 100, 1), tmp / "top.jpg")
        save_jpeg(make_texture(100, 100, 2), tmp / "sub" / "nested.jpg")

        found = set(fd.find_images(tmp, recursive=True))
        assert found == {tmp / "top.jpg", tmp / "sub" / "nested.jpg"}, (
            f"expected both top-level and nested files, got {found}"
        )
        print("  ok  recursive scan finds files in subdirectories")


def test_find_images_recursive_excludes_dest_dir() -> None:
    """Failure case this guards against: without exclusion, a re-scan of a
    directory that already has a _duplicates/ folder from a prior run would
    treat previously-moved files as fresh scan candidates."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        dest = tmp / "_duplicates"
        save_jpeg(make_texture(100, 100, 1), tmp / "top.jpg")
        save_jpeg(make_texture(100, 100, 2), tmp / "sub" / "nested.jpg")
        save_jpeg(make_texture(100, 100, 3), dest / "leftover.jpg")

        found = set(fd.find_images(tmp, recursive=True, exclude_dir=dest))
        assert found == {tmp / "top.jpg", tmp / "sub" / "nested.jpg"}, (
            f"expected dest_dir contents excluded, got {found}"
        )
        print("  ok  recursive scan excludes files already inside dest_dir")


def test_compute_dest_preserves_relative_path() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        dest_dir = tmp / "_duplicates"
        path = tmp / "sub1" / "photo.jpg"

        dest = fd._compute_dest(path, dest_dir, dry_run=False, recursive=True, scan_root=tmp)

        assert dest == dest_dir / "sub1" / "photo.jpg", f"expected mirrored subdir path, got {dest}"
        assert dest.parent.is_dir(), "parent directory must be created for a real (non-dry-run) move"
        print("  ok  recursive dest mirrors the file's subdirectory under dest_dir")


def test_compute_dest_collision_preserves_subdir() -> None:
    """The collision-suffix path must not drop the mirrored subdirectory --
    a naive `dest_dir / f'{stem}_dup{n}{suffix}'` would silently flatten the
    file into dest_dir's root, defeating the whole point of Stage 1.3."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        dest_dir = tmp / "_duplicates"
        # Simulate a leftover file already sitting at the mirrored dest path
        # from a prior run -- this is what actually triggers the collision
        # branch (two files never share a relative path within one scan).
        leftover = dest_dir / "sub1" / "photo.jpg"
        leftover.parent.mkdir(parents=True)
        leftover.write_bytes(b"leftover")

        incoming = tmp / "sub1" / "photo.jpg"
        dest = fd._compute_dest(incoming, dest_dir, dry_run=False, recursive=True, scan_root=tmp)

        assert dest == dest_dir / "sub1" / "photo_dup1.jpg", (
            f"collision suffix must stay inside the mirrored subdirectory, got {dest}"
        )
        print("  ok  collision suffix keeps the file inside its mirrored subdirectory")


def test_compute_dest_non_recursive_unchanged() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        dest_dir = tmp / "_duplicates"
        path = tmp / "photo.jpg"

        dest = fd._compute_dest(path, dest_dir, dry_run=False)
        assert dest == dest_dir / "photo.jpg"

        (dest_dir / "photo.jpg").parent.mkdir(parents=True, exist_ok=True)
        (dest_dir / "photo.jpg").write_bytes(b"leftover")
        dest2 = fd._compute_dest(path, dest_dir, dry_run=False)
        assert dest2 == dest_dir / "photo_dup1.jpg", f"expected flat collision suffix, got {dest2}"
        print("  ok  non-recursive _compute_dest behavior is unchanged (flat, same collision naming)")


def test_display_path_shows_relative_path_when_recursive() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        group = fd.Group(
            paths=[tmp / "sub1" / "a.jpg", tmp / "sub2" / "a.jpg"],
            results=[{}, {}],
            thumbnails=None,
            suggested_idx=0,
            current_pick=0,
            is_close_call=False,
        )
        app = fd.DuplicateReviewApp([group], tmp / "_duplicates", dry_run=True, recursive=True, scan_root=tmp)

        shown = {app._display_path(p) for p in group.paths}
        assert shown == {str(Path("sub1") / "a.jpg"), str(Path("sub2") / "a.jpg")}, (
            f"two same-named files in different subdirs must render distinguishably, got {shown}"
        )
        print("  ok  recursive display path disambiguates same-named files in different subdirectories")


def test_display_path_falls_back_to_name_when_not_recursive() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        group = fd.Group(
            paths=[tmp / "a.jpg"],
            results=[{}],
            thumbnails=None,
            suggested_idx=0,
            current_pick=0,
            is_close_call=False,
        )
        app = fd.DuplicateReviewApp([group], tmp / "_duplicates", dry_run=True)
        assert app._display_path(tmp / "a.jpg") == "a.jpg"
        print("  ok  non-recursive display path is just the filename, unchanged from before")


def test_build_groups_recursive_groups_across_subdirs() -> None:
    """Integration: a near-duplicate pair living in two different
    subdirectories is still grouped, and grouped/cache keys stay unique
    absolute paths regardless of nesting."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        p1 = tmp / "vacation" / "big.jpg"
        p2 = tmp / "backup" / "small.jpg"
        make_duplicate_pair(seed=10, p1=p1, p2=p2)

        groups = fd.build_groups(tmp, fd.DEFAULT_HASH_THRESHOLD, recursive=True, dest_dir=tmp / "_duplicates")

        assert len(groups) == 1, f"expected the cross-subdir pair to be grouped, got {len(groups)} group(s)"
        assert set(groups[0].paths) == {p1, p2}

        cache = fd.load_hash_cache(tmp)
        assert str(p1.resolve()) in cache and str(p2.resolve()) in cache, (
            "hash cache keys must remain unique absolute paths across different subdirectories"
        )
        print("  ok  build_groups(recursive=True) groups a near-duplicate pair across subdirectories")


def test_build_groups_recursive_excludes_prior_run_leftovers() -> None:
    """A file already sitting in dest_dir from a prior run must not be
    re-ingested as a scan candidate on the next recursive run."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        dest_dir = tmp / "_duplicates"
        save_jpeg(make_texture(200, 200, 20), tmp / "keep.jpg")
        save_jpeg(make_texture(200, 200, 21), dest_dir / "already_moved.jpg")

        groups = fd.build_groups(tmp, fd.DEFAULT_HASH_THRESHOLD, recursive=True, dest_dir=dest_dir)
        assert groups == [], "a lone top-level file plus an excluded dest_dir leftover must not form a group"
        print("  ok  build_groups(recursive=True) excludes dest_dir leftovers from a prior run")


def main() -> None:
    tests = [
        test_find_images_flat_default_ignores_subdirs,
        test_find_images_recursive_finds_nested,
        test_find_images_recursive_excludes_dest_dir,
        test_compute_dest_preserves_relative_path,
        test_compute_dest_collision_preserves_subdir,
        test_compute_dest_non_recursive_unchanged,
        test_display_path_shows_relative_path_when_recursive,
        test_display_path_falls_back_to_name_when_not_recursive,
        test_build_groups_recursive_groups_across_subdirs,
        test_build_groups_recursive_excludes_prior_run_leftovers,
    ]
    for test in tests:
        print(f"{test.__name__}:")
        test()
    print("all recursive-scan tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)
