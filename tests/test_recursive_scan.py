"""Regression tests for --recursive: subdirectory scanning, dest_dir
exclusion on re-scan, and relative-path preservation in move destinations
and the web UI's display labels.

Run: python3 test_recursive_scan.py
"""

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import duplicates_core as dc
import duplicates_web as web


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


def fake_session(directory: Path, recursive: bool) -> SimpleNamespace:
    return SimpleNamespace(params=SimpleNamespace(directory=directory, recursive=recursive))


def test_find_images_flat_default_ignores_subdirs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        save_jpeg(make_texture(100, 100, 1), tmp / "top.jpg")
        save_jpeg(make_texture(100, 100, 2), tmp / "sub" / "nested.jpg")

        found = dc.find_images(tmp)
        assert found == [tmp / "top.jpg"], f"default scan must stay top-level only, got {found}"
        print("  ok  default (non-recursive) scan ignores files in subdirectories")


def test_find_images_recursive_finds_nested() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        save_jpeg(make_texture(100, 100, 1), tmp / "top.jpg")
        save_jpeg(make_texture(100, 100, 2), tmp / "sub" / "nested.jpg")

        found = set(dc.find_images(tmp, recursive=True))
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

        found = set(dc.find_images(tmp, recursive=True, exclude_dir=dest))
        assert found == {tmp / "top.jpg", tmp / "sub" / "nested.jpg"}, (
            f"expected dest_dir contents excluded, got {found}"
        )
        print("  ok  recursive scan excludes files already inside dest_dir")


def test_find_images_recursive_skips_app_libraries_and_system_folders() -> None:
    """A recursive scan of ~/Pictures or a NAS share walks straight into
    folders another program owns, where every photo already has a smaller
    copy beside it: a Photos library's derivatives, Synology's @eaDir
    thumbnails, a recycle bin. Each of those pairs with its original as a
    "duplicate", and moving one out of a Photos library damages the
    library. AppleDouble "._" files (macOS metadata on a NAS) carry an
    image extension but are never images."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        wanted = [tmp / "top.jpg", tmp / "2019" / "06" / "nested.jpg", tmp / "My Library" / "pic.jpg"]
        unwanted = [
            tmp / "Photos Library.photoslibrary" / "resources" / "derivatives" / "x.jpeg",
            tmp / "Photos Library.photoslibrary" / "originals" / "x.jpeg",
            tmp / "2019" / "@eaDir" / "nested.jpg" / "SYNOPHOTO_THUMB_XL.jpg",
            tmp / "#recycle" / "old.jpg",
            tmp / ".Trashes" / "t.jpg",
            tmp / ".thumbnails" / "normal" / "t.png",
            tmp / "2019" / "._nested.jpg",
            tmp / "._top.jpg",
        ]
        for i, p in enumerate(wanted + unwanted):
            save_jpeg(make_texture(64, 64, 100 + i), p)

        found = set(dc.find_images(tmp, recursive=True))
        assert found == set(wanted), (
            f"expected only the user's own photos, got extra {sorted(found - set(wanted))} "
            f"and missing {sorted(set(wanted) - found)}"
        )
        flat = set(dc.find_images(tmp))
        assert flat == {tmp / "top.jpg"}, f"a top-level AppleDouble file must be skipped too, got {flat}"
        print("  ok  recursive scan skips app libraries, NAS thumbnail/recycle folders and AppleDouble files")


def test_find_images_scans_a_skipped_kind_of_folder_when_it_is_the_root() -> None:
    """The skip list prunes folders found *during* a walk. Pointing the
    scan at one of them on purpose is an explicit choice and still works."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "Old.photolibrary"
        save_jpeg(make_texture(64, 64, 1), root / "Masters" / "a.jpg")
        found = dc.find_images(root, recursive=True)
        assert found == [root / "Masters" / "a.jpg"], f"an explicitly chosen root must be scanned, got {found}"
        print("  ok  a library folder chosen as the scan root is still scanned")


def test_compute_dest_preserves_relative_path() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        dest_dir = tmp / "_duplicates"
        path = tmp / "sub1" / "photo.jpg"

        dest = dc._compute_dest(path, dest_dir, dry_run=False, recursive=True, scan_root=tmp)

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
        dest = dc._compute_dest(incoming, dest_dir, dry_run=False, recursive=True, scan_root=tmp)

        assert dest == dest_dir / "sub1" / "photo_dup1.jpg", (
            f"collision suffix must stay inside the mirrored subdirectory, got {dest}"
        )
        print("  ok  collision suffix keeps the file inside its mirrored subdirectory")


def test_compute_dest_non_recursive_unchanged() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        dest_dir = tmp / "_duplicates"
        path = tmp / "photo.jpg"

        dest = dc._compute_dest(path, dest_dir, dry_run=False)
        assert dest == dest_dir / "photo.jpg"

        (dest_dir / "photo.jpg").parent.mkdir(parents=True, exist_ok=True)
        (dest_dir / "photo.jpg").write_bytes(b"leftover")
        dest2 = dc._compute_dest(path, dest_dir, dry_run=False)
        assert dest2 == dest_dir / "photo_dup1.jpg", f"expected flat collision suffix, got {dest2}"
        print("  ok  non-recursive _compute_dest behavior is unchanged (flat, same collision naming)")


def test_display_path_shows_relative_path_when_recursive() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        session = fake_session(tmp, recursive=True)
        paths = [tmp / "sub1" / "a.jpg", tmp / "sub2" / "a.jpg"]

        shown = {web._display_path(session, p) for p in paths}
        assert shown == {str(Path("sub1") / "a.jpg"), str(Path("sub2") / "a.jpg")}, (
            f"two same-named files in different subdirs must render distinguishably, got {shown}"
        )
        print("  ok  recursive display path disambiguates same-named files in different subdirectories")


def test_display_path_falls_back_to_name_when_not_recursive() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        session = fake_session(tmp, recursive=False)
        assert web._display_path(session, tmp / "a.jpg") == "a.jpg"
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

        cache: dict = {}
        groups = dc.build_groups(tmp, dc.DEFAULT_HASH_THRESHOLD, recursive=True,
                                 dest_dir=tmp / "_duplicates", hash_cache=cache)

        assert len(groups) == 1, f"expected the cross-subdir pair to be grouped, got {len(groups)} group(s)"
        assert set(groups[0].paths) == {p1, p2}

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

        groups = dc.build_groups(tmp, dc.DEFAULT_HASH_THRESHOLD, recursive=True, dest_dir=dest_dir)
        assert groups == [], "a lone top-level file plus an excluded dest_dir leftover must not form a group"
        print("  ok  build_groups(recursive=True) excludes dest_dir leftovers from a prior run")


def main() -> None:
    tests = [
        test_find_images_flat_default_ignores_subdirs,
        test_find_images_recursive_finds_nested,
        test_find_images_recursive_excludes_dest_dir,
        test_find_images_recursive_skips_app_libraries_and_system_folders,
        test_find_images_scans_a_skipped_kind_of_folder_when_it_is_the_root,
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
