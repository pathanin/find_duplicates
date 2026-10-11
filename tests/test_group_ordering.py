"""build_groups() orders each group best-scoring-first, so the suggested
file is always index 0 -- leftmost preview and leftmost value column in
both front ends.

The reorder permutes two parallel lists (Group.paths and Group.results).
Permuting one and not the other would silently attach every file's metrics
to a different file's name -- the table would show the right numbers under
the wrong photo, and confirming would move the wrong files. Nothing else in
the codebase re-derives that pairing, so it can only be caught here:
test_paths_and_results_stay_aligned is the real point of this file.

Run: python3 test_group_ordering.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import duplicates_core as dc


def save_jpeg(img: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 85])


def make_quality_split_group(directory: Path) -> tuple[Path, Path]:
    """A near-duplicate pair whose filename order deliberately disagrees
    with its quality order: the *worse* file sorts first by name. Without
    that, "the best file is at index 0" would pass even if the sort never
    ran, since find_images() already returns paths sorted by name."""
    rng = np.random.default_rng(7)
    base = rng.integers(0, 255, size=(150, 200, 3), dtype=np.uint8)
    small = directory / "a_small.jpg"
    big = directory / "z_big.jpg"
    save_jpeg(cv2.resize(base, (400, 300), interpolation=cv2.INTER_CUBIC), small)
    save_jpeg(cv2.resize(base, (1600, 1200), interpolation=cv2.INTER_CUBIC), big)
    return small, big


def build_one_group(directory: Path) -> dc.Group:
    groups = dc.build_groups(directory, dc.DEFAULT_HASH_THRESHOLD, dest_dir=directory / "_duplicates")
    assert len(groups) == 1, f"expected exactly one duplicate group, got {len(groups)}"
    return groups[0]


def test_best_scoring_file_is_first() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        small, big = make_quality_split_group(tmp)
        group = build_one_group(tmp)

        scores = [r["quality_score"] for r in group.results]
        assert scores == sorted(scores, reverse=True), f"group is not ordered best-first: {scores}"
        assert group.suggested_idx == 0, f"suggested_idx should always be 0, got {group.suggested_idx}"
        assert group.current_pick == 0, f"current_pick should start at 0, got {group.current_pick}"
        assert group.paths[0] == big, (
            f"the higher-quality file should be first, got {group.paths[0].name}"
        )
        print("  ok  build_groups puts the best-scoring file at index 0")


def test_detector_filename_order_would_have_failed() -> None:
    """Proves the fixture isn't vacuous: the file that must end up first is
    NOT the one plain filename order would put there, so a build_groups that
    skipped the reorder entirely would fail test_best_scoring_file_is_first
    rather than passing by coincidence."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        small, big = make_quality_split_group(tmp)
        by_name = dc.find_images(tmp)
        assert by_name[0] == small, "fixture broken: the worse file should sort first by name"
        assert by_name[0] != big, "fixture is vacuous: name order already matches quality order"
        print("  ok  filename order disagrees with quality order, so the sort is doing real work")


def test_paths_and_results_stay_aligned() -> None:
    """The reorder permutes Group.paths and Group.results together. If the
    two ever drift apart, every metric row is attributed to the wrong photo
    and Confirm moves the wrong files. file_size is the cross-check: it's
    recorded in the result dict by analyze_paths and is independently
    readable from the path on disk."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        make_quality_split_group(tmp)
        group = build_one_group(tmp)

        for i, (path, result) in enumerate(zip(group.paths, group.results)):
            assert result["file_size"] == path.stat().st_size, (
                f"index {i}: results[{i}]['file_size']={result['file_size']} but "
                f"{path.name} is {path.stat().st_size} bytes on disk -- paths and "
                "results were permuted out of step with each other"
            )
        print("  ok  every path still lines up with its own metrics after the reorder")


def test_tied_scores_keep_filename_order() -> None:
    """Boundary: byte-identical copies score identically, so quality can't
    separate them. Equal-length names tie on the length key too and fall
    through to the name itself, so they stay in filename order -- identical
    copies must not shuffle between scans for no visible reason."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        rng = np.random.default_rng(11)
        base = rng.integers(0, 255, size=(150, 200, 3), dtype=np.uint8)
        img = cv2.resize(base, (800, 600), interpolation=cv2.INTER_CUBIC)
        names = ["a.jpg", "b.jpg", "c.jpg"]
        for name in names:
            save_jpeg(img, tmp / name)

        group = build_one_group(tmp)
        scores = [r["quality_score"] for r in group.results]
        assert len(set(scores)) == 1, f"identical copies should tie on score, got {scores}"
        assert [p.name for p in group.paths] == names, (
            f"tied files must keep filename order, got {[p.name for p in group.paths]}"
        )
        print("  ok  files tied on score keep filename order")


def test_tied_copies_suggest_the_original_not_the_copy() -> None:
    """Regression test: the tool used to suggest keeping the *copy*.

    Byte-identical duplicates score exactly equal, so the tie-break is the
    entire decision -- and plain filename order loses it, because " 2.jpg"
    and " copy.jpg" both sort ahead of ".jpg" (space is 0x20, period is
    0x2e). Every macOS-duplicated photo in a library hit this. The shorter
    name wins a tie now, since every convention for naming a derived copy
    appends to the original's name.

    This is the destructive path: --auto moves whatever lands at index 0's
    expense, so getting it backwards moved the original into _duplicates/
    and kept the copy."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        rng = np.random.default_rng(13)
        base = rng.integers(0, 255, size=(150, 200, 3), dtype=np.uint8)
        img = cv2.resize(base, (800, 600), interpolation=cv2.INTER_CUBIC)
        original = tmp / "holiday.jpg"
        save_jpeg(img, original)
        # The three shapes a copy actually arrives in, all of which sort
        # before the original by name.
        copies = [tmp / "holiday 2.jpg", tmp / "holiday copy.jpg", tmp / "holiday (1).jpg"]
        for copy in copies:
            shutil.copyfile(original, copy)

        assert dc.find_images(tmp)[0] != original, (
            "fixture is vacuous: the original already sorts first by name"
        )

        group = build_one_group(tmp)
        scores = [r["quality_score"] for r in group.results]
        assert len(set(scores)) == 1, f"identical copies should tie on score, got {scores}"
        assert group.paths[0] == original, (
            f"the original should be suggested, not a copy -- got {group.paths[0].name}"
        )
        print("  ok  a tie between an original and its copies suggests the original")


def test_identical_copies_are_marked_and_the_close_call_skips_them() -> None:
    """Byte-identical copies tie on every metric, so with one at index 1
    the old rule -- top two within CLOSE_CALL_MARGIN -- flagged a group like
    {original, its copy, a small export} as a close call, though the real
    choice (original against the small export) wasn't close at all. The
    reviewer was told to look harder at two files that are the same bytes.
    Identical copies are marked instead, and the close call measures the
    top file against the best file that actually differs from it."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        rng = np.random.default_rng(17)
        base = rng.integers(0, 255, size=(150, 200, 3), dtype=np.uint8)
        original = tmp / "beach.jpg"
        save_jpeg(cv2.resize(base, (1600, 1200), interpolation=cv2.INTER_CUBIC), original)
        copy = tmp / "beach copy.jpg"
        shutil.copyfile(original, copy)
        small = tmp / "beach small.jpg"
        save_jpeg(cv2.resize(base, (400, 300), interpolation=cv2.INTER_CUBIC), small)

        group = build_one_group(tmp)
        assert group.paths == [original, copy, small], [p.name for p in group.paths]
        scores = [r["quality_score"] for r in group.results]
        assert scores[0] == scores[1], "fixture: the copy must tie the original, as the old rule saw it"
        assert scores[0] - scores[2] >= dc.CLOSE_CALL_MARGIN, (
            f"fixture: the small export must clearly lose, got {scores}"
        )
        assert group.identical_to == [None, 0, None], group.identical_to
        assert group.all_identical is False
        assert group.is_close_call is False, "a tie with an identical copy is not a close call"
        print("  ok  an identical copy is marked as such, and doesn't make the group a close call")


def test_a_group_of_identical_copies_is_all_identical() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        rng = np.random.default_rng(19)
        base = rng.integers(0, 255, size=(150, 200, 3), dtype=np.uint8)
        original = tmp / "dog.jpg"
        save_jpeg(cv2.resize(base, (800, 600), interpolation=cv2.INTER_CUBIC), original)
        shutil.copyfile(original, tmp / "dog 2.jpg")
        shutil.copyfile(original, tmp / "dog (1).jpg")

        group = build_one_group(tmp)
        assert group.identical_to == [None, 0, 0], group.identical_to
        assert group.all_identical is True
        assert group.is_close_call is False, "keeping any one of identical files loses nothing"
        assert isinstance(group.all_identical, bool) and isinstance(group.is_close_call, bool)
        print("  ok  a group of byte-identical copies is all_identical and not a close call")


def test_same_pixels_different_bytes_is_still_a_close_call() -> None:
    """The other side of the boundary: identical means identical *bytes*.
    The same pixels saved at two PNG compression levels decode alike and
    tie on every metric, but they are different files -- say a tie, not
    "same file"."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        rng = np.random.default_rng(23)
        base = rng.integers(0, 255, size=(150, 200, 3), dtype=np.uint8)
        img = cv2.resize(base, (800, 600), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(str(tmp / "fast.png"), img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        cv2.imwrite(str(tmp / "small.png"), img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        assert (tmp / "fast.png").read_bytes() != (tmp / "small.png").read_bytes(), "fixture: bytes must differ"

        group = build_one_group(tmp)
        assert group.identical_to == [None, None], group.identical_to
        assert group.all_identical is False
        assert group.is_close_call is True, "an exact tie between different files is a close call"
        print("  ok  pixel-identical files with different bytes are a close call, not 'identical'")


def main() -> None:
    for fn in (
        test_best_scoring_file_is_first,
        test_detector_filename_order_would_have_failed,
        test_paths_and_results_stay_aligned,
        test_tied_scores_keep_filename_order,
        test_tied_copies_suggest_the_original_not_the_copy,
        test_identical_copies_are_marked_and_the_close_call_skips_them,
        test_a_group_of_identical_copies_is_all_identical,
        test_same_pixels_different_bytes_is_still_a_close_call,
    ):
        print(f"{fn.__name__}:")
        fn()
    print("all group-ordering tests passed")


if __name__ == "__main__":
    main()
