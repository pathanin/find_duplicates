"""Regression tests for --auto: the non-interactive apply_group()/
auto_apply_groups() path that skips the review UI and keeps each group's
suggested (top-scored) pick automatically.

Run: python3 test_auto_mode.py
"""

import io
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import duplicates_core as dc


def fake_result(file_size: int, quality: float) -> dict:
    return {
        "dimensions": (100, 100),
        "file_size": file_size,
        "sharpness_normalized": 100.0,
        "effective_resolution_fraction": 0.9,
        "effective_resolution_px_equiv": 9000.0,
        "noise_sigma": 1.0,
        "blockiness": 0.1,
        "brisque": None,
        "niqe": None,
        "quality_score": quality,
    }


def make_real_group(directory: Path, n: int, prefix: str, suggested_idx: int = 0) -> dc.Group:
    """Files that actually exist on disk -- dry_run=False callers really call
    shutil.move, so a synthetic non-existent Path would fail for the wrong
    reason. Each non-suggested file gets a lower quality_score so
    suggested_idx is unambiguous."""
    paths = []
    for i in range(n):
        p = directory / f"{prefix}_{i}.png"
        PILImage.new("RGB", (20, 20), (10 * i, 0, 0)).save(p)
        paths.append(p)
    thumbs = [PILImage.new("RGB", (20, 20)) for _ in range(n)]
    results = [
        fake_result(paths[i].stat().st_size, quality=1.0 if i == suggested_idx else 0.5) for i in range(n)
    ]
    return dc.Group(
        paths=paths,
        results=results,
        thumbnails=thumbs,
        suggested_idx=suggested_idx,
        current_pick=suggested_idx,
        is_close_call=False,
    )


def test_apply_group_standalone() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        group = make_real_group(directory, n=3, prefix="g")
        dest_dir = directory / "_duplicates"
        manifest: list[dict] = []

        entry = dc.apply_group(group, 0, keep_idx=0, dest_dir=dest_dir, dry_run=False, manifest=manifest)

        assert group.status == "pending", "apply_group must not set status -- the caller decides"
        assert len(entry["moved"]) == 2
        assert manifest == [entry]
        assert group.paths[0].exists(), "kept file must stay at its original location"
        assert not group.paths[1].exists() and not group.paths[2].exists()
        assert (dest_dir / "g_1.png").exists() and (dest_dir / "g_2.png").exists()
        print("  ok  apply_group() moves non-kept files the same way apply_pick does")


def read_move_log(dest_dir: Path) -> list[list[str]]:
    """The log's rows as [time, action, from, to], comment lines dropped."""
    import csv
    with open(dest_dir / dc.MOVE_LOG_NAME, newline="", encoding="utf-8") as f:
        return [row for row in csv.reader(f, delimiter="\t") if row and not row[0].startswith("#")]


def test_apply_group_and_unapply_write_a_move_log() -> None:
    """The manifest is in memory only, so once the process exits it can no
    longer say where a file came from -- and a collision rename
    (name_dup1.jpg) has already lost the original name. A log next to the
    moved files keeps that record on disk, and records restores too, so it
    never claims a file is in dest_dir after undo brought it back."""
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        group = make_real_group(directory, n=3, prefix="log")
        dest_dir = directory / "_duplicates"
        manifest: list[dict] = []

        dc.apply_group(group, 0, keep_idx=0, dest_dir=dest_dir, dry_run=False, manifest=manifest)
        rows = read_move_log(dest_dir)
        assert [r[1] for r in rows] == ["moved", "moved"], rows
        assert {(r[2], r[3]) for r in rows} == {
            (str(group.paths[1]), str(dest_dir / "log_1.png")),
            (str(group.paths[2]), str(dest_dir / "log_2.png")),
        }, rows

        dc.unapply(manifest, 0)
        rows = read_move_log(dest_dir)
        assert [r[1] for r in rows] == ["moved", "moved", "restored", "restored"], rows
        assert group.paths[1].exists() and group.paths[2].exists()
        print("  ok  apply_group logs each move and unapply logs each restore, in <dest>/moves.log")


def test_move_log_failure_does_not_fail_the_move() -> None:
    """The log is a record, not part of the move: a log that can't be
    written (here a directory squatting on its name) must not turn a
    successful move into a failed confirm."""
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        group = make_real_group(directory, n=2, prefix="nolog")
        dest_dir = directory / "_duplicates"
        (dest_dir / dc.MOVE_LOG_NAME).mkdir(parents=True)
        manifest: list[dict] = []

        entry = dc.apply_group(group, 0, keep_idx=0, dest_dir=dest_dir, dry_run=False, manifest=manifest)
        assert len(entry["moved"]) == 1 and not group.paths[1].exists(), "the move itself must still happen"
        dc.unapply(manifest, 0)
        assert group.paths[1].exists(), "and so must the restore"
        print("  ok  an unwritable move log never fails the move or the restore")


def test_auto_dry_run_reports_the_bytes_it_would_move() -> None:
    """bytes_reclaimed stays 0 under dry run (nothing moved), but a preview
    that can't say how much it would free is half a preview."""
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        group = make_real_group(directory, n=3, prefix="w")
        expected = group.paths[1].stat().st_size + group.paths[2].stat().st_size
        summary = dc.auto_apply_groups([group], directory / "_duplicates", dry_run=True)
        assert summary["bytes_moved"] == expected, summary
        assert summary["bytes_reclaimed"] == 0
        assert len(summary["entries"]) == 1 and len(summary["entries"][0]["moved"]) == 2, summary
        print("  ok  a dry run reports the bytes it would move, and every per-group entry")


def test_run_auto_dry_run_names_every_file() -> None:
    """--auto --dry-run is documented as "show what would happen", and used
    to print only counts. It has to name what it keeps and what it moves,
    or there is nothing to preview."""
    import contextlib
    import find_duplicates as fd
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        rng = np.random.default_rng(41)
        base = rng.integers(0, 255, size=(150, 200, 3), dtype=np.uint8)
        original = directory / "beach.jpg"
        save_jpeg(cv2.resize(base, (800, 600), interpolation=cv2.INTER_CUBIC), original)
        shutil.copyfile(original, directory / "beach copy.jpg")
        dest_dir = directory / "_duplicates"

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            fd._run_auto(directory, dest_dir, dc.DEFAULT_HASH_THRESHOLD, recursive=False, dry_run=True)
        text = out.getvalue()
        assert "keep beach.jpg" in text, text
        assert "would move beach copy.jpg" in text, text
        assert "Nothing was moved" in text, text
        assert (directory / "beach copy.jpg").exists() and not dest_dir.exists()
        print("  ok  --auto --dry-run names the file it keeps and each file it would move")


def test_auto_apply_groups_all_pending() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        groups = [
            make_real_group(directory, n=2, prefix="a", suggested_idx=0),
            make_real_group(directory, n=3, prefix="b", suggested_idx=1),
        ]
        dest_dir = directory / "_duplicates"

        summary = dc.auto_apply_groups(groups, dest_dir, dry_run=False)

        assert summary["confirmed"] == 2 and summary["failed"] == 0
        assert summary["files_moved"] == 1 + 2
        assert all(g.status == "confirmed" for g in groups)
        assert groups[0].current_pick == 0 and groups[1].current_pick == 1
        print("  ok  every pending group is confirmed using its suggested pick")


def test_auto_apply_groups_skips_already_confirmed() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        already = make_real_group(directory, n=2, prefix="done")
        already.status = "confirmed"
        pending = make_real_group(directory, n=2, prefix="new")
        dest_dir = directory / "_duplicates"

        summary = dc.auto_apply_groups([already, pending], dest_dir, dry_run=False)

        assert summary["confirmed"] == 1, "already-confirmed group must not be reprocessed"
        assert already.paths[1].exists(), "an already-confirmed group's files must be left untouched"
        assert not pending.paths[1].exists()
        print("  ok  auto_apply_groups skips groups that aren't 'pending'")


def test_auto_apply_groups_dry_run() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        group = make_real_group(directory, n=3, prefix="d")
        dest_dir = directory / "_duplicates"

        summary = dc.auto_apply_groups([group], dest_dir, dry_run=True)

        assert summary["confirmed"] == 1 and summary["files_moved"] == 2
        assert summary["bytes_reclaimed"] == 0, "dry run must not report bytes as reclaimed -- nothing moved"
        assert group.paths[1].exists() and group.paths[2].exists(), "dry run must not move any files"
        assert not dest_dir.exists(), "dry run must not even create dest_dir"
        print("  ok  dry run reports counts but moves nothing and reclaims 0 bytes")


def test_auto_apply_groups_respects_close_call() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        group = make_real_group(directory, n=2, prefix="c", suggested_idx=1)
        group.is_close_call = True
        dest_dir = directory / "_duplicates"

        summary = dc.auto_apply_groups([group], dest_dir, dry_run=False)

        assert summary["confirmed"] == 1
        assert group.current_pick == 1, "auto mode must not second-guess a close call -- suggested_idx wins"
        assert group.paths[1].exists() and not group.paths[0].exists()
        print("  ok  a close-call group still auto-picks suggested_idx without hesitation")


def test_auto_summary_bytes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        group = make_real_group(directory, n=3, prefix="s", suggested_idx=0)
        expected_bytes = group.paths[1].stat().st_size + group.paths[2].stat().st_size
        dest_dir = directory / "_duplicates"

        summary = dc.auto_apply_groups([group], dest_dir, dry_run=False)

        assert summary["bytes_reclaimed"] == expected_bytes, (
            f"expected {expected_bytes} bytes reclaimed (sum of moved files' real sizes pre-move), "
            f"got {summary['bytes_reclaimed']}"
        )
        print("  ok  bytes_reclaimed matches the real on-disk size of every moved file")


class FlakyMove:
    """Real move on every call except the Nth, which raises -- simulates a
    disk-full/permission error partway through a multi-group auto run."""

    def __init__(self, real_move: object, fail_on_call: int):
        self.real_move = real_move
        self.fail_on_call = fail_on_call
        self.calls = 0

    def __call__(self, src, dst):
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise OSError("simulated failure partway through the run")
        return self.real_move(src, dst)


def test_auto_apply_groups_continues_after_one_group_fails() -> None:
    """A failure moving one group's files must not abort the whole run --
    earlier/later groups still get processed, and the failure is reported
    rather than raised, since --auto is meant to run unattended."""
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        # group0: 1 move (succeeds). group1: 2 moves -- 1st succeeds, 2nd
        # raises. Calls happen in group order, so this is call #3 overall.
        group0 = make_real_group(directory, n=2, prefix="ok")
        group1 = make_real_group(directory, n=3, prefix="bad")
        dest_dir = directory / "_duplicates"

        flaky = FlakyMove(dc.shutil.move, fail_on_call=3)
        dc.shutil.move = flaky
        try:
            summary = dc.auto_apply_groups([group0, group1], dest_dir, dry_run=False)
        finally:
            dc.shutil.move = flaky.real_move

        assert summary["confirmed"] == 1 and summary["failed"] == 1
        assert group0.status == "confirmed"
        assert group1.status == "pending", "the failed group must stay 'pending', same as apply_pick's invariant"
        assert len(summary["failures"]) == 1
        assert summary["failures"][0]["group"] == 1
        assert summary["failures"][0]["files_moved"] == 1, "the one file moved before the raise must be reported"
        assert summary["files_moved"] == 1 + 1, "group0's move plus group1's one successful move before the failure"
        assert not group1.paths[1].exists(), "group1's first (successful) move must have really happened"
        assert group1.paths[2].exists(), "group1's second (failed) move must leave the file untouched"
        print("  ok  a mid-run failure is reported without aborting or losing earlier groups' progress")


class ExplodingIsSymlink(Path):
    """A Path whose is_symlink() raises, simulating e.g. EACCES on a parent
    directory when apply_group checks the kept file before its move loop --
    i.e. before the try/finally that normally guarantees a manifest entry."""

    def is_symlink(self):
        raise OSError("simulated EACCES checking the kept file")


def test_auto_apply_groups_isolates_a_failure_before_the_move_loop() -> None:
    """Narrower failure mode than the FlakyMove test above: apply_group can
    raise from is_symlink()/resolve() on the *kept* file, before its
    try/finally is even entered, so no manifest entry exists for that group
    at all. auto_apply_groups must not assume manifest always grew by one --
    otherwise it would misattribute a later group's manifest entry to the
    one that actually failed (or IndexError outright on group 0)."""
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        group0 = make_real_group(directory, n=2, prefix="bad")
        group0.paths[0] = ExplodingIsSymlink(group0.paths[0])
        group1 = make_real_group(directory, n=2, prefix="ok")
        dest_dir = directory / "_duplicates"

        summary = dc.auto_apply_groups([group0, group1], dest_dir, dry_run=False)

        assert summary["failed"] == 1 and summary["confirmed"] == 1
        assert group0.status == "pending"
        assert summary["failures"][0]["group"] == 0
        assert summary["failures"][0]["files_moved"] == 0, (
            "no manifest entry existed for the failed group -- this must read 0, not borrow group1's count"
        )
        assert group1.status == "confirmed"
        assert not group1.paths[1].exists(), "group1 must still succeed normally after group0's early failure"
        print("  ok  a failure before apply_group's move loop stays isolated to its own group, not misattributed")


def make_texture(h: int, w: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(h // 8 + 1, w // 8 + 1, 3), dtype=np.uint8)
    return cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)


def save_jpeg(img: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 85])


def test_recursive_auto_combined() -> None:
    """--recursive and --auto compose: a near-duplicate pair in different
    subdirectories is found, grouped, and auto-applied with dest paths
    mirroring the source subdirectory structure."""
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        rng = np.random.default_rng(99)
        base = rng.integers(0, 255, size=(150, 200, 3), dtype=np.uint8)
        big = cv2.resize(base, (1600, 1200), interpolation=cv2.INTER_CUBIC)
        small = cv2.resize(base, (400, 300), interpolation=cv2.INTER_CUBIC)
        p_big = directory / "vacation" / "big.jpg"
        p_small = directory / "backup" / "small.jpg"
        save_jpeg(big, p_big)
        save_jpeg(small, p_small)
        dest_dir = directory / "_duplicates"

        groups = dc.build_groups(directory, dc.DEFAULT_HASH_THRESHOLD, recursive=True, dest_dir=dest_dir)
        assert len(groups) == 1

        summary = dc.auto_apply_groups(groups, dest_dir, dry_run=False, recursive=True, scan_root=directory)

        assert summary["confirmed"] == 1 and summary["files_moved"] == 1
        kept_still_here = p_big.exists() or p_small.exists()
        moved_to_mirrored_subdir = (dest_dir / "vacation" / "big.jpg").exists() or (
            dest_dir / "backup" / "small.jpg"
        ).exists()
        assert kept_still_here and moved_to_mirrored_subdir, (
            "one file stays at its original nested path, the other lands under the same "
            "mirrored subdirectory inside dest_dir"
        )
        print("  ok  --recursive and --auto compose: cross-subdir group applied with dest mirroring the source")


def test_auto_keeps_the_original_of_a_byte_identical_copy() -> None:
    """End-to-end over the destructive path: build_groups -> auto_apply_groups
    on an original and its copy.

    Every other test here hands auto_apply_groups a Group it built by hand
    with suggested_idx already decided, so none of them can see how
    build_groups decided it. That left the commonest duplicate in a real
    library uncovered: byte-identical copies score exactly equal, the
    tie-break is the whole decision, and filename order got it backwards
    (" copy.jpg" sorts ahead of ".jpg"). --auto doesn't second-guess close
    calls, so it moved the original into _duplicates/ and kept the copy."""
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        rng = np.random.default_rng(31)
        base = rng.integers(0, 255, size=(150, 200, 3), dtype=np.uint8)
        original = directory / "holiday.jpg"
        save_jpeg(cv2.resize(base, (800, 600), interpolation=cv2.INTER_CUBIC), original)
        copy = directory / "holiday copy.jpg"
        shutil.copyfile(original, copy)
        dest_dir = directory / "_duplicates"

        groups = dc.build_groups(directory, dc.DEFAULT_HASH_THRESHOLD, dest_dir=dest_dir)
        assert len(groups) == 1, f"expected one group, got {len(groups)}"
        assert groups[0].is_close_call, "identical copies must read as a close call"

        summary = dc.auto_apply_groups(groups, dest_dir, dry_run=False)

        assert summary["confirmed"] == 1 and summary["files_moved"] == 1, summary
        assert original.exists(), "--auto moved the original away and kept the copy"
        assert not copy.exists() and (dest_dir / "holiday copy.jpg").exists(), (
            "the copy is what should have been moved into dest_dir"
        )
        print("  ok  --auto keeps the original and moves the copy")


def main() -> None:
    tests = [
        test_apply_group_standalone,
        test_apply_group_and_unapply_write_a_move_log,
        test_move_log_failure_does_not_fail_the_move,
        test_auto_dry_run_reports_the_bytes_it_would_move,
        test_run_auto_dry_run_names_every_file,
        test_auto_apply_groups_all_pending,
        test_auto_apply_groups_skips_already_confirmed,
        test_auto_apply_groups_dry_run,
        test_auto_apply_groups_respects_close_call,
        test_auto_summary_bytes,
        test_auto_apply_groups_continues_after_one_group_fails,
        test_auto_apply_groups_isolates_a_failure_before_the_move_loop,
        test_recursive_auto_combined,
        test_auto_keeps_the_original_of_a_byte_identical_copy,
    ]
    for test in tests:
        print(f"{test.__name__}:")
        test()
    print("all auto-mode tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)
