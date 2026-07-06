"""Regression test: DuplicateReviewApp._apply must not lose track of files
it already moved if a later move in the same group raises partway through
(disk full, permission error, etc).

Before this fix, the manifest entry for a group was only appended *after*
the whole move loop completed -- so a mid-loop exception meant any files
already relocated to _duplicates/ had no decisions.json record at all,
making them unrecoverable via _unapply. _apply() now appends whatever was
actually moved from inside a `finally`, so a partial failure still leaves
an accurate, reversible manifest entry.

Run: python3 test_manifest_crash_safety.py
"""

import json
import sys
import tempfile
from pathlib import Path

from PIL import Image as PILImage

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


def make_real_group(directory: Path, n: int) -> fd.Group:
    """Unlike most other tests here, this one needs files that actually
    exist on disk -- dry_run=False below means _apply really calls
    shutil.move, so a synthetic non-existent Path would fail for the wrong
    reason."""
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
    group = make_real_group(directory, n=3)
    return fd.DuplicateReviewApp(
        [group], directory / "_duplicates", dry_run=False, manifest_path=directory / "decisions.json"
    )


class FlakyMove:
    """Real move on the 1st call, raises on the 2nd -- simulates a disk-full
    or permission error partway through a multi-file group's move loop."""

    def __init__(self, real_move):
        self.real_move = real_move
        self.calls = 0

    def __call__(self, src, dst):
        self.calls += 1
        if self.calls == 2:
            raise OSError("simulated failure partway through the group")
        return self.real_move(src, dst)


def test_partial_move_failure_is_still_recorded_in_the_manifest() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        app = new_app(directory)
        group = app.groups[0]
        p0, p1, p2 = group.paths  # p0 is kept; p1, p2 would be moved

        flaky = FlakyMove(fd.shutil.move)
        fd.shutil.move = flaky
        try:
            raised = False
            try:
                app._apply(0, keep_idx=0)
            except OSError:
                raised = True
        finally:
            fd.shutil.move = flaky.real_move

        assert raised, "expected the simulated move failure to propagate out of _apply"
        assert group.status == "confirmed", "status is set at the top of _apply before any move is attempted"

        assert len(app.manifest) == 1, f"expected exactly one manifest entry, got {app.manifest}"
        entry = app.manifest[0]
        assert entry["group"] == 0
        assert entry["kept"] == str(p0)
        assert len(entry["moved"]) == 1, (
            f"expected only the file moved before the failure to be recorded, got {entry['moved']}"
        )
        moved_from = Path(entry["moved"][0]["from"])
        moved_to = Path(entry["moved"][0]["to"])
        assert moved_from == p1, "the first file in iteration order should be the one that succeeded"
        assert not moved_from.exists() and moved_to.exists(), "the succeeded move must actually be on disk"
        assert p2.exists(), "the file that failed to move must be left untouched at its original location"

        on_disk = json.loads(app.manifest_path.read_text())
        assert on_disk == app.manifest, "decisions.json must reflect the partial manifest even after the crash"
    print("  ok  a move failure partway through a group still records what was actually moved")


def test_detector_without_finally_would_lose_the_partial_move() -> None:
    """Proof the check above can actually fail: replaying the same failure
    against a replica of the pre-fix _apply() (manifest append only after
    the whole loop completes) must produce an EMPTY manifest, silently
    losing track of the file that really did move to disk."""

    def apply_without_finally(app: fd.DuplicateReviewApp, i: int, keep_idx: int) -> None:
        group = app.groups[i]
        group.status = "confirmed"
        group.current_pick = keep_idx
        moved = []
        for idx, path in enumerate(group.paths):
            if idx == keep_idx:
                continue
            dest = app._dest_for(path)
            if not app.dry_run:
                fd.shutil.move(str(path), str(dest))
            moved.append({"from": str(path), "to": str(dest)})
        app.manifest.append(
            {"group": i, "kept": str(group.paths[keep_idx]), "moved": moved, "dry_run": app.dry_run}
        )
        app._write_manifest()

    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        app = new_app(directory)
        p1 = app.groups[0].paths[1]

        flaky = FlakyMove(fd.shutil.move)
        fd.shutil.move = flaky
        try:
            try:
                apply_without_finally(app, 0, keep_idx=0)
            except OSError:
                pass
        finally:
            fd.shutil.move = flaky.real_move

        assert not p1.exists(), "sanity check: the first move should still have really happened on disk"
        assert app.manifest == [], (
            "expected the pre-fix replica to lose the manifest entry entirely, proving the "
            "try/finally in the real _apply is load-bearing"
        )
    print("  ok  without the try/finally, the same failure would have lost the manifest entry (not vacuous)")


def main() -> None:
    for test in (
        test_partial_move_failure_is_still_recorded_in_the_manifest,
        test_detector_without_finally_would_lose_the_partial_move,
    ):
        print(f"{test.__name__}:")
        test()
    print("all manifest crash-safety tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)
