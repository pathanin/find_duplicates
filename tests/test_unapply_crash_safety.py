"""Regression tests for DuplicateReviewApp._unapply (the reverse of _apply,
used when a confirmed group is re-picked or un-confirmed back to skipped).

_apply already had crash-safety tests (test_manifest_crash_safety.py):
a move failure partway through a group's forward move must not lose track
of files already relocated to _duplicates/. _unapply never got the matching
coverage for the reverse direction, and had two real bugs:

1. It used Path.rename() to move files back, not shutil.move(). --dest can
   point at a different filesystem than the scanned directory (an explicit
   CLI feature), and plain rename() raises OSError (EXDEV) cross-device
   where shutil.move() falls back to copy+remove. _apply already uses
   shutil.move() for exactly this reason; _unapply didn't match it, so
   restoring a pick after choosing a different one could hard-fail whenever
   --dest was cross-device.

2. Its finally block removed the whole manifest entry unconditionally, even
   if the restore loop only partially succeeded before raising. A partial
   failure meant losing decisions.json's only record of files still sitting
   in dest_dir/ -- silently breaking the "never delete, always track in
   decisions.json" invariant on the one path meant to guard it.

Run: python3 test_unapply_crash_safety.py
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
    or permission error partway through a multi-file group's restore loop."""

    def __init__(self, real_move):
        self.real_move = real_move
        self.calls = 0

    def __call__(self, src, dst):
        self.calls += 1
        if self.calls == 2:
            raise OSError("simulated failure partway through the restore")
        return self.real_move(src, dst)


def test_full_restore_removes_the_manifest_entry() -> None:
    """Baseline: a clean, fully-successful _unapply still removes the
    manifest entry entirely (no regression on the common case)."""
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        app = new_app(directory)
        p0, p1, p2 = app.groups[0].paths

        app._apply(0, keep_idx=0)
        assert len(app.manifest) == 1
        assert not p1.exists() and not p2.exists()

        app._unapply(0)
        assert app.manifest == [], "a fully-successful restore must remove the manifest entry"
        assert p1.exists() and p2.exists(), "both moved files must be back at their original paths"
        on_disk = json.loads(app.manifest_path.read_text())
        assert on_disk == [], "decisions.json must reflect the now-empty manifest"
    print("  ok  a clean restore still removes the manifest entry (no regression)")


def test_partial_restore_failure_keeps_the_unrestored_file_tracked() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        app = new_app(directory)
        p0, p1, p2 = app.groups[0].paths

        app._apply(0, keep_idx=0)
        assert len(app.manifest) == 1
        entry_before = app.manifest[0]
        moved_from_names = [Path(m["from"]).name for m in entry_before["moved"]]
        assert moved_from_names == [p1.name, p2.name]

        flaky = FlakyMove(fd.shutil.move)
        fd.shutil.move = flaky
        try:
            raised = False
            try:
                app._unapply(0)
            except OSError:
                raised = True
        finally:
            fd.shutil.move = flaky.real_move

        assert raised, "expected the simulated restore failure to propagate out of _unapply"
        assert p1.exists(), "the file whose restore succeeded before the failure must be back on disk"
        assert not p2.exists(), "the file whose restore failed must still be sitting in dest_dir/"

        assert len(app.manifest) == 1, (
            f"a partial restore failure must NOT drop the manifest entry entirely, got {app.manifest}"
        )
        entry_after = app.manifest[0]
        assert entry_after["group"] == 0
        assert len(entry_after["moved"]) == 1, (
            f"only the file still sitting in dest_dir/ should remain tracked, got {entry_after['moved']}"
        )
        assert Path(entry_after["moved"][0]["from"]).name == p2.name

        on_disk = json.loads(app.manifest_path.read_text())
        assert on_disk == app.manifest, "decisions.json must reflect the partial manifest even after the crash"
    print("  ok  a restore failure partway through a group still tracks the file left in dest_dir/")


def test_detector_without_the_fix_would_lose_the_untracked_file() -> None:
    """Proof the check above can actually fail: replaying the same failure
    against a replica of the pre-fix _unapply (manifest entry removed
    unconditionally in `finally`) must drop tracking of the file that's
    still sitting in dest_dir/, unable to be moved back."""

    def unapply_pre_fix(app: fd.DuplicateReviewApp, i: int) -> None:
        entry = next((m for m in app.manifest if m["group"] == i), None)
        if not entry:
            return
        restored = []
        try:
            for moved in entry["moved"]:
                src = Path(moved["from"])
                dst = Path(moved["to"])
                if dst.exists() and not src.exists():
                    fd.shutil.move(str(dst), str(src))
                    restored.append(moved)
        finally:
            if restored:
                entry["moved"] = restored
            app.manifest.remove(entry)  # unconditional -- this is the bug
            app._write_manifest()

    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        app = new_app(directory)
        p1, p2 = app.groups[0].paths[1], app.groups[0].paths[2]
        app._apply(0, keep_idx=0)

        flaky = FlakyMove(fd.shutil.move)
        fd.shutil.move = flaky
        try:
            try:
                unapply_pre_fix(app, 0)
            except OSError:
                pass
        finally:
            fd.shutil.move = flaky.real_move

        assert p1.exists(), "sanity check: the first restore should still have really happened on disk"
        assert not p2.exists(), "sanity check: the second file should still be stuck in dest_dir/"
        assert app.manifest == [], (
            "expected the pre-fix replica to lose the manifest entry entirely, proving the "
            "remaining-vs-restored bookkeeping in the real _unapply is load-bearing"
        )
    print("  ok  without the fix, the same failure would have lost track of the file in dest_dir/ (not vacuous)")


def main() -> None:
    for test in (
        test_full_restore_removes_the_manifest_entry,
        test_partial_restore_failure_keeps_the_unrestored_file_tracked,
        test_detector_without_the_fix_would_lose_the_untracked_file,
    ):
        print(f"{test.__name__}:")
        test()
    print("all unapply crash-safety tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)
