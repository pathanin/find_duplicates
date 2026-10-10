"""Keeping more than one file per group (Group.also_keep).

A group can hold a real duplicate *and* a false positive -- a similar but
different photo the hash pulled in. Before also_keep the only choices were
"keep one, move the rest" or skip the whole group, which left the real
duplicates to be hunted down by hand. Pinned indices sit alongside
current_pick: the kept set is {current_pick} | also_keep, and pins are
independent of the pick so flipping the stage never drops one.

What this locks in:
  - apply_group never moves a pinned file, and records the pins in the
    manifest (also_kept) so a confirmed group can tell when they changed;
  - pick_needs_reapply compares the whole kept *set*: a pin change on a
    confirmed group needs a reapply, but promoting an already-pinned file
    to primary moves nothing and so must not trigger an unapply/reapply;
  - the symlink guard covers every kept file, not just the primary --
    moving the target of a pinned symlink would leave it dangling;
  - POST /api/group/{i}/keep carries the same guards as /pick (bounds,
    stale ?g=, non-streaming scan) and a confirm after it moves only the
    unpinned files.

Run: python3 tests/test_keep_multiple.py
"""

import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import duplicates_core as dc
import duplicates_web as web

TOKEN = "test-token-not-a-secret"


def _fake_group(paths: list[Path], pick: int = 0, also_keep: list[int] | None = None) -> dc.Group:
    return dc.Group(
        paths=paths, results=[{"file_size": 1} for _ in paths], thumbnails=None,
        suggested_idx=0, current_pick=pick, is_close_call=False,
        also_keep=list(also_keep or []),
    )


def _touch(directory: Path, names: list[str]) -> list[Path]:
    out = []
    for n in names:
        p = directory / n
        p.write_bytes(n.encode())
        out.append(p)
    return out


def test_apply_group_leaves_pinned_files_in_place() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        paths = _touch(d, ["a.jpg", "b.jpg", "c.jpg", "d.jpg"])
        g = _fake_group(paths, pick=0, also_keep=[2])
        manifest: list[dict] = []
        dc.apply_pick(g, 0, 0, d / "_dup", False, manifest)
        assert paths[0].exists() and paths[2].exists(), "primary and pinned files must stay"
        assert not paths[1].exists() and not paths[3].exists(), "unpinned files must move"
        moved = sorted(Path(m["from"]).name for m in manifest[0]["moved"])
        assert moved == ["b.jpg", "d.jpg"], moved
        assert manifest[0]["also_kept"] == [str(paths[2])], manifest[0]
    print("  ok  apply_group moves only files outside {pick} | also_keep")


def test_no_pins_behaves_exactly_as_before() -> None:
    """Both sides of the boundary: without pins every non-pick file moves."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        paths = _touch(d, ["a.jpg", "b.jpg", "c.jpg"])
        g = _fake_group(paths)
        manifest: list[dict] = []
        dc.apply_pick(g, 0, 0, d / "_dup", False, manifest)
        assert [p.exists() for p in paths] == [True, False, False]
        assert manifest[0]["also_kept"] == []
    print("  ok  an unpinned group still keeps exactly one file")


def test_pin_change_on_confirmed_group_needs_reapply() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        paths = _touch(d, ["a.jpg", "b.jpg", "c.jpg"])
        g = _fake_group(paths)
        manifest: list[dict] = []
        dc.apply_pick(g, 0, 0, d / "_dup", False, manifest)
        assert not dc.pick_needs_reapply(manifest, 0, g), "freshly confirmed: nothing to reapply"
        g.also_keep = [1]
        assert dc.pick_needs_reapply(manifest, 0, g), "adding a pin after confirm must need a reapply"
        dc.unapply(manifest, 0)
        dc.apply_pick(g, 0, 0, d / "_dup", False, manifest)
        assert [p.exists() for p in paths] == [True, True, False], "reapply must restore the newly pinned file"
        g.also_keep = []
        assert dc.pick_needs_reapply(manifest, 0, g), "removing a pin after confirm must need a reapply"
    print("  ok  adding or removing a pin on a confirmed group needs a reapply")


def test_swapping_primary_within_kept_set_needs_no_reapply() -> None:
    """Pinned {0, 2} with pick 0, then pick 2 with 0 pinned: the same two
    files stay, so re-confirming must not unapply and reapply anything."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        paths = _touch(d, ["a.jpg", "b.jpg", "c.jpg"])
        g = _fake_group(paths, pick=0, also_keep=[2])
        manifest: list[dict] = []
        dc.apply_pick(g, 0, 0, d / "_dup", False, manifest)
        g.current_pick = 2
        g.also_keep = [0]
        assert not dc.pick_needs_reapply(manifest, 0, g)
        g.also_keep = []
        assert dc.pick_needs_reapply(manifest, 0, g), "dropping 0 from the kept set must still be detected"
    print("  ok  a different primary over the same kept set is not a reapply")


def test_legacy_manifest_entry_without_also_kept() -> None:
    g = _fake_group([Path("/x/a.jpg"), Path("/x/b.jpg")])
    manifest = [{"group": 0, "kept": "/x/a.jpg", "moved": [], "dry_run": True}]
    assert not dc.pick_needs_reapply(manifest, 0, g)
    print("  ok  a manifest entry without also_kept reads as no pins")


def test_symlink_to_pinned_target_is_not_moved() -> None:
    """b.jpg is pinned and is a symlink to c.jpg. Moving c.jpg would leave
    the kept b.jpg dangling, so c.jpg must stay even though it's unpinned."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        a, c = _touch(d, ["a.jpg", "c.jpg"])
        b = d / "b.jpg"
        b.symlink_to(c)
        g = _fake_group([a, b, c], pick=0, also_keep=[1])
        manifest: list[dict] = []
        dc.apply_pick(g, 0, 0, d / "_dup", False, manifest)
        assert b.exists() and c.exists(), "a pinned symlink's target must not be moved out from under it"
        assert manifest[0]["moved"] == []
    print("  ok  the symlink guard covers pinned files, not just the primary")


# --- HTTP layer -------------------------------------------------------------

def _make_dup_set(directory: Path, seed: int, n: int) -> None:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(150, 200, 3), dtype=np.uint8)
    sizes = [(1600, 1200), (1200, 900), (800, 600), (400, 300)]
    for i in range(n):
        w, h = sizes[i % len(sizes)]
        img = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(str(directory / f"dup_{seed}_{i}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])


def _client(directory: Path):
    params = web.ScanParams(
        directory=directory, threshold=dc.DEFAULT_HASH_THRESHOLD, recursive=False,
        dest_dir=directory / "_duplicates", dry_run=False,
    )
    app = web.create_app(params, TOKEN)
    return TestClient(app), app


def _wait_ready(client: TestClient) -> dict:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        data = client.get("/api/state", params={"token": TOKEN}).json()
        if data["status"] not in ("scanning", "idle"):
            return data
        time.sleep(0.05)
    raise AssertionError("scan did not finish")


def test_keep_route_then_confirm_moves_only_unpinned() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        _make_dup_set(directory, seed=21, n=3)
        client, app = _client(directory)
        with client:
            data = _wait_ready(client)
            gen = data["generation"]
            g = app.state.session.groups[0]
            assert len(g.paths) == 3, f"fixture must group all three, got {len(g.paths)}"
            p0, p1, p2 = g.paths

            r = client.post("/api/group/0/keep", params={"token": TOKEN, "g": gen}, json={"idx": 2, "keep": True})
            assert r.status_code == 200, r.text
            assert r.json()["also_keep"] == [2]
            assert client.get("/api/state", params={"token": TOKEN}).json()["groups"][0]["also_keep"] == [2]

            r = client.post("/api/group/0/confirm", params={"token": TOKEN, "g": gen})
            assert r.status_code == 200 and r.json()["status"] == "confirmed", r.text
            assert p0.exists() and p2.exists() and not p1.exists()
            assert r.json()["needs_reapply"] is False

            r = client.post("/api/group/0/keep", params={"token": TOKEN}, json={"idx": 2, "keep": False})
            assert r.json()["also_keep"] == [] and r.json()["needs_reapply"] is True
            r = client.post("/api/group/0/confirm", params={"token": TOKEN})
            assert r.status_code == 200
            assert p0.exists() and not p1.exists() and not p2.exists(), "unpinned then reconfirmed: p2 must move too"

            r = client.post("/api/group/0/keep", params={"token": TOKEN}, json={"idx": 2, "keep": True})
            r = client.post("/api/group/0/skip", params={"token": TOKEN})
            assert all(p.exists() for p in (p0, p1, p2)), "skip must restore everything"
            assert r.json()["restored"] == 2, (
                f"skip must report what the manifest moved (2), not paths minus today's pins (1): {r.json()['restored']}"
            )
        print("  ok  /keep pins a file, confirm leaves it, unpin + reconfirm moves it, skip restores all")


def test_keep_route_guards() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        _make_dup_set(directory, seed=22, n=2)
        client, app = _client(directory)
        with client:
            gen = _wait_ready(client)["generation"]
            body = {"idx": 1, "keep": True}
            assert client.post("/api/group/0/keep", params={"token": TOKEN}, json={"idx": 5, "keep": True}).status_code == 400
            assert client.post("/api/group/9/keep", params={"token": TOKEN}, json=body).status_code == 404
            assert client.post("/api/group/0/keep", params={"token": TOKEN, "g": gen - 1}, json=body).status_code == 409
            assert client.post("/api/group/0/keep", json=body).status_code == 401
            session = app.state.session
            session.status = "scanning"
            try:
                assert client.post("/api/group/0/keep", params={"token": TOKEN}, json=body).status_code == 409
            finally:
                session.status = "ready"
            assert session.groups[0].also_keep == [], "no rejected request may have pinned anything"
            r = client.post("/api/group/0/keep", params={"token": TOKEN, "g": gen}, json=body)
            assert r.status_code == 200 and r.json()["also_keep"] == [1], "the same request with valid guards succeeds"
        print("  ok  /keep rejects bad idx/group/stale g/no token/mid-rescan, accepts a valid one")


def main() -> None:
    tests = [
        test_apply_group_leaves_pinned_files_in_place,
        test_no_pins_behaves_exactly_as_before,
        test_pin_change_on_confirmed_group_needs_reapply,
        test_swapping_primary_within_kept_set_needs_no_reapply,
        test_legacy_manifest_entry_without_also_kept,
        test_symlink_to_pinned_target_is_not_moved,
        test_keep_route_then_confirm_moves_only_unpinned,
        test_keep_route_guards,
    ]
    for test in tests:
        print(f"{test.__name__}:")
        test()
    print("all keep-multiple tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)
