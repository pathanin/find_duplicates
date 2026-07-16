"""Tests for the FastAPI web front end (duplicates_web.py): token auth on
every endpoint (including image bytes), and the confirm/skip/re-pick
destructive path exercised through real HTTP requests -- the web
equivalents of test_manifest_crash_safety.py, test_unapply_crash_safety.py,
and test_repick_confirmed_group.py, which cover the same invariants for the
TUI. Both front ends share the same duplicates_core primitives
(apply_pick/unapply/pick_needs_reapply), but only a real end-to-end test
through the HTTP layer -- not a unit test of those primitives in isolation
-- would have caught the is_close_call/numpy.bool_ JSON-serialization bug
this file also guards against (found via live browser testing, see the
"Fix is_close_call not being JSON-serializable" commit).

Requires httpx < 0.28: FastAPI's TestClient (starlette 0.35.x, current at
the time this was written) constructs its underlying httpx.Client with the
`app=` shortcut, which httpx 0.28 removed. `pip install "httpx<0.28"` if
this file fails with `TypeError: Client.__init__() got an unexpected
keyword argument 'app'` -- that's an httpx/starlette version mismatch, not
a real bug. httpx itself is test-only, not a runtime dependency of the web
front end (contrib/install.sh does not install it).

Run: python3 tests/test_web_api.py
"""

import shutil
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


def make_texture(h: int, w: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(h // 8 + 1, w // 8 + 1, 3), dtype=np.uint8)
    return cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)


def make_duplicate_set(directory: Path, seed: int, n: int) -> list[Path]:
    """n differently-sized exports of the same base texture, same JPEG
    quality throughout -- same shape as test_fast_scan.py's
    make_duplicate_pair, generalized to n copies, so a real build_groups()
    scan reliably groups them together under DEFAULT_HASH_THRESHOLD."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(150, 200, 3), dtype=np.uint8)
    sizes = [(1600, 1200), (1200, 900), (800, 600), (400, 300), (200, 150)]
    paths = []
    for i in range(n):
        w, h = sizes[i % len(sizes)]
        img = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)
        p = directory / f"dup_{seed}_{i}.jpg"
        cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        paths.append(p)
    return paths


def make_close_call_pair(directory: Path, seed: int) -> list[Path]:
    """Two exports at identical size/quality from the same base texture --
    their metrics land in score_group's hi==lo degenerate range (tied
    scores), guaranteeing is_close_call=True. Used to prove the API
    actually serializes a close-call group correctly (regression test for
    the numpy.bool_ bug)."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(300, 400, 3), dtype=np.uint8)
    paths = []
    for i in range(2):
        p = directory / f"close_{seed}_{i}.jpg"
        cv2.imwrite(str(p), base, [cv2.IMWRITE_JPEG_QUALITY, 90])
        paths.append(p)
    return paths


def _wait_ready(client: TestClient, timeout_s: float = 30.0) -> dict:
    deadline = time.monotonic() + timeout_s
    data = None
    while time.monotonic() < deadline:
        r = client.get("/api/state", params={"token": TOKEN})
        data = r.json()
        if data["status"] not in ("scanning", "idle"):
            return data
        time.sleep(0.05)
    raise AssertionError(f"scan did not finish in {timeout_s}s, last status: {data}")


def _make_client(directory: Path, dest_dir: Path, dry_run: bool = True):
    params = web.ScanParams(
        directory=directory, threshold=dc.DEFAULT_HASH_THRESHOLD, recursive=False,
        dest_dir=dest_dir, dry_run=dry_run,
    )
    app = web.create_app(params, TOKEN)
    return TestClient(app), app


def test_data_endpoint_requires_token() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        client, _ = _make_client(directory, directory / "_duplicates")
        with client:
            r = client.get("/api/state")
            assert r.status_code == 401, f"expected 401 with no token, got {r.status_code}"
            r = client.get("/api/state", params={"token": "wrong"})
            assert r.status_code == 401, f"expected 401 with wrong token, got {r.status_code}"
        print("  ok  /api/state rejects missing/wrong token")


def test_image_endpoint_requires_token() -> None:
    """Explicit PLAN requirement: token gating covers image bytes too, not
    just the JSON API -- an <img src> is exactly the kind of request a
    casual scan of "protected endpoints" might miss."""
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        make_duplicate_set(directory, seed=1, n=2)
        client, _ = _make_client(directory, directory / "_duplicates")
        with client:
            _wait_ready(client)
            r = client.get("/api/thumb/0/0")
            assert r.status_code == 401, f"expected 401 with no token on an image endpoint, got {r.status_code}"
        print("  ok  /api/thumb rejects requests with no token")


def test_index_sets_cookie_and_cookie_alone_authenticates() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        client, _ = _make_client(directory, directory / "_duplicates")
        with client:
            r = client.get("/", params={"token": TOKEN})
            assert r.status_code == 200
            assert web.COOKIE_NAME in client.cookies, "expected GET / to set the auth cookie"
            r = client.get("/api/state")  # no query token, cookie should carry auth
            assert r.status_code == 200, f"expected cookie-only auth to work, got {r.status_code}"
        print("  ok  GET / sets an auth cookie that alone authenticates later requests")


def test_initial_scan_populates_groups() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        make_duplicate_set(directory, seed=2, n=3)
        client, _ = _make_client(directory, directory / "_duplicates")
        with client:
            data = _wait_ready(client)
            assert data["status"] == "ready"
            assert len(data["groups"]) == 1, f"expected the 3 near-duplicates as one group, got {data['groups']}"
            assert data["groups"][0]["file_count"] == 3
        print("  ok  the CLI-seeded scan runs on startup and populates groups")


def test_group_and_thumb_404_for_out_of_range_index() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        make_duplicate_set(directory, seed=3, n=2)
        client, _ = _make_client(directory, directory / "_duplicates")
        with client:
            _wait_ready(client)
            assert client.get("/api/group/99", params={"token": TOKEN}).status_code == 404
            assert client.get("/api/thumb/0/99", params={"token": TOKEN}).status_code == 404
        print("  ok  out-of-range group/file indices 404 instead of crashing")


def test_scan_rejects_invalid_directory_and_out_of_range_threshold() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        client, _ = _make_client(directory, directory / "_duplicates")
        with client:
            _wait_ready(client)
            r = client.post(
                "/api/scan", params={"token": TOKEN},
                json={"directory": str(directory / "does-not-exist")},
            )
            assert r.status_code == 400, f"expected 400 for a nonexistent directory, got {r.status_code}"
            r = client.post(
                "/api/scan", params={"token": TOKEN},
                json={"directory": str(directory), "threshold": 999},
            )
            assert r.status_code == 422, f"expected 422 for an out-of-range threshold, got {r.status_code}"
        print("  ok  POST /api/scan validates directory existence and threshold range")


def test_close_call_group_serializes_without_500() -> None:
    """Regression test for the numpy.bool_/JSON bug: is_close_call must be
    a real JSON boolean, not something json.dumps chokes on, for both
    /api/state's summaries and /api/group/{i}'s detail."""
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        make_close_call_pair(directory, seed=4)
        client, _ = _make_client(directory, directory / "_duplicates")
        with client:
            data = _wait_ready(client)
            assert data["status"] == "ready", data
            assert len(data["groups"]) == 1
            assert data["groups"][0]["is_close_call"] is True, (
                f"expected the identical-quality pair to be a close call, got {data['groups'][0]}"
            )
            r = client.get("/api/group/0", params={"token": TOKEN})
            assert r.status_code == 200, f"expected group detail to serialize cleanly, got {r.status_code}: {r.text}"
            assert r.json()["is_close_call"] is True
        print("  ok  a close-call group serializes cleanly through /api/state and /api/group/{i}")


def test_confirm_moves_files_and_skip_unapplies() -> None:
    """dry_run=False: confirm must really move the non-kept file to
    dest_dir, and a subsequent skip must restore it -- mirrors
    DuplicateReviewApp.action_confirm/action_skip's real-filesystem
    behavior, via the same duplicates_core primitives."""
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        dest_dir = directory / "_duplicates"
        paths = make_duplicate_set(directory, seed=5, n=2)
        client, app = _make_client(directory, dest_dir, dry_run=False)
        with client:
            _wait_ready(client)
            group = app.state.session.groups[0]
            keep_idx = group.current_pick
            other_idx = 1 - keep_idx
            kept_path, moved_path = group.paths[keep_idx], group.paths[other_idx]

            r = client.post(f"/api/group/0/pick", params={"token": TOKEN}, json={"idx": keep_idx})
            assert r.status_code == 200
            r = client.post("/api/group/0/confirm", params={"token": TOKEN})
            assert r.status_code == 200 and r.json()["status"] == "confirmed"
            assert kept_path.exists(), "kept file must stay at its original location"
            assert not moved_path.exists(), "non-kept file must be moved away from its original location"
            assert any(dest_dir.rglob(moved_path.name)), "non-kept file must land somewhere under dest_dir"

            r = client.post("/api/group/0/skip", params={"token": TOKEN})
            assert r.status_code == 200 and r.json()["status"] == "skipped"
            assert moved_path.exists(), "skip must unapply the confirm, restoring the moved file"
            assert not any(dest_dir.rglob(moved_path.name)), "restored file must no longer be under dest_dir"
        print("  ok  confirm moves the non-kept file; skip unapplies and restores it")


def test_repick_after_confirm_moves_the_new_non_kept_file() -> None:
    """Confirm keep=A, then re-pick to keep=B, then confirm again: A must
    come back to its original location and B's former sibling (now A) must
    be the one that moves -- mirrors test_repick_confirmed_group.py."""
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        dest_dir = directory / "_duplicates"
        paths = make_duplicate_set(directory, seed=6, n=2)
        client, app = _make_client(directory, dest_dir, dry_run=False)
        with client:
            _wait_ready(client)
            group = app.state.session.groups[0]
            p0, p1 = group.paths

            client.post("/api/group/0/pick", params={"token": TOKEN}, json={"idx": 0})
            r = client.post("/api/group/0/confirm", params={"token": TOKEN})
            assert r.status_code == 200
            assert p0.exists() and not p1.exists(), "expected p0 kept, p1 moved after first confirm"

            client.post("/api/group/0/pick", params={"token": TOKEN}, json={"idx": 1})
            r = client.post("/api/group/0/confirm", params={"token": TOKEN})
            assert r.status_code == 200, r.text
            assert p1.exists() and not p0.exists(), (
                "expected re-confirm to restore p1 and move p0 instead"
            )
        print("  ok  re-picking an already-confirmed group and re-confirming swaps which file is kept")


def test_confirm_partial_failure_leaves_group_pending() -> None:
    """The web equivalent of test_manifest_crash_safety.py: a move that
    raises partway through a multi-file group must leave the group
    "pending" (not silently marked confirmed) and the manifest must still
    record exactly what was actually moved before the failure."""
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        dest_dir = directory / "_duplicates"
        make_duplicate_set(directory, seed=7, n=3)
        client, app = _make_client(directory, dest_dir, dry_run=False)

        class FlakyMove:
            def __init__(self, real_move):
                self.real_move = real_move
                self.calls = 0

            def __call__(self, src, dst):
                self.calls += 1
                if self.calls == 2:
                    raise OSError("simulated failure partway through the group")
                return self.real_move(src, dst)

        with client:
            _wait_ready(client)
            real_move = shutil.move
            flaky = FlakyMove(real_move)
            shutil.move = flaky
            try:
                r = client.post("/api/group/0/confirm", params={"token": TOKEN})
            finally:
                shutil.move = real_move

            assert r.status_code == 500, f"expected the simulated failure to surface as a 500, got {r.status_code}"
            group = app.state.session.groups[0]
            assert group.status == "pending", (
                f"group must stay pending after a partial move failure, got {group.status!r}"
            )
            entry = next((m for m in app.state.session.manifest if m["group"] == 0), None)
            assert entry is not None, "the partial move must still be recorded in the manifest"
            assert len(entry["moved"]) == 1, (
                f"expected exactly the one move that succeeded before the failure, got {entry['moved']}"
            )
        print("  ok  a partial move failure leaves the group pending with an accurate manifest entry")


def test_scanning_status_blocks_mutating_endpoints() -> None:
    """A rescan replaces session.groups/manifest wholesale on completion --
    a pick/confirm/skip that lands mid-scan must be rejected (409), not
    silently mutate a Group that's about to be discarded. See the
    _require_not_scanning guard in duplicates_web.py."""
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        make_duplicate_set(directory, seed=8, n=2)
        client, app = _make_client(directory, directory / "_duplicates")
        with client:
            _wait_ready(client)
            session = app.state.session
            session.status = "scanning"
            try:
                assert client.post(
                    "/api/group/0/pick", params={"token": TOKEN}, json={"idx": 0}
                ).status_code == 409
                assert client.post("/api/group/0/confirm", params={"token": TOKEN}).status_code == 409
                assert client.post("/api/group/0/skip", params={"token": TOKEN}).status_code == 409
            finally:
                session.status = "ready"
        print("  ok  pick/confirm/skip all reject a request while a scan is in flight")


def test_rescan_bumps_generation_and_resets_group_status() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        make_duplicate_set(directory, seed=9, n=2)
        client, app = _make_client(directory, directory / "_duplicates")
        with client:
            data = _wait_ready(client)
            assert data["generation"] == 1
            client.post("/api/group/0/skip", params={"token": TOKEN})

            r = client.post("/api/scan", params={"token": TOKEN}, json={"directory": str(directory)})
            assert r.status_code == 200
            data = _wait_ready(client)
            assert data["generation"] == 2, f"expected generation to bump on a successful rescan, got {data}"
            assert data["groups"][0]["status"] == "pending", "a rescan must produce a fresh, unreviewed group set"
        print("  ok  a rescan bumps generation and resets group status")


def main() -> None:
    tests = [
        test_data_endpoint_requires_token,
        test_image_endpoint_requires_token,
        test_index_sets_cookie_and_cookie_alone_authenticates,
        test_initial_scan_populates_groups,
        test_group_and_thumb_404_for_out_of_range_index,
        test_scan_rejects_invalid_directory_and_out_of_range_threshold,
        test_close_call_group_serializes_without_500,
        test_confirm_moves_files_and_skip_unapplies,
        test_repick_after_confirm_moves_the_new_non_kept_file,
        test_confirm_partial_failure_leaves_group_pending,
        test_scanning_status_blocks_mutating_endpoints,
        test_rescan_bumps_generation_and_resets_group_status,
    ]
    for test in tests:
        print(f"{test.__name__}:")
        test()
    print("all web-api tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)
