"""Tests for stopping the program from the review page (POST /api/quit).

Two separate claims, one test apiece:

  * the route is wired and token-gated, proved in-process with TestClient
    against a patched duplicates_web.request_exit -- the real one would
    SIGINT the test runner;
  * the 200 reaches the browser *before* the process dies, proved only by a
    real uvicorn in a subprocess. If that response is lost the page never
    draws its goodbye and the user sees a dead review screen instead, which
    is the whole failure this feature exists to avoid.

The subprocess child is test_shutdown.py's: build_groups slowed to a crawl,
so quitting mid-scan with an SSE client attached is what gets measured.

Run: python3 tests/test_quit.py
"""

import json
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import duplicates_core as dc
import duplicates_web as web

TOKEN = "test-token-not-a-secret"
SCAN_SECONDS = 30

CHILD = f"""
import sys, time
sys.path.insert(0, {str(ROOT)!r})
import duplicates_web
_real = duplicates_web.build_groups
def slow(*a, **k):
    time.sleep({SCAN_SECONDS})
    return _real(*a, **k)
duplicates_web.build_groups = slow
sys.argv = ["find_duplicates.py", {str(ROOT / "tests" / "Test-image")!r},
            "--no-browser", "--port", "{{port}}"]
import find_duplicates
find_duplicates.main()
"""


def _make_client(directory: Path):
    params = web.ScanParams(
        directory=directory, threshold=dc.DEFAULT_HASH_THRESHOLD, recursive=False,
        dest_dir=directory / "_duplicates", dry_run=True,
    )
    return TestClient(web.create_app(params, TOKEN))


def _patched_exit():
    """Swap in a recorder for request_exit and hand back the call log. The
    real one signals this very process."""
    calls = []
    original = web.request_exit
    web.request_exit = lambda: calls.append(time.monotonic())
    return calls, original


def test_quit_requires_a_token() -> None:
    """An unauthenticated POST must not stop the server -- the token is the
    only thing standing between a LAN-exposed review and anyone on it."""
    calls, original = _patched_exit()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            with _make_client(Path(tmp)) as client:
                assert client.post("/api/quit").status_code == 401
                assert client.post("/api/quit", params={"token": "wrong"}).status_code == 401
        assert not calls, "an unauthorized request reached request_exit"
    finally:
        web.request_exit = original
        web.shutting_down.clear()
    print("  ok  /api/quit rejects a missing or wrong token without exiting")


def test_quit_answers_then_asks_the_process_to_stop() -> None:
    calls, original = _patched_exit()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            with _make_client(Path(tmp)) as client:
                r = client.post("/api/quit", params={"token": TOKEN})
                assert r.status_code == 200, r.status_code
                assert r.json() == {"status": "stopping"}, r.json()
        assert len(calls) == 1, f"expected exactly one exit request, got {len(calls)}"
    finally:
        web.request_exit = original
        web.shutting_down.clear()
    print("  ok  an authorized /api/quit returns 200 and requests the exit once")


def _run_until_serving(port: int) -> tuple[subprocess.Popen, str]:
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", CHILD.format(port=port)],
        cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    deadline = time.time() + 20
    while time.time() < deadline:
        line = proc.stdout.readline()
        if "Open:" in line:
            time.sleep(1)  # let uvicorn finish binding
            return proc, line.split("token=")[1].strip()
    proc.kill()
    raise AssertionError("server never printed its URL")


def test_quit_stops_a_real_server_after_answering_the_browser() -> None:
    port = 8955
    proc, token = _run_until_serving(port)
    try:
        # The realistic case: the page is open, so its progress stream is
        # holding a request that uvicorn's graceful shutdown would otherwise
        # wait on for the rest of the (deliberately slow) scan.
        stream = urllib.request.urlopen(f"http://127.0.0.1:{port}/api/progress?token={token}")
        assert stream.readline().startswith(b"data:")

        started = time.time()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/quit?token={token}", method="POST", data=b"",
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            assert r.status == 200, r.status
            body = json.loads(r.read())
        assert body == {"status": "stopping"}, body

        proc.wait(timeout=10)
        elapsed = time.time() - started
        output = proc.stdout.read()
        assert elapsed < 5, f"quit took {elapsed:.1f}s -- it waited on the scan or the stream"
        assert proc.returncode == 0, f"exited {proc.returncode}: {output}"
        assert "Traceback" not in output, output
        assert "Stopped from the review page." in output, output
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=10)
    print("  ok  /api/quit answers the browser, then exits 0 mid-scan with a client attached")


def test_the_server_is_really_gone() -> None:
    """Belt and braces on the one thing the user asked for: after a quit the
    port answers nothing at all."""
    port = 8956
    proc, token = _run_until_serving(port)
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/quit?token={token}", method="POST", data=b"",
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        r.read()
    proc.wait(timeout=10)
    proc.stdout.read()
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/api/state?token={token}", timeout=5)
    except (urllib.error.URLError, OSError):
        pass
    else:
        raise AssertionError("the server still answered after quitting")
    print("  ok  nothing answers on the port once the quit has gone through")


def main():
    for test in (test_quit_requires_a_token,
                 test_quit_answers_then_asks_the_process_to_stop,
                 test_quit_stops_a_real_server_after_answering_the_browser,
                 test_the_server_is_really_gone):
        print(f"{test.__name__}:")
        test()
    print("all quit tests passed")


if __name__ == "__main__":
    main()
