"""Regression tests for Ctrl-C shutdown.

Two past failures: a SIGINT during a scan hung until the whole scan
finished (build_groups ran in the loop's default executor, and asyncio's
teardown joins those threads), and a quick second Ctrl-C during shutdown
printed an ERROR traceback (uvicorn re-raises the signal through
asyncio.run's own SIGINT handler, which cancels the lifespan task
mid-shutdown).

Spawns real subprocesses -- the bug only exists in the signal/loop
teardown that an in-process test never runs.

Run: python3 test_shutdown.py
"""

import pathlib
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Stands in for a big library: the scan blocks long enough that a shutdown
# waiting on it is unmistakable.
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


def run_until_serving(port: int) -> tuple[subprocess.Popen, str]:
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


def stop(proc: subprocess.Popen, extra_sigint_after: float | None = None) -> tuple[float, str]:
    started = time.time()
    proc.send_signal(signal.SIGINT)
    if extra_sigint_after is not None:
        time.sleep(extra_sigint_after)
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
    output = proc.stdout.read()
    proc.wait(timeout=SCAN_SECONDS + 10)
    return time.time() - started, output


def test_sigint_during_scan_exits_promptly():
    proc, _ = run_until_serving(8951)
    elapsed, output = stop(proc)
    assert elapsed < 5, f"Ctrl-C waited {elapsed:.1f}s on the in-flight scan"
    assert "Traceback" not in output, output
    print("  ok  a single Ctrl-C mid-scan exits without waiting for the scan")


def test_second_sigint_during_shutdown_is_quiet():
    proc, _ = run_until_serving(8952)
    elapsed, output = stop(proc, extra_sigint_after=0.05)
    assert elapsed < 5, f"double Ctrl-C took {elapsed:.1f}s"
    assert "Traceback" not in output, output
    print("  ok  a second Ctrl-C during shutdown prints no traceback")


def test_sigint_with_an_open_sse_stream_exits_promptly():
    """The realistic Ctrl-C: a browser sits on the review page holding
    /api/progress open. That stream is an in-flight task, and uvicorn's
    graceful shutdown waits on those with no timeout by default."""
    proc, token = run_until_serving(8953)
    stream = urllib.request.urlopen(f"http://127.0.0.1:8953/api/progress?token={token}")
    assert stream.readline().startswith(b"data:")
    elapsed, output = stop(proc)
    assert elapsed < 5, f"Ctrl-C waited {elapsed:.1f}s on the open SSE stream"
    assert "Traceback" not in output, output
    print("  ok  a Ctrl-C with an SSE client attached exits without waiting on it")


def test_startup_lines_survive_a_redirect():
    """os._exit skips the interpreter's buffer flush, and stdout to a file
    is block-buffered -- without an explicit flush both startup lines
    (including the tokened URL) vanish from a headless box's log."""
    with tempfile.NamedTemporaryFile("r", suffix=".log") as log:
        proc = subprocess.Popen(
            [sys.executable, "-c", CHILD.format(port=8954)],
            cwd=str(ROOT), stdout=log.file, stderr=subprocess.STDOUT, text=True,
        )
        time.sleep(3)
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=SCAN_SECONDS + 10)
        written = pathlib.Path(log.name).read_text()
    assert "Open:" in written, f"startup output lost on exit: {written!r}"
    print("  ok  the startup lines are flushed even when stdout is a redirect")


def main():
    for test in (test_sigint_during_scan_exits_promptly,
                 test_second_sigint_during_shutdown_is_quiet,
                 test_sigint_with_an_open_sse_stream_exits_promptly,
                 test_startup_lines_survive_a_redirect):
        print(f"{test.__name__}:")
        test()
    print("all shutdown tests passed")


if __name__ == "__main__":
    main()
