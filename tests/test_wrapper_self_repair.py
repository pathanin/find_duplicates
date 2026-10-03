"""Locks in the `find-duplicates` wrapper rebuilding a dead venv on its own.

A venv is tied to one Python minor version: its compiled wheels (numpy,
opencv, pillow) only load in the interpreter they were built for. Once
Homebrew removes that interpreter, every run died with "venv/bin/python3:
No such file or directory" until the user found and reran install.sh. The
wrapper now notices the missing interpreter, reruns libexec/setup_venv.sh,
and carries on.

Renders the real wrapper from install.sh's heredoc, then runs it against a
fake DATA_DIR whose setup_venv.sh is a stub -- no network, no pip.

Run: python3 tests/test_wrapper_self_repair.py
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INSTALL = ROOT / "install.sh"

# Stands in for setup_venv.sh: logs the call, then builds a "venv" whose
# python3 just echoes its arguments. FAIL=1 makes it fail like a dead network.
STUB_SETUP = """#!/bin/sh
echo setup >> "$LOG"
[ "${FAIL:-0}" = 1 ] && exit 1
rm -rf "$1"
mkdir -p "$1/bin"
printf '#!/bin/sh\\necho ran "$@"\\n' > "$1/bin/python3"
chmod +x "$1/bin/python3"
touch "$1/.complete"
"""


def render_wrapper(data_dir: Path, venv_dir: Path) -> Path:
    m = re.search(r'^cat > "\$WRAPPER" <<EOS\n(.*?)^EOS\n', INSTALL.read_text(), re.M | re.S)
    assert m, "install.sh has no wrapper heredoc"
    script = f'DATA_DIR="{data_dir}"; VENV_DIR="{venv_dir}"\ncat <<EOS\n{m.group(1)}EOS\n'
    body = subprocess.run(["sh", "-c", script], capture_output=True, text=True, check=True).stdout
    wrapper = data_dir / "find-duplicates"
    wrapper.write_text(body)
    wrapper.chmod(0o755)
    return wrapper


def run(wrapper: Path, log: Path, fail: bool = False) -> subprocess.CompletedProcess:
    env = {"PATH": "/usr/bin:/bin", "LOG": str(log), "FAIL": "1" if fail else "0"}
    return subprocess.run([str(wrapper), "--help"], capture_output=True, text=True, env=env)


def setup_calls(log: Path) -> int:
    return len(log.read_text().splitlines()) if log.exists() else 0


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp)
        venv = data / "venv"
        log = data / "log"
        (data / "libexec").mkdir()
        (data / "libexec/setup_venv.sh").write_text(STUB_SETUP)
        wrapper = render_wrapper(data, venv)

        # Dead venv: python3 is a dangling link, as after a Homebrew removal.
        (venv / "bin").mkdir(parents=True)
        (venv / "bin/python3").symlink_to(data / "gone/python3")
        (venv / ".complete").touch()

        r = run(wrapper, log, fail=True)
        assert r.returncode != 0, r
        assert "ran" not in r.stdout, r.stdout
        assert "install.sh" in r.stderr, r.stderr
        print("ok  a failed rebuild exits non-zero and names install.sh")

        r = run(wrapper, log)
        assert r.returncode == 0, r
        assert r.stdout.strip() == f"ran {data}/libexec/find_duplicates.py --help", r.stdout
        assert r.stderr, "rebuild should say why the first run is slow"
        print("ok  dangling python triggers a rebuild, then runs the tool")

        calls = setup_calls(log)
        r = run(wrapper, log)
        assert r.returncode == 0 and r.stderr == "", r
        assert setup_calls(log) == calls, "healthy venv must not rebuild"
        print("ok  healthy venv runs straight through, silently")

        (venv / ".complete").unlink()
        run(wrapper, log)
        assert setup_calls(log) == calls + 1, "half-built venv must rebuild"
        print("ok  venv without the .complete marker is rebuilt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
