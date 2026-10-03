"""Locks in install.sh building its venv from Homebrew's stable opt/ path.

The bug: `python3 -m venv` writes the base interpreter's *resolved* path into
the venv -- bin/python3.14 -> /opt/homebrew/Cellar/python@3.14/3.14.7/...
A routine `brew upgrade` to 3.14.8 deletes that Cellar dir, and every
`find-duplicates` run then dies with "venv/bin/python3: No such file or
directory". /opt/homebrew/opt/python@3.14 is a symlink Homebrew moves forward
on each upgrade, so a venv created from it survives patch releases.

Runs install.sh's `stable_python` function in sh against a fake Cellar/opt
tree; the rest of install.sh (network, pip) never executes.

Run: python3 tests/test_install_stable_python.py
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INSTALL = ROOT / "install.sh"


def stable_python(base_executable: str, version: str) -> str:
    m = re.search(r"^stable_python\(\) \{\n.*?^\}\n", INSTALL.read_text(), re.M | re.S)
    assert m, "install.sh has no stable_python() function"
    script = m.group(0) + 'stable_python "$1" "$2"\n'
    out = subprocess.run(
        ["sh", "-c", script, "sh", base_executable, version],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        prefix = Path(tmp)
        cellar_py = prefix / "Cellar/python@3.14/3.14.8/Frameworks/Python.framework/Versions/3.14/bin/python3.14"
        opt_py = prefix / "opt/python@3.14/bin/python3.14"
        for p in (cellar_py, opt_py):
            p.parent.mkdir(parents=True)
            p.write_text("")
            p.chmod(0o755)

        assert stable_python(str(cellar_py), "3.14") == str(opt_py)
        print("ok  Cellar interpreter maps to the opt/ symlink")

        opt_py.unlink()
        assert stable_python(str(cellar_py), "3.14") == str(cellar_py)
        print("ok  falls back to the given path when opt/ is missing")

    assert stable_python("/usr/bin/python3", "3.9") == "/usr/bin/python3"
    print("ok  non-Homebrew interpreter is left alone")
    return 0


if __name__ == "__main__":
    sys.exit(main())
