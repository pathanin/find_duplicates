"""Machine-checks that install.sh ships every module the tool imports.

The bug this locks out: name_hint.py was added to the repo and imported by
find_duplicates.py, but install.sh's hand-written copy list never learned
about it. The install itself succeeded -- nothing there reads an import
graph -- and every `find-duplicates` run afterwards died on
`ModuleNotFoundError: No module named 'name_hint'`. Running from a clone
hid it completely, because there the module sits next to its importer.

Two halves, both needed: REQUIRED_FILES has to name every local module
(that's the list), and the copy step has to be driven BY that list rather
than a second hand-written one (that's what keeps the list honest).

Run: python3 tests/test_install_file_list.py
"""

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INSTALL = ROOT / "install.sh"


def local_modules() -> set[str]:
    """Every top-level .py in the repo root -- the candidate import names."""
    return {p.stem for p in ROOT.glob("*.py")}


def imported_locally() -> set[str]:
    """Local modules reachable from find_duplicates.py, walked transitively
    the way the interpreter would."""
    known = local_modules()
    seen: set[str] = set()
    queue = ["find_duplicates"]
    while queue:
        name = queue.pop()
        if name in seen:
            continue
        seen.add(name)
        tree = ast.parse((ROOT / f"{name}.py").read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                targets = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                targets = [(node.module or "").split(".")[0]]
            else:
                continue
            queue.extend(t for t in targets if t in known and t not in seen)
    return seen


def required_files() -> list[str]:
    m = re.search(r'^REQUIRED_FILES="([^"]+)"', INSTALL.read_text(), re.M)
    assert m, "install.sh has no REQUIRED_FILES assignment"
    return m.group(1).split()


def test_every_imported_module_is_installed() -> None:
    listed = set(required_files())
    for name in sorted(imported_locally()):
        assert f"{name}.py" in listed, (
            f"{name}.py is imported at runtime but missing from install.sh's "
            f"REQUIRED_FILES -- an installed find-duplicates would die on its import"
        )
    print(f"  ok  all {len(imported_locally())} imported modules are in REQUIRED_FILES")


def test_required_files_all_exist() -> None:
    for f in required_files():
        assert (ROOT / f).is_file(), f"install.sh lists {f}, which is not in the repo"
    print("  ok  every file REQUIRED_FILES names exists in the repo")


def test_the_copy_step_is_driven_by_that_list() -> None:
    """A second hand-written list in the cp is how the drift happened. The
    copy has to iterate REQUIRED_FILES, so one edit covers both uses."""
    text = INSTALL.read_text()
    copy_block = re.search(
        r'mkdir -p "\$DATA_DIR/libexec".*?cp -r "\$REPO_ROOT/static"', text, re.S,
    )
    assert copy_block, "couldn't find install.sh's libexec copy step"
    body = copy_block.group(0)
    assert "$REQUIRED_FILES" in body, (
        "the libexec copy step doesn't iterate REQUIRED_FILES -- a second list "
        "here is exactly what left name_hint.py out of the install"
    )
    stray = re.findall(r'"\$REPO_ROOT/(\w+\.py)"', body)
    assert not stray, f"the copy step names modules by hand: {stray}"
    print("  ok  the libexec copy step is driven by REQUIRED_FILES, not a second list")


def main():
    for test in (test_every_imported_module_is_installed,
                 test_required_files_all_exist,
                 test_the_copy_step_is_driven_by_that_list):
        print(f"{test.__name__}:")
        test()
    print("all install file-list tests passed")


if __name__ == "__main__":
    main()
