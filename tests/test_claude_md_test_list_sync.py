"""Guard against CLAUDE.md's test list drifting from tests/ on disk.

This bug has already happened twice (test_score_group.py, then
test_unapply_crash_safety.py): a new tests/test_*.py file gets added
without a matching `python3 tests/test_X.py` line in CLAUDE.md's "## Tests"
fenced code block, so anyone running "the test suite" off that doc silently
skips it. This test fails loudly the moment the two go out of sync again,
in either direction (a file on disk with no CLAUDE.md line, or a CLAUDE.md
line pointing at a file that no longer exists).

Run: python3 test_claude_md_test_list_sync.py
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = REPO_ROOT / "tests"
CLAUDE_MD = REPO_ROOT / "CLAUDE.md"


def _files_on_disk() -> set[str]:
    return {p.name for p in TESTS_DIR.glob("test_*.py")}


def _files_in_claude_md() -> set[str]:
    text = CLAUDE_MD.read_text()
    match = re.search(r"## Tests\n.*?```bash\n(.*?)```", text, re.DOTALL)
    assert match, "CLAUDE.md has no '## Tests' fenced bash block to check"
    return set(re.findall(r"tests/(test_\w+\.py)", match.group(1)))


def test_every_disk_file_is_listed() -> None:
    missing = _files_on_disk() - _files_in_claude_md()
    assert not missing, (
        f"tests/ has file(s) not listed in CLAUDE.md's '## Tests' section: {sorted(missing)}"
    )


def test_every_listed_file_exists_on_disk() -> None:
    dangling = _files_in_claude_md() - _files_on_disk()
    assert not dangling, (
        f"CLAUDE.md's '## Tests' section lists file(s) that don't exist in tests/: {sorted(dangling)}"
    )


def main() -> None:
    tests = [
        ("every disk file is listed in CLAUDE.md", test_every_disk_file_is_listed),
        ("every CLAUDE.md-listed file exists on disk", test_every_listed_file_exists_on_disk),
    ]
    for name, fn in tests:
        try:
            fn()
            print(f"ok {name}")
        except Exception as e:
            print(f"FAIL {name}: {e}")
            sys.exit(1)
    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    main()
