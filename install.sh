#!/bin/sh
# Creates an isolated venv with plain pip (fast, wheel-based install) and
# puts a `find-duplicates` wrapper on PATH.
#
# Usage (already have a clone):
#   ./install.sh
#
# Usage (no clone needed):
#   curl -LsSf https://raw.githubusercontent.com/pathanin/find_duplicates/main/install.sh | sh
#
# Written in POSIX sh, not bash: `curl ... | sh` runs this under the
# invoker's /bin/sh regardless of the shebang above, so bash-only syntax
# (arrays, [[ ]], `pipefail`) would silently misbehave there even though
# ./install.sh under bash would be fine.

set -eu

if [ "$#" -gt 0 ]; then
  echo "error: install.sh takes no arguments (got: $*)" >&2
  exit 1
fi

REPO_TARBALL="https://github.com/pathanin/find_duplicates/archive/refs/heads/main.tar.gz"
DATA_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/find-duplicates"
VENV_DIR="$DATA_DIR/venv"
BIN_DIR="$HOME/.local/bin"

# Every module the tool imports, in one list: it validates a candidate repo
# root below AND is what gets copied into libexec, so a module added to the
# repo can't reach one half and miss the other. name_hint.py did exactly
# that -- the install ran clean and every run then died on its import.
# duplicates_core.py + compare_image_quality.py are the shared scan/score/
# move pipeline; duplicates_web.py + find_duplicates.py + static/ are the
# (only) front end; name_hint.py is the optional filename hint.
REQUIRED_FILES="duplicates_core.py compare_image_quality.py duplicates_web.py find_duplicates.py name_hint.py"

have_required_files() {
  # $1 is the candidate root directory.
  for f in $REQUIRED_FILES; do
    [ -f "$1/$f" ] || return 1
  done
  [ -d "$1/static" ] || return 1
  [ -f "$1/setup_venv.sh" ] || return 1
  return 0
}

# When run as `./install.sh` or `bash install.sh` from a clone, $0 points at
# a real file next to the rest of the repo -- use that, no network needed
# beyond what pip already requires. When run as `curl ... | sh`, $0 is just
# "sh"/"-sh" (there is no on-disk script to locate), so fall through to
# downloading a tarball of the repo instead.
REPO_ROOT=""
case "${0:-}" in
  */install.sh|install.sh)
    if [ -f "$0" ]; then
      CANDIDATE="$(cd "$(dirname "$0")" && pwd)"
      if have_required_files "$CANDIDATE"; then
        REPO_ROOT="$CANDIDATE"
      fi
    fi
    ;;
esac

CLEANUP_DIR=""
cleanup() {
  # `[ -n ] && rm` would end a clone install with status 1 (the trap's
  # last status becomes the script's), failing `./install.sh && ...`.
  [ -z "$CLEANUP_DIR" ] || rm -rf "$CLEANUP_DIR"
}
trap cleanup EXIT

if [ -z "$REPO_ROOT" ]; then
  echo "==> Downloading find_duplicates source (main branch)"
  if ! command -v tar >/dev/null 2>&1; then
    echo "error: need tar to unpack the downloaded source" >&2
    exit 1
  fi
  CLEANUP_DIR="$(mktemp -d)"
  ARCHIVE="$CLEANUP_DIR/src.tar.gz"
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf "$REPO_TARBALL" -o "$ARCHIVE"
  elif command -v wget >/dev/null 2>&1; then
    wget -q "$REPO_TARBALL" -O "$ARCHIVE"
  else
    echo "error: need curl or wget to download the source" >&2
    exit 1
  fi
  tar -xzf "$ARCHIVE" -C "$CLEANUP_DIR"
  # GitHub's branch tarball extracts to a single top-level <repo>-<branch>/ dir.
  EXTRACTED="$(find "$CLEANUP_DIR" -mindepth 1 -maxdepth 1 -type d -name 'find_duplicates-*')"
  if [ -z "$EXTRACTED" ] || ! have_required_files "$EXTRACTED"; then
    echo "error: downloaded source is missing expected files" >&2
    exit 1
  fi
  REPO_ROOT="$EXTRACTED"
fi

mkdir -p "$DATA_DIR"
sh "$REPO_ROOT/setup_venv.sh" "$VENV_DIR"

echo "==> Installing scripts"
# Clear libexec rather than copying over it: cp only ever adds, so a module
# dropped from the repo between versions (find_duplicates-web.py, removed
# with the Textual TUI) would sit here forever, along with a __pycache__
# holding its stale .pyc. DATA_DIR is rooted at XDG_DATA_HOME or $HOME, so
# this can't degrade into an rm -rf of a bare /libexec.
rm -rf "$DATA_DIR/libexec"
mkdir -p "$DATA_DIR/libexec"
for f in $REQUIRED_FILES; do
  cp "$REPO_ROOT/$f" "$DATA_DIR/libexec/"
done
cp -r "$REPO_ROOT/static" "$DATA_DIR/libexec/static"
cp "$REPO_ROOT/setup_venv.sh" "$DATA_DIR/libexec/"

mkdir -p "$BIN_DIR"
WRAPPER="$BIN_DIR/find-duplicates"
echo "==> Writing wrapper to $WRAPPER"
# The venv dies with the Python it was built from (a Homebrew upgrade to a
# new minor version, then python@X.Y removed), so the wrapper checks for its
# interpreter and the .complete marker setup_venv.sh writes last, and
# rebuilds on the spot instead of failing every run until install.sh is
# found again. Both tests are file checks: nothing is spent on a healthy run.
cat > "$WRAPPER" <<EOS
#!/bin/sh
if [ ! -x "$VENV_DIR/bin/python3" ] || [ ! -f "$VENV_DIR/.complete" ]; then
  echo "find-duplicates: its Python environment is missing or incomplete (often a Homebrew Python upgrade); rebuilding it..." >&2
  if ! sh "$DATA_DIR/libexec/setup_venv.sh" "$VENV_DIR" >&2; then
    echo "find-duplicates: rebuild failed; rerun install.sh to repair it." >&2
    exit 1
  fi
fi
exec "$VENV_DIR/bin/python3" "$DATA_DIR/libexec/find_duplicates.py" "\$@"
EOS
chmod +x "$WRAPPER"

# Drop a wrapper from a prior TUI/web-split install: it would exec a
# find_duplicates-web.py that no longer exists in libexec/.
rm -f "$BIN_DIR/find-duplicates-web"

echo
echo "Installed."
echo "Run: find-duplicates --help"

case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *)
    echo
    echo "Note: $BIN_DIR is not on your PATH yet. Add this to your shell rc file:"
    echo "  export PATH=\"$BIN_DIR:\$PATH\""
    ;;
esac
