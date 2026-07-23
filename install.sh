#!/bin/sh
# Creates an isolated venv with plain pip (fast, wheel-based install) and
# puts a `find-duplicates` (TUI) and/or `find-duplicates-web` (browser UI)
# wrapper on PATH.
#
# Usage (already have a clone):
#   ./install.sh [--tui|--web|--all]
#
# Usage (no clone needed):
#   curl -LsSf https://raw.githubusercontent.com/pathanin/find_duplicates/main/install.sh | sh
#   curl -LsSf https://raw.githubusercontent.com/pathanin/find_duplicates/main/install.sh | sh -s -- --tui
#
#   --tui   Textual terminal UI only (adds textual/textual-image)
#   --web   Browser UI only (adds fastapi/uvicorn) -- e.g. for a headless
#           NAS box that will never run a TUI in a terminal.
#   --all   Both (default).
#
# Written in POSIX sh, not bash: `curl ... | sh` runs this under the
# invoker's /bin/sh regardless of the shebang above, so bash-only syntax
# (arrays, [[ ]], `pipefail`) would silently misbehave there even though
# ./install.sh under bash would be fine.

set -eu

REPO_TARBALL="https://github.com/pathanin/find_duplicates/archive/refs/heads/main.tar.gz"
DATA_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/find-duplicates"
VENV_DIR="$DATA_DIR/venv"
BIN_DIR="$HOME/.local/bin"

COMPONENT="all"
for arg in "$@"; do
  case "$arg" in
    --tui) COMPONENT="tui" ;;
    --web) COMPONENT="web" ;;
    --all) COMPONENT="all" ;;
    *)
      echo "error: unknown argument '$arg' (expected --tui, --web, or --all)" >&2
      exit 1
      ;;
  esac
done
WANT_TUI=0
WANT_WEB=0
case "$COMPONENT" in
  tui) WANT_TUI=1 ;;
  web) WANT_WEB=1 ;;
  all) WANT_TUI=1; WANT_WEB=1 ;;
esac

# duplicates_core.py + compare_image_quality.py are the shared scan/score/
# move pipeline both front ends import -- required no matter which
# component(s) are selected. find_duplicates.py (TUI) and
# duplicates_web.py + find_duplicates-web.py + static/ (web) are only
# required for their respective component.
REQUIRED_FILES="duplicates_core.py compare_image_quality.py"
[ "$WANT_TUI" = "1" ] && REQUIRED_FILES="$REQUIRED_FILES find_duplicates.py"
[ "$WANT_WEB" = "1" ] && REQUIRED_FILES="$REQUIRED_FILES duplicates_web.py find_duplicates-web.py"

have_required_files() {
  # $1 is the candidate root directory.
  for f in $REQUIRED_FILES; do
    [ -f "$1/$f" ] || return 1
  done
  if [ "$WANT_WEB" = "1" ] && [ ! -d "$1/static" ]; then
    return 1
  fi
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
  [ -n "$CLEANUP_DIR" ] && rm -rf "$CLEANUP_DIR"
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

if ! command -v python3 >/dev/null 2>&1; then
  echo "error: python3 not found. Install Python 3.10+ first." >&2
  exit 1
fi

PY_VERSION="$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
PY_OK="$(python3 -c 'import sys; print(1 if sys.version_info >= (3, 10) else 0)')"
if [ "$PY_OK" != "1" ]; then
  echo "error: python3 is $PY_VERSION, but 3.10+ is required." >&2
  exit 1
fi

echo "==> Using python3 $PY_VERSION"
echo "==> Installing: $COMPONENT"

echo "==> Creating venv at $VENV_DIR"
mkdir -p "$DATA_DIR"
python3 -m venv "$VENV_DIR"

echo "==> Installing shared dependencies (prebuilt wheels via pip)"
"$VENV_DIR/bin/pip" install --upgrade pip --quiet
"$VENV_DIR/bin/pip" install --quiet \
  numpy \
  opencv-python-headless \
  pillow \
  pillow-heif

if [ "$WANT_TUI" = "1" ]; then
  echo "==> Installing TUI dependencies"
  "$VENV_DIR/bin/pip" install --quiet textual textual-image
fi
if [ "$WANT_WEB" = "1" ]; then
  echo "==> Installing web dependencies"
  "$VENV_DIR/bin/pip" install --quiet fastapi uvicorn
fi

echo "==> Installing scripts"
mkdir -p "$DATA_DIR/libexec"
cp "$REPO_ROOT/duplicates_core.py" "$REPO_ROOT/compare_image_quality.py" "$DATA_DIR/libexec/"
mkdir -p "$BIN_DIR"

if [ "$WANT_TUI" = "1" ]; then
  cp "$REPO_ROOT/find_duplicates.py" "$DATA_DIR/libexec/"
  WRAPPER="$BIN_DIR/find-duplicates"
  echo "==> Writing wrapper to $WRAPPER"
  cat > "$WRAPPER" <<EOS
#!/bin/sh
exec "$VENV_DIR/bin/python3" "$DATA_DIR/libexec/find_duplicates.py" "\$@"
EOS
  chmod +x "$WRAPPER"
fi

if [ "$WANT_WEB" = "1" ]; then
  cp "$REPO_ROOT/duplicates_web.py" "$REPO_ROOT/find_duplicates-web.py" "$DATA_DIR/libexec/"
  rm -rf "$DATA_DIR/libexec/static"
  cp -r "$REPO_ROOT/static" "$DATA_DIR/libexec/static"
  WEB_WRAPPER="$BIN_DIR/find-duplicates-web"
  echo "==> Writing wrapper to $WEB_WRAPPER"
  cat > "$WEB_WRAPPER" <<EOS
#!/bin/sh
exec "$VENV_DIR/bin/python3" "$DATA_DIR/libexec/find_duplicates-web.py" "\$@"
EOS
  chmod +x "$WEB_WRAPPER"
fi

echo
echo "Installed."
[ "$WANT_TUI" = "1" ] && echo "Run: find-duplicates --help"
[ "$WANT_WEB" = "1" ] && echo "Run: find-duplicates-web --help"

case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *)
    echo
    echo "Note: $BIN_DIR is not on your PATH yet. Add this to your shell rc file:"
    echo "  export PATH=\"$BIN_DIR:\$PATH\""
    ;;
esac
