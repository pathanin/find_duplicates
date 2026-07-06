#!/usr/bin/env bash
# Creates an isolated venv with plain pip (fast, wheel-based install of
# numpy/opencv-python-headless/pillow/textual) and puts a `find-duplicates`
# wrapper on PATH.
#
# Usage: ./contrib/install.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/find-duplicates"
VENV_DIR="$DATA_DIR/venv"
BIN_DIR="$HOME/.local/bin"
WRAPPER="$BIN_DIR/find-duplicates"

for f in find_duplicates.py compare_image_quality.py; do
  if [[ ! -f "$REPO_ROOT/$f" ]]; then
    echo "error: $f not found next to this script (expected at $REPO_ROOT/$f)" >&2
    exit 1
  fi
done

if ! command -v python3 >/dev/null 2>&1; then
  echo "error: python3 not found. Install Python 3.10+ first." >&2
  exit 1
fi

PY_VERSION="$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
PY_OK="$(python3 -c 'import sys; print(1 if sys.version_info >= (3, 10) else 0)')"
if [[ "$PY_OK" != "1" ]]; then
  echo "error: python3 is $PY_VERSION, but 3.10+ is required." >&2
  exit 1
fi

echo "==> Using python3 $PY_VERSION"

echo "==> Creating venv at $VENV_DIR"
mkdir -p "$DATA_DIR"
python3 -m venv "$VENV_DIR"

echo "==> Installing dependencies (prebuilt wheels via pip)"
"$VENV_DIR/bin/pip" install --upgrade pip --quiet
"$VENV_DIR/bin/pip" install --quiet \
  numpy \
  opencv-python-headless \
  pillow \
  textual \
  textual-image

echo "==> Installing scripts"
mkdir -p "$DATA_DIR/libexec"
cp "$REPO_ROOT/find_duplicates.py" "$REPO_ROOT/compare_image_quality.py" "$DATA_DIR/libexec/"

echo "==> Writing wrapper to $WRAPPER"
mkdir -p "$BIN_DIR"
cat > "$WRAPPER" <<EOS
#!/bin/bash
exec "$VENV_DIR/bin/python3" "$DATA_DIR/libexec/find_duplicates.py" "\$@"
EOS
chmod +x "$WRAPPER"

echo
echo "Installed. Run: find-duplicates --help"

case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *)
    echo
    echo "Note: $BIN_DIR is not on your PATH yet. Add this to your shell rc file:"
    echo "  export PATH=\"$BIN_DIR:\$PATH\""
    ;;
esac
