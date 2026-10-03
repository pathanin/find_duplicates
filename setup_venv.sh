#!/bin/sh
# Builds (or rebuilds) the tool's venv at $1 with plain pip.
#
# Two callers: install.sh on a fresh install, and the `find-duplicates`
# wrapper when the venv's python is gone. A venv only ever works with the
# Python minor version it was built from -- its numpy/opencv/pillow wheels
# are compiled for it -- so when Homebrew removes that interpreter the only
# repair is a rebuild against whatever python3 is current. Installed into
# libexec/ so the wrapper can run it without the repo.
#
# POSIX sh, like install.sh.

set -eu

if [ "$#" -ne 1 ]; then
  echo "usage: setup_venv.sh VENV_DIR" >&2
  exit 1
fi
VENV_DIR="$1"

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

# `venv` links its python at the base interpreter's resolved path, which for
# Homebrew is a versioned Cellar dir (.../Cellar/python@3.14/3.14.7/...).
# The next `brew upgrade` deletes that dir and the wrapper dies with "No
# such file or directory". Homebrew's opt/python@X.Y symlink follows
# upgrades, so build the venv from that instead when it exists.
stable_python() {
  # $1 is the base interpreter path, $2 its X.Y version.
  case "$1" in
    */Cellar/python@*/*)
      formula="${1#*/Cellar/}"
      formula="${formula%%/*}"
      opt="${1%%/Cellar/*}/opt/$formula/bin/python$2"
      if [ -x "$opt" ]; then
        echo "$opt"
        return
      fi
      ;;
  esac
  echo "$1"
}
VENV_PYTHON="$(stable_python "$(python3 -c 'import sys; print(sys._base_executable)')" "$PY_VERSION")"

echo "==> Creating venv at $VENV_DIR"
# Remove any existing venv rather than letting `venv` upgrade it in place:
# with its interpreter gone, `venv`'s upgrade path stats the dangling
# symlink and dies with the same "No such file or directory" being fixed.
rm -rf "$VENV_DIR"
mkdir -p "$(dirname "$VENV_DIR")"
"$VENV_PYTHON" -m venv "$VENV_DIR"

echo "==> Installing dependencies (prebuilt wheels via pip)"
"$VENV_DIR/bin/pip" install --upgrade pip --quiet
"$VENV_DIR/bin/pip" install --quiet \
  numpy \
  opencv-python-headless \
  pillow \
  pillow-heif \
  fastapi \
  uvicorn

# Written last: the wrapper rebuilds any venv without it, so a pip run cut
# short by a dropped network can't leave a venv that starts and then dies
# on `import cv2`.
touch "$VENV_DIR/.complete"
