"""
find_duplicates-web.py

Browser-based front end for the duplicate-image tool: same scan/group/
score/apply pipeline as find_duplicates.py's Textual TUI (both import it
from duplicates_core.py), but the review UI is a web page instead of a
terminal app -- LAN-capable, so a photo library on a NAS/headless box can
be reviewed from another machine's browser.

The actual FastAPI app lives in duplicates_web.py (importable, so tests can
drive it directly); this script is just the CLI entry point that seeds the
first scan from argv and hands the app to uvicorn. Server runs until
Ctrl-C -- it does NOT exit when a review finishes, unlike --auto mode.

Usage:
    python find_duplicates-web.py [directory] [--threshold N] [--dest DIR]
                                   [--recursive] [--dry-run]
                                   [--host HOST] [--port PORT]

Requires:
    pip install opencv-python-headless numpy pillow pillow-heif fastapi uvicorn
"""

import argparse
import secrets
import sys
import webbrowser
from pathlib import Path

import uvicorn

from duplicates_core import DEFAULT_HASH_THRESHOLD
from duplicates_web import ScanParams, create_app

DEFAULT_PORT = 8737


def _threshold_arg(s: str) -> int:
    v = int(s)
    if not 0 <= v <= 64:
        raise argparse.ArgumentTypeError(f"threshold must be 0-64, got {v}")
    return v


def main() -> None:
    parser = argparse.ArgumentParser(description="Browser-based review UI for the duplicate-image tool.")
    parser.add_argument("directory", nargs="?", default=".", type=Path)
    parser.add_argument(
        "--threshold",
        type=_threshold_arg,
        default=DEFAULT_HASH_THRESHOLD,
        help="Max Hamming distance (0-64) to consider two images duplicates. Lower = stricter. Default: %(default)s",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Folder to move non-kept duplicates into (default: <directory>/_duplicates)",
    )
    parser.add_argument(
        "--recursive", "-r", action="store_true", help="Scan subdirectories too, not just the top level."
    )
    parser.add_argument("--dry-run", action="store_true", help="Don't move any files, just show what would happen.")
    parser.add_argument(
        "--host", default="127.0.0.1", help="Bind address. Use 0.0.0.0 to expose on the LAN. Default: %(default)s"
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Default: %(default)s")
    parser.add_argument(
        "--no-browser", action="store_true", help="Don't try to auto-open the URL in a browser."
    )
    args = parser.parse_args()

    directory = args.directory
    if not directory.exists():
        print(f"Error: directory '{directory}' does not exist.", file=sys.stderr)
        sys.exit(1)
    if not directory.is_dir():
        print(f"Error: '{directory}' is not a directory.", file=sys.stderr)
        sys.exit(1)
    directory = directory.resolve()
    dest_dir = (args.dest or (directory / "_duplicates")).resolve()

    token = secrets.token_urlsafe(32)
    params = ScanParams(
        directory=directory, threshold=args.threshold, recursive=args.recursive,
        dest_dir=dest_dir, dry_run=args.dry_run,
    )
    app = create_app(params, token)

    display_host = "localhost" if args.host in ("0.0.0.0", "::") else args.host
    url = f"http://{display_host}:{args.port}/?token={token}"
    print(f"Scanning {directory} ...")
    print(f"Open: {url}")
    if args.host in ("127.0.0.1", "localhost") and not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
