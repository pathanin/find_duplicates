"""
find_duplicates.py

CLI entry point for the duplicate-image tool: scans a directory for images
that look like the same photo saved at different sizes/qualities, groups
them with a perceptual hash, scores each candidate using
compare_image_quality.analyze(), and lets you confirm which one to keep --
interactively from a browser page (LAN-capable, so a photo library on a
NAS/headless box can be reviewed from another machine), or fully
unattended with --auto. Non-kept files are moved to ./_duplicates/, never
deleted -- restoring one is a manual move back out of that folder (with
--recursive, a moved file's subdirectory structure is mirrored under
_duplicates/, so the original relative location is still recoverable from
the path alone).

The scan/group/score/move pipeline lives in duplicates_core.py; the actual
FastAPI app lives in duplicates_web.py (importable, so tests can drive it
directly). This script just seeds the first scan from argv, then either
runs it non-interactively (--auto) or hands the app to uvicorn.

Usage:
    python find_duplicates.py [directory] [--threshold N] [--dest DIR]
                               [--recursive] [--auto] [--dry-run]
                               [--host HOST] [--port PORT]

Requires:
    pip install opencv-python-headless numpy pillow pillow-heif fastapi uvicorn
"""

import argparse
import asyncio
import os
import secrets
import socket
import sys
import webbrowser
from pathlib import Path

import uvicorn

from duplicates_core import DEFAULT_HASH_THRESHOLD, auto_apply_groups, build_groups, humansize
import duplicates_web
from duplicates_web import ScanParams, create_app

DEFAULT_PORT = 8737


def _lan_ip() -> str | None:
    """Best-effort discovery of this machine's LAN-facing IP, for the
    printed URL when bound to 0.0.0.0 -- "localhost" there would be
    actively wrong for the plan's "review from another machine" use case
    (it only ever resolves on the host itself). Connecting a UDP socket
    doesn't send any packets (UDP is connectionless); it just asks the OS
    to pick the local address it would route through to reach the target,
    which is a common trick for finding the outbound-facing interface
    without depending on hostname resolution being configured sanely."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return None


def _threshold_arg(s: str) -> int:
    v = int(s)
    if not 0 <= v <= 64:
        raise argparse.ArgumentTypeError(f"threshold must be 0-64, got {v}")
    return v


def _run_auto(directory: Path, dest_dir: Path, threshold: int, recursive: bool, dry_run: bool) -> None:
    """Non-interactive path: scan, keep each group's suggested (top-scored)
    file automatically, and exit -- never starts the web server. One bad
    group (disk full, permission error) doesn't abort the run; see
    auto_apply_groups."""
    print(f"Scanning {directory} ...")
    groups = build_groups(directory, threshold, recursive=recursive, dest_dir=dest_dir)
    if not groups:
        print("No potential duplicate groups found.")
        return

    print(f"Found {len(groups)} potential duplicate group(s). Auto-applying suggested picks...")
    summary = auto_apply_groups(groups, dest_dir, dry_run, recursive=recursive, scan_root=directory)
    reclaimed = "(dry run)" if dry_run else humansize(summary["bytes_reclaimed"])
    print(
        f"\nDone. {summary['confirmed']} group(s) confirmed, {summary['files_moved']} file(s) "
        f"{'would be moved' if dry_run else 'moved'} to {dest_dir}. Reclaimed: {reclaimed}"
    )
    if summary["failed"]:
        print(f"\n{summary['failed']} group(s) FAILED and were left pending:", file=sys.stderr)
        for f in summary["failures"]:
            print(
                f"  group {f['group']}: {f['error']} "
                f"({f['files_moved']} file(s)/{humansize(f['bytes_moved'])} moved before the failure)",
                file=sys.stderr,
            )
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Find and review potential duplicate images by quality.")
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
        "--auto",
        "--yes",
        action="store_true",
        help="Non-interactive: skip the review UI and keep each group's suggested (top-scored) file automatically.",
    )
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

    if args.auto:
        _run_auto(directory, dest_dir, args.threshold, args.recursive, args.dry_run)
        return

    token = secrets.token_urlsafe(32)
    params = ScanParams(
        directory=directory, threshold=args.threshold, recursive=args.recursive,
        dest_dir=dest_dir, dry_run=args.dry_run,
    )
    app = create_app(params, token)

    # 0.0.0.0/:: bind to "any interface" -- not itself a valid address to
    # browse to. The plan's killer use case is reviewing from *another*
    # machine on the LAN, so show that machine's actual reachable address
    # rather than "localhost" (which would only ever resolve on the host
    # itself and actively mislead a remote reviewer).
    if args.host in ("0.0.0.0", "::"):
        display_host = _lan_ip() or socket.gethostname()
    else:
        display_host = args.host
    url = f"http://{display_host}:{args.port}/?token={token}"
    print(f"Scanning {directory} ...")
    print(f"Open: {url}")
    if args.host in ("127.0.0.1", "localhost") and not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    class _Server(uvicorn.Server):
        def handle_exit(self, sig, frame):
            # Before uvicorn starts waiting for in-flight requests: an open
            # SSE progress stream has to be told to end, or the wait lasts
            # as long as the scan it is reporting on.
            duplicates_web.shutting_down.set()
            super().handle_exit(sig, frame)

    server = _Server(
        uvicorn.Config(app, host=args.host, port=args.port, log_level="warning")
    )
    # Driven on a bare loop rather than server.run()/asyncio.run(): the runner
    # installs its own SIGINT handler, and uvicorn re-raising the signal
    # through it turns a quick double Ctrl-C into a KeyboardInterrupt that
    # cancels the lifespan task mid-shutdown and dumps a traceback.
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(server.serve())
    except KeyboardInterrupt:
        pass
    # Exit without waiting on the interpreter's teardown: a scan thread may
    # still be running, and nothing is pending on disk -- the manifest and
    # caches are in-memory only, and a graceful shutdown has already let any
    # in-flight file move finish. os._exit skips the buffer flush too, which
    # would eat both startup lines under a redirect (a headless box's log).
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
