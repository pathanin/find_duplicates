# find-duplicates

Scan a directory for near-duplicate images — the same photo exported at different sizes or quality levels — and pick the best one to keep. Two front ends share one scan/score/move pipeline: a terminal UI, and a browser UI (LAN-capable, so a photo library sitting on a NAS or headless box can be reviewed from another machine).

```bash
python3 find_duplicates.py [directory] [--threshold N] [--dest DIR] [--recursive] [--auto] [--dry-run]
python3 find_duplicates-web.py [directory] [--threshold N] [--dest DIR] [--recursive] [--dry-run] [--host H] [--port N]
```

## How it works

1. **Perceptual hash** — each image gets a 64-bit DCT hash. Images within a configurable Hamming distance (default 10/64) are grouped as duplicates.
2. **Quality scoring** — every image in a group is evaluated on sharpness, effective resolution (FFT-based, resistant to fake upscaling), noise, and blockiness. Scores are min-max normalized within each group and weighted into a composite quality score.
3. **Interactive review** — pick a keeper per group, either in a Textual TUI or a browser page; both show thumbnails and per-image metrics with the same keyboard shortcuts. The TUI's `--auto` skips review entirely and keeps each group's suggested (top-scored) file automatically.
4. **Safe cleanup** — non-kept files are moved to `_duplicates/` (never deleted), so you can always move one back by hand if you change your mind. Pass `--dry-run` to preview without moving anything.

## Install

Requires Python 3.10+. The install script creates an isolated venv via pip
(prebuilt wheels, seconds) and puts `find-duplicates` / `find-duplicates-web`
on your PATH.

One-liner (no clone needed) — installs both front ends:

```bash
curl -LsSf https://raw.githubusercontent.com/pathanin/find_duplicates/main/install.sh | sh
```

Add `--tui` or `--web` to install just one front end:

```bash
curl -LsSf https://raw.githubusercontent.com/pathanin/find_duplicates/main/install.sh | sh -s -- --tui
```

Or, from a clone:

```bash
git clone https://github.com/pathanin/find_duplicates.git
cd find_duplicates
./install.sh                  # both front ends; add --tui or --web for just one
```

## Terminal UI

```bash
find-duplicates [directory] [options]
```

| Key | Action |
|---|---|
| `←` `→` | Change keeper selection |
| `1`–`9` | Jump to image by number |
| `c` / `Enter` | Confirm keep |
| `s` / `Delete` | Skip group |
| `o` | Open full-res in Preview |
| `?` / `F1` | Help |
| `q` / `Escape` | Finish |

## Web UI

```bash
find-duplicates-web [directory] [options]
```

Prints a tokened URL (`http://127.0.0.1:8737/?token=...` by default) and opens it in your browser; the server keeps running until Ctrl-C rather than exiting when a review finishes. The page's control panel lets you change directory/threshold/recursive/dest and trigger a rescan without restarting the process. To review from another device on your network, bind to all interfaces:

```bash
find-duplicates-web /path/to/photos --host 0.0.0.0
```

...and open the printed LAN URL (with its token) from the other machine's browser. The URL's token is required for every request — treat it like a password on a shared network.

Keyboard shortcuts match the TUI (arrows, `c`/`Enter`, `s`/`Delete`/`Backspace`, `o`, `?`/`F1`, digit keys to jump). The sidebar marks each group's status: `◻` pending, `✔` confirmed, `—` skipped, `⚠` close call (top two picks scored nearly the same).

## CLI options

### `find_duplicates.py` (TUI)

| Flag | Default | Description |
|---|---|---|
| `directory` | `.` | Directory to scan |
| `--threshold` | `10` | Max Hamming distance (0–64) for duplicate match |
| `--dest` | `./_duplicates` | Where to move non-kept files |
| `--recursive`, `-r` | — | Scan subdirectories too, not just the top level |
| `--auto`, `--yes` | — | Non-interactive: skip the review UI, keep each group's suggested file automatically |
| `--dry-run` | — | Show what would happen, don't move anything |

### `find_duplicates-web.py` (browser)

| Flag | Default | Description |
|---|---|---|
| `directory` | `.` | Directory to scan |
| `--threshold` | `10` | Max Hamming distance (0–64) for duplicate match |
| `--dest` | `./_duplicates` | Where to move non-kept files |
| `--recursive`, `-r` | — | Scan subdirectories too, not just the top level |
| `--dry-run` | — | Show what would happen, don't move anything |
| `--host` | `127.0.0.1` | Bind address; use `0.0.0.0` to expose on the LAN |
| `--port` | `8737` | Port to listen on |
| `--no-browser` | — | Don't auto-open the URL in a browser |

## Tests

```bash
python3 tests/test_preview_render.py
python3 tests/test_help_and_labels.py
python3 tests/test_footer_click_safety.py
python3 tests/test_keyboard_aliases.py
python3 tests/test_fast_scan.py
python3 tests/test_web_api.py
python3 tests/test_web_progress.py
```

See `CLAUDE.md` for the full test list.

## License

MIT
