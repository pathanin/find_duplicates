# find-duplicates

Scan a directory for near-duplicate images — the same photo exported at different sizes or quality levels — and pick the best one to keep, from a browser page (LAN-capable, so a photo library sitting on a NAS or headless box can be reviewed from another machine).

```bash
python3 find_duplicates.py [directory] [--threshold N] [--dest DIR] [--recursive] [--auto] [--dry-run] [--host H] [--port N]
```

## How it works

1. **Perceptual hash** — each image gets a 64-bit DCT hash. Images within a configurable Hamming distance (default 10/64) are grouped as duplicates.
2. **Quality scoring** — every image in a group is evaluated on sharpness, effective resolution (FFT-based, resistant to fake upscaling), noise, and blockiness. Scores are min-max normalized within each group and weighted into a composite quality score.
3. **Interactive review** — pick a keeper per group from a browser page: it shows thumbnails and per-image metrics with keyboard shortcuts. `--auto` skips review entirely and keeps each group's suggested (top-scored) file automatically, without starting the web server.
4. **Safe cleanup** — non-kept files are moved to `_duplicates/` (never deleted), so you can always move one back by hand if you change your mind. Pass `--dry-run` to preview without moving anything.

Byte-identical copies score exactly equal, so the tie-break decides: the shorter filename wins, because every convention for naming a copy *appends* to the original (`" copy"`, `" 2"`, `" (1)"`, `"-edited"`). That keeps `holiday.jpg` over `holiday copy.jpg`.

## Optional: the filename hint

The quality metrics read pixels, so they can't see that `Downloads/IMG_4821 (1).jpg` is a re-download of `Originals/IMG_4821.jpg`. With a [TypeSafe](https://typesafe.ai) API key, the review page adds a hint line reading the *names*: which file looks like the original, and whether a group is really one photo stored twice rather than two similar shots.

```bash
find-duplicates --set-typesafe-key      # prompts, saves, exits
```

The key is stored in `~/.config/find_duplicates/typesafe-key`, readable by you only. It's prompted rather than passed as a flag so it stays out of `ps` and your shell history. `$TYPESAFE_API_KEY` overrides the file for a single run:

```bash
TYPESAFE_API_KEY=sk-... find-duplicates ~/Pictures
```

To turn the hint off again, delete that file and unset the variable.

**Entirely optional.** With no key, no network, or an API error the hint is simply absent and everything else — scanning, scoring, the suggested pick, `--auto` — behaves exactly the same. The hint is advisory in any case: it never changes which file is suggested or moved, and `--auto` never calls it.

## Install

Requires Python 3.10+. The install script creates an isolated venv via pip
(prebuilt wheels, seconds) and puts `find-duplicates` on your PATH.

One-liner:

```bash
curl -LsSf https://raw.githubusercontent.com/pathanin/find_duplicates/main/install.sh | sh
```

Or, from a clone:

```bash
git clone https://github.com/pathanin/find_duplicates.git
cd find_duplicates
./install.sh
```

## Usage

```bash
find-duplicates [directory] [options]
```

Prints a tokened URL (`http://127.0.0.1:8737/?token=...` by default) and opens it in your browser; the server keeps running until you stop it rather than exiting when a review finishes — Ctrl-C in the terminal, or **Quit** in the page's top bar (`q`), which matters when the scan is running on a machine whose terminal isn't in front of you. The page's control panel lets you change directory/threshold/recursive/dest and trigger a rescan without restarting the process. To review from another device on your network, bind to all interfaces:

```bash
find-duplicates /path/to/photos --host 0.0.0.0
```

...and open the printed LAN URL (with its token) from the other machine's browser. The URL's token is required for every request — treat it like a password on a shared network.

Keyboard shortcuts: arrows to change the keeper selection, digit keys to jump to an image by number, `c`/`Enter` to confirm, `s`/`Delete`/`Backspace` to skip, `z`/click the stage to inspect at 1:1, `o` to open the full-res original, `q` to quit the server, `?`/`F1` for help. The sidebar marks each group's status (pending/kept/skipped) with a shape-and-color dot; a "close call" (top two picks scored nearly the same) shows on the row's tooltip and the active group's ledger note, not as a sidebar mark of its own — most groups in a real scan are close calls, so flagging every row would stop meaning anything.

## CLI options

| Flag | Default | Description |
|---|---|---|
| `directory` | `.` | Directory to scan |
| `--threshold` | `10` | Max Hamming distance (0–64) for duplicate match |
| `--dest` | `./_duplicates` | Where to move non-kept files |
| `--recursive`, `-r` | — | Scan subdirectories too, not just the top level |
| `--dry-run` | — | Show what would happen, don't move anything |
| `--auto`, `--yes` | — | Non-interactive: skip the review UI, keep each group's suggested file automatically, and exit (no web server) |
| `--host` | `127.0.0.1` | Bind address; use `0.0.0.0` to expose on the LAN |
| `--port` | `8737` | Port to listen on |
| `--no-browser` | — | Don't auto-open the URL in a browser |
| `--set-typesafe-key` | — | Prompt for a TypeSafe API key, save it, and exit — enables the optional filename hint |

## Tests

```bash
python3 tests/test_web_api.py
python3 tests/test_web_progress.py
python3 tests/test_help_and_labels.py
python3 tests/test_fast_scan.py
python3 tests/test_auto_mode.py
python3 tests/test_unapply_crash_safety.py
```

See `CLAUDE.md` for the full test list.
