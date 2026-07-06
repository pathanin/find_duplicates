# find-duplicates

Scan a directory for near-duplicate images — the same photo exported at different sizes or quality levels — and pick the best one to keep.

```
python3 find_duplicates.py [directory] [--threshold N] [--dest DIR] [--dry-run]
```

## How it works

1. **Perceptual hash** — each image gets a 64-bit DCT hash. Images within a configurable Hamming distance (default 10/64) are grouped as duplicates.
2. **Quality scoring** — every image in a group is evaluated on sharpness, effective resolution (FFT-based, resistant to fake upscaling), noise, and blockiness. Scores are min-max normalized within each group and weighted into a composite quality score.
3. **Interactive review** — a Textual TUI shows thumbnails and per-image metrics. Navigate with `←` `→`, pick your keeper with `c`, skip the group with `s`.
4. **Safe cleanup** — non-kept files are moved to `_duplicates/` (never deleted). Every decision is logged to `decisions.json`. Pass `--dry-run` to preview without moving anything.

## Install

```bash
# Homebrew
brew install pathanin/find_duplicates/find-duplicates

# Or run directly (Python 3.10+)
pip install opencv-python-headless numpy textual textual-image pillow
python3 find_duplicates.py
```

## Usage

| Key | Action |
|---|---|
| `←` `→` | Change keeper selection |
| `1`–`9` | Jump to image by number |
| `c` / `Enter` | Confirm keep |
| `s` / `Delete` | Skip group |
| `o` | Open full-res in Preview |
| `?` / `F1` | Help |
| `q` / `Escape` | Finish |

## CLI options

| Flag | Default | Description |
|---|---|---|
| `directory` | `.` | Directory to scan |
| `--threshold` | `10` | Max Hamming distance (0–64) for duplicate match |
| `--dest` | `./_duplicates` | Where to move non-kept files |
| `--dry-run` | — | Show what would happen, don't move anything |

## Tests

\`\`\`bash
python3 tests/test_preview_render.py
python3 tests/test_help_and_labels.py
python3 tests/test_footer_click_safety.py
python3 tests/test_keyboard_aliases.py
python3 tests/test_fast_scan.py
\`\`\`

## License

MIT
