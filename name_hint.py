"""
name_hint.py

Which duplicate is the original, judged from filenames alone.

The pixel metrics in compare_image_quality.py can't see a filename, and
duplicates very often differ by name rather than by pixels -- "x.jpg" vs
"x copy 2.jpg", "Originals/IMG_4821.jpg" vs "Downloads/IMG_4821 (1).jpg",
"sunset.jpg" vs "thumbs/sunset_thumb.jpg". That's the signal a human reviewer
uses on every close call, and CLAUDE.md's own note says close calls are most
groups on a real library. Writing it as regexes means a growing pile of
`\\(\\d+\\)$`, ` copy`, `-edited`, `_v2_final`, `IMG-\\d{8}-WA\\d+` patterns
that each cover one naming convention and mis-fire on the next.

So this asks TypeSafe's Jev instead: one Choice ("which name reads as the
original") and one Noul ("are these really re-exports of one photo, or
separate frames that merely look alike"). The Noul catches the false
positives CONFIRM_HASH_THRESHOLD deliberately lets through -- a burst series
shares a filename stem but isn't one photo stored twice.

The key comes from $TYPESAFE_API_KEY or ~/.config/find_duplicates/typesafe-key
(`find_duplicates.py --set-typesafe-key` writes the latter).

Advisory only, by design. Nothing here feeds quality_score, suggested_idx or
the file moves; the hint is shown next to the close-call note and the
reviewer still decides. Optional like brisque/niqe: no API key, no network
or any API error yields None and the app behaves exactly as before.
"""

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

API_URL = "https://api.typesafe.ai/v1/systemone"
# A file, not just $TYPESAFE_API_KEY, because the tool's own use case is a
# NAS or headless box you reach over SSH -- an env var set in one shell
# doesn't survive the next login or a systemd unit. `--set-typesafe-key`
# writes this; the env var still wins when both are present.
KEY_PATH = Path.home() / ".config" / "find_duplicates" / "typesafe-key"
MODEL = "jev-latest"
TIMEOUT = 10.0

_KEEP_Q = {
    "type": "choice",
    "instructions": "These files are pixel-level duplicates of a single photo, listed by path. "
                    "Which one is the original, rather than a copy produced later by an operating "
                    "system duplicate, a download, a messaging app, a thumbnail, or an edit?",
}
_SAME_PHOTO_Q = {
    "type": "noul",
    "instructions": "Do these paths describe the same single photo stored more than once, or "
                    "separate shots that merely look alike?",
    "criteria": {
        "true": "One photo stored repeatedly -- a copy suffix, a re-download, a resize, a re-encode, an edit",
        "false": "Different frames, different crops, or consecutive shots from one burst or series",
    },
}

def load_key() -> str | None:
    """The API key from $TYPESAFE_API_KEY, else KEY_PATH, else None."""
    key = os.environ.get("TYPESAFE_API_KEY")
    if key:
        return key.strip()
    try:
        return KEY_PATH.read_text().strip() or None
    except OSError:  # missing, unreadable, a directory -- all mean "no key"
        return None


def save_key(key: str) -> Path:
    """Write *key* to KEY_PATH readable by this user only, and return the
    path. Chmod before the write so the secret is never briefly world
    readable."""
    KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    KEY_PATH.touch(mode=0o600, exist_ok=True)
    KEY_PATH.chmod(0o600)
    KEY_PATH.write_text(key.strip() + "\n")
    return KEY_PATH


# Keyed by the paths themselves, so it survives a rescan the way hash_cache
# does. Only successes are stored -- caching a None would pin a transient
# network failure for the life of the process.
_cache: dict[tuple[str, ...], dict] = {}


def _ask(paths: tuple[str, ...], api_key: str) -> dict | None:
    questions = {
        "keep": {**_KEEP_Q, "criteria": {p: None for p in paths}},
        "same_photo": _SAME_PHOTO_Q,
    }
    payload = json.dumps({"state": list(paths), "model": MODEL, "questions": questions}).encode()
    request = urllib.request.Request(API_URL, payload, {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    })
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            answers = json.load(response)["answers"]
        return {
            "keep": paths.index(answers["keep"]["choice"]),
            "confidence": float(answers["keep"]["confidence"]),
            "same_photo": float(answers["same_photo"]["noul"]),
        }
    except (urllib.error.URLError, OSError, ValueError, KeyError):
        # ValueError covers both bad JSON and a `choice` that isn't one of
        # the paths we offered; either way the hint is simply unavailable.
        return None


def name_hint(paths: tuple[str, ...]) -> dict | None:
    """{"keep": index, "confidence": 0-1, "same_photo": 0-1} for *paths*, or
    None when the hint is unavailable. *paths* must be a tuple so it can key
    the cache."""
    if len(paths) < 2:
        return None
    if paths in _cache:
        return _cache[paths]
    api_key = load_key()
    if not api_key:
        return None
    result = _ask(paths, api_key)
    if result is not None:
        _cache[paths] = result
    return result
