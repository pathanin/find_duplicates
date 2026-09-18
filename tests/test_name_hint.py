"""name_hint must stay optional, and its key must be easy to set.

The hint is advisory: no API key, an unreachable API or a malformed response
all have to read as "no hint" and leave the review untouched, exactly like
brisque/niqe being absent. A raised exception here would 500 the group route
the reviewer is sitting on.

load_key's precedence is locked here too: $TYPESAFE_API_KEY wins over the
saved file, so a one-off `TYPESAFE_API_KEY=... python3 find_duplicates.py`
can override a stored key without editing anything.

Also locks two details that are easy to lose in a refactor. The cache stores
successes only -- caching a None would pin one transient network failure for
the life of the process. And the answer is converted to an *index* into the
paths tuple at the boundary, so nothing downstream matches on path strings.
"""

import io
import json
import sys
import tempfile
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import name_hint as nh

PATHS = ("Pictures/Originals/IMG_4821.jpg", "Downloads/IMG_4821 (1).jpg")


def fake_response(answers):
    def urlopen(request, timeout=None):
        body = json.dumps({"answers": answers, "usage": {}}).encode()
        stream = io.BytesIO(body)
        stream.__enter__ = lambda: stream
        stream.__exit__ = lambda *a: False
        return stream
    return urlopen


def raising(exc):
    def urlopen(request, timeout=None):
        raise exc
    return urlopen


GOOD = {
    "keep": {"type": "choice", "choice": PATHS[0], "confidence": 0.91},
    "same_photo": {"type": "noul", "noul": 0.88},
}


def test_key_file_round_trip() -> None:
    """save_key/load_key are what --set-typesafe-key is built on: the tool
    runs on a NAS or headless box reached over SSH, where an env var set in
    one shell doesn't survive the next login. The file must be readable by
    its owner only -- it holds a credential."""
    import os
    saved_env = os.environ.pop("TYPESAFE_API_KEY", None)
    saved_path = nh.KEY_PATH
    with tempfile.TemporaryDirectory() as tmp:
        nh.KEY_PATH = Path(tmp) / ".config" / "find_duplicates" / "typesafe-key"
        try:
            assert nh.load_key() is None, "no file and no env var means no key"

            written = nh.save_key("  secret-key-123  ")
            assert written == nh.KEY_PATH
            assert nh.load_key() == "secret-key-123", "surrounding whitespace must be stripped"
            mode = nh.KEY_PATH.stat().st_mode & 0o777
            assert mode == 0o600, f"key file must be owner-only, got {oct(mode)}"
            print("  ok  save_key writes an owner-only file that load_key reads back")

            os.environ["TYPESAFE_API_KEY"] = "env-key"
            assert nh.load_key() == "env-key", "the env var must win over the saved file"
            del os.environ["TYPESAFE_API_KEY"]
            print("  ok  $TYPESAFE_API_KEY overrides the saved file")

            # An unreadable or malformed file is "no key", never a crash --
            # it would otherwise take down every scan on that machine.
            nh.KEY_PATH.write_text("   \n")
            assert nh.load_key() is None, "a blank file means no key"
            nh.KEY_PATH.unlink()
            nh.KEY_PATH.mkdir()
            assert nh.load_key() is None, "a directory where the file should be means no key"
            print("  ok  a blank or unreadable key file degrades to None")
        finally:
            nh.KEY_PATH = saved_path
            if saved_env is not None:
                os.environ["TYPESAFE_API_KEY"] = saved_env


def main() -> None:
    test_key_file_round_trip()
    nh._cache.clear()

    # No key: no call at all, no hint. KEY_PATH is redirected somewhere
    # empty as well as the env var cleared -- otherwise this test passes or
    # fails depending on whether whoever runs it has saved a real key.
    calls = []
    nh.urllib.request.urlopen = lambda *a, **k: calls.append(1)
    import os
    saved = os.environ.pop("TYPESAFE_API_KEY", None)
    nh.KEY_PATH = Path(tempfile.gettempdir()) / "find-duplicates-no-such-key-file"
    assert nh.name_hint(PATHS) is None
    assert not calls, "must not reach the network without an API key"
    os.environ["TYPESAFE_API_KEY"] = "test-key"
    print("ok no key -> None, no request")

    # A single file is not a group.
    assert nh.name_hint(("only.jpg",)) is None
    print("ok single path -> None")

    # Happy path: the chosen path comes back as its index.
    nh.urllib.request.urlopen = fake_response(GOOD)
    hint = nh.name_hint(PATHS)
    assert hint == {"keep": 0, "confidence": 0.91, "same_photo": 0.88}, hint
    print("ok answer -> index, floats")

    # Cached: a second ask never hits the transport again.
    nh.urllib.request.urlopen = raising(AssertionError("should have been cached"))
    assert nh.name_hint(PATHS)["keep"] == 0
    print("ok success cached")

    # Failures degrade to None and are *not* cached.
    for exc in (urllib.error.URLError("down"), OSError("timed out")):
        nh._cache.clear()
        nh.urllib.request.urlopen = raising(exc)
        assert nh.name_hint(PATHS) is None, exc
        nh.urllib.request.urlopen = fake_response(GOOD)
        assert nh.name_hint(PATHS) is not None, "a failure must not be cached"
        nh._cache.clear()
    print("ok transport failure -> None, not cached")

    # A `choice` that isn't one of the offered paths must not raise.
    nh._cache.clear()
    nh.urllib.request.urlopen = fake_response({
        "keep": {"type": "choice", "choice": "something-else.jpg", "confidence": 0.5},
        "same_photo": {"type": "noul", "noul": 0.5},
    })
    assert nh.name_hint(PATHS) is None
    print("ok unknown choice -> None")

    # A response missing a question's answer entirely.
    nh._cache.clear()
    nh.urllib.request.urlopen = fake_response({"keep": {"choice": PATHS[0], "confidence": 0.5}})
    assert nh.name_hint(PATHS) is None
    print("ok missing answer -> None")

    if saved is not None:
        os.environ["TYPESAFE_API_KEY"] = saved
    else:
        os.environ.pop("TYPESAFE_API_KEY", None)
    print("all name_hint tests passed")


if __name__ == "__main__":
    main()
