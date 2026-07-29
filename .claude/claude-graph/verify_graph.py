#!/usr/bin/env python3
"""Check .claude/graph.md against the repo and report where it has started lying.

Invoked from hooks/hooks.json on Stop (blocking) and SessionStart (advisory).
Python 3 stdlib only. Exits 0 always — a broken checker must never break a turn.

Three checks:
  1. dead path    — a path named in the graph no longer exists            (hard)
  2. dead symbol  — a symbol in a Key-symbols cell is gone from its module (hard)
  3. unmapped dir — a code directory with no row in the graph             (soft)

New symbols appearing in a mapped module are NOT drift: "Key symbols" is a
curated selection, not an inventory. Flagging those would nag on every edit.
"""

import glob
import json
import os
import re
import subprocess
import sys

# Imported, never redefined. These two scripts drifted once: build_graph.py
# gained 15 extensions (.ps1, .tf, .jl, …) and this file kept its own older
# copy, so it could not read a PowerShell module and reported that module's real
# symbols as dead — a false block on every repo in a newly added language.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_extensions import CODE_EXT, SKIP_PARTS  # noqa: E402

MAX_BYTES = 512 * 1024
UNMAPPED_MIN_FILES = 3
BACKTICKED = re.compile(r"`([^`\n]+)`")


def sh(args, cwd):
    return subprocess.run(
        args, cwd=cwd, capture_output=True, text=True, timeout=20
    ).stdout


def skipped(path):
    return any(part in SKIP_PARTS for part in path.split("/"))


def parse_graph(text):
    """Return (paths, symbols_by_module) mentioned in the graph.

    Paths: every backticked token anywhere in the file that looks like a repo
    path. Symbols: backticked tokens in the third column of table rows only.
    """
    paths, symbols = set(), {}
    for line in text.splitlines():
        if line.lstrip().startswith("<!--"):
            continue
        for token in BACKTICKED.findall(line):
            token = token.strip()
            if "/" in token or os.path.splitext(token)[1] in CODE_EXT:
                paths.add(token)  # trailing slash preserved; it is a signal
        if not line.lstrip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3 or set(cells[0]) <= set("-: "):
            continue
        owner = BACKTICKED.findall(cells[0])
        if not owner:
            continue
        names = [s.strip() for s in BACKTICKED.findall(cells[2])]
        if names:
            symbols.setdefault(owner[0].strip().rstrip("/"), set()).update(names)
    return paths, symbols


def symbol_pattern(name):
    r"""Presence test for `name`, anchored on identifier characters.

    `\b` is defined against `\w`, so it does not match before a leading `$` or
    after a trailing one: `\b\$destroy\b` never matches the literal text
    `$destroy`. That reported every `$`-prefixed symbol as deleted — Svelte's
    `$set`/`$on`/`$destroy`, jQuery, PHP variables, shell. Anchor on "not an
    identifier character" instead, and only on the sides where the name itself
    starts or ends with one.
    """
    esc = re.escape(name)
    left = r"(?<![\w$])" if (name[0].isalnum() or name[0] in "_$") else ""
    right = r"(?![\w$])" if (name[-1].isalnum() or name[-1] in "_$") else ""
    return left + esc + right


ROOT_ROW = ("(repo root)", ".")


def module_text(root, module, cache):
    """Concatenated source of every code file under `module`."""
    if module in cache:
        return cache[module]
    abs_module = os.path.join(root, module)
    blob = []
    if module in ROOT_ROW:
        # build_graph.py labels files sitting directly in the repo root
        # `(repo root)`, which names no directory on disk — so this loop used to
        # skip the row outright and never checked a symbol listed under it. A
        # ghost symbol injected there passed verification on croc while the same
        # injection was caught on gin, whose first row is a real path. Read the
        # root's direct children only, matching how the generator groups them
        # (by `os.path.dirname`); walking would pull in the whole repo.
        files = [os.path.join(root, f) for f in os.listdir(root)
                 if os.path.isfile(os.path.join(root, f))]
    elif os.path.isfile(abs_module):
        files = [abs_module]
    else:
        files = []
        for dirpath, dirnames, filenames in os.walk(abs_module):
            dirnames[:] = [d for d in dirnames if d not in SKIP_PARTS]
            files.extend(os.path.join(dirpath, f) for f in filenames)
    for f in files:
        if os.path.splitext(f)[1] not in CODE_EXT:
            continue
        try:
            if os.path.getsize(f) > MAX_BYTES:
                continue
            with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                blob.append(fh.read())
        except OSError:
            continue
    cache[module] = "\n".join(blob)
    return cache[module]


def check(root):
    graph_path = os.path.join(root, ".claude", "graph.md")
    if not os.path.isfile(graph_path):
        return None
    with open(graph_path, "r", encoding="utf-8", errors="ignore") as fh:
        text = fh.read()
    if "claude-graph: no-verify" in text:
        return None

    paths, symbols = parse_graph(text)
    tracked = [p for p in sh(["git", "ls-files"], root).splitlines() if p]
    top_level = {p.split("/")[0] for p in tracked}

    dead_paths = []
    for raw in sorted(paths):
        p = raw.rstrip("/")
        if p.startswith(("@", "~", "http")) or skipped(p):
            continue
        # `*`/`?` mark an actual glob pattern, not a claim that one path
        # exists. Honor it if anything matches.
        #
        # `[` used to be in this charset too, on the same theory. But a
        # bracketed path is far more often a literal directory or file name —
        # Next.js/SvelteKit/Expo Router's dynamic-route convention writes
        # `[hostId]`, `[...notfound]` straight into the filesystem — than an
        # author-written glob. Routing it through `glob.glob` here means the
        # bracket is read as a character class instead of literal text, so
        # `mobile/app/h/[hostId]/accounts.tsx` (a real, tracked file) matches
        # nothing and blocks. That is a false block on every Stop hook for any
        # repo using the convention: it hit 2 of 3 repos in one benchmark
        # batch. Falling bracketed paths through to the literal checks below
        # fixes real ones; a bracketed path that is genuinely dead still has
        # no literal match and still gets reported.
        if any(ch in p for ch in "*?"):
            if not glob.glob(os.path.join(root, p)):
                dead_paths.append(p)
            continue
        # Distinguish repo paths from package specifiers. A trailing slash or a
        # code extension settles it outright; otherwise the first segment has to
        # name something the repo actually has, so `lodash/merge` and
        # `github.com/foo/bar` are left alone.
        strong = raw.endswith("/") or os.path.splitext(p)[1] in CODE_EXT
        if not strong and p.split("/")[0] not in top_level:
            continue
        if os.path.exists(os.path.join(root, p)):
            continue
        # A bare filename is a legitimate way to name a file the row already
        # locates ("`static/` … `app.js`"), so resolve it anywhere in the repo
        # before calling it dead. Only a path with a directory component is
        # held to its exact location.
        if "/" not in p and any(os.path.basename(t) == p for t in tracked):
            continue
        dead_paths.append(p)

    dead_syms, cache = [], {}
    for module, names in sorted(symbols.items()):
        if module not in ROOT_ROW and (
                module in dead_paths
                or not os.path.exists(os.path.join(root, module))):
            continue
        blob = module_text(root, module, cache)
        if not blob:
            continue
        for name in sorted(names):
            if not re.search(symbol_pattern(name), blob):
                dead_syms.append((module, name))

    mapped = {p.rstrip("/") for p in paths}
    counts = {}
    for f in tracked:
        if skipped(f) or os.path.splitext(f)[1] not in CODE_EXT or "/" not in f:
            continue
        counts[os.path.dirname(f)] = counts.get(os.path.dirname(f), 0) + 1
    unmapped = sorted(
        d for d, n in counts.items()
        if n >= UNMAPPED_MIN_FILES
        and not any(d == m or d.startswith(m + "/") for m in mapped)
    )
    return dead_paths, dead_syms, unmapped


def main():
    event = sys.argv[1] if len(sys.argv) > 1 else "Stop"
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except ValueError:
        payload = {}
    if payload.get("stop_hook_active"):
        return  # already re-prompted once this turn; never loop

    cwd = payload.get("cwd") or os.getcwd()
    root = sh(["git", "rev-parse", "--show-toplevel"], cwd).strip()
    if not root:
        return
    result = check(root)
    if result is None:
        return
    dead_paths, dead_syms, unmapped = result

    lines = []
    if dead_paths:
        lines.append(
            "Dead paths in .claude/graph.md: "
            + ", ".join("`%s`" % p for p in dead_paths[:8])
        )
    if dead_syms:
        lines.append(
            "Symbols named in .claude/graph.md that no longer exist: "
            + ", ".join("`%s` in `%s`" % (n, m) for m, n in dead_syms[:8])
        )

    if event == "SessionStart":
        if unmapped:
            lines.append(
                "Directories with no row in .claude/graph.md: "
                + ", ".join("`%s`" % d for d in unmapped[:8])
            )
        if lines:
            print(json.dumps({"hookSpecificOutput": {
                "hookEventName": "SessionStart",
                "additionalContext": ".claude/graph.md is stale. "
                + " ".join(lines)
                + " Update the affected rows before relying on the graph to locate code.",
            }}))
        return

    if lines:
        print(json.dumps({"decision": "block", "reason": (
            "The repo graph now points at code that moved or was removed, so it "
            "will send agents to the wrong place. " + " ".join(lines)
            + " Update only the affected rows in .claude/graph.md — re-derive the "
            "path and key symbols from the current code, leave unrelated rows "
            "alone — then refresh the generated-at comment in its header."
        )}))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass
    sys.exit(0)
