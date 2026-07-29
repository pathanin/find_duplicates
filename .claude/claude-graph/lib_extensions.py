#!/usr/bin/env python3
"""Which files count as source, shared by the generator and the verifier.

Single-sourced deliberately. When these lists were duplicated, adding a language
to build_graph.py left verify_graph.py unable to read it, so the staleness check
reported that language's real symbols as dead and blocked the turn. Any change
here must hold for both scripts: the generator decides what to index, the
verifier decides what it can confirm still exists, and a symbol the generator
can see but the verifier cannot is a guaranteed false positive.
"""

SKIP_PARTS = {
    ".git", "node_modules", "vendor", "dist", "build", "target", "__pycache__",
    ".venv", "venv", ".next", ".cache", "coverage", "testdata", "fixtures",
}

CODE_EXT = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".rb", ".java", ".kt",
    ".c", ".h", ".cc", ".cpp", ".hpp", ".cs", ".php", ".swift", ".scala",
    ".sh", ".bash", ".zsh", ".lua", ".vue", ".svelte", ".ex", ".exs",
    ".mjs", ".cjs", ".mts", ".cts", ".ps1", ".psm1", ".pl", ".pm", ".r", ".jl",
    ".dart", ".m", ".mm", ".groovy", ".tf", ".sql", ".qml", ".proto",
    # Both have validated fixtures and RECALL cases in test_symbols.py
    # (haskell.hs, zig.zig) but were missing here, so the generator's own file
    # filter dropped every Haskell/Zig source file before extraction ever ran —
    # a repo in either language would graph as if those files did not exist,
    # with no warning, since unsupported_languages() only inspects files that
    # were read in the first place.
    ".hs", ".zig",
}
