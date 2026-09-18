"""brisque/niqe are optional, and a *broken* install must cost one attempt.

A missing package is cheap to retry. A package that imports and then fails
is not: brisque 0.2.0 computes its whole feature set before dying on modern
numpy, which measured 418 ms per image -- analyze() went from 25 ms to
443 ms -- to return None every time. analyze() is the per-image scan
bottleneck, so anyone following the docstring's `pip install brisque[...]`
silently got a 17x slower scan and an "n/a" column either way.

So each metric latches after its first failure. These tests inject fake
packages through sys.modules (the real imports are inside the functions),
count how often each is reached, and check the happy path still works --
a latch that fires on a working install would be worse than the bug.

Run: python3 tests/test_optional_metrics.py
"""

import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import compare_image_quality as ciq

IMG = np.zeros((32, 32, 3), dtype=np.uint8)


def fake_brisque(score=None, raises=False):
    """A stand-in `brisque` module counting how often BRISQUE() is reached."""
    module = types.ModuleType("brisque")
    module.calls = 0

    class BRISQUE:
        def __init__(self, url=False):
            module.calls += 1
            if raises:
                # Where the real package dies: after the expensive part.
                raise TypeError("only 0-dimensional arrays can be converted to Python scalars")

        def score(self, img):
            return score

    module.BRISQUE = BRISQUE
    return module


def fake_pyiqa(raises=False):
    module = types.ModuleType("pyiqa")
    module.calls = 0

    def create_metric(name, device=None):
        module.calls += 1
        if raises:
            raise RuntimeError("model download failed")
        return lambda path: 3.5

    module.create_metric = create_metric
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    return module, torch


def install(**modules):
    saved = {k: sys.modules.get(k) for k in modules}
    sys.modules.update(modules)
    return saved


def restore(saved):
    for name, module in saved.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def test_working_brisque_is_not_latched() -> None:
    ciq._brisque_unavailable = False
    module = fake_brisque(score=42.0)
    saved = install(brisque=module)
    try:
        assert ciq.brisque_score(IMG) == 42.0
        assert ciq.brisque_score(IMG) == 42.0
        assert module.calls == 2, f"a working install must run every time, got {module.calls} calls"
        assert ciq._brisque_unavailable is False
    finally:
        restore(saved)
    print("  ok  a working brisque runs on every image")


def test_broken_brisque_is_tried_once() -> None:
    ciq._brisque_unavailable = False
    module = fake_brisque(raises=True)
    saved = install(brisque=module)
    try:
        assert ciq.brisque_score(IMG) is None
        assert module.calls == 1
        for _ in range(5):
            assert ciq.brisque_score(IMG) is None
        assert module.calls == 1, (
            f"a broken brisque must be attempted once, not per image -- got {module.calls} calls"
        )
    finally:
        restore(saved)
        ciq._brisque_unavailable = False
    print("  ok  a broken brisque costs one attempt, not one per image")


def test_missing_brisque_still_returns_none() -> None:
    """Boundary: the package absent entirely, which is the common case."""
    ciq._brisque_unavailable = False
    saved = install()
    real = sys.modules.pop("brisque", None)
    try:
        assert ciq.brisque_score(IMG) is None
    finally:
        if real is not None:
            sys.modules["brisque"] = real
        restore(saved)
        ciq._brisque_unavailable = False
    print("  ok  a missing brisque returns None")


def test_broken_niqe_is_tried_once() -> None:
    ciq._niqe_unavailable = False
    module, torch = fake_pyiqa(raises=True)
    saved = install(pyiqa=module, torch=torch)
    try:
        assert ciq.niqe_score("whatever.jpg") is None
        for _ in range(5):
            assert ciq.niqe_score("whatever.jpg") is None
        assert module.calls == 1, (
            f"create_metric builds a torch model per call -- got {module.calls} calls"
        )
    finally:
        restore(saved)
        ciq._niqe_unavailable = False
    print("  ok  a broken niqe costs one attempt, not one per image")


def test_analyze_still_reports_none_for_both() -> None:
    """The latch must not change what analyze() puts in the result dict --
    the metrics table and score_group both key off these being None."""
    ciq._brisque_unavailable = True
    ciq._niqe_unavailable = True
    try:
        import tempfile
        import cv2
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.jpg"
            rng = np.random.default_rng(3)
            cv2.imwrite(str(path), rng.integers(0, 255, (120, 160, 3), dtype=np.uint8))
            result = ciq.analyze(str(path))
        assert result["brisque"] is None and result["niqe"] is None, result
        assert result["sharpness_normalized"] is not None, "the real metrics must be unaffected"
    finally:
        ciq._brisque_unavailable = False
        ciq._niqe_unavailable = False
    print("  ok  analyze() still reports brisque/niqe as None once latched")


def main() -> None:
    for fn in (
        test_working_brisque_is_not_latched,
        test_broken_brisque_is_tried_once,
        test_missing_brisque_still_returns_none,
        test_broken_niqe_is_tried_once,
        test_analyze_still_reports_none_for_both,
    ):
        print(f"{fn.__name__}:")
        fn()
    print("all optional-metric tests passed")


if __name__ == "__main__":
    main()
