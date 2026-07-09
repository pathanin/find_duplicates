"""Tests for score_group -- the money-math-sensitive scoring core.

score_group min-max normalizes per metric within the group, then combines
via METRIC_WEIGHTS into quality_score.  These tests cover the edge cases
that can arise from degenerate metrics (NaN, inf), same-valued metrics,
and empty-ish groups.

Run: python3 test_score_group.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import find_duplicates as fd


def _result(**kw):
    """Build a result dict with default values for every metric, then
    overlay caller-supplied overrides."""
    base = {
        "effective_resolution_px_equiv": 1000.0,
        "sharpness_normalized": 50.0,
        "effective_resolution_fraction": 0.8,
        "noise_sigma": 10.0,
        "blockiness": 5.0,
        "brisque": 20.0,
        "niqe": 3.0,
    }
    base.update(kw)
    return base


def test_scores_sorted_by_descending_quality() -> None:
    """Clear winner: higher on every positive metric, lower on every
    negative metric."""
    results = [_result(sharpness_normalized=100.0), _result(sharpness_normalized=20.0)]
    fd.score_group(results)
    assert results[0]["quality_score"] > results[1]["quality_score"], (
        f"first should score higher: {results[0]['quality_score']} vs {results[1]['quality_score']}"
    )


def test_scores_are_within_zero_one() -> None:
    """Normalized score should be between 0 and 1 for any valid inputs."""
    results = [_result(sharpness_normalized=10.0), _result(sharpness_normalized=90.0)]
    fd.score_group(results)
    for r in results:
        assert 0.0 <= r["quality_score"] <= 1.0, f"quality_score {r['quality_score']} out of [0,1]"


def test_equal_inputs_produce_equal_scores() -> None:
    """Two identical results should get identical quality scores."""
    results = [_result(), _result()]
    fd.score_group(results)
    assert results[0]["quality_score"] == results[1]["quality_score"]


def test_slightly_better_image_scores_higher() -> None:
    """A marginally better image should score higher than a marginally worse
    one -- the magnitude depends on the number of metrics and their values,
    so we just check ordering, not the diff."""
    results = [
        _result(sharpness_normalized=100.0, effective_resolution_px_equiv=2000),
        _result(sharpness_normalized=99.0, effective_resolution_px_equiv=1990),
    ]
    fd.score_group(results)
    assert results[0]["quality_score"] > results[1]["quality_score"], (
        f"best image must have highest score: {results[0]['quality_score']} vs {results[1]['quality_score']}"
    )


def test_hi_lo_degenerate_range() -> None:
    """When all group members have the same value for a metric, the
    range is hi==lo and score_group uses lo + 1e-9 to avoid div-by-zero."""
    results = [_result(noise_sigma=10.0), _result(noise_sigma=10.0)]
    fd.score_group(results)
    for r in results:
        assert math.isfinite(r["quality_score"]), f"quality_score should be finite, got {r['quality_score']}"


def test_nan_value_skips_that_metric() -> None:
    """A NaN metric value should cause that metric to be excluded from
    scoring for the entire group (not produce NaN scores everywhere)."""
    results = [
        _result(sharpness_normalized=50.0, noise_sigma=float("nan")),
        _result(sharpness_normalized=100.0, noise_sigma=5.0),
    ]
    fd.score_group(results)
    for r in results:
        assert math.isfinite(r["quality_score"]), f"quality_score should be finite, got {r['quality_score']}"


def test_inf_value_skips_that_metric() -> None:
    """An inf metric value should cause that metric to be excluded from
    scoring for the entire group."""
    results = [
        _result(sharpness_normalized=float("inf")),
        _result(sharpness_normalized=100.0),
    ]
    fd.score_group(results)
    for r in results:
        assert math.isfinite(r["quality_score"]), f"quality_score should be finite, got {r['quality_score']}"


def test_none_value_skips_that_metric() -> None:
    """A None metric value (e.g. optional BRISQUE/NIQE not installed) should
    cause that metric to be skipped for the whole group."""
    results = [
        _result(brisque=None),
        _result(brisque=15.0),
    ]
    fd.score_group(results)
    for r in results:
        assert math.isfinite(r["quality_score"]), f"quality_score should be finite, got {r['quality_score']}"


def test_all_metrics_skipped_falls_back_to_zero() -> None:
    """If every metric is NaN/inf/None, the total_weight becomes 0 and we
    default to 1.0 divisor, producing quality_score=0.0 for all."""
    results = [
        _result(sharpness_normalized=float("nan"), effective_resolution_px_equiv=float("nan"),
                effective_resolution_fraction=float("nan"), noise_sigma=float("nan"),
                blockiness=float("nan"), brisque=float("nan"), niqe=float("nan")),
        _result(sharpness_normalized=float("nan"), effective_resolution_px_equiv=float("nan"),
                effective_resolution_fraction=float("nan"), noise_sigma=float("nan"),
                blockiness=float("nan"), brisque=float("nan"), niqe=float("nan")),
    ]
    fd.score_group(results)
    for r in results:
        assert r["quality_score"] == 0.0, f"quality_score should be 0.0 when all metrics skipped, got {r['quality_score']}"


def test_single_result_is_finite() -> None:
    """A group with only one member has all hi==lo degenerate ranges.
    Positive-weight metrics contribute 0 at the low end; negative-weight
    ('lower better') metrics contribute their full abs(weight) because
    (1 - 0) * abs(weight) = abs(weight).  The score is between 0 and 1
    and finite -- that's the property to lock down, not the exact value."""
    results = [_result()]
    fd.score_group(results)
    assert math.isfinite(results[0]["quality_score"]), (
        f"single-item group score should be finite, got {results[0]['quality_score']}"
    )
    assert 0.0 < results[0]["quality_score"] < 1.0, (
        f"single-item group score should be in (0,1), got {results[0]['quality_score']}"
    )


def test_empty_results_scores_zero() -> None:
    """An empty results list should not crash and sets no score on anything."""
    fd.score_group([])


def test_negative_weight_metric_orders_correctly() -> None:
    """For a 'lower better' metric (negative weight), the smaller value
    should contribute positively to the score."""
    results = [
        _result(noise_sigma=5.0, sharpness_normalized=50.0),
        _result(noise_sigma=100.0, sharpness_normalized=50.0),
    ]
    fd.score_group(results)
    assert results[0]["quality_score"] > results[1]["quality_score"], (
        f"image with lower noise should score higher: {results[0]['quality_score']} vs {results[1]['quality_score']}"
    )


def main() -> None:
    tests = [
        ("scores sorted by descending quality", test_scores_sorted_by_descending_quality),
        ("scores are within 0-1", test_scores_are_within_zero_one),
        ("equal inputs produce equal scores", test_equal_inputs_produce_equal_scores),
        ("slightly better scores higher", test_slightly_better_image_scores_higher),
        ("hi==lo degenerate range", test_hi_lo_degenerate_range),
        ("NaN value skips metric", test_nan_value_skips_that_metric),
        ("inf value skips metric", test_inf_value_skips_that_metric),
        ("None value skips metric", test_none_value_skips_that_metric),
        ("all metrics skipped", test_all_metrics_skipped_falls_back_to_zero),
        ("single result is finite", test_single_result_is_finite),
        ("empty results", test_empty_results_scores_zero),
        ("negative weight orders correctly", test_negative_weight_metric_orders_correctly),
    ]
    for name, fn in tests:
        try:
            fn()
            print(f"ok {name}")
        except Exception as e:
            print(f"FAIL {name}: {e}")
            sys.exit(1)
    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    main()
