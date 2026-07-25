"""Regression tests for metric-row labeling in duplicates_core.py.

A raw number is meaningless without knowing which direction is "better", so
every METRIC_ROWS label must say so (or say it isn't scored at all, for
dimensions/file size), and every weighted metric must have both a display
row and a help description -- both frontends render their help sheet
straight off METRIC_WEIGHTS/METRIC_DESCRIPTIONS/METRIC_ROWS rather than
hardcoding it, so any of the three drifting out of sync silently breaks
that.

Run: python3 test_help_and_labels.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import duplicates_core as dc

REFERENCE_ONLY_ROWS = {"Dimensions", "File size"}  # not part of the score; explained in the help screen instead


def test_every_scored_metric_row_states_its_direction() -> None:
    for row in dc.METRIC_ROWS:
        if row.label in REFERENCE_ONLY_ROWS:
            continue
        assert "better" in row.label, (
            f"metric row {row.label!r} doesn't state a direction -- "
            "a bare number here is meaningless without one"
        )
    print("  ok  every scored METRIC_ROWS label states a direction")


def test_metric_row_direction_matches_kind_and_weight_sign() -> None:
    """MetricRow.direction is a computed property (see duplicates_core.py):
    for a "metric" row it's *defined* as sign(METRIC_WEIGHTS[row.key]), so it
    can never mathematically disagree with its own key's weight -- asserting
    that agreement would be vacuous, always true by construction. What can
    still drift is `key` itself pointing at the wrong metric, or the
    human-readable label wording disagreeing with what `key` implies. This
    reuses the co_consts introspection from
    test_every_weighted_metric_has_a_display_row to check the first (a
    "metric" row's declared key must be the same dict key its own `fn`
    lambda actually reads), and checks the second directly against the
    label text."""
    for row in dc.METRIC_ROWS:
        if row.kind == "reference":
            assert row.direction == 0 and row.key is None, f"{row.label!r}: reference row should have no key/direction"
            continue
        if row.kind == "score":
            assert row.direction == 1, f"{row.label!r}: the score row should always be direction 1"
            continue
        assert row.kind == "metric", f"unrecognized MetricRow.kind {row.kind!r} for {row.label!r}"
        assert row.key is not None, f"{row.label!r} is a metric row but has no METRIC_WEIGHTS key"
        assert row.key in dc.METRIC_WEIGHTS, f"{row.label!r}'s key {row.key!r} isn't in METRIC_WEIGHTS"
        referenced = {c for c in row.fn.__code__.co_consts if isinstance(c, str)}
        assert row.key in referenced, (
            f"{row.label!r} declares key {row.key!r} but its own lambda never reads r[{row.key!r}] -- "
            "the declared key and the value actually being formatted have drifted apart"
        )
        label_says_higher = "higher better" in row.label
        label_says_lower = "lower better" in row.label
        assert label_says_higher or label_says_lower, f"{row.label!r} doesn't state a direction"
        expected = 1 if label_says_higher else -1
        assert row.direction == expected, (
            f"{row.label!r} says {'higher' if label_says_higher else 'lower'} better, "
            f"but key {row.key!r}'s weight ({dc.METRIC_WEIGHTS[row.key]!r}) implies direction {row.direction}"
        )
    print("  ok  every metric row's key matches what it formats, and its label agrees with that key's weight sign")


def test_every_weighted_metric_has_a_description() -> None:
    for name in dc.METRIC_WEIGHTS:
        assert name in dc.METRIC_DESCRIPTIONS, f"METRIC_DESCRIPTIONS is missing weighted metric {name!r}"
    print("  ok  every weighted metric has a help description")


def test_every_weighted_metric_has_a_display_row() -> None:
    """METRIC_WEIGHTS and METRIC_ROWS are two separate structures kept in
    sync by convention only: a metric added to METRIC_WEIGHTS but never given
    a METRIC_ROWS row would silently affect the score without ever being
    shown to the user. Each row's rendering function is a lambda that
    dict-subscripts its result by the metric's key, e.g. `r['niqe']` -- that
    key literal shows up in the lambda's compiled constants, so we can check
    every weighted metric is actually referenced by some row without needing
    to render the table."""
    referenced = set()
    for row in dc.METRIC_ROWS:
        referenced.update(c for c in row.fn.__code__.co_consts if isinstance(c, str))
    for name in dc.METRIC_WEIGHTS:
        assert name in referenced, (
            f"{name!r} is scored (in METRIC_WEIGHTS) but no METRIC_ROWS row references it -- "
            "it would silently affect quality_score without ever being displayed"
        )
    print("  ok  every weighted metric has a corresponding METRIC_ROWS display row")


def test_detector_catches_a_dropped_display_row() -> None:
    """Proof the check above can actually fail: with the NIQE row removed,
    the check must flag 'niqe' as no longer referenced."""
    rows_without_niqe = [row for row in dc.METRIC_ROWS if "NIQE" not in row.label]
    assert len(rows_without_niqe) == len(dc.METRIC_ROWS) - 1, "expected to drop exactly one row (NIQE)"
    referenced = set()
    for row in dc.METRIC_ROWS:
        if "NIQE" in row.label:
            continue
        referenced.update(c for c in row.fn.__code__.co_consts if isinstance(c, str))
    assert "niqe" not in referenced, "dropping the NIQE row should have removed 'niqe' from referenced keys"
    print("  ok  the coupling check correctly flags a dropped display row (not vacuous)")


def main() -> None:
    for test in (
        test_every_scored_metric_row_states_its_direction,
        test_metric_row_direction_matches_kind_and_weight_sign,
        test_every_weighted_metric_has_a_description,
        test_every_weighted_metric_has_a_display_row,
        test_detector_catches_a_dropped_display_row,
    ):
        print(f"{test.__name__}:")
        test()
    print("all help/label tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)
