import pytest

from kron_reduce.evaluation.unified_discovery_m3 import _choose
from kron_reduce.evaluation.unified_discovery_m3_scores import checkpoint, equivalent

STEPS = ("0.0025", "0.00125")
GATES = {"max_fp": 1, "diagnostic_fp": 2, "stable_median_limit": .21, "tn_mean_limit": .24, "witness_ratio": .8}
REFERENCE = {dt: {"signed_median": 2.55, "absolute_median": 2.62} for dt in STEPS}


def _candidate(name, support, *, median=.195, tn=.225, peak=.82, fp=1, signed=2.03, absolute=2.61):
    val = {dt: {"stability": {"allocated": 107, "tp": 12, "fp": fp, "unsettled_predictions": 0,
            "numerical_failures": 0, "uncovered": 0}, "task3_six_term_score": .2,
          "stable_errors": {"qualified_stable_8s": 95, "median_8s": median, "tn_mean_8s": tn},
          "first_500ms_peaks": {"qualified": 12, "positive_median": peak}} for dt in STEPS}
    witness = {dt: {"qualified": 33, "signed_median": signed, "absolute_median": absolute} for dt in STEPS}
    train = {dt: {"stability": {"allocated": 346, "uncovered": 0, "numerical_failures": 0},
                  "task3_six_term_score": .2} for dt in STEPS}
    row = {"name": name, "support": support, "completed_updates": 15, "scheduled_updates": 15}
    return row | {"gate": checkpoint(val, witness, REFERENCE, GATES, 0, train=train)}


@pytest.mark.parametrize("bad", ("stable", "numerical", "witness"))
def test_refined_step_is_an_independent_hard_gate(bad):
    candidate = _candidate("full", ["G1", "G2"])
    stable = candidate["gate"]["val"]
    witness = candidate["gate"]["witness"]
    dt = STEPS[1]
    assert candidate["gate"]["passed"]
    if bad == "stable":
        stable[dt]["stable_errors"]["median_8s"] = .21001
    elif bad == "numerical":
        stable[dt]["stability"]["numerical_failures"] = 1
    else:
        witness[dt]["absolute_median"] = REFERENCE[dt]["absolute_median"]
    assert not checkpoint(stable, witness, REFERENCE, GATES, 0)["passed"]


def test_parsimony_takes_only_strict_complete_deletions_in_band():
    full = _candidate("full", ["G1", "G2", "G3"])
    relaxed = _candidate("relaxed", ["G1"], fp=2)
    near = _candidate("near", ["G1", "G2"], peak=.831)
    worse = _candidate("worse", ["G1"], peak=.86)
    incomplete = _candidate("incomplete", ["G2"])
    incomplete["completed_updates"] = 12
    assert relaxed["gate"]["relaxed_pass"] and not relaxed["gate"]["passed"]
    assert equivalent(near, full, .02) and not equivalent(worse, near, .02)
    assert _choose([full, relaxed, near, worse, incomplete], .02) == (near, True)
    assert _choose([relaxed, incomplete], .02)[1] is False
