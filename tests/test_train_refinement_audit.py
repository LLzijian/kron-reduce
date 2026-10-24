from __future__ import annotations

import pytest

from kron_reduce.evaluation.train_refinement_audit import (
    compare_saved_replay,
    summarize_refinement,
    verify_saved_replay,
)


def test_recursive_saved_replay_comparison_requires_complete_structure_and_exact_flags() -> None:
    expected = {"diagnostic": {"rho": 0.25, "hit": False, "index": None,
                               "nested": [1.0, 2.0]}}
    close = {"diagnostic": {"rho": 0.25 + 5e-13, "hit": False, "index": None,
                            "nested": [1.0, 2.0]}}
    assert compare_saved_replay(close, expected)["qualified"]
    assert not compare_saved_replay({"diagnostic": {**close["diagnostic"], "hit": 0}}, expected)["qualified"]
    assert not compare_saved_replay({"diagnostic": {"rho": 0.25, "hit": False}}, expected)["qualified"]
    assert not compare_saved_replay({"diagnostic": {**expected["diagnostic"], "extra": 1}}, expected)["qualified"]


def test_old_replay_verification_keeps_missing_events_and_summary_mismatches() -> None:
    ids = ("e1", "e2")
    row = {"status": "ok", "diagnostic": {"event_mse": 0.2}}
    good = verify_saved_replay(
        event_ids=ids, event_weights=(0.5, 0.5), counts=(80, 300),
        rebuilt_diagnostics={"e1": row, "e2": row}, saved_diagnostics={"e1": row, "e2": row},
        rebuilt_summary={"main_nrmse": 0.2}, saved_summary={"main_nrmse": 0.2 + 5e-13},
    )
    assert good["qualified"]
    bad = verify_saved_replay(
        event_ids=ids, event_weights=(0.5, 0.5), counts=(80, 300),
        rebuilt_diagnostics={"e1": row}, saved_diagnostics={"e1": row, "e2": row},
        rebuilt_summary={"main_nrmse": 0.2}, saved_summary={"main_nrmse": 0.21},
    )
    assert not bad["qualified"]
    assert bad["event_mismatch_count"] == 1
    assert bad["summary_mismatches"]


def _fixture(*, one_failed_refinement: bool = False, same_line: bool = False):
    def bound():
        return {"truth": {"rho": 0.2}, "prediction": {"rho": 0.19},
                "E_atom": 0.02, "E_vector": 0.03, "bound_verified": True}

    def representative():
        return {"start_index": 11, "end_index": 15,
                "coarse_margin_same_window": 0.2, "fine_margin_same_window": 0.19,
                "fixed_five_point_audit": bound()}

    ids = tuple(f"e{i:03d}" for i in range(258))
    metadata = {
        event_id: {"source_dataset": "source-a" if i % 2 == 0 else "source-b",
                   "fault_line": "Line_1" if same_line or i < 2 else f"Line_{i % 5}"}
        for i, event_id in enumerate(ids)
    }
    rows = {}
    refinements = {}
    for seed in (324, 325, 326):
        for step in (0.005, 0.0025):
            for event_id in ids:
                diagnostics = {
                    "event_mse": 0.25,
                    "future_points": 150,
                    "horizon_scores": {
                        "1.0": {"normalized_mse": 0.2},
                        "3.0": {"normalized_mse": 0.25},
                    },
                    "history_rho": {**bound(), "truth": {"rho": 0.2, "triggered": False}},
                    "future_rho": {**bound(), "truth": {"rho": 0.2, "triggered": True},
                                   "prediction": {"rho": 0.19, "triggered": False}},
                    "full_rho": bound(),
                    "representative_truth_window": {
                        **representative(),"prediction_margin_same_window":0.2 if step==.005 else .19,
                        "fixed_five_point_audit":{**bound(),"prediction":{"rho":0.2 if step==.005 else .19}}},
                    "representative_window_excess_over_event_mean_C_minus_pM": 0.1,
                }
                rows[(seed, step, event_id)] = {
                    "status": "ok", "physical": {"qualified": True}, "diagnostic": diagnostics,
                }
        for i, event_id in enumerate(ids):
            refinements[(seed, event_id)] = {
                "max_normalized_difference": 2e-6 if not (one_failed_refinement and i == 0) else 2e-5,
                "same_rule_decision": True,
                "trigger_within_two_points": True,
                "margin_bounds_verified": True,
                "coarse_fine_rule_margin_comparison": {
                    "history_rho": bound(), "future_rho": bound(), "full_rho": bound(),
                    "representative_truth_window": representative(),
                },
            }
    checks = {"source_scores": {"qualified": True}, "old_saved_replay": {"qualified": True}}
    return ids, metadata, rows, refinements, checks


def test_complete_population_support_is_deduplicated_by_physical_line_and_never_authorizes_training() -> None:
    ids, metadata, rows, refinements, checks = _fixture()
    result = summarize_refinement(ids, metadata, rows, refinements, checks, budget_complete=True)
    assert result["coverage"] == {
        "expected_rows": 1548, "present_rows": 1548, "missing_rows": 0, "extra_rows": 0,
        "expected_refinements": 774, "present_refinements": 774,
        "missing_refinements": 0, "extra_refinements": 0,
    }
    assert result["scientific_verdict"] == "ACCEPT"
    assert result["time_aggregation_proposal_supported"] is True
    assert len(result["supporting_events"]) == 258
    assert len(result["distinct_physical_fault_lines"]) == 5
    assert result["promotion_decision"] == "KEEP_BASELINE"
    assert result["new_training_authorized"] is False
    assert result["nonlinear_memory_supported_by_this_audit"] is False
    assert result["selected_candidate"] is None
    assert result["weighted_trajectory_summaries"]["324-0.005"]["event_count"] == 258


def test_one_failed_refinement_blocks_global_acceptance_even_when_support_events_exist() -> None:
    ids, metadata, rows, refinements, checks = _fixture(one_failed_refinement=True)
    result = summarize_refinement(ids, metadata, rows, refinements, checks, budget_complete=True)
    assert result["gates"]["support_event_count"] == 258
    assert result["gates"]["all_774_refinements_pass"] is False
    assert result["scientific_verdict"] == "REJECT"
    assert result["time_aggregation_proposal_supported"] is False
    assert result["new_training_authorized"] is False
    assert len(result["refinement_failures"]) == 3


def test_same_physical_line_across_sources_counts_once_and_source_failure_blocks() -> None:
    ids, metadata, rows, refinements, checks = _fixture(same_line=True)
    result = summarize_refinement(ids, metadata, rows, refinements, checks, budget_complete=True)
    assert len({r["source_dataset"] for r in result["supporting_events"]}) == 2
    assert result["gates"]["support_line_count"] == 1
    assert result["time_aggregation_proposal_supported"] is False
    checks["source_scores"]["qualified"] = False
    blocked = summarize_refinement(ids, metadata, rows, refinements, checks, budget_complete=True)
    assert blocked["scientific_verdict"] == "REJECT"
    assert blocked["time_aggregation_proposal_supported"] is False


def test_budget_and_missing_rows_fail_closed() -> None:
    ids, metadata, rows, refinements, checks = _fixture()
    del rows[(324, 0.0025, ids[0])]
    result = summarize_refinement(ids, metadata, rows, refinements, checks, budget_complete=False)
    assert result["coverage"]["missing_rows"] == 1
    assert result["gates"]["budget_complete"] is False
    assert result["scientific_verdict"] == "REJECT"
    assert result["time_aggregation_proposal_supported"] is False


def test_missing_or_false_true_path_bound_blocks_acceptance_even_with_other_flags_true() -> None:
    ids, metadata, rows, refinements, checks = _fixture()
    diagnostic = rows[(324, 0.005, ids[0])]["diagnostic"]
    diagnostic["history_rho"].pop("E_atom")
    result = summarize_refinement(ids, metadata, rows, refinements, checks, budget_complete=True)
    assert result["gates"]["all_774_paths_physical_and_status_pass"] is False
    assert result["scientific_verdict"] == "REJECT"

    ids, metadata, rows, refinements, checks = _fixture()
    rows[(324, 0.005, ids[0])]["diagnostic"]["full_rho"]["bound_verified"] = False
    result = summarize_refinement(ids, metadata, rows, refinements, checks, budget_complete=True)
    assert result["scientific_verdict"] == "REJECT"


@pytest.mark.parametrize("damage", ["rho_exceeds_atom", "atom_exceeds_vector", "nan_bound"])
def test_true_path_bound_arithmetic_must_pass_with_original_tolerance(damage: str) -> None:
    ids, metadata, rows, refinements, checks = _fixture()
    bound = rows[(324, 0.0025, ids[1])]["diagnostic"]["future_rho"]
    if damage == "rho_exceeds_atom":
        bound["prediction"]["rho"] = 0.3
    elif damage == "atom_exceeds_vector":
        bound["E_atom"] = 0.04
    else:
        bound["E_vector"] = float("nan")
    result = summarize_refinement(ids, metadata, rows, refinements, checks, budget_complete=True)
    assert result["scientific_verdict"] == "REJECT"
    assert result["gates"]["all_774_paths_physical_and_status_pass"] is False


def test_flat_refinement_flag_cannot_replace_four_coarse_fine_bounds() -> None:
    ids, metadata, rows, refinements, checks = _fixture()
    item = refinements[(325, ids[0])]
    item["coarse_fine_rule_margin_comparison"] = {"bound_verified": True}
    result = summarize_refinement(ids, metadata, rows, refinements, checks, budget_complete=True)
    assert result["gates"]["all_774_refinements_pass"] is False
    assert result["scientific_verdict"] == "REJECT"


@pytest.mark.parametrize(
    "damage", ["missing_category", "false_flag", "bad_inequality", "wrong_window", "wrong_pair_margin"])
def test_coarse_fine_bounds_and_shared_representative_window_are_required(damage: str) -> None:
    ids, metadata, rows, refinements, checks = _fixture()
    item = refinements[(326, ids[2])]
    comparison = item["coarse_fine_rule_margin_comparison"]
    if damage == "missing_category":
        comparison.pop("full_rho")
    elif damage == "false_flag":
        comparison["history_rho"]["bound_verified"] = False
    elif damage == "bad_inequality":
        comparison["future_rho"]["prediction"]["rho"] = 0.5
    elif damage == "wrong_pair_margin":
        comparison["representative_truth_window"]["fine_margin_same_window"] = 0.4
    else:
        comparison["representative_truth_window"]["start_index"] = 12
    result = summarize_refinement(ids, metadata, rows, refinements, checks, budget_complete=True)
    assert result["gates"]["all_774_refinements_pass"] is False
    assert result["scientific_verdict"] == "REJECT"


def test_coarse_fine_representative_window_must_match_both_true_path_windows() -> None:
    ids, metadata, rows, refinements, checks = _fixture()
    rows[(324, 0.0025, ids[0])]["diagnostic"]["representative_truth_window"]["end_index"] = 16
    result = summarize_refinement(ids, metadata, rows, refinements, checks, budget_complete=True)
    assert result["gates"]["all_774_refinements_pass"] is False
    assert result["scientific_verdict"] == "REJECT"


def test_all_missing_window_indices_cannot_pass_by_equal_none_pairs():
    ids, metadata, rows, refinements, checks = _fixture()
    for step in (.005,.0025):
        window = rows[324,step,ids[0]]["diagnostic"]["representative_truth_window"]
        window.pop("start_index")
        window.pop("end_index")
    window = refinements[324,ids[0]]["coarse_fine_rule_margin_comparison"]["representative_truth_window"]
    window.pop("start_index")
    window.pop("end_index")
    result = summarize_refinement(ids,metadata,rows,refinements,checks,budget_complete=True)
    assert result["scientific_verdict"] == "REJECT"


def test_fixed_margin_must_match_the_prediction_on_each_saved_path():
    ids,metadata,rows,refinements,checks = _fixture()
    rows[324,.0025,ids[0]]["diagnostic"]["representative_truth_window"]["prediction_margin_same_window"] = .4
    result = summarize_refinement(ids,metadata,rows,refinements,checks,budget_complete=True)
    assert result["scientific_verdict"] == "REJECT"
