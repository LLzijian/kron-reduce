from __future__ import annotations

from pathlib import Path

import pytest

from kron_reduce.workflows import p1_a04_postprocess as post


def _row(
    event_id: str,
    *,
    source: str = "source-a",
    horizon: int = 1,
    status: str = "completed",
    target_sum: float = 0.0,
    target_square: float = 10.0,
    rmse: float = 0.1,
    coi: float = 1e-14,
) -> dict:
    output = {
        "count_per_node": horizon * 100 + 1,
        "sse_by_node": [1.0] * 10,
        "target_sum_by_node": [target_sum] * 10,
        "target_sum_squares_by_node": [target_square] * 10,
        "rmse": rmse,
        "coi_drift": coi,
    }
    return {
        "event_id": event_id,
        "source_dataset": source,
        "fault_line": "line-a",
        "horizon_seconds": horizon,
        "status": status,
        "outputs": {"delta": dict(output), "omega": dict(output)},
    }


def test_aggregate_uses_per_event_per_node_centering_and_retains_tail() -> None:
    rows = [_row("e1", rmse=0.1), _row("e2", rmse=0.3, coi=2e-14)]
    result = post._aggregate_rows(
        rows,
        expected_ids={"e1", "e2"},
        source=None,
        horizon=1,
        output="omega",
    )
    assert result["pooled_sse"] == pytest.approx(20.0)
    assert result["pooled_sst_per_event_node_centered"] == pytest.approx(200.0)
    assert result["pooled_r2"] == pytest.approx(0.9)
    assert result["worst_10pct_mean_rmse"] == pytest.approx(0.3)
    assert result["worst_10pct_event_ids"] == ["e2"]
    assert result["maximum_coi_drift"] == pytest.approx(2e-14)


def test_aggregate_does_not_replace_local_centering_with_global_variance() -> None:
    # Each event/node is temporally constant. A global mean across the two
    # events would manufacture variance, while the registered local centering
    # correctly leaves zero SST and refuses an R2 claim.
    count = 101
    rows = [
        _row("e1", target_sum=float(count), target_square=float(count)),
        _row("e2", target_sum=-float(count), target_square=float(count)),
    ]
    with pytest.raises(ValueError, match="target variance is zero"):
        post._aggregate_rows(
            rows,
            expected_ids={"e1", "e2"},
            source=None,
            horizon=1,
            output="omega",
        )


@pytest.mark.parametrize(
    "rows,expected,message",
    [
        ([_row("e1")], {"e1", "e2"}, "missing or duplicated"),
        ([_row("e1", status="failed")], {"e1"}, "failed integration"),
    ],
)
def test_aggregate_fails_closed_on_population_or_integration_failure(
    rows: list[dict], expected: set[str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        post._aggregate_rows(
            rows,
            expected_ids=expected,
            source=None,
            horizon=1,
            output="delta",
        )


def test_metric_payload_requires_one_object_not_a_list() -> None:
    payload = {"role": "internal_audit"}
    assert post._metric_payload({"metrics": payload}) is payload
    with pytest.raises(TypeError, match="one internal-audit metric"):
        post._metric_payload({"metrics": [payload]})


def _question(name: str) -> dict:
    if name == "Q2_why_long_horizon_speed_error_is_large":
        output = {
            "event_count": 43,
            "worst_10pct_mean_rmse": 1.0,
            "worst_10pct_event_ids": [f"e{i}" for i in range(5)],
        }
        return {
            "machine_checks": {"complete": True},
            "claim_boundary": "bounded",
            "evidence": {
                "arm4_interpretation": "frozen-core sensitivity; not a refitted conditional upper bound",
                "five_arm": {
                    "arms": {
                        str(arm): {
                            "aggregates": {
                                "ALL": {
                                    str(horizon): {
                                        "delta": dict(output),
                                        "omega": dict(output),
                                    }
                                    for horizon in (1, 5, 10)
                                }
                            }
                        }
                        for arm in range(1, 6)
                    }
                },
            },
        }
    return {
        "machine_checks": {"complete": True},
        "claim_boundary": "bounded",
        "evidence": {},
    }


def _diagnosis() -> dict:
    names = {
        "Q1_modules_change_prediction_or_degenerate_to_B2",
        "Q2_why_long_horizon_speed_error_is_large",
        "Q3_modal_audit_shared_poles_and_local_observability",
    }
    return {
        "questions": {name: _question(name) for name in names},
        "scientific_verdict": None,
        "promotion_decision": "KEEP_BASELINE",
        "level_6_claimed": False,
    }


def _grid() -> dict:
    return {
        "event_count": 178,
        "shifted_event_count": 69,
        "no_extrapolation_verified": True,
    }


def _retry_access() -> dict:
    return {
        "only_postclear_time_read": True,
        "raw_state_or_pm_read": False,
        "test_ids_read": [],
        "other_folds_read": [],
    }


def _controlled_access() -> dict:
    return {
        "test_ids_read": [],
        "other_folds_read": [],
        "labels_read": False,
        "pe_or_voltage_read": False,
        "source_or_fault_metadata_used_for_routing": False,
        "future_pm_used_as_main_model_input": False,
    }


def test_adversarial_contract_passes_complete_bounded_diagnosis() -> None:
    checks = post._adversarial_checks(
        _diagnosis(), _grid(), _retry_access(), _controlled_access()
    )
    assert checks
    assert all(checks.values())
    assert post._review_passes({"status": "PASS", "checks": checks})
    assert not post._review_passes({"status": "PASS", "checks": {}})
    assert not post._review_passes({"status": "FAIL", "checks": checks})


@pytest.mark.parametrize(
    "attack",
    [
        "missing_question",
        "failed_machine_check",
        "empty_boundary",
        "wrong_grid_count",
        "test_leakage",
        "source_routing",
        "future_pm",
        "observed_pm_overclaim",
        "tail_omission",
        "premature_level6",
        "premature_promotion",
    ],
)
def test_adversarial_contract_fails_closed(attack: str) -> None:
    diagnosis, grid = _diagnosis(), _grid()
    retry, controlled = _retry_access(), _controlled_access()
    if attack == "missing_question":
        diagnosis["questions"].pop(
            "Q1_modules_change_prediction_or_degenerate_to_B2"
        )
    elif attack == "failed_machine_check":
        diagnosis["questions"][
            "Q1_modules_change_prediction_or_degenerate_to_B2"
        ]["machine_checks"]["complete"] = False
    elif attack == "empty_boundary":
        diagnosis["questions"][
            "Q3_modal_audit_shared_poles_and_local_observability"
        ]["claim_boundary"] = ""
    elif attack == "wrong_grid_count":
        grid["event_count"] = 177
    elif attack == "test_leakage":
        controlled["test_ids_read"] = ["forbidden"]
    elif attack == "source_routing":
        controlled["source_or_fault_metadata_used_for_routing"] = True
    elif attack == "future_pm":
        controlled["future_pm_used_as_main_model_input"] = True
    elif attack == "observed_pm_overclaim":
        diagnosis["questions"]["Q2_why_long_horizon_speed_error_is_large"][
            "evidence"
        ]["arm4_interpretation"] = "conditional upper bound"
    elif attack == "tail_omission":
        del diagnosis["questions"]["Q2_why_long_horizon_speed_error_is_large"][
            "evidence"
        ]["five_arm"]["arms"]["5"]["aggregates"]["ALL"]["10"]["omega"][
            "worst_10pct_mean_rmse"
        ]
    elif attack == "premature_level6":
        diagnosis["level_6_claimed"] = True
    elif attack == "premature_promotion":
        diagnosis["promotion_decision"] = "PROMOTE"
    checks = post._adversarial_checks(diagnosis, grid, retry, controlled)
    assert not all(checks.values())


def test_history_evidence_requires_one_exact_stage(monkeypatch) -> None:
    event = {
        "action": "advance",
        "to": "controlled_experiment",
        "evidence": {"result_dir": "run-a"},
    }
    monkeypatch.setattr(post, "load_state", lambda *_: {"history": [event]})
    root = Path("unused")
    assert post._history_evidence(root, "wf", "controlled_experiment") == event[
        "evidence"
    ]
    monkeypatch.setattr(post, "load_state", lambda *_: {"history": [event, event]})
    with pytest.raises(RuntimeError, match="exactly one"):
        post._history_evidence(root, "wf", "controlled_experiment")


def test_report_is_concise_and_keeps_model_unpromoted() -> None:
    acceptance = {
        "scientific_verdict": "ACCEPT",
        "claim_boundary": "diagnosis only",
        "question_answers": {
            "Q1": {"answer": "first", "claim_boundary": "one"},
            "Q2": {"answer": "second", "claim_boundary": "two"},
            "Q3": {"answer": "third", "claim_boundary": "three"},
        },
    }
    report = post._report_text(acceptance)
    assert "`ACCEPT`" in report
    assert "`KEEP_BASELINE`" in report
    assert "当前候选不作为已接受部署模型" in report
    assert all(value in report for value in ("first", "second", "third"))


@pytest.mark.parametrize(
    "merge_evidence",
    [
        {"merge_commit": "a" * 40},
        {"git_commit": "b" * 40},
    ],
)
def test_run_report_uses_canonical_merge_commit_from_new_or_legacy_evidence(
    monkeypatch, merge_evidence: dict
) -> None:
    monkeypatch.setattr(
        post,
        "_history_evidence",
        lambda *_: merge_evidence,
    )
    assert post._report_merge_commit(Path("unused"), "workflow") == (
        merge_evidence.get("merge_commit") or merge_evidence["git_commit"]
    )
