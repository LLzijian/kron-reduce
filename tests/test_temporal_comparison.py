from copy import deepcopy

from kron_reduce.evaluation.temporal_comparison import LOCAL_METRICS, SEEDS, STEPS, outcome


def _evidence():
    baselines = {
        seed: {
            "original_main_nrmse": 1.0,
            "original_complete3_nrmse": 1.0,
            "main_nrmse": 1.0,
            "complete3_nrmse": 1.0,
            "fn": 1,
            "fp": 0,
        }
        for seed in SEEDS
    }
    records = {}
    for group in ("mean", "time"):
        for seed in SEEDS:
            records[group, seed] = {
                "val_main_nrmse": 0.9,
                "val_complete3_nrmse": 1.0,
                "fn": 0,
                "fp": 0,
                "tp": 14,
                "tn": 74,
                "unresolved": 0,
                "development_evaluation_refusals": 0,
                "eligible_events": 88,
                "train_main_mse_by_step": {step: 0.5 for step in STEPS},
                "local_metrics_by_step": {
                    step: {metric: 1.0 if group == "mean" else 0.9 for metric in LOCAL_METRICS}
                    for step in STEPS
                },
            }
    gates = {"paths": True, "physics": True, "replay": True, "budget": True}
    return records, baselines, gates


def test_frozen_zero_acceptance_does_not_select_or_promote():
    records, baselines, gates = _evidence()
    result = outcome(records, baselines, validity_gates=gates)
    assert result["validity_passed"] is True
    assert result["group_results"]["mean"]["zero"] is True
    assert result["group_results"]["time"]["zero"] is True
    assert result["campaign_verdict"] == "ACCEPT"
    assert result["mechanism_supported"] is True
    assert result["selected_candidate"] is None
    assert result["promotion_decision"] == "KEEP_BASELINE"
    assert result["model_promoted"] is False
    assert result["new_training_authorized"] is False


def test_raw_fn_improvement_requires_mechanism_gates_for_attribution():
    records, baselines, gates = _evidence()
    for seed in SEEDS:
        records["mean", seed]["fn"] = 1
        records["mean", seed]["tp"] = 13
        records["time", seed]["train_main_mse_by_step"]["0.005"] = 0.6
    result = outcome(records, baselines, validity_gates=gates)
    assert result["raw_fn_improved_against_mean"] is True
    assert result["mechanism_supported"] is False
    assert result["detection_improved"] is False


def test_retry_requires_one_qualifying_group_with_its_own_fn_improvement():
    records, baselines, gates = _evidence()
    for group in ("mean", "time"):
        for seed in SEEDS:
            records[group, seed]["fn"] = 1
            records[group, seed]["tp"] = 13
    result = outcome(records, baselines, validity_gates=gates)
    assert result["campaign_verdict"] == "REJECT"
    records["time", 324]["fn"] = 0
    records["time", 324]["tp"] = 14
    result = outcome(records, baselines, validity_gates=gates)
    assert result["campaign_verdict"] == "RETRY"


def test_group_utility_failure_does_not_erase_valid_candidate_mechanism():
    records, baselines, gates = _evidence()
    for seed in SEEDS:
        baselines[seed]["fn"] = 2
        records["mean", seed]["val_main_nrmse"] = 1.1
        records["mean", seed]["fn"] = 2
        records["mean", seed]["tp"] = 12
        records["time", seed]["fn"] = 1
        records["time", seed]["tp"] = 13
    result = outcome(records, baselines, validity_gates=gates)
    assert result["validity_passed"] is True
    assert result["group_results"]["mean"]["utility_passed"] is False
    assert result["group_results"]["time"]["utility_passed"] is True
    assert result["mechanism_supported"] is True
    assert result["detection_improved"] is True
    assert result["campaign_verdict"] == "RETRY"


def test_strict_local_improvement_boundary_and_train_guard():
    records, baselines, gates = _evidence()
    for seed in SEEDS:
        for metric in LOCAL_METRICS:
            records["time", seed]["local_metrics_by_step"]["0.005"][metric] = 1.0 - 1e-12
    result = outcome(records, baselines, validity_gates=gates)
    assert result["mechanism_supported"] is False
    assert result["campaign_verdict"] == "ACCEPT"
    records, baselines, gates = _evidence()
    records["time", 324]["train_main_mse_by_step"]["0.005"] = 0.5000001
    result = outcome(records, baselines, validity_gates=gates)
    assert result["mechanism_supported"] is False


def test_tail_metric_improvement_cannot_replace_main_validation_guard():
    records, baselines, gates = _evidence()
    for seed in SEEDS:
        records["time", seed]["val_main_nrmse"] = 0.900001
        for step in STEPS:
            for metric in LOCAL_METRICS:
                records["time", seed]["local_metrics_by_step"][step][metric] = 0.1
    result = outcome(records, baselines, validity_gates=gates)
    assert result["mechanism_supported"] is False


def test_missing_row_bad_number_or_bad_gate_fails_validity_closed():
    records, baselines, gates = _evidence()
    del records["time", 326]
    result = outcome(records, baselines, validity_gates=gates)
    assert result["validity_passed"] is False
    assert result["campaign_verdict"] == "REJECT"
    assert result["mechanism_supported"] is False
    records, baselines, gates = _evidence()
    records["mean", 324]["val_main_nrmse"] = float("nan")
    result = outcome(records, baselines, validity_gates=gates)
    assert result["validity_passed"] is False
    records, baselines, gates = _evidence()
    gates["physics"] = 1
    result = outcome(records, baselines, validity_gates=gates)
    assert result["validity_passed"] is False


def test_fp_or_incomplete_population_blocks_v_even_if_metrics_look_good():
    records, baselines, gates = _evidence()
    records["time", 324]["fp"] = 1
    records["time", 324]["tn"] = 73
    result = outcome(records, baselines, validity_gates=gates)
    assert result["validity_passed"] is True
    assert result["group_results"]["time"]["utility_passed"] is False
    records, baselines, gates = _evidence()
    records["mean", 325]["eligible_events"] = 87
    result = outcome(records, baselines, validity_gates=gates)
    assert result["validity_passed"] is False


def test_groups_are_not_spliced_for_retry_and_no_group_zero_is_reject():
    records, baselines, gates = _evidence()
    for seed in SEEDS:
        records["mean", seed]["fn"] = 1
        records["mean", seed]["tp"] = 13
        records["time", seed]["fn"] = 2
        records["time", seed]["tp"] = 12
    records["time", 324]["fn"] = 0
    records["time", 324]["tp"] = 14
    result = outcome(records, baselines, validity_gates=gates)
    assert result["group_results"]["mean"]["utility_passed"] is True
    assert result["group_results"]["time"]["utility_passed"] is False
    assert result["campaign_verdict"] == "REJECT"


def test_baseline_missing_or_nonfinite_blocks_all_official_decisions():
    records, baselines, gates = _evidence()
    baselines[325] = deepcopy(baselines[325])
    baselines[325]["main_nrmse"] = float("inf")
    result = outcome(records, baselines, validity_gates=gates)
    assert result["validity_passed"] is False
    assert result["campaign_verdict"] == "REJECT"
