from __future__ import annotations

import csv
import json
import math
import shutil
import tomllib
from pathlib import Path

import numpy as np
import pytest

from kron_reduce.workflows import kiki_temporal_training as workflow
from kron_reduce.workflows.kiki_stability_comparison import summarize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUTH = workflow.AUTH
EXPERIMENT = workflow.EXPERIMENT
CONFIG_REL = Path("configs/experiments/paper_kiki_a11_temporal.toml")


def _authorization_fixture(destination: Path):
    policy = json.loads((PROJECT_ROOT / "configs/research_policy.json").read_text(encoding="utf-8"))
    manifest = json.loads((PROJECT_ROOT / "configs/project_manifest.json").read_text(encoding="utf-8"))
    approval = policy["development_evaluations"][AUTH]
    relatives = [CONFIG_REL, Path("configs/research_policy.json"), Path("configs/project_manifest.json")]
    relatives.extend(Path(approval[key]) for key in (
        "development_protocol_path", "design_path", "scientific_review_path"))
    for relative in relatives:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT_ROOT / relative, target)
    policy["development_evaluations"][AUTH]["status"] = "user_approved"
    (destination / "configs/research_policy.json").write_text(
        json.dumps(policy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    config = tomllib.loads((destination / CONFIG_REL).read_text(encoding="utf-8"))
    experiment = manifest["experiments"][EXPERIMENT]
    return destination, config, experiment


def _bound(rho=0.2):
    return {
        "bound_verified": True,
        "truth": {"rho": rho},
        "prediction": {"rho": rho + 0.01},
        "E_atom": 0.02,
        "E_vector": 0.03,
    }


def _diagnostic(value):
    fixed = _bound()
    rep = {
        "fixed_five_point_audit": fixed,
        "prediction_margin_same_window": fixed["prediction"]["rho"],
    }
    return {
        "history_rho": _bound(),
        "future_rho": _bound(),
        "full_rho": _bound(),
        "representative_truth_window": rep,
        "representative_window_error_density_C_over_p": value,
    }


def test_a11_authorization_accepts_reconstructed_approval_and_rejects_consumed(tmp_path):
    active_policy = json.loads(
        (PROJECT_ROOT / "configs/research_policy.json").read_text(encoding="utf-8"))
    evaluations = active_policy["development_evaluations"]
    for suffix in ("a04", "a05", "a06", "a07", "a08", "a09", "a10"):
        key = f"kiki_{suffix}_development_20261024" if suffix != "a04" else "kiki_a04_development_20261023"
        assert evaluations[key]["status"] == "consumed"
    root, config, experiment = _authorization_fixture(tmp_path / "approved")
    assert workflow.authorization(root, experiment, config)["status"] == "user_approved"
    consumed_root, consumed_config, consumed_experiment = _authorization_fixture(tmp_path / "consumed")
    path = consumed_root / "configs/research_policy.json"
    policy = json.loads(path.read_text(encoding="utf-8"))
    policy["development_evaluations"][AUTH]["status"] = "consumed"
    path.write_text(json.dumps(policy), encoding="utf-8", newline="\n")
    with pytest.raises(PermissionError, match="active fixed-budget authorization"):
        workflow.authorization(consumed_root, consumed_experiment, consumed_config)


def test_a11_authorization_rejects_config_and_bound_protocol_changes(tmp_path):
    root, config, experiment = _authorization_fixture(tmp_path / "config")
    changed = dict(config, max_total_rhs=config["max_total_rhs"] + 1)
    with pytest.raises(PermissionError, match="configuration differs"):
        workflow.authorization(root, experiment, changed)

    doc_root, doc_config, doc_experiment = _authorization_fixture(tmp_path / "protocol")
    protocol = doc_root / doc_config["development_protocol_path"]
    protocol.write_bytes(protocol.read_bytes() + b"tampered\n")
    with pytest.raises(PermissionError, match="approved document bytes changed"):
        workflow.authorization(doc_root, doc_experiment, doc_config)


def test_a11_local_statistics_preserve_weights_and_duration_bucket_denominator():
    event_ids = ["a", "b", "c", "d"]
    weights = [0.1, 0.2, 0.3, 0.4]
    counts = [40, 150, 300, 100]
    values = [1.0, 2.0, 3.0, 4.0]
    rows = [{"event_id": event_id, "status": "ok", "diagnostic": _diagnostic(value)}
            for event_id, value in zip(event_ids, values, strict=True)]
    result = workflow.local_statistics(event_ids, weights, counts, rows)
    assert result["complete"] is True
    buckets = result["duration_buckets"]
    assert buckets["through_1s"]["event_count"] == 2
    assert buckets["through_1s"]["original_weight_mass"] == pytest.approx(0.5)
    assert buckets["between_1s_and_3s"]["original_weight_mass"] == pytest.approx(0.2)
    assert buckets["scored_at_3s"]["original_weight_mass"] == pytest.approx(0.3)
    for metric, total in result["metrics"].items():
        restored = math.fsum(bucket["same_denominator_contribution"][metric]
                             for bucket in buckets.values())
        assert restored == pytest.approx(total)

    failed_rows = [*rows[:-1], {"event_id": "d", "status": "failed"}]
    failed = workflow.local_statistics(event_ids, weights, counts, failed_rows)
    assert len(failed_rows) == len(event_ids)
    assert failed == {"complete": False, "metrics": None}


def test_a11_initial_risk_reconstructs_unequal_saved_point_loss_lengths():
    temperature = workflow.TAU
    rows = [
        {"diagnostic": {"normalized_point_e2": [0.03]}},
        {"diagnostic": {"normalized_point_e2": [0.1, 0.2, 0.4, 0.7]}},
        {"diagnostic": {"normalized_point_e2": np.linspace(0.01, 0.8, 137)}},
    ]
    weights = [0.2, 0.3, 0.5]
    expected = []
    for row in rows:
        losses = np.asarray(row["diagnostic"]["normalized_point_e2"], dtype=np.float64)
        n = len(losses)
        n1 = min(n, 100)
        c = 0.5 / n + 0.5 / n1 * (np.arange(n) < n1)
        logits = np.log(c) + losses / temperature
        peak = float(np.max(logits))
        expected.append(temperature * (peak + math.log(math.fsum(np.exp(logits - peak)))))
    assert workflow.initial_risk(rows, weights, temperature) == pytest.approx(
        math.fsum(w * value for w, value in zip(weights, expected, strict=True)), abs=1e-14)


def test_a11_fixed_path_and_rhs_budgets_recompute_from_registered_counts():
    train_counts = [300] * 200 + [209] * 58
    train_counts[-1] += 25
    val_counts = [300] * 74 + [38] * 13 + [40]
    assert len(train_counts) == 258 and sum(train_counts) == 72147
    assert len(val_counts) == 88 and sum(val_counts) == 22734

    fits = 6
    training_updates = fits * 240 * 2 * (258 * 10 + sum(train_counts))
    training_rhs = training_updates * 4
    assert training_updates == 215213760
    assert training_rhs == 860855040

    prediction_paths = 6 * (258 * 2 + 88 * 2) + 3 * 88 + 30
    assert prediction_paths == 4446

    train_eval_updates = 6 * (258 * 60 + 6 * sum(train_counts))
    train_eval_physical = 6 * (258 * 24 + 2 * sum(train_counts))
    val_eval_updates = 6 * 88 * (60 + 6 * 800)
    val_eval_physical = 6 * 88 * (22 + 2 * 801)
    source_val_updates = 3 * 88 * (40 + 4 * 800)
    source_val_physical = 3 * 88 * (11 + 801)
    candidate_replay_updates = 6 * 2 * (20 + 600 + 40 + 1200)
    candidate_replay_physical = 24 * (11 + 301)
    source_replay_updates = 6 * (40 + 1200)
    source_replay_physical = 6 * (11 + 301)
    eval_rhs = 4 * (train_eval_updates + val_eval_updates + source_val_updates
                    + candidate_replay_updates + source_replay_updates)
    eval_rhs += (train_eval_physical + val_eval_physical + source_val_physical
                 + candidate_replay_physical + source_replay_physical)
    assert eval_rhs == 26549604
    assert training_rhs + eval_rhs == 887404644


def test_a11_preserved_failure_marks_every_allocation_unresolved_without_retry(tmp_path):
    allocated = [
        {"event_id": f"event-{i}", "role": "train" if i < 3 else "val"}
        for i in range(5)
    ]
    state = {
        "stage": "evaluation",
        "fit_records": [{"group": "mean", "seed": 324, "status": "failed"}],
        "allocated": allocated,
    }
    result_dir = tmp_path / "failed-run"
    result_dir.mkdir()
    workflow.preserve_failure(result_dir, "synthetic-a11-run", state, RuntimeError("synthetic stop"))
    receipt = json.loads((result_dir / "execution_failure.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert receipt["reason"] == "synthetic stop"
    assert receipt["all_six_reserved_fits"] == [
        {"group": group, "seed": seed}
        for group in workflow.GROUPS for seed in workflow.SEEDS
    ]
    assert receipt["planned_prediction_paths"] == 4446
    assert receipt["planned_updates"] == 1440
    assert receipt["retries_allowed"] is False
    assert receipt["scientific_verdict"] == "REJECT"
    assert len(receipt["allocated_events"]) == len(allocated)
    assert [row["event_id"] for row in receipt["allocated_events"]] == [
        row["event_id"] for row in allocated
    ]
    assert all(row["campaign_status"] == "unresolved_failed_campaign"
               and row["counted_as_correct"] is False for row in receipt["allocated_events"])
    assert sorted(p.name for p in result_dir.iterdir()) == [
        "execution_failure.json", "failure_artifact_hashes.json"
    ]


class _MetadataProvider:
    def __init__(self, source_root, rows):
        self.source_root = source_root
        self.rows = rows


def _write_dataset_index(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["event_id", "label", "event_time_s"])
        writer.writeheader()
        writer.writerows(rows)


def test_a11_val_truth_metadata_filters_ids_before_parsing_nonval_sentinels(tmp_path):
    source = "synthetic-source"
    registered = [
        {"role": "train", "source_dataset": source, "raw_id": "train-1", "event_id": "train-id"},
        {"role": "val", "source_dataset": source, "raw_id": "val-1", "event_id": "val-id-1"},
        {"role": "val", "source_dataset": source, "raw_id": "val-2", "event_id": "val-id-2"},
    ]
    _write_dataset_index(tmp_path / source / "dataset_index.csv", [
        {"event_id": "train-1", "label": "stable", "event_time_s": "not-a-time"},
        {"event_id": "source-val", "label": "unstable", "event_time_s": "also-not-a-time"},
        {"event_id": "test-1", "label": "unstable", "event_time_s": "invalid"},
        {"event_id": "val-1", "label": "unstable", "event_time_s": "1.25"},
        {"event_id": "val-2", "label": "stable", "event_time_s": ""},
    ])
    result = workflow.val_truth_metadata(_MetadataProvider(tmp_path, registered))
    assert result == {
        "val-id-1": {"label": "unstable", "true_event_time_record_s": 1.25},
        "val-id-2": {"label": "stable", "true_event_time_record_s": None},
    }


def test_a11_val_truth_metadata_rejects_duplicate_or_missing_internal_val(tmp_path):
    source = "synthetic-source"
    rows = [
        {"role": "val", "source_dataset": source, "raw_id": "v1", "event_id": "v1"},
        {"role": "val", "source_dataset": source, "raw_id": "v2", "event_id": "v2"},
    ]
    index_path = tmp_path / source / "dataset_index.csv"
    duplicate = [
        {"event_id": "v1", "label": "unstable", "event_time_s": "1"},
        {"event_id": "v1", "label": "stable", "event_time_s": "2"},
        {"event_id": "v2", "label": "stable", "event_time_s": ""},
    ]
    _write_dataset_index(index_path, duplicate)
    with pytest.raises(ValueError, match="duplicate val metadata identity"):
        workflow.val_truth_metadata(_MetadataProvider(tmp_path, rows))

    _write_dataset_index(index_path, duplicate[:1])
    with pytest.raises(ValueError, match="population incomplete"):
        workflow.val_truth_metadata(_MetadataProvider(tmp_path, rows))


def test_a11_classification_counts_maps_real_los_summary_and_retains_unresolved():
    rows = []
    for index in range(88):
        unstable = index < 14
        rows.append({
            "prospective_eligible": True,
            "label": "unstable" if unstable else "stable",
            "qualified": True,
            "predicted_los": unstable,
            "window_settled": not unstable,
        })
    summary = summarize(rows)
    summary["development_evaluation_refusals"] = 0
    counts = workflow.classification_counts(summary)
    assert counts["unresolved"] == summary["numerical_unresolved"] == 0
    assert counts["eligible_events"] == summary["eligible"] == 88
    assert counts["tp"] + counts["fn"] == summary["eligible_unstable"] == 14
    assert counts["tn"] + counts["fp"] == summary["eligible_stable"] == 74

    rows[0]["qualified"] = False
    summary_with_unresolved = summarize(rows)
    summary_with_unresolved["development_evaluation_refusals"] = 0
    counts_with_unresolved = workflow.classification_counts(summary_with_unresolved)
    assert counts_with_unresolved["unresolved"] == summary_with_unresolved["numerical_unresolved"] == 1
    assert counts_with_unresolved["eligible_events"] == summary_with_unresolved["eligible"] == 88
    assert counts_with_unresolved["tp"] + counts_with_unresolved["fn"] == 13
    assert counts_with_unresolved["tn"] + counts_with_unresolved["fp"] == 74
