from __future__ import annotations

import json
import shutil
import tomllib
from pathlib import Path

import numpy as np
import pytest

from kron_reduce.workflows.ten_machine_physical_discovery import (
    _causal_schedule_invariance_audit,
    _historical_audit,
)


class _CausalStub:
    def __init__(self, *, consume_schedule: bool = False) -> None:
        self.consume_schedule = consume_schedule

    def rollout_with_pm(self, initial_state, time_grid, exogenous):
        time = np.asarray(time_grid, dtype=np.float64)
        pm_clear = np.asarray(initial_state["Pm_clear"], dtype=np.float64)
        pm = np.repeat(pm_clear[np.newaxis, :], len(time), axis=0)
        if self.consume_schedule:
            pm = np.asarray(exogenous["Pm"], dtype=np.float64)
        delta = np.zeros_like(pm)
        omega = np.zeros_like(pm)
        return delta, omega, pm


def _causality_sample() -> dict:
    time = np.linspace(0.0, 0.2, 21)
    pm_clear = np.linspace(-0.9, 0.9, 10)
    observed = np.repeat(pm_clear[np.newaxis, :], len(time), axis=0)
    return {
        "event_id": "synthetic-causality",
        "model_inputs": {"static": {"Pm_clear": pm_clear}},
        "training_context": {
            "targets": {"t": time},
            "Pm_observed": observed,
        },
    }


def test_causal_pm_counterfactual_detects_future_schedule_consumption() -> None:
    models = {
        "M0-GMD-PM": _CausalStub(),
        "M0-PSD-PM": _CausalStub(),
    }
    passing = _causal_schedule_invariance_audit(models, _causality_sample(), 21)
    assert passing["status"] == "PASS"
    assert all(row["max_abs_prediction_error"] == 0.0 for row in passing["rows"])
    assert all(
        row["future_schedule_max_abs_perturbation"] > 1.0
        for row in passing["rows"]
    )

    models["M0-PSD-PM"] = _CausalStub(consume_schedule=True)
    failing = _causal_schedule_invariance_audit(models, _causality_sample(), 21)
    assert failing["status"] == "FAIL"
    failed = next(row for row in failing["rows"] if row["method"] == "M0-PSD-PM")
    assert failed["max_abs_prediction_error"] > 1.0


def test_raw_ten_machine_protocol_is_registered_and_test_locked(root: Path) -> None:
    config_path = (
        root
        / "configs"
        / "experiments"
        / "paper_ten_machine_physical_discovery_f1_seed324.toml"
    )
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads(
        (root / "configs" / "project_manifest.json").read_text(encoding="utf-8")
    )
    workflow = json.loads(
        (
            root
            / "configs"
            / "workflows"
            / "ten_machine_physical_discovery_v1.json"
        ).read_text(encoding="utf-8")
    )

    assert config["representation"] == "generator_10"
    assert config["test_access"] == "forbidden"
    assert config["fold"] == "F1" and config["seed"] == 324
    assert config["minimum_train_steps"] == 100
    assert config["minimum_validation_steps"] == 100
    assert config["training_pm_mode"] == "observed"
    assert config["claim_boundaries"]["observed_pm_is_conditional"] is True
    assert config["claim_boundaries"]["test_performance"] is False
    assert (
        manifest["experiments"]["paper_ten_machine_physical_discovery_f1_seed324"][
            "reads_test_performance"
        ]
        is False
    )
    assert workflow["approval_required"] == ["merge_gate"]
    assert "M2A-LYAP" in workflow["locked_claims"]
    resume = manifest["experiments"][
        "paper_ten_machine_physical_discovery_resume_f1_seed324"
    ]
    assert resume["stage"] == "diagnostic"
    assert resume["resumes_failed_run_after_render_only_failure"] is True
    assert resume["reads_test_performance"] is False
    for experiment_name in (
        "paper_ten_machine_physical_discovery_adversarial_f1_seed324",
        "paper_ten_machine_physical_discovery_acceptance_f1_seed324",
    ):
        assert (
            manifest["experiments"][experiment_name]["config"]
            == "paper_ten_machine_physical_discovery_resume_f1_seed324.toml"
        )


@pytest.mark.parametrize("selection_available", [False, True])
def test_historical_mainline_representation_claim_has_hashed_evidence(
    root: Path, tmp_path: Path, selection_available: bool,
) -> None:
    for relative in (
        "configs/experiments/paper_m0_gmd_pm_round6_f1_single_seed.toml",
        "docs/m0_gmd_pm_round6_protocol.md",
        "configs/project_manifest.json",
        "configs/experiments/paper_m0_3eg_cluster_compare_f2_seed324.toml",
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, target)
    gmd = tmp_path / (
        "results/backup/m0_gmd_horizon_balanced_backbone/"
        "m0-gmd-horizon-balanced-f1-seed324-20260731/selection_marker.json"
    )
    gmd.parent.mkdir(parents=True)
    gmd.write_text(json.dumps({"selected_modal_rank": 2}), encoding="utf-8")
    gmd_pm = tmp_path / (
        "results/diagnostics/m0-gmd-pm-modal-tgov-bound-f1-seed324-20260801-r4/"
        "selection_marker.json"
    )
    if selection_available:
        gmd_pm.parent.mkdir(parents=True)
        gmd_pm.write_text(
            json.dumps({"source_accuracy_model_frozen": "synthetic-reference"}),
            encoding="utf-8",
        )

    audit = _historical_audit(tmp_path)
    assert audit["accepted_round6_representation"] == "clustered_7"
    assert audit["accepted_downstream_chain_representation"] == "clustered_7"
    assert audit["later_raw10_runs_were_diagnostic_only"] is True
    assert audit["accepted_gmd_selected_modal_rank"] == 2
    assert audit["accepted_gmd_pm_selection_available"] is selection_available
    assert audit["accepted_gmd_pm_source_accuracy_model"] == (
        "synthetic-reference" if selection_available else None
    )
    assert len(audit["evidence_files"]) == 6
    for name, evidence in audit["evidence_files"].items():
        available = name != "accepted_gmd_pm_selection" or selection_available
        assert evidence["available"] is available
        if available:
            assert len(evidence["sha256"]) == 64
        else:
            assert evidence["sha256"] is None

    if selection_available:
        gmd_pm.write_text("{", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            _historical_audit(tmp_path)
    gmd.unlink()
    with pytest.raises(FileNotFoundError):
        _historical_audit(tmp_path)
