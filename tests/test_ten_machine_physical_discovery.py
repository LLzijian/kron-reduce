from __future__ import annotations

import json
import tomllib
from pathlib import Path

import numpy as np

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


def test_historical_mainline_representation_claim_has_hashed_evidence(
    root: Path,
) -> None:
    audit = _historical_audit(root)
    assert audit["accepted_round6_representation"] == "clustered_7"
    assert audit["accepted_downstream_chain_representation"] == "clustered_7"
    assert audit["later_raw10_runs_were_diagnostic_only"] is True
    assert len(audit["evidence_files"]) >= 5
    assert all(
        len(value["sha256"]) == 64 for value in audit["evidence_files"].values()
    )
