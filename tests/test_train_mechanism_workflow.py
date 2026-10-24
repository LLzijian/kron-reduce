from __future__ import annotations

import json
import shutil
import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from kron_reduce.data.history_contract import CausalStateHistory
from kron_reduce.evaluation.audit_budget import AuditRhsBudget
from kron_reduce.models.nonlinear_latent import (
    LearnedNonlinearModel,
    NonlinearFitSettings,
    NonlinearPassiveClosure,
)
from kron_reduce.physics.coordinates import coi_project
from kron_reduce.workflows.kiki_train_mechanism import (
    AUTH,
    EXPERIMENT,
    SEEDS,
    STEPS,
    authorization,
    compare_source_scores,
    counted_model,
    integration_substeps,
    path_physics,
    summarize_audit,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_REL = Path("configs/experiments/paper_kiki_a09_train_mechanism.toml")


def _copy_auth_fixture(destination: Path):
    config_source = PROJECT_ROOT / CONFIG_REL
    policy_source = PROJECT_ROOT / "configs/research_policy.json"
    manifest_source = PROJECT_ROOT / "configs/project_manifest.json"
    config_bytes = config_source.read_bytes()
    policy = json.loads(policy_source.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_source.read_text(encoding="utf-8"))
    approval = policy["development_evaluations"][AUTH]
    relatives = [
        CONFIG_REL,
        Path("configs/research_policy.json"),
        Path("configs/project_manifest.json"),
        Path(approval["development_protocol_path"]),
        Path(approval["design_path"]),
        Path(approval["scientific_review_path"]),
    ]
    for relative in relatives:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT_ROOT / relative, target)
    # Reconstruct the pre-execution approval only inside this isolated fixture;
    # the repository's consumed authorization is never changed or reused.
    policy["development_evaluations"][AUTH]["status"] = "user_approved"
    (destination / "configs/research_policy.json").write_text(
        json.dumps(policy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    copied_config = tomllib.loads((destination / CONFIG_REL).read_text(encoding="utf-8"))
    copied_manifest = json.loads(
        (destination / "configs/project_manifest.json").read_text(encoding="utf-8"))
    experiment = copied_manifest["experiments"][EXPERIMENT]
    assert experiment == manifest["experiments"][EXPERIMENT]
    assert (destination / CONFIG_REL).read_bytes() == config_bytes
    return destination, copied_config, experiment


def test_authorization_accepts_copied_registry_and_document_hashes(tmp_path):
    root, config, experiment = _copy_auth_fixture(tmp_path / "valid")
    approved = authorization(root, experiment, config)
    assert approved["status"] == "user_approved"


def test_authorization_rejects_budget_change_and_consumed_approval(tmp_path):
    root, config, experiment = _copy_auth_fixture(tmp_path / "budget")
    changed = dict(config)
    changed["max_total_rhs"] += 1
    with pytest.raises(PermissionError, match="differs from its approval"):
        authorization(root, experiment, changed)

    consumed_root, consumed_config, consumed_experiment = _copy_auth_fixture(tmp_path / "consumed")
    policy_path = consumed_root / "configs/research_policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["development_evaluations"][AUTH]["status"] = "consumed"
    policy_path.write_text(json.dumps(policy), encoding="utf-8", newline="\n")
    with pytest.raises(PermissionError, match="active fixed-model"):
        authorization(consumed_root, consumed_experiment, consumed_config)


def test_authorization_rejects_changed_bound_approval_document(tmp_path):
    root, config, experiment = _copy_auth_fixture(tmp_path / "tampered-doc")
    protocol = root / config["development_protocol_path"]
    protocol.write_bytes(protocol.read_bytes() + b"tampered\n")
    with pytest.raises(PermissionError, match="approved document bytes changed"):
        authorization(root, experiment, config)


def test_integration_substeps_preserve_each_original_clock_interval_ceil():
    clock = np.array([0.0, 0.01, 0.025, 0.04])
    assert integration_substeps(clock, 0.01) == 1 + 2 + 2
    assert integration_substeps(clock, 0.005) == 2 + 3 + 3
    assert integration_substeps(np.array([0.0, 0.01, 0.02]), 0.01) == 2
    assert integration_substeps(np.array([0.0, 0.01, 0.02]), 0.005) == 4
    with pytest.raises(ValueError, match="increasing"):
        integration_substeps(np.array([0.0, 0.02, 0.01]), 0.01)


def _synthetic_model_and_history():
    rng = np.random.default_rng(20261024)
    mass = np.linspace(1.0, 3.7, 10)
    incidence = np.zeros((10, 2))
    incidence[[0, 1], 0] = (1.0, -1.0)
    incidence[[3, 8], 1] = (0.7, -0.7)
    coupling = incidence @ np.diag([0.13, 0.08])
    projection = np.eye(10) - np.ones((10, 10)) / 10
    damping = 0.04 * projection
    latent_skew = np.array([[0.0, 0.03], [-0.03, 0.0]])
    latent_dissipation = np.diag([0.08, 0.11])
    closure = NonlinearPassiveClosure(
        mass=mass,
        edge_stiffness=np.linspace(0.01, 0.05, 45),
        damping=damping,
        coupling=coupling,
        latent_skew=latent_skew,
        latent_dissipation=latent_dissipation,
    )
    power = np.linspace(-0.2, 0.2, 10)
    power -= np.mean(power)
    equilibrium = coi_project(np.linspace(-0.15, 0.12, 10), mass)
    delta = np.stack([coi_project(rng.normal(0.0, 0.01, 10), mass) for _ in range(11)])
    omega = np.stack([coi_project(rng.normal(0.0, 0.004, 10), mass) for _ in range(11)])
    history = CausalStateHistory(
        time=np.arange(11, dtype=float) * 0.01 - 0.1,
        delta=delta,
        omega=omega,
        constant_power=power,
    )
    encoder = rng.normal(0.0, 0.02, size=(2, 80))
    model = LearnedNonlinearModel(
        closure=closure,
        encoder=encoder,
        feature_scale=np.ones(80),
        target_scale=np.linspace(0.8, 1.2, 20),
        settings=NonlinearFitSettings(latent_dimension=2, frames=3),
        fit_record={"synthetic_only": True},
    )
    return model, history, equilibrium, mass


def _assert_byteexact(left, right):
    assert left.dtype == right.dtype
    assert left.shape == right.shape
    assert left.tobytes() == right.tobytes()


def test_counted_closure_matches_original_initialize_rollout_and_physical_calls():
    reference, history, equilibrium, _mass = _synthetic_model_and_history()
    original_z, original_history_path = reference.initialize(history, equilibrium, max_step=0.01)
    future_time = np.array([0.0, 0.01, 0.025, 0.04])
    original_future = reference.closure.rollout(
        history.delta[-1], history.omega[-1], original_z, future_time,
        history.constant_power, equilibrium, max_step=0.005,
    )

    counted, _, _, _ = _synthetic_model_and_history()
    budget = AuditRhsBudget(integration_limit=1000, physical_limit=1000, total_limit=2000,
                            rollout_limit=10)
    counted_model(counted, budget)
    with budget.segment({"synthetic": "history"}, "history_integration", 40):
        counted_z, counted_history_path = counted.initialize(history, equilibrium, max_step=0.01)
    future_rhs = 4 * integration_substeps(future_time, 0.005)
    with budget.segment({"synthetic": "future"}, "future_integration", future_rhs):
        counted_future = counted.closure.rollout(
            history.delta[-1], history.omega[-1], counted_z, future_time,
            history.constant_power, equilibrium, max_step=0.005,
        )
    _assert_byteexact(counted_z, original_z)
    _assert_byteexact(counted_history_path, original_history_path)
    _assert_byteexact(counted_future, original_future)
    assert [(r["attempted"], r["returned"], r["rollout_calls_started"],
             r["successful_rk4_updates"]) for r in budget.records] == [(40, 40, 1, 10), (32, 32, 1, 8)]

    sample = SimpleNamespace(history=history, equilibrium_delta=equilibrium)
    saved_states = len(counted_history_path) + len(counted_future)
    with budget.segment({"synthetic": "physical"}, "physical_checks", saved_states):
        diagnostics = path_physics(counted, counted_history_path, counted_future, sample)
    assert diagnostics["qualified"]
    physical_record = budget.records[-1]
    assert physical_record["attempted"] == physical_record["returned"] == saved_states
    assert physical_record["successful_rk4_updates"] is None
    assert budget.snapshot()["integration_rhs"]["returned"] == 72
    assert budget.snapshot()["physical_rhs"]["returned"] == saved_states


def _audit_inputs(*, supported_events=2, same_line_across_sources=False):
    event_ids = [f"event-{i:03d}" for i in range(258)]
    metadata = {}
    for i, event_id in enumerate(event_ids):
        metadata[event_id] = {
            "source_dataset": "source-A" if i % 2 == 0 else "source-B",
            "fault_line": "Line_1" if i >= supported_events or i == 0 or same_line_across_sources else f"Line_{i+1}",
        }
    rows = {}
    for seed in SEEDS:
        for step in STEPS:
            for i, event_id in enumerate(event_ids):
                supported = i < supported_events
                rows[seed, step, event_id] = {
                    "status": "ok", "physical": {"qualified": True},
                    "diagnostic": {
                        "observed_rule_hit": False,
                        "prediction_rule_hit": False,
                        "history_rho": {"truth": {"triggered": False}},
                        "future_rho": {"truth": {"rho": 0.2 if supported else -0.2},
                                        "prediction": {"triggered": False}},
                        "representative_window_excess_over_event_mean_C_minus_pM":
                            0.2 if supported else -0.1,
                    },
                }
    refinements = {
        (seed, event_id): {"qualified": True}
        for seed in SEEDS for event_id in event_ids
    }
    source_checks = {str(seed): {"qualified": True} for seed in SEEDS}
    return event_ids, metadata, rows, refinements, source_checks


def _summary(inputs, *, budget_complete=True):
    return summarize_audit(*inputs[:1], *inputs[1:], budget_complete=budget_complete)


def test_summary_requires_two_events_on_two_distinct_physical_lines():
    same_line = _audit_inputs(supported_events=2, same_line_across_sources=True)
    result_same = _summary(same_line)
    assert result_same["audit_all_checks_passed"] is True
    assert len(result_same["time_local_error_evidence_events"]) == 2
    assert result_same["distinct_physical_fault_lines"] == ["Line_1"]
    assert result_same["time_aggregation_proposal_supported"] is False

    distinct = _audit_inputs(supported_events=2, same_line_across_sources=False)
    result_distinct = _summary(distinct)
    assert result_distinct["audit_all_checks_passed"] is True
    assert len(result_distinct["time_local_error_evidence_events"]) == 2
    assert len(result_distinct["distinct_physical_fault_lines"]) == 2
    assert result_distinct["time_aggregation_proposal_supported"] is True
    assert result_distinct["new_training_authorized"] is False
    assert result_distinct["promotion_decision"] == "KEEP_BASELINE"
    assert result_distinct["model_promoted"] is False


@pytest.mark.parametrize("failure", ["physical", "numerical", "source", "budget"])
def test_summary_rejects_any_physical_numerical_source_or_budget_failure(failure):
    inputs = _audit_inputs()
    event_ids, _metadata, rows, refinements, source_checks = inputs
    budget_complete = True
    if failure == "physical":
        rows[SEEDS[0], STEPS[0], event_ids[0]]["physical"]["qualified"] = False
    elif failure == "numerical":
        refinements[SEEDS[0], event_ids[0]]["qualified"] = False
    elif failure == "source":
        source_checks[str(SEEDS[0])]["qualified"] = False
    elif failure == "budget":
        budget_complete = False
    result = _summary(inputs, budget_complete=budget_complete)
    assert result["audit_all_checks_passed"] is False
    assert result["scientific_verdict"] == "REJECT"
    assert result["time_aggregation_proposal_supported"] is False
    assert result["new_training_authorized"] is False
    assert result["promotion_decision"] == "KEEP_BASELINE"
    assert result["model_promoted"] is False


@pytest.mark.parametrize("boundary", ["rho", "excess"])
def test_summary_strictly_rejects_evidence_equal_to_one_e_minus_twelve(boundary):
    inputs = _audit_inputs()
    event_ids, _, rows, _, _ = inputs
    for seed in SEEDS:
        for step in STEPS:
            row = rows[seed, step, event_ids[0]]["diagnostic"]
            if boundary == "rho":
                row["future_rho"]["truth"]["rho"] = 1e-12
            else:
                row["representative_window_excess_over_event_mean_C_minus_pM"] = 1e-12
    result = _summary(inputs)
    assert event_ids[0] not in {r["event_id"] for r in result["time_local_error_evidence_events"]}
    assert result["time_aggregation_proposal_supported"] is False
    assert result["new_training_authorized"] is False
    assert result["promotion_decision"] == "KEEP_BASELINE"


def _source_comparison_inputs():
    expected = {}
    diagnostics = {}
    keys = {
        "normalized_mse": 0.4,
        "normalized_delta_mse_contribution": 0.2,
        "normalized_omega_mse_contribution": 0.2,
        "physical_delta_mse_rad2": 0.01,
        "physical_omega_mse_rad2_per_s2": 0.02,
    }
    source_keys = {
        "normalized_mse": "normalized_mse",
        "normalized_delta_mse_contribution": "normalized_delta_mse_contribution",
        "normalized_omega_mse_contribution": "normalized_omega_mse_contribution",
        "physical_delta_mse_rad2": "delta_mse_rad2",
        "physical_omega_mse_rad2_per_s2": "omega_mse_rad2_per_s2",
    }
    for i in range(258):
        event_id = f"source-event-{i:03d}"
        expected[event_id] = {
            horizon: {source_key: keys[key] for key, source_key in source_keys.items()}
            for horizon in ("1.0", "3.0")
        }
        diagnostics[event_id] = {"status": "ok", "diagnostic": {
            "horizon_scores": {h: dict(keys) for h in ("1.0", "3.0")}
        }}
    return diagnostics, expected


def test_compare_source_scores_rejects_missing_failed_and_mismatched_events():
    diagnostics, expected = _source_comparison_inputs()
    assert compare_source_scores(diagnostics, expected)["qualified"] is True

    missing = dict(diagnostics)
    missing.pop(next(iter(missing)))
    assert compare_source_scores(missing, expected)["qualified"] is False

    failed = dict(diagnostics)
    event = next(iter(failed))
    failed[event] = {"status": "failed"}
    assert compare_source_scores(failed, expected)["qualified"] is False

    mismatch = {key: dict(value) for key, value in diagnostics.items()}
    event = next(iter(mismatch))
    mismatch[event] = json.loads(json.dumps(diagnostics[event]))
    mismatch[event]["diagnostic"]["horizon_scores"]["1.0"]["normalized_mse"] += 2e-12
    assert compare_source_scores(mismatch, expected)["qualified"] is False
    malformed = json.loads(json.dumps(expected))
    del malformed[event]["1.0"]["normalized_mse"]
    assert compare_source_scores(diagnostics, malformed)["qualified"] is False
