from __future__ import annotations

import json
import tomllib

import numpy as np

from kron_reduce.data import (
    cumulative_cohorts,
    fit_stratification_policy,
    stratify_samples,
    trajectory_information,
)
from kron_reduce.physics import (
    aggregate_model_sample,
    discover_nested_partitions,
    reconstruct_cluster_state,
)
from kron_reduce.physics.graph import complete_incidence


def _sample(event_id: str, amplitude: float, frequency: float, steps: int = 1001) -> dict:
    time = np.arange(steps, dtype=float) * 0.01
    mass = np.asarray([2.0, 3.0, 4.0, 5.0])
    basis = np.asarray([1.0, -1.0, 0.5, -0.5])
    basis -= np.sum(basis * mass) / np.sum(mass)
    omega = amplitude * np.sin(2.0 * np.pi * frequency * time)[:, None] * basis
    delta = np.cumsum(omega, axis=0) * 0.01
    pm = np.gradient(omega, time, axis=0) * mass[None, :]
    pm -= np.sum(pm, axis=1, keepdims=True) * mass[None, :] / np.sum(mass)
    incidence, edges = complete_incidence(4)
    return {
        "event_id": event_id,
        "model_inputs": {
            "static": {
                "delta0": delta[[0]],
                "omega0": omega[[0]],
                "Pm_equilibrium": np.zeros(4),
                "Pm_clear": pm[[0]],
                "Pm0": pm[[0]],
                "equilibrium_delta": np.zeros(4),
                "mass": mass,
                "incidence": incidence,
            },
            "exogenous": {"t": time, "Pm": pm},
        },
        "training_context": {
            "targets": {"t": time, "delta": delta, "omega": omega},
            "Pm_observed": pm,
            "Pm_equilibrium": np.zeros(4),
            "claim_type": "conditional_rollout_not_causal_forecast",
            "representation": "generator_10",
            "edges": edges,
            "metadata": {
                "source_dataset": "synthetic",
                "fault_line": "none",
                "outcome_label": "forbidden-value",
            },
        },
    }


def test_afsd_stratification_is_train_frozen_and_label_free() -> None:
    train = [
        _sample("low", 0.1, 0.4),
        _sample("mid", 0.4, 0.7),
        _sample("high", 1.0, 1.3),
        _sample("high2", 1.5, 1.7),
    ]
    policy, train_rows = fit_stratification_policy(
        train, minimum_structure_steps=100, report_horizons=(100, 500, 1000)
    )
    changed = _sample("validation", 0.6, 0.9)
    changed["training_context"]["metadata"]["outcome_label"] = "opposite"
    validation = stratify_samples([changed], policy)[0]

    assert policy.labels_used is False
    assert policy.validation_or_test_used is False
    assert policy.created_from_case_ids == tuple(sample["event_id"] for sample in train)
    assert validation["quality_grade"] in {"Q-A", "Q-B"}
    assert validation["amplitude_level"] in {"A-low", "A-mid", "A-high"}
    assert {row["event_id"] for row in train_rows} == {"low", "mid", "high", "high2"}


def test_short_or_nonfinite_trajectory_refuses_structure_selection() -> None:
    short = _sample("short", 0.1, 0.5, steps=20)
    row = trajectory_information(short, report_horizons=(100, 500, 1000))
    assert row["eligible_h100"] is False
    assert row["hankel_rank"] == 0


def test_cumulative_corpora_use_frozen_train_edges() -> None:
    samples = [_sample(f"e{index}", 0.1 + index * 0.2, 0.4 + 0.1 * index) for index in range(9)]
    policy, rows = fit_stratification_policy(samples, minimum_structure_steps=100)
    cohorts, diagnostics = cumulative_cohorts(
        samples,
        rows,
        activation_edges=policy.activation_edges,
        minimum_cases=2,
        allow_train_fallback=True,
    )
    ids = {name: {sample["event_id"] for sample in group} for name, group in cohorts.items()}
    assert ids["C1"] <= ids["C2"] <= ids["C3"]
    assert diagnostics["nested"] is True


def test_nested_partition_discovery_and_reconstruction_are_deterministic() -> None:
    samples = [_sample(f"e{index}", 0.2 + 0.1 * index, 0.5 + 0.1 * index) for index in range(5)]
    hierarchy, diagnostics = discover_nested_partitions(
        samples, cluster_counts=(2, 3), maximum_steps=101
    )
    assert set(hierarchy) == {2, 3}
    assert diagnostics["labels_used"] is False
    assert diagnostics["validation_or_test_used"] is False
    reduced = aggregate_model_sample(samples[0], hierarchy[2], "adaptive_k2")
    reconstructed = reconstruct_cluster_state(
        reduced["training_context"]["targets"]["omega"],
        samples[0]["model_inputs"]["static"]["mass"],
        hierarchy[2],
    )
    assert reconstructed.shape == samples[0]["training_context"]["targets"]["omega"].shape
    assert reduced["model_inputs"]["static"]["mass"].shape == (2,)
    assert abs(float(np.sum(reduced["model_inputs"]["static"]["Pm_equilibrium"]))) < 1e-12


def test_afsd_protocol_is_registered_and_test_locked(root) -> None:
    config = tomllib.loads(
        (root / "configs/experiments/paper_m0_afsd_f1_seed324.toml").read_text(
            encoding="utf-8"
        )
    )
    manifest = json.loads(
        (root / "configs/project_manifest.json").read_text(encoding="utf-8")
    )
    workflow = json.loads(
        (root / "configs/workflows/m0_afsd_v1.json").read_text(encoding="utf-8")
    )
    assert config["fold"] == "F1" and config["seed"] == 324
    assert config["pm_mode"] == "observed" and config["test_access"] == "forbidden"
    assert config["report_horizons"] == [100, 500, 1000]
    assert manifest["experiments"]["paper_m0_afsd_f1_seed324"][
        "reads_test_performance"
    ] is False
    assert "absolute_true_system_order" in workflow["locked_claims"]
    assert "merge_gate" in workflow["approval_required"]
