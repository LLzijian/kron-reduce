from __future__ import annotations

import json

import numpy as np
import pytest

from kron_reduce.models import (
    EDGE_MEMORY,
    NODE_HIGH_ORDER,
    OPTIONAL_BLOCKS,
    RESONANT_MEMORY,
    PrunableEnergyModel,
    RunContext,
    StructureSelectionResult,
    block_registry,
)
from kron_reduce.physics.coordinates import coi_project, coi_project_power
from kron_reduce.physics.graph import complete_incidence
from kron_reduce.selection import decide_clustering


def _manual_model(
    n_nodes: int,
    coupling: np.ndarray,
    *,
    active_blocks: tuple[str, ...] = (),
    max_internal_step: float = 0.0025,
) -> PrunableEnergyModel:
    model = PrunableEnergyModel(
        active_blocks=active_blocks,
        memory_taus=(0.2,),
        resonances=((1.1, 0.12),),
        node_tau=0.25,
        max_internal_step=max_internal_step,
    )
    n_edges = len(coupling)
    model.node_damping = np.full(n_nodes, 0.22)
    model.coupling = np.asarray(coupling, dtype=float)
    model.potential_coefficients = np.full((3, n_edges), 0.04)
    model.edge_damping_coefficients = np.full((3, n_edges), 0.03)
    model.node_high_order_gain = np.full(n_nodes, 0.08)
    model.memory_gains = np.full((1, n_edges), 0.06)
    model.resonant_gains = np.full((1, n_edges), 0.05)
    return model


def _sample(
    event_id: str,
    model: PrunableEnergyModel,
    mass: np.ndarray,
    incidence: np.ndarray,
    delta0: np.ndarray,
    omega0: np.ndarray,
    pm: np.ndarray,
    time: np.ndarray,
    group: str,
) -> dict:
    static = {
        "delta0": coi_project(delta0, mass)[None, :],
        "omega0": coi_project(omega0, mass)[None, :],
        "Pm_equilibrium": np.zeros(len(mass)),
        "Pm_clear": pm[[0]],
        "Pm0": pm[[0]],
        "equilibrium_delta": np.zeros(len(mass)),
        "mass": mass.copy(),
        "incidence": incidence.copy(),
    }
    delta, omega = model.rollout(static, time, {"Pm": pm, "interpolation": "linear"})
    return {
        "event_id": event_id,
        "model_inputs": {"static": static, "exogenous": {"t": time, "Pm": pm}},
        "training_context": {
            "targets": {"t": time, "delta": delta, "omega": omega},
            "Pm_observed": pm.copy(),
            "Pm_equilibrium": np.zeros(len(mass)),
            "claim_type": "conditional_rollout_not_causal_forecast",
            "representation": f"synthetic_{len(mass)}",
            "metadata": {"source_dataset": "synthetic", "fault_line": group},
        },
    }


def _coherent_corpus(*, heterogeneous_input: bool = False) -> list[dict]:
    mass = np.ones(4)
    incidence, edges = complete_incidence(4)
    coupling = np.asarray(
        [18.0 if set(edge) in ({0, 1}, {2, 3}) else 0.15 for edge in edges]
    )
    truth = _manual_model(4, coupling)
    truth.max_internal_step = 0.01
    time = np.linspace(0.0, 4.0, 401)
    slow = np.asarray([1.0, 1.0, -1.0, -1.0])
    internal_left = np.asarray([1.0, -1.0, 0.0, 0.0])
    internal_right = np.asarray([0.0, 0.0, 1.0, -1.0])
    samples = []
    for index in range(9):
        delta0 = (0.05 + 0.004 * index) * slow
        delta0 += 3e-4 * ((-1.0) ** index) * internal_left
        delta0 += 3e-4 * ((-1.0) ** (index // 2)) * internal_right
        omega0 = 0.006 * np.cos(index) * slow
        pm = np.zeros((len(time), 4))
        if heterogeneous_input:
            forcing = 0.08 * np.sin(2.0 * np.pi * 0.7 * time + 0.2 * index)
            pm[:, 0] = forcing
            pm[:, 1] = -forcing
            pm = coi_project_power(pm, mass)
        samples.append(
            _sample(
                f"event-{index}",
                truth,
                mass,
                incidence,
                delta0,
                omega0,
                pm,
                time,
                f"group-{index}",
            )
        )
    return samples


def test_clusterability_recovers_exact_coherent_partition() -> None:
    decision = decide_clustering(
        _coherent_corpus(),
        bootstrap_draws=20,
        max_steps=401,
        delays=8,
    )
    assert decision.use_clustering is True
    assert decision.n_clusters == 2
    assert {frozenset(group) for group in decision.partition} == {
        frozenset((0, 1)),
        frozenset((2, 3)),
    }
    assert decision.bootstrap_support >= 0.8
    assert decision.score <= 1.0


def test_clusterability_rejects_intra_cluster_input_information() -> None:
    decision = decide_clustering(
        _coherent_corpus(heterogeneous_input=True),
        bootstrap_draws=12,
        max_steps=401,
        delays=8,
    )
    assert decision.use_clustering is False
    assert decision.n_clusters == decision.n_original


def test_b2_truth_fits_and_has_no_supported_optional_blocks() -> None:
    mass = np.ones(2)
    incidence = np.asarray([[1.0], [-1.0]])
    truth = _manual_model(2, np.asarray([1.4]), max_internal_step=0.005)
    time = np.linspace(0.0, 10.0, 1001)
    samples = []
    for index in range(4):
        forcing = 0.05 * np.sin(2 * np.pi * (0.35 + 0.05 * index) * time)
        pm = np.column_stack((forcing, -forcing))
        samples.append(
            _sample(
                f"b2-{index}",
                truth,
                mass,
                incidence,
                np.asarray([0.03 + 0.004 * index, -0.03 - 0.004 * index]),
                np.zeros(2),
                pm,
                time,
                f"line-{index}",
            )
        )
    context = RunContext(
        horizon_steps=1001,
        max_cases=4,
        ridge_values=(1e-10,),
        integral_windows=(1, 5, 20, 80),
        fit_max_steps=1001,
        weak_max_windows_per_case_horizon=12,
        max_internal_step=0.01,
    )
    fitted = PrunableEnergyModel(active_blocks=())
    fitted.fit(samples[:3], samples[3:], context)
    summary = fitted.validation_summary(samples[3:], context)
    proposal = PrunableEnergyModel.propose_candidates(
        samples[:3], fitted.node_damping, fitted.coupling, max_steps=1001
    )
    assert summary["failure_rate"] == 0.0
    assert summary["raw_mean_rmse"] < 1e-3
    assert proposal["active_blocks"] == ()
    assert np.allclose(fitted.node_damping, truth.node_damping, rtol=0.08, atol=0.03)
    assert np.allclose(fitted.coupling, truth.coupling, rtol=0.05, atol=0.03)


def test_all_dynamic_ports_satisfy_energy_supply_identity() -> None:
    mass = np.asarray([1.0, 1.5, 2.0])
    incidence, _ = complete_incidence(3)
    model = _manual_model(3, np.asarray([1.2, 0.9, 1.1]), active_blocks=OPTIONAL_BLOCKS)
    static = {
        "delta0": np.zeros((1, 3)),
        "omega0": np.zeros((1, 3)),
        "Pm_equilibrium": np.zeros(3),
        "Pm_clear": np.zeros((1, 3)),
        "equilibrium_delta": np.zeros(3),
        "mass": mass,
        "incidence": incidence,
    }
    delta = coi_project(np.asarray([0.08, -0.03, -0.01]), mass)
    omega = coi_project(np.asarray([0.04, -0.02, -0.005]), mass)
    hidden = model.initial_hidden(omega, static)
    hidden["node"] += np.asarray([0.01, -0.02, 0.015])
    hidden["memory"] += 0.01
    hidden["resonance_q"] += 0.005
    hidden["resonance_p"] -= 0.008
    dd, dw, dh = model.vector_field_augmented(delta, omega, hidden, np.zeros(3), static)
    flat_hidden = model._pack_hidden(hidden)
    flat_derivative = model._pack_hidden(dh)
    step = 1e-6
    forward = model.storage(
        delta + step * dd,
        omega + step * dw,
        static,
        flat_hidden + step * flat_derivative,
    )
    backward = model.storage(
        delta - step * dd,
        omega - step * dw,
        static,
        flat_hidden - step * flat_derivative,
    )
    numerical_rate = (forward - backward) / (2 * step)
    expected_rate = -model.dissipation_rate(delta, omega, hidden, static)
    assert abs(numerical_rate - expected_rate) < 1e-8
    audit = model.physics_audit(static)
    assert audit["status"] == "pass"
    assert audit["storage_hessian_min_eigenvalue"] > 1e-8


def test_full_supported_model_recovers_every_optional_block() -> None:
    mass = np.ones(2)
    incidence = np.asarray([[1.0], [-1.0]])
    truth = _manual_model(
        2,
        np.asarray([1.2]),
        active_blocks=OPTIONAL_BLOCKS,
        max_internal_step=0.01,
    )
    time = np.linspace(0.0, 5.0, 501)
    samples = []
    for index in range(4):
        forcing = 0.06 * np.sin(2 * np.pi * (0.3 + 0.07 * index) * time)
        pm = np.column_stack((forcing, -forcing))
        samples.append(
            _sample(
                f"full-{index}",
                truth,
                mass,
                incidence,
                np.asarray([0.10 + 0.015 * index, -0.10 - 0.015 * index]),
                np.zeros(2),
                pm,
                time,
                f"full-line-{index}",
            )
        )
    context = RunContext(
        horizon_steps=501,
        max_cases=4,
        ridge_values=(1e-10, 1e-8),
        integral_windows=(1, 5, 20, 80),
        fit_max_steps=501,
        weak_max_windows_per_case_horizon=16,
        max_internal_step=0.01,
    )
    fitted = PrunableEnergyModel(
        active_blocks=OPTIONAL_BLOCKS,
        memory_taus=(0.2,),
        resonances=((1.1, 0.12),),
    )
    base = PrunableEnergyModel(active_blocks=())
    base.fit(samples[:3], samples[3:], context)
    proposal = PrunableEnergyModel.propose_candidates(
        samples[:3],
        base.node_damping,
        base.coupling,
        bootstrap_draws=40,
    )
    fitted.fit(samples[:3], samples[3:], context)
    summary = fitted.validation_summary(samples[3:], context)
    assert summary["failure_rate"] == 0.0
    assert summary["raw_mean_rmse"] < 1e-3
    assert set(fitted.diagnostics["block_parameter_norms"]) == set(OPTIONAL_BLOCKS)
    assert min(fitted.diagnostics["block_parameter_norms"].values()) > 1e-3
    assert set(proposal["active_blocks"]) == set(OPTIONAL_BLOCKS)
    assert min(proposal["support"].values()) >= 0.8


def test_pruning_noninferiority_is_enforced_at_every_time_window() -> None:
    reference = {
        "failure_rate": 0.0,
        "horizon_standardized_loss": {"1s": 1.0, "5s": 1.0, "10s": 1.0},
    }
    passing = {
        "failure_rate": 0.0,
        "horizon_standardized_loss": {"1s": 1.02, "5s": 0.9, "10s": 1.0},
    }
    failing = {
        "failure_rate": 0.0,
        "horizon_standardized_loss": {"1s": 0.8, "5s": 1.0201, "10s": 0.8},
    }
    assert PrunableEnergyModel._rollout_noninferior(reference, passing, 0.02)
    assert not PrunableEnergyModel._rollout_noninferior(reference, failing, 0.02)


def test_integration_free_margin_uses_fixed_postclear_equilibrium() -> None:
    mass = np.ones(2)
    model = _manual_model(2, np.asarray([1.2]))
    static = {
        "delta0": np.asarray([[0.08, -0.08]]),
        "omega0": np.zeros((1, 2)),
        "Pm_equilibrium": np.zeros(2),
        "Pm_clear": np.zeros((1, 2)),
        "equilibrium_delta": np.zeros(2),
        "mass": mass,
        "incidence": np.asarray([[1.0], [-1.0]]),
    }
    result = model.stability_margin(static)
    assert result["certified"] is True
    assert result["integration_free"] is True
    assert result["critical_energy"] > result["storage_at_clearing"]
    assert result["margin"] > 0.0
    assert result["predicted_stable"] is True


def test_registry_and_workflow_are_single_corpus_single_model(root) -> None:
    registry = block_registry()
    assert set(OPTIONAL_BLOCKS).issubset(registry)
    assert registry["B2_BACKBONE"].removable is False
    for name in (NODE_HIGH_ORDER, EDGE_MEMORY, RESONANT_MEMORY):
        assert "reciprocal" in registry[name].parameter_constraint
    workflow = json.loads(
        (root / "configs" / "workflows" / "data_level_structure_selection_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert workflow["scientific_target"] == "one_data_level_cluster_decision_and_one_prunable_energy_model"
    assert workflow["stages"][-1] == "merge_gate"
    manifest = json.loads((root / "configs" / "project_manifest.json").read_text(encoding="utf-8"))
    names = {
        "paper_data_level_v1_hypothesis_f1_seed324",
        "paper_data_level_v1_controlled_f1_seed324",
        "paper_data_level_v1_residual_f1_seed324",
        "paper_data_level_v1_adversarial_f1_seed324",
        "paper_data_level_v1_acceptance_f1_seed324",
    }
    assert names.issubset(manifest["experiments"])
    assert all(
        manifest["experiments"][name]["workflow"] == "data_level_structure_selection_v1"
        for name in names
    )
    program = manifest["research_programs"]["ieee39_modular_energy_paper_v1"]
    assert program["single_model_per_corpus"] is True
    assert program["source_routing"] is False


def test_structure_selection_result_refuses_model_routing() -> None:
    common = {
        "cluster_decision": {"use_clustering": False},
        "active_blocks": ("B2_BACKBONE",),
        "pruning_trace": (),
        "rollout_metrics": {},
        "modes": (),
        "stability": {},
    }
    result = StructureSelectionResult(
        **common,
        module_excitation_support={"B2_BACKBONE": {"all": 1.0}},
        module_identifiability={"B2_BACKBONE": {"status": "structural"}},
        module_deletion_effect={"B2_BACKBONE": {"removable": False}},
        module_cross_fold_stability={"B2_BACKBONE": {"folds_present": 1}},
    )
    assert result.to_dict()["model_count"] == 1
    with pytest.raises(ValueError, match="exactly one"):
        StructureSelectionResult(**common, model_count=2)
    with pytest.raises(ValueError, match="routing"):
        StructureSelectionResult(**common, source_routing=True)
