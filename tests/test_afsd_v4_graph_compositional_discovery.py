from __future__ import annotations

import json
import tomllib
from dataclasses import replace

import numpy as np

from kron_reduce.evaluation.compositional_search import (
    SearchState,
    add_module,
    deterministic_beam_search,
    exhaustive_oracle,
    module_subset_states,
    oracle_agreement,
    paired_module_mcts_trigger,
)
from kron_reduce.models.compositional_modules import (
    module_registry,
    residual_module_proposal,
    validate_module_registry,
    validate_module_spec,
)
from kron_reduce.physics.graph_quotient import (
    beam_graph_contraction,
    conditional_predictive_information_gain,
    input_heterogeneity,
    internal_mode_basis,
    intra_cluster_energy,
    mass_orthogonal_projectors,
    projection_certificate,
)
from kron_reduce.workflows.m0_afsd_v4_graph_compositional_discovery import (
    ROLLOUT_LABEL,
    _hypothesis_registry,
    _synthetic_graph_gates,
    _synthetic_module_search_gates,
    _validate_scope,
)


def _paths(*, hidden: bool, heterogeneous_input: bool = False, seed: int = 324):
    rng = np.random.default_rng(seed)
    result = []
    time = np.arange(180, dtype=float) * 0.01
    for index in range(8):
        phase = 0.23 * index
        common_a = np.sin(2.0 * time + phase)
        common_b = np.cos(1.2 * time - phase)
        inner = 0.25 * np.sin(5.0 * time + phase) if hidden else np.zeros_like(time)
        delta = np.column_stack((common_a + inner, common_a - inner, common_b, common_b))
        delta += 0.001 * rng.normal(size=delta.shape)
        omega = np.gradient(delta, time, axis=0)
        forcing = np.zeros_like(delta)
        forcing[:, 0] = np.sin(0.8 * time)
        forcing[:, 1] = forcing[:, 0]
        if heterogeneous_input:
            forcing[:, 1] *= -1.0
        result.append(
            {
                "delta": delta,
                "omega": omega,
                "u": forcing,
                "fault_line": f"L{index}",
            }
        )
    return result


def test_mass_projector_certificates_and_full_resolution_exact():
    mass = np.array([2.0, 3.0, 5.0, 7.0])
    partition = ((0, 1), (2, 3))
    membership, projector, complement = mass_orthogonal_projectors(mass, partition)
    assert np.allclose(projector @ projector, projector)
    assert np.allclose(projector.T @ np.diag(mass), np.diag(mass) @ projector)
    assert np.allclose(projector @ complement, 0.0)
    assert np.allclose(membership.T @ mass, [5.0, 12.0])
    assert projection_certificate(mass, partition)["status"] == "PASS"
    identity = projection_certificate(mass, ((0,), (1,), (2,), (3,)))
    assert identity["full_resolution_exact"] is True


def test_exact_cluster_has_low_intra_energy_and_hidden_mode_is_recovered():
    mass = np.ones(4)
    partition = ((0, 1), (2, 3))
    coherent = _paths(hidden=False)
    hidden = _paths(hidden=True)
    assert intra_cluster_energy(coherent, mass, partition)["delta"] < 1e-4
    assert intra_cluster_energy(hidden, mass, partition)["delta"] > 0.01
    basis, audit = internal_mode_basis(hidden, mass, partition, 1)
    membership, _, _ = mass_orthogonal_projectors(mass, partition)
    assert audit["status"] == "completed"
    assert np.allclose(basis.T @ (mass[:, None] * basis), np.eye(1), atol=1e-10)
    assert np.allclose(basis.T @ (mass[:, None] * membership), 0.0, atol=1e-10)


def test_input_heterogeneity_cannot_be_hidden_by_cluster():
    mass = np.ones(4)
    partition = ((0, 1), (2, 3))
    homogeneous = input_heterogeneity(_paths(hidden=False), mass, partition)
    heterogeneous = input_heterogeneity(
        _paths(hidden=False, heterogeneous_input=True), mass, partition
    )
    assert homogeneous < 1e-12
    assert heterogeneous > 0.4


def test_cpig_detects_predictive_hidden_coordinate():
    mass = np.ones(4)
    partition = ((0, 1), (2, 3))
    paths = _paths(hidden=True)
    rows = conditional_predictive_information_gain(
        paths[:5], paths[5:], mass, partition, horizons=(5,), bootstrap_draws=50
    )
    assert rows[0]["status"] == "completed"
    assert np.isfinite(rows[0]["cpig"])


def test_graph_contraction_obeys_certificate_and_is_deterministic():
    distance = np.array([[0.0, 0.1, 4.0], [0.1, 0.0, 3.0], [4.0, 3.0, 0.0]])

    def certificate(partition):
        acceptable = all(set(group) != {0, 1, 2} for group in partition)
        return {"status": "HARD_CLUSTER_ADMISSIBLE" if acceptable else "FULL_RESOLUTION_REQUIRED"}

    first = beam_graph_contraction(distance, certificate, beam_width=3, minimum_clusters=1)
    second = beam_graph_contraction(distance, certificate, beam_width=3, minimum_clusters=1)
    assert first["best_partition"] == second["best_partition"]
    assert len(first["best_partition"]) == 2
    assert [0, 1] in first["best_partition"]


def test_module_grammar_is_complete_and_rejects_illegal_time_constant():
    registry = module_registry()
    assert len(registry) == 10
    assert validate_module_registry(registry)["status"] == "PASS"
    bad = replace(registry["NODE_LAG_STORAGE"], parameter_domain={"tau": "free"})
    assert validate_module_spec(bad)["status"] == "FAIL"


def test_residual_proposals_distinguish_static_memory_and_insufficient():
    time = np.linspace(0.0, 8.0, 800)
    signal = np.sin(2.0 * time) + 0.2 * np.sin(4.0 * time)
    static = residual_module_proposal(signal, edge_angle=signal)
    assert static["proposals"] == ["STATIC_NONLINEAR_POTENTIAL"]
    lagged = np.roll(signal, 8)
    lagged[:8] = 0.0
    memory = residual_module_proposal(lagged, edge_speed=signal)
    assert memory["proposals"] == ["EDGE_LAG_STORAGE"]
    insufficient = residual_module_proposal(np.zeros_like(signal), edge_speed=signal)
    assert insufficient["status"] == "INSUFFICIENT_INFORMATION"


def test_saturation_is_not_replaced_by_high_order_static_polynomial():
    signal = np.linspace(-2.0, 2.0, 1000)
    residual = np.clip(signal, -0.4, 0.4)
    proposal = residual_module_proposal(residual, node_speed=signal)
    assert proposal["proposals"] == ["SATURATION_OR_DEADBAND"]


def _module_evaluator(target: tuple[str, ...], pair_synergy: bool = False):
    def evaluate(state: SearchState, tier: int):
        selected = set(state.active_modules)
        truth = set(target)
        if pair_synergy and truth and not truth.issubset(selected):
            error = 2.0 + 0.1 * len(selected)
        else:
            error = float(len(selected ^ truth))
        return {
            "hard_gate": "PASS",
            "heldout_error": error,
            "description_length": len(selected),
            "tier": tier,
        }

    return evaluate


def test_beam_matches_exhaustive_oracle_and_uses_fewer_full_fits():
    initial = SearchState(((0,), (1,)))
    modules = ("EDGE_DAMPING", "EDGE_LAG_STORAGE", "NODE_DAMPING")
    evaluator = _module_evaluator(("EDGE_LAG_STORAGE",))

    def actions(state):
        for module in modules:
            if module not in state.active_modules:
                yield add_module(state, module)

    beam = deterministic_beam_search(initial, actions, evaluator, beam_width=3, max_depth=3)
    oracle = exhaustive_oracle(module_subset_states(initial, modules), evaluator)
    agreement = oracle_agreement(beam, oracle)
    assert agreement["status"] == "PASS"
    assert agreement["full_fit_reduction"] > 0


def test_candidate_order_does_not_change_search_conclusion():
    initial = SearchState(((0,), (1,)))
    evaluator = _module_evaluator(("EDGE_DAMPING",))

    def actions(order):
        return lambda state: (
            add_module(state, module) for module in order if module not in state.active_modules
        )

    left = deterministic_beam_search(
        initial, actions(("NODE_DAMPING", "EDGE_DAMPING")), evaluator, beam_width=2, max_depth=2
    )
    right = deterministic_beam_search(
        initial, actions(("EDGE_DAMPING", "NODE_DAMPING")), evaluator, beam_width=2, max_depth=2
    )
    assert left["selected"] == right["selected"]


def test_paired_module_scenario_triggers_mcts_gate_when_beam_cannot_keep_pair():
    initial = SearchState(((0,), (1,)))
    modules = tuple(f"M{index}" for index in range(12))
    evaluator = _module_evaluator(("M8", "M9"), pair_synergy=True)
    audit = paired_module_mcts_trigger(initial, modules, evaluator, beam_widths=(3, 5, 10))
    assert audit["status"] == "MCTS_TRIGGERED"
    assert audit["oracle"]["selected"] is not None


def test_structure_serialization_is_fixed_support():
    state = SearchState(
        ((0, 1), (2,)),
        internal_modes=1,
        equivalent_graph=((0, 1),),
        active_modules=("EDGE_DAMPING",),
    )
    restored = json.loads(json.dumps(state.to_dict()))
    assert restored["internal_modes"] == 1
    assert restored["active_modules"] == ["EDGE_DAMPING"]


def _v4_config(root):
    return tomllib.loads(
        (
            root
            / "configs"
            / "experiments"
            / "paper_m0_afsd_v4_graph_compositional_discovery_f1_seed324.toml"
        ).read_text(encoding="utf-8")
    )


def test_v4_scope_is_locked_to_f1_seed324_and_conditional_observed_pm(root):
    config = _v4_config(root)
    _validate_scope(config)
    assert config["rollout_label"] == ROLLOUT_LABEL
    invalid = {**config, "pm_mode": "zoh"}
    try:
        _validate_scope(invalid)
    except RuntimeError as error:
        assert "locked v4 scope" in str(error)
    else:
        raise AssertionError("v4 accepted an unregistered Pm policy")


def test_v4_workflow_orders_scientific_stages_and_locks_merge(root):
    workflow = json.loads(
        (
            root / "configs" / "workflows" / "m0_afsd_v4_graph_compositional_discovery.json"
        ).read_text(encoding="utf-8")
    )
    ordered = [
        "hypothesis_registration",
        "controlled_experiment",
        "residual_diagnosis",
        "adversarial_review",
        "acceptance",
    ]
    assert [workflow["stages"].index(stage) for stage in ordered] == sorted(
        workflow["stages"].index(stage) for stage in ordered
    )
    assert workflow["approval_required"] == ["merge_gate"]


def test_v4_synthetic_gates_are_registered_before_f1(root):
    config = _v4_config(root)
    assert _synthetic_graph_gates(config)["status"] == "PASS"
    assert _synthetic_module_search_gates(config)["status"] == "PASS"
    registry = _hypothesis_registry(config, "synthetic-test")
    assert registry["registered_before_controlled_experiment"] is True
    assert registry["claim_boundaries"]["test_performance"] is False
