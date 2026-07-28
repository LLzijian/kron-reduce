from __future__ import annotations

import numpy as np

from kron_reduce.data import ModelDataRequest, adapt_event
from kron_reduce.evaluation.energy import evaluate_energy_gate, supply_violation_loss
from kron_reduce.models import DynamicM2AModel, EnergyM2AModel, GovernorM0Model, M0Model, M2AModel, RunContext
from kron_reduce.models.base import sample_rollout_inputs
from kron_reduce.models.integration import integration_convergence_audit, rk4_rollout, torch_rk4_rollout
from kron_reduce.physics.coordinates import coi_basis, coi_project, coi_project_power


def _two_node_static():
    return {
        "delta0": np.array([[0.04, -0.04]]),
        "omega0": np.array([[0.01, -0.01]]),
        "Pm_equilibrium": np.zeros(2),
        "Pm_clear": np.zeros((1, 2)),
        "Pm0": np.zeros((1, 2)),
        "equilibrium_delta": np.zeros(2),
        "mass": np.ones(2),
        "incidence": np.array([[1.0], [-1.0]]),
    }


def _synthetic_sample():
    static = _two_node_static()
    truth = M0Model(np.array([0.35, 0.35]), np.array([1.2]), max_internal_step=0.0025)
    t = np.linspace(0.0, 1.0, 101)
    input_signal = 0.08 * np.sin(2 * np.pi * t)
    pm = np.column_stack((input_signal, -input_signal))
    delta, omega = truth.rollout(static, t, {"Pm": pm, "interpolation": "linear"})
    return {
        "event_id": "synthetic",
        "model_inputs": {"static": static, "exogenous": {"t": t, "Pm": pm}},
        "training_context": {
            "targets": {"t": t, "delta": delta, "omega": omega},
            "Pm_observed": pm,
            "claim_type": "conditional_rollout_not_causal_forecast",
        },
    }


def test_pm_semantics_and_idempotent_coi(store):
    event_id = next(e for e in store.event_ids() if store.metadata(e).valid_length >= 20)
    sample = adapt_event(
        store.load_event(event_id), inertia_prior=store.dataset_metadata.M_prior,
        request=ModelDataRequest.initial_state_rollout("zoh"),
    )
    static = sample["model_inputs"]["static"]
    assert np.array_equal(static["Pm0"], static["Pm_clear"])
    assert abs(np.sum(static["Pm_equilibrium"])) < 1e-10
    assert np.max(np.abs(sample["training_context"]["Pm_observed"].sum(axis=1))) < 1e-10
    assert np.allclose(coi_project(static["delta0"], static["mass"]), static["delta0"])
    assert np.allclose(coi_project_power(static["Pm_equilibrium"], static["mass"]), static["Pm_equilibrium"])


def test_raw_coordinate_coi_basis_and_shift_invariance():
    mass = np.array([1.0, 2.0, 4.0])
    basis = coi_basis(mass)
    assert np.linalg.norm(mass @ basis) < 1e-12
    static = {
        "incidence": np.array([[1.0, 1.0, 0.0], [-1.0, 0.0, 1.0], [0.0, -1.0, -1.0]]),
        "equilibrium_delta": np.zeros(3), "mass": mass,
        "Pm_equilibrium": np.zeros(3),
    }
    node = np.array([[0.2, -0.1], [0.1, 0.3], [-0.2, 0.1]])
    edge = np.arange(9, dtype=float).reshape(3, 3) * 0.01
    delta = np.array([0.1, -0.03, -0.01]); omega = coi_project(np.array([0.2, -0.1, 0.0]), mass); pm = np.zeros(3)
    force = M2AModel._residual_force_numpy(node, edge, delta, omega, pm, static)
    shifted = M2AModel._residual_force_numpy(node, edge, delta + 7.0, omega, pm, static)
    assert abs(np.sum(force)) < 1e-12
    assert np.allclose(force, shifted, atol=1e-12)


def test_round2_projected_fourier_library_is_balanced_and_equilibrium_anchored():
    static = _two_node_static()
    node = np.arange(10, dtype=float).reshape(2, 5) * 0.01
    edge = np.arange(9, dtype=float).reshape(1, 9) * 0.01
    equilibrium = static["equilibrium_delta"]
    force = M2AModel._residual_force_numpy(
        node, edge, equilibrium, np.zeros(2), static["Pm_equilibrium"], static,
        "projected_fourier_v2",
    )
    assert np.allclose(force, 0.0, atol=1e-14)
    moved = M2AModel._residual_force_numpy(
        node, edge, np.array([0.2, -0.2]), np.array([0.1, -0.1]),
        np.array([0.03, -0.03]), static, "projected_fourier_v2",
    )
    shifted = M2AModel._residual_force_numpy(
        node, edge, np.array([5.2, 4.8]), np.array([0.1, -0.1]),
        np.array([0.03, -0.03]), static, "projected_fourier_v2",
    )
    assert abs(np.sum(moved)) < 1e-12
    assert np.allclose(moved, shifted, atol=1e-12)


def test_integral_weak_form_recovers_known_parameters():
    sample = _synthetic_sample()
    model = M0Model()
    result = model.fit([sample], [sample], RunContext(horizon_steps=100, max_cases=1, integral_windows=(1, 5, 20)))
    assert np.allclose(model.damping, [0.35, 0.35], rtol=0.05, atol=0.02)
    assert np.allclose(model.coupling, [1.2], rtol=0.05, atol=0.03)
    assert result.details["n_integration_windows"] > 0


def test_observed_pm_validation_uses_linear_rk_stage_interpolation():
    sample = _synthetic_sample()
    _, _, exogenous = sample_rollout_inputs(sample)
    assert exogenous["interpolation"] == "linear"
    model = M0Model(
        np.array([0.35, 0.35]), np.array([1.2]), max_internal_step=0.0025
    )
    model.validation_scales = {"omega": 1.0, "edge_angle": 1.0}
    summary = model.validation_summary(
        [sample],
        RunContext(horizon_steps=100, max_cases=1),
    )
    assert summary["standardized_loss"] < 1e-20


def test_round3_linear_residual_projection_cannot_add_antidamping():
    rng = np.random.default_rng(324)
    theta = rng.normal(size=7 + 4 + 20 + 8)
    constrained = M2AModel._project_stability_constraints(
        theta,
        n_nodes=7,
        n_edges=20,
        feature_version="projected_fourier_tied_v2",
        n_node_features=5,
        n_edge_features=9,
    )
    node, edge = M2AModel._unpack_solution(
        constrained, 7, 20, "projected_fourier_tied_v2", 5, 9
    )
    assert np.all(node[:, 0] <= 0.0)
    assert np.all(edge[:, 5] >= 0.0)
    assert np.allclose(edge[:, 5], edge[0, 5])


def test_numpy_torch_rk4_match_and_converge():
    import torch

    static = _two_node_static(); t = np.array([0.0, 0.037, 0.09]); pm = np.zeros((3, 2))
    def nf(d, w, p, s): return w, -2.0 * d - 0.1 * w
    nd, nw = rk4_rollout(nf, static["delta0"], static["omega0"], t, pm, static, interpolation="zoh", max_step=0.01)
    ts = {k: torch.as_tensor(v, dtype=torch.float64) for k, v in static.items() if isinstance(v, np.ndarray)}
    def tf(d, w, p, s): return w, -2.0 * d - 0.1 * w
    td, tw = torch_rk4_rollout(tf, ts["delta0"], ts["omega0"], torch.as_tensor(t), torch.as_tensor(pm), ts, interpolation="zoh", max_step=0.01)
    assert np.allclose(nd, td.detach().numpy(), atol=1e-13)
    assert np.allclose(nw, tw.detach().numpy(), atol=1e-13)
    audit = integration_convergence_audit(nf, static["delta0"], static["omega0"], t, pm, static, interpolation="zoh", base_step=0.02)
    assert audit["convergent"]


def test_dynamic_memory_save_load_reproducible(tmp_path):
    static = _two_node_static(); base = M0Model(np.array([0.3, 0.4]), np.array([1.2])); static_m2a = M2AModel(base, np.zeros((2, 2)), np.zeros((1, 3)))
    model = DynamicM2AModel(static_m2a, np.array([0.1]), np.array([0.2]), np.array([-0.1]), True)
    t = np.linspace(0, 0.1, 11); pm = np.zeros((len(t), 2)); expected = model.rollout(static, t, {"Pm": pm, "interpolation": "zoh"})
    path = tmp_path / "dynamic.npz"; model.save(path); loaded = DynamicM2AModel.load(path); actual = loaded.rollout(static, t, {"Pm": pm, "interpolation": "zoh"})
    assert np.allclose(expected[0], actual[0]); assert np.allclose(expected[1], actual[1])


def test_governor_and_dynamic_candidates_fit_without_future_inputs():
    sample = _synthetic_sample(); context = RunContext(horizon_steps=40, max_cases=1, sindy_thresholds=(0.0, 1e-3), ridge_values=(1e-6,))
    base = M0Model(); base.fit([sample], [sample], context)
    governor = GovernorM0Model(base); fit_governor = governor.fit([sample], [sample], context)
    gd, gw = governor.rollout(sample["model_inputs"]["static"], sample["training_context"]["targets"]["t"][:40], {})
    assert fit_governor.details["optimizer"] == "single_constrained_integral_least_squares"
    assert np.isfinite(gd).all() and np.isfinite(gw).all()
    dynamic = DynamicM2AModel(M2AModel(M0Model(base.damping, base.coupling)))
    dynamic.static_model.base.validation_scales = dict(base.validation_scales)
    fit_dynamic = dynamic.fit([sample], [sample], context)
    assert fit_dynamic.details["epochs"] == 0
    assert dynamic.physics_audit()["future_state_initialization"] is False


def test_m2a_uses_exact_m0_fallback_when_all_learned_candidates_fail(monkeypatch):
    sample = _synthetic_sample()
    context = RunContext(
        horizon_steps=40, max_cases=1, sindy_thresholds=(0.0, 1e-3),
        ridge_values=(1e-6,),
    )
    base = M0Model()
    base.fit([sample], [sample], context)
    model = M2AModel(base)

    def reject_learned_residuals(*_args, **_kwargs):
        raise FloatingPointError("synthetic unstable residual")

    monkeypatch.setattr(model, "_validation_summary", reject_learned_residuals)
    result = model.fit([sample], [sample], context)

    assert result.status == "completed"
    assert result.details["candidate_kind"] == "exact_M0_fallback"
    assert result.details["solution_kind"] == "one_step_rejected_to_M0"
    assert result.details["active_parameters"] == 0
    assert result.details["one_step_relative_improvement"] == 0.0


def test_port_hamiltonian_gate_and_artificial_violation():
    sample = _synthetic_sample(); static = sample["model_inputs"]["static"]
    model = EnergyM2AModel(
        M0Model(np.array([0.3, 0.4]), np.array([1.2])),
        node_damping=np.array([0.3, 0.4]), edge_damping=np.array([1e-10]),
        kappa=np.array([1.2]), decay=np.array([1.0]), memory_coupling=np.array([0.0]),
    )
    gate = evaluate_energy_gate(model, [sample], tolerance=2e-5)
    assert gate.certificate_passed
    assert gate.reported_name == "M2A-LYAP"
    assert gate.discrete_energy_convergent
    assert supply_violation_loss(1.0, 0.0) > 0

    class AntiDamped(EnergyM2AModel):
        def vector_field_augmented(self, delta, omega, z, pm, static):
            dd, dw, dz = super().vector_field_augmented(delta, omega, z, pm, static)
            return dd, dw + 5.0 * omega, dz

    bad = AntiDamped(model.base, model.node_damping, model.edge_damping, model.kappa, model.decay, model.memory_coupling)
    rejected = evaluate_energy_gate(bad, [sample], tolerance=2e-5)
    assert not rejected.certificate_passed
    assert rejected.reported_name != "M2A-LYAP"
