"""Synthetic tests for deterministic nested-memory initializations."""

import numpy as np
import pytest

from kron_reduce.data.history_contract import CausalStateHistory
from kron_reduce.models.nested_memory import (
    BASE_ENCODER_PENALTY,
    ENCODER_PENALTY_DENOMINATOR,
    NEW_LATENT_DISSIPATION,
    nested_memory_start,
)
from kron_reduce.physics.coordinates import coi_basis, coi_project


def nested_fixture():
    rng = np.random.default_rng(8042)
    n, features = 10, 80
    basis = coi_basis(np.ones(n))
    mass = np.linspace(0.8, 2.2, n)
    encoder = rng.normal(size=(2, features))
    encoder[1] += 0.4 * encoder[0]
    source = {
        "mass": mass,
        "edge_stiffness": np.linspace(0.7, 1.8, n * (n - 1) // 2),
        "damping": (basis * np.linspace(0.2, 0.8, n - 1)) @ basis.T,
        "coupling": basis @ rng.normal(scale=0.1, size=(n - 1, 2)),
        "latent_skew": np.array([[0.0, 0.13], [-0.13, 0.0]]),
        "latent_dissipation": np.diag([0.3, 0.6]),
        "encoder": encoder,
        "feature_scale": np.linspace(0.5, 1.5, features),
        "target_scale": np.linspace(0.7, 1.3, 2 * n),
    }
    # Build full-rank causal inputs with deterministic, nonuniform event weights.
    rows = rng.normal(size=(12, features))
    weights = np.arange(1.0, len(rows) + 1.0)
    weights /= weights.sum()
    return source, rows, weights


def test_nested_start_is_deterministic_copied_and_preserves_visible_embedding():
    source, rows, weights = nested_fixture()
    source_before = {key: value.copy() for key, value in source.items()}
    features_before, weights_before = rows.copy(), weights.copy()
    r2, audit2 = nested_memory_start(source, rows, weights, group="r2")
    r4, audit4 = nested_memory_start(source, rows, weights, group="r4")
    again, audit_again = nested_memory_start(source, rows, weights, group="r4")

    assert audit4["source_array_sha256"] == audit_again["source_array_sha256"]
    assert audit4["residual_directions_sha256"] == audit_again["residual_directions_sha256"]
    assert audit4["full_start_array_sha256"] == audit_again["full_start_array_sha256"]
    assert all(np.array_equal(r4[key], again[key]) for key in r4)
    assert all(np.array_equal(source[key], source_before[key]) for key in source)
    assert np.array_equal(rows, features_before) and np.array_equal(weights, weights_before)
    assert r2["encoder"].shape == (2, 80) and r4["encoder"].shape == (4, 80)
    assert np.array_equal(r4["encoder"][:2], source["encoder"])
    assert np.allclose(r4["encoder"][:2] @ rows.T, r2["encoder"] @ rows.T, atol=0, rtol=0)
    assert audit2["target_latent_dimension"] == 2
    assert audit4["target_latent_dimension"] == 4
    assert audit4["old_encoder_direction_overlap_frobenius"] <= 1e-10
    assert audit4["direction_orthogonality_frobenius"] <= 1e-12
    assert audit4["new_to_old_encoder_squared_norm_ratio"] == pytest.approx(0.05**2)


def test_nested_start_rejects_degenerate_encoder_and_exhausted_feature_space():
    source, rows, weights = nested_fixture()
    degenerate = {**source, "encoder": np.vstack((source["encoder"][0], source["encoder"][0]))}
    with pytest.raises(ValueError, match="encoder lacks two independent rows"):
        nested_memory_start(degenerate, rows, weights, group="r2")

    # Every input row lies in the old encoder row space; the residual should
    # be rejected despite floating-point projection noise.
    coordinates = np.random.default_rng(9).normal(size=(len(rows), 2))
    exhausted = coordinates @ source["encoder"]
    with pytest.raises(ValueError, match="lack two independent residual directions"):
        nested_memory_start(source, exhausted, weights, group="r4")


def test_r2_and_r4_share_identical_source_memory_and_r4_keeps_energy_structure():
    source, rows, weights = nested_fixture()
    r2, _ = nested_memory_start(source, rows, weights, group="r2")
    r4, audit = nested_memory_start(source, rows, weights, group="r4")
    for name in ("mass", "edge_stiffness", "damping", "feature_scale", "target_scale"):
        assert np.array_equal(r2[name], r4[name])
    assert np.array_equal(r4["coupling"][:, :2], r2["coupling"])
    assert np.count_nonzero(r4["coupling"][:, 2:]) == 0
    assert np.array_equal(r4["latent_skew"][:2, :2], r2["latent_skew"])
    assert np.count_nonzero(r4["latent_skew"][2:, :]) == 0
    assert np.array_equal(r4["latent_dissipation"][:2, :2], r2["latent_dissipation"])
    assert np.allclose(r4["latent_dissipation"][2:, 2:], NEW_LATENT_DISSIPATION * np.eye(2))
    assert np.array_equal(r4["encoder"][:2], r2["encoder"])
    assert audit["optimizer_initialization"] == "fresh_adam_zero_moments_step_zero"
    assert audit["future_targets_used"] is False and audit["labels_used"] is False

    # The passive closure constructor enforces the COI, skew and PSD structure.
    from kron_reduce.models.nonlinear_latent import NonlinearPassiveClosure

    closure = NonlinearPassiveClosure(**{key: r4[key] for key in (
        "mass", "edge_stiffness", "damping", "coupling", "latent_skew", "latent_dissipation"
    )})
    n = len(r4["mass"])
    random_state = np.random.default_rng(13).normal(scale=0.02, size=2 * n + 4)
    state = np.r_[
        coi_project(random_state[:n], r4["mass"]),
        coi_project(random_state[n : 2 * n], r4["mass"]),
        random_state[2 * n :],
    ]
    power = np.random.default_rng(14).normal(scale=0.1, size=len(r4["mass"]))
    identity = closure.energy_identity(state, power, np.zeros_like(power))
    assert identity["identity_error"] == pytest.approx(0.0, abs=1e-10)


def test_nested_start_preserves_old_encoder_penalty_scale_and_limits_increase():
    source, rows, weights = nested_fixture()
    _, audit2 = nested_memory_start(source, rows, weights, group="r2")
    r4, audit4 = nested_memory_start(source, rows, weights, group="r4")
    coefficient = BASE_ENCODER_PENALTY / ENCODER_PENALTY_DENOMINATOR
    old_penalty = coefficient * np.sum(source["encoder"] ** 2)
    new_penalty = coefficient * np.sum(r4["encoder"] ** 2)
    assert audit2["initial_encoder_penalty"] == pytest.approx(old_penalty)
    assert audit4["initial_encoder_penalty"] == pytest.approx(new_penalty)
    assert audit4["initial_encoder_penalty_increase"] == pytest.approx(new_penalty - old_penalty)
    assert new_penalty / old_penalty == pytest.approx(1.0025, abs=1e-12)
    assert audit4["encoder_penalty_setting"] == BASE_ENCODER_PENALTY * 2
    assert audit2["encoder_penalty_setting"] == BASE_ENCODER_PENALTY


def test_nontrivial_inputs_support_gradients_in_new_memory_blocks():
    import torch

    source, rows, weights = nested_fixture()
    start, _ = nested_memory_start(source, rows, weights, group="r4")
    # A small differentiable visible-memory objective demonstrates that the
    # new coupling and encoder blocks are connected to nondegenerate inputs.
    coupling = torch.tensor(start["coupling"], dtype=torch.float64, requires_grad=True)
    encoder = torch.tensor(start["encoder"], dtype=torch.float64, requires_grad=True)
    features = torch.tensor(rows, dtype=torch.float64)
    target = torch.tensor(np.random.default_rng(45).normal(size=(len(rows), 10)), dtype=torch.float64)
    weights_t = torch.tensor(weights, dtype=torch.float64)
    prediction = (features @ encoder.T) @ coupling.T
    loss = torch.sum(weights_t[:, None] * (prediction - target) ** 2)
    loss.backward()
    assert coupling.grad is not None and torch.isfinite(coupling.grad).all()
    assert encoder.grad is not None and torch.isfinite(encoder.grad).all()
    assert torch.linalg.vector_norm(coupling.grad[:, 2:]) > 0
    assert torch.linalg.vector_norm(encoder.grad[:2]) > 0


def _synthetic_history(mass):
    n = len(mass)
    clock = np.arange(11, dtype=np.float64) * 0.01 - 0.1
    base_q = coi_project(np.linspace(-0.04, 0.06, n), mass)
    base_v = coi_project(np.linspace(0.03, -0.02, n), mass)
    shape = np.linspace(-1.0, 1.0, n)
    delta = np.stack([coi_project(base_q + t * shape, mass) for t in clock])
    omega = np.stack([coi_project(base_v + 0.2 * t * shape, mass) for t in clock])
    power = np.linspace(-0.2, 0.3, n)
    power -= power.mean()
    assert np.linalg.norm(delta) > 0 and np.linalg.norm(omega) > 0 and np.linalg.norm(power) > 0
    return CausalStateHistory(clock, delta, omega, power)


def test_full_history_initialization_and_future_rollout_keep_visible_path_equal():
    from kron_reduce.models.nonlinear_latent import (
        LearnedNonlinearModel,
        NonlinearFitSettings,
        NonlinearPassiveClosure,
    )

    source, rows, weights = nested_fixture()
    r2, _ = nested_memory_start(source, rows, weights, group="r2")
    r4, _ = nested_memory_start(source, rows, weights, group="r4")
    history = _synthetic_history(source["mass"])
    equilibrium = coi_project(np.linspace(-0.03, 0.05, 10), source["mass"])
    settings2 = NonlinearFitSettings(latent_dimension=2, frames=3, epochs=1, horizons_s=(0.03,))
    settings4 = NonlinearFitSettings(latent_dimension=4, frames=3, epochs=1, horizons_s=(0.03,))

    def build_model(start, settings):
        closure = NonlinearPassiveClosure(**{key: start[key] for key in (
            "mass", "edge_stiffness", "damping", "coupling", "latent_skew", "latent_dissipation"
        )})
        return LearnedNonlinearModel(
            closure, start["encoder"], start["feature_scale"], start["target_scale"], settings, {}
        )

    model2, model4 = build_model(r2, settings2), build_model(r4, settings4)
    assert np.linalg.norm(r4["encoder"][2:]) > 0
    initial2, history_path2 = model2.initialize(history, equilibrium)
    initial4, history_path4 = model4.initialize(history, equilibrium)
    assert np.allclose(history_path2[:, :20], history_path4[:, :20], atol=1e-12, rtol=0)
    assert np.allclose(initial2, initial4[:2], atol=1e-12, rtol=0)

    future_clock = np.array([0.0, 0.01, 0.02, 0.03])
    future2 = model2.predict(history, future_clock, equilibrium)
    future4 = model4.predict(history, future_clock, equilibrium)
    assert np.allclose(future2[:, :20], future4[:, :20], atol=1e-12, rtol=0)


def test_nested_memory_gradient_blocks_match_and_zero_start_is_a_dead_block():
    import torch

    source, rows, weights = nested_fixture()
    start2, _ = nested_memory_start(source, rows, weights, group="r2")
    start4, _ = nested_memory_start(source, rows, weights, group="r4")
    features = torch.tensor(rows, dtype=torch.float64)
    target = torch.tensor(np.random.default_rng(45).normal(size=(len(rows), 10)), dtype=torch.float64)
    weights_t = torch.tensor(weights, dtype=torch.float64)

    def gradients(start):
        coupling = torch.tensor(start["coupling"], dtype=torch.float64, requires_grad=True)
        encoder = torch.tensor(start["encoder"], dtype=torch.float64, requires_grad=True)
        prediction = (features @ encoder.T) @ coupling.T
        loss = torch.sum(weights_t[:, None] * (prediction - target) ** 2)
        loss.backward()
        return loss.detach(), coupling.grad.detach(), encoder.grad.detach()

    loss2, coupling_grad2, encoder_grad2 = gradients(start2)
    loss4, coupling_grad4, encoder_grad4 = gradients(start4)
    assert loss2.item() == pytest.approx(loss4.item(), abs=1e-12)
    assert torch.allclose(coupling_grad2, coupling_grad4[:, :2], atol=1e-12, rtol=0)
    assert torch.allclose(encoder_grad2, encoder_grad4[:2], atol=1e-12, rtol=0)
    assert torch.linalg.vector_norm(coupling_grad4[:, 2:]) > 0

    # If both the new encoder and its cross-coupling are zero, the new
    # bilinear block is disconnected and neither block receives a gradient.
    dead = {key: value.copy() for key, value in start4.items()}
    dead["encoder"][2:] = 0.0
    dead["coupling"][:, 2:] = 0.0
    _, dead_coupling_grad, dead_encoder_grad = gradients(dead)
    assert torch.count_nonzero(dead_coupling_grad[:, 2:]) == 0
    assert torch.count_nonzero(dead_encoder_grad[2:]) == 0

    # The existing encoder penalty gives exactly the same old-row gradient;
    # the two added rows account for only a 0.25% initial increase.
    penalty_coefficient = BASE_ENCODER_PENALTY / ENCODER_PENALTY_DENOMINATOR
    old2 = torch.tensor(start2["encoder"], dtype=torch.float64, requires_grad=True)
    total4 = torch.tensor(start4["encoder"], dtype=torch.float64, requires_grad=True)
    penalty2 = penalty_coefficient * torch.sum(old2**2)
    penalty4 = penalty_coefficient * torch.sum(total4**2)
    penalty2.backward()
    penalty4.backward()
    assert torch.equal(old2.grad, total4.grad[:2])
    assert penalty4.item() / penalty2.item() == pytest.approx(1.0025, abs=1e-12)


def test_r4_full_start_runs_two_synthetic_adam_updates_with_430_parameters():
    from kron_reduce.models.learned_latent import event_weights, visible_scales
    from kron_reduce.models.nonlinear_latent import (
        EquilibriumLearningSample,
        NonlinearFitSettings,
        NonlinearPassiveClosure,
        _features,
        fit_nonlinear_model,
    )

    source, _, _ = nested_fixture()
    mass = source["mass"]
    teacher = NonlinearPassiveClosure(**{key: source[key] for key in (
        "mass", "edge_stiffness", "damping", "coupling", "latent_skew", "latent_dissipation"
    )})
    history_clock = np.arange(11, dtype=np.float64) * 0.01 - 0.1
    forecast_clock = np.array([0.0, 0.01, 0.02, 0.03])
    shape = np.linspace(-1.0, 1.0, len(mass))
    samples = []
    allocation = (
        ("source-a", "line-1"),
        ("source-a", "line-1"),
        ("source-a", "line-2"),
        ("source-b", "line-3"),
    )
    for index, (source_name, line) in enumerate(allocation):
        q0 = coi_project(np.linspace(-0.035, 0.045, len(mass)) * (1 + 0.1 * index), mass)
        v0 = coi_project(np.linspace(0.025, -0.018, len(mass)) * (1 + 0.12 * index), mass)
        delta = np.stack([
            coi_project(q0 + (0.08 + 0.01 * index) * t * shape, mass)
            for t in history_clock
        ])
        omega = np.stack([
            coi_project(v0 + (0.17 + 0.02 * index) * t * shape, mass)
            for t in history_clock
        ])
        power = np.linspace(-0.13, 0.21, len(mass)) * (1 + 0.1 * index)
        power -= power.mean()
        equilibrium = coi_project(np.linspace(-0.025, 0.04, len(mass)) * (1 + 0.05 * index), mass)
        history = CausalStateHistory(history_clock, delta, omega, power)
        seed = np.array([0.015 * (index + 1), -0.008 * (index + 1)])
        target_path = teacher.rollout(
            delta[-1], omega[-1], seed, forecast_clock, power, equilibrium
        )
        samples.append(EquilibriumLearningSample(
            f"s{index}", source_name, line, "train", history, forecast_clock,
            target_path[:, : len(mass)], target_path[:, len(mass) : 2 * len(mass)], equilibrium,
        ))
    samples = tuple(samples)
    weights = event_weights(samples)
    features = np.stack([
        _features(sample.history, sample.equilibrium_delta, 3, "frames") for sample in samples
    ])
    feature_scale = np.maximum(np.sqrt(weights @ features**2), 1e-5)
    target_scale = visible_scales(samples)
    source = {**source, "feature_scale": feature_scale, "target_scale": target_scale}
    normalized_features = features / feature_scale
    start, audit = nested_memory_start(source, normalized_features, weights, group="r4")
    settings = NonlinearFitSettings(
        latent_dimension=4,
        frames=3,
        epochs=2,
        learning_rate=0.003,
        encoder_penalty=2e-4,
        history_weight=0.0,
        horizons_s=(0.03,),
        seed=4902,
    )
    snapshots = []

    def save_snapshot(step, model, state):
        snapshots.append((step, model, {key: np.array(value, copy=True) for key, value in state.items()}))

    fitted = fit_nonlinear_model(
        samples,
        mass,
        settings,
        full_start=start,
        checkpoint_steps=(2,),
        checkpoint_callback=save_snapshot,
    )
    assert len(snapshots) == 1
    step, checkpoint_model, optimizer_state = snapshots[0]
    assert step == 2
    assert audit["optimizer_updated_scalars"] == 430
    assert fitted.fit_record["parameter_count"] == 430
    assert fitted.fit_record["optimizer_updated_scalars"] == 430
    assert fitted.fit_record["initial_optimizer_step"] == 0
    assert fitted.fit_record["total_optimizer_steps"] == 2
    assert fitted.fit_record["planned_total_optimizer_steps"] == 2
    assert fitted.fit_record["initial_optimizer_moments_zero"] is True
    assert fitted.fit_record["exact_optimizer_resume"] is False
    assert fitted.fit_record["actual_optimizer_steps"] == 2
    assert max(fitted.fit_record["full_start_roundtrip_errors"].values()) <= 1e-12
    assert checkpoint_model.fit_record["actual_optimizer_steps"] == 2
    assert int(optimizer_state["completed_optimizer_steps"]) == 2
    assert int(optimizer_state["total_optimizer_steps"]) == 2
    assert optimizer_state["initial_optimizer_step"] == 0
    for name in ("raw", "port", "skew", "decay", "encoder"):
        assert np.all(optimizer_state[f"adam_step__{name}"] == 2)
        assert np.isfinite(optimizer_state[f"adam_exp_avg__{name}"]).all()
        assert np.isfinite(optimizer_state[f"adam_exp_avg_sq__{name}"]).all()
    assert np.linalg.norm(fitted.closure.coupling[:, 2:]) > 0
    assert all(np.isfinite(row["objective"]) for row in fitted.fit_record["loss_trace"])
