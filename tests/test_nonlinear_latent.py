from dataclasses import replace

import numpy as np
import pytest

from kron_reduce.data.history_contract import CausalStateHistory
from kron_reduce.models.nonlinear_latent import (
    EquilibriumLearningSample,
    LearnedNonlinearModel,
    NonlinearFitSettings,
    NonlinearPassiveClosure,
    complete_incidence,
    fit_nonlinear_model,
)
from kron_reduce.physics.coordinates import coi_basis, coi_project


def examples():
    mass = np.array([1.0, 1.2, 1.5])
    basis = coi_basis(np.ones(3))
    truth = NonlinearPassiveClosure(
        mass,
        np.array([1.3, 0.7, 1.1]),
        0.2 * basis @ basis.T,
        basis[:, :1],
        np.zeros((1, 1)),
        np.ones((1, 1)) * 0.3,
    )
    rng = np.random.default_rng(324)
    samples = []
    for i in range(4):
        eq = coi_project(rng.normal(0, 0.2, 3), mass)
        q = coi_project(rng.normal(0, 0.3, 3), mass)
        v = coi_project(rng.normal(0, 0.1, 3), mass)
        path = truth.rollout(q, v, rng.normal(size=1), np.arange(71) * 0.01, np.zeros(3), eq)
        history = CausalStateHistory(
            np.arange(21) * 0.01 - 0.2, path[:21, :3], path[:21, 3:6], np.zeros(3)
        )
        samples.append(
            EquilibriumLearningSample(
                f"e{i}",
                "A",
                f"L{i}",
                "train",
                history,
                np.arange(51) * 0.01,
                path[20:, :3],
                path[20:, 3:6],
                eq,
            )
        )
    return truth, tuple(samples)


def test_sine_energy_derivative_and_coi():
    truth, samples = examples()
    sample = samples[0]
    state = np.r_[sample.history.delta[-1], sample.history.omega[-1], 0.7]
    power = np.array([0.2, -0.3, 0.1])
    derivative = truth.rhs(state, power, sample.equilibrium_delta)
    eps = 1e-6
    finite = (
        truth.energy(state + eps * derivative, sample.equilibrium_delta)
        - truth.energy(state - eps * derivative, sample.equilibrium_delta)
    ) / (2 * eps)
    identity = truth.energy_identity(state, power, sample.equilibrium_delta)
    assert identity["identity_error"] < 1e-12
    assert abs(finite - identity["energy_rate"]) < 1e-9
    assert abs(derivative[3:6] @ truth.mass) < 1e-12
    assert np.max(np.abs(truth.rhs(np.zeros(7), np.zeros(3), sample.equilibrium_delta))) < 1e-14


def test_same_visible_state_different_hidden_futures_and_equilibrium():
    truth, samples = examples()
    s = samples[0]
    args = (s.history.delta[-1], s.history.omega[-1])
    p1 = truth.rollout(
        *args, np.ones(1), s.forecast_time, s.history.constant_power, s.equilibrium_delta
    )
    p2 = truth.rollout(
        *args, -np.ones(1), s.forecast_time, s.history.constant_power, s.equilibrium_delta
    )
    p3 = truth.rollout(*args, np.ones(1), s.forecast_time, s.history.constant_power, np.zeros(3))
    assert np.max(np.abs(p1[:, :6] - p2[:, :6])) > 0.1
    assert np.max(np.abs(p1 - p3)) > 0.001


def test_nonlinear_rk4_order_and_coi_preservation():
    truth, samples = examples()
    s = samples[0]
    paths = [
        truth.rollout(
            s.history.delta[-1],
            s.history.omega[-1],
            np.ones(1),
            s.forecast_time,
            s.history.constant_power,
            s.equilibrium_delta,
            max_step=h,
        )
        for h in (0.01, 0.005, 0.0025)
    ]
    assert np.max(np.abs(paths[0][:, :3] @ truth.mass)) < 1e-12
    assert np.linalg.norm(paths[0] - paths[2]) > 10 * np.linalg.norm(paths[1] - paths[2])


def test_learning_deterministic_checkpoint_causal_and_loss_replay(tmp_path):
    truth, samples = examples()
    settings = NonlinearFitSettings(
        latent_dimension=1, epochs=1, horizons_s=(0.2, 0.5), learning_rate=1e-15
    )
    first = fit_nonlinear_model(samples, truth.mass, settings)
    second = fit_nonlinear_model(samples, truth.mass, settings)
    assert np.array_equal(first.encoder, second.encoder)
    s = samples[0]
    prediction = first.predict(s.history, s.forecast_time, s.equilibrium_delta)
    changed = replace(
        s,
        target_delta=s.target_delta + np.r_[0.0, np.ones(50)][:, None],
        source_dataset="arbitrary",
        fault_line="other",
    )
    assert np.array_equal(
        prediction, first.predict(changed.history, changed.forecast_time, changed.equilibrium_delta)
    )
    loss = 0.0
    for row in samples:
        z, history_path = first.initialize(row.history, row.equilibrium_delta)
        forecast_path = first.predict(row.history, row.forecast_time, row.equilibrium_delta)
        assert np.array_equal(forecast_path[0, 6:], z)
        history_loss = np.mean(
            (
                (history_path[1:, :6] - np.c_[row.history.delta[1:], row.history.omega[1:]])
                / first.target_scale
            )
            ** 2
        )
        error = (
            (forecast_path[1:, :6] - np.c_[row.target_delta[1:], row.target_omega[1:]])
            / first.target_scale
        ) ** 2
        loss += (history_loss + (np.mean(error[:20]) + np.mean(error[:50])) / 2) / len(samples)
    loss += settings.encoder_penalty * np.mean(first.encoder**2)
    assert abs(loss - first.fit_record["loss_trace"][0]["objective"]) < 1e-10
    first.save(tmp_path / "model.npz")
    loaded = LearnedNonlinearModel.load(tmp_path / "model.npz")
    assert np.array_equal(
        prediction, loaded.predict(s.history, s.forecast_time, s.equilibrium_delta)
    )
    with pytest.raises(FileExistsError):
        first.save(tmp_path / "model.npz")


@pytest.mark.parametrize("role", ["val", "audit"])
def test_locked_roles_cannot_fit(role):
    truth, samples = examples()
    with pytest.raises(ValueError, match="train only"):
        fit_nonlinear_model(
            [replace(samples[0], role=role)], truth.mass, NonlinearFitSettings(epochs=1)
        )


def test_failed_predictions_remain_in_population(monkeypatch):
    truth, samples = examples()
    model = fit_nonlinear_model(
        samples, truth.mass, NonlinearFitSettings(latent_dimension=0, epochs=1, horizons_s=(0.5,))
    )

    def fail(*args, **kwargs):
        raise FloatingPointError("divergence")

    monkeypatch.setattr(model, "predict", fail)
    result = model.score(samples)
    assert result["failed_event_count"] == 4
    assert result["normalized_rmse"] is None
    assert result["failures_imputed_as_zero"] is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"edge_stiffness": np.array([-1.0, 1.0, 1.0])},
        {"coupling": np.ones((3, 1))},
        {"damping": -np.eye(3)},
    ],
)
def test_invalid_passive_operators_rejected(kwargs):
    truth, _ = examples()
    with pytest.raises(ValueError):
        replace(truth, **kwargs)


def test_complete_graph_is_balanced():
    incidence = complete_incidence(10)
    assert incidence.shape == (10, 45)
    assert np.array_equal(incidence.sum(axis=0), np.zeros(45))
