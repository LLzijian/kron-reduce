"""Synthetic regression tests for tilted risk and exact Adam continuation."""

from dataclasses import replace

import numpy as np
import pytest
from test_joint_refinement import make_full_start
from test_paired_training import tiny_train

from kron_reduce.models.nonlinear_latent import _tilted_risk, fit_nonlinear_model


def snapshot_model(model):
    c = model.closure
    return {
        "mass": c.mass.copy(), "edge_stiffness": c.edge_stiffness.copy(),
        "damping": c.damping.copy(), "coupling": c.coupling.copy(),
        "latent_skew": c.latent_skew.copy(), "latent_dissipation": c.latent_dissipation.copy(),
        "encoder": model.encoder.copy(), "feature_scale": model.feature_scale.copy(),
        "target_scale": model.target_scale.copy(),
    }


def checkpoint_fit(samples, mass, settings):
    saved = {}

    def callback(step, model, state):
        saved[step] = (model, {k: np.array(v, copy=True) for k, v in state.items()})

    result = fit_nonlinear_model(
        samples, mass, settings, full_start=make_full_start(mass, samples, settings),
        checkpoint_steps=(3,), checkpoint_callback=callback,
    )
    return result, saved[3]


def test_adam_resume_matches_uninterrupted_six_steps():
    mass, _, samples, settings = tiny_train()
    full = make_full_start(mass, samples, settings)
    continuous_settings = replace(settings, epochs=6)
    direct_state = {}

    def direct_callback(step, model, state):
        direct_state.update({k: np.array(v, copy=True) for k, v in state.items()})

    direct = fit_nonlinear_model(
        samples, mass, continuous_settings, full_start=full,
        checkpoint_steps=(6,), checkpoint_callback=direct_callback,
    )
    first_settings = replace(settings, epochs=3)
    first_full = full
    split_state = {}

    def first_callback(step, model, state):
        split_state["model"] = model
        split_state["state"] = {k: np.array(v, copy=True) for k, v in state.items()}

    first = fit_nonlinear_model(
        samples, mass, first_settings, full_start=first_full,
        checkpoint_steps=(3,), checkpoint_callback=first_callback,
    )
    assert first.fit_record["total_optimizer_steps"] == 3
    resumed_state = {}

    def resumed_callback(step, model, state):
        resumed_state.update({k: np.array(v, copy=True) for k, v in state.items()})

    resumed = fit_nonlinear_model(
        samples, mass, replace(settings, epochs=3), full_start=snapshot_model(split_state["model"]),
        optimizer_start=split_state["state"], checkpoint_steps=(3,), checkpoint_callback=resumed_callback,
    )
    assert resumed.fit_record["total_optimizer_steps"] == 6
    assert resumed_state["total_optimizer_steps"] == direct_state["total_optimizer_steps"] == 6
    for key, direct_value in direct_state.items():
        if key.startswith(("parameter__", "adam_exp_avg__", "adam_exp_avg_sq__", "adam_step__")):
            assert np.array_equal(resumed_state[key], direct_value), key
    for name in ("mass", "edge_stiffness", "damping", "coupling", "latent_skew", "latent_dissipation"):
        assert np.allclose(getattr(resumed.closure, name), getattr(direct.closure, name), atol=1e-12, rtol=0)
    assert np.allclose(resumed.encoder, direct.encoder, atol=1e-12, rtol=0)


@pytest.mark.parametrize("tau", [0, -1, float("nan"), float("inf"), True])
def test_rejects_invalid_risk_temperature(tau):
    mass, _, samples, settings = tiny_train()
    with pytest.raises(ValueError, match="risk_temperature"):
        fit_nonlinear_model(samples, mass, settings, risk_temperature=tau)


def test_tilted_risk_converges_to_weighted_mean_for_large_temperature():
    mass, _, samples, settings = tiny_train()
    model = fit_nonlinear_model(samples, mass, replace(settings, epochs=1), risk_temperature=1e8)
    trace = model.fit_record["loss_trace"][0]
    assert trace["risk_temperature"] == 1e8
    assert trace["future_objective"] == pytest.approx(trace["future_loss"], abs=2e-8)
    assert np.isfinite(trace["tilted_future_risk"])


def test_tilted_risk_value_weights_and_autograd_match_formula():
    import torch

    losses = torch.tensor([0.2, 1.1, 2.0], dtype=torch.float64, requires_grad=True)
    weights = torch.tensor([0.6, 0.3, 0.1], dtype=torch.float64)
    tau = 0.7
    objective, gradient_weights = _tilted_risk(losses, weights, tau)
    logits = torch.log(weights) + losses.detach() / tau
    expected_weights = torch.softmax(logits, dim=0)
    expected_value = tau * torch.logsumexp(logits, dim=0)
    assert objective.item() == pytest.approx(expected_value.item(), rel=1e-14, abs=1e-14)
    assert torch.allclose(gradient_weights, expected_weights, rtol=1e-14, atol=1e-14)
    assert gradient_weights.sum().item() == pytest.approx(1.0, abs=1e-14)
    objective.backward()
    assert torch.allclose(losses.grad, expected_weights, rtol=1e-14, atol=1e-14)


def test_tilted_risk_extreme_logits_remain_finite_and_normalized():
    import torch

    losses = torch.tensor([1e6, -1e6, 0.0], dtype=torch.float64, requires_grad=True)
    weights = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    objective, gradient_weights = _tilted_risk(losses, weights, 0.01)
    assert torch.isfinite(objective)
    assert torch.isfinite(gradient_weights).all()
    assert gradient_weights.sum().item() == pytest.approx(1.0, abs=1e-14)
    assert torch.allclose(gradient_weights, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))
    objective.backward()
    assert torch.isfinite(losses.grad).all()
    assert torch.allclose(losses.grad, gradient_weights)


def test_optimizer_resume_rejects_malformed_state_and_mode_conflicts():
    mass, _, samples, settings = tiny_train()
    initial = make_full_start(mass, samples, settings)
    saved = {}

    def callback(step, model, state):
        saved.update({k: np.array(v, copy=True) for k, v in state.items()})
        saved["model"] = model

    fit_nonlinear_model(samples, mass, settings, full_start=initial,
                        checkpoint_steps=(settings.epochs,), checkpoint_callback=callback)
    full = snapshot_model(saved["model"])
    state = {k: v for k, v in saved.items() if k != "model"}
    with pytest.raises(ValueError, match="requires the matching full_start"):
        fit_nonlinear_model(samples, mass, settings, optimizer_start=state)
    with pytest.raises(ValueError, match="paired_start"):
        fit_nonlinear_model(samples, mass, settings, full_start=full, optimizer_start=state,
                            paired_start={})

    invalid = []
    bad = {k: np.array(v, copy=True) for k, v in state.items()}
    bad["parameter__raw"] = bad["parameter__raw"].copy()
    bad["parameter__raw"].flat[0] = np.nan
    invalid.append(bad)
    bad = {k: np.array(v, copy=True) for k, v in state.items()}
    bad["adam_exp_avg_sq__raw"] = bad["adam_exp_avg_sq__raw"].copy()
    bad["adam_exp_avg_sq__raw"].flat[0] = -1
    invalid.append(bad)
    bad = {k: np.array(v, copy=True) for k, v in state.items()}
    bad["adam_step__raw"] = np.asarray(0.0)
    invalid.append(bad)
    bad = {k: np.array(v, copy=True) for k, v in state.items()}
    bad["adam_learning_rate"] = np.asarray(settings.learning_rate * 2)
    invalid.append(bad)
    for corrupted in invalid:
        with pytest.raises(ValueError):
            fit_nonlinear_model(samples, mass, settings, full_start=full, optimizer_start=corrupted)

    changed_physical = dict(full, edge_stiffness=full["edge_stiffness"] * 1.01)
    with pytest.raises(ValueError):
        fit_nonlinear_model(samples, mass, settings, full_start=changed_physical, optimizer_start=state)
