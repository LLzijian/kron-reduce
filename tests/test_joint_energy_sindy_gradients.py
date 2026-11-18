"""Regression checks for the complete causal training derivative."""
import numpy as np
import pytest
import torch

from kron_reduce.evaluation import joint_energy_sindy_synthetic as synthetic
from kron_reduce.evaluation import kiki_joint_energy_sindy as real
from kron_reduce.evaluation.joint_energy_sindy_training import causal_rollout
from kron_reduce.models.joint_energy_sindy import JointEnergySINDy


class Budget:
    def spend(self, *_):
        pass


def test_five_step_history_gradient_matches_reestimated_finite_difference(record_property):
    """Both finite-difference sides repeat all five GN steps and the warmup."""
    torch.set_num_threads(1)
    model = JointEnergySINDy.from_mass(np.linspace(1., 1.9, 10))
    generator = torch.Generator().manual_seed(231)
    initial = .05*torch.randn((1, 50), generator=generator, dtype=torch.float64)
    for start in (0, 10):
        block = initial[:, start:start+10]
        block -= (block@model.mass/model.mass.sum())[:, None]
    power = .03*torch.randn((1, 10), generator=generator, dtype=torch.float64)
    power -= power.mean(dim=-1, keepdim=True)
    equilibrium = .1*torch.sin(torch.arange(10, dtype=torch.float64))[None]
    with torch.no_grad():
        observed = model.rollout(initial, power, equilibrium, 15)
        model.log_global_stiffness.add_(.08)
    scale = torch.full((20,), .1, dtype=torch.float64)
    settings = {"memory_radius": 1., "prior_weight": 1e-6, "gradient_mode": "full"}

    def objective():
        forecast = causal_rollout(model, observed[:, :11, :10], observed[:, :11, 10:20],
                                  power, equilibrium, scale, 5, settings=settings,
                                  spend=Budget().spend)
        return ((forecast[:, 1:, :20]-observed[:, 11:, :20])/scale).square().mean()

    parameters = list(model.parameters())
    direction = [torch.randn(p.shape, generator=generator, dtype=p.dtype) for p in parameters]
    norm = torch.sqrt(sum(d.square().sum() for d in direction))
    direction = [d/norm for d in direction]
    value = objective()
    gradients = torch.autograd.grad(value, parameters)
    assert all(torch.isfinite(g).all() for g in gradients)
    automatic = sum((g*d).sum() for g, d in zip(gradients, direction, strict=True))
    original = [p.detach().clone() for p in parameters]
    epsilon = 2e-5
    values = []
    try:
        with torch.no_grad():
            for sign in (1., -1.):
                for p, base, d in zip(parameters, original, direction, strict=True):
                    p.copy_(base+sign*epsilon*d)
                values.append(objective())
    finally:
        with torch.no_grad():
            for p, base in zip(parameters, original, strict=True):
                p.copy_(base)
    finite_difference = (values[0]-values[1])/(2*epsilon)
    record_property("autograd_directional_derivative", float(automatic))
    record_property("finite_difference_directional_derivative", float(finite_difference))
    record_property("directional_relative_error", float(
        (automatic-finite_difference).abs()/torch.maximum(automatic.abs(), finite_difference.abs())))
    assert abs(float(automatic)) > 1e-10
    torch.testing.assert_close(automatic, finite_difference, rtol=3e-4, atol=1e-11)


@pytest.mark.parametrize("entry", ("synthetic", "real"))
def test_training_entry_keeps_causal_initializer_gradient(monkeypatch, entry):
    """A differentiable inferred-origin signal must reach both train losses."""
    model = JointEnergySINDy.from_mass(np.linspace(1., 1.9, 10))
    module = synthetic if entry == "synthetic" else real

    def inferred_forecast(model, q, v, power, equilibrium, scale, count, **kwargs):
        # Isolate the initialization path from direct ODE-parameter derivatives.
        return (2.+model.log_global_stiffness).expand(len(q), count+1, 50)

    monkeypatch.setattr(module, "causal_rollout", inferred_forecast)
    settings = {"memory_radius": 1., "prior_weight": 1e-6, "gradient_mode": "full",
                "fit_frames": 3}
    scale = np.ones(20)
    if entry == "synthetic":
        observed = np.zeros((1, 14, 20))
        value = synthetic.trajectory_objective(
            model, observed, np.array([0]), model._tensor(np.zeros((1, 10))),
            model._tensor(np.zeros((1, 10))), model._tensor(scale), model._tensor(scale),
            3, settings, Budget(), training=True)
    else:
        events = [{"x": np.zeros((14, 20)), "p0": np.zeros(10), "equilibrium": np.zeros(10)}]
        value = real.train_objective(model, events, 1, {"history": scale, "target": scale},
                                     Budget(), settings, {})
    gradient, = torch.autograd.grad(value, (model.log_global_stiffness,))
    torch.testing.assert_close(gradient, torch.tensor(4., dtype=torch.float64))
