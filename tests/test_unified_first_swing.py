import json
import sys
from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from kron_reduce.evaluation import kiki_first_swing
from kron_reduce.evaluation.first_swing_loss import (
    momentum_window_loss,
    objective_terms,
    observed_velocity_increment,
    one_sided_velocity_loss,
    training_weights,
)
from kron_reduce.evaluation.kiki_machine_local_full import train_updates
from kron_reduce.evaluation.machine_local_topology import make_model
from kron_reduce.evaluation.machine_local_topology_truth import parameters, truth_rollout
from kron_reduce.models.machine_local_topology import predict_from_topology_history


def _unmetered(*_):
    pass


def test_weights_use_only_train_history_and_break_energy_ties_by_source_id():
    events = []
    values = [
        ("source-b:04", "stable", 2),
        ("source-a:04", "stable", 2),
        ("source-a:03", "unstable", 3),
        ("source-a:02", "unstable", 1),
        ("source-a:01", "stable", 1),
        ("source-b:01", "stable", 1),
        ("source-b:02", "stable", 1),
    ]
    for event_id, label, amplitude in values:
        path = np.zeros((12, 20))
        path[:10, 10:12] = [amplitude, -amplitude]
        path[-1, 10:12] = [1e6, -1e6] if event_id == "source-a:01" else [0., 0.]
        events.append({"id": event_id, "role": "train", "label": label, "x": path, "mass": np.ones(10)})

    weights = training_weights(events, 6., .15)
    assert set(weights) == {row["id"] for row in events}
    assert {key for key, value in weights.items() if value == 6.} == {"source-a:04", "source-a:03", "source-a:02"}
    assert all(value in (1., 6.) for value in weights.values())


def test_velocity_penalty_respects_observed_sign_valid_length_and_half_second_cap():
    observed = torch.ones((2, 53, 1), dtype=torch.float64) * 2.
    observed[1] *= -1.
    forecast = observed.clone()
    forecast[0, :, 0] = -1e6
    forecast[0, [1, 2], 0] = forecast.new_tensor([1., 3.])
    forecast[1, :, 0] = -3.
    forecast[1, [0, 1, 51, 52], 0] = 2.
    forecast.requires_grad_()

    loss = one_sided_velocity_loss(observed, forecast, observed.new_tensor([2.]), np.array([2, 52]))
    torch.testing.assert_close(loss, observed.new_tensor([.125, .08]))
    loss.sum().backward()
    expected = torch.zeros_like(forecast)
    expected[0, 1, 0], expected[1, 1, 0] = -.25, .04
    torch.testing.assert_close(forecast.grad, expected)


def test_momentum_penalty_uses_complete_windows_and_differences_the_integral():
    observed = torch.arange(41, dtype=torch.float64).reshape(1, 41, 1).repeat(3, 1, 1) + 7.
    increment = .5 * (observed - 7.)
    increment[0, 20:] = 1e6
    increment[1, 40] = 1e6
    increment[2, 40] = 38.
    scale = observed.new_tensor([2.])

    loss = momentum_window_loss(observed, increment, scale, np.array([19, 39, 40]))
    torch.testing.assert_close(loss, observed.new_tensor([0., 25., 20.5]))
    incomplete = momentum_window_loss(observed[:1, :20], increment[:1, :20], scale, np.array([19]))
    torch.testing.assert_close(incomplete, observed.new_zeros(1))


class _AnalyticObserver:
    n_nodes = memory_dimension = latent_dimension = 1
    mass = torch.ones(1, dtype=torch.float64)

    def _tensor(self, value):
        return torch.as_tensor(value, dtype=self.mass.dtype)

    def _rhs(self, power, equilibrium):
        def field(state):
            _, velocity, hidden = state.split(1, dim=-1)
            return torch.cat((velocity, velocity + hidden, power), dim=-1)

        return field


def test_observed_path_impulse_carries_hidden_state_and_stops_at_truth_end():
    model = _AnalyticObserver()
    seconds = torch.arange(5, dtype=torch.float64) * .01
    visible = torch.stack((torch.zeros_like(seconds), 1. + 3. * seconds), dim=-1).repeat(2, 1, 1)
    visible[1, 3:, 1] = 1e6
    z0 = seconds.new_tensor([[.4], [-.2]]).requires_grad_()
    power = seconds.new_tensor([[2.], [-1.]])
    zeros = seconds.new_zeros((2, 1))

    impulse = observed_velocity_increment(model, visible, z0, power, zeros, np.array([4, 2]),
                                          step=.005, spend=_unmetered)
    eligible_time = seconds.expand(2, -1).clone()
    eligible_time[1].clamp_max_(.02)
    expected = ((z0.detach() + 1.) * eligible_time + (power + 3.) / 2. * eligible_time.square())[..., None]
    torch.testing.assert_close(impulse, expected, atol=1e-12, rtol=0)
    impulse[:, -1].sum().backward()
    torch.testing.assert_close(z0.grad, seconds.new_tensor([[.04], [.02]]), atol=1e-12, rtol=0)


def test_adaptive_gn_uses_zero_start_history_residual_and_keeps_gradient():
    model = make_model(parameters(), "T0M1", "cpu", train_scales=True)
    n = model.n_nodes
    state = model.mass.new_zeros((2, 3*n))
    state[1, 2*n:2*n+2] = state.new_tensor([.8, -.8])
    zeros = state[:, :n]
    scale = state.new_full((2*n,), .01)
    with torch.no_grad():
        observed = model.rollout(state, zeros, zeros, 10, max_step=.005, spend=_unmetered)

    settings = {"max_step": .005, "spend": _unmetered, "memory_radius": .1,
                "prior_weight": 1e-6, "adaptive_prior_reference": .1}
    args = model, observed[:, :, :n], observed[:, :, n:2*n], zeros, zeros, scale
    result = predict_from_topology_history(*args, 2, **settings)
    r0 = result["zero_history_nrmse"]
    prior = result["adaptive_prior_weight"]
    torch.testing.assert_close(r0, (observed[:, 1:, :2*n] / scale).square().mean(dim=(1, 2)).sqrt())
    torch.testing.assert_close(prior, 1e-6 / (1. + (r0 / .1).square()))
    assert prior[1] < prior[0]
    hidden_norm = result["hidden_start"].norm(dim=-1)
    assert torch.all(hidden_norm < .2)
    assert hidden_norm[1] > .1
    torch.testing.assert_close(result["forecast"][:, 0, :2*n], observed[:, -1, :2*n])
    with torch.no_grad():
        origin_only = predict_from_topology_history(*args, 0, **settings)
    torch.testing.assert_close(result["hidden_start"], origin_only["hidden_start"])
    gradients = torch.autograd.grad(result["forecast"][:, -1, n:2*n].square().sum(),
                                    (model.port_parameter, model.rate_parameter))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_first_swing_objective_updates_shared_model_on_known_truth(tmp_path, device):
    params = parameters()
    mass = params[0]
    start = np.zeros((1, 10))
    motion = start.copy()
    motion[0, :2] = [.13, -.11]
    motion -= (motion@mass)[:, None]/mass.sum()
    path = truth_rollout(params, motion, motion, start, start, start, count=52,
                         max_step=.00125, spend=_unmetered, angle_damping_rho=2/3, spatial_mix=1.)
    event = {"id": "fixture/unstable", "role": "train", "label": "unstable", "x": path[0, :, :20],
             "p0": start[0], "equilibrium": params[-1], "mass": mass}
    candidate = make_model(params, "T0M1", device, train_scales=True)
    candidate.set_angle_damping(2/3, spatial_mix=1.)
    with torch.no_grad():
        candidate.log_stiffness_factor.fill_(.1)
    settings = {"updates": 1, "train_batch": 1, "prior_weight": 1e-6, "origin_energy_weight": .1,
                "origin_energy_radius_multiplier": 2., "lambda_asym": 1., "lambda_momentum": .5}
    scales = {"history": np.ones(20), "target": np.ones(20), "memory_radius": 1.}
    objective = partial(objective_terms, settings=settings, weights={event["id"]: 8.})
    with pytest.raises(ValueError, match="restricted to train"):
        objective(candidate, [event | {"role": "val"}], 1, scales, SimpleNamespace(spend=_unmetered))
    updates = train_updates(candidate, [event], scales, SimpleNamespace(spend=_unmetered), tmp_path, settings,
                            stiffness_bounds=(-.7, .7), damping_bounds=(-.7, .7), minimum_port=.5,
                            objective_fn=objective)
    row = next(updates)
    updates.close()
    assert row["weighted_rollout"] == pytest.approx(8.*row["train_nmse_1s_3s"])
    assert row["weak_momentum"] > 0. and row["gradient_norm"] > 0.
    assert torch.all(candidate.port_parameter.exp() >= .5)
    assert candidate.log_stiffness_factor.item() != .1


def test_partial_score_remains_a_failed_checkpoint():
    stability = {"allocated": 107, "uncovered": 0, "numerical_failures": 1, "tp": 11,
                 "fp": 0, "unsettled_predictions": 0}
    stable = {"8": {"qualified": 94, "qualified_subset_diagnostic": {"mean": .2}}}
    result = {"stability": stability, "stable_complete": stable, "task3_six_term_score": None}
    record = kiki_first_swing.checkpoint_metrics(result, {}, 6)
    assert record["passed"] is False
    assert record["rank"] == [2, 1, -11, 0, 0, None, None, 6]


def test_initial_freeze_binds_source_backbone_before_val(tmp_path, monkeypatch):
    params = parameters()
    mass = params[0]
    event = {"id": "fixture/stable", "role": "train", "label": "stable", "x": np.zeros((22, 20)), "mass": mass}
    scales = kiki_first_swing.scales_from_train([event], mass)
    encoded = json.dumps({key: np.asarray(value).tolist() for key, value in scales.items()}).encode()
    plan_path = tmp_path/"plan.toml"
    plan_path.write_text("fixture = true\n", encoding="utf-8")
    plan = {"plan_path": plan_path.name, "operation_limits": {"checkpoint_loads": 2},
            "settings": {"minimum_port": .01, "large_event_weight": 8., "history_kinetic_fraction": .15}}
    source = {"task3_root": str(tmp_path), "sources": {}, "backbone": {"path": "anchor.npz", "sha256": "fixture"}}
    directory = tmp_path/"result"
    directory.mkdir()

    class InitialFreezeChecked(Exception):
        pass

    class SyntheticReader:
        def __init__(self, root, contract, journal):
            assert contract["allowed_roles"] == ["train", "val"]
            self.static = {"mass": mass}

        def load(self, role, freeze_path=None):
            if role == "train":
                return [event]
            assert role == "val"
            frozen = json.loads(freeze_path.read_bytes())
            assert frozen["backbone"] == source["backbone"]
            assert frozen["phase"] == "initial"
            assert (tmp_path/frozen["checkpoint_path"]).is_file()
            raise InitialFreezeChecked

    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.setattr(torch, "set_num_threads", lambda *_: None)
    monkeypatch.setattr(torch, "use_deterministic_algorithms", lambda *_: None)
    monkeypatch.setattr("kron_reduce.evaluation.kiki_machine_local_topology.git_head", lambda _: "fixture")
    monkeypatch.setattr(kiki_first_swing, "Task3FullReader", SyntheticReader)
    monkeypatch.setattr(kiki_first_swing.LearnedNonlinearModel, "load",
                        lambda _: SimpleNamespace(closure=SimpleNamespace(mass=mass)))
    monkeypatch.setattr(kiki_first_swing, "make_model", lambda *_, **__: make_model(params, "T0M1", "cpu",
                                                                                 train_scales=True))
    monkeypatch.setattr(kiki_first_swing, "load_snapshot", lambda *_: None)
    with pytest.raises(InitialFreezeChecked):
        kiki_first_swing.execute(tmp_path, directory, plan, source,
                                {"backbone": b"", "checkpoint": b"", "preprocessing": encoded})
