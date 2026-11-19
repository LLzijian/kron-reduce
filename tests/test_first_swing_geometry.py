from types import SimpleNamespace

import numpy as np
import pytest
import torch

from kron_reduce.evaluation import first_swing_geometry


def test_probe_accumulates_weighted_signed_geometry_in_named_parameter_order(monkeypatch):
    model = torch.nn.Module()
    model.port_parameter = torch.nn.Parameter(torch.ones(10, dtype=torch.float64))
    model.rate_parameter = torch.nn.Parameter(torch.ones(10, dtype=torch.float64))
    model.log_stiffness_factor = torch.nn.Parameter(torch.tensor(.2, dtype=torch.float64))
    model.log_damping_factor = torch.nn.Parameter(torch.tensor(-.1, dtype=torch.float64))
    history = np.zeros((11, 20))
    mass = np.ones(10)
    events = [{"id": f"fixture/{i:03d}", "role": "train", "label": "unstable" if i == 128 else "stable",
               "x": history, "mass": mass} for i in range(129)]
    settings = {"large_event_weight": 8., "history_kinetic_fraction": .15, "train_batch": 128}
    visits, charged = [], []

    def objective(candidate, batch, population, scales, budget, settings, *, weights):
        assert candidate is model and population == len(events)
        visits.append([row["id"] for row in batch])
        scale = sum(weights[row["id"]] for row in batch)/population
        port = 3.*candidate.port_parameter[0]
        stiffness = 4.*candidate.log_stiffness_factor
        rollout = scale*(port+stiffness)
        velocity = scale*(port-stiffness+12.*candidate.log_damping_factor)
        momentum = 0.*candidate.rate_parameter[-1] if len(batch) == 128 else port.new_zeros(())
        return rollout+velocity+momentum, {"weighted_rollout": rollout, "one_sided_velocity": velocity,
                                          "weak_momentum": momentum}

    monkeypatch.setattr(first_swing_geometry, "objective_terms", objective)
    probe = first_swing_geometry.gradient_probe(model, events, {}, SimpleNamespace(spend=charged.append), settings)
    assert visits == [[row["id"] for row in batch] for batch in (events[:128], events[128:])]
    assert charged == ["gradients"]*5
    assert probe["terms"] == ["weighted_rollout", "one_sided_velocity", "weak_momentum"]
    scalars = ["log_stiffness_factor", "log_damping_factor"]
    assert probe["parameter_coordinates"] == [f"{name}[{i}]" for name in ("port_parameter", "rate_parameter")
                                               for i in range(10)] + scalars
    # Twenty high-history ties and the last unstable event carry weight eight.
    scale = (21*8.+108.)/129.
    rollout = [3.*scale]+[0.]*19+[4.*scale, 0.]
    velocity = [3.*scale]+[0.]*19+[-4.*scale, 12.*scale]
    assert probe["gradients"]["weighted_rollout"] == pytest.approx(rollout)
    assert probe["gradients"]["one_sided_velocity"] == pytest.approx(velocity)
    assert probe["gradients"]["weak_momentum"] == [0.]*22
    assert probe["norms"] == pytest.approx({"weighted_rollout": 5.*scale, "one_sided_velocity": 13.*scale,
                                          "weak_momentum": 0.})
    assert probe["cosines"][0][:2] == pytest.approx([1., -7./65.])
    assert probe["cosines"][1][:2] == pytest.approx([-7./65., 1.])
    assert [row[2] for row in probe["cosines"]] == probe["cosines"][2] == [None]*3
    assert probe["scalars"]["weighted_rollout"] == pytest.approx(dict(zip(scalars, rollout[-2:], strict=True)))
    assert probe["scalars"]["one_sided_velocity"] == pytest.approx(dict(zip(scalars, velocity[-2:], strict=True)))
    assert all(parameter.grad is None for parameter in model.parameters())
