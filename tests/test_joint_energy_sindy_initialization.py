import math

import numpy as np
import pytest
import torch

from kron_reduce.models.joint_energy_sindy import JointEnergySINDy


def test_physical_seed_round_trips_without_disabling_small_modules():
    seed = {
        "mobility": [.2, .2],
        "topology_storage": [1., 1.],
        "saturation": .1,
        "angle_damping": .05,
        "edge_damping": .01,
        "speed_damping": .01,
        "topology_port": 0.,
        "harmonics": [0., 0.],
        "memory_decay_factor": [.01, -.01]*10,
    }
    model = JointEnergySINDy.from_mass(np.linspace(1., 1.9, 10), physical_seed=seed)
    physical = model.effective_coefficients()
    for name, expected in seed.items():
        torch.testing.assert_close(physical[name], model._tensor(expected), atol=1e-15, rtol=1e-14)
    assert torch.all(model.group_gate == 1.)
    # Small positive coefficients retain a trainable, finite raw coordinate.
    sum(physical[name].sum() for name in seed).backward()
    for name in ("edge_damping_parameter", "speed_damping_parameter",
                 "saturation_parameter", "angle_damping_parameter"):
        gradient = getattr(model, name).grad
        assert torch.isfinite(gradient).all() and torch.all(gradient > 0.)


def test_omitting_physical_seed_preserves_historical_initialization():
    model = JointEnergySINDy.from_mass(np.ones(10))
    physical = model.effective_coefficients()
    assert physical["edge_damping"].item() == pytest.approx(math.log(2.))
    assert physical["speed_damping"].item() == pytest.approx(math.log(2.))
    assert physical["angle_damping"].item() == pytest.approx(.475)


@pytest.mark.parametrize("bad", [
    {"speed_damping": 0.},
    {"speed_damping": -1.},
    {"angle_damping": .95},
    {"harmonics": [.5, 0.]},
    {"mobility": [.1]},
    {"topology_storage": [float("nan"), 1.]},
    {"memory_decay_factor": [0.]*20},
    {"not_a_coefficient": .1},
])
def test_invalid_physical_seed_does_not_partially_change_model(bad):
    model = JointEnergySINDy.from_mass(np.ones(10))
    before = {key: value.clone() for key, value in model.state_dict().items()}
    with pytest.raises(ValueError):
        model.set_physical_seed({"saturation": .1, **bad})
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], atol=0., rtol=0.)
