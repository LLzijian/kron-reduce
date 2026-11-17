import numpy as np
import torch

from kron_reduce.evaluation.joint_energy_sindy_synthetic import _energy
from kron_reduce.evaluation.joint_energy_sindy_truth import truth_rollout
from kron_reduce.models.joint_energy_sindy import JointEnergySINDy


def test_joint_equation_agrees_with_independent_truth_and_balances_energy():
    rng = np.random.default_rng(143)
    mass = np.linspace(1., 1.9, 10)
    model = JointEnergySINDy.from_mass(mass)
    stiffness = np.linspace(.2, 1.2, 45)
    damping = np.linspace(.1, .3, 10)
    model.set_visible_seed(stiffness, damping)
    with torch.no_grad():
        model.harmonics.copy_(model._tensor((-.17, .09)))
        model.topology_port_parameter.fill_(.31)
    terms = {key: value.detach().numpy() for key, value in model.effective_coefficients().items()}
    coefficients = {name: terms[name] for name in ("tau", "mobility", "harmonics", "angle_damping",
                    "edge_damping", "speed_damping", "topology_port", "memory_decay_factor")}
    coefficients.update(stiffness=stiffness, node_damping=damping, storage=terms["topology_storage"],
                        ports=terms["port"], saturation=terms["saturation"], port_topology=terms["topology_port"])
    x = rng.normal(0., .35, (2, 50))
    p = rng.normal(0., .05, (2, 10)) @ terms["projection"].T
    eq = rng.normal(0., .2, (2, 10))
    for offset in (0, 10):
        block = x[:, offset:offset+10]
        block -= (block@mass/mass.sum())[:, None]
    expected = truth_rollout(mass, x, p, eq, coefficients, count=2, step=.005, spend=lambda *_: None)
    with torch.no_grad():
        actual = model.rollout(x, p, eq, 2, max_step=.005)
    np.testing.assert_allclose(actual.numpy(), expected, atol=2e-12, rtol=2e-12)

    class Budget:
        def spend(self, *_):
            pass

    assert _energy(model, x, model._tensor(p), model._tensor(eq), Budget()) <= 1e-12

    direction = model._tensor(rng.normal(0., 1., (2, 50, 2)))
    state, power, equilibrium = map(model._tensor, (x, p, eq))
    _, tangent = model._rhs(power, equilibrium, tangent=True)(state, direction)
    step = 5e-7
    expected_tangent = torch.stack([
        (model.rhs(state+step*d, power, equilibrium)-model.rhs(state-step*d, power, equilibrium))/(2*step)
        for d in direction.unbind(dim=-1)
    ], dim=-1)
    torch.testing.assert_close(tangent, expected_tangent, atol=2e-8, rtol=2e-7)
