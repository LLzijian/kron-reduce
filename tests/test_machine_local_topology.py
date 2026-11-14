import numpy as np
import pytest
import torch

from kron_reduce.evaluation.machine_local_topology import energy_residual, make_model
from kron_reduce.evaluation.machine_local_topology_truth import parameters
from kron_reduce.models.machine_local_topology import _ball


@pytest.mark.parametrize("support", ["T0M0", "T1M0", "T0M1", "T1M1"])
def test_topology_memory_energy_and_tangent(support):
    model = make_model(parameters(), support, "cpu")
    rng = np.random.default_rng(17)
    width = 20+model.latent_dimension
    x = model._tensor(rng.normal(0., .1, (2, width)))
    for visible in (x[:, :10], x[:, 10:20]):
        visible -= ((visible@model.mass)/model.mass.sum())[:, None]
    equilibrium = model._tensor(rng.normal(0., .08, (2, 10)))
    p = model._tensor(rng.normal(0., .03, (2, 10))) @ model.power_projection.T
    a = model.node_correction(x)
    torch.testing.assert_close(a.sum(-1), x.new_zeros(2), atol=1e-12, rtol=0)
    torch.testing.assert_close(model.edge_stiffness(x), model.stiffness*torch.exp(a@model.incidence.abs()))
    assert energy_residual(model, x, p, equilibrium, lambda *_: None) < 1e-12

    direction = model._tensor(rng.normal(size=(2, width)))
    field, tangent = model._rhs(p, equilibrium, tangent=True)(x, direction[..., None])
    torch.testing.assert_close(field, model.rhs(x, p, equilibrium))
    epsilon = 1e-6
    numerical = (model.rhs(x+epsilon*direction, p, equilibrium)
                 -model.rhs(x-epsilon*direction, p, equilibrium))/(2*epsilon)
    torch.testing.assert_close(tangent[..., 0], numerical, atol=1e-9, rtol=1e-8)
    torch.testing.assert_close(field[:, 20+model.memory_dimension:],
                               torch.zeros_like(field[:, 20+model.memory_dimension:]))


def test_initialization_ball_preserves_interior_and_bounds_exterior():
    x = torch.tensor([[.3, .4], [6., 8.], [0., 0.]], dtype=torch.float64, requires_grad=True)
    result = _ball(x, 1.)
    torch.testing.assert_close(result, x.new_tensor([[.3, .4], [.6, .8], [0., 0.]]))
    result.sum().backward()
    assert torch.isfinite(x.grad).all()
