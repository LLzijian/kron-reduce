"""Small synthetic checks for frozen memory coefficient substitutions."""

import numpy as np
import torch

from kron_reduce.evaluation.machine_local_equation import physical
from kron_reduce.evaluation.machine_local_memory_diagnosis import (
    MEMORY,
    VISIBLE,
    kernel,
    substitute,
)
from kron_reduce.models.machine_local_equation import JointMachineLocalModel
from kron_reduce.models.nonlinear_latent import NonlinearPassiveClosure, complete_incidence


def _model(*, offset=0.0):
    n = 10
    incidence = complete_incidence(n)
    stiffness = np.linspace(0.2, 0.5, incidence.shape[1])
    edge_damping = np.full(incidence.shape[1], 0.03)
    damping = (incidence * edge_damping) @ incidence.T
    backbone = NonlinearPassiveClosure(
        np.linspace(1.0, 2.0, n), stiffness, damping,
        np.zeros((n, 0)), np.zeros((0, 0)), np.zeros((0, 0)), "sine",
    )
    model = JointMachineLocalModel(
        backbone, port=np.linspace(0.2, 0.4, n) + offset,
        tau=np.linspace(0.1, 0.5, n), harmonic=True, modulated_damping=True,
    )
    with torch.no_grad():
        model.port_parameter.add_(offset)
        model.rate_parameter.add_(offset)
        model.edge_log_factor.fill_(offset)
        model.damping_log_factor.fill_(-offset)
        model.harmonic_strength.fill_(0.2 + offset)
        model.modulated_damping_strength.fill_(0.3 + offset)
    return model


def test_memory_diagnosis_substitutions_preserve_frozen_terms_and_kernel_controls():
    selected, truth = _model(), _model(offset=0.1)
    before = {name: value.detach().clone() for name, value in selected.state_dict().items()}
    visible = substitute(selected, truth, VISIBLE)
    memory = substitute(selected, truth, MEMORY)

    assert all(torch.equal(getattr(visible, name), getattr(selected, name)) for name in MEMORY)
    assert all(torch.equal(getattr(memory, name), getattr(truth, name)) for name in MEMORY)
    for candidate, replaced in ((visible, set(VISIBLE)), (memory, set(MEMORY))):
        for name, value in candidate.state_dict().items():
            if name not in replaced:
                assert torch.equal(value, before[name])

    np.testing.assert_array_equal(kernel(visible), kernel(selected))
    np.testing.assert_array_equal(kernel(memory), kernel(truth))

    # The inherited physical transfer is C (sI + diag(1/tau))^-1 C^T.
    for candidate in (selected, memory, visible):
        _, _, coupling, tau, _ = physical(candidate)
        expected = np.einsum(
            "ij,tj,kj->tik", coupling,
            np.exp(-np.geomspace(.01, 3., 32)[:, None] / tau), coupling,
        )
        np.testing.assert_array_equal(kernel(candidate), expected)


def test_memory_substitution_retains_inherited_tangent_transfer():
    selected, truth = _model(), _model(offset=0.08)
    replaced = substitute(selected, truth, MEMORY)
    q = torch.linspace(-0.04, 0.05, 10, dtype=torch.float64)[None]
    v = torch.linspace(0.03, -0.02, 10, dtype=torch.float64)[None]
    z = torch.linspace(-0.02, 0.02, 10, dtype=torch.float64)[None]
    state = torch.cat((q, v, z), dim=1)
    direction = torch.linspace(-0.3, 0.4, state.shape[1], dtype=torch.float64)[None]
    power = torch.zeros_like(q)
    equilibrium = torch.zeros_like(q)
    _, tangent = torch.autograd.functional.jvp(
        lambda value: replaced.rhs(value, power, equilibrium), state, direction,
    )
    field, response = replaced._rhs(power, equilibrium, tangent=True)(state, direction[:, :, None])
    torch.testing.assert_close(field, replaced.rhs(state, power, equilibrium), rtol=0, atol=0)
    torch.testing.assert_close(response[:, :, 0], tangent, rtol=2e-12, atol=2e-12)
