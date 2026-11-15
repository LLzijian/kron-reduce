import math

import numpy as np

from kron_reduce.evaluation import topology_dynamic_momentum as dynamic_momentum
from kron_reduce.evaluation.topology_momentum import momentum_system
from kron_reduce.evaluation.topology_spatial_momentum import spatial_damping_root


def _blocks():
    mass = np.array([2.0, 3.0], dtype=np.float64)
    incidence = np.array([[1.0], [-1.0]], dtype=np.float64)
    stiffness = np.array([4.0], dtype=np.float64)
    damping = np.array([[0.6, -0.6], [-0.6, 0.6]], dtype=np.float64)
    coupling = np.array([[0.22, 0.05], [-0.14, 0.17]], dtype=np.float64)
    tau = np.array([0.3, 0.4], dtype=np.float64)
    return mass, incidence, stiffness, damping, coupling, tau


def _fixed_q_path(blocks, frames=2):
    mass = blocks[0]
    # This relative angle already satisfies the mass-weighted COI constraint.
    q = np.tile(np.array([[1.2, -0.8]], dtype=np.float64), (frames, 1))
    velocity = np.tile(np.array([[0.15, -0.10]], dtype=np.float64), (frames, 1))
    assert np.allclose(q @ mass, 0.0)
    return np.concatenate((q, velocity), axis=1)


def _spend_noop(*_args):
    pass


def _ode_terms(c, potential, *, rate, kappa, limit):
    mobility = rate / kappa
    saturated = math.tanh(c / limit)
    multiplier = math.exp(2.0 * limit * saturated)
    dc = -rate * c - mobility * 2.0 * (1.0 - saturated**2) * multiplier * potential
    return dc, multiplier - 1.0


def _rk4_fixed_q_impulse(path, equilibrium, blocks, *, rate, limit, duration, steps):
    mass, incidence, stiffness = blocks[:3]
    n = len(mass)
    projection = np.eye(n) - mass[:, None] * np.ones((1, n)) / mass.sum()
    edge = path[0, :n] @ incidence
    reference = equilibrium @ incidence
    shifted = np.sin(edge + reference) - np.sin(reference)
    base_force = -(shifted * stiffness) @ (projection @ incidence).T
    edge_potential = np.cos(reference) - np.cos(edge + reference) - edge * np.sin(reference)
    potential = float(np.dot(stiffness, edge_potential))
    kappa = float(stiffness.sum())
    y = np.zeros(n + 1, dtype=np.float64)  # c followed by the raw projected impulse
    dt = duration / steps

    def rhs(state):
        dc, change = _ode_terms(state[0], potential, rate=rate, kappa=kappa, limit=limit)
        return np.concatenate(([dc], change * base_force))

    for _ in range(steps):
        k1 = rhs(y)
        k2 = rhs(y + 0.5 * dt * k1)
        k3 = rhs(y + 0.5 * dt * k2)
        k4 = rhs(y + dt * k3)
        y += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return y[0], y[1:]


def test_zero_rate_is_exact_frozen_stiffness_and_nonzero_rate_has_projected_sign():
    blocks = _blocks()
    path = _fixed_q_path(blocks)
    equilibrium = np.array([0.04, -0.02], dtype=np.float64)
    limit = 0.31

    zero = dynamic_momentum.coupling_impulse(
        path, equilibrium, blocks, 0.01, rate=0.0, limit=limit, substeps=4, spend=_spend_noop
    )
    np.testing.assert_array_equal(zero["c"], np.zeros(len(path)))
    np.testing.assert_array_equal(zero["interval_impulses"], np.zeros((len(path) - 1, len(blocks[0]))))
    np.testing.assert_array_equal(np.exp(2.0 * limit * np.tanh(zero["c"] / limit)), np.ones(len(path)))

    rate = 2.0
    changed = dynamic_momentum.coupling_impulse(
        path, equilibrium, blocks, 0.01, rate=rate, limit=limit, substeps=4, spend=_spend_noop
    )
    projection = np.eye(2) - blocks[0][:, None] * np.ones((1, 2)) / blocks[0].sum()
    projected_edge = (projection @ blocks[1])[:, 0]
    assert changed["c"][0] == 0.0 and changed["c"][1] < 0.0
    assert changed["interval_impulses"][0] @ projected_edge > 0.0


def test_fixed_q_coordinate_rk4_refines_and_raw_impulse_has_physical_units():
    blocks = _blocks()
    path = _fixed_q_path(blocks)
    equilibrium = np.array([0.04, -0.02], dtype=np.float64)
    rate, limit, step = 2.0, 0.31, 0.01
    outputs = {
        substeps: dynamic_momentum.coupling_impulse(
            path, equilibrium, blocks, step, rate=rate, limit=limit,
            substeps=substeps, spend=_spend_noop,
        )
        for substeps in (2, 4)
    }
    reference_c, reference_impulse = _rk4_fixed_q_impulse(
        path, equilibrium, blocks, rate=rate, limit=limit, duration=step, steps=8192
    )
    errors = {n: abs(outputs[n]["c"][1] - reference_c) for n in (2, 4)}
    assert errors[2] > errors[4] > 0.0
    assert errors[2] / errors[4] > 8.0
    np.testing.assert_allclose(
        outputs[4]["interval_impulses"][0], reference_impulse, rtol=1e-6, atol=1e-9
    )


def test_c_coordinate_starts_at_zero_and_is_not_reset_between_history_intervals():
    blocks = _blocks()
    path = _fixed_q_path(blocks, frames=3)
    equilibrium = np.array([0.04, -0.02], dtype=np.float64)
    result = dynamic_momentum.coupling_impulse(
        path, equilibrium, blocks, 0.01, rate=1.4, limit=0.31,
        substeps=4, spend=_spend_noop,
    )
    assert result["c"][0] == 0.0
    assert result["c"][1] != 0.0
    assert result["c"][2] != result["c"][1]

    expected_c, _ = _rk4_fixed_q_impulse(
        path, equilibrium, blocks, rate=1.4, limit=0.31, duration=0.02, steps=8192
    )
    np.testing.assert_allclose(result["c"][-1], expected_c, rtol=2e-8, atol=2e-11)


def test_future_residual_uses_raw_impulse_units_and_full_history_c_origin():
    blocks = _blocks()
    nframes = 13
    mass = blocks[0]
    q0 = np.array([0.7, -7.0 / 15.0], dtype=np.float64)
    q_step = np.array([0.002, -0.004 / 3.0], dtype=np.float64)
    q = np.stack([q0 + i * q_step for i in range(nframes)])
    velocity = np.stack([np.array([0.12 + .001*i, -0.08 - .001*i]) for i in range(nframes)])
    path = np.concatenate((q, velocity), axis=1)
    equilibrium = np.array([0.04, -0.02], dtype=np.float64)
    power = np.array([0.015, -0.015], dtype=np.float64)
    starts = np.array([[[0.03, -0.02], [0.01, 0.015]], [[-0.02, 0.025], [0.04, -0.01]]], dtype=np.float64)
    root = spatial_damping_root(blocks)
    rate, limit, step = 1.1, 0.31, 0.01

    full = dynamic_momentum.coupling_impulse(
        path, equilibrium, blocks, step, rate=rate, limit=limit, substeps=4, spend=_spend_noop
    )
    scored = dynamic_momentum.coupling_residuals(
        path, power, equilibrium, blocks, step, _spend_noop, starts,
        root=root, rate=rate, limit=limit,
    )
    assert abs(full["c"][10]) > 1e-10
    np.testing.assert_allclose(scored["coupling_origin"], full["c"][10], rtol=0.0, atol=0.0)

    future = path[10:]
    operator, rhs, _ = momentum_system(future, power, equilibrium, blocks, step)
    intervals = len(future) - 1
    normalization = 1.0 / (mass * step * np.sqrt(intervals))
    raw_future = full["interval_impulses"][10:]
    direction = (raw_future * normalization[None, :]).ravel()
    baseline = rhs - operator @ starts[0, 0]
    expected_base = float(baseline @ baseline)
    expected_changed = float((baseline - direction) @ (baseline - direction))
    np.testing.assert_allclose(scored["sse"]["constant"]["0.0025"],
                               [expected_base, expected_changed], rtol=2e-12, atol=2e-12)
