"""Check spatial dynamic coupling momentum diagnostics."""
from __future__ import annotations

import numpy as np
from scipy.linalg import helmert

from kron_reduce.evaluation.machine_local_topology_truth import _incidence, parameters
from kron_reduce.evaluation.topology_dynamic_momentum import coupling_impulse, coupling_residuals
from kron_reduce.evaluation.topology_momentum import momentum_system, project_residual
from kron_reduce.evaluation.topology_spatial_momentum import _force_directions, spatial_damping_root


def _fixture(frames=16):
    mass, stiffness, damping, coupling, tau, equilibrium = parameters()
    incidence = _incidence()
    t = np.arange(frames, dtype=np.float64) * 0.01
    n = len(mass)
    q = np.stack([0.07*np.sin(2.1*t + .31*i) for i in range(n)], axis=1)
    q -= (q @ mass / mass.sum())[:, None]
    v = np.stack([0.04*np.cos(1.7*t + .23*i) for i in range(n)], axis=1)
    v -= (v @ mass / mass.sum())[:, None]
    power = 0.03*np.sin(np.arange(n)*.41)
    power -= power.mean()
    equilibrium = np.linspace(-.08, .09, n)
    blocks = (mass, incidence, stiffness, damping, np.diag(coupling), tau)
    return np.column_stack((q, v)), power, equilibrium, blocks


def _counter():
    counts = {}

    def spend(name, amount=1):
        counts[name] = counts.get(name, 0) + amount

    return counts, spend


def test_spatial_impulse_zero_state_has_zero_log_sum_and_unit_multipliers():
    mass, stiffness, damping, coupling, tau, equilibrium = parameters()
    incidence = _incidence()
    blocks = (mass, incidence, stiffness, damping, np.diag(coupling), tau)
    path = np.zeros((12, 2*len(mass)))
    counts, spend = _counter()
    result = coupling_impulse(
        path, np.zeros_like(equilibrium), blocks, .01,
        rate=.7, limit=np.log(2.)/np.sqrt(2.), substeps=4, spend=spend, spatial=True,
    )
    np.testing.assert_array_equal(result["c"], 0.)
    np.testing.assert_array_equal(result["interval_impulses"], 0.)
    np.testing.assert_array_equal(result["edge_log_sum"], 0.)
    assert result["multiplier_range"] == [1., 1.]
    assert result["edge_log_sum_max_abs"] == 0.
    assert counts == {"dynamic_intervals": 11, "dynamic_rhs": 4*4*11}


def test_spatial_impulse_is_bounded_and_fine_step_refines_coarse_step():
    path, _power, equilibrium, blocks = _fixture()
    common = {"rate": .7, "limit": .3, "spend": lambda *_args: None, "spatial": True}
    coarse = coupling_impulse(path, equilibrium, blocks, .01, substeps=2, **common)
    fine = coupling_impulse(path, equilibrium, blocks, .01, substeps=4, **common)
    edge_basis = np.abs(blocks[1]).T @ helmert(len(blocks[0]), full=False).T
    def geometry(result):
        radial = result["c"]/np.sqrt(1.+np.sum(result["c"]**2, axis=1, keepdims=True)/(.3**2))
        log_edge = radial @ edge_basis.T
        return log_edge, np.exp(log_edge)
    coarse_log, coarse_multiplier = geometry(coarse)
    fine_log, fine_multiplier = geometry(fine)
    combined_multiplier = np.concatenate((coarse_multiplier.ravel(), fine_multiplier.ravel()))
    assert np.max(np.abs(coarse_log.sum(axis=1))) < 1e-12
    assert np.max(np.abs(fine_log.sum(axis=1))) < 1e-12
    assert combined_multiplier.min() >= 0.5 - 1e-12
    assert combined_multiplier.max() <= 2.0 + 1e-12
    assert coarse["edge_log_sum_max_abs"] < 1e-12
    assert fine["edge_log_sum_max_abs"] < 1e-12
    assert np.linalg.norm(fine["c"][-1]) > 0.
    assert np.linalg.norm(fine["interval_impulses"]) > 0.
    assert np.linalg.norm(fine["interval_impulses"]-coarse["interval_impulses"]) < 2e-5
    assert np.linalg.norm(fine["c"]-coarse["c"]) < 2e-7


def test_coupling_residual_uses_signed_impulse_and_same_fixed_origins_for_both_anchors():
    path, power, equilibrium, blocks = _fixture(frames=17)
    mass = blocks[0]
    starts = np.zeros((2, 2, len(mass)))
    starts[0, 0] = np.linspace(-.02, .03, len(mass))
    starts[0, 1] = -starts[0, 0]
    starts[1, 0] = np.linspace(.01, -.025, len(mass))
    starts[1, 1] = -starts[1, 0]
    root = spatial_damping_root(blocks)
    counts, spend = _counter()
    result = coupling_residuals(
        path, power, equilibrium, blocks, .01, spend, starts,
        root=root, rate=.7, limit=.3, spatial=True,
    )
    assert result["future_intervals"] == len(path)-1-10
    assert np.asarray(result["coupling_origin"]).shape == (9,)
    assert result["spatial_geometry_qualified"] is True
    assert result["gaps"]["coarse_zero_sum"] <= 1e-12
    assert result["gaps"]["fine_zero_sum"] <= 1e-12
    assert result["combined_multiplier_range"][0] >= .5-1e-12
    assert result["combined_multiplier_range"][1] <= 2.+1e-12
    assert set(result["sse"]) == {"constant", "spatial"}
    assert all(set(value) == {"0.0025", "0.00125", "free"} for value in result["sse"].values())
    assert counts["dynamic_intervals"] == 2*(len(path)-1)
    assert counts["weak_intervals"] == len(path)-1-10
    assert counts["svd"] == 1

    future = path[10:]
    operator, rhs, _ = momentum_system(future, power, equilibrium, blocks, .01)
    projection = project_residual(operator, rhs)
    impulse = coupling_impulse(
        path, equilibrium, blocks, .01, rate=.7, limit=.3, substeps=4,
        spend=lambda *_args: None, spatial=True,
    )["interval_impulses"][10:]
    direction = (impulse / (mass*.01*np.sqrt(len(future)-1))[None, :]).ravel()
    free_direction = direction-projection["basis"] @ (projection["basis"].T @ direction)
    np.testing.assert_allclose(list(result["direction_sse"].values()),
                               [direction @ direction, free_direction @ free_direction])
    damping = _force_directions(future, blocks, .01, 4, root)[:, :2].sum(axis=1)
    for anchor_index, (anchor, base) in enumerate((("constant", rhs), ("spatial", rhs-damping))):
        for i, start_name in enumerate(("0.0025", "0.00125")):
            baseline = base-operator @ starts[anchor_index, i]
            expected = baseline-direction
            actual_pair = result["sse"][anchor][start_name]
            np.testing.assert_allclose(actual_pair[0], baseline@baseline, rtol=2e-13, atol=1e-15)
            np.testing.assert_allclose(actual_pair[1], expected@expected, rtol=2e-13, atol=1e-15)
    assert projection["rank"] == result["rank"]
