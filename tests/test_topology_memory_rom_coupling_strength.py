"""Check equal starting coupling amplitudes and the short-time error calculation."""
import numpy as np
import pytest
import torch

from kron_reduce.evaluation.machine_local_topology_truth import parameters
from kron_reduce.evaluation.topology_memory_rom_coupling_strength import (
    _hidden_errors,
    _match_initial_coupling,
    _momentum_error,
)
from kron_reduce.models.nonlinear_latent import NonlinearPassiveClosure
from kron_reduce.models.topology_memory_rom import TopologyMemoryROM


def test_hidden_errors_separate_coupling_groups_and_steps():
    truth = np.zeros((1, 11, 40))
    truth[..., 30] = .04
    histories = np.repeat(truth[None, ...], 2, axis=0)
    histories[0, 0, 0, 30] += .004
    histories[1, 0, -1, 31] += .02
    errors = _hidden_errors(histories, truth, common=True, branch=True)
    np.testing.assert_allclose(errors["overall"][:, 0], [[.1, 0.], [0., 0.]])
    np.testing.assert_allclose(errors["branch"][:, 0], [[0., 0.], [0., .2]])
    np.testing.assert_array_equal(errors["memory"], 0.)


def _paired_cases():
    count, nodes = 48, 10
    pair = np.repeat(np.arange(count // 2), 2)
    roles = np.array([("train", "val", "test")[i % 3] for i in pair])
    q = np.zeros((count, nodes), dtype=np.float64)
    v = np.zeros_like(q)
    c = np.zeros((count, 1), dtype=np.float64)
    xi = np.tile(np.linspace(-1., 1., nodes - 1), (count, 1))
    return {"q": q, "v": v, "c": c, "xi": xi,
            "roles": roles.copy(), "groups": [f"g{i:02d}" for i in pair]}


def _model(*, common, spatial):
    mass, stiffness, damping, port, tau, _ = parameters()
    empty = np.zeros((0, 0), dtype=np.float64)
    backbone = NonlinearPassiveClosure(mass, stiffness, damping,
                                       np.zeros((10, 0)), empty, empty)
    return TopologyMemoryROM(backbone, port=port, tau=tau,
                             common=common, spatial=spatial)


def _log_edge_multipliers(model, coordinates):
    n = model.n_nodes
    base = torch.zeros((len(coordinates), 3 * n + model.dynamic_dimension),
                       dtype=model.mass.dtype)
    base[:, 3 * n:] = torch.as_tensor(coordinates, dtype=base.dtype)
    ratio = model.edge_stiffness(base) / model.effective_stiffness
    return torch.log(ratio).detach().cpu().numpy()


def test_match_initial_coupling_pairs_equalize_log_strength_and_preserve_layout():
    cases = _paired_cases()
    roles_before = cases["roles"].copy()
    groups_before = cases["groups"].copy()
    mass = parameters()[0]

    assert _match_initial_coupling(cases, mass, target_rms=.08,
                                   angle_spans=(.45, .65)) is None

    assert np.array_equal(cases["roles"], roles_before)
    assert cases["groups"] == groups_before
    assert len(set(cases["groups"])) == 24
    for start in range(0, 48, 2):
        assert cases["groups"][start] == cases["groups"][start + 1]
        assert cases["roles"][start] == cases["roles"][start + 1]
        np.testing.assert_array_equal(cases["q"][start], cases["q"][start + 1])
        np.testing.assert_array_equal(cases["v"][start], cases["v"][start + 1])

    q = cases["q"]
    np.testing.assert_allclose(q @ mass, 0., atol=2e-14, rtol=0.)
    spans = np.ptp(q, axis=1)
    assert np.all(np.isfinite(spans))
    np.testing.assert_allclose(spans, np.repeat(np.linspace(.45, .65, 24), 2), atol=2e-12, rtol=0.)

    common_model = _model(common=True, spatial=False)
    spatial_model = _model(common=False, spatial=True)
    common_logs = _log_edge_multipliers(common_model, cases["c"])
    spatial_logs = _log_edge_multipliers(spatial_model, cases["xi"])
    common_rms = float(np.sqrt(np.mean(common_logs ** 2)))
    spatial_rms = float(np.sqrt(np.mean(spatial_logs ** 2)))
    assert common_rms > 0.
    assert spatial_rms > 0.
    assert common_rms == pytest.approx(spatial_rms, rel=2e-10, abs=2e-12)


def test_momentum_error_is_zero_for_identical_increment_paths():
    target = np.zeros((3, 21, 20), dtype=np.float64)
    forecast = target.copy()
    # Keep identical nonzero origins to exercise subtraction of each path's
    # own origin before forming momentum increments.
    target[:, 0, 10:20] = .125
    forecast[:, 0, 10:20] = .125
    result = _momentum_error(forecast, target, np.linspace(1., 2., 10),
                             np.ones(10))
    np.testing.assert_allclose(result, np.zeros(3), atol=0., rtol=0.)


def test_momentum_error_normalizes_full_twenty_increment_window():
    target = np.zeros((2, 21, 20), dtype=np.float64)
    forecast = target.copy()
    forecast[:, 1:, 10:20] = 1.
    result = _momentum_error(forecast, target, np.ones(10), np.ones(10))
    np.testing.assert_allclose(result, np.ones(2), atol=1e-14, rtol=0.)


def test_momentum_error_marks_nonfinite_event_as_nan():
    target = np.zeros((2, 21, 20), dtype=np.float64)
    forecast = target.copy()
    forecast[1, 7, 13] = np.inf
    result = _momentum_error(forecast, target, np.ones(10), np.ones(10))
    assert result.shape == (2,)
    assert result[0] == 0.
    assert np.isnan(result[1])
