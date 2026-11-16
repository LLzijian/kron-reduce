from types import SimpleNamespace

import numpy as np

from kron_reduce.evaluation.kiki_topology_momentum import _model_blocks, _momentum_work
from kron_reduce.evaluation.machine_local_topology import make_model
from kron_reduce.evaluation.machine_local_topology_truth import parameters
from kron_reduce.evaluation.topology_momentum_work import account_path


def test_impulse_work_signs_on_synthetic_two_machine_swing():
    mass = np.array([1., 1.])
    incidence = np.array([[1.], [-1.]])
    damping = np.array([[.4, 0.], [0., .4]])
    root = np.sqrt(.4)*np.eye(2)
    coupling = np.array([[.15], [-.15]])
    blocks = (mass, incidence, np.array([2.]), damping, coupling, np.array([.5]))
    path = np.array([[.2, -.2, .3, -.3], [.21, -.21, .3, -.3]])
    result = account_path(path, np.array([1., -1.]), np.zeros(2), blocks, .01,
                          np.array([.2]), root, 4)

    for name in ("applied", "restoring", "damping", "initial_memory", "driven_memory"):
        assert result["impulses"][name].shape == (1, 2)
        assert result["work"][name].shape == (1, 2)
    assert result["impulses"]["applied"][0, 0] > 0.
    assert result["impulses"]["restoring"][0, 0] < 0.
    assert result["impulses"]["damping"][0, 0] < 0.
    assert result["impulses"]["initial_memory"][0, 0] < 0.
    assert result["impulses"]["driven_memory"][0, 0] < 0.
    assert result["work"]["applied"][0, 0] > 0.
    assert result["work"]["restoring"][0, 0] < 0.
    assert result["work"]["damping"][0, 0] < 0.
    assert result["hidden_energy"].shape == (2,)
    assert result["memory_balance_residual"].shape == (1,)


def test_memory_terms_keep_their_original_start_across_all_intervals():
    mass = np.array([1., 2.])
    incidence = np.array([[1.], [-1.]])
    damping = np.diag([.3, .6])
    root = np.diag(np.sqrt(np.diag(damping)))
    coupling = np.diag([.2, .35])
    tau = np.array([.04, .4])
    velocity = np.array([.8, -.4])
    acceleration = np.array([1.4, -.7])
    step = .01
    time = np.arange(11)*step
    angles = np.array([.2, -.1])+time[:, None]*velocity+.5*time[:, None]**2*acceleration
    path = np.column_stack((angles, velocity+time[:, None]*acceleration))
    equilibrium = np.array([.1, -.05])
    blocks = (mass, incidence, np.array([3.]), damping, coupling, tau)
    start = np.array([.2, -.4])
    projection = np.eye(2)-mass[:, None]/mass.sum()
    port = projection @ coupling
    result = account_path(path, np.array([.6, -.6]), equilibrium, blocks, step, start, root, 8)
    coarse = account_path(path, np.array([.6, -.6]), equilibrium, blocks, step, start, root, 4)
    decay = np.exp(-time[:, None]/tau)
    exact_free = np.diff(-decay, axis=0)*tau*start
    drive = velocity @ port
    ramp = acceleration @ port
    time_integral = step-np.diff(-decay, axis=0)*tau
    exact_driven = time_integral*tau*drive+(.5*np.diff(time**2)[:, None]-time_integral*tau)*tau*ramp
    np.testing.assert_allclose(result["impulses"]["initial_memory"], -exact_free @ port.T, atol=1e-15)
    np.testing.assert_allclose(result["impulses"]["driven_memory"], -exact_driven @ port.T, atol=1e-15)
    np.testing.assert_allclose(result["memory_balance_residual"], 0., atol=1e-14)
    for key in ("impulses", "work"):
        for term in result[key]:
            np.testing.assert_allclose(result[key][term], coarse[key][term], atol=1e-12)


def test_projected_constant_input_closes_momentum_and_kinetic_work():
    mass = np.array([1., 2.])
    incidence = np.array([[1.], [-1.]])
    zeros = np.zeros((2, 2))
    power = np.array([2., 1.])
    acceleration = np.array([1., -.5])
    step = .01
    time = np.arange(5)[:, None]*step
    path = np.column_stack((.5*time**2*acceleration, time*acceleration))
    blocks = (mass, incidence, np.zeros(1), zeros, zeros, np.array([.1, .2]))
    bank = account_path(path, power, np.zeros(2), blocks, step, np.zeros(2), zeros, 8)
    np.testing.assert_allclose(bank["momentum_residual"], 0., atol=1e-15)
    np.testing.assert_allclose(bank["kinetic_residual"], 0., atol=1e-15)
    np.testing.assert_allclose(bank["hidden_energy"], 0.)


def test_observer_diagnostic_keeps_origin_reset_separate(tmp_path):
    params = parameters()
    model = make_model(params, "T0M1", "cpu").eval()
    mass = np.asarray(params[0])
    state = np.zeros((15, 20))
    state[:, 0], state[:, 1] = .06, -.06*mass[0]/mass[1]
    event = {"id": "fixture", "label": "stable", "group": "fixture", "x": state,
             "mass": mass, "equilibrium": params[-1], "p0": np.zeros(10), "omega_s": 377.}
    plan = {"history_scale": np.ones(20), "memory_radius": 1.,
            "gates": dict.fromkeys(("quadrature_max", "memory_balance_max", "balance_identity_max"), 1e-7)}
    budget = SimpleNamespace(spend=lambda *_args: None)
    result = _momentum_work(tmp_path, [event], model, _model_blocks(model), plan, budget)
    assert result["qualified"] == 1
    with np.load(tmp_path/"momentum-work-intervals.npz", allow_pickle=False) as saved:
        assert saved["0_history_impulses"].shape == (10, 5, 10)
        assert saved["0_observed_future_impulses"].shape == (4, 5, 10)
        np.testing.assert_array_equal(saved["0_observed_future_observed_acceleration"], 0.)
