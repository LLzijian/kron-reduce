import numpy as np
import pytest
import torch

from kron_reduce.data.history_contract import CausalStateHistory
from kron_reduce.models.machine_local import DiagonalMemoryModel
from kron_reduce.models.machine_local_forecast import FrozenMachineLocalForecaster
from kron_reduce.models.nonlinear_history import predict_from_history
from kron_reduce.models.nonlinear_latent import NonlinearPassiveClosure


def _forecaster():
    mass = np.array([1.0, 1.2, 0.8])
    closure = NonlinearPassiveClosure(
        mass, np.array([1.5, 1.2, 1.8]),
        0.02*np.array([[2., -1., -1.], [-1., 2., -1.], [-1., -1., 2.]]),
        np.eye(3)-np.ones((3, 3))/3, np.zeros((3, 3)), np.eye(3), "sine",
    )
    model = DiagonalMemoryModel(
        closure, port_kind="local", port=np.array([.3, .4, .5]),
        tau=np.array([.08, .2, .7]), encoder=np.zeros((3, 24)), feature_scale=np.ones(24),
    )
    return FrozenMachineLocalForecaster(model, np.ones(6), max_step=.005)


def test_public_predict_matches_registered_gauss_newton_and_resets_visible_origin():
    forecaster = _forecaster()
    q = np.array([[0., 0., 0.], [.01, -.005, -.002], [.02, -.01, -.004],
                  [.03, -.015, -.006], [.04, -.02, -.008], [.05, -.025, -.01],
                  [.06, -.03, -.012], [.07, -.035, -.014], [.08, -.04, -.016],
                  [.09, -.045, -.018], [.1, -.05, -.02]])
    v = np.gradient(q, .01, axis=0)
    # Keep both observed vectors in the center-of-inertia subspace.
    mass = forecaster.model.mass.detach().numpy()
    q -= (q @ mass / mass.sum())[:, None]
    v -= (v @ mass / mass.sum())[:, None]
    power = np.array([.12, -.04, -.08])
    equilibrium = np.array([.06, -.025, -.01])
    history = CausalStateHistory(-np.arange(10, -1, -1)*.01, q, v, power)
    actual = forecaster.predict(history, equilibrium, 3)
    with torch.no_grad():
        expected = predict_from_history(
            forecaster.model, torch.tensor(q[None]), torch.tensor(v[None]),
            torch.tensor(power[None]), torch.tensor(equilibrium[None]), forecaster.history_scale,
            3, max_step=.005, spend=lambda *_: None,
        )["forecast"][0].numpy()
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(actual[0, :6], np.r_[q[-1], v[-1]])


def test_checkpoint_freezes_prediction_contract_and_rejects_other_initializer(tmp_path):
    forecaster = _forecaster()
    state = forecaster.state_dict()
    assert state["_extra_state"]["gauss_newton_updates"] == 3
    assert state["_extra_state"]["initializer"] == "nonlinear_history_gauss_newton_v1"
    restored = _forecaster()
    restored.load_state_dict(state)
    path = tmp_path / "frozen.pt"
    forecaster.save_checkpoint(path)
    standalone = FrozenMachineLocalForecaster.load_checkpoint(path)
    assert standalone.get_extra_state() == forecaster.get_extra_state()
    bad = dict(state)
    bad["_extra_state"] = dict(state["_extra_state"], gauss_newton_updates=2)
    with pytest.raises(ValueError, match="forecast contract"):
        _forecaster().load_state_dict(bad)
