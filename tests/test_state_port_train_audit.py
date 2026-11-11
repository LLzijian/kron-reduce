import json
from contextlib import contextmanager

import numpy as np
import pytest
from scipy.linalg import helmert

from kron_reduce.evaluation import state_port_train_audit as training_audit
from kron_reduce.models.state_port import StatePortClosure


def toy_model():
    n, r = 3, 2
    basis = helmert(n, full=False).T
    return StatePortClosure(
        mass=np.array([1.0, 1.3, 1.7]),
        edge_stiffness=np.array([0.8, 1.1, 0.6]),
        damping=0.15 * basis @ basis.T,
        coupling=0.2 * basis[:, :r],
        latent_skew=np.array([[0.0, 0.08], [-0.08, 0.0]]),
        latent_dissipation=0.25 * np.eye(r),
        delta=np.zeros((n - 1, r)),
        port_kind="state",
    )


def toy_batch(model):
    q = np.array([[0.12, -0.04, 0.0], [-0.08, 0.03, 0.02]])
    v = np.array([[0.02, -0.01, 0.0], [-0.01, 0.03, -0.02]])
    q -= (q @ model.mass)[:, None] / model.mass.sum()
    v -= (v @ model.mass)[:, None] / model.mass.sum()
    initial = np.c_[q, v, [[0.3, -0.2], [-0.1, 0.25]]]
    power = np.array([[0.06, -0.02, -0.04], [-0.03, 0.05, -0.02]])
    equilibrium = np.array([[0.04, -0.02, 0.01], [-0.03, 0.02, 0.01]])
    inputs = {"constant_power": power, "equilibrium": equilibrium, "target_scale": np.ones(6)}
    return initial, inputs


class ImmediateOperations:
    def __init__(self, caps=None):
        self.records = []
        self.caps = {} if caps is None else caps

    @contextmanager
    def scope(self, _name):
        yield

    def call(self, category, function, *, tag=None):
        value = function()
        self.records.append({"category": category, "tag": tag, "value": value})
        return value


def test_calls_records_attempted_event_batch_and_failed_numpy_state():
    calls = training_audit.Calls({"torch_batch_rhs": 1, "torch_event_rhs": 2})
    state = np.arange(16, dtype=float).reshape(2, 8)
    calls("attempted", batch_size=2, state=state)
    state[:] = -1
    failure = FloatingPointError("toy failure")
    calls("failed", batch_size=2, state=state, error=failure)
    assert calls.counts["torch_batch_rhs"] == 1
    assert calls.counts["torch_event_rhs"] == 2
    assert not calls.returned
    assert calls.last_rhs["backend"] == "torch"
    np.testing.assert_array_equal(calls.last_rhs["state"], np.arange(16).reshape(2, 8))


def test_fit_scope_rejects_overrun_before_call_and_underrun_on_close():
    over_ops = ImmediateOperations({"scope": {"logical_limits": {"probe": 1}}})
    over_calls = training_audit.Calls({"probe": 2})
    with pytest.raises(RuntimeError, match="scope budget exhausted: probe"), training_audit._scope(
        over_ops, over_calls, "scope"
    ):
        over_calls.add("probe")
        over_calls.add("probe")
    assert over_calls.counts["probe"] == 1

    under_ops = ImmediateOperations({"scope": {"logical_limits": {"probe": 2}}})
    under_calls = training_audit.Calls({"probe": 2})
    with pytest.raises(RuntimeError, match="scope calls differ from frozen allocation: scope"), training_audit._scope(
        under_ops, under_calls, "scope"
    ):
        under_calls.add("probe")
    assert under_calls.counts["probe"] == 1


def test_torch_operation_records_strict_json_metadata():
    torch = pytest.importorskip("torch")
    ops = ImmediateOperations()
    value = training_audit._torch_operation(
        ops, "probe", lambda: torch.zeros((2, 3), dtype=torch.float64), "toy"
    )
    assert tuple(value.shape) == (2, 3)
    record = ops.records[0]
    encoded = json.dumps({key: record[key] for key in ("category", "tag", "value")}, allow_nan=False)
    assert json.loads(encoded)["value"] == {"shape": [2, 3], "dtype": "torch.float64"}


def test_fit_updates_only_theta_for_forty_steps_and_saves_state(tmp_path):
    pytest.importorskip("torch")
    model = toy_model()
    initial, inputs = toy_batch(model)
    counts = np.array([2, 3], dtype=np.int64)
    target = np.zeros((2, 4, 6), dtype=np.float64)
    weights = np.array([0.4, 0.6])
    saved_parameters = {name: np.array(getattr(model, name), copy=True) for name in (
        "mass", "edge_stiffness", "damping", "coupling", "latent_skew", "latent_dissipation"
    )}
    limits = {
        "fit_attempts": 1,
        "training_forward": 40, "torch_batch_rhs": 480, "torch_event_rhs": 800,
        "backward": 40, "gradient_clip": 40, "optimizer_update": 40,
    }
    calls = training_audit.Calls(limits)
    fit_limits = {"training_forward": 1, "backward": 1, "gradient_clip": 1,
                  "optimizer_update": 1, "torch_batch_rhs": 12, "torch_event_rhs": 20}
    scope_caps = {f"fit-start-{tmp_path.name}": {"logical_limits": {"fit_attempts": 1}}}
    scope_caps.update({f"fit-{tmp_path.name}-{update:02d}": {"logical_limits": fit_limits}
                       for update in range(1, 41)})
    ops = ImmediateOperations(scope_caps)

    delta = training_audit._fit(
        model, initial, inputs, counts, target, weights, calls, ops, tmp_path
    )
    assert delta.shape == model.delta.shape == (2, 2)
    assert calls.counts == calls.returned == limits
    for name, saved in saved_parameters.items():
        np.testing.assert_array_equal(getattr(model, name), saved)
    assert (tmp_path / "update-40.json").is_file()
    assert (tmp_path / "update-40.npz").is_file()
    update = json.loads((tmp_path / "update-40.json").read_text(encoding="utf-8"))
    arrays = np.load(tmp_path / "update-40.npz", allow_pickle=False)
    theta = arrays[update["theta_after"]["array"]]
    assert theta.shape == (2, 2) and np.isfinite(theta).all()
    assert np.linalg.norm(theta) > 0
    final_metadata = json.loads((tmp_path / "final.json").read_text(encoding="utf-8"))
    final_arrays = np.load(tmp_path / "final.npz", allow_pickle=False)
    np.testing.assert_array_equal(final_arrays[final_metadata["delta"]["array"]], delta)


def test_write_json_serialization_failure_does_not_create_partial_file(tmp_path):
    output = tmp_path / "strict.json"
    with pytest.raises(TypeError):
        training_audit.write_json(output, {"numpy_bool": np.bool_(True)})
    assert not output.exists()
