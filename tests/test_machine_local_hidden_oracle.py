"""Hand-built arrays verify the contrast without running the 64 scientific cases."""

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from kron_reduce.evaluation import machine_local_hidden_oracle as evaluation
from kron_reduce.models.machine_local import DiagonalMemoryModel
from kron_reduce.models.nonlinear_latent import NonlinearPassiveClosure


@pytest.fixture(autouse=True)
def forbid_scientific_rollouts(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("unit checks must not integrate the scientific population")

    monkeypatch.setattr(NonlinearPassiveClosure, "rollout", forbidden)
    monkeypatch.setattr(DiagonalMemoryModel, "rollout", forbidden)
    monkeypatch.setattr(evaluation, "make_inputs", forbidden)


def _budget():
    return SimpleNamespace(spend=lambda *_args: None, scope=lambda _name: nullcontext())


def _outputs(count, oracle):
    result = {"forecast": np.zeros((count, 801, 30))}
    if not oracle:
        result.update(history=np.zeros((count, 11, 30)), iterates=np.zeros((count, 4, 10)),
                      increments=np.zeros((count, 3, 10)), singular=np.ones((count, 3, 10)),
                      residuals=np.zeros((count, 3, 200)))
    return result


def test_frozen_roles_and_initial_state_formula_are_exact():
    mass = 1 + 0.1 * np.arange(10)
    states = evaluation.initial_states(mass)
    assert states.shape == (64, 30)
    assert {role: list(range(64))[ix] for role, ix in evaluation.ROLES.items()} == {
        "train": list(range(32)), "val": list(range(32, 48)), "test": list(range(48, 64))}
    np.testing.assert_allclose(states[:, :10] @ mass, 0, atol=1e-13)
    np.testing.assert_allclose(states[:, 10:20] @ mass, 0, atol=1e-13)
    for j in (0, 1, 22, 63):
        angle = np.arange(10) + 2*np.pi*(j+0.5)/64
        q = 0.03*np.sin(angle)
        v = 0.02*np.cos(angle)
        v[(j//2) % 10] += 16*(j % 2)
        expected = np.r_[q - q @ mass/mass.sum(), v - v @ mass/mass.sum(), 0.05*np.cos(angle)]
        np.testing.assert_allclose(states[j], expected, rtol=0, atol=4e-15)
    for ix in evaluation.ROLES.values():
        indices = np.arange(64)[ix]
        assert np.count_nonzero(indices % 2) * 2 == len(indices)


def test_only_visible_stiffness_changes_between_the_two_models():
    base = evaluation.fixture("local", backbone="sine")
    models = evaluation.make_models(base)
    matched, mismatch = models["matched"], models["mismatch"]
    assert matched.backbone_kind == mismatch.backbone_kind == "sine"
    assert matched.port_kind == mismatch.port_kind == "local"
    before, after = matched.state_dict(), mismatch.state_dict()
    assert before.keys() == after.keys()
    for key in before:
        torch.testing.assert_close(after[key], before[key] * (1.10 if key == "stiffness" else 1),
                                   rtol=0, atol=0)
    np.testing.assert_array_equal(base.edge_stiffness, np.full(45, 0.12))


def test_oracle_uses_complete_eleventh_frame_and_never_initializes_history(monkeypatch):
    paths = np.arange(2*14*30, dtype=float).reshape(2, 14, 30)
    calls = []

    def rollout(initial, power, equilibrium, count, *, max_step, spend):
        calls.append((initial.copy(), power.copy(), equilibrium.copy(), count, max_step))
        return torch.tensor(np.repeat(initial[:, None], count+1, axis=1), dtype=torch.float64)

    monkeypatch.setattr(evaluation, "predict_from_history",
                        lambda *a, **kw: pytest.fail("oracle must not solve hidden history"))
    model = SimpleNamespace(rollout=rollout)
    result = evaluation.predict(model, paths, np.ones(20), 0.00125, True, _budget().spend)
    assert set(result) == {"forecast"}
    np.testing.assert_array_equal(calls[0][0], paths[:, 10])
    np.testing.assert_array_equal(calls[0][1], np.zeros((2, 10)))
    np.testing.assert_array_equal(calls[0][2], np.zeros((2, 10)))
    assert calls[0][3:] == (800, 0.00125)
    np.testing.assert_array_equal(result["forecast"][:, 0], paths[:, 10])


def test_gn_receives_only_causal_history_and_preserves_three_update_records(monkeypatch):
    paths = np.arange(2*14*30, dtype=float).reshape(2, 14, 30)
    hscale = np.arange(1, 21, dtype=float)
    calls = []

    def history(model, q, v, power, equilibrium, scale, count, *, max_step, spend):
        calls.append((q, v, power, equilibrium, scale, count, max_step))
        return {"forecast": torch.zeros((2, 801, 30)), "history": torch.zeros((2, 11, 30)),
                "iterates": torch.arange(80).reshape(4, 2, 10),
                "gn_hidden_start": torch.zeros((3, 2, 10)), "gn_singular": torch.ones((3, 2, 10)),
                "iteration_residuals": torch.zeros((3, 2, 200))}

    monkeypatch.setattr(evaluation, "predict_from_history", history)
    result = evaluation.predict(object(), paths, hscale, 0.0025, False, _budget().spend)
    q, v, power, equilibrium, scale, count, step = calls[0]
    np.testing.assert_array_equal(q, paths[:, :11, :10])
    np.testing.assert_array_equal(v, paths[:, :11, 10:20])
    np.testing.assert_array_equal(power, np.zeros((2, 10)))
    np.testing.assert_array_equal(equilibrium, power)
    assert scale is hscale and (count, step) == (800, 0.0025)
    assert result["iterates"].shape == (2, 4, 10)
    np.testing.assert_array_equal(result["iterates"][1, 2], np.arange(50, 60))
    assert result["increments"].shape == (2, 3, 10)
    assert result["residuals"].shape == (2, 3, 200)


@pytest.mark.parametrize("oracle", [False, True])
def test_failed_batch_is_isolated_once_at_same_step_without_blocking_other_step(oracle):
    paths = np.zeros((3, 12, 30))
    paths[:, 0, 0] = np.arange(3)
    calls = []

    def predictor(_model, part, _hscale, step, requested_oracle, _spend):
        ids = tuple(part[:, 0, 0].astype(int))
        calls.append((step, ids))
        assert requested_oracle is oracle
        if step == 0.0025 and (len(part) > 1 or ids == (1,)):
            raise FloatingPointError("hand-built failed path")
        return _outputs(len(part), oracle)

    arrays, failures = evaluation.predict_batches(object(), paths, np.ones(20), oracle,
                                                  _budget(), predictor=predictor)
    assert calls == [(0.0025, (0, 1, 2)), (0.0025, (0,)), (0.0025, (1,)),
                     (0.0025, (2,)), (0.00125, (0, 1, 2))]
    assert len(failures) == 2 and failures[1]["index"] == 1
    assert np.isnan(arrays["fine_forecast"][1]).all()
    assert np.isfinite(arrays["fine_forecast"][[0, 2]]).all()
    assert np.isfinite(arrays["refined_forecast"]).all()


def test_normally_returned_failed_output_does_not_add_isolation_calls():
    calls = []

    def predictor(_model, part, _hscale, step, _oracle, _spend):
        calls.append(step)
        return {"forecast": np.full((len(part), 801, 30), np.nan)}

    arrays, failures = evaluation.predict_batches(object(), np.zeros((2, 12, 30)), np.ones(20),
                                                  True, _budget(), predictor=predictor)
    assert calls == [0.0025, 0.00125] and failures == []
    assert np.isnan(arrays["fine_forecast"]).all()


def test_batch_size_is_at_most_sixteen_with_original_event_order():
    paths = np.zeros((17, 12, 30))
    paths[:, 0, 0] = np.arange(17)
    calls = []

    def predictor(_model, part, _hscale, step, _oracle, _spend):
        calls.append((step, tuple(part[:, 0, 0].astype(int))))
        result = _outputs(len(part), True)
        result["forecast"][:, :, 0] = part[:, 0, 0][:, None]
        return result

    arrays, failures = evaluation.predict_batches(object(), paths, np.ones(20), True,
                                                  _budget(), predictor=predictor)
    assert calls == [(0.0025, tuple(range(16))), (0.00125, tuple(range(16))),
                     (0.0025, (16,)), (0.00125, (16,))]
    np.testing.assert_array_equal(arrays["fine_forecast"][:, 0, 0], np.arange(17))
    assert failures == []


def test_trajectory_scores_use_future_squared_error_and_physical_units():
    truth, forecast = np.zeros((811, 30)), np.zeros((801, 30))
    forecast[0] = 1e6
    forecast[1:, :10], forecast[1:, 10:20] = 1, 2
    scale = np.r_[np.full(10, 2), np.ones(10)]
    horizons = evaluation.trajectory_scores(forecast, truth, scale)
    assert set(horizons) == {"1", "2", "3", "5", "8"}
    for row in horizons.values():
        assert row["nmse"] == pytest.approx(2.125)
        assert row["nrmse"] == pytest.approx(np.sqrt(2.125))
        assert (row["q_rmse"], row["v_rmse"], row["q_max_abs"], row["v_max_abs"]) == (1, 2, 1, 2)
    forecast[20, 3] = np.nan
    assert evaluation.trajectory_scores(forecast, truth, scale) is None


def test_unresolved_predictions_and_numerical_failures_never_become_stable(monkeypatch):
    monkeypatch.setattr(evaluation, "class_for", lambda *_args: {"label": None, "trigger_index": None})
    paths = np.zeros((2, 811, 30))
    arrays = {key+"_forecast": np.zeros((2, 801, 30)) for key in evaluation.STEPS}
    arrays["fine_forecast"][1] = np.nan
    truth = [{"label": "stable", "trigger_index": None}, {"label": "unstable", "trigger_index": 30}]
    row = evaluation.describe(arrays, paths, truth, np.ones(20), np.ones(20), np.ones(10), True)
    stability = row["stability"]
    assert stability["accuracy"] is None and stability["accuracy_on_resolved"] is None
    assert stability["resolved"] == 0 and stability["unresolved"] == 2
    assert stability["numerical_failures"] == 1 and stability["coverage"] == 0
    assert stability["recall_on_resolved"] is None and stability["false_alarm_rate_on_resolved"] is None
    assert row["horizons"] is None and row["recovery_pass"] is False


def test_finite_unqualified_step_preserves_raw_errors_but_nulls_main_scores(monkeypatch):
    monkeypatch.setattr(evaluation, "class_for", lambda *_args: {"label": "stable", "trigger_index": None})
    paths = np.zeros((1, 811, 30))
    arrays = {key+"_forecast": np.zeros((1, 801, 30)) for key in evaluation.STEPS}
    arrays["refined_forecast"][:, 1:, 0] = 0.001
    arrays["refined_forecast"][:, 1:, 1] = -0.001
    row = evaluation.describe(arrays, paths, [{"label": "stable", "trigger_index": None}],
                              np.ones(20), np.ones(20), np.ones(10), True)
    assert row["events"][0]["trajectory"]["3"]["nmse"] == 0
    assert row["events"][0]["qualified"] is False
    assert row["horizons"] is None and row["numerical_pass"] is False
    assert row["stability"]["accuracy"] is None


def test_paired_differences_preserve_signed_mse_and_missing_cases():
    values = {"matched_gn": (4, 8), "matched_oracle": (1, 3),
              "mismatch_gn": (5, 7), "mismatch_oracle": (10, 12)}
    records = {group: {role: {"events": [{"qualified": True, "trajectory": {"3": {"nmse": value}}}
                                        for value in pair]}
                       for role in evaluation.ROLES} for group, pair in values.items()}
    result = evaluation.paired_differences(records)
    for row in result.values():
        assert row["matched_initialization"] == {"per_event": [3, 5], "mean": 4, "allocated": 2, "paired_count": 2}
        assert row["dynamics_at_true_origin"] == {"per_event": [9, 9], "mean": 9, "allocated": 2, "paired_count": 2}
        assert row["mismatch_initialization"] == {"per_event": [-5, -5], "mean": -5, "allocated": 2, "paired_count": 2}
    records["mismatch_gn"]["test"]["events"][0]["trajectory"] = None
    assert evaluation.paired_differences(records)["test"]["mismatch_initialization"] == {
        "per_event": [None, -5], "mean": None, "allocated": 2, "paired_count": 1}
    records["matched_gn"]["val"]["events"][1]["qualified"] = False
    assert evaluation.paired_differences(records)["val"]["matched_initialization"] == {
        "per_event": [3, None], "mean": None, "allocated": 2, "paired_count": 1}


@pytest.mark.parametrize("truth_pass", [False, True])
def test_truth_gate_and_full_group_role_completion_use_only_mock_paths(tmp_path, monkeypatch, truth_pass):
    paths = np.zeros((64, 811, 30))
    paths[:, :, 0] = np.arange(64)[:, None]
    coarse = paths.copy()
    if not truth_pass:
        coarse[0, 30, 0] += 0.01
    monkeypatch.setattr(evaluation, "make_inputs", lambda *_args: (paths, coarse, np.ones(20), np.ones(20)))
    monkeypatch.setattr(evaluation, "class_for", lambda p, *_args: {
        "label": "unstable" if int(p[0, 0]) % 2 else "stable",
        "trigger_index": 20 if int(p[0, 0]) % 2 else None})
    models = {name: SimpleNamespace(name=name, requires_grad_=lambda _flag: None,
                                   state_dict=lambda: {"dummy": torch.zeros(1)})
              for name in ("matched", "mismatch")}
    monkeypatch.setattr(evaluation, "make_models", lambda _base: models)
    calls = []

    def batches(model, part, _hscale, oracle, _budget):
        assert truth_pass
        assert (tmp_path / "frozen_inputs_and_models.json").is_file()
        calls.append((model.name, oracle, len(part), int(part[0, 0, 0])))
        return {"marker": np.zeros(1)}, []

    def describe(_arrays, part, _classes, _scale, _hscale, _mass, _oracle):
        return {"events": [{"qualified": False, "trajectory": {"3": {"nmse": 2}}} for _ in part], "horizons": {},
                "stability": {"accuracy": None}, "numerical_pass": False, "recovery_pass": False}

    monkeypatch.setattr(evaluation, "predict_batches", batches)
    monkeypatch.setattr(evaluation, "describe", describe)
    summary = evaluation.validate(tmp_path, tmp_path, _budget(), {})
    assert summary["truth_pass"] is truth_pass and summary["scientific_pass"] is False
    assert summary["evaluation_complete"] is truth_pass
    if truth_pass:
        assert calls == [(name, method == "oracle", count, start)
                         for name, method in (group.split("_") for group in evaluation.GROUPS)
                         for count, start in ((32, 0), (16, 32), (16, 48))]
        assert set(summary["records"]) == set(evaluation.GROUPS)
        assert all(set(rows) == set(evaluation.ROLES) for rows in summary["records"].values())
    else:
        assert calls == [] and summary["records"] == {}


def test_static_operation_counts_match_frozen_clock_without_integrating():
    clock = np.arange(811)*0.01
    truth = [4*sum(max(1, int(np.ceil(h/step-1e-12))) for h in np.diff(clock))
             for step in (0.00125, 0.000625)]
    assert truth == [25924, 52228]
    assert 64*sum(truth) == 5001728
    batch_count, substep_sum = 2*4, 4+8
    gn_rhs = batch_count*4*840*substep_sum
    oracle_rhs = batch_count*4*800*substep_sum
    tangent_rhs = batch_count*4*30*substep_sum
    assert (gn_rhs, oracle_rhs, tangent_rhs) == (322560, 307200, 11520)
    assert 2*64*2*3 == 768
    assert 128+batch_count*2*5+batch_count*2 == 224
    assert 224+2*64*2*5+2*64*2 == 1760
    assert gn_rhs+oracle_rhs+(2*64)*4*(840+800)*substep_sum == 10705920
