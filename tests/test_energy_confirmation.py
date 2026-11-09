from __future__ import annotations

import json
from collections import Counter
from contextlib import contextmanager

import numpy as np
import pytest

from kron_reduce.evaluation import energy_confirmation as confirmation


def test_observation_is_eleven_point_coi_projected_periodic_noise():
    truth = np.zeros((11, 22))
    mass = np.arange(1.0, 11.0)
    result = confirmation.observation(truth, mass, "periodic")
    noise = result["noise"]
    a = np.array([1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1])
    b = np.array([0, 1, 1, 0, -1, -1, 0, 1, 1, 0, -1])

    assert result["observed"].shape == (11, 20)
    assert noise.shape == (11, 20)
    np.testing.assert_allclose(mass @ result["q_direction"], 0.0, atol=1e-14)
    np.testing.assert_allclose(mass @ result["v_direction"], 0.0, atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(result["q_direction"]), 1.0, atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(result["v_direction"]), 1.0, atol=1e-14)
    np.testing.assert_allclose(noise[:, :10], 2e-5 * a[:, None] * result["q_direction"])
    np.testing.assert_allclose(noise[:, 10:], 7e-5 * b[:, None] * result["v_direction"])
    np.testing.assert_array_equal(result["observed"], truth[:, :20] + noise)


def test_observation_ramp_has_registered_endpoints_and_constant_q_offset():
    truth = np.zeros((11, 22))
    mass = np.linspace(0.5, 2.0, 10)
    result = confirmation.observation(truth, mass, "ramp")
    noise = result["noise"]

    constant_q = np.repeat((2e-5 * result["v_direction"])[None, :], 11, axis=0)
    np.testing.assert_allclose(noise[:, :10], constant_q)
    ramp = (np.arange(11) - 5) / 5
    np.testing.assert_allclose(noise[:, 10:], 7e-5 * ramp[:, None] * result["q_direction"])
    np.testing.assert_allclose(noise[0, 10:], -7e-5 * result["q_direction"])
    np.testing.assert_allclose(noise[-1, 10:], 7e-5 * result["q_direction"])
    np.testing.assert_allclose(noise[0, :10], noise[-1, :10])


def test_observation_clean_is_exact_and_invalid_mode_or_shape_is_rejected():
    truth = np.arange(11 * 22, dtype=float).reshape(11, 22)
    clean = confirmation.observation(truth, np.ones(10), "clean")
    assert clean["observed"].shape == (11, 20)
    assert not np.any(clean["noise"])
    np.testing.assert_array_equal(clean["observed"], truth[:, :20])

    with pytest.raises(ValueError, match="unregistered observation mode"):
        confirmation.observation(truth, np.ones(10), "random")
    with pytest.raises(ValueError, match="eleven full states"):
        confirmation.observation(np.zeros((10, 22)), np.ones(10), "clean")


def _metric_fixture(*, noise_mode="clean", z=None, z_t=None):
    observed = np.arange(11 * 20, dtype=float).reshape(11, 20)
    z = np.array([0.0, 0.0]) if z is None else np.asarray(z, dtype=float)
    z_t = np.array([0.0, 0.0]) if z_t is None else np.asarray(z_t, dtype=float)
    history = np.zeros((11, 22))
    history[0] = np.r_[observed[0], z]
    history[-1, 20:] = z_t
    reset = np.r_[observed[-1], z_t]
    future = np.repeat(reset[None, :], 801, axis=0)
    truth_history = history.copy()
    truth_future = future.copy()
    context = {
        "noise_mode": noise_mode,
        "observed": observed,
        "truth_history": truth_history,
        "truth_future": truth_future,
        "matrix": np.eye(2),
        "right": np.zeros(2),
    }
    path = {
        "history": history,
        "reset": reset,
        "future": future,
        "reset_energy_change": 0.0,
    }
    errors = {horizon: {"visible_max_abs": 0.0} for horizon in ("3", "8")}
    return context, path, z, errors


@pytest.mark.parametrize(
    ("noise_mode", "expected_hidden", "expected_future"),
    [("clean", 1e-3, 1e-3), ("periodic", 1e-2, 1e-2), ("ramp", 1e-2, 1e-2)],
)
def test_metrics_selects_hidden_and_future_thresholds_from_noise_mode(
        noise_mode, expected_hidden, expected_future):
    context, path, z, errors = _metric_fixture(noise_mode=noise_mode)
    tolerances = {
        "hidden_clean": 1e-3, "hidden_noisy": 1e-2,
        "future_clean": 1e-3, "future_noisy": 1e-2,
        "noninferiority": 1e-8,
    }

    result = confirmation.metrics(
        context, z, path, errors, errors, errors, tolerances,
    )

    assert result["state_tolerance"] == expected_hidden
    assert result["future_tolerance"] == expected_future
    assert result["future_3s"] is True
    assert result["future_8s"] is True


def test_metrics_zero_initial_energy_ratio_is_null_and_reset_mismatch_fails():
    context, path, z, errors = _metric_fixture(z=np.zeros(2), z_t=np.zeros(2))
    tolerances = {
        "hidden_clean": 1e-3, "hidden_noisy": 1e-2,
        "future_clean": 1e-3, "future_noisy": 1e-2,
        "noninferiority": 1e-8,
    }
    metrics = confirmation.metrics(context, z, path, errors, errors, errors, tolerances)
    assert metrics["hidden_energy_origin_over_initial"] is None

    bad_path = {**path, "reset": path["reset"].copy()}
    bad_path["reset"][0] += 1.0
    with pytest.raises(FloatingPointError, match="causal initialization/reset identity"):
        confirmation.metrics(context, z, bad_path, errors, errors, errors, tolerances)


class FakeOperations:
    def __init__(self, directory, plan):
        self.directory = directory
        self.plan = plan
        self.counts = Counter()
        self.returns = Counter()
        self.journal = type("Journal", (), {"closed_scopes": [], "close": lambda _self: None})()
        self.clock_checks = []

    @contextmanager
    def scope(self, _name):
        yield

    def call(self, category, function, *, tag=""):
        del tag
        self.counts[category] += 1
        value = function()
        self.returns[category] += 1
        return value


def _install_mock_execution(monkeypatch, *, active_case="CW_periodic", rank_failure=False):
    instances, saved_records, pair_calls = [], [], []

    def make_ops(directory, plan):
        ops = FakeOperations(directory, plan)
        ops.partial = {}

        @contextmanager
        def tracked_scope(name):
            ops.partial = {}
            try:
                yield
            except BaseException:
                ops.journal.closed_scopes.append({"scope_id": name, "status": "failed"})
                raise
            else:
                ops.journal.closed_scopes.append({"scope_id": name, "status": "complete"})

        ops.scope = tracked_scope
        instances.append(ops)
        return ops

    monkeypatch.setattr(confirmation, "Operations", make_ops)
    monkeypatch.setattr(confirmation, "git_head", lambda _root: "mock-source-commit")
    monkeypatch.setattr(confirmation, "version", lambda _name: "0.mock")
    monkeypatch.setattr(confirmation, "save_record",
                        lambda _directory, name, record: saved_records.append((name, record)))
    monkeypatch.setattr(confirmation, "model_parameters", lambda _model: {"mock": True})
    monkeypatch.setattr(confirmation, "endpoint_check",
                        lambda *_args, **_kwargs: {"maximum_error": 0.0})

    def spend(ops, **quotas):
        for category, count in quotas.items():
            for _ in range(count):
                ops.call(category, lambda: None, tag=f"stubbed_{category}")

    class Model:
        mass = np.ones(10)

    def make_model(_base, ops, *, weak=False):
        spend(ops, probe=1)
        model = Model()
        model.family = "weak" if weak else "main"
        return model, np.zeros(10), np.zeros(22)

    def linear_bundle(_model, _power, ops, _tolerance):
        spend(ops, probe=1, expm=7, physical_rhs=3)
        identity = np.eye(23)
        return {"transition": identity, "endpoints": {10: identity, 800: identity}}

    def setup_exact_path(_transition, _initial, intervals, ops):
        assert intervals == 810
        spend(ops, rollouts=1)
        return np.zeros((811, 22))

    def setup_check_path(_model, _power, path, ops, _tolerances):
        assert path.shape == (811, 22)
        spend(ops, physical_rhs=3)
        return {"mock_checks": True}

    def sampled(_transition, intervals, ops):
        assert intervals == 10
        spend(ops, probe=1)
        return np.zeros((4, 2))

    def decompose(_matrix, ops, natural=0.0):
        del natural
        spend(ops, svd=1)
        return {"rank": 2, "s": np.ones(2), "u": np.eye(4, 2),
                "vh": np.eye(2), "threshold": 1e-12}

    def integral_system(_model, q, v, _power, ops):
        assert q.shape == v.shape == (11, 10)
        spend(ops, expm=1, force=60, svd=2, ls=2)
        rank = 1 if rank_failure else 2
        return {
            "matrix": np.zeros((10, 2)), "right4": np.zeros(10),
            "svd": {"rank": rank, "s": np.ones(2), "u": np.eye(10, 2),
                    "vh": np.eye(2), "threshold": 1e-12},
            "pc_svd": {"rank": 2},
            "z4": None if rank_failure else np.zeros(2),
        }

    def path(context, z):
        history = np.zeros((11, 22))
        history[:, 20:] = z
        history[0] = np.r_[context["observed"][0], z]
        reset = np.r_[context["observed"][-1], z]
        future = np.repeat(reset[None, :], 801, axis=0)
        return {"history": history, "reset": reset, "future": future,
                "reset_energy_change": 0.0}

    def pair(context, _model, z, _method, ops, _tolerances):
        pair_calls.append(context["case_id"])
        spend(ops, probe=1, rollouts=2, physical_rhs=6)
        return path(context, z)

    def solve(context, radius_id, ops):
        assert radius_id == "fixed_radius"
        spend(ops, probe=1)
        is_active = context["case_id"] == active_case
        return {
            "status": "active" if is_active else "inactive",
            "z": np.array([0.25, -0.5]) if is_active else context["unconstrained"].copy(),
            "radius": 1.0, "secular_evaluations": 2 if is_active else 0,
            "iterations": 1 if is_active else 0,
        }

    def build(context, _model, solution, _radius_id, ops, _tolerances):
        if solution["status"] == "active":
            # One conditional pair; all its numerical operations are represented
            # by counted no-op calls, while its state arrays remain artificial.
            pair_calls.append(context["case_id"])
            spend(ops, probe=1, rollouts=2, physical_rhs=6)
            candidate_paths = path(context, solution["z"])
            return {"paths": candidate_paths, "reused_unconstrained": False}
        return {"paths": context["baseline_paths"], "reused_unconstrained": True}

    def errors_stub(_estimate, _truth, ops):
        spend(ops, probe=1)
        return {horizon: {"visible_max_abs": 0.0} for horizon in ("3", "8")}

    monkeypatch.setattr(confirmation, "make_model", make_model)
    monkeypatch.setattr(confirmation, "linear_bundle", linear_bundle)
    monkeypatch.setattr(confirmation, "exact_path", setup_exact_path)
    monkeypatch.setattr(confirmation, "check_path", setup_check_path)
    monkeypatch.setattr(confirmation, "sampled_operator", sampled)
    monkeypatch.setattr(confirmation, "decompose", decompose)
    monkeypatch.setattr(confirmation, "integral_system", integral_system)
    monkeypatch.setattr(confirmation, "_run_pair", pair)
    monkeypatch.setattr(confirmation, "solve_candidate", solve)
    monkeypatch.setattr(confirmation, "build_candidate_paths", build)
    monkeypatch.setattr(confirmation, "errors", errors_stub)
    return instances, saved_records, pair_calls


def test_execute_mocked_setup_failure_writes_failure_counts_without_science(tmp_path, monkeypatch):
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    instances = []

    def make_ops(directory, plan):
        ops = FakeOperations(directory, plan)
        instances.append(ops)
        return ops

    monkeypatch.setattr(confirmation, "Operations", make_ops)
    monkeypatch.setattr(confirmation, "git_head", lambda _root: "artificial-commit")
    monkeypatch.setattr(confirmation, "version", lambda _name: "0.mock")

    def mocked_model_failure(_base, ops, *, weak=False):
        assert weak is False
        ops.call("probe", lambda: {"mock": True}, tag="mock_setup_probe")
        raise RuntimeError("mock setup stop")

    monkeypatch.setattr(confirmation, "make_model", mocked_model_failure)
    plan = {
        "cases": [{"id": name, "group": "main", "noise": "clean"}
                  for name in confirmation.CASES],
        "tolerances": confirmation.TOLERANCES, "radius": 1.0,
        "base": {"mock": True},
    }

    with pytest.raises(RuntimeError, match="mock setup stop"):
        confirmation.execute(plan=plan, root=tmp_path, result_dir=result_dir)

    failure = json.loads((result_dir / "failure.json").read_text(encoding="utf-8"))
    assert failure["process_status"] == "FAIL"
    assert failure["operator_attempted"] == {"probe": 1}
    assert failure["operator_returned"] == {"probe": 1}
    assert failure["retry_allowed"] is False
    assert instances[0].counts == instances[0].returns == {"probe": 1}
    assert not (result_dir / "summary.json").exists()


def test_execute_mock_success_writes_all_cases_methods_scopes_and_exact_counts(tmp_path, monkeypatch):
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    instances, saved_records, pair_calls = _install_mock_execution(monkeypatch)
    plan = {
        "cases": [
            {"id": name, "group": "weak" if name.startswith("CW") else "main",
             "noise": name.split("_")[1] if name.split("_")[1] != "clean" else "clean"}
            for name in confirmation.CASES
        ],
        "tolerances": confirmation.TOLERANCES, "radius": 1.0,
        "base": {"mock": True},
    }

    summary = confirmation.execute(plan=plan, root=tmp_path, result_dir=result_dir)

    assert summary["process_status"] == "PASS"
    assert summary["case_count"] == 6
    assert summary["method_count"] == 24
    assert summary["scope_count"] == 8
    assert summary["active_fixed_count"] == 1
    assert summary["operator_attempted"] == {
        "probe": 111, "rollouts": 40, "physical_rhs": 126,
        "expm": 20, "svd": 14, "ls": 12, "force": 360,
    }
    assert summary["operator_returned"] == summary["operator_attempted"]
    assert summary["confirmation_gates"] == {
        "fixed_all_absolute": True, "fixed_all_noninferior": True,
        "weak_noisy_constraint_exercised": True, "confirmation_criteria_met": True,
    }
    assert summary["path_transition_actions"] == 17010
    assert summary["truth_endpoint_actions"] == 4
    assert len(instances[0].journal.closed_scopes) == 8
    assert pair_calls.count("CW_periodic") == 4
    case_records = {name: record for name, record in saved_records if name in confirmation.CASES}
    assert set(case_records) == set(confirmation.CASES)
    assert all(set(record["paths"]) == set(confirmation.METHODS) for record in case_records.values())
    assert all(set(record["metrics"]) == set(confirmation.METHODS) for record in case_records.values())
    assert (result_dir / "execution_freeze.json").is_file()
    assert (result_dir / "summary.json").is_file()
    evidence = json.loads((result_dir / "stage_evidence.json").read_text(encoding="utf-8"))
    assert evidence["status"] == "PASS" and evidence["method_count"] == 24
    artifact_manifest = json.loads((result_dir / "artifact_hashes.json").read_text(encoding="utf-8"))
    assert "execution_freeze.json" in artifact_manifest
    assert "summary.json" in artifact_manifest
    assert "stage_evidence.json" in artifact_manifest


def test_execute_rank_unresolved_is_recorded_before_any_candidate_paths(tmp_path, monkeypatch):
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    instances, saved_records, pair_calls = _install_mock_execution(
        monkeypatch, rank_failure=True,
    )
    plan = {
        "cases": [
            {"id": name, "group": "weak" if name.startswith("CW") else "main",
             "noise": name.split("_")[1] if name.split("_")[1] != "clean" else "clean"}
            for name in confirmation.CASES
        ],
        "tolerances": confirmation.TOLERANCES, "radius": 1.0,
        "base": {"mock": True},
    }

    with pytest.raises(FloatingPointError, match="integral rank gate failed"):
        confirmation.execute(plan=plan, root=tmp_path, result_dir=result_dir)

    failure = json.loads((result_dir / "failure.json").read_text(encoding="utf-8"))
    assert failure["completed_cases"] == []
    assert failure["retry_allowed"] is False
    assert instances[0].partial["rank_unresolved"]["status"] == "unresolved"
    assert instances[0].partial["rank_unresolved"]["candidate_propagated"] is False
    assert pair_calls == []
    assert [name for name, _record in saved_records] == ["setup-main", "setup-weak"]
