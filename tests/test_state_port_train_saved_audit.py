import hashlib
import json

import numpy as np
import pytest

from kron_reduce.evaluation.constant_c_observability import save_record
from kron_reduce.evaluation.energy_cap_scores import score_pair
from kron_reduce.evaluation.scoped_audit_journal import ScopedAuditJournal
from scripts import audit_state_port_train_saved as audit


def test_decode_record_checks_json_declarations_and_spends_once(tmp_path):
    (tmp_path / "trace.json").write_text(json.dumps({
        "theta": {"array": "trace__theta", "shape": [2, 2], "dtype": "<f8"},
        "adam_step": {"array": "trace__step", "shape": [], "dtype": "<f4"},
    }), encoding="utf-8")
    np.savez_compressed(tmp_path / "trace.npz", trace__theta=np.eye(2),
                        trace__step=np.asarray(3, dtype=np.float32))
    decoded = set()
    record, arrays = audit.decode_record(tmp_path, "trace", decoded)
    np.testing.assert_array_equal(record["theta"], np.eye(2))
    assert record["adam_step"].shape == ()
    assert record["adam_step"].dtype == np.float32
    assert set(arrays) == {"trace__theta", "trace__step"}
    with pytest.raises(PermissionError, match="already decoded"):
        audit.decode_record(tmp_path, "trace", decoded)


def test_event_loss_uses_only_available_future_points():
    path = np.zeros((2, 4, 22))
    truth = np.zeros((2, 4, 20))
    path[0, 1, :20] = 1
    path[1, 1:4, :20] = 2
    values = audit.event_losses(path, truth, np.array([1, 3]), np.ones(20))
    np.testing.assert_allclose(values, [1.0, 4.0])
    desc = audit.describe_path(path, truth, np.array([1, 3]), np.ones(20))
    assert desc["event_ell_mean"] == pytest.approx(2.5)
    assert desc["pooled_normalized_mse"] == pytest.approx(13 / 4)
    assert desc["q_rmse_rad"] == pytest.approx((13 / 4) ** 0.5)
    assert desc["v_rmse_rad_per_s"] == pytest.approx((13 / 4) ** 0.5)


def test_adam_trace_recursion_is_checked_with_numpy_only():
    gradient = np.array([[0.1, -0.2], [0.3, 0.4]])
    m = 0.1 * gradient
    v = 0.001 * gradient ** 2
    theta = -0.01 * (m / 0.1) / (np.sqrt(v / 0.001) + 1e-8)
    row = {"gradient_clipped": gradient, "theta_before": np.zeros((2, 2)),
           "theta_after": theta,
           "adam": {"exp_avg": m, "exp_avg_sq": v}}
    valid, actual_m, actual_v = audit.adam_step_matches(row, np.zeros_like(m), np.zeros_like(v), 1)
    assert valid
    np.testing.assert_allclose(actual_m, m)
    np.testing.assert_allclose(actual_v, v)
    row["theta_after"] = theta + 0.01
    assert not audit.adam_step_matches(row, np.zeros_like(m), np.zeros_like(v), 1)[0]


def test_physics_check_recomputes_saved_identity_normalization():
    path = np.zeros((1, 2, 22))
    counts = np.array([1])
    inputs = {"constant_power": np.zeros((1, 10)), "equilibrium": np.zeros((1, 10))}
    params = {"mass": np.ones(10), "damping": np.zeros((10, 10)),
              "latent_dissipation": np.eye(2)}
    point_rows = []
    for point in (0, 0, 1):
        point_rows.append({"index": 0, "point": point, "energy_rate": 1e-9,
                           "supply_minus_dissipation": 0.0, "denominator": 1.0,
                           "identity_error": 1e-9, "normalized_error": 1e-9,
                           "hidden_energy": 0.0, "total_energy": 0.0})
    physics = {"points": point_rows, "qualified": True, "max_mass_coi": 0.0}
    valid, result = audit.check_physics_record(physics, path, counts, inputs, params)
    assert valid
    assert result["max_normalized_identity_error"] == pytest.approx(1e-9)
    point_rows[0]["normalized_error"] = 0.0
    valid, _ = audit.check_physics_record(physics, path, counts, inputs, params)
    assert not valid


def test_artifact_mismatch_stops_before_npz_decode(tmp_path):
    (tmp_path / "extra.txt").write_text("x", encoding="utf-8")
    (tmp_path / "artifact_hashes.json").write_text(json.dumps({
        "extra.txt": hashlib.sha256(b"x").hexdigest(),
    }), encoding="utf-8")
    check = audit.Audit()
    with pytest.raises(RuntimeError, match="before any decode"):
        audit.verify_artifact_manifest(tmp_path, check)
    assert any(row["name"] == "run.npz_roster" and not row["passed"] for row in check.checks)


def test_journal_fixture_checks_scope_ranges_and_zero_byte_scope(tmp_path):
    directory = tmp_path / "operations"
    journal = ScopedAuditJournal(directory)
    scope_limits = {
        "toy_scope": {"probe": 1, "rollouts": 1, "logical_limits": {}},
        "empty_backend": {"probe": 0, "rollouts": 0, "logical_limits": {}},
    }
    journal.begin("toy-scope", scope_limits["toy_scope"])
    journal.append({"category": "probe", "index": 1, "tag": "toy", "transition": "attempted"})
    journal.append({"category": "probe", "index": 1, "tag": "toy", "transition": "returned"})
    journal.append({"category": "rollout", "seed": 1, "event_id": "toy",
                    "step": 0.005, "transition": "attempted"})
    journal.append({"category": "rollout", "seed": 1, "event_id": "toy",
                    "step": 0.005, "transition": "returned"})
    journal.finish({"status": "completed_within_budget"})
    journal.begin("empty-backend", scope_limits["empty_backend"])
    journal.finish({"status": "completed_within_budget"})
    journal.close()
    plan = {"scope_limits": scope_limits, "logical_operation_limits": {},
            "operator_limits": {"probe": 1, "rollouts": 1}}
    result = audit.Audit()
    detail = audit.audit_journal(tmp_path, plan, result, expected_scope_count=2)
    assert detail == {"scopes": 2, "operator_calls": 2, "transitions": 4,
                      "log_bytes": (directory / "calls.jsonl").stat().st_size}
    assert not result.mismatches


def test_score_pair_record_round_trips_nested_tree_and_numpy_values(tmp_path):
    history = np.zeros((11, 20))
    truth = np.zeros((2, 20))
    coarse = np.zeros((2, 22))
    fine = np.zeros((2, 22))
    record = score_pair(history=history, truth=truth, coarse=coarse, fine=fine,
                        equilibrium=np.zeros(10), mass=np.ones(10), omega_s=1.0,
                        scale=np.ones(20), clock=np.array([0.0, 0.01]))
    record["fixture_tuple"] = (np.float64(1.25), np.array([2.0, 3.0]))

    save_record(tmp_path, "score", record)
    decoded, _ = audit.decode_record(tmp_path, "score", set())
    assert audit.compare_tree(decoded, record)
