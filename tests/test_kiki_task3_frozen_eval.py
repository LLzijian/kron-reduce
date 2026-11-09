from types import SimpleNamespace

import numpy as np
import pytest

from kron_reduce.workflows import kiki_task3_frozen_eval as frozen
from kron_reduce.workflows.kiki_task3_frozen_eval import evaluate_model, forecast, run


def test_unregistered_run_cannot_reach_cache(tmp_path):
    with pytest.raises(PermissionError, match="not registered active"):
        run(tmp_path)
    assert not list(tmp_path.iterdir())


def test_copied_or_spent_authorization_is_refused_before_task3_reads(tmp_path, monkeypatch):
    checkout = tmp_path / "bound"
    monkeypatch.setattr(frozen, "reviewed_plan", lambda root: {"checkout": str(checkout)})
    monkeypatch.setattr(frozen, "checked_bytes", lambda *args: pytest.fail("unexpected TASK3 read"))
    with pytest.raises(PermissionError, match="bound to one checkout"):
        run(tmp_path)
    marker = checkout / ".project_state" / f"{frozen.AUTH}.spent.json"
    marker.parent.mkdir(parents=True)
    frozen.write_json(marker, {"run_id": "synthetic"})
    with pytest.raises(FileExistsError):
        frozen.write_json(marker, {"run_id": "duplicate"})
    with pytest.raises(PermissionError, match="already spent"):
        run(checkout)


def test_forecast_preserves_observed_origin_and_uses_zero_common():
    mass = np.arange(1, 11, dtype=float)
    data = np.arange(22, dtype=float)
    model = SimpleNamespace(
        closure=SimpleNamespace(mass=mass),
        predict=lambda history, clock, equilibrium, max_step: np.tile(data, (len(clock), 1)),
    )
    origin = np.arange(20, dtype=float) + 100
    output = forecast(model, None, origin, np.zeros(10), 0.005)
    assert output.shape == (801, 20)
    np.testing.assert_array_equal(output[0], origin)
    np.testing.assert_array_equal(output[1, :10], data[:10])
    np.testing.assert_allclose(output[1:, 10:] @ mass, 0, atol=1e-12)


def test_one_refinement_keeps_failed_slot_and_binds_budget(tmp_path):
    calls = []
    journal = []

    def predict(history, clock, equilibrium, max_step):
        calls.append(max_step)
        if max_step == 0.01:
            raise FloatingPointError("synthetic failed path")
        return np.zeros((len(clock), 20))

    def evaluate(fine, events, comparison):
        failed = not np.isfinite(comparison).all()
        return {
            "selection_score": None if failed else 0.0,
            "counts": {"numerical_unresolved": int(failed)},
            "records": [],
        }

    model = SimpleNamespace(closure=SimpleNamespace(mass=np.ones(10)), predict=predict)
    events = [
        {
            "x": np.zeros((22, 20)),
            "p0": np.zeros(10),
            "id": "synthetic",
            "equilibrium_delta": np.zeros(10),
        }
    ]
    plan = {
        "max_model_paths": 3,
        "max_rk4_updates": 5740,
        "max_rhs_calls": 22960,
        "output_convention": "test",
    }
    budget = {"paths_reserved": 0, "rk4_updates_reserved": 0}
    result = evaluate_model(
        model, events, evaluate, plan, budget, journal, "frozen", "val", tmp_path
    )
    assert calls == [0.005, 0.01, 0.0025]
    assert result["initial_numerical_qualification"]["counts"]["numerical_unresolved"] == 1
    assert result["task3_six_term_score_zero_common"] == 0
    assert len(result["path_failures"]) == 1
    assert budget == {"paths_reserved": 3, "rk4_updates_reserved": 5740}
