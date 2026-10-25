"""Journaled A17 path composition on synthetic arrays only."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from kron_reduce.evaluation import input_consistency_paths as paths
from kron_reduce.evaluation.input_consistency_evidence import (
    error_record,
    save_manifest,
    save_record,
)
from kron_reduce.evaluation.input_consistency_journal import InputConsistencyJournal
from kron_reduce.evaluation.scoped_audit_journal import ScopeJournalError
from kron_reduce.models.nonlinear_latent import NonlinearPassiveClosure
from kron_reduce.physics.coordinates import coi_project


def _integral_fixture() -> tuple[dict, np.ndarray, dict]:
    U = np.zeros((100, 2))
    U[0, 0] = U[1, 1] = 1.0
    singular = np.array([3.0, 2.0])
    Vh = np.array([[0.0, -1.0], [1.0, 0.0]])
    matrix = (U * singular) @ Vh
    old_z = np.array([0.2, -0.1])
    old_rhs = matrix @ old_z
    saved_z = Vh.T @ ((U.T @ old_rhs) / singular)
    phi = np.array([[0.9, 0.1], [-0.1, 0.9]])
    w = np.array([0.01, -0.02])
    clock = np.linspace(0.0, 0.1, 11)
    integral = {
        "Ltilde": matrix, "U": U, "s": singular, "Vh": Vh, "btilde4": old_rhs,
        "c": np.ones(10), "S": np.ones(10), "h": np.diff(clock), "time": clock,
        "zh2": saved_z.copy(), "zh4": saved_z.copy(),
        "zTobs2": phi @ saved_z + w, "zTobs4": phi @ saved_z + w,
        "PhiT": phi, "wT": w,
    }
    arguments = {
        "rank": 2,
        "threshold": float(64 * np.finfo(np.float64).eps * 100 * singular[0]),
        "natural_norm": 0.0,
        "determinant_lower": 36.0,
        "trace_upper": 13.0,
    }
    return integral, matrix @ np.array([0.03, -0.04]), arguments


def _closure() -> NonlinearPassiveClosure:
    mass = np.linspace(0.1, 1.3, 10)
    projector = np.eye(10) - np.ones((10, 10)) / 10
    coupling = np.column_stack((np.arange(10) - 4.5, (-1.0) ** np.arange(10))) * 0.01
    return NonlinearPassiveClosure(
        mass, np.full(45, 0.04), 0.03 * projector, coupling,
        np.array([[0.0, 0.6], [-0.6, 0.0]]), np.diag([0.2, 0.4]),
    )


def _initial_state(model: NonlinearPassiveClosure, z: np.ndarray) -> dict:
    return {
        "q": coi_project(np.linspace(-0.01, 0.01, 10), model.mass),
        "v": coi_project(np.sin(np.arange(10)) * 0.01, model.mass),
        "z": z.copy(),
    }


def _dispatch(journal: InputConsistencyJournal):
    return lambda category, function: journal.call(category, function)


def _successful_reservation() -> dict[str, int]:
    caps = Counter(paths.initializer_success_caps())
    for linear, count, updates in (
        (True, 11, 20),
        (True, 11, 40),
        (False, 3, 4),
        (False, 3, 8),
    ):
        caps.update(paths.path_success_caps(
            point_count=count, rk4_updates=updates, linear=linear
        ))
    caps.update({key: value * 2 for key, value in paths.reset_success_caps().items()})
    return dict(caps)


def test_ls_history_reset_future_pair_is_journaled_saved_and_closed(tmp_path: Path) -> None:
    model = _closure()
    integral, impulse, init_args = _integral_fixture()
    journal = InputConsistencyJournal(tmp_path / "composed-run")
    journal.begin("synthetic-composition", _successful_reservation())
    dispatch = _dispatch(journal)

    initialized = paths.estimate_initial(integral, impulse, dispatch=dispatch, **init_args)
    assert initialized["status"] == "complete"
    assert initialized["ls"]["z0"].shape == (2,)

    pattern = np.arange(10, dtype=np.float64) - 4.5
    equilibrium = np.linspace(-0.05, 0.05, 10)
    target_scale = np.ones(20)
    original = _initial_state(model, initialized["z0"])
    histories, resets, futures = [], [], []

    for step in (0.005, 0.0025):
        history_clock = np.linspace(0.0, 0.1, 11)
        history = paths.evaluate_fixed_path(
            model, original["q"], original["v"], original["z"], history_clock,
            np.arange(11, dtype=np.float64)[:, None] * pattern * 0.001,
            equilibrium, max_step=step, linear=True, target_scale=target_scale,
            dispatch=dispatch,
        )
        histories.append(history)
        assert history["status"] == "complete"
        assert history["derived_actual"]["successful_rk4_updates"] == (20 if step == 0.005 else 40)
        assert history["call_counts"]["integration_rhs"]["callback_returned"] == (
            80 if step == 0.005 else 160
        )

        reset = paths.reset_path_origin(
            model, history["path"], history["path"][-1, :20], equilibrium,
            dispatch=dispatch,
        )
        resets.append(reset)
        assert reset["status"] == "complete"
        assert np.array_equal(reset["reset"]["origin"][20:], history["path"][-1, 20:])

        origin = reset["reset"]["origin"]
        future = paths.evaluate_fixed_path(
            model, origin[:10], origin[10:20], origin[20:], np.array([0.0, 0.01, 0.02]),
            pattern * 0.0005, equilibrium, max_step=step, linear=False,
            target_scale=target_scale, dispatch=dispatch,
        )
        futures.append(future)
        assert future["status"] == "complete"
        assert future["derived_actual"]["successful_rk4_updates"] == (4 if step == 0.005 else 8)
        assert future["call_counts"]["integration_rhs"]["callback_returned"] == (
            16 if step == 0.005 else 32
        )

    all_records = {"initializer": initialized, "histories": histories,
                   "visible_resets": resets, "futures": futures}
    caps = _successful_reservation()
    report_path = journal.directory / "composition.json"

    def persist():
        save_record(journal.directory, "composition", all_records)
        return save_manifest(journal.directory)

    closed = journal.persist_and_finish(persist, status="complete")
    journal.close()

    assert closed["durably_closed"] is True
    assert closed["actual_counters"]["attempted"] == caps
    assert closed["actual_counters"]["returned"] == caps
    assert closed["actual_counters"]["failed"] == {}
    assert len(histories) == len(resets) == len(futures) == 2
    assert all(record["checks"]["physical_judgement"]["qualified"] is True
               for record in (*histories, *futures))

    actual_callback_counts = Counter()
    for record in (initialized, *histories, *resets, *futures):
        for category, counts in record["call_counts"].items():
            actual_callback_counts[category] += counts["callback_returned"]
            assert counts["callback_attempted"] == counts["callback_returned"]
            assert counts["callback_failed"] == counts["dispatch_failed"] == 0
            assert counts["dispatch_contract_failed"] == 0
    assert actual_callback_counts == Counter(caps)
    assert actual_callback_counts["integration_rhs"] == 288
    assert actual_callback_counts["input_eval"] == 240
    assert actual_callback_counts["physical_rhs"] == 28
    assert "structural_eigvalsh" not in actual_callback_counts
    assert "closure_constructions" not in actual_callback_counts

    transitions = [json.loads(line) for line in (journal.directory / "calls.jsonl").read_text().splitlines()]
    attempted = Counter(row["category"] for row in transitions if row["transition"] == "attempted")
    returned = Counter(row["category"] for row in transitions if row["transition"] == "returned")
    assert attempted == returned == Counter(caps)
    artifact_manifest = json.loads((journal.directory / "artifact_hashes.json").read_text())
    assert artifact_manifest == closed["artifact_hashes"]
    assert report_path.exists() and (journal.directory / "composition.npz").exists()
    saved = json.loads(report_path.read_text())
    assert saved["histories"][0]["derived_actual"]["successful_rk4_updates"] == 20
    assert saved["histories"][1]["derived_actual"]["successful_rk4_updates"] == 40
    assert saved["futures"][0]["derived_actual"]["successful_rk4_updates"] == 4
    assert saved["futures"][1]["derived_actual"]["successful_rk4_updates"] == 8
    assert (journal.directory / "synthetic-composition-closed.json").exists()


def test_malicious_future_clock_fails_before_rhs_and_retains_failure_record(tmp_path: Path) -> None:
    model = _closure()
    journal = InputConsistencyJournal(tmp_path / "bad-future-clock")
    journal.begin("synthetic-bad-clock", {"path_evaluations": 1, "path_preflight": 1})
    args = {
        **_initial_state(model, np.array([0.2, -0.1])),
        "closure": model,
        "clock": np.array([0.0, 0.01, 3.01]),
        "power": (np.arange(10, dtype=np.float64) - 4.5) * 0.001,
        "equilibrium": np.linspace(-0.05, 0.05, 10),
        "max_step": 0.005,
        "linear": False,
        "target_scale": np.ones(20),
    }

    with pytest.raises(ScopeJournalError) as caught:
        paths.evaluate_fixed_path(dispatch=_dispatch(journal), **args)
    partial = getattr(caught.value, "input_consistency_paths_partial_record", None)
    assert partial is not None
    assert partial["status"] == "global_failure"
    assert partial["call_counts"].get("integration_rhs", {}).get("callback_attempted", 0) == 0
    assert journal.attempted["integration_rhs"] == 0
    assert journal.returned["integration_rhs"] == 0
    assert journal.poisoned is True
    assert journal.closed_scopes == []
    assert not (journal.directory / "synthetic-bad-clock-closed.json").exists()

    save_record(journal.directory, "bad-clock-error", error_record(caught.value))
    saved_error = json.loads((journal.directory / "bad-clock-error.json").read_text())
    assert saved_error["retry_allowed"] is False
    assert "input_consistency_paths_partial_record" in saved_error
    journal.close()

