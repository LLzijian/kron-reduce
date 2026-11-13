import copy
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_kiki_machine_local_full_saved.py"
spec = importlib.util.spec_from_file_location("audit_saved", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def _case(monkeypatch, *, mutate=False):
    event_id = "ieee39_postfault/pilot_0001"
    event = {"id": event_id, "x": np.zeros((12, 20)), "mass": np.ones(10),
             "equilibrium_delta": np.zeros(10), "omega_s": audit.OMEGA_S,
             "label": "stable", "group": "line_1", "event_time_s": None}
    classification = {"event_id": event_id, "label": "stable", "category": "window_settled",
                      "predicted_los": False, "window_settled": True, "qualified": True}
    recomputed = {
        "summary": {"1": {"mean": 1.25}},
        "unstable_prefix": {"n_events": 0},
        "stable_complete": {"1": {"mean": 1.25}},
        "counts": {"eligible": 1},
        "records": [{"event_id": event_id, "horizon": "1", "rmse": 0.25},
                    {"event_id": event_id, "horizon": "prefix", "rmse": 0.4}],
        "classifications": [classification],
        "selection_score": 0.25,
    }

    def task3_eval(*_args):
        return recomputed

    def classify(*_args):
        return {"predicted_los": False, "window_settled": True}

    stability_events = [{"id": event_id, "truth_label": "stable", "predicted_label": "stable",
                         "qualified": True, "prospective_eligible": True,
                         "numerical_failure": False, "unsettled": False}]
    from kron_reduce.evaluation.trajectory_stability import summarize
    stability = summarize(stability_events)
    stability.update(recall_on_resolved=None, miss_rate_on_resolved=None,
                     false_alarm_rate_on_resolved=0.0)
    saved = copy.deepcopy({key: value for key, value in recomputed.items() if key != "selection_score"})
    saved.update(stability=stability, stability_events=stability_events, task3_six_term_score=0.25)
    saved["initialization"] = [{"id": event_id, "step": 0.0025, "history_nrmse": 0.2,
                                "hidden_start_energy": 0.3, "hidden_origin_energy": 0.4}]
    if mutate:
        saved["records"][0]["rmse"] = 0.5
    monkeypatch.setattr(audit, "load_saved_role", lambda *_args: (
        {"fine": np.zeros((1, 801, 20)), "refined": np.zeros((1, 801, 20))},
        [event], saved, "truth-hash",
    ))
    result = audit.audit_role(Path("."), Path("."), "train", {"run_files": {"train": {"npz_sha256": "npz"}}},
                              [event_id], task3_eval, classify, summarize)
    return result


def test_saved_metric_comparison_passes_exact_and_flags_mutated_record(monkeypatch):
    exact = _case(monkeypatch)
    assert exact["mismatch_count"] == 0, exact["mismatches"]
    assert exact["train_diagnosis"]["count"] == 1
    assert exact["train_diagnosis"]["history_residual_top_quartile"]["prefix_task3_rmse_mean"] == 0.4
    result = _case(monkeypatch, mutate=True)
    assert result["mismatch_count"] == 1
    assert result["mismatches"][0]["path"] == "train.records[0].rmse"
