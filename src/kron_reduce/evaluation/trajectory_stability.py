"""100 ms trajectory-derived stability reports with explicit coverage."""
from __future__ import annotations

import numpy as np

from kron_reduce.data.history_contract import CausalStateHistory
from kron_reduce.workflows.kiki_stability_comparison import confusion_metrics, los_classification


def classify(history_q, history_v, forecast, equilibrium, mass):
    if np.shape(history_q) != (11, 10) or np.shape(history_v) != (11, 10):
        raise ValueError("stability reporting requires eleven ten-machine observations")
    history = CausalStateHistory(np.arange(-10, 1)*.01, history_q, history_v, np.zeros(10))
    row = los_classification(history, np.asarray(forecast), equilibrium, mass, 2*np.pi*60)
    # The shared historical classifier uses a 200 ms origin for its time field.
    hit = row["trigger_index"]
    row["predicted_los_time_forecast_s"] = None if hit is None else (hit-10)*.01
    row["label"] = ("unstable" if row["predicted_los"] else "stable" if row["window_settled"] else None)
    if not row["finite_full_path"]:
        row["label"] = None
    return row


def summarize(events):
    active = [r for r in events if r["truth_label"] in ("stable", "unstable") and r["prospective_eligible"]]
    u = [r for r in active if r["truth_label"] == "unstable"]
    s = [r for r in active if r["truth_label"] == "stable"]
    resolved = lambda r: r["qualified"] and r["predicted_label"] in ("stable", "unstable")
    tp = sum(resolved(r) and r["predicted_label"] == "unstable" for r in u)
    fn = sum(resolved(r) and r["predicted_label"] == "stable" for r in u)
    fp = sum(resolved(r) and r["predicted_label"] == "unstable" for r in s)
    tn = sum(resolved(r) and r["predicted_label"] == "stable" for r in s)
    row = confusion_metrics(tp, fn, fp, tn, sum(not resolved(r) for r in u), sum(not resolved(r) for r in s))
    row.update(allocated=len(events), resolved=tp+tn+fp+fn, uncovered=len(events)-len(active),
               unresolved=len(active)-tp-tn-fp-fn,
               coverage=(tp+tn+fp+fn)/len(events) if events else None,
               correct_fraction_lower_bound=(tp+tn)/len(events) if events else None,
               accuracy_on_resolved=(tp+tn)/(tp+tn+fp+fn) if tp+tn+fp+fn else None,
               numerical_failures=sum(r.get("numerical_failure", False) for r in events),
               unsettled_predictions=sum(r.get("unsettled", False) for r in events),
               weighting="event_equal", forecast_horizon_s=8., positive_class="unstable")
    if row["uncovered"]:
        for key in ("accuracy", "balanced_accuracy", "recall", "fsr", "fur", "precision", "f1"):
            row[key] = None
    return row
