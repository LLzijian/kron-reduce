"""Audit frozen post-fault models using sealed train/val prediction artifacts."""

from __future__ import annotations

from collections import Counter

import numpy as np
import torch
from scipy.stats import rankdata

from kron_reduce.physics.coordinates import coi_project

STEPS = ("0.0025", "0.00125")
ORIGIN = 10


def archived_cohort(roster, hard, m3, role):
    """Reconstruct the TASK3 view from the frozen V4 truth archive."""
    ids = [e["id"] for e in roster]
    labels = [e["label"] for e in roster]
    if role not in ("train", "val") or hard["ids"].tolist() != ids or m3["ids"].tolist() != ids:
        raise ValueError("frozen M4 archives do not match the authorized roster")
    if hard["labels"].tolist() != labels or Counter(labels) != ({"stable": 311, "unstable": 35}
                                                            if role == "train" else {"stable": 95, "unstable": 12}):
        raise ValueError("frozen M4 archive labels changed")
    lengths, truth = hard["lengths"], hard["truth"]
    mass, equilibrium = hard["mass"], hard["equilibrium"]
    if (lengths.shape != (len(ids),) or lengths.dtype.kind not in "iu" or (lengths < 12).any()
            or (lengths > 811).any() or truth.shape != (lengths.sum(), 20)
            or mass.shape != (len(ids), 10) or equilibrium.shape != mass.shape
            or not all(a.dtype == np.float64 and np.isfinite(a).all() for a in (truth, mass, equilibrium))
            or (mass <= 0).any()):
        raise ValueError("invalid archived TASK3 truth or reference state")
    forecasts = {"hard_r": dict(zip(STEPS, (hard["fine"], hard["refined"]), strict=True)),
                 "m3": dict(zip(STEPS, (m3["main"], m3["refined"]), strict=True))}
    clips = np.split(truth, np.cumsum(lengths)[:-1])
    events = []
    for meta, raw, m, reference in zip(roster, clips, mass, equilibrium, strict=True):
        x = np.c_[coi_project(raw[:, :10], m), coi_project(raw[:, 10:], m)]
        events.append(meta | {"role": role, "x": x, "raw_x": raw, "mass": m,
                              "equilibrium": reference, "omega_s": float(2*np.pi*60)})
    origin = np.stack([e["x"][ORIGIN] for e in events])
    for by_step in forecasts.values():
        for path in by_step.values():
            if path.shape != (len(ids), 801, 20) or path.dtype != np.float64:
                raise ValueError("invalid frozen prediction roster or precision")
            if not np.allclose(path[:, 0], origin, rtol=0, atol=1e-12):
                raise ValueError("frozen predictions changed the observed forecast origin")
    return events, forecasts


def archive_diagnostics(roster, hard, m3):
    ids = [e["id"] for e in roster]
    expected = [(e["id"], e["label"]) for e in roster]
    data = {}
    for step in STEPS:
        hard_rows = [r for r in hard["initialization"] if float(step) == r["step"]]
        saved = m3["cases"][step]
        hard_classes, m3_classes = hard["stability_events"], saved["classification"]
        if ([r["id"] for r in hard_rows] != ids or list(saved["origin_memory_energy"]) != ids
                or any([(r["id"], r["truth_label"]) for r in rows] != expected
                       for rows in (hard_classes, m3_classes))):
            raise ValueError("frozen per-event diagnostics differ from the authorized roster")
        data[step] = {"hard_r": ([r["hidden_origin_energy"] for r in hard_rows], hard_classes),
                      "m3": (list(saved["origin_memory_energy"].values()), m3_classes)}
    return data


def _auc(positive, value):
    n = int(positive.sum())
    m = len(value)-n
    if not n or not m or not np.isfinite(value).all():
        return None
    ranks = rankdata(value, method="average")
    return float((ranks[positive].sum()-n*(n+1)/2)/(m*n))


@torch.no_grad()
def energies(model, events, memory):
    if model.topology_dimension or model.memory_dimension != 10:
        raise ValueError("archived scalar memory energy applies to fixed ten-memory systems")
    state = np.stack([e["x"][ORIGIN] for e in events])
    equilibrium = np.stack([e["equilibrium"] for e in events])
    inertia = model.mass.cpu().numpy()
    if not np.allclose(np.stack([e["mass"] for e in events]), inertia, rtol=0, atol=1e-12):
        raise ValueError("archived and model inertia differ")
    kinetic = .5*(state[:, 10:]**2 @ inertia)
    visible = model.storage(model._tensor(np.c_[state, np.zeros((len(events), 10))]),
                            model._tensor(equilibrium)).cpu().numpy()
    hidden = np.asarray(memory, dtype=np.float64)
    if hidden.shape != (len(events),) or (hidden < 0).any() or not np.isfinite(hidden).all():
        raise ValueError("invalid archived origin memory energy")
    quantities = {"total": visible+hidden, "kinetic": kinetic, "potential": visible-kinetic, "memory": hidden}
    positive = np.array([e["label"] == "unstable" for e in events])
    summary = {k: {"auroc": _auc(positive, v), "stable_q25_median_q75": np.quantile(v[~positive], [.25,.5,.75]).tolist(),
                   "unstable_q25_median_q75": np.quantile(v[positive], [.25,.5,.75]).tolist()}
               for k, v in quantities.items()}
    return {"summary": summary, "rows": [{"id": e["id"], "label": e["label"],
            **{k: float(v[i]) for k, v in quantities.items()}} for i, e in enumerate(events)]}


def _partition(path, hit, event):
    angle = path[hit, :10]+event["equilibrium"]
    order = np.argsort(angle, kind="stable")
    gap = np.diff(angle[order])
    boundary = int(np.argmax(gap))+1
    low, high = np.sort(order[:boundary]), np.sort(order[boundary:])
    speed, mass = path[hit, 10:], event["mass"]
    return {"trigger_s_from_origin": (hit-ORIGIN)*.01, "leading": high.tolist(), "trailing": low.tolist(),
            "max_angular_gap_rad": float(gap[boundary-1]), "relative_group_speed_rad_s": float(
                np.average(speed[high], weights=mass[high])-np.average(speed[low], weights=mass[low]))}


def _crossing_edges(partition):
    return {tuple(sorted((i, j))) for i in partition["leading"] for j in partition["trailing"]}


def separation_cuts(events, paths, qualifications):
    """Compare leading groups at each trajectory's own first TASK3 LOS trigger."""
    from task3.protocol import classify

    rows = []
    for event, pred, qualified in zip(events, paths, qualifications, strict=True):
        if event["label"] != "unstable":
            continue
        raw = event["raw_x"]
        scoring = event | {"x": raw, "equilibrium_delta": event["equilibrium"]}
        observed_hit = classify(raw[ORIGIN:], scoring)["trigger_index"]
        estimated_hit = classify(pred, scoring)["trigger_index"] if qualified["qualified"] else None
        if qualified["qualified"] and (estimated_hit is not None) != (qualified["predicted_label"] == "unstable"):
            raise ValueError("archived cut trigger and frozen classification disagree")
        truth = None if observed_hit is None else _partition(event["x"], observed_hit, event)
        pred_full = np.concatenate((event["x"][:ORIGIN+1], pred[1:]))
        estimated = None if estimated_hit is None else _partition(pred_full, estimated_hit, event)
        row = {"id": event["id"], "truth": truth, "prediction": estimated, "numerically_qualified": qualified["qualified"],
               "exact_leading": None, "exact_cut": None, "cut_jaccard": None}
        if truth is not None and estimated is not None:
            a, b = _crossing_edges(truth), _crossing_edges(estimated)
            row.update(exact_leading=truth["leading"] == estimated["leading"], exact_cut=a == b,
                       cut_jaccard=len(a & b)/len(a | b))
        rows.append(row)
    paired = [r for r in rows if r["exact_cut"] is not None]
    exact, directional = sum(r["exact_cut"] for r in paired), sum(r["exact_leading"] for r in paired)
    observed = sum(r["truth"] is not None for r in rows)
    return {"allocated": len(rows), "observed_trigger": observed, "predicted_trigger": sum(r["prediction"] is not None for r in rows),
            "paired": len(paired), "exact_cut": exact, "exact_leading": directional,
            "exact_cut_rate_paired": exact/len(paired) if paired else None,
            "exact_leading_rate_paired": directional/len(paired) if paired else None,
            "exact_cut_rate_observed": exact/observed if observed else None,
            "cut_jaccard_mean_paired": float(np.mean([r["cut_jaccard"] for r in paired])) if paired else None,
            "rows": rows}
