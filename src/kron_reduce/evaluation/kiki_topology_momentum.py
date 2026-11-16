"""Train-only observed-path weak-momentum audit; performs no model forecasts."""
from __future__ import annotations

import json
import math
from collections import Counter
from functools import partial
from pathlib import Path

import numpy as np
import torch

from kron_reduce.data.durable_audit_ledger import AppendOnlyAuditJournal
from kron_reduce.data.task3_full import Task3FullReader
from kron_reduce.evaluation.constant_c_observability import write_json
from kron_reduce.evaluation.kiki_machine_local_r20_r21 import verified
from kron_reduce.evaluation.kiki_topology_bridge import load_snapshot, make_model, predict
from kron_reduce.evaluation.machine_local_clean_learning import NUMERICAL_ERRORS
from kron_reduce.evaluation.topology_angle_momentum import (
    angle_momentum,
    angle_scale_residuals,
    damping_sse,
    fit_train_damping,
)
from kron_reduce.evaluation.topology_dynamic_momentum import ANCHORS, STARTS, coupling_residuals
from kron_reduce.evaluation.topology_modal_momentum import modal_probe, modal_residuals
from kron_reduce.evaluation.topology_momentum import (
    direction_changes,
    inspect_momentum,
    origin_transfer,
)
from kron_reduce.evaluation.topology_momentum_work import account_path
from kron_reduce.evaluation.topology_restoring_momentum import (
    fit_restoring_train,
    restoring_momentum,
    restoring_sse,
)
from kron_reduce.evaluation.topology_spatial_momentum import (
    fit_spatial_train,
    spatial_damping_root,
    spatial_momentum,
    spatial_sse,
)
from kron_reduce.evaluation.train_mechanism_audit import _row_margin, _window_result
from kron_reduce.models.nonlinear_latent import LearnedNonlinearModel

TRAIN_COUNT = 346
UNSTABLE_COUNT = 35
MAX_FRAME = 311
DT = 0.01
ORIGIN_FRAME = 10


def _array(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().copy()
    return np.asarray(value, dtype=np.float64).copy()


def _model_blocks(model):
    return tuple(_array(value) for value in (
        model.mass, model.incidence, model.effective_stiffness,
        model.effective_damping, model.coupling, model.tau,
    ))


def _groups(rows):
    groups = {"train": rows}
    for label in ("stable", "unstable"):
        groups[f"label/{label}"] = [row for row in rows if row["label"] == label]
    faults = sorted({str(row["fault_group"]) for row in rows})
    for fault in faults:
        fault_rows = [row for row in rows if str(row["fault_group"]) == fault]
        groups[f"fault/{fault}"] = fault_rows
        for label in ("stable", "unstable"):
            label_rows = [row for row in fault_rows if row["label"] == label]
            if label_rows:
                groups[f"label/{label}/fault/{fault}"] = label_rows
    return groups


def _summarize(rows):
    count = len(rows)
    if not count:
        return {"events": 0, "full_rank": 0, "rank_deficient": 0,
                "condition_median": None, "condition_max": None,
                "orthogonal_sse": None, "rhs_sse": None,
                "orthogonal_fraction": None, "ball": None}
    conditions = [float(row["condition"]) for row in rows if row["condition"] is not None]
    rhs = sum(float(row["rhs_sse"]) for row in rows)
    orthogonal = sum(float(row["orthogonal_sse"]) for row in rows)
    ball_rows = [row["ball"] for row in rows if row.get("ball") is not None]
    ball = None
    if ball_rows:
        residuals = [float(row["residual_sse"]) for row in ball_rows
                     if row.get("residual_sse") is not None]
        norms = [float(np.linalg.norm(row["state"])) for row in ball_rows if row.get("state") is not None]
        limits = {float(row["radius_bound"]) for row in rows if row.get("radius_bound") is not None}
        ball_residual = sum(residuals) if len(residuals) == len(ball_rows) else None
        ball = {"solves": len(ball_rows),
                "resolved": sum(row["status"] not in ("numerical_failure", "unresolved") for row in ball_rows),
                "unresolved": sum(row["status"] == "unresolved" for row in ball_rows),
                "numerical_failures": sum(row["status"] == "numerical_failure" for row in ball_rows),
                "constraint_active": sum(row.get("constraint_active") is True for row in ball_rows),
                "state_norm_max": max(norms) if norms else None,
                "radius_bound": next(iter(limits)) if len(limits) == 1 else None,
                "residual_sse": ball_residual,
                "residual_ratio_to_algebraic": ball_residual/orthogonal if ball_residual is not None and orthogonal > 0 else None}
    return {"events": count,
            "full_rank": sum(row["rank"] == row["dimension"] for row in rows),
            "rank_deficient": sum(row["rank"] < row["dimension"] for row in rows),
            "condition_defined": len(conditions),
            "condition_median": float(np.median(conditions)) if conditions else None,
            "condition_max": max(conditions) if conditions else None,
            "orthogonal_sse": orthogonal, "rhs_sse": rhs,
            "orthogonal_fraction": orthogonal/rhs if rhs > 0 else None,
            "ball": ball}


def _decorate(row, event, *, kind, memory_inertia=1.0, radius=None, window=None):
    result = {"id": event["id"], "role": "train", "label": event["label"],
              "fault_group": str(event["group"]), "source_dataset": event["source_dataset"],
              "kind": kind, "memory_inertia": float(memory_inertia),
              "intervals": row["intervals"], "dimension": 10, "rank": row["rank"],
              "condition": row["condition"], "singular": row["singular"],
              "rhs_sse": row["rhs_sse"], "orthogonal_sse": row["orthogonal_sse"],
              "algebraic_norm": row["algebraic_norm"], "directions": row["directions"]}
    result["orthogonal_fraction"] = (row["orthogonal_sse"]/row["rhs_sse"]
                                     if row["rhs_sse"] > 0 else None)
    if "ball" in row:
        result["ball"] = row["ball"]
        result["radius_bound"] = radius
        result["ball_state_norm"] = (float(np.linalg.norm(row["ball"]["state"]))
                                      if row["ball"].get("state") is not None else None)
    if window is not None:
        result["window"] = window
    return result


def _inspect(events, blocks, budget, *, kind, radius, inertia=1.0, paths=None):
    rows = []
    for index, event in enumerate(events):
        path = event["x"] if paths is None else paths[index]
        result = inspect_momentum(path, event["p0"], event["equilibrium"], blocks, DT,
                                  budget.spend, radius=radius, memory_inertia=inertia)
        rows.append(_decorate(result, event, kind=kind, memory_inertia=inertia,
                              radius=radius))
    return rows


def _matched_windows(events, blocks, plan, budget):
    unstable = [event for event in events if event["label"] == "unstable"]
    if len(unstable) != UNSTABLE_COUNT:
        raise RuntimeError(f"expected exactly {UNSTABLE_COUNT} frozen train unstable cases")
    rows = []
    for event in unstable:
        available = event["x"][:min(MAX_FRAME, len(event["x"]))]
        margins = _row_margin(available, event["equilibrium"], event["mass"], event["omega_s"])
        end = _window_result(margins, min_end=ORIGIN_FRAME+1)["first_trigger_end_index"]
        if end is None:
            raise RuntimeError(f"no qualifying five-point truth window in train scope: {event['id']}")
        start = end-4
        window = {"start_index": start, "end_index": end,
                  "start_time_s": start*DT, "end_time_s": end*DT,
                  "margin": float(np.min(margins[start:end+1])), "truth_label": event["label"]}
        result = inspect_momentum(available[:end+1], event["p0"], event["equilibrium"], blocks, DT,
                                  budget.spend, radius=plan["memory_radius"])
        rows.append(_decorate(result, event, kind="matched_unstable_five_point",
                              radius=plan["memory_radius"], window=window))
    return rows


def _direction_report(rows, paired_groups=None):
    if not rows:
        return {}
    groups = _groups(rows)
    if paired_groups:
        groups.update(paired_groups)
    changes = direction_changes(rows, groups)
    return {"groups": {key: _summarize(value) for key, value in groups.items()},
            "train_fit_and_paired_application": changes}


def _verify_access_journal(path, expected):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows:
        raise RuntimeError("train access journal is empty")
    reads = Counter(row.get("source") for row in rows if row.get("stage") == "read_reserved")
    if reads != Counter({"static": 1, "development": 1}):
        raise RuntimeError("access journal contains reads outside static plus train development cache")
    decoded = [row.get("event_id") for row in rows if row.get("stage") == "decoded"]
    reserved = [row for row in rows if row.get("stage") == "cohort_reserved"]
    completed = [row for row in rows if row.get("stage") == "cohort_complete"]
    if (decoded != [event_id for event_id, _ in expected]
            or reserved != [{"stage": "cohort_reserved", "role": "train", "count": TRAIN_COUNT}]
            or completed != [{"stage": "cohort_complete", "role": "train", "count": TRAIN_COUNT}]
            or any(row.get("stage") == "cohort_failed" for row in rows)
            or any(row.get("role") not in (None, "train") for row in rows)):
        raise RuntimeError("access journal does not prove one complete train-only read")


def _origin_contrast(directory, events, blocks, radius, budget):
    rows = [{"id": event["id"], "label": event["label"], **origin_transfer(
        event["x"][:MAX_FRAME], event["p0"], event["equilibrium"], blocks, DT, budget.spend, radius=radius,
    )} for event in events]
    write_json(directory/"origin-transfer.json", rows)
    status = Counter(row["status"] for row in rows)
    groups = {}
    for name in ("train", "stable", "unstable"):
        matching = rows if name == "train" else [row for row in rows if row["label"] == name]
        solved = [row for row in matching if row["status"] == "solved"]
        sse = {method: math.fsum(row["future_sse"][method] for row in solved)
               for method in ("unconstrained", "constrained")}
        ratio = (sse["unconstrained"]/sse["constrained"]
                 if len(solved) == len(matching) and sse["constrained"] > 0. else None)
        groups[name] = {"allocated": len(matching), "solved": len(solved),
                        "sse": sse if len(solved) == len(matching) else None,
                        "ratio": ratio, "active": sum(row["constraint_active"] for row in solved),
                        "max_sampling_difference": max((row["sampling_difference"] for row in solved), default=None),
                        "sampling_qualified": sum(row["sampling_difference"] <= .05 for row in solved)}
    passed = (status == {"solved": len(rows)} and groups["train"]["sampling_qualified"] == len(rows)
              and all(row["ratio"] is not None and row["ratio"] <= .8 for row in groups.values()))
    return {"role": "train", "allocated": len(rows), "engineering_pass": True, "scientific_pass": passed,
            "statuses": dict(status), "memory_radius": radius, "groups": groups}


def _angle_fit(directory, events, blocks, radius, budget):
    rows = []
    for event in events:
        path = event["x"][:MAX_FRAME]
        result = angle_momentum(path, event["p0"], event["equilibrium"], blocks, DT,
                                budget.spend, check_quadrature=True)
        rows.append({"id": event["id"], "role": "train", "label": event["label"],
                     "fault_group": str(event["group"]), "intervals": len(path)-1, **result})
    write_json(directory/"angle-fit-rows.json", rows)
    budget.spend("fits", 2)
    fit = fit_train_damping(rows)
    gap = max(row["quadrature_gap"] for row in rows)
    valid = fit is not None and gap <= 1e-10
    groups = {"train": rows} | {label: [row for row in rows if row["label"] == label]
                               for label in ("stable", "unstable")}
    residuals = {name: damping_sse(group, fit) for name, group in groups.items()} if valid else None
    return {"role": "train", "allocated": len(rows), "memory_radius": radius,
            "engineering_pass": valid, "fit_supported": valid, "train_fits": fit if valid else None,
            "scientific_pass": valid and all(row["angle"] < row["original"] for row in residuals.values()),
            "quadrature_gap": gap, "full_rank": sum(row["rank"] == 10 for row in rows),
            "retained_intervals": sum(row["intervals"] for row in rows), "residuals": residuals}


def _spatial_fit(directory, events, blocks, radius, budget):
    budget.spend("numpy_damping_roots")
    root = spatial_damping_root(blocks)
    rows = []
    for event in events:
        path = event["x"][:MAX_FRAME]
        row = {"id": event["id"], "role": "train", "label": event["label"], "intervals": len(path)-1}
        try:
            result = spatial_momentum(path, event["p0"], event["equilibrium"], blocks, DT, budget.spend, root=root)
            row.update(result, qualified=True)
        except (FloatingPointError, np.linalg.LinAlgError) as error:
            row.update(qualified=False, reason=str(error))
        rows.append(row)
    write_json(directory/"spatial-fit-rows.json", rows)
    complete = all(row["qualified"] for row in rows)
    fit, gap = None, None
    if complete:
        budget.spend("fits")
        fit = fit_spatial_train(rows)
        gap = max(row["quadrature_gap"] for row in rows)
    valid = fit is not None and gap <= 1e-10
    groups = {"train": rows} | {label: [row for row in rows if row["label"] == label] for label in ("stable", "unstable")}
    residuals = {name: {"events": len(group), "global": spatial_sse(group, 0.),
                        "spatial": spatial_sse(group, fit["value"]), "trace": sum(row["trace_sse"] for row in group)}
                 for name, group in groups.items()} if valid else None
    return {"role": "train", "allocated": len(rows), "memory_radius": radius,
            "engineering_pass": valid, "fit_supported": valid, "train_fit": fit if valid else None,
            "scientific_pass": valid and all(row["spatial"] < row["global"] for row in residuals.values()),
            "quadrature_gap": gap, "full_rank": sum(row.get("rank") == 10 for row in rows),
            "retained_intervals": sum(row["intervals"] for row in rows), "residuals": residuals}


@torch.no_grad()
def _causal_memory(events, model, plan, budget, *, at_origin=False, full_history=False):
    steps = (.0025, .00125)
    shape = (11, 30) if full_history else (10,)
    starts = np.full((len(events), len(steps), *shape), np.nan)

    def estimate(indices, step, slot):
        output = predict(model, [events[i] for i in indices], plan["history_scale"], step, budget.spend,
                         count=0, memory_radius=plan["memory_radius"], prior_weight=1e-6)
        state = output["history"] if full_history else output["hidden_start"]
        if at_origin and not full_history:
            state = output["history"][:, -1, 20:]
        starts[indices, slot] = state.cpu().numpy()

    errors = []
    for slot, step in enumerate(steps):
        for offset in range(0, len(events), 32):
            indices = list(range(offset, min(offset+32, len(events))))
            try:
                estimate(indices, step, slot)
            except NUMERICAL_ERRORS as error:
                errors.append({"step": step, "batch_offset": offset, "error": str(error)})
                for i in indices:
                    try:
                        estimate([i], step, slot)
                    except NUMERICAL_ERRORS as local:
                        errors.append({"step": step, "id": events[i]["id"], "error": str(local)})
    return starts, errors


def _relative_gap(left, right):
    return float(np.max(np.abs(left-right)/np.maximum(1., np.abs(right))))


def _separation_window(event, path):
    margins = _row_margin(path, event["equilibrium"], event["mass"], event["omega_s"])
    witness = _window_result(margins, min_end=ORIGIN_FRAME+4)["first_trigger_end_index"]
    if witness is None:
        return None
    angle = path[witness-4, :10]+event["equilibrium"]
    high, low = int(np.argmax(angle)), int(np.argmin(angle))
    relative = path[ORIGIN_FRAME:, 10+high]-path[ORIGIN_FRAME:, 10+low]
    outgoing = np.flatnonzero(relative > 0.)
    start = int(outgoing[0]) if len(outgoing) else None
    crossing = np.flatnonzero(relative[start+1:] <= 0.) if start is not None else []
    end = start+1+int(crossing[0]) if len(crossing) else None
    return {"machines": [high, low], "witness_start": witness-4, "witness_end": witness,
            "outward_start": start, "outward_end": end}


def _work_window(bank, start, stop, mass, machines=None):
    impulses, work = bank["impulses"], bank["work"]
    piece = slice(start, stop)
    impulse = {key: values[piece].sum(0) for key, values in impulses.items()}
    impulse["residual"] = bank["momentum_residual"][piece].sum(0)
    total_work = {key: float(values[piece].sum()) for key, values in work.items()}
    total_work["residual"] = float(bank["kinetic_residual"][piece].sum())
    absolute = {key: float(np.abs(values[piece]).sum()) for key, values in work.items()}
    row = {"intervals": stop-start, "impulses": {key: value.tolist() for key, value in impulse.items()},
           "work": total_work, "absolute_work": absolute,
           "hidden_energy": [float(bank["hidden_energy"][i]) for i in (start, stop)],
           "memory_dissipation": float(bank["memory_dissipation"][piece].sum())}
    if machines is not None:
        high, low = machines
        row["separation_increment"] = {key: float(value[high]/mass[high]-value[low]/mass[low])
                                       for key, value in impulse.items()}
    return row


@torch.no_grad()
def _momentum_work(directory, events, model, blocks, plan, budget):
    model.set_angle_damping(2./3., spatial_mix=1.)
    budget.spend("numpy_damping_roots")
    root = spatial_damping_root(blocks)
    histories, errors = _causal_memory(events, model, plan, budget, full_history=True)
    mass = blocks[0]
    rows, stored, reset = [], {}, []
    names = ("applied", "restoring", "damping", "initial_memory", "driven_memory")
    for event, (coarse, fine) in zip(events, histories, strict=True):
        path = event["x"][:MAX_FRAME]
        row = {"id": event["id"], "label": event["label"], "fault_group": str(event["group"]),
               "qualified": False}
        rows.append(row)
        if not np.isfinite([coarse, fine]).all():
            row["failure"] = "missing causal history solution"
            continue
        observed = fine[-1].copy()
        observed[:20] = path[ORIGIN_FRAME]
        budget.spend("storage_calls")
        storage = model.storage(model._tensor(np.stack((fine[-1], observed))), model._tensor(event["equilibrium"]))
        delta = observed[:20]-fine[-1, :20]
        dk = .5*float(mass @ (observed[10:20]**2-fine[-1, 10:20]**2))
        row.update(history_nrmse=float(np.sqrt(np.mean(((fine[:, :20]-path[:11])/plan["history_scale"])**2))),
                   history_hidden_norm=float(np.linalg.norm(fine[0, 20:])),
                   reset={"angle": delta[:10].tolist(), "velocity": delta[10:].tolist(),
                          "momentum": (mass*delta[10:]).tolist(), "kinetic": dk,
                          "potential": float(storage.diff()[0])-dk, "storage": float(storage.diff()[0])})
        reset.append(row["history_nrmse"])
        witness = _separation_window(event, path)
        row["separation"] = witness
        controls = {"history": (path[:11], 0), "observed_future": (path[10:], 10)}
        gaps = {"quadrature_max": 0., "memory_balance_max": 0., "balance_identity_max": 0.}
        row["gaps"], row["windows"], row["initializer_sensitivity"] = gaps, {}, {}
        if len(path) <= ORIGIN_FRAME+1:
            row["failure"] = "no observed interval after forecast origin"
            continue
        try:
            for label, (segment, frame) in controls.items():
                intervals = len(segment)-1
                accounts = []
                for points, start in ((4, fine), (8, fine), (8, coarse)):
                    budget.spend("work_gauss_intervals", intervals)
                    budget.spend("work_gauss_nodes", points*intervals)
                    accounts.append(account_path(segment, event["p0"], event["equilibrium"], blocks, DT,
                                                 start[frame, 20:], root, points))
                quad, bank, alternate = accounts
                gaps["quadrature_max"] = max(gaps["quadrature_max"], quad["memory_impulse_quadrature_gap"],
                                             bank["memory_impulse_quadrature_gap"])
                for key in ("impulses", "work"):
                    gaps["quadrature_max"] = max(gaps["quadrature_max"], *(
                        _relative_gap(quad[key][name], bank[key][name]) for name in names))
                change = np.diff(segment[:, 10:], axis=0)
                kinetic = .5*np.diff(segment[:, 10:]**2, axis=0)*mass
                gaps["balance_identity_max"] = max(gaps["balance_identity_max"],
                    _relative_gap(sum(bank["impulses"].values())+bank["momentum_residual"], change*mass),
                    _relative_gap(sum(bank["work"].values())+bank["kinetic_residual"], kinetic))
                gaps["memory_balance_max"] = max(gaps["memory_balance_max"],
                    _relative_gap(bank["memory_balance_residual"], np.zeros(intervals)))
                row["initializer_sensitivity"][label] = _relative_gap(
                    alternate["impulses"]["initial_memory"], bank["impulses"]["initial_memory"])
                velocity_integral = .5*DT*(segment[1:, 10:]+segment[:-1, 10:])
                kinematic = np.diff(segment[:, :10], axis=0)-velocity_integral
                budget.spend("storage_calls")
                potential = model.storage(model._tensor(np.pad(segment[:, :20], ((0, 0), (0, 10)))),
                                          model._tensor(event["equilibrium"])).cpu().numpy()
                potential -= .5*(segment[:, 10:]**2 @ mass)
                arrays = {"impulses": np.stack([bank["impulses"][name] for name in names], axis=1),
                          "work": np.stack([bank["work"][name] for name in names], axis=1),
                          "observed_acceleration": change/DT, "momentum_residual": bank["momentum_residual"],
                          "kinetic_residual": bank["kinetic_residual"], "hidden_energy": bank["hidden_energy"],
                          "memory_dissipation": bank["memory_dissipation"], "kinematic_residual": kinematic,
                          "restoring_work_plus_potential_change": bank["work"]["restoring"].sum(1)+np.diff(potential)}
                if not all(np.isfinite(value).all() for value in arrays.values()):
                    raise FloatingPointError("nonfinite momentum or work ledger")
                stored.update({f"{len(rows)-1}_{label}_{key}": value for key, value in arrays.items()})
                row["windows"][label] = _work_window(bank, 0, intervals, mass)
                if label == "observed_future":
                    row["windows"]["early_500ms"] = _work_window(bank, 0, min(50, intervals), mass)
                    if witness is not None:
                        stop = witness["witness_end"]-ORIGIN_FRAME
                        row["windows"]["through_witness"] = _work_window(bank, 0, stop, mass, witness["machines"])
                        if witness["outward_end"] is not None:
                            row["windows"]["first_outward_swing"] = _work_window(
                                bank, witness["outward_start"], witness["outward_end"], mass, witness["machines"])
            row["qualified"] = all(gaps[key] <= limit for key, limit in plan["gates"].items())
        except FloatingPointError as error:
            row["failure"] = str(error)

    cutoff = float(np.median(reset)) if reset else None
    for row in rows:
        value = row.get("history_nrmse")
        row["history_residual_group"] = None if value is None else ("higher" if value > cutoff else "lower")
    stable = [i for i, row in enumerate(rows) if row["label"] == "stable" and row.get("qualified")]
    speed = [float(np.sqrt(mass @ event["x"][ORIGIN_FRAME, 10:]**2)) for event in events]
    matches = []
    for i, row in enumerate(rows):
        if row["label"] != "unstable":
            continue
        same_fault = [j for j in stable if rows[j]["fault_group"] == row["fault_group"]]
        partner = min(same_fault, key=lambda j: (abs(speed[i]-speed[j]), rows[j]["id"]), default=None)
        match = {"unstable": row["id"], "stable": rows[partner]["id"] if partner is not None else None}
        if partner is not None and row.get("qualified") and "through_witness" in row["windows"]:
            high, low = row["separation"]["machines"]
            target = row["windows"]["through_witness"]
            impulses = stored[f"{partner}_observed_future_impulses"]
            stop = target["intervals"]
            if len(impulses) >= stop:
                vector = impulses[:stop].sum(0)
                match["unstable_increment"] = target["separation_increment"]
                match["stable_increment"] = dict(zip(names, (vector[:, high]/mass[high]-vector[:, low]/mass[low]).tolist(),
                                                     strict=True))
                residual = stored[f"{partner}_observed_future_momentum_residual"][:stop].sum(0)
                match["stable_increment"]["residual"] = float(residual[high]/mass[high]-residual[low]/mass[low])
                match["intervals"] = stop
        matches.append(match)
    write_json(directory/"momentum-work-rows.json", rows)
    write_json(directory/"history-estimation-errors.json", errors)
    stored["causal_history"] = histories
    np.savez_compressed(directory/"momentum-work-intervals.npz", **stored)
    qualified = sum(row["qualified"] for row in rows)
    return {"role": "train", "allocated": len(rows), "qualified": qualified,
            "engineering_pass": qualified == TRAIN_COUNT, "scientific_pass": False,
            "history_nrmse_median": cutoff, "matched_stable": matches, "component_order": list(names)}


def _fixed_origin_diagnosis(directory, events, model, blocks, plan, budget):
    modal = plan["contrast"] == "modal_damping"
    model.set_angle_damping(2./3., spatial_mix=1. if modal else 0.)
    if modal:
        budget.spend("numpy_damping_roots")
        budget.spend("reference_eigh")
        probe = modal_probe(blocks)
        assess = partial(modal_residuals, probe=probe)
        metadata = {"modal_probe": {key: value.tolist() if isinstance(value, np.ndarray) else value
                                    for key, value in probe.items() if key != "root"}}
        row_filename, stable_limit = "modal-damping-rows.json", .99
    else:
        assess = angle_scale_residuals
        metadata = {"angle_scales_rad": [1., 1.25]}
        row_filename, stable_limit = "angle-scale-rows.json", .95
    starts, errors = _causal_memory(events, model, plan, budget)

    rows = []
    for i, event in enumerate(events):
        row = {"id": event["id"], "label": event["label"]}
        path = event["x"][:MAX_FRAME]
        if len(path) > 11 and np.isfinite(starts[i]).all():
            try:
                result = assess(path, event["p0"], event["equilibrium"], blocks, DT, budget.spend, starts[i])
                ok = result["rank"] == 10 and max(result["quadrature_gap"].values()) <= 1e-10
                row.update(result, qualified=ok)
            except (FloatingPointError, np.linalg.LinAlgError) as error:
                row.update(qualified=False, reason=str(error))
        else:
            row.update(qualified=False, reason="missing history estimate or future interval")
        rows.append(row)
    write_json(directory/row_filename, rows)
    write_json(directory/"history-estimation-errors.json", errors)
    groups = {}
    for role in ("train", "stable", "unstable"):
        selected = rows if role == "train" else [row for row in rows if row["label"] == role]
        valid = [row for row in selected if row["qualified"]]
        ratios = {}
        for observer in ("0.0025", "0.00125", "free"):
            pair = ([math.fsum(row["sse"][observer][i] for row in valid) for i in range(2)]
                    if len(selected) == len(valid) else None)
            ratios[observer] = {"sse": pair, "ratio": pair[1]/pair[0] if pair and pair[0] > 0. else None}
        groups[role] = {"allocated": len(selected), "qualified": len(valid), "effects": ratios}
    engineering = groups["train"]["qualified"] == TRAIN_COUNT
    science = engineering and all(entry["ratio"] is not None and entry["ratio"] <= limit
                                  for role, limit in (("train", 1.01), ("stable", stable_limit), ("unstable", 1.01))
                                  for entry in groups[role]["effects"].values())
    return {"role": "train", "allocated": len(events), "engineering_pass": engineering,
            "scientific_pass": science, "groups": groups, "memory_radius": plan["memory_radius"],
            "rho": 2./3., "trajectory_lengths": [len(event["x"]) for event in events]} | metadata


def _restoring_fit(directory, events, blocks, radius, budget):
    budget.spend("numpy_damping_roots")
    root = spatial_damping_root(blocks)
    rows = []
    for event in events:
        path = event["x"][:MAX_FRAME]
        row = {"id": event["id"], "role": "train", "label": event["label"], "intervals": len(path)-1}
        try:
            row.update(restoring_momentum(path, event["p0"], event["equilibrium"], blocks, DT, budget.spend,
                                          root=root))
        except (FloatingPointError, np.linalg.LinAlgError) as error:
            row.update(qualified=False, reason=str(error))
        rows.append(row)
    write_json(directory/"restoring-fit-rows.json", rows)
    fit = None
    if all(row["qualified"] for row in rows):
        budget.spend("fits", 2)
        fit = fit_restoring_train(rows)
    groups = {"train": rows} | {key: [row for row in rows if row["label"] == key] for key in ("stable", "unstable")}
    qualified = fit is not None
    residuals = ({key: {"events": len(group), **restoring_sse(group, fit)} for key, group in groups.items()}
                 if qualified else None)
    science = bool(qualified and residuals["train"]["shape"] <= .95*residuals["train"]["base"]
                   and all(row["shape"] < row["base"] and row["shape"] <= row["scale"] for row in residuals.values()))
    return {"role": "train", "allocated": len(rows), "memory_radius": radius,
            "engineering_pass": qualified, "scientific_pass": science, "fit_supported": qualified, "train_fit": fit,
            "quadrature_gap": max((row["quadrature_gap"] for row in rows if "quadrature_gap" in row), default=None),
            "full_rank": sum(row.get("rank") == 10 for row in rows),
            "retained_intervals": sum(row["intervals"] for row in rows), "residuals": residuals}


def _dynamic_coupling_diagnosis(directory, events, model, blocks, plan, budget):
    settings, limits = plan["coupling"], plan["gates"]
    budget.spend("numpy_damping_roots")
    root = spatial_damping_root(blocks)
    states, errors = {}, {}
    for anchor, rho in zip(ANCHORS, (0., 2./3.), strict=True):
        model.set_angle_damping(rho, spatial_mix=1.)
        states[anchor], errors[anchor] = _causal_memory(events, model, plan, budget, at_origin=True)
    rows = []
    for index, event in enumerate(events):
        path = event["x"][:MAX_FRAME]
        origins = np.stack([states[anchor][index] for anchor in ANCHORS])
        row = {"id": event["id"], "label": event["label"], "intervals": len(path)-1}
        if len(path) > ORIGIN_FRAME+1 and np.isfinite(origins).all():
            try:
                result = coupling_residuals(path, event["p0"], event["equilibrium"], blocks, DT, budget.spend,
                                            origins, root=root, **settings)
                row.update(result, qualified=result["rank"] == 10
                           and result.get("spatial_geometry_qualified", True)
                           and all(result["gaps"][key] <= value for key, value in limits["gaps"].items()))
            except (FloatingPointError, np.linalg.LinAlgError) as error:
                row.update(qualified=False, reason=str(error))
        else:
            row.update(qualified=False, reason="missing causal memory origin or future intervals")
        rows.append(row)
    write_json(directory/"dynamic-coupling-rows.json", rows)
    write_json(directory/"history-estimation-errors.json", errors)
    groups = {}
    for label in ("train", "stable", "unstable"):
        group = rows if label == "train" else [row for row in rows if row["label"] == label]
        valid = [row for row in group if row["qualified"]]
        effects = {}
        for anchor in ANCHORS:
            effects[anchor] = {}
            for start in STARTS:
                pair = ([math.fsum(row["sse"][anchor][start][i] for row in valid) for i in range(2)]
                        if group and len(valid) == len(group) else None)
                effects[anchor][start] = {"sse": pair, "ratio": pair[1]/pair[0] if pair and pair[0] > 0 else None}
        groups[label] = {"allocated": len(group), "qualified": len(valid), "effects": effects}
    engineering = groups["train"]["qualified"] == TRAIN_COUNT
    checks = {label: all(row["ratio"] is not None and row["ratio"] <= limits["ratios"][label]
                         for starts in group["effects"].values() for row in starts.values())
              for label, group in groups.items()}
    return {"role": "train", "allocated": len(events), "engineering_pass": engineering,
            "scientific_pass": engineering and all(checks.values()), "checks": checks, "groups": groups,
            "coupling": settings | {"stiffness": float(blocks[2].sum())}, "memory_radius": plan["memory_radius"],
            "retained_intervals": sum(row["intervals"] for row in rows),
            "trajectory_lengths": [len(event["x"]) for event in events]}


def execute(root, directory, plan, budget):
    """Audit all official train cases using saved causal inputs; never forecasts."""
    root, directory = Path(root).resolve(), Path(directory)
    sources = plan["sources"]
    if sources.get("allowed_roles") != ["train"]:
        raise PermissionError("momentum audit is restricted to the official train role")
    roster = sources["rosters"]["train"]
    contrast = plan.get("contrast", "force_blocks")
    fixed_origin = ("angle_scale", "modal_damping", "dynamic_coupling", "spatial_coupling", "momentum_work")
    contrasts = ("force_blocks", "origin_transfer", "angle_damping_fit", "spatial_damping_fit", "restoring_shape_fit",
                 *fixed_origin)
    if len(roster) != TRAIN_COUNT or contrast not in contrasts:
        raise ValueError("plan must bind the complete train roster and a supported contrast")
    factors = [float(x) for x in plan.get("inertias", [])]
    if contrast == "force_blocks" and (len(set(factors)) != 4 or len(factors) != 4
            or any(not math.isfinite(x) or x <= 0 or x == 1. for x in factors)):
        raise ValueError("inertia contrasts must be four distinct positive non-unit factors")
    radius = float(plan["memory_radius"])
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("the frozen original T0M1 memory radius must be positive")

    journal = AppendOnlyAuditJournal(directory/"access.jsonl")
    try:
        budget.spend("checkpoint_loads")
        backbone = LearnedNonlinearModel.load(verified(root, plan["backbone"])).closure
        model = make_model(backbone, "T0M1", train_scales=True, device="cuda" if contrast in fixed_origin else "cpu")
        budget.spend("checkpoint_loads")
        load_snapshot(model, verified(root, plan["checkpoint"]))
        if model.topology_dimension != 0 or model.memory_dimension != 10:
            raise RuntimeError("momentum diagnostic requires the frozen T0M1 model")
        blocks = _model_blocks(model)
        if not np.array_equal(blocks[0], _array(backbone.mass)):
            raise RuntimeError("frozen T0M1 mass differs from its backbone")

        reader = Task3FullReader(root, sources, journal)
        events = reader.load("train")
        if len(events) != TRAIN_COUNT:
            raise RuntimeError(f"expected the complete train cohort ({TRAIN_COUNT}), got {len(events)}")
        observed = [(event["id"], event["label"]) for event in events]
        expected = [(row["id"], row["label"]) for row in roster]
        counts = Counter(label for _, label in observed)
        if observed != expected or counts != Counter(stable=311, unstable=35):
            raise RuntimeError("reader output differs from frozen complete train IDs or labels")
        lengths = [len(event["x"]) for event in events]
        if any(length < 11 for length in lengths):
            raise RuntimeError("a train trajectory does not contain the full 100 ms history")
        _verify_access_journal(directory/"access.jsonl", expected)
        if not np.allclose(reader.static["mass"], blocks[0], rtol=0, atol=1e-12):
            raise RuntimeError("TASK3 train inertia differs from the frozen original T0M1 model")

        if contrast == "origin_transfer":
            return _origin_contrast(directory, events, blocks, radius, budget) | {"trajectory_lengths": lengths}
        if contrast == "angle_damping_fit":
            return _angle_fit(directory, events, blocks, radius, budget)
        if contrast in ("dynamic_coupling", "spatial_coupling"):
            return _dynamic_coupling_diagnosis(directory, events, model, blocks, plan, budget)
        if contrast == "momentum_work":
            return _momentum_work(directory, events, model, blocks, plan, budget)
        if contrast in fixed_origin:
            return _fixed_origin_diagnosis(directory, events, model, blocks, plan, budget)
        if contrast == "spatial_damping_fit":
            return _spatial_fit(directory, events, blocks, radius, budget)
        if contrast == "restoring_shape_fit":
            return _restoring_fit(directory, events, blocks, radius, budget)

        history_paths = [event["x"][:11] for event in events]
        full_paths = [event["x"][:min(MAX_FRAME, len(event["x"]))] for event in events]
        history = _inspect(events, blocks, budget, kind="history_0_to_100ms",
                           radius=radius, paths=history_paths)
        full = _inspect(events, blocks, budget, kind="full_available_to_3s",
                        radius=radius, paths=full_paths)
        matched = _matched_windows(events, blocks, plan, budget)

        inertia_results = {}
        for factor in factors:
            rows = _inspect(events, blocks, budget,
                            kind="full_available_inertia_contrast", radius=None, inertia=factor,
                            paths=full_paths)
            inertia_results[str(factor)] = rows

        for filename, rows in (("history-100ms.json", history),
                               ("full-3s.json", full),
                               ("matched-unstable-windows.json", matched)):
            write_json(directory/filename, rows)
        for factor, rows in inertia_results.items():
            write_json(directory/f"inertia-{factor}.json", rows)

        matched_pair = {"matched_unstable_five_point": matched}
        nominal_directions = _direction_report(full, matched_pair)
        inertia_groups = {factor: {key: _summarize(group) for key, group in _groups(rows).items()}
                          for factor, rows in inertia_results.items()}
        by_scope = {
            "history_0_to_100ms": {key: _summarize(value) for key, value in _groups(history).items()},
            "full_available_to_3s": nominal_directions["groups"],
            "matched_unstable_five_point": {"train/unstable": _summarize(matched)},
            "inertia_contrasts": inertia_groups,
        }
        changes = nominal_directions["train_fit_and_paired_application"]
        stable, unstable, matched_label = "label/stable", "label/unstable", "matched_unstable_five_point"
        floor = plan["minimum_train_improvement"]
        routes = []
        for name in ("stiffness", "damping"):
            candidate = changes[name]
            ratios = {group: candidate["by_group"][group]["ratio"] for group in ("train", stable, unstable, matched_label)}
            if (candidate["delta_multiplier"] is not None and 1.+candidate["delta_multiplier"] > 0
                    and all(value is not None and value < 1. for value in ratios.values())
                    and ratios["train"] <= 1.-floor):
                routes.append({"direction": name, "multiplier": 1.+candidate["delta_multiplier"],
                               "train_improvement": 1.-ratios["train"]})
        nominal_groups = nominal_directions["groups"]
        for factor, groups in inertia_groups.items():
            ratios = {name: groups[name]["orthogonal_sse"]/nominal_groups[name]["orthogonal_sse"]
                      for name in ("train", stable, unstable) if nominal_groups[name]["orthogonal_sse"] > 0}
            if len(ratios) == 3 and ratios["train"] <= 1.-floor and ratios[stable] < 1. and ratios[unstable] < 1.:
                routes.append({"direction": "memory_inertia", "multiplier": float(factor),
                               "train_improvement": 1.-ratios["train"]})
        routes.sort(key=lambda row: (-row["train_improvement"], row["direction"], row["multiplier"]))
        summary = {
            "status": "PASS", "engineering_pass": True, "scientific_pass": True,
            "role": "train", "allocated": len(events),
            "class_counts": dict(counts), "ids": [event["id"] for event in events],
            "trajectory_lengths": lengths,
            "observed_path_limit_frames": MAX_FRAME,
            "matched_unstable_windows": len(matched),
            "memory_radius": radius, "inertia_factors": factors,
            "checkpoint_bindings": {key: {"path": plan[key]["path"], "sha256": plan[key]["sha256"]}
                                     for key in ("backbone", "checkpoint")},
            "operation_counts": dict(budget.counts),
            "by_scope_and_group": by_scope,
            "nominal_force_direction_changes": nominal_directions["train_fit_and_paired_application"],
            "supported_directions": routes,
        }
        return summary
    finally:
        journal.close()
