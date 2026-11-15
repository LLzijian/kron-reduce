"""Train-only observed-path weak-momentum audit; performs no model forecasts."""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from kron_reduce.data.durable_audit_ledger import AppendOnlyAuditJournal
from kron_reduce.data.task3_full import Task3FullReader
from kron_reduce.evaluation.constant_c_observability import write_json
from kron_reduce.evaluation.kiki_machine_local_r20_r21 import verified
from kron_reduce.evaluation.kiki_topology_bridge import load_snapshot, make_model
from kron_reduce.evaluation.topology_momentum import (
    direction_changes,
    inspect_momentum,
    origin_transfer,
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


def execute(root, directory, plan, budget):
    """Audit all official train cases using saved causal inputs; never forecasts."""
    root, directory = Path(root).resolve(), Path(directory)
    sources = plan["sources"]
    if sources.get("allowed_roles") != ["train"]:
        raise PermissionError("momentum audit is restricted to the official train role")
    roster = sources["rosters"]["train"]
    contrast = plan.get("contrast", "force_blocks")
    if len(roster) != TRAIN_COUNT or contrast not in ("force_blocks", "origin_transfer"):
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
        model = make_model(backbone, "T0M1", train_scales=True, device="cpu")
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
