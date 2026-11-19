"""Zero-update M1 diagnostics on the original frozen equation."""

from __future__ import annotations

import time
from functools import partial

import numpy as np
import torch

from kron_reduce.evaluation.constant_c_observability import write_json
from kron_reduce.evaluation.first_swing_geometry import gradient_probe
from kron_reduce.evaluation.first_swing_loss import observed_velocity_increment
from kron_reduce.evaluation.kiki_first_swing import STEPS, _witnesses
from kron_reduce.evaluation.kiki_machine_local_full import evaluate, inputs
from kron_reduce.evaluation.kiki_topology_momentum import ORIGIN_FRAME
from kron_reduce.evaluation.machine_local_clean_learning import NUMERICAL_ERRORS
from kron_reduce.models.machine_local_topology import predict_from_topology_history
from kron_reduce.physics.coordinates import coi_project


@torch.no_grad()
def zero_history_reference(model, events, scale, spend, *, batch_size, floor):
    if not events or any(event["role"] != "train" for event in events):
        raise ValueError("the residual reference requires the frozen train roster")
    residuals = {}
    n = model.n_nodes
    for offset in range(0, len(events), batch_size):
        batch = events[offset:offset+batch_size]
        q, v, power, equilibrium, scale_tensor = map(model._tensor, inputs(batch, scale))
        initial = torch.cat((q[:, 0], v[:, 0], q.new_zeros((len(batch), model.latent_dimension))), dim=1)
        history = model.rollout(initial, power, equilibrium, 10, max_step=STEPS[0], spend=spend)
        truth = torch.cat((q[:, 1:], v[:, 1:]), dim=-1)
        values = ((truth-history[:, 1:, :2*n])/scale_tensor).square().mean(dim=(1, 2)).sqrt()
        if not torch.isfinite(values).all():
            raise FloatingPointError("nonfinite zero-memory history reference")
        residuals.update(zip((event["id"] for event in batch), values.tolist(), strict=True))
    median = float(np.median(list(residuals.values())))
    return {"step": STEPS[0], "median": median, "reference": max(floor, median), "zero_history_nrmse": residuals}


def speed_peak_gap(event, forecast):
    """A signed first-500-ms peak deficit on the common observed grid."""
    count = min(50, len(event["x"])-11)
    mass = event["mass"]
    n = len(mass)
    truth = event["x"][11:11+count, n:2*n]
    prediction = forecast[1:1+count, n:2*n]
    if count <= 0 or prediction.shape != truth.shape or not np.isfinite(prediction).all():
        return None
    true_peak = float(np.max(np.abs(coi_project(truth, mass))))
    predicted_peak = float(np.max(np.abs(coi_project(prediction, mass))))
    return {"points": count, "true_peak_rad_s": true_peak, "predicted_peak_rad_s": predicted_peak,
            "gap_rad_s": true_peak-predicted_peak}


def _adaptive_predict(model, events, step, scales, budget, settings, reference, histories):
    result = predict_from_topology_history(model, *inputs(events, scales["history"]), 800,
        max_step=step, spend=budget.spend, memory_radius=scales["memory_radius"],
        prior_weight=settings["prior_weight"], adaptive_prior_reference=reference)
    for event, residual, prior, hidden in zip(events, result["zero_history_nrmse"].tolist(),
            result["adaptive_prior_weight"].tolist(), result["history"][:, -1, 20:].cpu().numpy(), strict=True):
        histories[step][event["id"]] = (residual, prior, hidden)
    return {**result, "gn_singular": result["singular"][None]}


@torch.no_grad()
def _witness_audit(model, events, histories, baseline, budget):
    witnesses, missing = _witnesses(events)
    if missing != baseline["without_witness"]:
        raise RuntimeError("the frozen M1 train witness roster changed")
    output = {}
    for step in STEPS:
        fixed = baseline["comparisons"]["first_swing"][str(step)]["rows"]
        scored = {event["id"]: None for event, _ in witnesses}
        numerical_error = None
        present = [(e, w) for e, w in witnesses if e["id"] in histories[step]]
        if present:
            lengths = np.array([w["witness_end"]-ORIGIN_FRAME for _, w in present])
            count = int(lengths.max())
            paths = [e["x"][ORIGIN_FRAME:w["witness_end"]+1] for e, w in present]
            path = model._tensor(np.stack([np.pad(p, ((0, count+1-len(p)), (0, 0)), mode="edge") for p in paths]))
            starts = model._tensor(np.stack([histories[step][e["id"]][2] for e, _ in present]))
            _, _, power, equilibrium, _ = inputs([e for e, _ in present], np.ones(20))
            try:
                impulse = observed_velocity_increment(model, path, starts, model._tensor(power),
                    model._tensor(equilibrium), lengths, step=step, spend=budget.spend)
                residuals = (path[..., 10:]-path[:, :1, 10:]-impulse).cpu().numpy()
                for i, (e, w) in enumerate(present):
                    high, low = w["machines"]
                    scored[e["id"]] = float(residuals[i, lengths[i], high]-residuals[i, lengths[i], low])
            except NUMERICAL_ERRORS as error:
                numerical_error = str(error)
        rows = []
        for (event, window), old in zip(witnesses, fixed, strict=True):
            if event["id"] != old["id"] or any(old[key] != value for key, value in window.items()):
                raise RuntimeError("paired witness windows disagree with the frozen run")
            rows.append({"id": event["id"], **window, "fixed": old["signed_residual_rad_s"],
                         "adaptive": scored[event["id"]]})
        pairs = [(r["fixed"], r["adaptive"]) for r in rows if r["adaptive"] is not None and r["fixed"] is not None]
        summary = {"qualified": len(pairs), "allocated": len(rows), "improved": sum(abs(a) < abs(f) for f, a in pairs),
                   "numerical_error": numerical_error}
        for index, mode in enumerate(("fixed", "adaptive")):
            values = [p[index] for p in pairs] if len(pairs) == len(rows) else []
            summary[mode] = {"signed_median": float(np.median(values)) if values else None,
                             "absolute_median": float(np.median(np.abs(values))) if values else None}
        summary["median_absolute_change"] = (float(np.median([abs(a)-abs(f) for f, a in pairs]))
                                             if len(pairs) == len(rows) else None)
        output[str(step)] = summary | {"rows": rows}
    return output


def _label(row):
    if not row["prospective_eligible"]:
        return "uncovered"
    if not row["qualified"]:
        return "numerical_failure"
    return row["predicted_label"] or "undecided"


def _classification_pairs(events, baseline, adaptive):
    pairs = {}
    for event, fixed, candidate in zip(events, baseline["stability_events"], adaptive["stability_events"], strict=True):
        if event["id"] != fixed["id"] or event["id"] != candidate["id"] or (
                event["label"] != fixed["truth_label"] or event["label"] != candidate["truth_label"]):
            raise RuntimeError("case labels or the paired roster changed")
        pairs[event["id"]] = {"truth": event["label"], "fixed": _label(fixed), "adaptive": _label(candidate)}
    return pairs


def _peak_audit(events, before, after):
    ids = [event["id"] for event in events]
    with np.load(before, allow_pickle=False) as fixed, np.load(after, allow_pickle=False) as adaptive:
        if fixed["ids"].tolist() != ids or adaptive["ids"].tolist() != ids:
            raise RuntimeError("saved val forecast rosters differ")
        rows = {e["id"]: {} for e in events if e["label"] == "unstable"}
        for key, step in zip(("fine", "refined"), STEPS, strict=True):
            for e, f, a in zip(events, fixed[key], adaptive[key], strict=True):
                if e["id"] in rows:
                    rows[e["id"]][str(step)] = {"fixed": speed_peak_gap(e, f), "adaptive": speed_peak_gap(e, a)}
    return rows


def _history_stats(result):
    stats = {}
    for step in STEPS:
        rows = [r for r in result["initialization"] if r["step"] == step]
        stats[str(step)] = {"qualified": len(rows),
            "median_posterior_history_nrmse": float(np.median([r["history_nrmse"] for r in rows])) if rows else None,
            "max_origin_energy": max((r["hidden_origin_energy"] for r in rows), default=None)}
    return stats


def execute(model, reader, directory, budget, settings, scales, plan, baseline):
    started = time.perf_counter()
    train = reader.load("train")
    if not np.allclose(reader.static["mass"], model.mass.detach().cpu().numpy(), rtol=0., atol=1e-12):
        raise RuntimeError("probe inertia differs from the frozen equation")
    model.requires_grad_(True)
    try:
        geometry = gradient_probe(model, train, scales, budget, settings)
    finally:
        model.requires_grad_(False)
    for i, name in enumerate(geometry["terms"]):
        reference = baseline["gradients"][name]
        if not (np.isclose(geometry["norms"][name], reference["norm"], rtol=1e-10, atol=1e-12)
                and np.isclose(geometry["cosines"][i][0], reference["cosine_to_rollout"], rtol=1e-10, atol=1e-12)):
            raise RuntimeError(f"probe A did not reproduce the signed M1 origin for {name}")
    write_json(directory/"gradient-probe.json", geometry)
    calibration = zero_history_reference(model, train, scales["history"], budget.spend,
        batch_size=settings["eval_batch"], floor=plan["r_ref_floor"])
    reference = calibration["reference"]
    write_json(directory/"train-reference.json", calibration)
    freeze = {**baseline["freeze"], "adaptive_prior_reference": reference, "reference_role": "train"}
    freeze_path = directory/"freeze.json"
    write_json(freeze_path, freeze)
    val = reader.load("val", freeze_path=freeze_path)

    histories = {step: {} for step in STEPS}
    predictor = partial(_adaptive_predict, model, scales=scales, budget=budget, settings=settings,
                        reference=reference, histories=histories)
    scores, cases = {}, {}
    for role, events in (("train", train), ("val", val)):
        adaptive = evaluate(model, events, scales["history"], budget, directory, role, settings, predictor=predictor)
        fixed = baseline["scores"][role]
        cases[role] = _classification_pairs(events, fixed, adaptive)
        keys = ("stability", "stable_complete", "unstable_prefix", "task3_six_term_score")
        scores[role] = {mode: {key: result[key] for key in keys} | {"history": _history_stats(result)}
                        for mode, result in (("fixed", fixed), ("adaptive", adaptive))}
    for event_id, expected in calibration["zero_history_nrmse"].items():
        result = histories[STEPS[0]].get(event_id)
        if result is not None and not np.isclose(result[0], expected, rtol=1e-10, atol=1e-12):
            raise RuntimeError("zero-start history residual drifted after freezing train calibration")
    write_json(directory/"classification-pairs.json", cases)
    write_json(directory/"adaptive-priors.json", {
        str(step): {key: {"zero_history_nrmse": row[0], "prior_weight": row[1]} for key, row in values.items()}
        for step, values in histories.items()})
    witness = _witness_audit(model, train, histories, baseline["witness"], budget)
    write_json(directory/"train-witnesses.json", witness)
    peak = _peak_audit(val, baseline["val_forecast"], directory/"val_predictions.npz")
    write_json(directory/"val-speed-peaks.json", peak)
    return {"r_ref": reference, "scores": scores, "baseline_missed_or_undecided": {
        role: {key: row for key, row in values.items()
               if row["truth"] == "unstable" and row["fixed"] in {"stable", "undecided"}}
        for role, values in cases.items()}, "witness": {
        step: {key: value for key, value in row.items() if key != "rows"} for step, row in witness.items()},
        "wall_seconds": time.perf_counter()-started}
