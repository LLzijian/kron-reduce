"""Causal and observed-path scoring on the frozen discovery cohorts."""

from __future__ import annotations

import math
from itertools import product

import numpy as np
import torch

from kron_reduce.evaluation.constant_c_observability import write_json
from kron_reduce.evaluation.first_swing_balanced_fit import trajectory_groups
from kron_reduce.evaluation.first_swing_gn_decomposition import _distribution
from kron_reduce.evaluation.first_swing_loss import observed_velocity_increment
from kron_reduce.evaluation.first_swing_probes import speed_peak_gap
from kron_reduce.evaluation.kiki_first_swing import STEPS
from kron_reduce.evaluation.kiki_machine_local_full import inputs
from kron_reduce.evaluation.kiki_topology_momentum import ORIGIN_FRAME
from kron_reduce.evaluation.machine_local_clean_learning import NUMERICAL_ERRORS
from kron_reduce.evaluation.trajectory_stability import summarize
from kron_reduce.models.unified_discovery_m3 import MODULES, predict_m3


def _forecast(model, batch, scales, prior, budget, step, count):
    return predict_m3(model, *inputs(batch, scales["history"]), count, max_step=step, spend=budget.spend,
                      memory_radius=scales["memory_radius"], prior_weight=prior)


@torch.no_grad()
def score(model, events, scales, settings, budget, *, directory=None):
    """Score each step with the original TASK3 comparison and its paired stability gate."""
    from task3.c2.evaluation import evaluate as task3_score

    role = events[0]["role"]
    if role not in ("train", "val") or any(event["role"] != role for event in events):
        raise ValueError("discovery scoring is restricted to complete train or val")
    paths = {step: np.full((len(events), 801, 20), np.nan) for step in STEPS}
    origin_energies = {step: {} for step in STEPS}
    failures = []

    def forward(batch, offset, step):
        result = _forecast(model, batch, scales, settings["prior_weight"], budget, step, 800)
        paths[step][offset:offset+len(batch)] = result["forecast"][..., :20].cpu().numpy()
        energy = .5*result["history"][:, -1, 20:30].square().sum(dim=1)
        origin_energies[step].update(zip((e["id"] for e in batch), energy.tolist(), strict=True))

    for start in range(0, len(events), settings["eval_batch"]):
        batch = events[start:start+settings["eval_batch"]]
        for step in STEPS:
            try:
                forward(batch, start, step)
            except NUMERICAL_ERRORS as error:
                failures.append({"start": start, "count": len(batch), "step": step, "error": str(error)})
                for i, event in enumerate(batch):
                    try:
                        forward([event], start+i, step)
                    except NUMERICAL_ERRORS as individual:
                        failures.append({"id": event["id"], "step": step, "error": str(individual)})
    scoring = [e | {"x": e["raw_x"], "equilibrium_delta": e["equilibrium"],
                    "event_time_s": e.get("event_time_s")} for e in events]
    paired = {step: task3_score(paths[step], scoring, paths[other]) for step, other in
              zip(STEPS, reversed(STEPS), strict=True)}
    stepwise, records = {}, {}
    for step, other in zip(STEPS, reversed(STEPS), strict=True):
        result = paired[step]
        cases = []
        for event, current, refined in zip(events, result["classifications"], paired[other]["classifications"], strict=True):
            label = "unstable" if current["predicted_los"] else "stable" if current["window_settled"] else None
            same = (current["predicted_los"], current["window_settled"]) == (refined["predicted_los"], refined["window_settled"])
            qualified = current["qualified"] and refined["qualified"] and same
            cases.append({"id": event["id"], "truth_label": event["label"], "predicted_label": label,
                "qualified": qualified, "prospective_eligible": True, "numerical_failure": not qualified,
                "unsettled": qualified and label is None})
        stability = summarize(cases)
        for name, numerator, denominator in (("recall_on_resolved", stability["tp"], stability["tp"]+stability["fn"]),
            ("miss_rate_on_resolved", stability["fn"], stability["tp"]+stability["fn"]),
            ("false_alarm_rate_on_resolved", stability["fp"], stability["fp"]+stability["tn"])):
            stability[name] = numerator/denominator if denominator else None
        peaks = {event["id"]: speed_peak_gap(event, path) for event, path in zip(events, paths[step], strict=True)
                 if event["label"] == "unstable"}
        values = [row["gap_rad_s"] if row else None for row in peaks.values()]
        result["stability_events"] = cases
        stepwise[str(step)] = {"stability": stability, "stable_errors": trajectory_groups(result),
            "stable_complete": result["stable_complete"], "unstable_prefix": result["unstable_prefix"],
            "task3_six_term_score": result["selection_score"], "first_500ms_peaks": _distribution(values, len(values)),
            "max_origin_memory_energy": max(origin_energies[step].values(), default=None)}
        records[str(step)] = {"trajectory": result["records"], "classification": cases, "first_500ms_peaks": peaks,
                              "origin_memory_energy": origin_energies[step]}
    if directory is not None:
        write_json(directory/f"{role}.json", {"steps": stepwise, "cases": records, "failures": failures})
        with (directory/f"{role}_predictions.npz").open("xb") as stream:
            np.savez(stream, ids=np.array([e["id"] for e in events]), main=paths[STEPS[0]], refined=paths[STEPS[1]])
    return stepwise


def _paths(model, events, frames):
    count = int(frames.max())
    values = [e["x"][ORIGIN_FRAME:ORIGIN_FRAME+length+1] for e, length in zip(events, frames, strict=True)]
    return model._tensor(np.stack([np.pad(value, ((0, count+1-len(value)), (0, 0)), mode="edge") for value in values]))


@torch.no_grad()
def witness_scores(model, rows, scales, settings, budget):
    events = [e for e, _ in rows]
    frames = np.array([w["witness_end"]-ORIGIN_FRAME for e, w in rows])
    path = _paths(model, events, frames)
    _, _, power, equilibrium, _ = inputs(events, scales["history"])
    output = {}
    for step in STEPS:
        error = None
        witness = [None]*len(rows)
        try:
            history = _forecast(model, events, scales, settings["prior_weight"], budget, step, 0)["history"]
            impulse = observed_velocity_increment(model, path, history[:, -1, 20:], model._tensor(power),
                                                  model._tensor(equilibrium), frames, step=step, spend=budget.spend)
            residual = (path[:, :, 10:]-path[:, :1, 10:]-impulse).cpu().numpy()
            for i, (_, w) in enumerate(rows):
                high, low = w["machines"]
                witness[i] = float(residual[i, frames[i], high]-residual[i, frames[i], low])
        except NUMERICAL_ERRORS as failed:
            error = str(failed)
        output[str(step)] = _distribution(witness, len(rows)) | {"error": error, "rows": [
            {"id": event["id"], **window, "signed_residual_rad_s": value}
            for (event, window), value in zip(rows, witness, strict=True)]}
    return output


@torch.no_grad()
def _screen_profile(model, train, scales, settings, budget):
    n = settings["screen_batch"]
    lengths = np.array([min(40, len(event["x"])-11) for event in train])
    labels = np.array([e["label"] for e in train])
    scale = model._tensor(scales["target"][10:])
    output = {}
    for step in STEPS:
        residuals = np.full((len(train), 2), np.nan)
        failures = []
        for start in range(0, len(train), n):
            events, frames = train[start:start+n], lengths[start:start+n]
            stop = start+len(events)
            try:
                history = _forecast(model, events, scales, settings["prior_weight"], budget, step, 0)["history"]
                path = _paths(model, events, frames)
                _, _, power, equilibrium, _ = inputs(events, scales["history"])
                impulse = observed_velocity_increment(model, path, history[:, -1, 20:], model._tensor(power),
                    model._tensor(equilibrium), frames, step=step, spend=budget.spend)
                error = (path[..., 10:]-path[:, :1, 10:]-impulse)/scale
                for j, (begin, end) in enumerate(((0, 20), (20, 40))):
                    if end < error.shape[1]:
                        values = (error[:, end]-error[:, begin]).square().mean(dim=1).cpu().numpy()
                        residuals[start:stop, j] = np.where(frames >= end, values, np.nan)
            except NUMERICAL_ERRORS as failed:
                failures.append({"ids": [e["id"] for e in events], "error": str(failed)})
        groups = {}
        for label in ("stable", "unstable"):
            selected = labels == label
            group = residuals[selected]
            available = np.stack((lengths[selected] >= 20, lengths[selected] >= 40), axis=1)
            n_windows = available.sum(axis=1)
            complete = np.array_equal(np.isfinite(group), available)
            known = int((n_windows > 0).sum())
            rms = float(np.sqrt(np.mean(np.nansum(group, axis=1)[n_windows > 0]/n_windows[n_windows > 0]))) if complete and known else None
            groups[label] = {"allocated": int(selected.sum()), "eligible": known,
                "first_window": int(available[:, 0].sum()), "second_window": int(available[:, 1].sum()),
                "qualified_windows": int(np.isfinite(group).sum()), "rms": rms}
        usable = not failures and all(group["rms"] is not None for group in groups.values())
        weights = {label: groups[label]["eligible"]*(8 if label == "unstable" else 1) for label in groups}
        objective = (sum(weights[label]*groups[label]["rms"]**2 for label in groups)/sum(weights.values())) if usable else None
        output[str(step)] = {"groups": groups, "weighted_mean_square": objective, "failures": failures}
    objectives = [row["weighted_mean_square"] for row in output.values()]
    return {"objective": float(np.mean(objectives)) if None not in objectives else None, "steps": output}


def screening(anchor, train, scales, settings, search, budget, directory):
    if len(train) != 346 or any(e["role"] != "train" for e in train):
        raise ValueError("weak-momentum screening requires the complete frozen train roster")
    reference = _screen_profile(anchor.with_support(()), train, scales, settings, budget)
    if reference["objective"] is None or any(row["rms"] <= 0. for s in reference["steps"].values() for row in s["groups"].values()):
        raise RuntimeError("incomplete or zero matched screening reference")
    result = {"reference": reference, "modules": {}}
    for module in MODULES:
        axes = search[module]
        candidates = []

        def run(values, axes=axes, module=module):
            params = dict(zip(axes, values, strict=True))
            return {"parameters": params, **_screen_profile(anchor.with_support((module,), params), train, scales, settings, budget)}

        candidates.extend(run(values) for values in product(*axes.values()))
        qualified = [row for row in candidates if row["objective"] is not None]
        if qualified:
            center = min(qualified, key=lambda row: (row["objective"], tuple(row["parameters"].values())))["parameters"]
            for key, values in axes.items():
                i = values.index(center[key])
                for j in (i-1, i+1):
                    if 0 <= j < len(values):
                        candidates.append(run((center | {key: (values[i]+values[j])/2.}).values()))
        qualified = [row for row in candidates if row["objective"] is not None]
        if qualified:
            chosen = min(qualified, key=lambda row: (row["objective"], tuple(row["parameters"].values())))
            deltas = {step: {label: row["rms"]/reference["steps"][step]["groups"][label]["rms"]-1.
                             for label, row in step_result["groups"].items()}
                      for step, step_result in chosen["steps"].items()}
            reject = any(row["stable"] > .05 or row["unstable"] > .02 for row in deltas.values())
            result["modules"][module] = {"status": "SCREENED_OUT" if reject else "RETAINED", "best": chosen,
                                        "relative_rms_change": deltas, "evaluations": candidates}
        else:
            result["modules"][module] = {"status": "NUMERICAL_FAILURE", "best": None, "evaluations": candidates}
        print(f"M3 {module} screening: {result['modules'][module]['status']}", flush=True)
    write_json(directory/"screening_table.json", result)
    return result


def checkpoint(val, witness, reference, settings, update, *, train=None):
    worst = lambda values: max((v if v is not None and math.isfinite(v) else math.inf for v in values), default=math.inf)
    numerical_ok = lambda row, n: (row["stability"]["allocated"] == n
        and row["stability"]["uncovered"]+row["stability"]["numerical_failures"] == 0
        and row["task3_six_term_score"] is not None)
    val_rows = tuple(val.values())
    stable = [row["stable_errors"] for row in val_rows]
    val_failed = sum(row["stability"]["numerical_failures"]+row["stability"]["uncovered"] for row in val_rows)
    numerical = (all(numerical_ok(row, 107) for row in val_rows)
                 and (train is None or all(numerical_ok(row, 346) for row in train.values())))
    tp = min(row["stability"]["tp"] for row in val_rows)
    fp = max(row["stability"]["fp"] for row in val_rows)
    unresolved = max(row["stability"]["unsettled_predictions"] for row in val_rows)
    med = worst(row["median_8s"] for row in stable)
    tn = worst(row["tn_mean_8s"] for row in stable)
    peak = worst(row["first_500ms_peaks"]["positive_median"] for row in val_rows)
    witness_median = worst(abs(w["signed_median"]) if w["signed_median"] is not None else None for w in witness.values())
    complete = (all(row["qualified_stable_8s"] == 95 for row in stable) and
                all(row["first_500ms_peaks"]["qualified"] == 12 for row in val_rows) and
                all(w["qualified"] == 33 for w in witness.values()))
    gates = {"numerical_and_coverage": numerical and complete and unresolved == 0,
             "unstable_recall": tp == 12, "stable_false_alarms": fp <= settings["max_fp"],
             "stable_trajectory": med <= settings["stable_median_limit"] and tn <= settings["tn_mean_limit"],
             "train_witnesses": complete and all(abs(w["signed_median"]) <= settings["witness_ratio"]*abs(reference[dt]["signed_median"])
                and w["absolute_median"] < reference[dt]["absolute_median"] for dt, w in witness.items())}
    passed = all(gates.values())
    relaxed = fp <= settings["diagnostic_fp"] and all(value for key, value in gates.items() if key != "stable_false_alarms")
    rank = [0 if passed else 1 if relaxed else 2 if numerical and complete else 3,
            val_failed, -tp, unresolved, fp, peak, witness_median, med, tn, update]
    return {"rank": [x if math.isfinite(x) else None for x in rank], "passed": bool(passed), "relaxed_pass": bool(relaxed),
            "gates": gates, "update": update, "val": val, "witness": witness}


def quality(candidate):
    rank = candidate["gate"]["rank"]
    return (*[math.inf if v is None else v for v in rank[:-1]], len(candidate["support"]), rank[-1], candidate["name"])


def equivalent(child, parent, tolerance):
    return all(before is not None and after is not None and after <= (1.+tolerance)*before
               for dt, row in child["gate"]["val"].items() for before, after in (
        (parent["gate"]["val"][dt]["stable_errors"]["tn_mean_8s"], row["stable_errors"]["tn_mean_8s"]),
        (parent["gate"]["val"][dt]["stable_errors"]["median_8s"], row["stable_errors"]["median_8s"]),
        (parent["gate"]["val"][dt]["first_500ms_peaks"]["positive_median"], row["first_500ms_peaks"]["positive_median"])))
