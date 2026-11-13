"""Full synthetic population separating hidden-state and dynamics errors."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from kron_reduce.evaluation.constant_c_observability import write_json
from kron_reduce.evaluation.machine_local_clean_learning import NUMERICAL_ERRORS, snapshot
from kron_reduce.evaluation.machine_local_nonlinear_history import class_for, inputs_for
from kron_reduce.evaluation.machine_local_observer import execute as execute_observer
from kron_reduce.evaluation.machine_local_observer import known_model
from kron_reduce.evaluation.machine_local_synthetic import fixture
from kron_reduce.evaluation.trajectory_stability import summarize
from kron_reduce.models.nonlinear_history import predict_from_history
from project_harness.evidence import sha256_file

ROLES = {"train": slice(0, 32), "val": slice(32, 48), "test": slice(48, 64)}
GROUPS = ("matched_gn", "matched_oracle", "mismatch_gn", "mismatch_oracle")
STEPS = {"fine": .0025, "refined": .00125}


def initial_states(mass):
    i, j = np.arange(10), np.arange(64)
    phase = 2*np.pi*(j+.5)/64
    projection = np.eye(10)-np.outer(mass, np.ones(10))/mass.sum()
    q = .03*np.sin(i[None, :]+phase[:, None]) @ projection
    v = .02*np.cos(i[None, :]+phase[:, None])
    v[j, (j//2) % 10] += 16*(j % 2)
    z = .05*np.cos(i[None, :]+phase[:, None])
    return np.c_[q, v @ projection, z]


def make_inputs(base, budget):
    paths, clock = [], np.arange(811)*.01
    for step in (.00125, .000625):
        values = []
        rhs_count = 4*sum(max(1, int(np.ceil(h/step-1e-12))) for h in np.diff(clock))
        for index, state in enumerate(initial_states(base.mass)):
            budget.spend("rollouts")
            budget.spend("numpy_rhs", rhs_count)
            values.append(base.rollout(state[:10], state[10:20], state[20:], clock,
                          np.zeros(10), np.zeros(10), max_step=step))
            if (index+1) % 16 == 0:
                print(f"truth step={step}: {index+1}/64", flush=True)
        paths.append(np.stack(values))
    reference = paths[1]
    scale = np.maximum(np.sqrt(np.mean(reference[:32, 11:311, :20]**2, axis=(0, 1))), 1e-5)
    hscale = np.maximum(np.sqrt(np.mean(reference[:32, :11, :20]**2, axis=(0, 1))), 1e-5)
    return reference, paths[0], scale, hscale


def make_models(base):
    return {name: known_model(backbone, "local") for name, backbone in
            (("matched", base), ("mismatch", replace(base, edge_stiffness=base.edge_stiffness*1.10)))}


def predict(model, paths, hscale, step, oracle, spend):
    """Oracle uses only the complete state at the common forecast origin."""
    try:
        with torch.no_grad():
            if oracle:
                zeros = np.zeros((len(paths), 10))
                result = {"forecast": model.rollout(paths[:, 10, :30], zeros, zeros, 800,
                                                    max_step=step, spend=spend)}
            else:
                result = predict_from_history(model, *inputs_for(paths, hscale), 800,
                                              max_step=step, spend=spend)
    except ValueError as error:
        if str(error) == "model inputs must be finite":
            raise FloatingPointError("nonfinite intermediate RK4 state") from error
        raise
    if not torch.isfinite(result["forecast"]).all():
        raise FloatingPointError("nonfinite forecast")
    arrays = {"forecast": result["forecast"].numpy()}
    if not oracle:
        arrays["history"] = result["history"].numpy()
        for target, source in (("iterates", "iterates"), ("increments", "gn_hidden_start"),
                               ("singular", "gn_singular"), ("residuals", "iteration_residuals")):
            arrays[target] = result[source].numpy().swapaxes(0, 1)
    return arrays


def predict_batches(model, paths, hscale, oracle, budget, predictor=predict):
    fields = {"forecast": (801, 30)}
    if not oracle:
        fields.update(history=(11, 30), iterates=(4, 10), increments=(3, 10),
                      singular=(3, 10), residuals=(3, 200))
    arrays = {prefix+"_"+key: np.full((len(paths), *shape), np.nan)
              for prefix in STEPS for key, shape in fields.items()}
    failures = []

    def forward(part, offset, prefix, step):
        result = predictor(model, part, hscale, step, oracle, budget.spend)
        for key in fields:
            arrays[prefix+"_"+key][offset:offset+len(part)] = result[key]

    for offset in range(0, len(paths), 16):
        part = paths[offset:offset+16]
        for prefix, step in STEPS.items():
            try:
                forward(part, offset, prefix, step)
            except NUMERICAL_ERRORS as error:
                failures.append({"offset": offset, "count": len(part), "step": step, "error": str(error)})
                for index, path in enumerate(part):
                    try:
                        forward(path[None], offset+index, prefix, step)
                    except NUMERICAL_ERRORS as individual:
                        failures.append({"index": offset+index, "step": step, "error": str(individual)})
    return arrays, failures


def trajectory_scores(forecast, truth, scale):
    if not np.isfinite(forecast).all():
        return None
    horizons = {}
    for seconds in (1, 2, 3, 5, 8):
        error = forecast[1:100*seconds+1, :20]-truth[11:11+100*seconds, :20]
        row = {"nmse": float(np.mean((error/scale)**2))}
        row["nrmse"] = float(np.sqrt(row["nmse"]))
        for name, channel in (("q", slice(0, 10)), ("v", slice(10, 20))):
            row[name+"_nrmse"] = float(np.sqrt(np.mean((error[:, channel]/scale[channel])**2)))
            row[name+"_rmse"] = float(np.sqrt(np.mean(error[:, channel]**2)))
            row[name+"_max_abs"] = float(np.max(np.abs(error[:, channel])))
        horizons[str(seconds)] = row
    return horizons


def describe(arrays, paths, truth_classes, scale, hscale, mass, oracle):
    events = []
    for index, path in enumerate(paths):
        fine, refined = (arrays[p+"_forecast"][index] for p in STEPS)
        a, b = (class_for(path, forecast, mass) for forecast in (fine, refined))
        finite = bool(np.isfinite(fine).all() and np.isfinite(refined).all())
        difference = float(np.max(np.abs((fine[:, :20]-refined[:, :20])/scale))) if finite else None
        coi = max(float(np.max(np.abs(x[:, sl] @ mass))) for x in (fine, refined)
                  for sl in (slice(0, 10), slice(10, 20))) if finite else None
        time_ok = (a["trigger_index"] is None and b["trigger_index"] is None or
                   a["trigger_index"] is not None and b["trigger_index"] is not None and
                   abs(a["trigger_index"]-b["trigger_index"]) <= 2)
        numerical = bool(finite and difference <= 1e-5 and coi <= 1e-10 and time_ok and a["label"] == b["label"])
        row = {"index": index, "truth_label": truth_classes[index]["label"], "predicted_label": a["label"],
               "prospective_eligible": truth_classes[index]["trigger_index"] is None or truth_classes[index]["trigger_index"] > 10,
               "qualified": numerical, "numerical_failure": not numerical, "unsettled": a["label"] is None,
               "prediction": a, "refined_prediction": b, "step_difference": difference, "coi_error": coi,
               "trajectory": trajectory_scores(fine, path, scale)}
        if not oracle:
            history = arrays["fine_history"][index]
            if np.isfinite(history).all():
                row["history_max_normalized_error"] = float(np.max(np.abs((history[:, :20]-path[:11, :20])/hscale)))
                row["history_nrmse"] = float(np.sqrt(np.mean(((history[:, :20]-path[:11, :20])/hscale)**2)))
                for label, point in (("start", 0), ("origin", 10)):
                    row["hidden_"+label+"_rmse"] = float(np.sqrt(np.mean((history[point, 20:]-path[point, 20:])**2)))
                    row["hidden_"+label+"_energy"] = float(np.sum(history[point, 20:]**2)/2)
                    row["true_hidden_"+label+"_energy"] = float(np.sum(path[point, 20:]**2)/2)
        events.append(row)
    complete = all(e["qualified"] and e["trajectory"] is not None for e in events)
    horizons = None
    if complete:
        horizons = {str(s): {"event_mean_nrmse": float(np.mean([e["trajectory"][str(s)]["nrmse"] for e in events])),
            "event_mean_nmse": float(np.mean([e["trajectory"][str(s)]["nmse"] for e in events])),
            "pooled_nrmse": float(np.sqrt(np.mean([e["trajectory"][str(s)]["nmse"] for e in events])))}
            for s in (1, 2, 3, 5, 8)}
        for seconds, values in horizons.items():
            for name in ("q", "v"):
                for metric in ("nrmse", "rmse"):
                    key = name+"_"+metric
                    values[key] = float(np.sqrt(np.mean([e["trajectory"][seconds][key]**2 for e in events])))
                key = name+"_max_abs"
                values[key] = max(e["trajectory"][seconds][key] for e in events)
    stability = summarize(events)
    for key, numerator, denominator in (("recall_on_resolved", stability["tp"], stability["tp"]+stability["fn"]),
            ("miss_rate_on_resolved", stability["fn"], stability["tp"]+stability["fn"]),
            ("false_alarm_rate_on_resolved", stability["fp"], stability["tn"]+stability["fp"])):
        stability[key] = numerator/denominator if denominator else None
    recovery = (complete and horizons["3"]["pooled_nrmse"] <= .001 and stability["accuracy"] == 1.
                and (oracle or all(e.get("hidden_start_rmse", np.inf) <= 1e-4
                    and e.get("hidden_origin_rmse", np.inf) <= 1e-4
                    and e.get("history_max_normalized_error", np.inf) <= 1e-7 for e in events)))
    return {"allocated": len(paths), "events": events, "horizons": horizons, "stability": stability,
            "numerical_pass": all(e["qualified"] for e in events), "recovery_pass": bool(recovery)}


def paired_differences(records):
    pairs = {"matched_initialization": ("matched_gn", "matched_oracle"),
             "dynamics_at_true_origin": ("mismatch_oracle", "matched_oracle"),
             "mismatch_initialization": ("mismatch_gn", "mismatch_oracle")}
    rows = {}
    for role in ROLES:
        rows[role] = {}
        for name, (left, right) in pairs.items():
            values = [a["trajectory"]["3"]["nmse"]-b["trajectory"]["3"]["nmse"]
                      if a["qualified"] and b["qualified"] and a["trajectory"] is not None and b["trajectory"] is not None else None
                      for a, b in zip(records[left][role]["events"], records[right][role]["events"], strict=True)]
            rows[role][name] = {"per_event": values, "allocated": len(values),
                "paired_count": sum(v is not None for v in values),
                "mean": float(np.mean(values)) if all(v is not None for v in values) else None}
    return rows


def validate(root, directory, budget, plan):
    base = fixture(backbone="sine")
    with budget.scope("truth"):
        paths, coarse, scale, hscale = make_inputs(base, budget)
        classes = [class_for(p, p[10:], base.mass) for p in paths]
        coarse_classes = [class_for(p, p[10:], base.mass) for p in coarse]
        difference = float(np.max(np.abs((paths[:, :, :20]-coarse[:, :, :20])/scale)))
        coverage = all({c["label"] for c in classes[ix]} == {"stable", "unstable"} for ix in ROLES.values())
        truth_pass = bool(coverage and difference <= 1e-6 and np.isfinite(paths).all()
            and all(a["label"] == b["label"] and a["trigger_index"] == b["trigger_index"]
                    and (a["trigger_index"] is None or a["trigger_index"] > 10)
                    for a, b in zip(classes, coarse_classes, strict=True)))
        np.savez_compressed(directory/"inputs.npz", paths=paths, coarse_paths=coarse, scale=scale,
            history_scale=hscale, mass=base.mass, ids=np.arange(64), roles=np.array(["train"]*32+["val"]*16+["test"]*16))
    write_json(directory/"truth.json", {"pass": truth_pass, "class_coverage_pass": coverage,
               "step_difference": difference, "classes": classes, "coarse_classes": coarse_classes})
    models = make_models(base)
    for name, model in models.items():
        model.requires_grad_(False)
        np.savez_compressed(directory/(name+"_parameters.npz"), **{k: v.numpy() for k, v in snapshot(model).items()})
    write_json(directory/"frozen_inputs_and_models.json", {p.name: sha256_file(p) for p in directory.glob("*.npz")})
    records = {}
    if truth_pass:
        for group in GROUPS:
            name, method = group.split("_")
            records[group] = {}
            for role, ix in ROLES.items():
                key = group+"_"+role
                with budget.scope(key):
                    arrays, failures = predict_batches(models[name], paths[ix], hscale, method == "oracle", budget)
                    np.savez_compressed(directory/(key+".npz"), **arrays)
                    row = describe(arrays, paths[ix], classes[ix], scale, hscale, base.mass, method == "oracle")
                    row["failures"] = failures
                records[group][role] = row
                write_json(directory/(key+".json"), row)
                print(f"{key}: horizons={row['horizons']} stability={row['stability']['accuracy']}", flush=True)
    numerical = bool(truth_pass and all(r["numerical_pass"] for rows in records.values() for r in rows.values()))
    recovery = bool(truth_pass and all(r["recovery_pass"] for group in ("matched_gn", "matched_oracle") for r in records[group].values()))
    differences = paired_differences(records) if truth_pass else {}
    write_json(directory/"paired_differences.json", differences)
    return {"scientific_pass": numerical and recovery, "truth_pass": truth_pass, "numerical_pass": numerical,
        "matched_recovery_pass": recovery, "evaluation_complete": truth_pass, "records": records,
        "paired_differences": differences, "fit_attempts": 0, "noise_injection": 0., "t0_allowed": False,
        "budget_accounting": "Attempted calls remain spent after batch failure. Returned scopes do not certify every internal call succeeded.",
        "next_route": "REVIEW_INITIALIZATION_DYNAMICS_SEPARATION" if truth_pass else "REVIEW_TRUTH_GATE_FAILURE",
        "claim_boundary": "Complete 64-case sine L10 known-truth comparison of matched/10% stiffness mismatch and causal/true hidden origin. No fitting, noise, real-data reads or real-system causal attribution."}


def execute(*, root, result_dir, stage, plan):
    return execute_observer(root=root, result_dir=result_dir, stage=stage, plan=plan, validator=validate)
