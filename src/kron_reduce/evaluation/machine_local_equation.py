"""Known-truth joint learning, explicit support refits and complete scoring."""
from __future__ import annotations

import sys
import time
from importlib.metadata import version

import numpy as np
import torch

from kron_reduce.evaluation.constant_c_observability import write_json
from kron_reduce.evaluation.kiki_machine_local_full import inputs, predict_checked
from kron_reduce.evaluation.machine_local_clean_learning import NUMERICAL_ERRORS
from kron_reduce.evaluation.machine_local_equation_cases import make_population
from kron_reduce.evaluation.machine_local_equation_fit import (
    choose_support,
    fit,
    freeze,
    make_model,
    refit_model,
    support_values,
)
from kron_reduce.evaluation.machine_local_hidden_oracle import trajectory_scores
from kron_reduce.evaluation.machine_local_synthetic import Budget
from kron_reduce.evaluation.trajectory_stability import classify, summarize
from kron_reduce.physics.coordinates import coi_basis
from project_harness.evidence import sha256_file
from project_harness.git_state import git_head

POPULATIONS = ("null", "harmonic", "damping")
ROLES = ("train", "val", "test")
STEPS = (.0025, .00125)
FORECAST_FIELDS = {"forecast": (801, 30), "history": (11, 30), "gn_singular": (3, 10)}


def align_classes(a, b):
    i, j = a["trigger_index"], b["trigger_index"]
    return (a["finite_full_path"] and b["finite_full_path"] and a["label"] == b["label"]
            and (i is None and j is None or i is not None and j is not None and abs(i-j) <= 2))


def score(model, population, role, budget, directory):
    events = population["roles"][role]
    truth = population["role_paths"][role]
    scale, hscale = population["target_scale"], population["history_scale"]
    mass = population["anchor"].mass
    results, failures = {}, []
    for step in STEPS:
        arrays = {key: np.full((len(events), *shape), np.nan) for key, shape in FORECAST_FIELDS.items()}

        def forward(batch, indices, *, step=step, arrays=arrays):
            with torch.no_grad():
                prediction = predict_checked(model, *inputs(batch, hscale), 800, max_step=step, spend=budget.spend)
            for key, array in arrays.items():
                value = prediction[key].cpu().numpy()
                array[indices] = value.swapaxes(0, 1) if key == "gn_singular" else value

        try:
            forward(events, slice(None))
        except NUMERICAL_ERRORS as error:
            failures.append({"step": step, "batch": True, "error": str(error)})
            for i, event in enumerate(events):
                try:
                    forward([event], slice(i, i+1))
                except NUMERICAL_ERRORS as individual:
                    failures.append({"step": step, "id": event["id"], "error": str(individual)})
        results[step] = arrays
    rows = []
    ref = truth["refined"]
    for i, event in enumerate(events):
        path = ref[i]
        args = (path[:11, :10], path[:11, 10:20])
        fine, refined = (results[step]["forecast"][i] for step in STEPS)
        raw = (truth["fine"][i, 10:], path[10:], fine, refined)
        canonical, actual, predicted, finer = (classify(*args, x[:, :20], event["equilibrium"], mass) for x in raw)
        finite = np.isfinite(fine).all() and np.isfinite(refined).all()
        step_gap = float(np.max(np.abs((fine[:, :20]-refined[:, :20])/scale))) if finite else None
        coi = float(np.max(np.abs(np.stack((fine, refined))[..., :20].reshape(2, 801, 2, 10)@mass))) if finite else None
        truth_gap = float(np.max(np.abs((truth["fine"][i, :, :20]-path[:, :20])/scale)))
        truth_ok = truth_gap <= 1e-6 and align_classes(canonical, actual)
        numerical = bool(finite and step_gap <= 1e-5 and coi <= 1e-10 and align_classes(predicted, finer))
        onset = actual["trigger_index"]
        history = results[STEPS[0]]["history"][i]
        singular = results[STEPS[0]]["gn_singular"][i]
        valid_history = np.isfinite(history).all() and np.isfinite(singular).all()
        rows.append({"id": event["id"], "truth_label": actual["label"], "predicted_label": predicted["label"],
                     "truth_trigger_index": onset, "prediction": predicted, "refined_prediction": finer,
                     "truth_step_max_error": truth_gap, "truth_numerical": truth_ok,
                     "step_max_error": step_gap, "coi_error": coi, "qualified": numerical,
                     "prospective_eligible": onset is None or onset > 10,
                     "numerical_failure": not numerical, "unsettled": numerical and predicted["label"] is None,
                     "truth_agreement": align_classes(actual, predicted),
                     "trajectory": trajectory_scores(fine, path, scale),
                     "history_max_error": float(np.max(np.abs((history[1:, :20]-path[1:11, :20])/hscale))) if valid_history else None,
                     "hidden_start_rmse": float(np.sqrt(np.mean((history[0, 20:]-path[0, 20:])**2))) if valid_history else None,
                     "hidden_origin_rmse": float(np.sqrt(np.mean((history[-1, 20:]-path[10, 20:])**2))) if valid_history else None,
                     "hidden_origin_energy": float(.5*np.sum(history[-1, 20:]**2)) if valid_history else None,
                     "condition_max": float(np.max(singular[:, 0]/singular[:, -1])) if valid_history else None})
    stability = summarize(rows)
    for key, numerator, denominator in (
        ("recall_on_resolved", stability["tp"], stability["tp"]+stability["fn"]),
        ("miss_rate_on_resolved", stability["fn"], stability["tp"]+stability["fn"]),
        ("false_alarm_rate_on_resolved", stability["fp"], stability["fp"]+stability["tn"]),
    ):
        stability[key] = numerator/denominator if denominator else None
    qualified = [row for row in rows if row["qualified"] and row["trajectory"] is not None]
    horizons = {}
    for seconds in (1, 2, 3, 5, 8):
        scores = [row["trajectory"][str(seconds)] for row in qualified]
        valid = len(scores) == len(rows)
        horizons[str(seconds)] = {
            "qualified": len(scores), "allocated": len(rows),
            "event_mean_nrmse": float(np.mean([r["nrmse"] for r in scores])) if valid else None,
            "pooled_nrmse": float(np.sqrt(np.mean([r["nmse"] for r in scores]))) if valid else None,
            "q_rmse_rad": float(np.sqrt(np.mean([r["q_rmse"]**2 for r in scores]))) if valid else None,
            "v_rmse_rad_s": float(np.sqrt(np.mean([r["v_rmse"]**2 for r in scores]))) if valid else None,
            "qualified_event_mean_nrmse": float(np.mean([r["nrmse"] for r in scores])) if scores else None,
        }
    record = {"horizons": horizons, "stability": stability, "events": rows, "failures": failures}
    np.savez_compressed(directory/(role+".npz"),
                        **{f"{label}_{key}": val for label, step in zip(("fine", "refined"), STEPS, strict=True)
                           for key, val in results[step].items()})
    write_json(directory/(role+".json"), record)
    return record


def physical(model):
    array = lambda tensor: tensor.detach().cpu().numpy()
    return (array(model.effective_stiffness), array(model.effective_damping), array(model.coupling),
            array(model.tau), support_values(model))


def function_checks(model, population, budget):
    budget.spend("function_checks")
    b = model.incidence.numpy()
    base = population["stiffness"]
    states = population["role_paths"]["val"]["refined"][:, 10::10].reshape(-1, 30)
    equilibria = np.repeat(np.stack([e["equilibrium"] for e in population["roles"]["val"]]), 81, axis=0)
    q, v = states[:, :10], states[:, 10:20]
    eta, ref, velocity = q@b, equilibria@b, v@b
    time_grid = np.geomspace(.01, 3., 32)

    def functions(stiffness, damping, coupling, tau, strengths):
        a, d = strengths
        force = ((np.sin(eta+ref)-np.sin(ref))*stiffness
                 + (np.sin(2*(eta+ref))-np.sin(2*ref))*base*a) @ b.T
        friction = v@damping.T+(8*d/b.shape[1])*(np.sin(eta)**2*velocity) @ b.T
        kernel = np.einsum("ij,tj,kj->tik", coupling, np.exp(-time_grid[:, None]/tau), coupling)
        return force, friction, kernel

    learned, truth = physical(model), physical(population["truth_model"])
    # The analytic power balance is checked at the same frozen val grid on CPU.
    power = np.repeat(np.stack([e["p0"] for e in population["roles"]["val"]]), 81, axis=0)
    k, _, _, tau, strengths = learned
    force, friction, kernel = functions(*learned)
    with torch.no_grad():
        budget.spend("torch_rhs")
        rhs = model.rhs(states, power, equilibria).numpy()
    z = states[:, 20:]
    mass = model.mass.numpy()
    dissipated = np.sum(v*friction, axis=1) + np.sum(z*z/tau, axis=1)
    supply = np.sum(v*power, axis=1)
    hdot = np.sum(force*rhs[:, :10]+mass*v*rhs[:, 10:20]+z*rhs[:, 20:], axis=1)
    n = coi_basis(mass)
    columns = n.T @ b
    slope_at_zero = np.cos(ref)*k + 2*strengths[0]*base*np.cos(2*ref)
    hessian = (columns*slope_at_zero[:, None, :]) @ columns.T
    budget.spend("curvature_spectra", len(hessian))
    values = np.linalg.eigvalsh(hessian)
    rel = lambda est, ref: float(np.linalg.norm(est-ref)/np.linalg.norm(ref))
    errors = {name: rel(value, reference) for name, value, reference in zip(
        ("potential_force", "visible_damping_force", "memory_kernel"), (force, friction, kernel),
        functions(*truth), strict=True)}
    parameters = {name: rel(value, reference) for name, value, reference in zip(
        ("stiffness", "damping_matrix", "coupling", "tau"), learned[:4], truth[:4], strict=True)}
    if not all(np.isfinite(x).all() for x in (values, hdot, dissipated, list(errors.values()), list(parameters.values()))):
        raise FloatingPointError("nonfinite function or energy diagnostic")
    return {"function_relative_errors": errors, "parameter_relative_errors": parameters,
            "energy_balance_relative_max": float(np.max(np.abs(hdot-supply+dissipated)/np.maximum(1., np.abs(supply)+np.abs(dissipated)))),
            "reference_min_coi_curvature": float(values.min()), "minimum_dissipation": float(dissipated.min())}


def case(root, directory, name, budget):
    population = make_population(name, budget)
    directory.mkdir()
    truth = population["truth_model"]
    models = {key: make_model(population, key) for key in ("a", "both")}
    a_state, b_state = (model.state_dict() for model in models.values())
    if any(not torch.equal(value, b_state[key]) for key, value in a_state.items()) or any(support_values(models["both"])):
        raise RuntimeError("joint candidates do not share an exact coefficient start")
    np.savez_compressed(directory/"truth.npz", **population["paths"], power=population["power"],
                        equilibrium=population["equilibrium"], history_scale=population["history_scale"],
                        target_scale=population["target_scale"],
                        **{role: np.array([e["truth_index"] for e in events]) for role, events in population["roles"].items()})
    initial_model = make_model(population, "a").requires_grad_(False)
    fits = {key: fit(model, population, directory/key, budget, 240, shrink=1e-6 if key == "both" else 0.)
            for key, model in models.items()}
    for support in ("harmonic", "damping", "neither"):
        models[support] = refit_model(population, models["both"], support)
        fits[support] = fit(models[support], population, directory/support, budget, 80, shrink=0.)
    selected, allowance = choose_support(models, fits)
    write_json(directory/"selection.json", {"selected_b": selected, "val_allowance": allowance,
               "validation_scores": {key: row["selected_val_nmse"] for key, row in fits.items()}})
    (directory/"known").mkdir()
    models["known"] = truth
    for key, model in models.items():
        freeze(model, directory/key)
    write_json(directory/"freeze.json", {"selected_b": selected, "source_commit": git_head(root),
               "models": {key: sha256_file(directory/key/"frozen.npz") for key in models},
               "preprocessing": sha256_file(directory/"truth.npz"), "selection_closed": True})
    def physical_diagnostic(model):
        try:
            return function_checks(model, population, budget)
        except (FloatingPointError, np.linalg.LinAlgError) as error:
            return {"failure": str(error)}

    initial = physical_diagnostic(initial_model)
    records, functions = {}, {}
    for key, model in models.items():
        functions[key] = physical_diagnostic(model)
        predictor = model.to("cuda")
        records[key] = {role: score(predictor, population, role, budget, directory/key) for role in ROLES}
        model.to("cpu")
        print(f"{name}/{key}: all 32 events scored", flush=True)
    expected = support_values(truth)
    actual = support_values(models[selected])
    support_pass = all(value == 0 if target == 0 else abs(value/target-1.) <= .1
                       for value, target in zip(actual, expected, strict=True))
    tolerances = {"potential_force": .03, "visible_damping_force": .05, "memory_kernel": .05}
    initial_gap = "failure" not in initial and any(initial["function_relative_errors"][k] > v for k, v in tolerances.items())

    def functions_pass(key):
        return all(functions[key].get("function_relative_errors", {}).get(k, np.inf) <= v for k, v in tolerances.items())

    function_pass = functions_pass(selected)
    train_val = [records[selected][role] for role in ("train", "val")]
    def trajectories_pass(rows):
        return all(r["horizons"]["3"]["pooled_nrmse"] is not None
                   and r["horizons"]["3"]["pooled_nrmse"] <= .01
                   and r["horizons"]["8"]["pooled_nrmse"] <= .03 for r in rows)

    trajectory_pass = trajectories_pass(train_val)
    a_pass = (name != "null" or (fits["a"]["selected_val_nmse"] is not None and fits["a"]["failure"] is None
              and trajectories_pass([records["a"][role] for role in ("train", "val")])
              and functions_pass("a")))
    known_pass = all(records["known"][role]["horizons"]["3"]["pooled_nrmse"] is not None
                     and records["known"][role]["horizons"]["3"]["pooled_nrmse"] <= .001
                     and {e["truth_label"] for e in records["known"][role]["events"]} == {"stable", "unstable"}
                     and all(e["truth_numerical"] and e["truth_agreement"] and e["prospective_eligible"]
                             and e["history_max_error"] is not None and e["history_max_error"] <= 1e-7
                             and e["hidden_start_rmse"] <= 1e-4 and e["hidden_origin_rmse"] <= 1e-4
                             for e in records["known"][role]["events"]) for role in ("train", "val"))
    physical_pass = all("failure" not in f and f["energy_balance_relative_max"] <= 1e-10 and f["minimum_dissipation"] >= -1e-12
                        and f["reference_min_coi_curvature"] > 1e-8 for f in functions.values())
    gates = {"initial_function_gap": initial_gap, "known_control_train_val": known_pass,
             "selected_training_valid": fits[selected]["selected_val_nmse"] is not None and fits[selected]["failure"] is None,
             "selected_trajectory_train_val": trajectory_pass, "selected_function_recovery_val": function_pass,
             "selected_support_and_amplitude": support_pass, "physical_checks": physical_pass,
             "original_equation_recovery_on_null": a_pass}
    return {"selected_b": selected, "selected_strengths": actual, "true_strengths": expected,
            "gates": gates, "scientific_pass": all(gates.values()), "fits": fits,
            "initial_functions": initial, "functions": functions,
            "records": {key: {role: {f: r[f] for f in ("horizons", "stability", "failures")}
                              for role, r in roles.items()} for key, roles in records.items()}}


def execute(root, directory, plan):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    start = time.perf_counter()
    budget = Budget(directory/"operations.jsonl", plan["operation_limits"])
    try:
        results = {name: case(root, directory/name, name, budget) for name in POPULATIONS}
    finally:
        budget.stream.close()
    passed = all(row["scientific_pass"] for row in results.values())
    stage = {"status": "PASS", "process_status": "PASS", "source_commit": git_head(root),
             "data_policy": "analytic_synthetic_only", "promotion_decision": "KEEP_BASELINE",
             "scientific_pass": passed, "claim_boundary": "Joint equations and family selection on fixed known-truth populations."}
    summary = stage | {"populations": results, "operation_counts": dict(budget.counts),
                       "elapsed_seconds": time.perf_counter()-start,
                       "runtime": {"python": sys.version, "torch_cuda": torch.version.cuda,
                                   "cuda_device": torch.cuda.get_device_name(), "fit_device": "cpu", "score_device": "cuda",
                                   **{name: version(name) for name in ("torch", "numpy", "scipy")}}}
    write_json(directory/"stage_evidence.json", stage)
    write_json(directory/"summary.json", summary)
    write_json(directory/"artifact_hashes.json", {p.relative_to(directory).as_posix(): sha256_file(p)
                for p in sorted(directory.rglob("*")) if p.is_file()})
    return summary
