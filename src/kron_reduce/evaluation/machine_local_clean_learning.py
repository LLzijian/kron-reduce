"""One clean training round; all frozen candidates receive val and test scores."""
from __future__ import annotations

import json
import time

import numpy as np
import torch

from kron_reduce.evaluation.constant_c_observability import write_json
from kron_reduce.evaluation.machine_local_observer import execute as execute_observer
from kron_reduce.evaluation.machine_local_observer import known_model
from kron_reduce.evaluation.machine_local_synthetic import synthetic_data
from kron_reduce.evaluation.state_port_training import loss
from kron_reduce.models.differentiable_history import predict_from_history
from kron_reduce.models.machine_local import DiagonalMemoryModel
from project_harness.evidence import sha256_file

ROLES = {"train": slice(0, 32), "val": slice(32, 48), "test": slice(48, 64)}
NUMERICAL_ERRORS = (FloatingPointError, torch.linalg.LinAlgError)


def snapshot(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def objective(model, inputs, target, scale, budget):
    result = predict_from_history(model, *inputs, 300, max_step=.005, spend=budget.spend)
    return loss(result["forecast"], target, np.full(len(target), 1/len(target)), scale,
                np.full(len(target), 300))


def gradient_check(model, inputs, target, scale, budget, directory):
    params = (model.port_parameter, model.rate_parameter)
    value = objective(model, inputs, target, scale, budget)
    budget.spend("gradients")
    grads = torch.autograd.grad(value, params)
    rng, rows = np.random.default_rng(2026111302), []
    for label, parameter, grad in zip(("port", "rate"), params, grads, strict=True):
        direction = torch.tensor(rng.normal(size=tuple(parameter.shape)), dtype=torch.float64)
        direction /= torch.linalg.vector_norm(direction)
        saved = parameter.detach().clone()
        exact = float(torch.sum(grad*direction))
        for epsilon in (1e-5, 1e-6):
            values = []
            try:
                for sign in (1, -1):
                    with torch.no_grad():
                        parameter.copy_(saved+sign*epsilon*direction)
                        values.append(float(objective(model, inputs, target, scale, budget)))
            finally:
                with torch.no_grad():
                    parameter.copy_(saved)
            finite = (values[0]-values[1])/(2*epsilon)
            rows.append({"parameter": label, "direction": direction.tolist(), "plus_minus_losses": values,
                         "epsilon": epsilon, "analytic": exact, "finite_difference": finite,
                         "pass": abs(finite-exact) <= 1e-9+1e-5*abs(exact)})
    write_json(directory/"gradient_check.json", {"pass": all(r["pass"] for r in rows), "checks": rows})
    return all(r["pass"] for r in rows)


def fit(model, inputs, target, scale, budget, directory):
    parameters = (model.port_parameter, model.rate_parameter)
    optimizer = torch.optim.Adam(parameters, lr=.01, betas=(.9, .999), eps=1e-8)
    last_evaluated, count, checkpoint_update = snapshot(model), 0, 0
    started = time.perf_counter()
    with (directory/"training.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        try:
            for _ in range(240):
                optimizer.zero_grad(set_to_none=True)
                value = objective(model, inputs, target, scale, budget)
                if not torch.isfinite(value):
                    raise FloatingPointError("nonfinite training loss")
                last_evaluated = snapshot(model)
                checkpoint_update = count
                budget.spend("gradients")
                value.backward()
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
                    raise FloatingPointError("nonfinite training gradient")
                norm = torch.nn.utils.clip_grad_norm_(parameters, 10.)
                if not torch.isfinite(norm):
                    raise FloatingPointError("nonfinite gradient norm")
                budget.spend("optimizer_updates")
                optimizer.step()
                count += 1
                if any(not torch.isfinite(p).all() for p in parameters):
                    raise FloatingPointError("nonfinite updated parameters")
                stream.write(json.dumps({"update": count, "train_main_nmse": float(value.detach()),
                                         "gradient_norm": float(norm), "elapsed_seconds": time.perf_counter()-started}, allow_nan=False)+"\n")
                stream.flush()
        except NUMERICAL_ERRORS as error:
            np.savez_compressed(directory/"failed_parameters.npz", **{k: v.numpy() for k, v in snapshot(model).items()})
            np.savez_compressed(directory/"last_finite_parameters.npz", **{k: v.numpy() for k, v in last_evaluated.items()})
            model.load_state_dict(last_evaluated)
            write_json(directory/"training_failure.json", {"status": "NUMERICAL_FAILURE_SNAPSHOT_FROZEN",
                "updates_attempted": count, "checkpoint_update": checkpoint_update, "message": str(error),
                "selection_rule": "last_finite_train_forward_not_best"})
            raise
    return {"status": "COMPLETED", "updates_attempted": count, "selection_rule": "fixed_final_update"}


def scores(future, history, truth, scale):
    row = {}
    for seconds in (1, 2, 3, 5, 8):
        error = future[:, 1:100*seconds+1, :20]-truth[:, 11:11+100*seconds, :20]
        for name, channel in (("q", slice(0, 10)), ("v", slice(10, 20)), ("all", slice(0, 20))):
            e = error[:, :, channel]
            row[f"nrmse_{seconds}s_{name}"] = float(np.sqrt(np.mean((e/scale[channel])**2)))
            row[f"rmse_{seconds}s_{name}"] = float(np.sqrt(np.mean(e**2)))
            row[f"max_abs_{seconds}s_{name}"] = float(np.max(np.abs(e)))
    row["main_nrmse"] = float(np.sqrt((row["nrmse_1s_all"]**2+row["nrmse_3s_all"]**2)/2))
    row["history_nrmse"] = float(np.sqrt(np.mean(((history[:, :, :20]-truth[:, :11, :20])/scale)**2)))
    for label, index in (("start", 0), ("origin", 10)):
        row[f"hidden_{label}_rmse"] = float(np.sqrt(np.mean((history[:, index, 20:]-truth[:, index, 20:30])**2)))
    return row


def evaluate_role(model, inputs, truth, scale, budget, directory, key):
    with torch.no_grad():
        result = predict_from_history(model, *inputs, 800, max_step=.0025, spend=budget.spend)
        fine = predict_from_history(model, *inputs, 800, max_step=.00125, spend=budget.spend)
    arrays = {k: v.detach().numpy() for k, v in result.items() if isinstance(v, torch.Tensor)}
    arrays["refined_forecast"] = fine["forecast"].numpy()
    np.savez_compressed(directory/(key+".npz"), **arrays)
    row = scores(arrays["forecast"], arrays["history"], truth, scale)
    row["step_difference"] = float(np.max(np.abs((arrays["forecast"][:, :, :20]-arrays["refined_forecast"][:, :, :20])/scale)))
    budget.spend("matrix_products", 2)
    row["coi_error"] = max(float(np.max(np.abs(arrays["forecast"][:, :, :10]@model.mass.numpy()))),
                            float(np.max(np.abs(arrays["forecast"][:, :, 10:20]@model.mass.numpy()))))
    closure = model.closure()
    budget.spend("numpy_rhs")
    energy = closure.energy_identity(arrays["forecast"][0, 0], np.asarray(inputs[2])[0], np.asarray(inputs[3])[0])
    row.update(condition_number=result["condition_number"], rank=result["rank"],
        energy_identity_error=float(energy["identity_error"]),
        status="PASS", evaluated_events=len(truth), allocated_events=len(truth), failed_events=0,
        accuracy_pass=row["main_nrmse"] <= .001 and row["nrmse_3s_all"] <= .001)
    return row


def evaluate_all(model, inputs, paths, scale, budget, directory, name, evaluator=evaluate_role):
    rows = {}
    for role, ix in ROLES.items():
        try:
            with budget.scope(name+"_"+role):
                subset = tuple(x[ix] for x in inputs[:-1])+(inputs[-1],)
                row = evaluator(model, subset, paths[ix], scale, budget, directory, name+"_"+role)
        except NUMERICAL_ERRORS as error:
            row = {"status": "NUMERICAL_FAILURE", "message": str(error), "allocated_events": len(paths[ix]),
                   "evaluated_events": 0, "failed_events": len(paths[ix]), "accuracy_pass": False}
        rows[role] = row
        write_json(directory/(name+"_"+role+".json"), row)
        print(f"{name} {role}: {row.get('nrmse_3s_all', row['status'])}", flush=True)
    return rows


def validate(root, directory, budget, plan):
    with budget.scope("truth"):
        base, paths, scale, hscale = synthetic_data("local", plan["truth_seed"], budget)
        np.savez_compressed(directory/"inputs.npz", paths=paths, target_scale=scale, history_scale=hscale,
                            roles=np.array(["train"]*32+["val"]*16+["test"]*16))
    write_json(directory/"frozen_inputs.json", {"inputs.npz": sha256_file(directory/"inputs.npz")})
    inputs = (paths[:, :11, :10].copy(), paths[:, :11, 10:20].copy(), paths[:, 0, 30:].copy(),
              np.zeros((64, 10)), hscale)
    model = DiagonalMemoryModel(base, port_kind="local", port=np.full(10, .4),
        tau=np.geomspace(.06, 1.6, 10), encoder=np.zeros((10, 80)), feature_scale=np.ones(80))
    model.encoder.requires_grad_(False)
    start = snapshot(model)
    train = tuple(x[:32] for x in inputs[:-1])+(hscale,)
    target = paths[:32, 10:311, :20].copy()
    try:
        with budget.scope("gradient"):
            small = tuple(x[:2] for x in train[:-1])+(hscale,)
            gradient_pass = gradient_check(model, small, target[:2], scale, budget, directory)
    except NUMERICAL_ERRORS as error:
        gradient_pass = False
        write_json(directory/"gradient_check.json", {"pass": False, "message": str(error)})
    if gradient_pass:
        try:
            with budget.scope("fit"):
                budget.spend("fits")
                training = fit(model, train, target, scale, budget, directory)
        except NUMERICAL_ERRORS:
            training = json.loads((directory/"training_failure.json").read_text(encoding="utf-8"))
    else:
        training = {"status": "GRADIENT_CHECK_FAILED", "updates_attempted": 0}
    final = snapshot(model)
    true_model = known_model(base, "local")
    frozen = {"truth": snapshot(true_model), "initial": start, "fitted": final}
    for name, state in frozen.items():
        np.savez_compressed(directory/(name+"_parameters.npz"), **{k: v.numpy() for k, v in state.items()})
    write_json(directory/"frozen_models.json", {p.name: sha256_file(p) for p in sorted(directory.glob("*_parameters.npz"))})
    records, parameter_metrics = {}, {}
    for name, state in frozen.items():
        model.load_state_dict(state)
        g, tau = np.exp(model.port_parameter.detach().numpy()), model.tau.detach().numpy()
        true_g, true_tau = .35+.025*np.arange(10), 1/np.diag(base.latent_dissipation)
        parameter_metrics[name] = {"g": g.tolist(), "tau": tau.tolist(),
            "relative_g_error": float(np.linalg.norm(g-true_g)/np.linalg.norm(true_g)),
            "relative_tau_error": float(np.linalg.norm(tau-true_tau)/np.linalg.norm(true_tau))}
        records[name] = evaluate_all(model, inputs, paths, scale, budget, directory, name)
    valid = all(row["status"] == "PASS" and row["step_difference"] <= 1e-5
                and row["coi_error"] <= 1e-10 and row["energy_identity_error"] <= 1e-10
                for rows in records.values() for row in rows.values())
    accuracy = all(row["accuracy_pass"] for row in records["fitted"].values())
    truth_pass = all(row["status"] == "PASS" and row["accuracy_pass"]
                     and max(row["hidden_start_rmse"], row["hidden_origin_rmse"]) <= 1e-4
                     for row in records["truth"].values())
    return {"scientific_pass": bool(valid and gradient_pass and truth_pass and accuracy and training["status"] == "COMPLETED"),
        "truth_control_pass": bool(truth_pass),
        "numerical_pass": bool(valid), "gradient_pass": gradient_pass, "accuracy_pass": accuracy,
        "evaluation_complete": all(set(rows) == set(ROLES) for rows in records.values()),
        "training": training, "records": records, "parameter_metrics": parameter_metrics, "noise_injection": 0.,
        "next_route": "REVIEW_CLEAN_DYNAMICS_RESULTS", "t0_allowed": False,
        "claim_boundary": "One clean local linear synthetic round. All frozen models receive train/val/test evaluation regardless of RMSE. Known-backbone learning of twenty port/time-scale parameters is not real-data confirmation."}


def execute(*, root, result_dir, stage, plan):
    return execute_observer(root=root, result_dir=result_dir, stage=stage, plan=plan, validator=validate)
