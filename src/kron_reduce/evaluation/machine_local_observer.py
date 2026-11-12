"""Fixed-dynamics validation of the discrete-history state initializer."""
from __future__ import annotations

import hashlib
import io
import sys
import time
from importlib.metadata import version

import numpy as np
import torch
from scipy.linalg import helmert

from kron_reduce.evaluation.constant_c_observability import write_json
from kron_reduce.evaluation.machine_local_initializer_diagnosis import check_paths, metrics
from kron_reduce.evaluation.machine_local_synthetic import (
    Budget,
    fixture,
    new_model,
    peak_memory,
    synthetic_data,
)
from kron_reduce.models.discrete_history_initializer import estimate_history
from kron_reduce.models.machine_local import DiagonalMemoryModel
from project_harness.evidence import sha256_file
from project_harness.git_state import git_head


def known_model(base, kind):
    port = .35 + .025*np.arange(10)
    if kind == "nonlocal":
        port = helmert(10, full=False) @ base.coupling
    return DiagonalMemoryModel(base, port_kind="global" if kind == "nonlocal" else "local",
        port=port, tau=1/np.diag(base.latent_dissipation),
        encoder=np.zeros((10, 80)), feature_scale=np.ones(80))


def audit_conditional_sources(inputs, cases, budget):
    """Close the previously saved normal equations from their train operands."""
    f = inputs["raw_features"][:32]/inputs["feature_scale"]
    target = inputs["paths"][:32, 11:311, :20]
    scale = inputs["target_scale"]
    w = np.r_[np.full(100, 1/150), np.full(200, 1/600)]
    checks = {}
    for name, arrays in cases.items():
        if not np.array_equal(arrays["time_weights"], w) or not np.array_equal(arrays["train_target"], target):
            raise RuntimeError("saved conditional train target or weights changed")
        d = arrays["training_response"][1:, :20]/scale[None, :, None]
        y = (target-arrays["zero_training_forecast"][:32, 1:, :20])/scale
        budget.spend("matrix_products", 4)
        gf = f.T@f/32
        gz = np.einsum("t,tir,tis->rs", w, d, d)/20
        rhs = np.einsum("t,tir,bti->br", w, d, y).T@f/(32*20)
        row = {key: float(np.max(np.abs(value-arrays[key]))) for key, value in
               (("feature_gram", gf), ("response_gram", gz), ("rhs", rhs))}
        if any(value > 1e-10 for value in row.values()):
            raise RuntimeError("saved conditional normal-equation operands disagree")
        checks[name] = row
    return checks


def forecast(model, observed, power, history_scale, step, budget):
    q, v, eq = observed[:, :, :10], observed[:, :, 10:], np.zeros((len(observed), 10))
    result = estimate_history(model, q, v, power, eq, history_scale, max_step=step, spend=budget.spend)
    if result["resolved"]:
        with torch.no_grad():
            result["forecast"] = model.rollout(result["origin"], power, eq, 800,
                max_step=step, spend=budget.spend).numpy()
    return result


def evaluate(name, model, paths, scale, history_scale, noise, directory, budget, gates):
    observed, power = paths[:, :11, :20]+noise, paths[:, 0, 30:]
    fine = forecast(model, observed, power, history_scale, .0025, budget)
    refined = forecast(model, observed, power, history_scale, .00125, budget)
    arrays = {"observed": observed, "power": power, "scale": scale, "history_scale": history_scale}
    for prefix, result in (("fine", fine), ("refined", refined)):
        arrays.update({prefix+"__"+key: val for key, val in result.items() if isinstance(val, np.ndarray)})
    arrays.update({"model__"+key: val.detach().numpy() for key, val in model.state_dict().items()})
    row = {"case": name, "resolved": fine["resolved"] and refined["resolved"],
           "rank": fine["rank"], "condition_number": fine["condition_number"],
           "refined_rank": refined["rank"], "refined_condition_number": refined["condition_number"]}
    if row["resolved"]:
        refinement = float(np.max(np.abs((fine["forecast"][:, :, :20]-refined["forecast"][:, :, :20])/scale)))
        coi = check_paths((fine["history"], fine["forecast"], refined["history"], refined["forecast"]), model.mass.numpy(), budget)
        score = metrics(fine["forecast"], fine["history"], paths, scale)
        hidden_error = max(float(np.max(np.abs(fine["history"][:, j, 20:]-paths[:, j, 20:30]))) for j in (0, 10))
        row.update(metrics=score, refinement_error=refinement, coi_error=coi,
                   hidden_max_abs_error=hidden_error,
                   history_residual_rms=float(np.sqrt(np.mean(fine["residual"]**2))),
                   numerical_pass=refinement <= gates["step_difference"] and coi <= 1e-10,
                   hidden_clean_pass=all(r[key] <= gates["clean_hidden_rmse"] for r in score.values()
                       for key in ("hidden_start_rmse", "hidden_origin_rmse")))
        threshold = gates["noisy_nrmse"] if np.any(noise) else gates["clean_nrmse"]
        row["prediction_pass"] = all(r[key] <= threshold for r in score.values()
            for key in ("main_nrmse", "nrmse_3s_all"))
    else:
        row.update(numerical_pass=False, prediction_pass=False, hidden_clean_pass=False)
    np.savez_compressed(directory/(name+".npz"), **arrays)
    write_json(directory/(name+".json"), row)
    return row


def validate(root, directory, budget, plan):
    saved = {}
    with budget.scope("sources"):
        for name, entry in plan["saved_sources"].items():
            raw = (root/entry["path"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
                raise RuntimeError("observer source hash changed")
            budget.spend("npz_decodes")
            with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
                saved[name] = {key: archive[key].copy() for key in archive.files}
        inputs, prior = saved["inputs"], saved["fitted"]
        old_paths, old_scale = inputs["paths"], inputs["target_scale"]
        if old_paths.shape != (64, 811, 40) or old_scale.shape != (20,):
            raise RuntimeError("saved observer population changed")
        source_checks = audit_conditional_sources(inputs, {k: saved[k] for k in ("truth_dynamics", "fitted_dynamics")}, budget)
        fitted = new_model(fixture(), "L10", prior["feature_scale"])
        budget.spend("checkpoint_loads")
        fitted.load_state_dict({key: torch.tensor(prior[key]) for key in fitted.state_dict()}, strict=True)
        old_history_scale = np.maximum(np.sqrt(np.mean(old_paths[:32, :11, :20]**2, axis=(0, 1))), 1e-5)
        np.savez_compressed(directory/"old_inputs.npz", paths=old_paths, target_scale=old_scale,
                            history_scale=old_history_scale)
        write_json(directory/"conditional_source_audit.json", source_checks)
    data = {}
    for kind, seed in plan["truth_seeds"].items():
        with budget.scope("truth_"+kind):
            base, paths, scale, hscale = synthetic_data(kind, seed, budget)
            p = np.eye(10)-np.outer(base.mass, np.ones(10))/base.mass.sum()
            noise = np.random.default_rng(seed+100).normal(size=(64, 11, 20))*hscale[None, None, :]*.01
            budget.spend("matrix_products", 2)
            noise[:, :, :10], noise[:, :, 10:] = noise[:, :, :10]@p, noise[:, :, 10:]@p
            np.savez_compressed(directory/(kind+"_inputs.npz"), paths=paths, target_scale=scale,
                history_scale=hscale, noise=noise, mass=base.mass, coupling=base.coupling,
                decay=base.latent_dissipation, roles=np.array(["train"]*32+["val"]*16+["test"]*16))
            data[kind] = (base, paths, scale, hscale, noise)
    write_json(directory/"frozen_inputs.json", {p.name: sha256_file(p) for p in sorted(directory.glob("*_inputs.npz"))})
    cases = [("old_truth", known_model(fixture(), "local"), old_paths, old_scale, old_history_scale, 0.),
             ("old_fitted", fitted, old_paths, old_scale, old_history_scale, 0.)]
    for kind, (base, paths, scale, hscale, noise) in data.items():
        for noisy in (False, True):
            cases.append((kind+("_noise01" if noisy else "_clean"), known_model(base, kind),
                          paths, scale, hscale, noise if noisy else 0.))
    rows = {}
    for name, *args in cases:
        with budget.scope(name):
            rows[name] = evaluate(name, *args, directory, budget, plan["gates"])
    with budget.scope("rank_control"):
        base = fixture("equal")
        model = known_model(base, "local")
        zeros = np.zeros((1, 11, 10))
        rank = estimate_history(model, zeros, zeros, np.zeros((1, 10)), np.zeros((1, 10)),
                                np.ones(20), max_step=.0025, spend=budget.spend)
        np.savez_compressed(directory/"rank_control.npz", **{k: v for k, v in rank.items() if isinstance(v, np.ndarray)})
        rank_pass = not rank["resolved"] and rank["rank"] == 9 and "origin" not in rank
        rank_row = {"rank": rank["rank"], "resolved": rank["resolved"], "pass": rank_pass,
                    "candidate_future_calls": 0}
        write_json(directory/"rank_control.json", rank_row)
    numerical = rank_pass and all(r["numerical_pass"] for r in rows.values())
    matched = [rows[k] for k in ("local_clean", "local_noise01", "nonlocal_clean", "nonlocal_noise01")]
    clean = all(rows[k]["prediction_pass"] and rows[k]["hidden_clean_pass"] for k in
                ("old_truth", "local_clean", "nonlocal_clean"))
    noise_pass = all(rows[k]["prediction_pass"] for k in ("local_noise01", "nonlocal_noise01"))
    passed = bool(numerical and clean and noise_pass and all(r["resolved"] for r in matched))
    return {"scientific_pass": passed, "numerical_pass": bool(numerical), "clean_recovery_pass": bool(clean),
            "noise_prediction_pass": bool(noise_pass), "rank_control": rank_row, "records": rows,
            "old_fitted_prediction_pass": rows["old_fitted"]["prediction_pass"],
            "learned_model_confirmed": False, "t0_allowed": False,
            "next_route": "REGISTER_OBSERVER_DYNAMICS_TRAINING" if passed else "OBSERVER_REPAIR_REQUIRED",
            "claim_boundary": "Fixed-dynamics observer checks; new seeds test state recovery in the same linear families. Known parameters are oracle controls, not learned-model confirmation or real-data results."}


def execute(*, root, result_dir, stage, plan):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    start = time.perf_counter()
    budget = Budget(result_dir/"operations.jsonl", plan["stages"][stage]["operation_limits"], plan["stages"][stage]["scope_limits"])
    try:
        summary = validate(root, result_dir, budget, plan)
        summary.update(process_status="PASS", promotion_decision="KEEP_BASELINE", real_data_reads=0,
            operations=dict(budget.counts), operations_returned=dict(budget.returned), scopes=budget.scopes,
            source_commit=git_head(root), wall_seconds=time.perf_counter()-start, process_peak_memory=peak_memory(),
            runtime={"python": sys.version, "executable": sys.executable,
                     **{name: version(name) for name in ("numpy", "scipy", "torch")}})
        write_json(result_dir/"summary.json", summary)
        write_json(result_dir/"stage_evidence.json", {key: summary[key] for key in (
            "process_status", "promotion_decision", "source_commit", "claim_boundary", "scientific_pass")}
            | {"status": "PASS", "data_policy": "analytic_synthetic_only"})
        return summary
    except BaseException as error:
        write_json(result_dir/"failure.json", {"error_type": type(error).__name__, "message": str(error),
            "operations_attempted": dict(budget.counts), "retry_allowed": False})
        raise
    finally:
        budget.stream.close()
        write_json(result_dir/"artifact_hashes.json", {p.name: sha256_file(p)
            for p in sorted(result_dir.iterdir()) if p.is_file() and p.name != "artifact_hashes.json"})
