"""Bounded separation of encoder optimization, generalization and dynamics error."""
from __future__ import annotations

import hashlib
import io
import sys
import time
from importlib.metadata import version

import numpy as np
import torch

from kron_reduce.evaluation.constant_c_observability import write_json
from kron_reduce.evaluation.machine_local_synthetic import (
    Budget,
    _features_batch,
    fixture,
    new_model,
    peak_memory,
)
from kron_reduce.models.conditional_encoder import causal_affine_maps, solve_conditional_encoder
from kron_reduce.models.machine_local import DiagonalMemoryModel
from project_harness.evidence import sha256_file
from project_harness.git_state import git_head


def metrics(future, history, truth, scale):
    if (not np.isfinite(future).all() or not np.isfinite(history).all()
            or np.max(np.abs(future)) > 1e12 or np.max(np.abs(history)) > 1e12):
        raise FloatingPointError("nonfinite diagnostic prediction")
    rows = {}
    for role, indices in (("train", slice(0, 32)), ("val", slice(32, 48)), ("test", slice(48, 64))):
        row = {}
        for seconds in (1, 3, 8):
            raw = future[indices, 1:100*seconds+1, :20]-truth[indices, 11:11+100*seconds, :20]
            for name, channel in (("all", slice(0, 20)), ("q", slice(0, 10)), ("v", slice(10, 20))):
                row[f"nrmse_{seconds}s_{name}"] = float(np.sqrt(np.mean((raw/scale)[:, :, channel]**2)))
                row[f"rmse_{seconds}s_{name}"] = float(np.sqrt(np.mean(raw[:, :, channel]**2)))
                row[f"max_abs_{seconds}s_{name}"] = float(np.max(np.abs(raw[:, :, channel])))
        row["main_nmse"] = (row["nrmse_1s_all"]**2+row["nrmse_3s_all"]**2)/2
        row["main_nrmse"] = float(np.sqrt(row["main_nmse"]))
        for name, position in (("start", 0), ("origin", 10)):
            z = history[indices, position, 20:]
            row[f"hidden_{name}_rmse"] = float(np.sqrt(np.mean((z-truth[indices, position, 20:30])**2)))
            row[f"hidden_{name}_energy_mean"] = float(np.mean(np.sum(z*z, axis=1)/2))
        rows[role] = row
    return rows


def check_paths(paths, mass, budget):
    worst = 0.
    for path in paths:
        if not np.isfinite(path).all() or np.max(np.abs(path)) > 1e12:
            raise FloatingPointError("nonfinite or divergent reconstructed path")
        budget.spend("matrix_products", 2)
        worst = max(worst, float(np.max(np.abs(path[:, :, :10]@mass))),
                    float(np.max(np.abs(path[:, :, 10:20]@mass))))
    return worst


def reconstruct(maps, features, encoder, budget):
    h0, f0, dh, df = maps
    budget.spend("matrix_products", 1+len(dh)+len(df))
    zh = features@encoder.T
    return h0+np.einsum("tir,br->bti", dh, zh), f0+np.einsum("tir,br->bti", df, zh)


def objective(path, target, scale, weights, encoder):
    main = float(np.mean(np.einsum("t,bti->bi", weights, ((path[:, 1:, :20]-target)/scale)**2)))
    return main+1e-4*float(np.sum(encoder**2))/160


def information(features, observed, directory, budget):
    budget.spend("svd", 2)
    _, singular, vh = np.linalg.svd(features[:32], full_matrices=False)
    threshold = 64*np.finfo(float).eps*max(features[:32].shape)*singular[0]
    rank = int(np.count_nonzero(singular > threshold))
    budget.spend("matrix_products", 2)
    projection = (features@vh[:rank].T)@vh[:rank]
    relative = np.linalg.norm(features-projection, axis=1)/np.maximum(np.linalg.norm(features, axis=1), 1e-15)
    blocks = observed["local_design"].reshape(11, 20, 10)
    scales = np.r_[np.full(10, .2), np.full(10, .1)]
    o3 = (blocks[[0, 5, 10]]/scales[None, :, None]).reshape(60, 10)
    _, s3, v3 = np.linalg.svd(o3, full_matrices=False)
    s11 = observed["local_scaled_singular"]
    rank3 = int(np.count_nonzero(s3 > 64*np.finfo(float).eps*max(o3.shape)*s3[0]))
    rank11 = int(np.count_nonzero(s11 > 64*np.finfo(float).eps*220*s11[0]))
    denominator = float(np.sum(features[32:48]**2))
    coverage = float(np.sum((features[32:48]-projection[32:48])**2)/denominator) if denominator else None
    ratio = float((s3[-1]/np.sqrt(3))/(s11[-1]/np.sqrt(11)))
    row = {"feature_rank": rank, "feature_rank_threshold": float(threshold),
           "feature_singular_values": singular.tolist(),
           "three_frame_singular_values": s3.tolist(), "eleven_frame_singular_values": s11.tolist(),
           "three_frame_per_sample_singular_values": (s3/np.sqrt(3)).tolist(),
           "eleven_frame_per_sample_singular_values": (s11/np.sqrt(11)).tolist(),
           "three_frame_rank": rank3, "eleven_frame_rank": rank11,
           "val_outside_train_span_energy_ratio": coverage,
           "val_zero_features": denominator == 0,
           "coverage_gap": coverage is not None and coverage > 1e-8,
           "three_frame_rank_loss": rank3 < 10 and rank11 == 10,
           "per_frame_minimum_singular_ratio": ratio, "three_frame_weakening": ratio < .1,
           "feature_outside_train_span_mean": {name: float(np.mean(relative[a:b])) for name, a, b in
               (("train", 0, 32), ("val", 32, 48), ("test", 48, 64))}}
    np.savez_compressed(directory/"information.npz", features=features, singular=singular, directions=vh,
                        projection=projection, outside_train_span=relative, three_frame_design=o3,
                        three_frame_singular=s3, three_frame_directions=v3, eleven_frame_singular=s11,
                        eleven_frame_design=(blocks/scales[None, :, None]).reshape(220, 10))
    write_json(directory/"information.json", row)
    return row


def evaluate_theta(name, model, truth, scale, raw_features, features, directory, budget):
    q, v, power = truth[:, :11, :10], truth[:, :11, 10:20], truth[:, 0, 30:]
    eq = np.zeros((64, 10))
    old_encoder = model.encoder.detach().numpy().copy()
    weights = np.r_[np.full(100, 1/150), np.full(200, 1/600)]
    training_maps = causal_affine_maps(model, q, v, power, eq, 300, max_step=.005, spend=budget.spend)
    target = truth[:32, 11:311, :20]
    budget.spend("fits")
    solution = solve_conditional_encoder(features[:32], target-training_maps[1][:32, 1:, :20],
        training_maps[3][1:, :20], weights, scale, penalty=1e-4/160, spend=budget.spend)
    encoder = solution["encoder"]
    h_train, f_train = reconstruct(training_maps, features, encoder, budget)
    _, f_old = reconstruct(training_maps, features, old_encoder, budget)
    old_objective = objective(f_old[:32], target, scale, weights, old_encoder)
    new_objective = objective(f_train[:32], target, scale, weights, encoder)
    with torch.no_grad():
        model.encoder.copy_(torch.tensor(encoder))
        direct = model.causal_rollout(q, v, power, eq, raw_features, 300,
                                     max_step=.005, spend=budget.spend).numpy()
    affine_error = float(np.max(np.abs(f_train-direct)))
    fine_maps = causal_affine_maps(model, q, v, power, eq, 800, max_step=.0025, spend=budget.spend)
    history, future = reconstruct(fine_maps, features, encoder, budget)
    finer_maps = causal_affine_maps(model, q, v, power, eq, 800, max_step=.00125, spend=budget.spend)
    finer_history, finer = reconstruct(finer_maps, features, encoder, budget)
    refinement = float(np.max(np.abs((future[:, :, :20]-finer[:, :, :20])/scale)))
    mass = model.mass.numpy()
    coi = check_paths((history, future, finer_history, finer, h_train, f_train, f_old, direct), mass, budget)
    symmetry = float(np.linalg.norm(solution["system"]-solution["system"].T)/max(1., float(np.linalg.norm(solution["system"]))))
    checks = {"affine": affine_error <= 1e-10, "symmetric": symmetry <= 1e-12,
              "optimality": solution["optimality_gap_upper_bound"] <= 1e-12,
              "objective_nonincrease": new_objective <= old_objective+1e-10,
              "refinement": refinement <= 1e-5, "coi": coi <= 1e-10}
    row = {"metrics": metrics(future, history, truth, scale), "checks": checks,
           "old_train_objective": old_objective, "conditional_train_objective": new_objective,
           "affine_error": affine_error, "symmetry_error": symmetry,
           "optimality_gap_upper_bound": solution["optimality_gap_upper_bound"],
           "refinement_error": refinement, "coi_error": coi}
    arrays = {k: value for k, value in solution.items() if isinstance(value, np.ndarray)}
    arrays.update(history=history, forecast=future, refined_forecast=finer, refined_history=finer_history,
                  training_history=h_train, training_forecast=f_train, direct_training_forecast=direct,
                  old_training_forecast=f_old, old_encoder=old_encoder, time_weights=weights,
                  zero_training_forecast=training_maps[1], training_response=training_maps[3],
                  zero_history=fine_maps[0], zero_forecast=fine_maps[1],
                  history_response=fine_maps[2], forecast_response=fine_maps[3], train_target=target)
    np.savez_compressed(directory/(name+".npz"), **arrays)
    write_json(directory/(name+".json"), row)
    return row


def diagnose(root, directory, budget, plan):
    saved = {}
    with budget.scope("sources"):
        for key, entry in plan["saved_sources"].items():
            path = root/entry["path"]
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
                raise RuntimeError("diagnostic source hash changed")
            budget.spend("npz_decodes")
            with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
                saved[key] = {name: archive[name].copy() for name in archive.files}
        data, prior = saved["truth"], saved["fitted"]
        truth, scale = data["paths"], data["target_scale"]
        if truth.shape != (64, 811, 40) or not np.array_equal(data["roles"], ["train"]*32+["val"]*16+["test"]*16):
            raise RuntimeError("diagnostic saved population changed")
        q, v, power = truth[:, :11, :10], truth[:, :11, 10:20], truth[:, 0, 30:]
        raw_features = _features_batch(q, v, power, np.zeros((64, 10)))
        if not np.array_equal(prior["feature_scale"], np.maximum(np.sqrt(np.mean(raw_features[:32]**2, axis=0)), 1e-5)):
            raise RuntimeError("saved feature scale is not the frozen train RMS")
        features = raw_features/prior["feature_scale"]
        model = new_model(fixture(), "L10", prior["feature_scale"])
        budget.spend("checkpoint_loads")
        model.load_state_dict({name: torch.tensor(prior[name]) for name in model.state_dict()}, strict=True)
        true_model = DiagonalMemoryModel(fixture(), port_kind="local", port=.35+.025*np.arange(10),
            tau=np.geomspace(.05, 2., 10), encoder=prior["encoder"], feature_scale=prior["feature_scale"])
        if (not np.array_equal(scale, np.maximum(np.sqrt(np.mean(truth[:32, 11:311, :20]**2, axis=(0, 1))), 1e-5))
                or any(np.max(np.abs(data[key]-value)) > 1e-12 for key, value in {
                    "coupling": true_model.coupling.detach().numpy(), "mass": true_model.mass.numpy(),
                    "stiffness": true_model.stiffness.numpy(), "damping": true_model.damping.numpy(),
                    "decay": np.diag(1/true_model.tau.detach().numpy())}.items())
                or np.max(np.abs(saved["observability"]["local_coupling"]-data["coupling"])) > 1e-12):
            raise RuntimeError("truth model or train-only scale mismatch")
        np.savez_compressed(directory/"diagnostic_inputs.npz", paths=truth, target_scale=scale,
            raw_features=raw_features, feature_scale=prior["feature_scale"], original_encoder=prior["encoder"],
            original_forecast=prior["forecast"], original_history=prior["history"], mass=data["mass"])
    with budget.scope("information"):
        info = information(features, saved["observability"], directory, budget)
    with budget.scope("oracle"):
        with torch.no_grad():
            oracle = true_model.rollout(truth[:, 10, :30], power, np.zeros((64, 10)), 800,
                max_step=.0025, spend=budget.spend).numpy()
            refined = true_model.rollout(truth[:, 10, :30], power, np.zeros((64, 10)), 800,
                max_step=.00125, spend=budget.spend).numpy()
        oracle_metrics = metrics(oracle, truth[:, :11, :30], truth, scale)
        oracle_pass = all(row[f"nrmse_{s}s_all"] <= 1e-5 for row in oracle_metrics.values() for s in (1, 3, 8))
        oracle_refinement = float(np.max(np.abs((oracle[:, :, :20]-refined[:, :, :20])/scale)))
        oracle_coi = check_paths((oracle, refined), data["mass"], budget)
        oracle_pass = bool(oracle_pass and oracle_coi <= 1e-10)
        np.savez_compressed(directory/"oracle.npz", forecast=oracle, refined_forecast=refined)
    rows = {}
    for name, candidate in (("truth_dynamics", true_model), ("fitted_dynamics", model)):
        with budget.scope(name):
            rows[name] = evaluate_theta(name, candidate, truth, scale, raw_features, features, directory, budget)
    original = metrics(prior["forecast"], prior["history"], truth, scale)
    technical = all(all(r["checks"].values()) for r in rows.values()) and oracle_refinement <= 1e-5
    flags = {}
    for name, row in rows.items():
        flags[name] = {role: {"clean_target_pass": all(row["metrics"][role][key] <= .001 for key in
            ("main_nrmse", "nrmse_3s_all")), "improves_original_nmse_5pct":
            original[role]["main_nmse"] > 0 and row["metrics"][role]["main_nmse"] <= .95*original[role]["main_nmse"]}
            for role in ("train", "val")}
    a, b = rows["truth_dynamics"]["metrics"]["val"], rows["fitted_dynamics"]["metrics"]["val"]
    dynamics_gap = b["main_nmse"] > 1.25*a["main_nmse"]+1e-12 and b["nrmse_3s_all"] > a["nrmse_3s_all"]
    route = ("DIAGNOSTIC_INVALID" if not technical or not oracle_pass else
             "INITIALIZER_LIMITATION_PRIORITY" if not flags["truth_dynamics"]["val"]["clean_target_pass"] else
             "DYNAMICS_REPAIR_PRIORITY" if not flags["fitted_dynamics"]["val"]["clean_target_pass"] else
             "CONDITIONAL_ENCODER_REPAIR_CANDIDATE")
    return {"scientific_pass": bool(technical and oracle_pass), "oracle_pass": oracle_pass,
            "oracle_refinement": oracle_refinement, "oracle_coi": oracle_coi, "oracle_metrics": oracle_metrics,
            "original_joint_metrics": original, "conditional": rows, "information": info,
            "diagnostic_flags": flags, "dynamics_gap": bool(dynamics_gap), "route": route,
            "t0_allowed": False, "claim_boundary": "Observed synthetic diagnostic; train-only conditional fits, val-based routing. No independent confirmation or real-data result."}


def execute(*, root, result_dir, stage, plan):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    start = time.perf_counter()
    budget = Budget(result_dir/"operations.jsonl", plan["stages"][stage]["operation_limits"], plan["stages"][stage]["scope_limits"])
    try:
        summary = diagnose(root, result_dir, budget, plan)
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
