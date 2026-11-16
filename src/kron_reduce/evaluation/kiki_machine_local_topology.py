"""Paired topology fits on the complete TASK3 train and frozen stability scoring."""

from __future__ import annotations

import math
import sys
import time
from functools import partial
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from kron_reduce.data.durable_audit_ledger import AppendOnlyAuditJournal
from kron_reduce.data.task3_full import Task3FullReader
from kron_reduce.evaluation.constant_c_observability import write_json
from kron_reduce.evaluation.kiki_machine_local_full import evaluate, train_updates
from kron_reduce.evaluation.kiki_machine_local_r20_r21 import bind, read_json, verified
from kron_reduce.evaluation.kiki_topology_bridge import load_snapshot, make_model, predict
from kron_reduce.evaluation.machine_local_clean_learning import snapshot
from kron_reduce.evaluation.machine_local_synthetic import Budget
from kron_reduce.models.nonlinear_latent import LearnedNonlinearModel
from project_harness.evidence import sha256_file
from project_harness.git_state import git_head

SUPPORTS = ("T0M1", "T1M1")


def scales_from_train(events, mass):
    history = np.mean([np.mean(row["x"][:11]**2, axis=0) for row in events], axis=0)
    target = np.mean([np.mean(row["x"][11:311]**2, axis=0) for row in events], axis=0)
    hscale = np.maximum(np.sqrt(history), 1e-5)
    return {"history": hscale, "target": np.maximum(np.sqrt(target), 1e-5),
            "memory_radius": 3.*math.sqrt(float(mass @ hscale[10:]**2))}


def freeze_pair(root, directory, phase, models, plan, training):
    checkpoints = {}
    for support, model in models.items():
        file = directory/f"{phase}-{support}.npz"
        np.savez_compressed(file, **{key: value.cpu().numpy() for key, value in snapshot(model).items()})
        checkpoints[support] = bind(root, file)
    index = directory/f"{phase}-checkpoints.json"
    write_json(index, checkpoints)
    freeze = {"input_contract": "measured_history_100ms_zoh_v1", "source_commit": git_head(root),
              "phase": phase, "fitting_closed": phase == "freeze", "selection_closed": phase == "freeze",
              "settings": plan["settings"], "backbone": plan["backbone"], "training": training,
              "fit_plan_sha256": sha256_file(root/plan["plan_path"])}
    for key, path in (("checkpoint", index), ("preprocessing", directory/"preprocessing.json")):
        binding = bind(root, path)
        freeze[key+"_path"], freeze[key+"_sha256"] = binding["path"], binding["sha256"]
    path = directory/f"{phase}.json"
    write_json(path, freeze)
    return path


def selection_rank(result, threshold, update, *, trajectory_first=False):
    stability = result["stability"]
    failures = stability["allocated"]-stability["resolved"]
    errors = stability["allocated"]-stability["tp"]-stability["tn"]
    score = result["task3_six_term_score"]
    invalid_trajectory = trajectory_first and (stability["uncovered"] or stability["numerical_failures"])
    if score is None or not math.isfinite(score) or invalid_trajectory:
        return (2, failures, errors, update)
    if trajectory_first:
        return (0, score, update)
    if score > threshold:
        return (1, score, failures, errors, update)
    return (0, failures, errors, score, update)


def saturation_rank(result, reference, threshold, update):
    stability, anchor = result["stability"], reference["stability"]
    unresolved = stability["allocated"]-stability["resolved"]
    score = result["task3_six_term_score"]
    stable_8s = result["stable_complete"]["8"]
    anchor_8s = reference["stable_complete"]["8"]
    if score is None or not math.isfinite(score):
        return (2, unresolved, stability["fp"], -stability["tp"], update)
    if (unresolved == 0 and stability["numerical_failures"] == 0 and stability["fp"] <= anchor["fp"]
            and score <= threshold and stable_8s["valid"] and stable_8s["qualified"] == anchor_8s["qualified"]
            and stable_8s["mean"] is not None and stable_8s["mean"] <= 1.05*anchor_8s["mean"]):
        return (0, -stability["tp"], score, update)
    return (1, unresolved, stability["fp"], -stability["tp"], score, update)


def score(model, events, scales, budget, directory, role, settings, *, save=True):
    run = partial(predict, model, scale=scales["history"], spend=budget.spend,
                  memory_radius=scales["memory_radius"], prior_weight=settings["prior_weight"])

    def predict_batch(batch, step):
        return run(batch, step=step)

    return evaluate(model, events, scales["history"], budget, directory, role, settings,
                    predictor=predict_batch, target_scale=scales["target"], save=save)


def fit_pair(models, train, val, scales, budget, directory, settings, *, saturation_reference=None):
    bounds = (math.log(.5), math.log(2.))
    trajectory_first = "spatial_rate_limit" in settings
    predictor = partial(predict, memory_radius=scales["memory_radius"], prior_weight=settings["prior_weight"])
    active = {support: train_updates(model, train, scales, budget, directory/support, settings,
                                    stiffness_bounds=bounds, damping_bounds=bounds,
                                    spatial_rate_limit=settings.get("spatial_rate_limit")
                                    if model.spatial_relaxation_rates is not None else None, predictor=predictor,
                                    learning_rate=settings["learning_rate"], gradient_clip=1.)
              for support, model in models.items()}
    best, latest, statuses, checks = {}, {}, {}, []
    stale, reason = 0, "update_cap"
    try:
        for update in range(settings["updates"]+1):
            if update:
                for support, steps in tuple(active.items()):
                    try:
                        latest[support] = next(steps)
                    except StopIteration as done:
                        statuses[support] = done.value
                        del active[support]
                if not active:
                    reason = "all_training_failed"
                    break
            if update % settings["val_interval"] and update != settings["updates"]:
                continue
            record, improved = {"update": update, "groups": {}}, False
            for support in active:
                model = models[support]
                budget.spend("val_checks")
                result = score(model, val, scales, budget, directory/support, "val", settings, save=False)
                rank = (selection_rank(result, settings["val_trajectory_limit"], update, trajectory_first=trajectory_first)
                        if saturation_reference is None else
                        saturation_rank(result, saturation_reference, settings["val_trajectory_limit"], update))
                previous = best.get(support)
                positive = saturation_reference is None or float(model.saturation_strength.detach()) > 0.
                if positive and (previous is None or rank < previous[0]):
                    best[support] = (rank, snapshot(model))
                    if saturation_reference is not None or previous is None or rank[0] != previous[0][0]:
                        improved = True
                    elif trajectory_first and rank[0] == 0:
                        improved |= rank[1] < previous[0][1]*(1.-settings["val_relative_tol"])
                    elif rank[0] == 0:
                        improved |= (rank[1:3] < previous[0][1:3]
                                     or rank[3] < previous[0][3]*(1.-settings["val_relative_tol"]))
                    elif rank[0] == 1:
                        improved |= rank[1] < previous[0][1]*(1.-settings["val_relative_tol"])
                    else:
                        improved |= rank[1:3] < previous[0][1:3]
                record["groups"][support] = {"rank": list(rank), "stability": result["stability"],
                    "trajectory_score": result["task3_six_term_score"], "train": latest.get(support)}
                if saturation_reference is not None:
                    record["groups"][support].update(memory_saturation=float(model.saturation_strength.detach()),
                                                      stable_8s=result["stable_complete"]["8"])
            checks.append(record)
            write_json(directory/f"val-check-{update:03d}.json", record)
            print(f"val checkpoint {update}: { {key: row['rank'] for key, row in record['groups'].items()} }",
                  flush=True)
            stale = 0 if improved else stale+1
            if update >= settings["min_updates"] and stale >= settings["val_patience"]:
                reason = "joint_val_plateau"
                break
    finally:
        for steps in active.values():
            steps.close()
    groups = {}
    for support, (rank, state) in best.items():
        models[support].load_state_dict(state)
        models[support].requires_grad_(False).eval()
        groups[support] = {"completed_updates": latest.get(support, {}).get("update", 0),
                           "selected_update": rank[-1], "selected_rank": list(rank),
                           "status": statuses.get(support, {}).get("status", "COMPLETED"),
                           "message": statuses.get(support, {}).get("message")}
        if models[support].spatial_relaxation_rates is not None:
            groups[support]["selected_relaxation_rates"] = models[support].spatial_relaxation_rates.tolist()
        harmonics = getattr(models[support], "restoring_harmonics", None)
        if harmonics is not None:
            groups[support]["selected_restoring_harmonics"] = harmonics.tolist()
        if saturation_reference is not None:
            groups[support]["selected_memory_saturation"] = float(models[support].saturation_strength)
    training = {"stop_reason": reason, "groups": groups, "val_checks": len(checks)}
    write_json(directory/"selection.json", training)
    return training


def execute(root, directory, plan, stage):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(Path(plan["task3_root"])/"src"))
    settings = plan["settings"]
    budget = Budget(directory/"operations.jsonl", plan["operation_limits"])
    journal = AppendOnlyAuditJournal(directory/"access.jsonl")
    started = time.perf_counter()
    try:
        reader = Task3FullReader(root, plan["sources"], journal)
        budget.spend("checkpoint_loads")
        backbone = LearnedNonlinearModel.load(verified(root, plan["backbone"])).closure
        spatial_start = plan.get("paired_start")
        harmonic_start = plan.get("harmonic_start")
        saturation_start = plan.get("saturation_start")
        fixed_start = harmonic_start or saturation_start
        saturation_reference = None
        if fixed_start is not None:
            model = make_model(backbone, "T0M1", train_scales=True, device="cuda",
                               learn_harmonics=harmonic_start is not None, memory_saturation=saturation_start is not None)
            model.set_angle_damping(2./3., spatial_mix=1.)
            models = {"memory_saturation" if saturation_start else "es_sindy": model}
            if stage == "fit":
                anchor = make_model(backbone, "T0M1", train_scales=True, device="cuda")
                budget.spend("checkpoint_loads")
                load_snapshot(anchor, verified(root, fixed_start["checkpoint"]))
                model.load_state_dict(snapshot(model) | snapshot(anchor), strict=True)
                model.requires_grad_(False).train()
                if saturation_start:
                    model.saturation_strength.fill_(settings["memory_saturation_start"]).requires_grad_(True)
                    receipt = read_json(verified(root, plan["saturation_val_reference"]))
                    saturation_reference = receipt["records"]["es_sindy"]["val"]
                else:
                    model.restoring_harmonics.requires_grad_(True)
                del anchor
        elif spatial_start is None:
            models = {support: make_model(backbone, support, train_scales=True, device="cuda") for support in SUPPORTS}
        else:
            models = {group: make_model(backbone, "T0M1", train_scales=True, device="cuda")
                      for group in ("fixed", "spatial")}
            for model in models.values():
                model.set_angle_damping(2./3., spatial_mix=1.)
                if stage == "fit":
                    budget.spend("checkpoint_loads")
                    load_snapshot(model, verified(root, spatial_start["checkpoint"]))
                    if not math.isclose(float(model.effective_stiffness.sum()), spatial_start["latent_stiffness"],
                                        rel_tol=1e-13, abs_tol=1e-12):
                        raise RuntimeError("initial restoring stiffness differs from the fixed coupling storage scale")
                    model.requires_grad_(True).train()
            models["spatial"].set_trainable_spatial_coupling(stiffness=spatial_start["latent_stiffness"],
                                                          limit=math.log(2.)/math.sqrt(1.6))
        for support in models:
            (directory/support).mkdir()
        training = None
        if stage == "fit":
            train = reader.load("train")
            if not np.allclose(reader.static["mass"], backbone.mass, rtol=0, atol=1e-12):
                raise ValueError("equation and TASK3 mass differ")
            scales = scales_from_train(train, backbone.mass)
            scales_start = spatial_start or fixed_start
            if scales_start is not None:
                frozen_scales = read_json(verified(root, scales_start["preprocessing"]))
                if set(frozen_scales) != set(scales) or any(not np.allclose(scales[key], frozen_scales[key],
                        rtol=1e-13, atol=1e-14) for key in scales):
                    raise RuntimeError("full train preprocessing differs from the common T0M1 anchor")
                scales = frozen_scales
            write_json(directory/"preprocessing.json", {key: np.asarray(value).tolist() for key, value in scales.items()})
            write_json(directory/"preprocessing-roster.json", {"ids": [row["id"] for row in train],
                                                              "history_points": 11})
            initial = freeze_pair(root, directory, "initial", models, plan, None)
            val = reader.load("val", freeze_path=initial)
            training = fit_pair(models, train, val, scales, budget, directory, settings,
                                saturation_reference=saturation_reference)
            freeze_pair(root, directory, "freeze", models, plan, training)
            roles = {"train": train, "val": val}
        else:
            freeze_path = verified(root, plan["frozen_model_manifest"])
            frozen = read_json(freeze_path)
            if (frozen["phase"] != "freeze" or frozen["fitting_closed"] is not True or frozen["selection_closed"] is not True
                    or frozen["settings"] != settings or frozen["backbone"] != plan["backbone"]
                    or frozen["fit_plan_sha256"] != plan["fit_plan_sha256"]):
                raise RuntimeError("test requires the exact selected pair and frozen fit inputs")
            assets = {key: verified(root, {"path": frozen[key+"_path"], "sha256": frozen[key+"_sha256"]})
                      for key in ("checkpoint", "preprocessing")}
            checkpoints, scales = (read_json(assets[key]) for key in ("checkpoint", "preprocessing"))
            if set(checkpoints) != set(models) or set(scales) != {"history", "target", "memory_radius"}:
                raise RuntimeError("frozen model pair or preprocessing differs")
            for support, model in models.items():
                budget.spend("checkpoint_loads")
                load_snapshot(model, verified(root, checkpoints[support]))
                rates = model.spatial_relaxation_rates
                if rates is not None and (torch.any(rates < 0.) or torch.any(rates > settings["spatial_rate_limit"])):
                    raise RuntimeError("frozen spatial relaxation rates exceed the signed range")
                harmonics = getattr(model, "restoring_harmonics", None)
                if harmonics is not None and torch.any(harmonics.abs() > settings["harmonic_bound"]):
                    raise RuntimeError("frozen restoring coefficients exceed the signed range")
                if saturation_start and not 0. < float(model.saturation_strength) <= settings["memory_saturation_bound"]:
                    raise RuntimeError("frozen saturation coefficient exceeds the positive scope")
            roles = {"test": reader.load("test", freeze_path=freeze_path)}

        if not np.allclose(reader.static["mass"], backbone.mass, rtol=0, atol=1e-12):
            raise ValueError("equation and TASK3 mass differ")
        records = {}
        for group, model in models.items():
            records[group] = {}
            for role, events in roles.items():
                result = score(model, events, scales, budget, directory/group, role, settings)
                records[group][role] = {key: result[key] for key in
                                        ("summary", "unstable_prefix", "stability", "task3_six_term_score")}
                if spatial_start is not None or fixed_start is not None:
                    records[group][role]["stable_complete"] = result["stable_complete"]
                if saturation_start and role == "train" and "matched_train_witnesses" in result:
                    records[group][role]["matched_train_witnesses"] = result["matched_train_witnesses"]
        summaries = [value["test"]["stability"] for value in records.values()] if stage == "test" else []
        pass_goal = any(s["allocated"] == s["resolved"] == 76 and s["accuracy"] is not None
                        and s["tp"]+s["tn"] >= 69 for s in summaries)
        if saturation_reference is not None:
            selected = records["memory_saturation"]["val"]
            pass_goal = (saturation_rank(selected, saturation_reference, settings["val_trajectory_limit"], 0)[0] == 0
                         and selected["stability"]["tp"] > saturation_reference["stability"]["tp"]
                         and training["groups"]["memory_saturation"]["status"] == "COMPLETED")
        evidence = {"status": "PASS", "process_status": "PASS", "source_commit": git_head(root),
                    "promotion_decision": "KEEP_BASELINE", "data_policy": "kiki_official_train_test_v1",
                    "claim_boundary": ("Fixed TASK3 comparison; one frozen fixed-topology candidate."
                                       if fixed_start else "Fixed TASK3 cohort and input contract; frozen paired-model comparison."),
                    "scientific_pass": pass_goal}
        summary = evidence | {"records": records, "operations": dict(budget.counts), "training": training,
                              "wall_seconds": time.perf_counter()-started, "runtime": {
            "python": sys.version, "executable": sys.executable, "device": "cuda",
            "cuda_build": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
            **{package: version(package) for package in ("torch", "numpy", "scipy", "filelock", "h5py")}}}
        write_json(directory/"summary.json", summary)
        write_json(directory/"stage_evidence.json", evidence)
        return summary
    finally:
        budget.stream.close()
        journal.close()
        write_json(directory/"artifact_hashes.json", {path.relative_to(directory).as_posix(): sha256_file(path)
                   for path in sorted(directory.rglob("*")) if path.is_file()})
