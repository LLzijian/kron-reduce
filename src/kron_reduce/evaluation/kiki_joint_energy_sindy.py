"""One from-scratch joint energy-SINDy fit and frozen TASK3 evaluation."""
from __future__ import annotations

import json
import math
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from kron_reduce.data.durable_audit_ledger import AppendOnlyAuditJournal
from kron_reduce.data.task3_full import Task3FullReader
from kron_reduce.evaluation.constant_c_observability import write_json
from kron_reduce.evaluation.joint_energy_sindy_training import (
    causal_rollout,
    checkpoint_eligible,
    group_regularization,
    prune_due,
    training_controls,
    update_learning_rate,
)
from kron_reduce.evaluation.joint_energy_sindy_weak import initialize as weak_initialize
from kron_reduce.evaluation.kiki_machine_local_full import evaluate, inputs
from kron_reduce.evaluation.kiki_machine_local_r20_r21 import bind, read_json, verified
from kron_reduce.evaluation.machine_local_clean_learning import NUMERICAL_ERRORS, snapshot
from kron_reduce.evaluation.machine_local_synthetic import Budget
from kron_reduce.evaluation.state_port_training import loss
from kron_reduce.models.joint_energy_sindy import GROUPS, JointEnergySINDy
from kron_reduce.models.machine_local_topology import predict_from_topology_history
from project_harness.evidence import sha256_file
from project_harness.git_state import git_head


def predict(model, events, scale, step, spend, *, count, memory_radius, prior_weight):
    result = predict_from_topology_history(model, *inputs(events, scale), count,
                                           max_step=step, spend=spend,
                                           memory_radius=memory_radius, prior_weight=prior_weight)
    return {"forecast": result["forecast"][:, :, :20], "history": result["history"],
            "hidden_start": result["hidden_start"], "gn_singular": result["singular"][None]}


def score(model, events, scales, budget, directory, role, settings, plan, *, save=True):
    def forecast(batch, step):
        return predict(model, batch, scales["history"], step, budget.spend, count=800,
                       memory_radius=settings["memory_radius"], prior_weight=settings["prior_weight"])

    return evaluate(model, events, scales["history"], budget, directory, role, settings,
                    predictor=forecast, target_scale=scales["target"], save=save)


def freeze(root, directory, phase, model, scales, plan, training):
    checkpoint = directory/f"{phase}-model.npz"
    np.savez_compressed(checkpoint, **{key: value.cpu().numpy() for key, value in snapshot(model).items()})
    assets = {"checkpoint": checkpoint, "preprocessing": directory/"preprocessing.json"}
    manifest = {"phase": phase, "fitting_closed": phase == "freeze",
                "selection_closed": phase == "freeze", "source_commit": git_head(root),
                "input_contract": "measured_history_100ms_zoh_v1",
                "fit_plan_sha256": sha256_file(root/plan["plan_path"]),
                "settings": plan["settings"],
                "training": training}
    for name, path in assets.items():
        binding = bind(root, path)
        manifest[name+"_path"], manifest[name+"_sha256"] = binding["path"], binding["sha256"]
    path = directory/f"{phase}.json"
    write_json(path, manifest)
    return path


def load_selected(model, path, pruned):
    current = model.state_dict()
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(current):
            raise ValueError("frozen joint state keys differ")
        values = {}
        for name, original in current.items():
            value = archive[name]
            if value.shape != tuple(original.shape) or value.dtype != np.float64 or not np.isfinite(value).all():
                raise ValueError(f"invalid frozen joint tensor: {name}")
            if name not in dict(model.named_parameters()) and name != "group_gate" and not np.array_equal(
                    value, original.cpu().numpy()):
                raise ValueError(f"frozen joint structural buffer differs: {name}")
            values[name] = torch.from_numpy(value.copy()).to(original.device)
    expected = np.array([0. if name in pruned else 1. for name in GROUPS])
    if set(pruned)-set(GROUPS) or not np.array_equal(values["group_gate"].cpu().numpy(), expected):
        raise ValueError("frozen optional-group mask differs from selected support")
    model.load_state_dict(values, strict=True)
    return model.requires_grad_(False).eval()


def train_objective(model, batch, population, scales, budget, settings, plan):
    counts = np.array([min(settings["fit_frames"], len(event["x"])-11) for event in batch])
    count = int(counts.max())
    target = np.stack([np.pad(event["x"][10:11+length], ((0, count-length), (0, 0)), mode="edge")
                       for event, length in zip(batch, counts, strict=True)])
    # The same coefficients determine both the inferred origin and the future.
    future = causal_rollout(model, *inputs(batch, scales["history"]), count,
                            settings=settings, spend=budget.spend)
    value = loss(future[:, :, :20], model._tensor(target),
                 model._tensor(np.full(len(batch), 1/population)), model._tensor(scales["target"]), counts)
    if not torch.isfinite(value):
        raise FloatingPointError("nonfinite train objective")
    return value


def rank(result, update):
    stability, value = result["stability"], result["task3_six_term_score"]
    missing = stability["allocated"]-stability["resolved"]
    if missing or value is None or not math.isfinite(value):
        return (1, missing, stability["numerical_failures"], update)
    return (0, value, -stability["tp"], stability["fp"], update)


def fit(model, train, val, scales, budget, directory, settings, plan):
    controls = training_controls(settings)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                                 lr=settings["learning_rate"], foreach=False, fused=False)
    best, fallback, latest, checks = None, None, None, []
    completed, reason, failure = 0, "update_cap", None
    pruned = []
    started = time.perf_counter()
    budget.spend("fits")
    with (directory/"training.jsonl").open("x", encoding="utf-8") as stream:
        for update in range(settings["updates"]+1):
            if update:
                limit = settings.get("max_training_seconds")
                if limit is not None and time.perf_counter()-started >= limit:
                    reason = "wall_time_cap"
                    break
                previous = snapshot(model)
                optimizer.zero_grad(set_to_none=True)
                learning_rate = update_learning_rate(optimizer, settings, update)
                objective = 0.
                try:
                    for offset in range(0, len(train), settings["train_batch"]):
                        value = train_objective(model, train[offset:offset+settings["train_batch"]],
                                                len(train), scales, budget, settings, plan)
                        budget.spend("gradients")
                        value.backward()
                        objective += float(value.detach())
                    penalty = group_regularization(model, settings)
                    if penalty is not None:
                        penalty.backward()
                    parameters = [p for p in model.parameters() if p.requires_grad]
                    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
                        raise FloatingPointError("nonfinite train gradient")
                    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
                    if not torch.isfinite(norm):
                        raise FloatingPointError("nonfinite gradient norm")
                    budget.spend("optimizer_updates")
                    optimizer.step()
                    if (any(not torch.isfinite(p).all() for p in parameters)
                            or any(not torch.isfinite(value).all() for state in optimizer.state.values()
                                   for value in state.values() if isinstance(value, torch.Tensor))):
                        raise FloatingPointError("nonfinite updated parameter or optimizer state")
                    completed = update
                except NUMERICAL_ERRORS as error:
                    np.savez_compressed(directory/"failed-parameters.npz",
                                        **{k: v.cpu().numpy() for k, v in snapshot(model).items()})
                    model.load_state_dict(previous)
                    reason = "numerical_failure"
                    failure = {"update_attempt": update, "batch_offset": offset,
                               "error_type": type(error).__name__, "message": str(error)}
                    write_json(directory/"training-failure.json", failure)
                    break
                if prune_due(settings, update):
                    norms = {key: float(value.detach()) for key, value in model.group_norms().items()}
                    pruned = [key for key, value in norms.items() if key not in ("common", "spatial")
                              and value < settings["support_threshold"]]
                    model.prune_groups(pruned)
                    write_json(directory/"pruning.json", {"update": update, "threshold": settings["support_threshold"],
                                                          "norms_before": norms, "pruned_groups": pruned})
                latest = {"update": update, "train_nmse_before_update": objective,
                          "train_nmse_1s": objective, "train_nmse_timing": "before_update",
                          "learning_rate": learning_rate,
                          "gradient_norm": float(norm),
                          "group_norms": {key: float(value.detach()) for key, value in model.group_norms().items()}}
                stream.write(json.dumps(latest, allow_nan=False)+"\n")
                stream.flush()
                print(f"joint update {update}/{settings['updates']} train NMSE={objective:.6g}", flush=True)
            if update % settings["val_interval"] and update != settings["updates"]:
                continue
            budget.spend("val_checks")
            result = score(model, val, scales, budget, directory, "val", settings, plan, save=False)
            with torch.no_grad():
                train_after = sum(float(train_objective(model, train[offset:offset+settings["train_batch"]],
                                      len(train), scales, budget, settings, plan))
                                  for offset in range(0, len(train), settings["train_batch"]))
            selected_rank = rank(result, update)
            row = {"update": update, "rank": list(selected_rank),
                   "task3_six_term_score": result["task3_six_term_score"],
                   "stability": result["stability"], "train": latest,
                   "train_nmse_after_update": train_after}
            checks.append(row)
            write_json(directory/f"val-check-{update:03d}.json", row)
            if checkpoint_eligible(settings, update):
                if best is None or selected_rank < best[0]:
                    best = (selected_rank, snapshot(model))
            elif fallback is None or selected_rank < fallback[0]:
                fallback = (selected_rank, snapshot(model))
            print(f"val checkpoint {update}: {selected_rank}", flush=True)
    if best is None and fallback is not None and failure:
        # A numerical failure before pruning still receives complete frozen cohort reporting.
        best, reason = fallback, "numerical_failure_before_pruning"
    if best is None:
        raise RuntimeError("no full-val checkpoint was assessed")
    model.load_state_dict(best[1])
    model.requires_grad_(False).eval()
    # Report support from the selected checkpoint, not from a later discarded update.
    pruned = [name for name, gate in zip(GROUPS, model.group_gate.tolist(), strict=True) if not gate]
    training = {"status": "NUMERICAL_FAILURE" if failure else ("STOPPED_WALL_TIME" if reason == "wall_time_cap" else "COMPLETED"),
                "stop_reason": reason, "completed_updates": completed,
                "selected_update": best[0][-1], "selected_rank": list(best[0]),
                "val_checks": len(checks), "failure": failure,
                "prune_update": settings.get("prune_update") if controls["support_policy"] == "legacy_threshold" else None,
                "support_threshold": settings.get("support_threshold") if controls["support_policy"] == "legacy_threshold" else None,
                "training_wall_seconds": time.perf_counter()-started, **controls,
                "pruned_groups": pruned, "active_groups": [key for key in model.group_norms() if key not in pruned],
                "selected_group_norms": {key: float(value) for key, value in model.group_norms().items()}}
    write_json(directory/"selection.json", {"training": training, "checks": checks})
    return training


def execute(root, directory, plan, stage):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(Path(plan["task3_root"])/"src"))
    if plan["settings"]["topology_radius"] != .6:
        raise ValueError("registered topology radius differs from causal observer")
    settings = plan["settings"]
    device = settings.get("device", "cuda")
    budget = Budget(directory/"operations.jsonl", plan["operation_limits"])
    journal = AppendOnlyAuditJournal(directory/"access.jsonl")
    started = time.perf_counter()
    try:
        reader = Task3FullReader(root, plan["sources"], journal)
        training, weak = None, None
        if stage == "fit":
            train = reader.load("train")
            mass = reader.static["mass"]
            scales = {"history": np.maximum(np.sqrt(np.mean([np.mean(e["x"][:11]**2, axis=0)
                                                             for e in train], axis=0)), 1e-5),
                      "target": np.maximum(np.sqrt(np.mean([np.mean(e["x"][11:311]**2, axis=0)
                                                            for e in train], axis=0)), 1e-5),
                      "mass": mass}
            write_json(directory/"preprocessing.json", {key: value.tolist() for key, value in scales.items()})
            write_json(directory/"train-scale-population.json", {"ids": [e["id"] for e in train],
                "history_points": 11, "target_points": [min(300, len(e["x"])-11) for e in train]})
            model = JointEnergySINDy.from_mass(mass, physical_seed=settings.get("physical_seed")).to(device)
            seed_policy = settings.get("weak_initialization", "train_integral")
            if seed_policy == "off":
                weak = {"policy": "off", "reason": "fit the complete physical seed without omitted-term weak calibration"}
            elif seed_policy == "train_integral":
                weak = weak_initialize(model, train, budget, frames=settings["weak_frames"])
                weak["policy"] = seed_policy
            else:
                raise ValueError("weak_initialization must be off or train_integral")
            write_json(directory/"weak-initialization.json", weak)
            write_json(directory/"initial-coefficients.json", {
                name: value.detach().cpu().tolist() for name, value in model.effective_coefficients().items()})
            initial = freeze(root, directory, "initial", model, scales, plan, None)
            val = reader.load("val", freeze_path=initial)
            training = fit(model, train, val, scales, budget, directory, settings, plan)
            freeze(root, directory, "freeze", model, scales, plan, training)
            roles = {"train": train, "val": val}
        elif stage == "test":
            frozen_path = verified(root, plan["frozen_model_manifest"])
            frozen = read_json(frozen_path)
            if (frozen["phase"] != "freeze" or frozen["fitting_closed"] is not True
                    or frozen["selection_closed"] is not True
                    or frozen["input_contract"] != "measured_history_100ms_zoh_v1"
                    or frozen["settings"] != settings
                    or frozen["fit_plan_sha256"] != plan["fit_plan_sha256"]):
                raise RuntimeError("test requires exact closed fit and selection contract")
            scales = read_json(verified(root, {"path": frozen["preprocessing_path"],
                                               "sha256": frozen["preprocessing_sha256"]}))
            if set(scales) != {"history", "target", "mass"}:
                raise RuntimeError("frozen preprocessing keys differ")
            model = JointEnergySINDy.from_mass(np.asarray(scales["mass"])).to(device)
            budget.spend("checkpoint_loads")
            load_selected(model, verified(root, {"path": frozen["checkpoint_path"],
                                                 "sha256": frozen["checkpoint_sha256"]}),
                          frozen["training"]["pruned_groups"])
            roles = {"test": reader.load("test", freeze_path=frozen_path)}
            if not np.allclose(reader.static["mass"], scales["mass"], rtol=0, atol=1e-12):
                raise ValueError("frozen model and TASK3 mass differ")
        else:
            raise ValueError("joint TASK3 stage must be fit or test")
        if any(not np.allclose(event["mass"], scales["mass"], rtol=0, atol=1e-12)
               for events in roles.values() for event in events):
            raise ValueError("model and TASK3 mass differ")
        records = {}
        for role, events in roles.items():
            result = score(model, events, scales, budget, directory, role, settings, plan)
            records[role] = {key: result[key] for key in
                             ("summary", "unstable_prefix", "stability", "task3_six_term_score")}
        evidence = {"status": "PASS", "process_status": "PASS", "source_commit": git_head(root),
                    "promotion_decision": "KEEP_BASELINE", "data_policy": "kiki_official_train_test_v1",
                    "scientific_pass": False,
                    "claim_boundary": "One from-scratch frozen joint model on the fixed TASK3 comparison cohorts."}
        summary = evidence | {"records": records, "operations": dict(budget.counts),
                              "training": training, "weak_initialization": weak,
                              "wall_seconds": time.perf_counter()-started,
                              "runtime": {"python": sys.version, "executable": sys.executable,
                                          "device": str(model.mass.device), "cuda_build": torch.version.cuda,
                                          "gpu": torch.cuda.get_device_name() if model.mass.is_cuda else None,
                                          **{p: version(p) for p in ("torch", "numpy", "scipy", "filelock", "h5py")}}}
        write_json(directory/"summary.json", summary)
        write_json(directory/"stage_evidence.json", evidence)
        return summary
    finally:
        budget.stream.close()
        journal.close()
        write_json(directory/"artifact_hashes.json", {path.relative_to(directory).as_posix(): sha256_file(path)
                   for path in sorted(directory.rglob("*")) if path.is_file()})
