"""Calibrate four fixed supports on TASK3 train and score the two frozen models."""

from __future__ import annotations

import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from kron_reduce.data.durable_audit_ledger import AppendOnlyAuditJournal
from kron_reduce.data.task3_full import Task3FullReader
from kron_reduce.evaluation.constant_c_observability import write_json
from kron_reduce.evaluation.kiki_machine_local_full import (
    evaluate,
    inputs,
    make_model,
    predict_checked,
)
from kron_reduce.evaluation.kiki_machine_local_r20_r21 import bind, monitor, read_json, verified
from kron_reduce.evaluation.machine_local_equation_fit import choose_support, freeze
from kron_reduce.evaluation.machine_local_memory_fit import momentum_fit
from kron_reduce.evaluation.machine_local_synthetic import Budget
from kron_reduce.models.machine_local_equation import JointMachineLocalModel
from kron_reduce.models.nonlinear_latent import LearnedNonlinearModel, NonlinearPassiveClosure
from project_harness.evidence import sha256_file
from project_harness.git_state import git_head

SUPPORTS = {"neither": (False, False), "harmonic": (True, False), "damping": (False, True), "both": (True, True)}


def load_model(path, support):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    empty = np.zeros((0, 0))
    backbone = NonlinearPassiveClosure(arrays["mass"], arrays["stiffness"], arrays["damping"],
                                       np.zeros((10, 0)), empty, empty)
    harmonic, damping = SUPPORTS[support]
    model = JointMachineLocalModel(backbone, port=np.ones(10), tau=np.ones(10),
                                   harmonic=harmonic, modulated_damping=damping)
    target = model.state_dict()
    if arrays.keys() != target.keys() or any(
        v.dtype != np.float64 or v.shape != target[k].shape or not np.isfinite(v).all()
        for k, v in arrays.items()
    ):
        raise ValueError("invalid frozen joint dynamics state")
    for key, value in model.named_buffers():
        if not np.array_equal(value.numpy(), arrays[key]):
            raise ValueError(f"invalid frozen joint dynamics buffer: {key}")
    model.load_state_dict({key: torch.from_numpy(value) for key, value in arrays.items()}, strict=True)
    return model.requires_grad_(False).eval()


def save_freeze(root, directory, phase, models, supports):
    files = {}
    for name, model in models.items():
        target = directory/phase/name
        target.mkdir(parents=True)
        freeze(model, target)
        files[name] = bind(root, target/"frozen.npz") | {"support": supports[name]}
    checkpoint = directory/phase/"checkpoints.json"
    write_json(checkpoint, files)
    preprocessing = directory/"preprocessing.json"
    finished = phase == "final"
    result = {
        "phase": phase, "source_commit": git_head(root), "fitting_closed": finished, "selection_closed": finished,
        "input_contract": "measured_history_100ms_zoh_v1",
        "checkpoint_path": bind(root, checkpoint)["path"], "checkpoint_sha256": sha256_file(checkpoint),
        "preprocessing_path": bind(root, preprocessing)["path"], "preprocessing_sha256": sha256_file(preprocessing),
        "train_scale_population": bind(root, directory/"train_scale_population.json"),
    }
    path = directory/("freeze.json" if finished else "initial.json")
    write_json(path, result)
    return path


def fit(root, directory, plan, reader, budget):
    budget.spend("checkpoint_loads")
    backbone = LearnedNonlinearModel.load(verified(root, plan["backbone"])).closure
    initial = make_model(backbone)
    models = {name: JointMachineLocalModel(
        backbone, port=initial.port_parameter.detach().exp().numpy(), tau=initial.tau.detach().numpy(),
        harmonic=flags[0], modulated_damping=flags[1]) for name, flags in SUPPORTS.items()}
    train = reader.load("train")
    scales = {"history": np.maximum(np.sqrt(np.mean([np.mean(e["x"][:11]**2, axis=0) for e in train], axis=0)), 1e-5),
              "target": np.maximum(np.sqrt(np.mean([np.mean(e["x"][11:311]**2, axis=0) for e in train], axis=0)), 1e-5)}
    write_json(directory/"preprocessing.json", {k: v.tolist() for k, v in scales.items()})
    write_json(directory/"train_scale_population.json", {"ids": [e["id"] for e in train],
        "history_points": 11, "target_points": [min(300, len(e["x"])-11) for e in train]})
    val = reader.load("val", freeze_path=save_freeze(root, directory, "initial", models, {k: k for k in models}))
    if not all(np.allclose(e["mass"], backbone.mass, rtol=0, atol=1e-12) for e in train+val):
        raise ValueError("backbone and data mass differ")

    def validate(model):
        budget.spend("val_checks")
        model.to("cuda")
        try:
            record = monitor(model, val, scales, budget, plan["settings"]["eval_batch"])
            return record["nmse"], record["event_nmse"]
        finally:
            model.to("cpu")

    fits = {}
    for support, model in models.items():
        fits[support] = momentum_fit(model, train, scales["target"], budget, start_frame=10,
                                     max_calls=plan["settings"]["momentum_calls"], monitor=validate)
        write_json(directory/f"fit-{support}.json", fits[support])
        print(f'{support}: {len(fits[support]["train_trace"])} train objectives; '
              f'val={fits[support]["selected_val_nmse"]}; {fits[support]["stop_reason"]}', flush=True)
    if any(row["selected_val_nmse"] is not None for row in fits.values()):
        selected, allowance = choose_support(models, fits)
    else:
        selected, allowance = "neither", None
    selected_models = {"a": models["neither"], "b": models[selected]}
    write_json(directory/"selection.json", {"chosen_support": selected, "allowance": allowance,
        "candidates": {k: {x: y for x, y in v.items() if x not in ("train_trace", "val_trace")}
                       for k, v in fits.items()}, "val_ids": [e["id"] for e in val]})
    save_freeze(root, directory, "final", selected_models, {"a": "neither", "b": selected})
    return selected_models, {"train": train, "val": val}, scales


def test(root, plan, reader, budget):
    path = verified(root, plan["frozen_model_manifest"])
    final = read_json(path)
    if (final["phase"] != "final" or final["fitting_closed"] is not True or final["selection_closed"] is not True
            or final["input_contract"] != "measured_history_100ms_zoh_v1"):
        raise RuntimeError("test requires closed model and preprocessing selection")
    verified(root, final["train_scale_population"])
    assets = {key: verified(root, {"path": final[key+"_path"], "sha256": final[key+"_sha256"]})
              for key in ("checkpoint", "preprocessing")}
    files = read_json(assets["checkpoint"])
    if files.keys() != {"a", "b"} or files["a"]["support"] != "neither":
        raise RuntimeError("frozen models differ from the registered comparison")
    models = {}
    for name, binding in files.items():
        budget.spend("checkpoint_loads")
        models[name] = load_model(verified(root, binding), binding["support"])
    return models, {"test": reader.load("test", freeze_path=path)}, read_json(assets["preprocessing"])


def execute(root, directory, plan, stage):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(Path(plan["task3_root"])/"src"))
    started = time.perf_counter()
    budget = Budget(directory/"operations.jsonl", plan["operation_limits"])
    journal = AppendOnlyAuditJournal(directory/"access.jsonl")
    try:
        reader = Task3FullReader(root, plan["sources"], journal)
        models, roles, scales = (fit(root, directory, plan, reader, budget) if stage == "fit"
                                 else test(root, plan, reader, budget))
        records = {}
        for name, model in models.items():
            path = directory/name
            path.mkdir()
            model.to("cuda").requires_grad_(False).eval()
            records[name] = {}

            def predict(batch, step, model=model):
                return predict_checked(model, *inputs(batch, scales["history"]), 800,
                                       max_step=step, spend=budget.spend)

            for role, events in roles.items():
                result = evaluate(model, events, scales["history"], budget, path, role, plan["settings"],
                                  predictor=predict, target_scale=scales["target"])
                records[name][role] = {key: result[key] for key in
                                       ("summary", "unstable_prefix", "stability", "task3_six_term_score")}
            model.to("cpu")
        evidence = {"status": "PASS", "process_status": "PASS", "source_commit": git_head(root),
                    "data_policy": "kiki_official_train_test_v1", "promotion_decision": "KEEP_BASELINE",
                    "scientific_pass": False, "claim_boundary": "Full frozen TASK3 comparison of train momentum calibration."}
        result = evidence | {"records": records, "operations": dict(budget.counts),
            "wall_seconds": time.perf_counter()-started, "runtime": {"python": sys.version,
            "executable": sys.executable, "device": "cuda", "cuda_build": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(), **{p: version(p) for p in ("torch", "numpy", "scipy", "filelock", "h5py")}}}
        write_json(directory/"summary.json", result)
        write_json(directory/"stage_evidence.json", evidence)
        return result
    finally:
        journal.close()
        budget.stream.close()
        write_json(directory/"artifact_hashes.json", {p.relative_to(directory).as_posix(): sha256_file(p)
                   for p in sorted(directory.rglob("*")) if p.is_file()})
