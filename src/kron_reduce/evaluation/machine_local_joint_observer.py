"""Paired first-frame constraint diagnosis and new-sample joint-state checks."""
from __future__ import annotations

import hashlib
import io
import json

import numpy as np

from kron_reduce.evaluation.constant_c_observability import write_json
from kron_reduce.evaluation.machine_local_observer import evaluate, known_model
from kron_reduce.evaluation.machine_local_observer import execute as execute_observer
from kron_reduce.evaluation.machine_local_synthetic import fixture, synthetic_data
from kron_reduce.models.joint_history_initializer import estimate_joint_history
from project_harness.evidence import sha256_file


def validate(root, directory, budget, plan):
    datasets, references = {}, {}
    with budget.scope("sources"):
        for kind in ("local", "nonlocal"):
            entry = plan["saved_sources"][kind]
            raw = (root/entry["path"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
                raise RuntimeError("joint observer source hash changed")
            budget.spend("npz_decodes")
            with np.load(io.BytesIO(raw), allow_pickle=False) as source:
                data = {key: source[key].copy() for key in source.files}
            if (data["paths"].shape != (64, 811, 40) or data["noise"].shape != (64, 11, 20)
                    or not np.array_equal(data["roles"], ["train"]*32+["val"]*16+["test"]*16)):
                raise RuntimeError("joint observer saved population changed")
            datasets["old_"+kind] = data
            np.savez_compressed(directory/("old_"+kind+"_inputs.npz"), **data)
            ref = plan["paired_references"][kind]
            text = (root/ref["path"]).read_bytes()
            if hashlib.sha256(text).hexdigest() != ref["sha256"]:
                raise RuntimeError("paired observer reference changed")
            references[kind] = json.loads(text)
    for kind, seed in plan["truth_seeds"].items():
        with budget.scope("truth_"+kind):
            base, paths, scale, hscale = synthetic_data(kind, seed, budget)
            p = np.eye(10)-np.outer(base.mass, np.ones(10))/base.mass.sum()
            noise = np.random.default_rng(seed+100).normal(size=(64, 11, 20))*hscale[None, None, :]*.01
            budget.spend("matrix_products", 2)
            noise[:, :, :10], noise[:, :, 10:] = noise[:, :, :10]@p, noise[:, :, 10:]@p
            data = {"paths": paths, "target_scale": scale, "history_scale": hscale, "noise": noise,
                "mass": base.mass, "coupling": base.coupling, "decay": base.latent_dissipation,
                "roles": np.array(["train"]*32+["val"]*16+["test"]*16)}
            np.savez_compressed(directory/(kind+"_inputs.npz"), **data)
            datasets[kind] = data
    write_json(directory/"frozen_inputs.json", {p.name: sha256_file(p) for p in sorted(directory.glob("*_inputs.npz"))})
    rows = {}
    for case in plan["cases"]:
        name, data_key, kind, noisy = case["name"], case["inputs"], case["kind"], case["noisy"]
        data = datasets[data_key]
        with budget.scope(name):
            rows[name] = evaluate(name, known_model(fixture(kind), kind), data["paths"],
                data["target_scale"], data["history_scale"], data["noise"] if noisy else 0.,
                directory, budget, plan["gates"], estimator=estimate_joint_history)
    with budget.scope("rank_control"):
        model = known_model(fixture("equal"), "local")
        zero = np.zeros((1, 11, 10))
        result = estimate_joint_history(model, zero, zero, np.zeros((1, 10)), np.zeros((1, 10)),
            np.ones(20), max_step=.0025, spend=budget.spend)
        np.savez_compressed(directory/"rank_control.npz", **{k: v for k, v in result.items() if isinstance(v, np.ndarray)})
        rank_pass = not result["resolved"] and result["rank"] == 27 and "origin" not in result
        rank_row = {"rank": result["rank"], "resolved": result["resolved"], "pass": rank_pass,
                    "candidate_future_calls": 0}
        write_json(directory/"rank_control.json", rank_row)
    paired = {}
    for kind in ("local", "nonlocal"):
        paired[kind] = {}
        for role in ("train", "val"):
            old = references[kind]["metrics"][role]["main_nmse"]
            candidate = rows["old_"+kind+"_noise01"]
            new = candidate["metrics"][role]["main_nmse"] if candidate["resolved"] else None
            paired[kind][role] = {"original_main_nmse": old, "joint_main_nmse": new,
                "ratio": new/old if old > 0 and new is not None else None,
                "improves_5pct": bool(old > 0 and new is not None and new <= .95*old)}
    supported = all(r["improves_5pct"] for roles in paired.values() for r in roles.values())
    numerical = rank_pass and all(r["numerical_pass"] for r in rows.values())
    clean = all(rows[k]["prediction_pass"] and rows[k]["hidden_clean_pass"] for k in ("local_clean", "nonlocal_clean"))
    noisy = all(rows[k]["prediction_pass"] for k in ("local_noise01", "nonlocal_noise01"))
    passed = bool(numerical and clean and noisy)
    return {"scientific_pass": passed, "numerical_pass": bool(numerical), "clean_recovery_pass": bool(clean),
            "noise_prediction_pass": bool(noisy), "rank_control": rank_row, "records": rows,
            "paired_mechanism_supported": supported, "paired_train_val": paired,
            "learned_model_confirmed": False, "t0_allowed": False,
            "next_route": "REGISTER_OBSERVER_DYNAMICS_TRAINING" if passed else "NOISE_MODEL_REPAIR_REQUIRED",
            "claim_boundary": "Joint initial-state estimation under known fixed dynamics. Paired mechanism uses prior train/val only; new seeds validate the frozen estimator. No learned-model or real-data confirmation."}


def execute(*, root, result_dir, stage, plan):
    return execute_observer(root=root, result_dir=result_dir, stage=stage, plan=plan, validator=validate)
