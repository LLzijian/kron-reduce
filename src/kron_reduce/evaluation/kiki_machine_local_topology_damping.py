"""Single-parameter TASK3 fit and frozen complete-roster scoring."""
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
from kron_reduce.data.task3_full import COUNTS, Task3FullReader
from kron_reduce.evaluation.constant_c_observability import write_json
from kron_reduce.evaluation.kiki_machine_local_full import batch_objective
from kron_reduce.evaluation.kiki_machine_local_r20_r21 import bind, read_json, verified
from kron_reduce.evaluation.kiki_machine_local_topology import freeze_pair, score
from kron_reduce.evaluation.kiki_topology_bridge import load_snapshot, make_model, predict
from kron_reduce.evaluation.machine_local_clean_learning import snapshot
from kron_reduce.evaluation.machine_local_synthetic import Budget
from kron_reduce.evaluation.machine_local_topology_delayed_damping import fit_profile
from kron_reduce.models.nonlinear_latent import LearnedNonlinearModel
from project_harness.evidence import sha256_file
from project_harness.git_state import git_head

FIELDS = ("summary", "unstable_prefix", "stable_complete", "counts", "stability", "task3_six_term_score")


def _fit(model, events, scales, settings, budget):
    predict_batch = partial(predict, memory_radius=scales["memory_radius"], prior_weight=settings["prior_weight"])
    batch_size = settings["train_batch"]

    def objective():
        return sum(float(batch_objective(model, events[i:i+batch_size], len(events), scales, budget,
                                         predictor=predict_batch)) for i in range(0, len(events), batch_size))

    before = snapshot(model)
    record = fit_profile(model, objective, settings["profile"], budget)
    if any(not torch.equal(value, before[key]) for key, value in model.state_dict().items()
           if key != "log_damping_factor"):
        raise RuntimeError("the fit changed a coefficient outside the damping scale")
    return record


def _summary(result, reference, events, role):
    expected = [(row["id"], row["label"]) for row in events]
    for score_record in (result, reference):
        rows = score_record["stability_events"]
        s = score_record["stability"]
        if (len(events) != COUNTS[role] or s["allocated"] != COUNTS[role]
                or [(row["id"], row["truth_label"]) for row in rows] != expected
                or s["resolved"] + s["unresolved"] + s["uncovered"] != len(events)):
            raise RuntimeError(f"incomplete or different {role} cohort")
    return {key: result[key] for key in FIELDS}


def val_guard(result, reference):
    s = result["stability"]
    a, b = (r["stable_complete"]["8"] for r in (reference, result))
    eligible = all(row["valid"] and row["n_events"] == row["qualified"] == 95 and row["unresolved"] == 0
                   and row["mean"] is not None and math.isfinite(row["mean"]) for row in (a, b))
    return bool(eligible and b["mean"] <= 1.05*a["mean"] and s["fp"] == 0 and s["unresolved"] <= 1
                and s["uncovered"] == s["numerical_failures"] == 0 and 107-s["tp"]-s["tn"] <= 3
                and result["task3_six_term_score"] is not None and result["task3_six_term_score"] <= .5)


def test_goal(stability):
    return (stability["allocated"] == stability["resolved"] == 76 and stability["numerical_failures"] == 0
            and stability["tp"]+stability["tn"] >= 69 and stability["accuracy"] is not None
            and stability["accuracy"] > .9)


def execute(root, directory, plan, stage):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(Path(plan["task3_root"])/"src"))
    budget = Budget(directory/"operations.jsonl", plan["operation_limits"])
    journal = AppendOnlyAuditJournal(directory/"access.jsonl")
    started, settings = time.perf_counter(), plan["settings"]
    try:
        reader = Task3FullReader(root, plan["sources"], journal)
        budget.spend("checkpoint_loads")
        backbone = LearnedNonlinearModel.load(verified(root, plan["backbone"])).closure
        model = make_model(backbone, "T0M1", train_scales=True, device="cuda")
        freeze_path = verified(root, plan["model_manifest"])
        source = read_json(freeze_path)
        index, scales = (read_json(verified(root, {"path": source[key+"_path"], "sha256": source[key+"_sha256"]}))
                         for key in ("checkpoint", "preprocessing"))
        if (set(scales) != {"history", "target", "memory_radius"} or scales["memory_radius"] != plan["memory_radius"]
                or any(np.shape(scales[key]) != (20,) or not np.isfinite(scales[key]).all()
                       or not np.all(np.asarray(scales[key]) > 0) for key in ("history", "target"))):
            raise ValueError("original TASK3 normalization or radius differs")
        budget.spend("checkpoint_loads")
        load_snapshot(model, verified(root, index["T0M1"]))
        references = {role: read_json(verified(root, item)) for role, item in plan["reference"].items()}
        fit = None
        if stage == "fit":
            train = reader.load("train")
            if not np.allclose(reader.static["mass"], backbone.mass, rtol=0, atol=1e-12):
                raise ValueError("TASK3 inertia differs from the frozen model")
            fit = _fit(model, train, scales, settings, budget)
            write_json(directory/"damping-profile.json", fit)
            write_json(directory/"preprocessing.json", scales)
            freeze_path = freeze_pair(root, directory, "freeze", {"T0M1": model}, plan,
                                      {key: value for key, value in fit.items() if key != "trace"})
            events = {"train": train, "val": reader.load("val", freeze_path=freeze_path)}
        else:
            events = {"test": reader.load("test", freeze_path=freeze_path)}

        if not np.allclose(reader.static["mass"], backbone.mass, rtol=0, atol=1e-12):
            raise ValueError("TASK3 inertia differs from the frozen model")
        folder = directory/"T0M1"
        folder.mkdir()
        records = {}
        for role, batch in events.items():
            result = score(model, batch, scales, budget, folder, role, settings)
            records[role] = _summary(result, references[role], batch, role)
        verdict = (fit["valid"] and val_guard(records["val"], references["val"]) if stage == "fit"
                   else test_goal(records["test"]["stability"]))
        evidence = {"status": "PASS", "process_status": "PASS", "engineering_pass": True,
                    "source_commit": git_head(root), "data_policy": "kiki_official_train_test_v1",
                    "promotion_decision": "KEEP_BASELINE", "scientific_pass": verdict,
                    "claim_boundary": "Single-scale damping fit on the fixed TASK3 cohorts, compared after freeze."}
        summary = evidence | {"records": {"T0M1": records}, "fit": fit,
                              "frozen_model_manifest": bind(root, freeze_path), "operations": dict(budget.counts),
                              "reference": {role: {key: record[key] for key in FIELDS}
                                            for role, record in references.items()},
                              "runtime": {"python": sys.version, "executable": sys.executable, "device": "cuda",
                                          "cuda_build": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
                                          **{p: version(p) for p in ("torch", "numpy", "scipy", "filelock", "h5py")}},
                              "wall_seconds": time.perf_counter()-started}
        write_json(directory/"summary.json", summary)
        write_json(directory/"stage_evidence.json", evidence)
        return summary
    finally:
        journal.close()
        budget.stream.close()
        write_json(directory/"artifact_hashes.json", {str(path.relative_to(directory).as_posix()): sha256_file(path)
                   for path in sorted(directory.rglob("*")) if path.is_file()})
