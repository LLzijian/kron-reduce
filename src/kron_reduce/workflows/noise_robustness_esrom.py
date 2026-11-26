"""Staged TASK3 noise study with a pre-access, one-use holdout gate."""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import tomllib
from contextlib import closing

import numpy as np
import torch

from kron_reduce.data.durable_audit_ledger import AppendOnlyAuditJournal
from kron_reduce.data.noise_stress_inputs import DevelopmentCache, blind_events, fixed_test_events
from kron_reduce.data.task3_full import REFERENCE
from kron_reduce.data.task3_measurement_noise import SCENARIOS, generate_noise_bank
from kron_reduce.evaluation.constant_c_observability import write_json
from kron_reduce.evaluation.noise_stress_predict import (
    MODELS,
    STEPS,
    infer,
    load_assets,
    model_for,
    observation_inputs,
    slice_inputs,
)
from kron_reduce.workflows.blind_confirmation_m5 import _reserve
from project_harness.archive_storage import read_run_artifact
from project_harness.evidence import sha256_file, verify_artifact_directory
from project_harness.git_state import git_head, verify_bound_file
from project_harness.loop import load_state, workflow_lock
from project_harness.manifest import load_manifest, project_root
from project_harness.policy import load_policy, validate_run_policy
from project_harness.runner import _run_experiment

WORKFLOW = "noise_robustness_esrom_v1"
CONFIG = f"configs/experiments/{WORKFLOW}.toml"
WORKFLOW_CONFIG = f"configs/workflows/{WORKFLOW}.json"
REPORT_DIR = "docs/reports/11_noise_robustness"
SEALED_FREEZE = f"{REPORT_DIR}/_sealed/freeze.json"
PYTEST_TEMP = "scratch/one_off/pytest_noise_kron_20261126"
CLAIM = "Measurement Noise & Outlier Stress Test of frozen V4 and ES-ROM."


def _json(path):
    return json.loads(path.read_bytes())


def _source(root, path):
    digest = sha256_file(root / path)
    verify_bound_file(root, path, digest)
    return digest


def _plans(root):
    digest = _source(root, CONFIG)
    plan = tomllib.loads((root / CONFIG).read_text(encoding="utf-8"))
    verify_bound_file(root, plan["baseline_plan"], plan["baseline_plan_sha256"])
    verify_bound_file(root, REFERENCE, plan["split_reference_sha256"])
    baseline = tomllib.loads((root / plan["baseline_plan"]).read_text(encoding="utf-8"))
    if (tuple(plan["steps_s"]) != STEPS or tuple(plan["models"]) != MODELS or
            tuple(plan["scenarios"]) != SCENARIOS or plan["counts"] !=
            {"train": 346, "val": 107, "test": 76, "blind_v2": 94} or
            (plan["observation_frames"], plan["sample_period_s"], plan["forecast_frames"]) != (11, .01, 800) or
            plan["device"] != "cuda:0" or plan["dtype"] != "float64"):
        raise ValueError("fixed roster, clocks, noise and hardware settings required")
    return plan, baseline, digest


def _runtime():
    if (sys.version_info[:2] != (3, 12) or not torch.cuda.is_available() or
            torch.cuda.get_device_name() != "NVIDIA GeForce RTX 4060 Ti" or
            os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"):
        raise RuntimeError("the registered deterministic Python 3.12/CUDA 4060 Ti runtime is required")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    return {"python": sys.version, "executable": sys.executable, "torch": torch.__version__,
            "numpy": np.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
            "dtype": "float64", "steps_s": STEPS}


def _verify_stage(root, name, digest):
    state = load_state(root, WORKFLOW)
    evidence = next((row["evidence"] for row in state["history"] if row.get("to") == name), None)
    if evidence is None or evidence["plan_sha256"] != digest:
        raise PermissionError(f"the bound {name} stage has not passed")
    binding = evidence["artifact_manifest"]
    directory = root / "results" / binding["run_id"]
    verify_artifact_directory(directory, strict=True, expected_manifest_sha256=binding["sha256"])
    return directory, evidence


def _synthetic(root, directory):
    temp = root / PYTEST_TEMP
    if temp.exists():
        raise FileExistsError(f"reserved pytest directory already exists: {temp}")
    from shutil import rmtree

    try:
        check = subprocess.run(
            [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp", PYTEST_TEMP,
             "tests/test_noise_robustness_mhe.py", "tests/test_task3_measurement_noise.py"],
            cwd=root, capture_output=True, check=False,
        )
    finally:
        if temp.exists():
            if temp.resolve() != temp or not temp.is_relative_to(root / "scratch" / "one_off"):
                raise RuntimeError("unexpected pytest temporary path")
            rmtree(temp)
    (directory / "pytest.log").write_bytes(check.stdout + check.stderr)
    if check.returncode:
        raise RuntimeError("the CUDA synthetic gate failed; see pytest.log")
    return {"real_data_reads": 0, "pytest_sha256": sha256_file(directory / "pytest.log")}


def _asset_parity(directory, root, plan, events, models, scales, assets):
    original = generate_noise_bank("val", "clean", 0, [event["id"] for event in events])
    inputs = observation_inputs(events, original, scales)
    ids = [e["id"] for e in events]
    checks = {}
    for name, member, receipt, batch_size, step_keys in (
        ("v4_hard_r", "hard_r/val_predictions.npz", assets["receipts"]["v4"], plan["v4_batch"], ("fine", "refined")),
        ("v4_soft_2r", "soft_2r/val_predictions.npz", assets["receipts"]["v4"], plan["v4_batch"], ("fine", "refined")),
        ("m3b_g2_orig", "support-G2/val_predictions.npz", assets["receipts"]["m3"], plan["m3b_batch"], ("main", "refined")),
    ):
        blob = read_run_artifact(root, receipt["run_id"], member, receipt["artifact_bindings"][member])
        model = model_for(models, name)
        with np.load(io.BytesIO(blob), allow_pickle=False) as expected:
            archived_ids = list(expected["ids"])
            if set(ids) != set(archived_ids) or len(archived_ids) != len(ids):
                raise ValueError(f"archived validation identities differ: {name}")
            order = [ids.index(event_id) for event_id in archived_ids]
            grouped_inputs = tuple(x[order] for x in inputs[:-1]) + (inputs[-1],)
            checks[name] = {}
            for step, key in zip(STEPS, step_keys, strict=True):
                actual = np.concatenate([
                    infer(model, name, slice_inputs(grouped_inputs, i, i + batch_size), step, plan, scales,
                          plan["candidates"][0], lambda *_: None)[0]
                    for i in range(0, len(ids), batch_size)
                ])
                target = expected[key]
                if actual.shape != target.shape:
                    raise ValueError(f"archived validation shape differs: {name}")
                exact = (actual.view(np.uint64) == target.view(np.uint64)).all(axis=(1, 2))
                errors = np.max(np.abs(actual - target), axis=(1, 2))
                checks[name][str(step)] = {"bitwise_equal": int(exact.sum()), "allocated": len(ids),
                    "max_absolute_error": float(errors.max()),
                    "nonidentical": [archived_ids[i] for i in np.flatnonzero(~exact)],
                    "matching_to_1e_minus_10": int((errors <= 1e-10).sum()),
                    "archive_sha256": receipt["artifact_bindings"][member]}
                print(f"clean replay {name} {step}: {int(exact.sum())}/{len(ids)} bit-identical", flush=True)
    write_json(directory / "clean_replay.json", checks)
    if any(row["bitwise_equal"] != len(ids) for by_step in checks.values() for row in by_step.values()):
        raise RuntimeError("the archived clean validation forecasts did not replay bitwise; see clean_replay.json")
    return {"archive_match": True, "val_cases": len(ids), "clean_noise_sha256": original.sha256}


def _write_evidence(directory, result):
    write_json(directory / "stage_evidence.json", result)
    write_json(directory / "artifact_hashes.json", {
        path.relative_to(directory).as_posix(): sha256_file(path)
        for path in sorted(directory.rglob("*")) if path.is_file()
    })


def _frozen_bindings(root, plan):
    files = [p.relative_to(root).as_posix() for p in sorted((root / "src/kron_reduce").rglob("*.py"))]
    files += [CONFIG, WORKFLOW_CONFIG, plan["protocol"], plan["baseline_plan"], REFERENCE,
              "configs/input_contracts.json", "requirements/constraints-py312.txt"]
    return {name: _source(root, name) for name in files}


def _freeze(root, directory, plan, digest, assets):
    current = load_state(root, WORKFLOW)["current_stage"]
    incoming, evidence = _verify_stage(root, current, digest)
    selection = _json(incoming / "selection.json")
    if evidence["scientific_verdict"] != "ACCEPT" or not selection["passed"] or current not in ("validation", "adapt"):
        raise PermissionError("full validation has not accepted a single candidate")
    checkpoint = directory / "m3b_g2.npz"
    preprocessing = directory / "preprocessing.json"
    checkpoint.write_bytes(assets["weight_bytes"]["m3"])
    preprocessing.write_bytes(assets["preprocessing_bytes"])
    freeze_path = directory / "freeze.json"
    write_json(freeze_path, {
        "claim_boundary": CLAIM, "source_commit": git_head(root), "plan_sha256": digest,
        "candidate": selection["candidate"], "optimizer_updates": selection["optimizer_updates"],
        "validation_run": evidence["artifact_manifest"], "files": _frozen_bindings(root, plan),
        "weights": assets["weights"], "checkpoint_path": checkpoint.relative_to(root).as_posix(),
        "checkpoint_sha256": sha256_file(checkpoint),
        "preprocessing_path": preprocessing.relative_to(root).as_posix(),
        "preprocessing_sha256": sha256_file(preprocessing), "selection_closed": True, "fitting_closed": True,
    })
    return {"frozen_model_manifest": {"path": freeze_path.relative_to(root).as_posix(),
                                     "sha256": sha256_file(freeze_path)}}


def _holdout(root, directory, run_id, plan, digest, source, models, scales, assets):
    from kron_reduce.evaluation.noise_stress_matrix import evaluate_matrix

    _, freeze_evidence = _verify_stage(root, "freeze", digest)
    expected = freeze_evidence["frozen_model_manifest"]["sha256"]
    verify_bound_file(root, SEALED_FREEZE, expected)
    freeze = _json(root / SEALED_FREEZE)
    if (not freeze["selection_closed"] or not freeze["fitting_closed"] or freeze["plan_sha256"] != digest
            or freeze["weights"] != assets["weights"] or freeze["files"] != _frozen_bindings(root, plan)):
        raise PermissionError("source, configuration or weights changed after the validation freeze")
    policy = load_policy(root)
    entries = load_manifest(root)["experiments"]
    for suffix, category, key in (("test", "official_test_evaluations", "test_authorization"),
                                  ("blind", "independent_confirmations", "blind_authorization")):
        name = f"{WORKFLOW}_{suffix}"
        entry = entries[name]
        scope = policy[category][entry[key]]
        if scope.get("frozen_model_manifest") != {"path": SEALED_FREEZE, "sha256": expected}:
            raise PermissionError("the scope must bind the accepted freeze")
        validate_run_policy(root, name, entry)
    _reserve(root, plan["scope"], run_id, git_head(root))
    with closing(AppendOnlyAuditJournal(directory / "access.jsonl")) as access:
        test = fixed_test_events(root, plan, source, access, root / SEALED_FREEZE)
        blind = blind_events(root, plan, access)
    metrics = evaluate_matrix(directory, {"test": test, "blind_v2": blind}, models, scales, plan,
                              candidates=[freeze["candidate"]])
    return {"test_cases": len(test), "blind_cases": len(blind), "test_trajectory_reads": len(test),
            "blind_container_opens": 1, "optimizer_updates": 0, "frozen_model_manifest":
            {"path": SEALED_FREEZE, "sha256": expected}, **metrics}


@torch.no_grad()
def run(root, experiment, stage, result_dir, run_id):
    validate_run_policy(root, WORKFLOW, experiment)
    plan, baseline, digest = _plans(root)
    write_json(result_dir / "runtime.json", _runtime())
    evidence = {"source_commit": git_head(root), "plan_sha256": digest, "claim_boundary": CLAIM}
    if stage == "synthetic":
        evidence.update(_synthetic(root, result_dir))
    else:
        source, models, scales, assets = load_assets(root, plan, baseline)
        evidence.update(test_trajectory_reads=0, blind_container_opens=0)
        if stage == "freeze":
            evidence.update(_freeze(root, result_dir, plan, digest, assets))
        elif stage == "blind":
            evidence.update(_holdout(root, result_dir, run_id, plan, digest, source, models, scales, assets))
        else:
            _verify_stage(root, "synthetic", digest)
            if stage in ("validation", "adapt"):
                _verify_stage(root, "clean", digest)
            if stage == "adapt":
                prior_dir, prior = _verify_stage(root, "validation", digest)
                if prior["scientific_verdict"] != "RETRY":
                    raise PermissionError("adaptation requires a failed zero-retrain validation")
            with closing(AppendOnlyAuditJournal(result_dir / "access.jsonl")) as access:
                val = DevelopmentCache(root, source, access).load("val")
                if stage == "clean":
                    evidence.update(_asset_parity(result_dir, root, plan, val, models, scales, assets))
                elif stage == "validation":
                    from kron_reduce.evaluation.noise_stress_matrix import (
                        evaluate_matrix,
                        select_candidate,
                    )

                    evidence.update(evaluate_matrix(result_dir, {"val": val}, models, scales, plan,
                                                    candidates=plan["candidates"]))
                    selection = select_candidate(result_dir, plan)
                    evidence.update(val_cases=len(val), optimizer_updates=0,
                                    scientific_verdict="ACCEPT" if selection["passed"] else "RETRY")
                elif stage == "adapt":
                    from kron_reduce.evaluation.noise_stress_fit import adapt

                    train = DevelopmentCache(root, source, access).load("train")
                    evidence.update(adapt(result_dir, prior_dir, train, val, models, scales, plan))
                else:
                    raise ValueError(f"unregistered execution stage: {stage}")
    _write_evidence(result_dir, evidence)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("synthetic", "clean", "validation", "adapt", "freeze", "blind"))
    args = parser.parse_args()
    root = project_root()
    with workflow_lock(root, WORKFLOW):
        directory = _run_experiment(root, WORKFLOW, args.stage)
    print(directory)


if __name__ == "__main__":
    main()
