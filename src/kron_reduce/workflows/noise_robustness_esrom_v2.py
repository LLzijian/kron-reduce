"""Registered V2 noise study and the single-use joint holdout release."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tomllib
from contextlib import closing

import torch

from kron_reduce.data.durable_audit_ledger import AppendOnlyAuditJournal
from kron_reduce.data.noise_stress_inputs import DevelopmentCache, blind_events, fixed_test_events
from kron_reduce.data.task3_full import REFERENCE
from kron_reduce.evaluation.constant_c_observability import write_json
from kron_reduce.evaluation.noise_stress_matrix_v2 import (
    V2,
    benchmark,
    evaluate_matrix,
    select_candidate,
)
from kron_reduce.evaluation.noise_stress_predict import STEPS, load_assets
from kron_reduce.workflows.blind_confirmation_m5 import _reserve
from kron_reduce.workflows.noise_robustness_esrom import (
    CLAIM,
    _json,
    _runtime,
    _source,
    _write_evidence,
)
from project_harness.archive_storage import read_run_artifact
from project_harness.evidence import sha256_file, verify_artifact_directory
from project_harness.git_state import git_head, verify_bound_file
from project_harness.loop import load_state, workflow_lock
from project_harness.manifest import load_manifest, project_root
from project_harness.policy import load_policy, validate_run_policy
from project_harness.runner import _run_experiment

WORKFLOW = "noise_robustness_esrom_v2"
CONFIG = f"configs/experiments/{WORKFLOW}.toml"
WORKFLOW_CONFIG = f"configs/workflows/{WORKFLOW}.json"
REPORT_DIR = "docs/reports/11_noise_robustness/v2"
SEALED_FREEZE = f"{REPORT_DIR}/_sealed/freeze_receipt.json"
PYTEST_TEMP = "scratch/one_off/pytest_noise_v2_20261127"
NOISE_ENGINE = "src/kron_reduce/data/task3_measurement_noise.py"


def _plans(root):
    digest = _source(root, CONFIG)
    settings = tomllib.loads((root / CONFIG).read_text(encoding="utf-8"))
    for path, sha in ((settings["protocol"], settings["protocol_sha256"]),
                      (settings["v1_plan"], settings["v1_plan_sha256"]),
                      (settings["v1_receipt"], settings["v1_receipt_sha256"]),
                      (NOISE_ENGINE, settings["noise_engine_sha256"])):
        verify_bound_file(root, path, sha)
    v1 = tomllib.loads((root / settings["v1_plan"]).read_text(encoding="utf-8"))
    if settings["gate"] != v1["gate"] or len(settings["candidates"]) != 8:
        raise ValueError("V2 requires the unchanged V1 gate and eight declared candidates")
    plan = v1 | settings
    plan["models"] = [*v1["models"][:-1], V2]
    plan["legacy_l2"] = v1["candidates"][0]
    plan["candidates"] = [
        candidate | {key: settings[key] for key in ("gate_rad", "huber_delta")}
        for candidate in settings["candidates"]
    ]
    for path, sha in ((plan["baseline_plan"], plan["baseline_plan_sha256"]),
                      (REFERENCE, plan["split_reference_sha256"])):
        verify_bound_file(root, path, sha)
    baseline = tomllib.loads((root / plan["baseline_plan"]).read_text(encoding="utf-8"))
    return plan, baseline, digest


def _phase(root, stage, digest):
    state = load_state(root, WORKFLOW)
    evidence = next((row["evidence"] for row in state["history"] if row.get("to") == stage), None)
    if evidence is None or evidence["plan_sha256"] != digest:
        raise PermissionError(f"the registered {stage} gate has not passed")
    binding = evidence["artifact_manifest"]
    directory = root / "results" / binding["run_id"]
    verify_artifact_directory(directory, strict=True, expected_manifest_sha256=binding["sha256"])
    return directory, evidence


def v1_reference(root, plan):
    receipt = _json(root / plan["v1_receipt"])
    evidence = receipt["stages"]["validation"]
    binding = evidence["artifact_manifest"]
    run_id = binding["run_id"]
    artifact = json.loads(read_run_artifact(root, run_id, "artifact_hashes.json", binding["sha256"]))
    if (artifact["metrics.json"] != evidence["metrics_sha256"]
            or artifact["noise_manifest.json"] != evidence["noise_manifest_sha256"]):
        raise ValueError("V1 comparison artifacts differ from the closed receipt")
    values = {name: json.loads(read_run_artifact(root, run_id, name, artifact[name]))
              for name in ("metrics.json", "noise_manifest.json", "latency.json")}
    if set(values["metrics.json"]["rosters"]) != {"val"} or set(values["noise_manifest.json"]) != {"val"}:
        raise ValueError("the reused comparison must contain val only")
    return values["metrics.json"]["rosters"]["val"], values["noise_manifest.json"], values["latency.json"]


def _synthetic(root, directory):
    temp = root / PYTEST_TEMP
    if temp.exists():
        raise FileExistsError(f"reserved pytest directory already exists: {temp}")
    try:
        check = subprocess.run(
            [sys.executable, "-B", "-m", "pytest", "-q", "-s", "-p", "no:cacheprovider",
             "--basetemp", PYTEST_TEMP, "tests/test_noise_robustness_mhe_v2.py"],
            cwd=root, capture_output=True, check=False,
        )
    finally:
        if temp.exists():
            if temp.resolve() != temp or not temp.is_relative_to(root / "scratch" / "one_off"):
                raise RuntimeError("unexpected pytest temporary path")
            shutil.rmtree(temp)
    (directory / "pytest.log").write_bytes(check.stdout + check.stderr)
    if check.returncode not in (0, 1):
        raise RuntimeError("the V2 synthetic pytest process could not complete; see pytest.log")
    prefix = "V2_PULSE_DIAGNOSTICS "
    rows = [json.loads(line.split(prefix, 1)[1]) for line in check.stdout.decode("utf-8").splitlines()
            if prefix + "{" in line]
    if (tuple(row["step_s"] for row in rows) != STEPS
            or any(row["device"] != "cuda:0" or len(row["cases"]) != 40 for row in rows)):
        raise RuntimeError("the synthetic pulse diagnostics lack full dual-step CUDA coverage")
    write_json(directory / "pulse_diagnostics.json", {"steps": {str(row["step_s"]): row for row in rows}})
    return {"real_data_reads": 0, "pytest_exit_code": check.returncode,
            "scientific_verdict": "REJECT" if check.returncode else "ACCEPT",
            "pytest_sha256": sha256_file(directory / "pytest.log"),
            "pulse_diagnostics_sha256": sha256_file(directory / "pulse_diagnostics.json")}


def _frozen_files(root, plan):
    files = [p.relative_to(root).as_posix() for group in ("src/kron_reduce", "project_harness")
             for p in (root / group).rglob("*.py")]
    files += [CONFIG, WORKFLOW_CONFIG, plan["protocol"], plan["v1_plan"], plan["v1_receipt"],
              plan["baseline_plan"], REFERENCE, "configs/input_contracts.json",
              "requirements/constraints-py312.txt", "tests/test_noise_robustness_mhe_v2.py",
              "tests/test_task3_measurement_noise.py", "tests/test_task3_noise_scores.py"]
    return {name: _source(root, name) for name in sorted(files)}


def _freeze(root, directory, plan, digest, assets):
    predecessor = load_state(root, WORKFLOW)["current_stage"]
    if predecessor not in ("validation", "adapt"):
        raise PermissionError("there is no eligible validation stage")
    incoming, evidence = _phase(root, predecessor, digest)
    selection = _json(incoming / "selection.json")
    if evidence["scientific_verdict"] != "ACCEPT" or not selection["passed"]:
        raise PermissionError("full val has not accepted a candidate")
    checkpoint = directory / "m3b_g2.npz"
    prep = directory / "preprocessing.json"
    checkpoint.write_bytes(assets["weight_bytes"]["m3"])
    prep.write_bytes(assets["preprocessing_bytes"])
    path = directory / "freeze_receipt.json"
    write_json(path, {
        "claim_boundary": CLAIM, "source_commit": git_head(root), "plan_sha256": digest,
        "candidate": selection["candidate"], "optimizer_updates": selection["optimizer_updates"],
        "validation_run": evidence["artifact_manifest"], "files": _frozen_files(root, plan),
        "weights": assets["weights"], "selection_closed": True, "fitting_closed": True,
        "checkpoint_path": checkpoint.relative_to(root).as_posix(), "checkpoint_sha256": sha256_file(checkpoint),
        "preprocessing_path": prep.relative_to(root).as_posix(), "preprocessing_sha256": sha256_file(prep),
    })
    return {"frozen_model_manifest": {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}}


def _holdout(root, directory, run_id, plan, digest, source, models, scales, assets):
    _, evidence = _phase(root, "freeze", digest)
    expected = evidence["frozen_model_manifest"]["sha256"]
    verify_bound_file(root, SEALED_FREEZE, expected)
    frozen = _json(root / SEALED_FREEZE)
    if (not frozen["selection_closed"] or not frozen["fitting_closed"] or frozen["plan_sha256"] != digest
            or frozen["weights"] != assets["weights"] or frozen["files"] != _frozen_files(root, plan)):
        raise PermissionError("code, plan or weights changed after the V2 freeze")
    binding = {"path": SEALED_FREEZE, "sha256": expected}
    policy = load_policy(root)
    entries = load_manifest(root)["experiments"]
    for suffix, category, key in (("test", "official_test_evaluations", "test_authorization"),
                                  ("blind", "independent_confirmations", "blind_authorization")):
        name = f"{WORKFLOW}_{suffix}"
        entry = entries[name]
        scope = policy[category][entry[key]]
        if scope.get("frozen_model_manifest") != binding:
            raise PermissionError("test and blind authorizations must bind the accepted freeze")
        validate_run_policy(root, name, entry)
    _reserve(root, plan["scope"], run_id, git_head(root))
    with closing(AppendOnlyAuditJournal(directory / "access.jsonl")) as access:
        test = fixed_test_events(root, plan, source, access, root / SEALED_FREEZE)
        blind = blind_events(root, plan, access)
    _, metrics = evaluate_matrix(directory, {"test": test, "blind_v2": blind}, models, scales, plan,
                                 candidates=[frozen["candidate"]])
    return {"test_cases": len(test), "blind_cases": len(blind), "test_trajectory_reads": len(test),
            "blind_container_opens": 1, "optimizer_updates": 0, "frozen_model_manifest": binding,
            **metrics}


def _development(root, result_dir, plan, digest, source, models, scales, *, adapt):
    _, synthetic = _phase(root, "synthetic", digest)
    if synthetic["scientific_verdict"] != "ACCEPT" or synthetic["pytest_exit_code"] != 0:
        raise PermissionError("the synthetic gate has not accepted V2")
    if adapt:
        prior, previous = _phase(root, "validation", digest)
        if previous["scientific_verdict"] != "RETRY":
            raise PermissionError("conditional adaptation requires a failed zero-retrain gate")
        initial = _json(prior / "selection.json")
    old, noise, latency = v1_reference(root, plan)
    with closing(AppendOnlyAuditJournal(result_dir / "access.jsonl")) as access:
        val = DevelopmentCache(root, source, access).load("val")
        if adapt:
            train = DevelopmentCache(root, source, access).load("train")

    def assess(directory, candidates, update):
        scores, metrics = evaluate_matrix(directory, {"val": val}, models, scales, plan,
                                          candidates=candidates, reference_noise=noise)
        return select_candidate(directory, plan, scores, old, candidates, update), metrics

    if not adapt:
        selected, evidence = assess(result_dir, plan["candidates"], 0)
        benchmark(result_dir, val, models["m3"], scales, plan, selected["candidate"], latency)
        return {"val_cases": len(val), "optimizer_updates": 0, **evidence,
                "scientific_verdict": "ACCEPT" if selected["passed"] else "RETRY"}

    from kron_reduce.evaluation.noise_stress_fit_v2 import fit_candidates

    def assess_checkpoint(directory, candidate, update):
        return assess(directory, [candidate], update)[0]

    selected = fit_candidates(result_dir, train, models, scales, plan, initial, assess_checkpoint)
    if selected["passed"]:
        benchmark(result_dir, val, models["m3"], scales, plan, selected["candidate"], latency)
    return {"train_cases": len(train), "val_cases": len(val), "optimizer_updates": selected["optimizer_updates"],
            "scientific_verdict": "ACCEPT" if selected["passed"] else "REJECT",
            "training_sha256": sha256_file(result_dir / "training_curves.json")}


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
        elif stage in ("validation", "adapt"):
            evidence.update(_development(root, result_dir, plan, digest, source, models, scales, adapt=stage == "adapt"))
        else:
            raise ValueError(f"unregistered execution stage: {stage}")
    _write_evidence(result_dir, evidence)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("synthetic", "validation", "adapt", "freeze", "blind"))
    args = parser.parse_args()
    root = project_root()
    with workflow_lock(root, WORKFLOW):
        directory = _run_experiment(root, WORKFLOW, args.stage)
    print(directory)


if __name__ == "__main__":
    main()
