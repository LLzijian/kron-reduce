"""Run the registered V3 gates and the single-use joint holdout."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tomllib
from contextlib import closing
from itertools import product

import torch

from kron_reduce.data.durable_audit_ledger import AppendOnlyAuditJournal
from kron_reduce.data.noise_stress_inputs import DevelopmentCache, blind_events, fixed_test_events
from kron_reduce.data.task3_full import REFERENCE
from kron_reduce.evaluation.constant_c_observability import write_json
from kron_reduce.evaluation.noise_stress_matrix_v3 import (
    V3,
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
from kron_reduce.workflows.noise_robustness_esrom_v2 import v1_reference
from project_harness.evidence import sha256_file, verify_artifact_directory
from project_harness.git_state import git_head, verify_bound_file
from project_harness.loop import load_state, workflow_lock
from project_harness.manifest import load_manifest, project_root
from project_harness.policy import load_policy, validate_run_policy
from project_harness.runner import _run_experiment

WORKFLOW = "noise_robustness_esrom_v3"
CONFIG = f"configs/experiments/{WORKFLOW}.toml"
WORKFLOW_CONFIG = f"configs/workflows/{WORKFLOW}.json"
REPORT_DIR = "docs/reports/11_noise_robustness/v3"
SEALED_FREEZE = f"{REPORT_DIR}/_sealed/freeze_receipt.json"
PYTEST_TEMP = "scratch/one_off/pytest_noise_v3_20261127"
V3_TEST = "tests/test_noise_robustness_mhe_v3.py"
V2_FIXTURE = "tests/test_noise_robustness_mhe_v2.py"
NOISE_ENGINE = "src/kron_reduce/data/task3_measurement_noise.py"
V2_SUMMARY = "docs/reports/11_noise_robustness/v2/_sealed/synthetic_summary.json"


def _plans(root):
    digest = _source(root, CONFIG)
    settings = tomllib.loads((root / CONFIG).read_text(encoding="utf-8"))
    for path, sha in ((settings["protocol"], settings["protocol_sha256"]),
                      (settings["v1_plan"], settings["v1_plan_sha256"]),
                      (settings["v1_receipt"], settings["v1_receipt_sha256"]),
                      (settings["v2_receipt"], settings["v2_receipt_sha256"]),
                      (V2_FIXTURE, settings["v2_synthetic_fixture_sha256"]),
                      (NOISE_ENGINE, settings["noise_engine_sha256"])):
        verify_bound_file(root, path, sha)
    v1 = tomllib.loads((root / settings["v1_plan"]).read_text(encoding="utf-8"))
    if (settings["gate"] != v1["gate"] or [r["name"] for r in settings["candidates"]] != list("ABCDEFGH")
            or settings["synthetic_gate"] != {
                "cases_per_track_step": 40, "track_b_truth_stiffness_damping_multiplier": 1.08,
                "track_a_minimum_reduction_vs_original_gn": .1,
            }):
        raise ValueError("the V3 roster, gates and candidate schedule must match registration")
    plan = v1 | settings
    plan["models"] = [*v1["models"][:-1], V3]
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
        raise PermissionError(f"the registered {stage} evidence does not match the plan")
    binding = evidence["artifact_manifest"]
    directory = root / "results" / binding["run_id"]
    verify_artifact_directory(directory, strict=True, expected_manifest_sha256=binding["sha256"])
    return directory, evidence


def _unit_tests(root, directory):
    temp = root / PYTEST_TEMP
    if temp.exists():
        raise FileExistsError(f"reserved pytest directory already exists: {temp}")
    try:
        check = subprocess.run(
            [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
             "--basetemp", PYTEST_TEMP, V3_TEST],
            cwd=root, capture_output=True, check=False,
        )
    finally:
        if temp.exists():
            if temp.resolve() != temp or not temp.is_relative_to(root / "scratch" / "one_off"):
                raise RuntimeError("unexpected pytest temporary path")
            shutil.rmtree(temp)
    log = directory / "pytest.log"
    log.write_bytes(check.stdout + check.stderr)
    if check.returncode:
        raise RuntimeError("V3 unit invariants did not pass; see pytest.log")
    return sha256_file(log)


def _baseline_parity(root, plan, rows):
    verify_bound_file(root, V2_SUMMARY, _json(root / plan["v2_receipt"])["evidence_files"][V2_SUMMARY])
    v2 = _json(root / V2_SUMMARY)["steps"]
    comparisons = {}
    for row in rows:
        if row["track"] != "A":
            continue
        step = str(row["step_s"])
        previous = v2[step]
        comparisons[step] = {}
        for label in ("orig", "v1"):
            values = [row["endpoint_target_peak_means"][ep][label] for ep in ("0", "10")]
            targets = [previous["endpoint_target_peak_means"][ep][label] for ep in ("0", "10")]
            values.append(row["models"][label]["aggregate_nrmse"])
            targets.append(previous["models"][label]["aggregate_nrmse"])
            if not all(math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-11)
                       for x, y in zip(values, targets, strict=True)):
                raise ValueError(f"track A changed the frozen V2 test case or baseline: {step}/{label}")
            comparisons[step][label] = max(abs(x - y) for x, y in zip(values, targets, strict=True))
    return comparisons


def _synthetic(root, directory, plan):
    pytest_sha = _unit_tests(root, directory)
    prefix, rows = "V3_PULSE_DIAGNOSTICS ", []
    with (
        (directory / "pulse_diagnostics.log").open("wb") as log,
        (directory / "pulse_diagnostics.stderr").open("wb") as errors,
        subprocess.Popen([sys.executable, "-B", V3_TEST], cwd=root, stdout=subprocess.PIPE,
                         stderr=errors) as process,
    ):
        for line in process.stdout:
            log.write(line)
            if line.startswith(prefix.encode()):
                row = json.loads(line[len(prefix):])
                rows.append(row)
                print(f"Synthetic {row['track']} {row['step_s']} s: {row['passing_candidates']}", flush=True)
        if process.wait():
            raise RuntimeError("the V3 synthetic comparison did not complete; see pulse_diagnostics.stderr")
    expected = tuple(product(("A", "B"), STEPS))
    cases = list(product(range(10), (0, 10), (1., -1.)))
    names = [candidate["name"] for candidate in plan["candidates"]]
    if tuple((row["track"], row["step_s"]) for row in rows) != expected:
        raise RuntimeError("the synthetic comparison did not cover both tracks and both steps")
    for row in rows:
        multiplier = 1. if row["track"] == "A" else plan["synthetic_gate"]["track_b_truth_stiffness_damping_multiplier"]
        if (row["device"] != "cuda:0" or row["device_name"] != "NVIDIA GeForce RTX 4060 Ti"
                or row["dtype"] != "torch.float64" or set(row["models"]) != {"orig", "v1", *names}
                or row["truth_stiffness_factor"] != multiplier or row["truth_damping_factor"] != multiplier
                or [(c["machine"], c["endpoint"], c["sign"]) for c in row["cases"]] != cases
                or not set(row["passing_candidates"]).issubset(names)):
            raise RuntimeError("the synthetic comparison differs from its registered coverage")
    qualifying = [name for name in names if all(name in row["passing_candidates"] for row in rows)]
    diagnostic_path = directory / "pulse_diagnostics.json"
    write_json(diagnostic_path, {"tracks": rows, "track_a_v2_baseline_parity": _baseline_parity(root, plan, rows),
                                "qualifying_candidates": qualifying})
    return {"real_data_reads": 0, "pytest_sha256": pytest_sha,
            "pulse_diagnostics_sha256": sha256_file(diagnostic_path), "qualifying_candidates": qualifying,
            "scientific_verdict": "ACCEPT" if qualifying else "REJECT"}


def _frozen_files(root, plan):
    files = [p.relative_to(root).as_posix() for group in ("src/kron_reduce", "project_harness")
             for p in (root / group).rglob("*.py")]
    files += [CONFIG, WORKFLOW_CONFIG, plan["protocol"], plan["v1_plan"], plan["v1_receipt"],
              plan["v2_receipt"], plan["baseline_plan"], REFERENCE, "configs/input_contracts.json",
              "requirements/constraints-py312.txt", V2_FIXTURE, V3_TEST,
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
    val_directory, _ = _phase(root, "validation", digest)
    expected = evidence["frozen_model_manifest"]["sha256"]
    verify_bound_file(root, SEALED_FREEZE, expected)
    frozen = _json(root / SEALED_FREEZE)
    if (not frozen["selection_closed"] or not frozen["fitting_closed"] or frozen["plan_sha256"] != digest
            or frozen["weights"] != assets["weights"] or frozen["files"] != _frozen_files(root, plan)):
        raise PermissionError("code, plan or weights changed after the V3 freeze")
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
    manifest = directory / "noise_manifest.json"
    write_json(manifest, _json(val_directory / "noise_manifest.json") | _json(manifest))
    metrics["noise_manifest_sha256"] = sha256_file(manifest)
    return {"test_cases": len(test), "blind_cases": len(blind), "test_trajectory_reads": len(test),
            "blind_container_opens": 1, "optimizer_updates": 0, "frozen_model_manifest": binding,
            **metrics}


def _development(root, directory, plan, digest, source, models, scales, *, adapt):
    _, synthetic = _phase(root, "synthetic", digest)
    if synthetic["scientific_verdict"] != "ACCEPT" or not synthetic["qualifying_candidates"]:
        raise PermissionError("both synthetic tracks must accept V3 before val")
    if adapt:
        prior, previous = _phase(root, "validation", digest)
        if previous["scientific_verdict"] != "RETRY":
            raise PermissionError("conditional adaptation requires a failed zero-retrain gate")
        initial = _json(prior / "selection.json")
    reference, noise, _ = v1_reference(root, plan)
    with closing(AppendOnlyAuditJournal(directory / "access.jsonl")) as access:
        val = DevelopmentCache(root, source, access).load("val")
        if adapt:
            train = DevelopmentCache(root, source, access).load("train")

    def assess(target, candidates, update):
        scores, metrics = evaluate_matrix(target, {"val": val}, models, scales, plan,
                                          candidates=candidates, reference_noise=noise)
        return select_candidate(target, plan, scores, reference, candidates, update), metrics

    if not adapt:
        selected, evidence = assess(directory, plan["candidates"], 0)
        if selected["passed"]:
            benchmark(directory, val, models, scales, plan, selected["candidate"])
        return {"val_cases": len(val), "optimizer_updates": 0, **evidence,
                "scientific_verdict": "ACCEPT" if selected["passed"] else "RETRY"}

    from kron_reduce.evaluation.noise_stress_fit_v3 import fit_candidates

    def assess_checkpoint(checkpoint, candidate, update):
        return assess(checkpoint, [candidate], update)[0]

    selected = fit_candidates(directory, train, models, scales, plan, initial, assess_checkpoint)
    if selected["passed"]:
        benchmark(directory, val, models, scales, plan, selected["candidate"])
    return {"train_cases": len(train), "val_cases": len(val), "optimizer_updates": selected["optimizer_updates"],
            "scientific_verdict": "ACCEPT" if selected["passed"] else "REJECT",
            "training_sha256": sha256_file(directory / "training_curves.json")}


@torch.no_grad()
def run(root, experiment, stage, result_dir, run_id):
    validate_run_policy(root, WORKFLOW, experiment)
    plan, baseline, digest = _plans(root)
    write_json(result_dir / "runtime.json", _runtime())
    evidence = {"source_commit": git_head(root), "plan_sha256": digest, "claim_boundary": CLAIM}
    if stage == "synthetic":
        evidence.update(_synthetic(root, result_dir, plan))
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
