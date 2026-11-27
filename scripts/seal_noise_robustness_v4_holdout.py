"""Seal a completed one-shot matrix after the V4 exclusive-write collision."""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import Counter
from pathlib import Path
from shutil import copyfile

from kron_reduce.evaluation.constant_c_observability import write_json
from kron_reduce.evaluation.noise_stress_matrix import _summary
from kron_reduce.evaluation.noise_stress_matrix_v4 import V4
from kron_reduce.workflows.noise_robustness_esrom import _write_evidence
from kron_reduce.workflows.noise_robustness_esrom_v4 import WORKFLOW, _phase, _plans
from project_harness.evidence import _verify_real_git_commit, sha256_file, verify_artifact_directory
from project_harness.loop import approval_recorded, load_state, workflow_lock
from project_harness.manifest import project_root
from scripts.register_noise_robustness_v4_holdout import _freeze_proof, _json_at, _reservation

STEPS = {"0.0025", "0.00125"}


def _verify_case_files(source, plan, scores, candidate):
    expected = set()
    for roster, conditions in scores["rosters"].items():
        if set(conditions) != set(plan["scenarios"]):
            raise ValueError(f"incomplete scenario list for {roster}")
        count = plan["counts"][roster]
        for scenario, models in conditions.items():
            repeats = 1 if scenario == "clean" else 5
            if set(models) != set(plan["models"]) or models[V4]["candidate"] != candidate:
                raise ValueError(f"model roster or frozen candidate mismatch for {roster}/{scenario}")
            for model, result in models.items():
                if len(result["repeats"]) != repeats:
                    raise ValueError(f"incomplete repetition list for {roster}/{scenario}/{model}")
                for rep, summary in enumerate(result["repeats"]):
                    relative = f"cases/{roster}/{scenario}/{model}/rep{rep}.json.gz"
                    expected.add(relative)
                    with gzip.open(source / relative, "rt", encoding="utf-8") as stream:
                        detail = json.load(stream)
                    ids = {row["id"] for row in detail["pairwise"]["rows"]}
                    if (set(detail["steps"]) != STEPS or len(ids) != count
                            or len(detail["pairwise"]["rows"]) != count
                            or summary["pairwise"]["allocated"] != count or _summary(detail) != summary):
                        raise ValueError(f"case summary or paired roster mismatch: {relative}")
                    for step in detail["steps"].values():
                        if len(step["rows"]) != count or {row["id"] for row in step["rows"]} != ids:
                            raise ValueError(f"each step must cover the paired roster: {relative}")
    return expected


def _verify_journal(source, freeze):
    rows = [json.loads(line) for line in (source / "access.jsonl").read_text(encoding="utf-8").splitlines()]
    counts = Counter((row["stage"], row.get("role")) for row in rows)
    if (rows[0]["stage"] != "freeze_verified" or rows[0]["sha256"] != freeze["sha256"]
            or counts["freeze_verified", None] != 1 or counts["decoded", "test"] != 76
            or counts["decode_container_reserved", "blind_v2"] != 1
            or counts["event_decoded", "blind_v2"] != 94
            or {row.get("role") for row in rows} != {None, "test", "blind_v2"}):
        raise ValueError("the prediction journal does not bind a single post-freeze read of each roster")


def _verify_noise(plan, original, val):
    if set(original) != {"test", "blind_v2"} or set(val) != {"val"}:
        raise ValueError("the noise manifests do not bind three separate rosters")
    combined = val | original
    for conditions in combined.values():
        if set(conditions) != set(plan["scenarios"]):
            raise ValueError("incomplete noise scenario roster")
        for scenario, repetitions in conditions.items():
            expected = {str(value) for value in range(1 if scenario == "clean" else 5)}
            if set(repetitions) != expected or any(re.fullmatch(r"[0-9a-f]{64}", sha) is None
                                                   for sha in repetitions.values()):
                raise ValueError("invalid noise repetition SHA-256")
    return combined


def _verify_bootstrap(source, plan):
    results, _ = _json_at(source / "blind_group_bootstrap.json")
    if set(results) != set(plan["scenarios"]):
        raise ValueError("incomplete blind fault-group bootstrap")
    for scenario, receipt in results.items():
        if (receipt["seed"] != plan["bootstrap_seed"] or receipt["replicates"] != 10000
                or receipt["noise_replicates"] != (1 if scenario == "clean" else 5)
                or receipt["groups_per_rep"] != 21 or set(receipt["by_step"]) != STEPS):
            raise ValueError(f"invalid blind fault-group bootstrap: {scenario}")


def seal(root, run_id, evidence_path):
    state = load_state(root, WORKFLOW)
    if state["current_stage"] != "freeze" or not approval_recorded(state, "blind"):
        raise PermissionError("sealing requires the existing approved and frozen joint scope")
    plan, _, digest = _plans(root)
    freeze, freeze_bytes = _freeze_proof(root, state, plan, digest)
    reservation, _ = _json_at(_reservation(root, plan["scope"]))
    if (re.fullmatch(rf"{WORKFLOW}-blind-[0-9]{{8}}T[0-9]+Z-[0-9a-f]{{8}}-[0-9a-f]{{8}}", run_id) is None
            or reservation["run_id"] != run_id or reservation["retry_allowed"] is not False):
        raise PermissionError("only the reserved joint prediction run can be sealed")
    _verify_real_git_commit(root, reservation["source_commit"])
    source = root / "results" / run_id
    if source.resolve(strict=True) != source:
        raise PermissionError("unsafe prediction results path")
    failure, _ = _json_at(source / "RUN_FAILED.json")
    if (failure["error_type"] != "FileExistsError" or failure["run_id"] != run_id
            or failure["automatic_retry"] is not False or failure["status"] != "failed"
            or str(source / "noise_manifest.json") not in failure["message"].replace("\\\\", "\\")):
        raise PermissionError("only the exclusive noise-manifest write collision can be recovered")

    scores, _ = _json_at(source / "metrics.json")
    allocated = scores["operations"]["allocated_event_predictions"]
    if set(scores["rosters"]) != {"test", "blind_v2"} or allocated != plan["limits"]["holdout_predictions"]:
        raise ValueError("the reservation did not finish its complete prediction matrix")
    expected = _verify_case_files(source, plan, scores, json.loads(freeze_bytes)["candidate"])
    expected.update({"access.jsonl", "blind_group_bootstrap.json", "metrics.json", "noise_manifest.json",
                     "RUN_FAILED.json", "runtime.json"})
    paths = list(source.rglob("*"))
    if any(path.is_symlink() for path in paths) or {p.relative_to(source).as_posix() for p in paths if p.is_file()} != expected:
        raise ValueError("unexpected or missing files in the original holdout results")
    _verify_bootstrap(source, plan)
    _verify_journal(source, freeze)
    original, _ = _json_at(source / "noise_manifest.json")
    val, evidence = _phase(root, "validation", digest)
    val_noise, _ = _json_at(val / "noise_manifest.json", evidence["noise_manifest_sha256"])
    noise = _verify_noise(plan, original, val_noise)

    target = source.with_name(f"{source.name}-sealed")
    target.mkdir(exist_ok=False)
    names = {"RUN_FAILED.json": "source_failure_receipt.json", "noise_manifest.json": "heldout_noise_manifest.json"}
    hashes = {name: sha256_file(source / name) for name in sorted(expected)}
    for name, checksum in hashes.items():
        output = target / names.get(name, name)
        output.parent.mkdir(parents=True, exist_ok=True)
        copyfile(source / name, output)
        if sha256_file(output) != checksum:
            raise ValueError(f"copy changed the original artifact: {name}")
    write_json(target / "noise_manifest.json", noise)
    recovery = target / "recovery_receipt.json"
    write_json(recovery, {"prediction_run_id": run_id, "additional_predictions": 0, "source_files": hashes})
    status = {
        "source_commit": reservation["source_commit"], "plan_sha256": digest,
        "claim_boundary": json.loads(freeze_bytes)["claim_boundary"], "test_cases": 76, "blind_cases": 94,
        "test_trajectory_reads": 76, "blind_container_opens": 1, "optimizer_updates": 0,
        "frozen_model_manifest": freeze, "noise_manifest_sha256": sha256_file(target / "noise_manifest.json"),
        "metrics_sha256": hashes["metrics.json"], "event_predictions": allocated,
        "receipt_recovery": {"prediction_run_id": run_id, "failure_sha256": hashes["RUN_FAILED.json"],
                             "provenance_sha256": sha256_file(recovery)},
    }
    _write_evidence(target, status)
    verify_artifact_directory(target, strict=True)
    binding = {"run_id": target.name, "path": "artifact_hashes.json",
               "sha256": sha256_file(target / "artifact_hashes.json")}
    write_json(evidence_path, status | {"status": "PASS", "result_dir": target.name, "artifact_manifest": binding,
                                       "result_artifact_manifest": binding})
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--evidence-out", required=True, type=Path)
    args = parser.parse_args()
    root = project_root()
    evidence = (root / args.evidence_out).resolve()
    if not evidence.is_relative_to(root / "scratch" / "one_off") or evidence.exists():
        raise ValueError("the external evidence must be a new file under scratch/one_off")
    with workflow_lock(root, WORKFLOW):
        target = seal(root, args.run_id, evidence)
    print(target)


if __name__ == "__main__":
    main()
