"""Dedicated dispatch for constructing and evaluating the sealed M5 cohort."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
import tomllib
from pathlib import Path

from project_harness.evidence import sha256_file
from project_harness.git_state import assert_clean, git_head, verify_bound_file
from project_harness.loop import approval_recorded, approve, load_state, workflow_lock
from project_harness.manifest import load_manifest, project_root
from project_harness.policy import load_policy, validate_run_policy
from project_harness.report_storage import locate_report
from project_harness.runner import _run_experiment

WORKFLOW = "blind_confirmation_m5_v1"
EXPERIMENTS = {"build": "blind_dataset_build_m5_v1", "blind": WORKFLOW}
COHORT = "data/kiki_ieee39_blind_v2"
PLAN = f"configs/experiments/{WORKFLOW}.toml"
ZERO = {"fits": 0, "gradients": 0, "optimizer_updates": 0}


def _read(path):
    return json.loads(path.read_bytes())


def _write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def _authorize(root, stage, experiment):
    key = experiment["development_authorization_id" if stage == "build" else "blind_authorization"]
    policy = load_policy(root)
    scope = policy["development_evaluations" if stage == "build" else "independent_confirmations"][key]
    state = load_state(root, WORKFLOW)
    expected_state = "scaffold" if stage == "build" else "build"
    expected = {"status": "active", "experiment": EXPERIMENTS[stage], "plan_path": PLAN,
                "max_executions": 1, "retry_allowed": False, "fit_attempts": 0, "gradients": 0,
                "optimizer_updates": 0, "test_cases": 0}
    expected.update({"allowed_roles": ["historical_clearing_frame_hashes"],
                     "historical_clearing_frames": 529, "fixed_test_clearing_frames_only": 76, "max_simulations": 210}
                    if stage == "build" else {"allowed_roles": ["blind"], "events": 94})
    if (state["current_stage"] != expected_state
            or any(type(scope.get(k)) is not type(v) or scope[k] != v for k, v in expected.items())):
        raise PermissionError("this stage requires its registered unused scope and predecessor")
    validate_run_policy(root, EXPERIMENTS[stage], experiment)
    verify_bound_file(root, PLAN, scope["plan_sha256"])
    plan = tomllib.loads((root / PLAN).read_text(encoding="utf-8"))
    for relative, digest in plan["bindings"].items():
        verify_bound_file(root, relative, digest)
    if stage == "build":
        if (root / COHORT).exists():
            raise FileExistsError("the destination for the independent cohort already exists")
    else:
        receipt_path = scope["build_receipt_path"]
        verify_bound_file(root, receipt_path, scope["build_receipt_sha256"])
        receipt = _read(root / locate_report(root, receipt_path, scope["build_receipt_sha256"]))
        prior = next(row["evidence"] for row in state["history"] if row.get("to") == "build")
        for key_name in ("dataset_manifest_sha256", "dataset_checksums_sha256"):
            if not scope[key_name] == receipt[key_name] == prior[key_name]:
                raise ValueError("the build receipt and seal authorization disagree")
        if (receipt["plan_sha256"] != scope["plan_sha256"] or receipt["run_id"] != prior["result_dir"]
                or receipt["status"] != "PASS"):
            raise ValueError("the original sealed build does not match this evaluation plan")
    return key, scope, plan


def _freeze_checks(root, plan):
    v4_path = plan["v4_receipt"]
    m3b_path = plan["m3b_receipt"]
    v4 = _read(root / locate_report(root, v4_path, plan["bindings"][v4_path]))
    m3b = _read(root / locate_report(root, m3b_path, plan["bindings"][m3b_path]))
    weights = v4["parameter_checkpoints"] | {"m3": m3b["frozen_parameters"]}
    for name, binding in weights.items():
        expected = plan["m3b_parameters_sha256" if name == "m3" else "v4_parameters_sha256"]
        if binding["sha256"] != expected or sha256_file(root / binding["path"]) != expected:
            raise ValueError(f"frozen checkpoint changed: {name}")
    manifests = [receipt["frozen_model_manifest"] for receipt in (v4, m3b)]
    for binding in manifests:
        if sha256_file(root / binding["path"]) != binding["sha256"]:
            raise ValueError("the frozen model manifest changed")
        manifest = _read(root / binding["path"])
        if (not manifest["selection_closed"] or not manifest["fitting_closed"]
                or sha256_file(root / manifest["preprocessing_path"]) != plan["preprocessing_sha256"]):
            raise ValueError("the frozen scoring setup changed")
    return weights


def _reserve(root, key, run_id, commit):
    common = Path(subprocess.check_output(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                                          cwd=root, text=True).strip()).resolve(strict=True)
    marker = common / "m4v2_m5_budgets" / f"{key}.json"
    marker.parent.mkdir(exist_ok=True)
    with marker.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump({"run_id": run_id, "source_commit": commit, "retry_allowed": False}, stream)
        stream.flush()
        os.fsync(stream.fileno())


def _build(root, directory, run_id, plan):
    from kron_reduce.data.blind_dataset_m5 import generate_and_seal

    files = {name: (root / plan[name]).resolve(strict=True)
             for name in ("source_contract", "registry_manifest", "generator", "generator_config", "kiki_src")}
    bindings = {files[name]: plan["bindings"][plan[name]] for name in files if name != "kiki_src"}
    bindings.update({files["kiki_src"] / path: digest for path, digest in plan["kiki_bindings"].items()})
    candidates = _read(root / plan["candidate_schedule"])
    if len(candidates) != 210:
        raise ValueError("the simulation budget requires the complete registered schedule")
    staging = directory / "staging"
    result = generate_and_seal(files | {"file_sha256": bindings, "candidates": candidates,
                                        "history_hdf": plan["history_hdf"]}, staging,
                               solver_workdir=root / "scratch" / "one_off" / run_id / "andes",
                               evidence_directory=directory)
    for name in ("dataset.h5", "manifest.json", "checksums.sha256", "splits.json"):
        (staging / name).chmod(0o444)
    staging.rename(root / COHORT)
    access = directory / "historical_access.jsonl"
    operations = directory / "attempts.jsonl"
    return result | {"historical_clearing_frames": 529, "fixed_test_clearing_frames_only": 76,
                     "test_full_trajectory_reads": 0,
                     "model_rollouts": 0, "access_sha256": sha256_file(access),
                     "operations_sha256": sha256_file(operations)}


def _sealed_paths(root, scope):
    basedir = root / COHORT
    names = {"dataset.h5", "dataset_index.csv", "manifest.json", "splits.json", "qa.json", "qa_report.md",
             "disjointness_audit.json"}
    rows = (basedir / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    hashes = {}
    for row in rows:
        digest, name = row.split("  ", maxsplit=1)
        if name not in names or name in hashes or re.fullmatch("[0-9a-f]{64}", digest) is None:
            raise ValueError("the checksum manifest has an invalid or duplicate member")
        hashes[name] = digest
    if hashes.keys() != names or sha256_file(basedir / "checksums.sha256") != scope["dataset_checksums_sha256"]:
        raise ValueError("the checksum manifest differs from the one-use scope")
    for name, digest in hashes.items():
        path = basedir / name
        if name == "dataset.h5":
            if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
                raise ValueError("sealed HDF5 failed its checksum")
        else:
            verify_bound_file(root, f"{COHORT}/{name}", digest)
    verify_bound_file(root, f"{COHORT}/checksums.sha256", scope["dataset_checksums_sha256"])
    return basedir, hashes


def _load_blind_events(basedir, hashes, access):
    import h5py
    import numpy as np

    from kron_reduce.physics.coordinates import coi_project, coi_project_power

    manifest = _read(basedir / "manifest.json")
    roster = _read(basedir / "splits.json")["blind"]
    with (basedir / "dataset_index.csv").open(newline="", encoding="utf-8") as stream:
        indexed = list(csv.DictReader(stream))
    labels = {item["event_id"]: item["label"] for item in indexed}
    if (manifest["dataset_id"] != Path(COHORT).name or manifest["cases"] != 94
            or manifest["labels"] != {"unstable": 32, "stable": 62}
            or _read(basedir / "qa.json")["status"] != "PASS"
            or _read(basedir / "disjointness_audit.json")["accepted_overlap_count"] != 0
            or list(labels) != roster or len(roster) != 94 or len(indexed) != 94
            or any(re.fullmatch("blind_[0-9]{4}", name) is None for name in roster)
            or list(labels.values()).count("stable") != 62 or list(labels.values()).count("unstable") != 32):
        raise ValueError("the independent cohort differs from the sealed roster")
    events = []
    access.append({"stage": "decode_container_reserved", "path": f"{COHORT}/dataset.h5",
                   "sha256": hashes["dataset.h5"], "role": "blind", "events": len(roster)})
    with h5py.File(basedir / "dataset.h5", "r") as store:
        for key, expected in (("dataset_id", manifest["dataset_id"]), ("case_sha256", manifest["case_sha256"]),
                              ("postfault_signature", manifest["restored_system_signature"]),
                              ("sample_interval_s", .01), ("omega_s", 2 * np.pi * 60)):
            if store.attrs[key] != expected:
                raise ValueError(f"the blind HDF5 {key} does not match the registered system")
        omega_s = float(store.attrs["omega_s"])
        mass = store["inertia"][...] / omega_s
        delta_star = store["equilibrium/delta"][...]
        pm_star = store["equilibrium/Pm"][...]
        if set(store["events"]) != set(roster) or any(x.shape != (10,) or not np.isfinite(x).all()
                                                       for x in (mass, delta_star, pm_star)) or np.any(mass <= 0):
            raise ValueError("the blind HDF5 roster or power-system data is invalid")

        equilibrium = coi_project(delta_star, mass)
        for row in indexed:
            name, label = row["event_id"], row["label"]
            identity = f"{manifest['dataset_id']}/{name}"
            event = store["events"][name]
            if (event.attrs["label"] != label or int(event.attrs["valid_length"]) != int(row["valid_length"])
                    or str(event.attrs["event_time_s"]) != row["event_time_s"]
                    or str(event.attrs["fault_bus"]) != row["fault_bus"]
                    or event.attrs["fault_line"] != row["fault_line"]):
                raise ValueError(f"the locked index disagrees with HDF5 for {identity}")
            length = min(int(row["valid_length"]), 811)
            if (label == "stable" and length != 811) or length < 12:
                raise ValueError(f"insufficient observations or stable coverage for {identity}")
            access.append({"stage": "event_decode_reserved", "role": "blind", "id": identity,
                           "selection": {"t": length, "delta": length, "omega": length, "Pm_row": 10}})
            t, delta, omega = (event[field][:length] for field in ("t", "delta", "omega"))
            observed_power = event["Pm"][10] - pm_star
            if (not np.array_equal(t, np.arange(length) * .01)
                    or delta.shape != (length, 10) or omega.shape != (length, 10)
                    or not all(np.isfinite(x).all() for x in (delta, omega, observed_power))):
                raise ValueError(f"invalid sealed samples for {identity}")
            q, v_raw = coi_project(delta - delta_star, mass), (omega - 1.) * omega_s
            p0 = coi_project_power(observed_power, mass)
            events.append({"id": identity, "role": "blind", "label": label,
                           "group": f"bus_{row['fault_bus']}/{row['fault_line']}",
                           "x": np.column_stack((coi_project(q, mass), coi_project(v_raw, mass))),
                           "raw_x": np.column_stack((q, v_raw)),
                           "p0": p0, "mass": mass, "equilibrium": equilibrium, "omega_s": omega_s,
                           "event_time_s": float(row["event_time_s"]) if label == "unstable" else None})
            access.append({"stage": "event_decoded", "role": "blind", "id": identity})
    return events


def _evaluate(root, directory, scope, plan):
    import numpy as np
    import torch

    from kron_reduce.data.durable_audit_ledger import AppendOnlyAuditJournal
    from kron_reduce.evaluation.blind_confirmation_m5 import MODEL_NAMES, execute
    from kron_reduce.evaluation.machine_local_synthetic import Budget
    from kron_reduce.workflows.postfault_m4_v2 import frozen_models

    torch.set_num_threads(1)
    if not torch.cuda.is_available() or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("the frozen evaluation requires CUDA and CUBLAS_WORKSPACE_CONFIG=:4096:8")
    torch.use_deterministic_algorithms(True)
    source, raw_models, scales, _ = frozen_models(root, plan, device="cuda")
    models = dict(zip(MODEL_NAMES, (raw_models[name].requires_grad_(False).eval()
                                   for name in ("hard_r", "soft_2r", "m3")), strict=True))
    basedir, hashes = _sealed_paths(root, scope)
    access = AppendOnlyAuditJournal(directory / "access.jsonl")
    budget = Budget(directory / "operations.jsonl", plan["operations"])
    try:
        events = _load_blind_events(basedir, hashes, access)
        for model in models.values():
            if not np.allclose(events[0]["mass"], model.mass.detach().cpu().numpy(), rtol=1e-12, atol=1e-12):
                raise ValueError("the sealed inertia disagrees with the frozen models")
        with budget.scope("blind"), torch.no_grad():
            result = execute(root, directory, plan, source, events, models, scales, budget)
    finally:
        access.close()
        budget.stream.close()
    return {"events": len(events), "blind_container_opens": 1, "test_trajectory_reads": 0,
            "operations": dict(budget.counts),
            "numerical_pass": result["numerical_pass"], "cuda_device": torch.cuda.get_device_name(),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "dataset_manifest_sha256": scope["dataset_manifest_sha256"],
            "dataset_checksums_sha256": scope["dataset_checksums_sha256"]}


def run(*, root, experiment, stage, result_dir, run_id):
    assert_clean(root)
    commit = git_head(root)
    key, scope, plan = _authorize(root, stage, experiment)
    if not approval_recorded(load_state(root, WORKFLOW), stage) or sys.version_info[:2] != (3, 12):
        raise PermissionError("the registered Python 3.12 runtime and operator approval are required")
    weights = _freeze_checks(root, plan)
    task3_root = Path(_read(root / plan["source_contract"])["task3_root"])
    for relative, digest in plan["task3_bindings"].items():
        if sha256_file(task3_root / relative) != digest:
            raise ValueError(f"the official TASK3 scoring source changed: {relative}")
    _reserve(root, key, run_id, commit)
    timed = time.perf_counter()
    try:
        summary = (_build(root, result_dir, run_id, plan) if stage == "build"
                   else _evaluate(root, result_dir, scope, plan))
        evidence = {"status": "PASS", "process_status": "PASS", "source_commit": commit,
                    "plan_sha256": scope["plan_sha256"], "claim_boundary": "Independent IEEE39 post-fault cohort.",
                    **ZERO, **summary}
        _write(result_dir / "stage_evidence.json", evidence)
        _write(result_dir / "run_receipt.json", evidence | {"run_id": run_id, "frozen_models": weights,
               "runtime": {"python": sys.version, "executable": sys.executable,
                           "wall_seconds": time.perf_counter() - timed}})
    finally:
        _write(result_dir / "artifact_hashes.json", {p.relative_to(result_dir).as_posix(): sha256_file(p)
               for p in sorted(result_dir.rglob("*")) if p.is_file()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=EXPERIMENTS)
    stage = parser.parse_args().stage
    root = project_root()
    assert_clean(root)
    experiment = load_manifest(root)["experiments"][EXPERIMENTS[stage]]
    _authorize(root, stage, experiment)
    approve(root, stage, WORKFLOW)
    with workflow_lock(root, WORKFLOW):
        print(_run_experiment(root, EXPERIMENTS[stage], stage, experiment=experiment), flush=True)
