"""Dedicated one-use A16 history-input diagnostic; no training or predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tomllib
from collections import Counter
from dataclasses import fields, is_dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
from filelock import FileLock

from kron_reduce.data.durable_audit_ledger import AppendOnlyAuditJournal
from kron_reduce.data.pm_impulse_reader import RUN, SEEDS, SOURCES, PmImpulseReader, validate_plan
from kron_reduce.evaluation.pm_impulse_audit import (
    Dual,
    NumericalError,
    ScalarLedger,
    SlotResult,
    aggregate_slots,
    audit_slot,
    build_history_impulse,
    expected_budget,
    prepare_source,
)
from kron_reduce.workflows.kiki_record20_input_audit import root_attribute_validator
from project_harness.evidence import sha256_file
from project_harness.git_state import assert_clean, git_head, verify_bound_file
from project_harness.manifest import load_manifest
from project_harness.policy import load_policy, validate_run_policy
from project_harness.runner import _run_experiment, _verify_development_text_bindings

AUTH = "kiki_a16_development_20261025"
EXPERIMENT = "paper_kiki_a16_pm_impulse"
WORKFLOW = "kiki_a16_pm_impulse_v1"
STAGE = "input_impulse"
MODULE = "kron_reduce.workflows.kiki_pm_impulse_audit"
MODE = "bounded_history_cache_audit_cli_only"
BOUNDARY = ("Uncertified original-train history-input diagnostic only. No new state, fitting, "
            "prediction, future attribution, labels, FSR, certification, promotion or deployment.")
ZERO_COMPUTE = {key: 0 for key in ("model_constructions", "fit_attempts", "optimizer_updates",
    "rhs", "rollouts", "force", "expm", "svd", "qr", "eig", "inverse", "solve", "new_latent_states")}


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pack(value, arrays, prefix="root"):
    if isinstance(value, Dual):
        return value.record()
    if is_dataclass(value):
        return {field.name: pack(getattr(value, field.name), arrays, f"{prefix}__{field.name}")
                for field in fields(value)}
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("object evidence is forbidden")
        arrays[prefix] = value
        return {"array_member": prefix, "shape": list(value.shape), "dtype": value.dtype.str,
                "sha256_c_order_bytes": hashlib.sha256(value.tobytes(order="C")).hexdigest()}
    if isinstance(value, dict):
        return {str(k): pack(v, arrays, f"{prefix}__{k}") for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if value and all(isinstance(v, Dual) for v in value):
            descriptor = pack(np.array([[v.point, v.lower, v.upper] for v in value]), arrays, prefix)
            descriptor["columns"] = ["point", "lower", "upper"]
            return descriptor
        return [pack(v, arrays, f"{prefix}__{i}") for i, v in enumerate(value)]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return {"value": None, "reason": "nonfinite_return", "representation": repr(value)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported evidence value {type(value).__name__}")


def save_record(directory, name, value):
    arrays = {}
    metadata = pack(value, arrays)
    js, npz = Path(directory) / f"{name}.json", Path(directory) / f"{name}.npz"
    with npz.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    write_json(js, metadata)
    return {js.name: sha256_file(js), npz.name: sha256_file(npz)}


def persistable_counts(snapshot):
    """Keep exact integer counters and explicitly preserve nonfinite returns."""
    counts = dict(snapshot)
    last = counts.get("last_numeric_return")
    if isinstance(last, dict):
        counts["last_numeric_return"] = dict(last)
        value = last.get("value")
        if isinstance(value, float) and not math.isfinite(value):
            counts["last_numeric_return"].update(value=None, representation=repr(value),
                                                  reason="nonfinite_numeric_return")
    return counts


def authorization(root, experiment, config):
    """All checks precede budget consumption and real-data byte access."""
    validate_run_policy(root, EXPERIMENT, experiment)
    _verify_development_text_bindings(root, experiment)
    approval = load_policy(root).get("development_evaluations", {}).get(AUTH, {})
    expected = {"status": "user_approved", "authority": "explicit_user_reply",
        "scientific_design_approver": "Astra Ultra", "experiment": EXPERIMENT,
        "workflow": WORKFLOW, "stage": STAGE, "module": MODULE, "dataset": "kiki_ieee39_v1",
        "max_campaign_executions": 1, "retry_allowed": False, "max_fit_attempts": 0,
        "max_rhs": 0, "internal_val_payload_allowed": False, "source_val_payload_allowed": False,
        "source_test_payload_allowed": False, "model_promotion_allowed": False,
        "deployment_predictions_allowed": False}
    if any(type(approval.get(k)) is not type(v) or approval.get(k) != v for k, v in expected.items()):
        raise PermissionError("A16 independent one-use authorization is inactive")
    if any(k not in approval or approval[k] != v for k, v in config.items()):
        raise PermissionError("A16 config differs from the approved plan")
    if (experiment != load_manifest(root)["experiments"].get(EXPERIMENT)
            or experiment.get("execution_mode") != MODE
            or config["input_contract"] != "measured_history_100ms_zoh_v1"):
        raise PermissionError("A16 execution identity mismatch")
    for pk, sk in (("read_plan_path", "read_plan_sha256"), ("compute_plan_path", "compute_plan_sha256")):
        verify_bound_file(root, config[pk], config[sk])
    verify_bound_file(root, approval["scientific_review_path"], approval["scientific_review_sha256"])
    bindings = approval.get("implementation_bindings", {})
    if "src/" + MODULE.replace(".", "/") + ".py" not in bindings:
        raise PermissionError("A16 lacks reviewed implementation bindings")
    for path, digest in bindings.items():
        verify_bound_file(root, path, digest)
    assert_clean(root)
    return approval


def bound_metadata(root, config):
    plan = validate_plan(read_json(root / config["read_plan_path"]))
    if (plan["authorization"] != AUTH or plan["source_run_id"] != RUN
            or plan["population"]["cache_rows"] != plan["events"]
            or len(plan["population"]["allocated_slots"]) != 268
            or len(plan["population"]["seed_event_slots"]) != 774):
        raise PermissionError("A16 frozen population identity differs")
    expected = {"npz_contexts": 778, "npz_members": 6978, "npz_array_elements": 502222,
        "npz_integrity_streams": 1556, "npz_integrity_bytes": 190630248, "hdf_contexts": 2,
        "hdf_array_selections": 520, "hdf_array_elements": 31258, "hdf_attributes": 18,
        "hdf_integrity_streams": 4, "hdf_integrity_bytes": 6870747346,
        "array_selections": 7498, "array_descriptors": 7498, "array_elements": 533480,
        "metadata_files": 783, "metadata_bytes": 56170683}
    if plan["limits"] != expected:
        raise PermissionError("A16 finite source budget differs")
    if (2 * sum(row["size_bytes"] for row in plan["npz_files"]) != expected["npz_integrity_bytes"]
            or 2 * sum(row["size_bytes"] for row in plan["source_inventory"].values()) != expected["hdf_integrity_bytes"]
            or len(plan["metadata_files"]) != expected["metadata_files"]
            or sum(row["size_bytes"] for row in plan["metadata_files"]) != expected["metadata_bytes"]):
        raise PermissionError("A16 source inventory lengths exceed or differ from their finite budget")
    return plan


def reserve(root, run_id, config):
    common = Path(subprocess.run(["git", "-C", str(root), "rev-parse", "--path-format=absolute",
        "--git-common-dir"], check=True, capture_output=True, text=True).stdout.strip()).resolve(strict=True)
    directory = common / "a16_pm_impulse_budgets" / AUTH
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir(exist_ok=False)
    receipt = {"authorization": AUTH, "run_id": run_id, "source_commit": git_head(root),
        "reserved_at": datetime.now(UTC).isoformat(), "spent": True, "retry_allowed": False,
        "read_plan_sha256": config["read_plan_sha256"], "compute_plan_sha256": config["compute_plan_sha256"]}
    write_json(directory / "reservation.json", receipt)
    return directory, receipt


def verify_metadata(plan, documents):
    manifest = documents[plan["source_manifest_path"]]
    population = documents[f"results/{RUN}/allocated_population.json"]
    if population != plan["population"]:
        raise ValueError("A16 saved population differs")
    if documents["configs/proposals/kiki_a14_integral_history_reads.json"]["population"] != population:
        raise ValueError("A16 original population differs")
    receipt = documents[plan["inherited_result_receipt_path"]]
    if (receipt.get("run_id") != RUN or receipt.get("V_pass") is not True
            or receipt.get("authorization_status") != "consumed"
            or receipt.get("result_artifact_manifest_sha256") != plan["source_manifest_sha256"]):
        raise ValueError("A16 inherited A14 validity differs")
    for row in plan["npz_files"]:
        relative = row["path"].removeprefix(f"results/{RUN}/")
        if manifest.get(relative) != row["sha256"]:
            raise ValueError("A16 NPZ differs from the sealed A14 manifest")
        descriptor = row["json_path"].removeprefix(f"results/{RUN}/")
        frozen = next(r for r in plan["metadata_files"] if r["path"] == row["json_path"])
        if manifest.get(descriptor) != frozen["sha256"]:
            raise ValueError("A16 member metadata differs from sealed A14 manifest")
    prefix = f"results/{RUN}/"
    for row in plan["metadata_files"]:
        if (row["path"].startswith(prefix) and row["path"] != plan["source_manifest_path"]
                and manifest.get(row["path"].removeprefix(prefix)) != row["sha256"]):
            raise ValueError("A16 source metadata differs from its sealed manifest")


def verify_arrays(row, values, document):
    """Only selected member descriptors may be used; never old numerical results."""
    descriptors = {}

    def visit(value):
        if isinstance(value, dict):
            if {"array_member", "shape", "dtype", "sha256_c_order_bytes"} <= value.keys():
                name = value["array_member"]
                if name in values:
                    if name in descriptors and value != descriptors[name]:
                        raise ValueError("conflicting selected descriptors")
                    descriptors[name] = value
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(document)
    if set(descriptors) != set(values):
        raise ValueError("missing independent member descriptor")
    for name, value in values.items():
        descriptor = descriptors[name]
        if (list(value.shape) != descriptor["shape"] or value.dtype.str != descriptor["dtype"]
                or hashlib.sha256(value.tobytes(order="C")).hexdigest() != descriptor["sha256_c_order_bytes"]):
            raise ValueError(f"A16 logical member identity differs: {row['path']}::{name}")


class IdentityArithmetic:
    """Fixed small identity arithmetic, separately accounted from new science."""

    def __init__(self, limits=None):
        self.counts = Counter()
        self.limits = limits

    def charge(self, key):
        if self.limits is not None and self.counts[key] >= self.limits.get(key, 0):
            raise RuntimeError("A16 identity scalar budget exceeded")
        self.counts[key] += 1

    def sub(self, a, b):
        self.charge("sub")
        value = float(a) - float(b)
        self.charge("finite_check")
        if not math.isfinite(value):
            raise ValueError("nonfinite source identity")
        return value

    def close(self, actual, expected):
        value = self.sub(actual, expected)
        self.charge("abs")
        if abs(value) > 1e-12:
            raise ValueError("A16 source mass anchor differs")

    def clocks(self, raw, saved, slots):
        first = np.array([self.sub(raw[j], raw[10]) for j in range(11)])
        if first.tobytes() != saved.tobytes():
            raise ValueError("A16 original history clock differs")
        second = np.array([self.sub(saved[j], saved[0]) for j in range(11)])
        h = np.array([self.sub(second[j + 1], second[j]) for j in range(10)])
        for slot in slots:
            if second.tobytes() != slot["time"].tobytes() or h.tobytes() != slot["h"].tobytes():
                raise ValueError("A16 saved time/interval differs")
        return h


class ScientificScopes:
    """Durable full-scope reservations, strict scalar limits, immutable outputs."""

    def __init__(self, directory, compute):
        self.directory = Path(directory)
        self.compute = compute
        self.journal = AppendOnlyAuditJournal(self.directory / "scientific_calls.jsonl")
        self.rows = []
        self.seen = set()
        self.failed = False

    def invoke(self, name, kind, function, *, allow_numerical_failure=False, prerequisite=None):
        if self.failed:
            raise RuntimeError("A16 scientific scopes are poisoned after failure")
        try:
            return self._invoke(name, kind, function,
                allow_numerical_failure=allow_numerical_failure, prerequisite=prerequisite)
        except BaseException:
            self.failed = True
            raise

    def _invoke(self, name, kind, function, *, allow_numerical_failure=False, prerequisite=None):
        if name in self.seen:
            raise RuntimeError("A16 scientific scope already spent")
        self.seen.add(name)
        limits = self.compute["per_invocation_maxima"][kind]
        if limits != expected_budget(kind):
            raise RuntimeError("A16 static scalar budget differs")
        self.journal.append({"scope": name, "kind": kind, "status": "reserved",
                             "limits": limits, "unknown_if_interrupted": True, "retry_allowed": False})
        ledger = ScalarLedger(limits)
        numerical_failure = False
        try:
            result = function(ledger)
        except NumericalError as error:
            if not allow_numerical_failure:
                raise
            result = None
            numerical_failure = True
            value = {"status": "numerical_failure", "complete": False, "error": str(error),
                     "partial_values": error.partial_record, "counts": error.counts}
        else:
            value = result
        counts = persistable_counts(ledger.snapshot())
        if (not counts["closed"] or any(n > limits.get(key, 0) for key, n in counts["attempted"].items())
                or any(n > counts["attempted"].get(key, 0) for key, n in counts["returned"].items())):
            raise RuntimeError("A16 scalar scope did not close within its reservation")
        execution = "computed"
        exact = None
        if prerequisite is not None:
            previous = next((row for row in self.rows if row["scope"] == prerequisite), None)
            if (kind != "slot" or previous is None or previous["kind"] != "impulse"
                    or previous["execution_status"] != "numerical_failure"
                    or not isinstance(result, SlotResult) or result.complete
                    or result.status != "upstream_numerical_failure"
                    or counts["attempted"] or counts["returned"]):
                raise RuntimeError("A16 skipped scope lacks its failed prerequisite")
            execution = "not_attempted_prerequisite_failed"
        elif numerical_failure or (isinstance(result, SlotResult) and result.status == "numerical_failure"):
            execution = "numerical_failure"
        elif result is None:
            raise RuntimeError("A16 scope returned no result without a recorded failure")
        elif kind == "aggregate":
            exact = expected_budget(kind, completed_slots=result.values["completed_slots"])
        elif kind == "slot":
            if not isinstance(result, SlotResult) or not result.complete:
                raise RuntimeError("A16 slot omitted computation without a failed prerequisite")
            exact = expected_budget(kind, resolved=result.status != "numerically_unresolved")
        else:
            exact = expected_budget(kind)
        if exact is not None and (counts["attempted"] != exact or counts["returned"] != exact):
            save_record(self.directory, name + "-count-mismatch", {"result": value, "counts": counts, "expected": exact})
            raise RuntimeError("A16 successful scope counts differ from its exact static schedule")
        artifacts = save_record(self.directory, name, value)
        self.journal.append({"scope": name, "kind": kind, "status": "closed",
                             "counts": counts, "artifacts": artifacts,
                             "execution_status": execution, "prerequisite": prerequisite,
                             "scientific_status": getattr(result, "status", "complete") if result else "numerical_failure"})
        self.rows.append({"scope": name, "kind": kind, "counts": counts, "artifacts": artifacts,
                          "execution_status": execution, "prerequisite": prerequisite})
        return result


def _execute(root, result_dir, config, plan, compute, receipt):
    source_root = (root / config["source_root"]).resolve(strict=True)
    write_json(result_dir / "frozen_plan.json", {"reservation": receipt, "config": config,
        "read_plan": plan, "compute_plan": compute, "zero_compute": ZERO_COMPUTE,
        "claim_boundary": BOUNDARY})
    write_json(result_dir / "environment.json", {"python": sys.version, "executable": sys.executable,
        "versions": {name: version(name) for name in ("numpy", "scipy", "h5py", "torch", "filelock")}})
    reader = PmImpulseReader(root, source_root, plan, result_dir)
    scopes = ScientificScopes(result_dir, compute)
    identity = IdentityArithmetic(compute["identity_arithmetic"])
    try:
        # Consume the one-use campaign before these source metadata and byte reads.
        documents = reader.read_metadata()
        verify_metadata(plan, documents)
        reader.integrity("before")
        cached = {}
        for row in plan["npz_files"]:
            values = reader.read_npz(row)
            verify_arrays(row, values, documents[row["json_path"]])
            cached[row["path"].removeprefix(f"results/{RUN}/")] = values
        registry = documents[plan["registry_manifest_path"]]
        raw = {}
        priors = {}
        source_attributes = {}
        for source in SOURCES:
            attrs, static, events = reader.read_source(source, lambda s, v: root_attribute_validator(s, v, registry))
            source_attributes[source] = {name: float(v) if isinstance(v, (float, np.floating)) else
                                         v.decode("utf-8") if isinstance(v, bytes) else str(v)
                                         for name, v in attrs.items()}
            save_record(result_dir, f"source-input-{source}",
                {"attributes": source_attributes[source], "static": static, "events": events})
            raw.update(events)
            priors[source] = scopes.invoke(f"source-{source}", "source",
                lambda ledger, static=static, attrs=attrs: prepare_source(static["inertia"], attrs["omega_s"],
                    static["equilibrium/Pm"], ledger=ledger))
        inputs = cached["permitted_inputs.npz"]
        static = {seed: cached[f"static-{seed}.npz"] for seed in SEEDS}
        scopes.journal.append({"scope": "identities", "status": "reserved", "limits": compute["identity_arithmetic"],
                               "retry_allowed": False, "unknown_if_interrupted": True})
        for prior in priors.values():
            for j in range(10):
                identity.close(prior.mass[j].point, inputs["mass"][j])
        for seed in SEEDS[1:]:
            if static[seed]["P"].tobytes() != static[SEEDS[0]]["P"].tobytes():
                raise ValueError("A16 matrices differ: cannot share input impulses across seeds")
        for event in plan["events"]:
            i = event["cache_index"]
            slots = [cached[f"seed-{seed}-event-{i:03d}/integral.npz"] for seed in SEEDS]
            identity.clocks(raw[event["event_id"]]["t"], inputs["history_time"][i], slots)
        if dict(identity.counts) != compute["identity_arithmetic"]:
            raise RuntimeError("A16 exact identity arithmetic count differs")
        write_json(result_dir / "identity_checks.json", {"counts": dict(identity.counts),
            "clock_order_bitwise": True, "shared_P_and_h_bitwise": True, "source_mass_absolute_tolerance": 1e-12})
        scopes.journal.append({"scope": "identities", "status": "closed", "counts": dict(identity.counts),
                               "artifact_sha256": sha256_file(result_dir / "identity_checks.json")})
        impulses = []
        for event in plan["events"]:
            i, source = event["cache_index"], event["source_dataset"]
            fixed = cached[f"seed-{SEEDS[0]}-event-{i:03d}/integral.npz"]
            impulse = scopes.invoke(f"impulse-{i:03d}", "impulse",
                lambda ledger, event=event, i=i, source=source, fixed=fixed:
                    build_history_impulse(priors[source], raw[event["event_id"]]["Pm"],
                        inputs["constant_power"][i], fixed["h"], static[SEEDS[0]]["P"], ledger=ledger),
                allow_numerical_failure=True)
            impulses.append(impulse)
        save_record(result_dir, "source_history", {"events": raw, "attributes": source_attributes})
        summaries, statuses, slots_receipt = {}, {}, []
        for seed in SEEDS:
            records = []
            for event, impulse in zip(plan["events"], impulses, strict=True):
                i = event["cache_index"]
                fixed = cached[f"seed-{seed}-event-{i:03d}/integral.npz"]
                def call(ledger, fixed=fixed, impulse=impulse, seed=seed):
                    if impulse is None:
                        ledger.closed = True
                        return SlotResult("upstream_numerical_failure", False, {}, ledger.snapshot(), "shared impulse failed")
                    return audit_slot(impulse, **{k: fixed[k] for k in ("Ltilde", "c", "S", "g4", "btilde4",
                        "dynamic_residual_scaled4", "force_residual4")},
                        force_residual_projector=static[seed]["force_residual_projector"], ledger=ledger)
                result = scopes.invoke(f"slot-{seed}-{i:03d}", "slot", call,
                    prerequisite=f"impulse-{i:03d}" if impulse is None else None)
                records.append(result)
                slots_receipt.append({"seed": seed, "cache_index": i, "event_id": event["event_id"],
                    "status": result.status, "complete": result.complete})
            statuses[str(seed)] = dict(Counter(row.status for row in records))
            summaries[str(seed)] = scopes.invoke(f"aggregate-{seed}", "aggregate",
                lambda ledger, records=records: aggregate_slots(records, plan["population"]["weights"], ledger=ledger))
        reader.integrity("after")
        read_counts = reader.finish()
        scopes.journal.close()
        if Counter(row["kind"] for row in scopes.rows) != {"source": 2, "impulse": 258, "slot": 774, "aggregate": 3}:
            raise RuntimeError("A16 scope inventory is incomplete")
        write_json(result_dir / "access_ledger.json", read_counts)
        write_json(result_dir / "scientific_scopes.json", scopes.rows)
        write_json(result_dir / "allocated_slots.json", {"original_population": plan["population"], "diagnostic_slots": slots_receipt})
        save_record(result_dir, "weighted_summaries", summaries)
        summary = {"diagnostic_status": "COMPLETE", "V_pass": all(row.complete for row in summaries.values()),
            "statuses_by_seed": statuses, "allocated_train": 268, "eligible_train": 258, "seed_event_slots": 774,
            "source_commit": receipt["source_commit"], "authorization": AUTH, "promotion_decision": "KEEP_BASELINE",
            "zero_compute": ZERO_COMPUTE, "scientific_scopes": len(scopes.rows),
            "complete_population_metrics_available": all(row.complete for row in summaries.values()),
            "claim_boundary": BOUNDARY, "new_missed_instability_result_established": False,
            "internal_val_ids_read": [], "source_val_ids_read": [], "test_ids_read": [], "labels_read": 0}
        write_json(result_dir / "summary.json", summary)
        write_json(result_dir / "stage_evidence.json", {"status": "PASS", "process_status": "PASS",
            "source_commit": receipt["source_commit"], "diagnostic_status": "COMPLETE",
            "promotion_decision": "KEEP_BASELINE", "data_policy": "kiki_original_train_only",
            "development_authorization_id": AUTH, "evidence_kind": "uncertified_train_history_input_diagnostic",
            "certified_envelope_established": False, "deployment_predictions_allowed": False, "model_promoted": False,
            "level_6_claimed": False, "hidden_training_targets_read": False, "labels_read": 0,
            "internal_val_ids_read": [], "source_val_ids_read": [], "test_ids_read": [],
            "result_dir": result_dir.name, "claim_boundary": BOUNDARY})
        manifest = {p.relative_to(result_dir).as_posix(): sha256_file(p) for p in sorted(result_dir.rglob("*")) if p.is_file()}
        write_json(result_dir / "artifact_hashes.json", manifest)
        return summary
    except BaseException as error:
        if hasattr(error, "partial_record"):
            save_record(result_dir, "fatal_scientific_partial", {"error_type": type(error).__name__,
                "partial_values": error.partial_record, "counts": getattr(error, "counts", {})})
        write_json(result_dir / "fatal_prefix.json", {"error_type": type(error).__name__, "message": str(error),
            "source_reservation": receipt, "identity_counts_prefix": dict(identity.counts),
            "closed_scopes_prefix": scopes.rows, "read_prefix": reader.ledger.snapshot(), "retry_allowed": False})
        raise


def run(*, root, experiment, stage, result_dir, run_id):
    if stage != STAGE:
        raise PermissionError("A16 stage differs")
    config = tomllib.loads((root / "configs/experiments" / experiment["config"]).read_text(encoding="utf-8"))
    approval = authorization(root, experiment, config)
    plan = bound_metadata(root, config)
    compute = read_json(root / config["compute_plan_path"])
    budget, receipt = reserve(root, run_id, config)
    try:
        summary = _execute(root, result_dir, config, plan, compute, receipt)
        authorization(root, experiment, config)
        if git_head(root) != receipt["source_commit"]:
            raise RuntimeError("A16 source HEAD changed")
        write_json(budget / "completed.json", {"run_id": run_id, "authorization_spent": True,
            "diagnostic_status": summary["diagnostic_status"], "V_pass": summary["V_pass"],
            "scientific_review_sha256": approval["scientific_review_sha256"],
            "artifact_manifest_sha256": sha256_file(result_dir / "artifact_hashes.json")})
    except BaseException as error:
        write_json(budget / "failed.json", {"run_id": run_id, "authorization_spent": True,
            "error_type": type(error).__name__, "message": str(error), "retry_allowed": False})
        raise


def execute_registered_audit(root):
    root = Path(root).resolve(strict=True)
    experiment = load_manifest(root)["experiments"][EXPERIMENT]
    config = tomllib.loads((root / "configs/experiments" / experiment["config"]).read_text(encoding="utf-8"))
    authorization(root, experiment, config)
    bound_metadata(root, config)
    lock_dir = root / ".project_state/locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with FileLock(lock_dir / f"{WORKFLOW}.lock", timeout=0):
        return _run_experiment(root, EXPERIMENT, STAGE)


def main():
    parser = argparse.ArgumentParser(description="One-use A16 train-history power diagnostic")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    print(execute_registered_audit(parser.parse_args().root))


if __name__ == "__main__":
    main()
