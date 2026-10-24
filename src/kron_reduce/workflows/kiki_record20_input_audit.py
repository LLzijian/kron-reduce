"""Dedicated A12 train input audit; imports perform no source data access.

This entrypoint intentionally does not call generic dataset verification or
the existing history provider: both have a broader source read scope.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tomllib
from collections import Counter
from datetime import UTC, datetime
from importlib.metadata import version
from numbers import Real
from pathlib import Path

import h5py
import numpy as np
from filelock import FileLock

from kron_reduce.data.durable_audit_ledger import (
    AppendOnlyAuditJournal,
    DurableAccessLedger,
    DurableAuditJournalError,
)
from kron_reduce.data.limited_hdf_audit import (
    ALLOWED_SOURCES,
    EVENT_ATTRS,
    ROOT_ATTRS,
    LimitedHdfReader,
    _selector_key,
)
from kron_reduce.evaluation.record20_coordinates import inspect_record20_arrays
from kron_reduce.evaluation.record20_input import audit_record20_rule
from kron_reduce.evaluation.record20_metadata import audit_postlabel_metadata
from project_harness.evidence import sha256_file
from project_harness.git_state import assert_clean, git_head, verify_bound_file
from project_harness.manifest import load_manifest
from project_harness.policy import load_policy, validate_run_policy
from project_harness.runner import _run_experiment, _verify_development_text_bindings

AUTH = "kiki_a12_development_20261024"
EXPERIMENT = "paper_kiki_a12_record20_input"
WORKFLOW = "kiki_a12_record20_input_v1"
STAGE = "input_audit"
MODULE = "kron_reduce.workflows.kiki_record20_input_audit"
CONTRACT = "measured_record_history_20ms_audit_v1"
ZERO_COMPUTE = {name: 0 for name in (
    "model_constructions", "model_loads", "fit_attempts", "optimizer_updates",
    "prediction_paths", "rollout_calls", "integration_rhs", "physical_rhs", "model_force_calls",
)}


def write_json(path, value):
    """Exclusively create and durably save evidence; never overwrite a run."""
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_npz(path, values):
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **values)
        stream.flush()
        os.fsync(stream.fileno())


def array_fingerprint(value):
    """Hash the declared float64 logical array, independently of its ZIP."""
    array = np.asarray(value)
    return {"shape": list(array.shape), "dtype": array.dtype.str,
            "sha256_c_order_bytes": hashlib.sha256(array.tobytes(order="C")).hexdigest()}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8", errors="strict")
    raise ValueError("source string attribute has an invalid native type")


def _typed(value):
    """Preserve native attribute types and invalid values without JSON NaN."""
    if isinstance(value, (bytes, np.bytes_)):
        return {"native_type": type(value).__name__, "bytes_hex": bytes(value).hex()}
    if isinstance(value, np.ndarray):
        return {"native_type": "ndarray", "dtype": str(value.dtype), "shape": list(value.shape),
                "elements": [_typed(x) for x in value.flat]}
    if isinstance(value, np.generic):
        return {"native_type": type(value).__name__, "value": _typed(value.item())}
    if isinstance(value, float) and not np.isfinite(value):
        return {"native_type": "float", "nonfinite_repr": repr(value)}
    if value is None or isinstance(value, (bool, int, float, str)):
        return {"native_type": type(value).__name__, "value": value}
    return {"native_type": type(value).__name__, "repr": repr(value)}


def authorization(root, experiment, config):
    """Metadata/HEAD checks only; no source files may be opened here."""
    validate_run_policy(root, EXPERIMENT, experiment)
    _verify_development_text_bindings(root, experiment)
    approval = load_policy(root).get("development_evaluations", {}).get(AUTH, {})
    expected = {
        "status": "user_approved", "authority": "explicit_user_reply",
        "scientific_design_approver": "Astra Ultra", "experiment": EXPERIMENT,
        "workflow": WORKFLOW, "stage": STAGE, "module": MODULE, "dataset": "kiki_ieee39_v1",
        "allocated_train": 268, "max_hdf_opens": 4, "max_array_selections": 1078,
        "max_descriptors": 1078, "max_attribute_reads": 2966,
        "max_logical_array_elements": 7501168, "max_integrity_bytes": 6870747346,
        "max_fit_attempts": 0, "max_optimizer_updates": 0, "max_prediction_paths": 0,
        "max_total_rhs": 0, "labels_only_after_prelabel_freeze": True,
        "internal_val_payload_allowed": False, "source_val_payload_allowed": False,
        "source_test_payload_allowed": False, "model_promotion_allowed": False,
        "deployment_predictions_allowed": False,
    }
    if any(type(approval.get(k)) is not type(v) or approval.get(k) != v for k, v in expected.items()):
        raise PermissionError("A12 requires its active bounded input-only authorization")
    if any(k not in approval or approval[k] != v for k, v in config.items()):
        raise PermissionError("A12 config differs from its registered approval")
    if (experiment != load_manifest(root)["experiments"].get(EXPERIMENT)
            or experiment.get("execution_mode") != "bounded_input_audit_cli_only"
            or config["input_contract"] != CONTRACT):
        raise PermissionError("A12 experiment/entrypoint/input identity differs")
    for path_key, sha_key in (("read_plan_path", "read_plan_sha256"),
                              ("source_evidence_path", "source_evidence_sha256")):
        verify_bound_file(root, config[path_key], config[sha_key])
    verify_bound_file(root, approval["scientific_review_path"], approval["scientific_review_sha256"])
    bindings = approval.get("implementation_bindings", {})
    if not bindings or MODULE.replace(".", "/") + ".py" not in {
            p.removeprefix("src/") for p in bindings}:
        raise PermissionError("A12 missing approved implementation bindings")
    for relative, digest in bindings.items():
        verify_bound_file(root, relative, digest)
    assert_clean(root)
    return approval


def bound_metadata(root, config):
    """Rebuild the fixed train whitelist from already registered metadata."""
    for path_key, sha_key in (("allocation_path", "allocation_sha256"),
                              ("registry_manifest_path", "registry_manifest_sha256")):
        if sha256_file(root / config[path_key]) != config[sha_key]:
            raise PermissionError(f"A12 frozen metadata changed: {path_key}")
    plan = read_json(root / config["read_plan_path"])
    allocation = read_json(root / config["allocation_path"])
    registry = read_json(root / config["registry_manifest_path"])
    rows = sorted((r for r in allocation["rows"] if r["source_role"] == r["role"] == "train"),
                  key=lambda r: (r["source_dataset"], r["raw_id"]))
    fields = ("event_id", "raw_id", "source_dataset", "fault_line", "role", "source_role", "declared_length")
    if len(rows) != 268 or [{k: r[k] for k in fields} for r in rows] != [
            {k: r[k] for k in fields} for r in plan["slots"]]:
        raise PermissionError("A12 allocation_mismatch")
    if (sum(r["declared_length"] for r in rows) != 357068
            or sum(r["declared_length"] >= 12 for r in rows) != 258):
        raise PermissionError("A12 fixed original population changed")
    for source in sorted(ALLOWED_SOURCES):
        digest = plan["source_inventory"][source]["dataset_h5_registered_sha256"]
        if (digest != allocation["source_bindings"][source]["dataset.h5"]
                or digest != registry["shards"][source]["sha256"]["dataset.h5"]):
            raise PermissionError("A12 source_binding_mismatch")
    if (plan["allocation_sha256"] != config["allocation_sha256"]
            or plan["registry_manifest_sha256"] != config["registry_manifest_sha256"]):
        raise PermissionError("A12 plan metadata bindings differ")
    return plan, registry


def budget_directory(root):
    result = subprocess.run(["git", "-C", str(root), "rev-parse", "--path-format=absolute",
                             "--git-common-dir"], check=True, capture_output=True, text=True)
    common = Path(result.stdout.strip()).resolve(strict=True)
    return common / "a12_input_budgets" / AUTH


def reserve_budget(root, run_id, config):
    directory = budget_directory(root)
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir(exist_ok=False)
    receipt = {"authorization": AUTH, "run_id": run_id, "source_commit": git_head(root),
               "reserved_at": datetime.now(UTC).isoformat(), "automatic_retry": False,
               "read_plan_sha256": config["read_plan_sha256"], "spent": True}
    write_json(directory / "reservation.json", receipt)
    return directory, receipt


def bounded_integrity_hash(path, expected_size, expected_sha, journal, phase):
    """Exactly one bounded byte pass; no HDF decoding or extra EOF read."""
    journal.append({"kind": "integrity_hash", "phase": phase, "path": str(path),
                    "status": "attempted", "planned_bytes": expected_size})
    digest, total = hashlib.sha256(), 0
    try:
        with Path(path).open("rb") as stream:
            before = os.fstat(stream.fileno())
            if before.st_size != expected_size:
                raise ValueError("source_binding_mismatch: size differs")
            while total < expected_size:
                block = stream.read(min(1024 * 1024, expected_size - total))
                if not block:
                    raise ValueError("source_binding_mismatch: unexpected EOF")
                total += len(block)
                digest.update(block)
            after = os.fstat(stream.fileno())
            if (before.st_size, before.st_mtime_ns, before.st_ino) != (
                    after.st_size, after.st_mtime_ns, after.st_ino):
                raise ValueError("source_binding_mismatch: file changed during hash")
        if digest.hexdigest() != expected_sha:
            raise ValueError("source_binding_mismatch: SHA256 differs")
    except BaseException as error:
        journal.append({"kind": "integrity_hash", "phase": phase, "path": str(path),
                        "status": "failed", "actual_bytes": total, "error_type": type(error).__name__})
        raise
    receipt = {"kind": "integrity_hash", "phase": phase, "path": str(path),
               "status": "returned", "actual_bytes": total, "sha256": digest.hexdigest()}
    journal.append(receipt)
    return receipt


def journal_metadata(ledger_plan, read_plan):
    lengths = {row["event_id"]: row["declared_length"] for row in read_plan["slots"]}
    result = {}
    for item in ledger_plan:
        parts = item["key"].split(":")
        category, source, name = parts[0], parts[1], parts[-1]
        event_id = parts[2] if len(parts) == 4 else None
        if category == "descriptor":
            selection = "shape,dtype"
        elif category in {"event-attr", "root-attr"}:
            selection = "single_attribute"
        elif category == "static-array":
            selection = "[:]"
        elif category == "event-array":
            n = lengths[event_id]
            selection = "[2,0:10]" if name == "Pm" else f"[0:{n}]" if name == "t" else f"[0:{n},0:10]"
        else:
            raise PermissionError("unregistered journal selector category")
        dataset_path = f"events/{event_id.split('/', 1)[1]}/{name}" if event_id else name
        result[item["key"]] = {"source": source, "event_id": event_id,
                               "role": "train" if event_id else "static_prior",
                               "dataset_or_attr": dataset_path, "selection": selection}
    return result


def root_attribute_validator(source, values, registry):
    expected = registry["shards"][source]["attributes"]
    if set(values) != set(ROOT_ATTRS):
        raise ValueError("source_binding_mismatch: missing root attributes")
    for key in ROOT_ATTRS:
        if key in {"omega_s", "mva_base"}:
            native = values[key]
            if (isinstance(native, (bool, np.bool_)) or not isinstance(native, Real)
                    or not np.isfinite(native) or float(native) != float(expected[key])):
                raise ValueError(f"source_static_prior_invalid: {key}")
        elif _text(values[key]) != expected[key]:
            raise ValueError(f"source_binding_mismatch: {key}")
    if float(values["mva_base"]) != 100 or abs(float(values["omega_s"]) - 120 * np.pi) > 1e-12:
        raise ValueError("source_static_prior_invalid: units")
    mapping = json.loads(_text(values["state_mapping"]))
    expected_first = [[i, "delta" if i < 10 else "omega", "GENROU", f"GENROU_{i % 10 + 1}"]
                      for i in range(20)]
    if mapping[:20] != expected_first:
        raise ValueError("source_static_prior_invalid: generator order")
    json.loads(_text(values["config"]))


def static_array_validator(source, values, static):
    keys = ("inertia[:]", "equilibrium/delta[:]", "equilibrium/Pm[:]")
    if any(k not in values or np.asarray(values[k]).shape != (10,)
           or not np.isfinite(values[k]).all() for k in keys) or np.any(values["inertia[:]"] <= 0):
        raise ValueError("source_static_prior_invalid: static arrays")
    mass = values["inertia[:]"] / values["omega_s"]
    if any(not np.allclose(mass, entry["mass"], atol=1e-12, rtol=0) for entry in static.values()):
        raise ValueError("source_static_prior_invalid: shared mass differs")
    static[source] = {"inertia": values[keys[0]], "equilibrium_delta": values[keys[1]],
                      "equilibrium_pm": values[keys[2]], "mass": mass,
                      "omega_s": float(values["omega_s"]),
                      "postfault_signature": _text(values["postfault_signature"])}


def _summarize_prelabel(slots):
    groups = {}
    for name, rows in (("all_268", slots), ("old100_common_258", [s for s in slots if s["old100_common"]]),
                       ("old100_unavailable_10", [s for s in slots if not s["old100_common"]])):
        groups[name] = {"allocated": len(rows), "input_supported": sum(s["input"].get("record20_input_supported", False) for s in rows),
                        "full_visible_record_valid": sum(s["input"].get("full_visible_record_valid", False) for s in rows),
                        "rule_valid": sum(s["rule"].get("status") == "valid" for s in rows),
                        "physical_lines": sorted({s["fault_line"] for s in rows}),
                        "origin20_continuous_violation_counts": dict(Counter(
                            str((s["rule"].get("origins", {}).get("2") or {}).get(
                                "continuous_joint_violations_ending_at_origin")) for s in rows))}
    return groups


def _execute(root, result_dir, config, plan, registry, receipt):
    source_root = Path(os.environ.get("KIKI_DATA_ROOT", config["source_root"]))
    source_root = (source_root if source_root.is_absolute() else root / source_root).resolve(strict=True)
    paths = {s: source_root / s / "dataset.h5" for s in sorted(ALLOWED_SOURCES)}
    write_json(result_dir / "frozen_plan.json", {"reservation": receipt, "config": config,
               "plan": plan, "source_paths": {s: str(p) for s, p in paths.items()},
               "compute": ZERO_COMPUTE, "source_val_ids_read": [], "internal_val_ids_read": [], "test_ids_read": []})
    write_json(result_dir / "environment.json", {"python_executable": sys.executable,
               "python_version": sys.version, "versions": {name: version(name) for name in
                   ("numpy", "scipy", "h5py", "torch", "filelock")}})
    io_journal = AppendOnlyAuditJournal(result_dir / "io_access.jsonl")
    reader, static, integrity, hdf_attempts = None, {}, [], []
    try:
        for source in sorted(ALLOWED_SOURCES):
            entry = plan["source_inventory"][source]
            integrity.append(bounded_integrity_hash(paths[source], entry["metadata_stat_only"]["size_bytes"],
                entry["dataset_h5_registered_sha256"], io_journal, "before"))

        def ledger_factory(ledger_plan, **kwargs):
            return DurableAccessLedger(ledger_plan, **kwargs, metadata_by_key=journal_metadata(ledger_plan, plan),
                                       journal_path=result_dir / "phase1_access.jsonl")

        def open_hdf(path, mode):
            if mode != "r" or Path(path) not in paths.values() or len(hdf_attempts) >= 4:
                raise PermissionError("budget_or_access_order_violation: HDF open")
            hdf_attempts.append(str(path))
            def log_open(status):
                try:
                    io_journal.append({"kind": "hdf_open", "path": str(path), "mode": mode,
                                       "attempt": len(hdf_attempts), "status": status})
                except (OSError, UnicodeError, ValueError, TypeError) as error:
                    raise DurableAuditJournalError("HDF-open journal failed") from error
            log_open("attempted")
            result = h5py.File(path, mode)
            try:
                log_open("returned")
            except DurableAuditJournalError:
                result.close()
                raise
            return result

        reader = LimitedHdfReader(plan, paths, file_opener=open_hdf, ledger_factory=ledger_factory,
            root_validator=lambda s, v: root_attribute_validator(s, v, registry),
            static_validator=lambda s, v: static_array_validator(s, v, static))
        values = {}
        for source in sorted(ALLOWED_SOURCES):
            values.update(reader.read_phase1_source(source))
        slots, raw_arrays, input_hashes = [], {}, {}
        for row in plan["slots"]:
            slot = {k: row[k] for k in ("slot", "event_id", "source_dataset", "raw_id", "fault_line", "declared_length")}
            slot.update({"old100_common": row["declared_length"] >= 12,
                         "input": {"record20_input_supported": False, "reasons": ["missing_event_or_dataset"]},
                         "rule": {"status": "invalid", "invalid_reason": "missing_event_or_dataset"}})
            selected = {}
            for name in ("t", "delta", "omega", "Pm"):
                key = _selector_key("event-array", row["source_dataset"], name, row["event_id"])
                if key in values:
                    selected[name] = values[key]
                    raw_arrays[f"s{row['slot']:04d}_{name}"] = values[key]
            if {"t", "delta", "omega"} <= set(selected) and row["source_dataset"] in static:
                priors = static[row["source_dataset"]]
                slot["input"], inputs, transformed = inspect_record20_arrays(
                    t=selected["t"], delta=selected["delta"], omega=selected["omega"],
                    origin_pm=selected.get("Pm"), **{k: priors[k] for k in
                        ("inertia", "equilibrium_delta", "equilibrium_pm", "omega_s")})
                for name, arr in transformed.items():
                    raw_arrays[f"s{row['slot']:04d}_transformed_{name}"] = arr
                if inputs is not None:
                    inputs_dir = result_dir / "inputs"
                    inputs_dir.mkdir(exist_ok=True)
                    write_npz(inputs_dir / f"slot-{row['slot']:04d}.npz", inputs)
                    input_hashes[str(row["slot"])] = {name: array_fingerprint(value)
                                                     for name, value in inputs.items()}
            if {"t", "delta", "omega"} <= set(selected) and row["source_dataset"] in static:
                slot["rule"] = audit_record20_rule(selected["t"], selected["delta"], selected["omega"],
                                                   static[row["source_dataset"]]["inertia"])
            slots.append(slot)
        for source, priors in static.items():
            for name, value in priors.items():
                if isinstance(value, np.ndarray):
                    raw_arrays[f"static_{source}_{name}"] = value
        write_npz(result_dir / "allowed_arrays.npz", raw_arrays)
        write_json(result_dir / "array_fingerprints.json", {
            "allowed_raw_and_transformed": {k: array_fingerprint(v) for k, v in raw_arrays.items()},
            "input_members_by_slot": input_hashes})
        write_json(result_dir / "prelabel_slots.json", slots)
        write_json(result_dir / "prelabel_summary.json", _summarize_prelabel(slots))
        write_json(result_dir / "native_static_metadata.json", {k: _typed(v) for k, v in values.items()
                   if k.startswith(("root-attr:", "static-array:"))})
        reader.ledger.close_phase1_journal()
        write_json(result_dir / "prelabel_ledger.json", reader.audit_log())
        # io_access includes no labels but stays open for final integrity checks;
        # its phase-1 prefix is separately snapshotted and sealed.
        io_prefix = (result_dir / "io_access.jsonl").read_bytes()
        with (result_dir / "prelabel_io_access.jsonl").open("xb") as stream:
            stream.write(io_prefix)
            stream.flush()
            os.fsync(stream.fileno())
        frozen_files = {p.relative_to(result_dir).as_posix(): sha256_file(p)
                        for p in sorted(result_dir.rglob("*")) if p.is_file() and p.name != "io_access.jsonl"}
        freeze = {"files": frozen_files, "labels_decoded": 0, "slots": 268,
                  "compute": ZERO_COMPUTE, "internal_val_ids_read": [], "source_val_ids_read": [], "test_ids_read": []}
        write_json(result_dir / "prelabel_freeze.json", freeze)
        freeze_sha = sha256_file(result_dir / "prelabel_freeze.json")
        reader.ledger.rotate_to_phase2_journal(result_dir / "phase2_access.jsonl")
        reader.seal_prelabel_freeze(freeze_sha)
        metadata = {}
        for source in sorted(ALLOWED_SOURCES):
            metadata.update(reader.read_phase2_source(source))
        postlabel, preserved = [], {}
        for row, slot in zip(plan["slots"], slots, strict=True):
            attrs = {}
            for name in EVENT_ATTRS:
                key = _selector_key("event-attr", row["source_dataset"], name, row["event_id"])
                if key in metadata:
                    attrs[name] = metadata[key]
                    preserved[key] = _typed(metadata[key])
            t = values.get(_selector_key("event-array", row["source_dataset"], "t", row["event_id"]), np.empty(0))
            checked = audit_postlabel_metadata(attrs, row, t, slot["rule"],
                       static.get(row["source_dataset"], {}).get("postfault_signature", ""))
            postlabel.append({"slot": row["slot"], "event_id": row["event_id"], "metadata": checked})
        write_json(result_dir / "native_event_metadata.json", preserved)
        write_json(result_dir / "postlabel_slots.json", postlabel)
        reader.ledger.close_journal()
        for source in sorted(ALLOWED_SOURCES):
            entry = plan["source_inventory"][source]
            integrity.append(bounded_integrity_hash(paths[source], entry["metadata_stat_only"]["size_bytes"],
                entry["dataset_h5_registered_sha256"], io_journal, "after"))
        io_journal.close()
        if sha256_file(result_dir / "prelabel_freeze.json") != freeze_sha or any(
                sha256_file(result_dir / p) != digest for p, digest in frozen_files.items()):
            raise RuntimeError("artifact_freeze_failure: prelabel evidence changed")
        access = reader.audit_log()
        write_json(result_dir / "access_ledger.json", access)
        write_json(result_dir / "integrity_receipts.json", integrity)
        planned = access["planned_counts"]
        counts = access["actual_counts"]
        budget_ok = (not access["poisoned"] and not access["denied_attempts"] and not access["violations"]
                     and all(counts[k]["returned"] == planned[k] and counts[k]["failed"] == 0 for k in planned)
                     and access["actual_array_elements"]["returned"] == 7501168
                     and sum(r["actual_bytes"] for r in integrity) == 6870747346
                     and len(hdf_attempts) == 4)
        metadata_ok = all(s["metadata"]["consistent"] for s in postlabel)
        supported = (budget_ok and metadata_ok and len(static) == 2
                     and all(s["input"].get("record20_input_supported", False)
                             and s["input"].get("full_visible_record_valid", False)
                             and s["rule"].get("status") == "valid" for s in slots))
        summary = {"execution_completed": True, "scientific_verdict": "ACCEPT" if supported else "REJECT",
                   "record20_train_contract_supported": supported, "budget_checks_pass": budget_ok,
                   "metadata_consistent_count": sum(s["metadata"]["consistent"] for s in postlabel),
                   "allocated": 268, "prelabel_groups": _summarize_prelabel(slots),
                   "prelabel_freeze_sha256": freeze_sha, "source_commit": receipt["source_commit"],
                   "compute": ZERO_COMPUTE, "new_training_authorized": False, "model_selection_performed": False,
                   "model_promotion_allowed": False, "internal_val_coverage_verified": False,
                   "zero_missed_instability_established": False, "deployment_predictions_allowed": False,
                   "automatic_retry": False, "claim_boundary": config["claim_boundary"]}
        write_json(result_dir / "summary.json", summary)
        evidence = {"status": "PASS", "process_status": "PASS", "source_commit": receipt["source_commit"],
                    "scientific_verdict": summary["scientific_verdict"], "promotion_decision": "KEEP_BASELINE",
                    "data_policy": "kiki_original_train_only", "source_val_ids_read": [], "internal_val_ids_read": [],
                    "test_ids_read": [], "hidden_training_targets_read": False, "level_6_claimed": False,
                    "development_authorization_id": AUTH, "evidence_kind": "uncertified_train_input_audit",
                    "certified_envelope_established": False, "deployment_predictions_allowed": False,
                    "model_promoted": False, "claim_boundary": config["claim_boundary"]}
        write_json(result_dir / "stage_evidence.json", evidence)
        write_json(result_dir / "artifact_hashes.json", {p.relative_to(result_dir).as_posix(): sha256_file(p)
                   for p in sorted(result_dir.rglob("*")) if p.is_file()})
        return summary
    except BaseException as error:
        # No resume or retry: preserve partial access and every unattempted slot.
        try:
            if reader is not None:
                reader.ledger.close_journal()
                write_json(result_dir / "failed_access_ledger.json", reader.audit_log())
            io_journal.close()
            write_json(result_dir / "input_audit_failure.json", {
                "error_type": type(error).__name__, "message": str(error), "automatic_retry": False,
                "execution_completed": False, "allocated": 268, "compute": ZERO_COMPUTE,
                "hdf_open_attempts": hdf_attempts})
        except BaseException as preservation_error:  # noqa: BLE001 - retain the original fatal cause.
            error.add_note(f"Failure evidence could not be fully saved: {preservation_error!r}")
        raise


def run(*, root, experiment, stage, result_dir, run_id):
    if stage != STAGE:
        raise PermissionError("A12 is restricted to its input audit stage")
    config = tomllib.loads((root / "configs/experiments" / experiment["config"]).read_text(encoding="utf-8"))
    approval = authorization(root, experiment, config)
    plan, registry = bound_metadata(root, config)
    budget_dir, receipt = reserve_budget(root, run_id, config)
    try:
        summary = _execute(root, result_dir, config, plan, registry, receipt)
        write_json(budget_dir / "completed.json", {"run_id": run_id, "scientific_verdict": summary["scientific_verdict"],
                   "artifact_manifest_sha256": sha256_file(result_dir / "artifact_hashes.json"),
                   "authorization_spent": True, "scientific_review_sha256": approval["scientific_review_sha256"]})
    except BaseException as error:
        write_json(budget_dir / "failed.json", {"run_id": run_id, "authorization_spent": True,
                   "automatic_retry": False, "error_type": type(error).__name__, "message": str(error)})
        raise


def execute_registered_audit(root):
    """Narrow outer wrapper: validate bindings, lock, then immutable inner run."""
    root = Path(root).resolve(strict=True)
    experiment = load_manifest(root)["experiments"][EXPERIMENT]
    config = tomllib.loads((root / "configs/experiments" / experiment["config"]).read_text(encoding="utf-8"))
    authorization(root, experiment, config)
    bound_metadata(root, config)
    lock_dir = root / ".project_state/locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with FileLock(lock_dir / f"{WORKFLOW}.lock", timeout=0):
        # The inner workflow repeats authorization checks before reserving its
        # one-use shared budget. No generic verify_data is reachable here.
        return _run_experiment(root, EXPERIMENT, STAGE)


def main():
    parser = argparse.ArgumentParser(description="Execute the one-use registered A12 input audit")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(execute_registered_audit(args.root))


if __name__ == "__main__":
    main()
