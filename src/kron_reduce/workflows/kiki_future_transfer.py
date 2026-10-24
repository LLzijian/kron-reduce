"""One-use A15: fixed A14 resets transferred to train future trajectories."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tomllib
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path

import numpy as np
from filelock import FileLock

from kron_reduce.data.durable_audit_ledger import AppendOnlyAuditJournal
from kron_reduce.data.future_transfer_reader import FutureTransferReader, fixed_file_specs
from kron_reduce.evaluation.audit_budget import AuditRhsBudget
from kron_reduce.evaluation.future_transfer_audit import (
    SEEDS,
    STEPS,
    compare_rule_counts,
    compare_rule_pair,
    preflight_prerequisite,
    preflight_truth,
    refinement,
    rule_audit,
)
from kron_reduce.evaluation.future_transfer_metrics import compare_future
from kron_reduce.evaluation.scoped_audit_journal import ScopedAuditJournal
from kron_reduce.models.nonlinear_latent import NonlinearPassiveClosure
from kron_reduce.workflows.kiki_integral_history_audit import save_record
from kron_reduce.workflows.kiki_record20_input_audit import (
    array_fingerprint,
    bounded_integrity_hash,
    write_json,
)
from project_harness.evidence import sha256_file
from project_harness.git_state import assert_clean, git_head, verify_bound_file
from project_harness.manifest import load_manifest
from project_harness.policy import load_policy, validate_run_policy
from project_harness.runner import _run_experiment, _verify_development_text_bindings

AUTH = "kiki_a15_development_20261024"
EXPERIMENT = "paper_kiki_a15_future_transfer"
WORKFLOW = "kiki_a15_future_transfer_v1"
STAGE = "future_transfer"
MODULE = "kron_reduce.workflows.kiki_future_transfer"
MODE = "bounded_future_transfer_cli_only"
INTEGRATION_RHS, PHYSICAL_RHS, TOTAL_RHS = 5194584, 434430, 5629014
RUNTIME = {"numpy": "2.5.2", "scipy": "1.18.0", "torch": "2.13.0+cpu",
           "h5py": "3.16.0", "filelock": "3.32.2"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def authorization(root, experiment, config):
    validate_run_policy(root, EXPERIMENT, experiment)
    _verify_development_text_bindings(root, experiment)
    approval = load_policy(root).get("development_evaluations", {}).get(AUTH, {})
    expected = {"status": "user_approved", "authority": "explicit_user_reply",
        "scientific_design_approver": "Astra Ultra", "experiment": EXPERIMENT,
        "workflow": WORKFLOW, "stage": STAGE, "module": MODULE, "dataset": "kiki_ieee39_v1",
        "max_campaign_executions": 1, "retry_allowed": False}
    if any(type(approval.get(k)) is not type(v) or approval.get(k) != v for k, v in expected.items()):
        raise PermissionError("A15 lacks its independent active one-use authorization")
    if any(k not in approval or approval[k] != v for k, v in config.items()):
        raise PermissionError("A15 configuration differs from its approved design")
    if (experiment != load_manifest(root)["experiments"].get(EXPERIMENT)
            or experiment.get("execution_mode") != MODE
            or config["input_contract"] != "measured_history_100ms_zoh_v1"):
        raise PermissionError("A15 execution identity mismatch")
    bindings = approval.get("implementation_bindings", {})
    if "src/" + MODULE.replace(".", "/") + ".py" not in bindings:
        raise PermissionError("A15 lacks reviewed implementation bindings")
    for relative, digest in bindings.items():
        verify_bound_file(root, relative, digest)
    for key in ("read_plan", "inventory"):
        verify_bound_file(root, config[f"{key}_path"], config[f"{key}_sha256"])
    verify_bound_file(root, approval["scientific_review_path"], approval["scientific_review_sha256"])
    assert_clean(root)
    return approval


def bound_metadata(root, config):
    plan = read_json(root / config["read_plan_path"])
    inventory = read_json(root / config["inventory_path"])
    indexed = {f'{item["source"]}/{item["relative_path"]}': item for item in inventory["npz_files"]}
    if (set(indexed) != {spec.key for spec in fixed_file_specs()}
            or len(inventory["npz_files"]) != 1559
            or sum(item["size_bytes_stat_only"] for item in indexed.values()) != 99537517):
        raise PermissionError("A15 registered file inventory differs")
    if (plan["metadata_files"] != inventory["metadata_files"] or plan["read_budget"] != inventory["totals"]
            or len(plan["npz_files"]) != 1559):
        raise PermissionError("A15 read plan and source inventory disagree")
    for file, spec in zip(plan["npz_files"], fixed_file_specs(), strict=True):
        if (file["file_id"] != spec.key or file["max_decode_context_opens"] != 1
                or file["max_integrity_hash_passes"] != 2):
            raise PermissionError("A15 file order or capacity differs from its fixed reader")
        members = file["allowed_members"]
        if len(members) != len(spec.members):
            raise PermissionError("A15 member count differs from fixed reader")
        for item, member in zip(members, spec.members, strict=True):
            if (item["name"] != member.name or item["shape"] != list(member.shape)
                    or np.dtype(item["dtype"]) != np.dtype(member.dtype)
                    or item["scalar_count"] != member.elements
                    or item["max_descriptor_reads"] != 1 or item["max_array_reads"] != 1):
                raise PermissionError("A15 explicit member plan differs from its fixed reader")
    return plan, inventory, indexed


def reserve_budget(root, run_id, config):
    common = Path(subprocess.run(["git", "-C", str(root), "rev-parse", "--path-format=absolute",
        "--git-common-dir"], check=True, capture_output=True, text=True).stdout.strip()).resolve(strict=True)
    directory = common / "a15_future_transfer_budgets" / AUTH
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir(exist_ok=False)
    receipt = {"authorization": AUTH, "run_id": run_id, "source_commit": git_head(root),
        "spent": True, "automatic_retry": False, "read_plan_sha256": config["read_plan_sha256"]}
    write_json(directory / "reservation.json", receipt)
    return directory, receipt


@contextmanager
def counted_structure(journal, counters):
    """Observe the original validator's eigvalsh calls without changing results."""
    original = np.linalg.eigvalsh

    def counted(*args, **kwargs):
        index = counters["eigvalsh_attempted"]
        if index >= 4653:
            raise RuntimeError("A15 eigvalsh budget exhausted before operator invocation")
        entry = {"category": "eigvalsh", "index": index}
        journal.append({**entry, "transition": "attempted"})
        counters["eigvalsh_attempted"] += 1
        try:
            result = original(*args, **kwargs)
        except BaseException as error:
            journal.append({**entry, "transition": "failed", "error_type": type(error).__name__})
            raise
        counters["eigvalsh_returned"] += 1
        journal.append({**entry, "transition": "returned"})
        if not np.isfinite(result).all():
            error = FloatingPointError("nonfinite structural eigvalsh return")
            error.partial_record = {"nonfinite_eigvalsh_return": result}
            raise error
        return result

    np.linalg.eigvalsh = counted
    try:
        yield
    finally:
        np.linalg.eigvalsh = original


def audited_future(closure, reset, clock, power, eq, step, budget, journal, identity, path_sink):
    """Call the unchanged scalar rollout; save a completed path before physics."""
    original = closure.rhs
    last = {}

    def counted(*args, **kwargs):
        def dispatch():
            entry = {"category": "rhs", "phase": budget._active["phase"],
                     "sequence": budget._active["attempted"], **identity}
            journal.append({**entry, "transition": "attempted"})
            last["rhs_input"] = args[0]
            try:
                result = original(*args, **kwargs)
            except BaseException as error:
                journal.append({**entry, "transition": "failed", "error_type": type(error).__name__})
                raise
            last["rhs_return"] = result
            journal.append({**entry, "transition": "returned"})
            return result
        return budget.invoke(dispatch)

    updates = int(sum(max(1, int(np.ceil(dt / step - 1e-12))) for dt in np.diff(clock)))
    object.__setattr__(closure, "rhs", counted)
    physical = []
    try:
        with budget.segment(identity, "future_integration", 4 * updates):
            budget.note_rollout_start()
            journal.append({"category": "rollout", "transition": "attempted", **identity})
            try:
                path = closure.rollout(reset[:10], reset[10:20], reset[20:], clock, power, eq, max_step=step)
            except BaseException as error:
                journal.append({"category": "rollout", "transition": "failed", **identity,
                                "error_type": type(error).__name__})
                raise
            journal.append({"category": "rollout", "transition": "returned", **identity})
        path_sink(path)
        if not np.isfinite(path).all() or not np.array_equal(path[0], reset):
            raise FloatingPointError("candidate path is nonfinite or changed its fixed initial state")
        with budget.segment(identity, "physical_checks", len(clock)):
            for state in path:
                check = closure.energy_identity(state, power, eq)
                check.update(q_mass_coi=abs(float(closure.mass @ state[:10])),
                             v_mass_coi=abs(float(closure.mass @ state[10:20])))
                physical.append(check)
                if not all(np.isfinite(value) for value in check.values()):
                    raise FloatingPointError("nonfinite physical identity diagnostic")
        qualified = all(row["identity_error"] <= 1e-8 and row["q_mass_coi"] <= 1e-10
                        and row["v_mass_coi"] <= 1e-10 for row in physical)
        return path, {"points": physical, "qualified": qualified}
    except (ArithmeticError, np.linalg.LinAlgError) as error:
        error.partial_record = {"last_rhs": last, "physical_points": physical,
                                "inner_partial": getattr(error, "partial_record", None)}
        raise
    finally:
        object.__delattr__(closure, "rhs")


def stage_evidence(receipt, summary, boundary):
    return {"status": "PASS", "process_status": "PASS", "source_commit": receipt["source_commit"],
        "scientific_verdict": summary["scientific_verdict"], "promotion_decision": "KEEP_BASELINE",
        "data_policy": "kiki_original_train_only", "source_val_ids_read": [], "internal_val_ids_read": [],
        "test_ids_read": [], "hidden_training_targets_read": False, "level_6_claimed": False,
        "development_authorization_id": AUTH, "evidence_kind": "uncertified_train_future_transfer",
        "certified_envelope_established": False, "deployment_predictions_allowed": False,
        "model_promoted": False, "claim_boundary": boundary}


def save_future_comparison(directory, key, *arguments):
    """Persist each group before moving to another; numerical failures are local."""
    valid = True
    try:
        result = compare_future(*arguments)
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as error:
        result = {"I_pass": False, "status": "numerical_failure",
            "error_type": type(error).__name__, "error": str(error),
            "partial": getattr(error, "partial_record", None)}
        valid = False
    save_record(directory, f"comparison-{key}", result)
    return result, valid


def _execute(root, result_dir, config, plan, inventory, indexed, receipt):
    # Metadata layout is bound and checked before any source array or model call.
    io = AppendOnlyAuditJournal(result_dir / "integrity_and_metadata.jsonl")
    scientific = ScopedAuditJournal(result_dir / "scientific_journal")
    reader = None
    budget = AuditRhsBudget(INTEGRATION_RHS, PHYSICAL_RHS, TOTAL_RHS, 1548)
    counters = {"model_constructions": 0, "eigvalsh_attempted": 0, "eigvalsh_returned": 0}
    population = plan["population"]
    slots = [{**row, "status": "not_attempted"} for row in population["seed_event_slots"]]
    paths = {key: root / "results" / item["run_id"] / item["relative_path"] for key, item in indexed.items()}
    integrity, comparisons, rule_comparisons = [], {}, {}
    candidate_paths, diagnoses = {}, {}
    try:
        runtime = {key: version(key) for key in RUNTIME}
        if runtime != RUNTIME:
            raise RuntimeError("A15 requires the pinned A06/A10 CPU runtime")
        write_json(result_dir / "environment.json", {"python_executable": sys.executable,
                   "python_version": sys.version, "versions": runtime})
        write_json(result_dir / "allocated_population.json", population)
        metadata = read_source_metadata(root, inventory, io)
        verify_metadata_semantics(metadata, plan, inventory)
        def before_open(key, path):
            item = indexed[key]
            integrity.append(bounded_integrity_hash(path, item["size_bytes_stat_only"],
                item["sha256_from_sealed_manifest"], io, "before"))

        reader = FutureTransferReader(paths, journal_path=result_dir / "source_member_access.jsonl",
                                      before_open=before_open)
        decoded = reader.read_prerequisite()
        pre = preflight_prerequisite(decoded, population)
        expected_counts = np.asarray(population["future_point_counts"], dtype=np.int64)
        declared_counts = np.asarray([min(300, r["declared_length"] - 11) for r in population["cache_rows"]])
        if not np.array_equal(pre["counts"], expected_counts) or not np.array_equal(pre["counts"], declared_counts):
            raise ValueError("A15 future counts differ from frozen metadata and declared lengths")
        fingerprints = verify_array_bindings(decoded, plan)
        frozen = save_record(result_dir, "frozen_inputs_and_resets", {
            "inputs": pre["inputs"], "parameters": {str(s): pre["parameters"][s] for s in SEEDS},
            "resets": {f"{s}-{h}": pre["resets"][s, h] for s in SEEDS for h in STEPS},
            "offsets": pre["offsets"], "counts": pre["counts"], "population": population})
        save_record(result_dir, "saved_controls", {f"{s}-{h}": pre["controls"][s, h] for s in SEEDS for h in STEPS})
        write_json(result_dir / "reset_freeze.json", {"artifact_hashes": frozen,
                   "all_resets_frozen_before_truth": True, "source_commit": receipt["source_commit"]})
        reader.seal_resets(sha256_file(result_dir / "reset_freeze.json"))
        truth = reader.read_truth()["A10/real_train_arrays.npz"]
        fingerprints.update(verify_array_bindings({"A10/real_train_arrays.npz": truth}, plan))
        truth_checks = preflight_truth(truth, pre)
        reader.close()
        save_record(result_dir, "permitted_truth", truth)
        write_json(result_dir / "source_member_fingerprints.json", fingerprints)
        write_json(result_dir / "preflight.json", {"truth": truth_checks,
            "baseline_refinement": pre["baseline_refinement"], "reset_identity_pass": True})
        inputs, offsets = pre["inputs"], pre["offsets"]
        models = {}
        for seed in SEEDS:
            scientific.begin(f"construction-{seed}", {"model_constructions": 1, "eigvalsh": 3})
            scientific.append({"category": "closure_construction", "seed": seed, "transition": "attempted"})
            counters["model_constructions"] += 1
            try:
                with counted_structure(scientific, counters):
                    models[seed] = NonlinearPassiveClosure(**pre["parameters"][seed], backbone="sine")
            except (np.linalg.LinAlgError, FloatingPointError, OverflowError) as error:
                scientific.append({"category": "closure_construction", "seed": seed,
                                   "transition": "failed", "error_type": type(error).__name__})
                for slot in slots:
                    if slot["seed"] == seed:
                        slot.update(status="prerequisite_failed", V_pass=False, error_type=type(error).__name__)
                write_json(result_dir / f"construction-{seed}-failure.json", {
                    "error_type": type(error).__name__, "error": str(error), "automatic_retry": False})
                saved = save_record(result_dir, f"construction-{seed}-partial", getattr(error, "partial_record", None))
                scientific.finish({"status": "failed", "artifact_hashes": saved})
                continue
            except BaseException as error:
                scientific.append({"category": "closure_construction", "seed": seed,
                                   "transition": "failed", "error_type": type(error).__name__})
                raise
            scientific.append({"category": "closure_construction", "seed": seed, "transition": "returned"})
            scientific.finish({"status": "complete"})
        all_valid = len(models) == 3
        for seed in SEEDS:
            if seed not in models:
                continue
            for index, row in enumerate(population["cache_rows"]):
                slot = slots[(seed - SEEDS[0]) * 258 + index]
                n = int(pre["counts"][index])
                a, b = offsets[index:index + 2]
                clock, observed = truth["forecast_time"][a:b], truth["future"][a:b]
                history, eq, power = (inputs[k][index] for k in ("history", "equilibrium", "constant_power"))
                common = (eq, inputs["mass"], float(truth["omega_s"][index]), inputs["target_scale"])
                directory = result_dir / f"seed-{seed}-event-{index:03d}"
                directory.mkdir(exist_ok=False)
                scientific.begin(f"seed-{seed}-event-{index:03d}", {
                    "slot": row, "eigvalsh": 6, "rollouts": 2, "integration_rhs": 24 * n,
                    "physical_rhs": 2 * (n + 1), "rk4_updates": 6 * n})
                slot["status"] = "attempted"
                saved, physical, pair, ds, old_ds = {}, {}, {}, {}, {}
                before = len(budget.records)
                try:
                    for step in STEPS:
                        baseline = pre["controls"][seed, step][a:b]
                        old_ds[step] = rule_audit(history, observed, baseline, *common,
                                                 inputs["history_time"][index], clock)
                        identity = {"seed": seed, "event_id": row["event_id"], "step": step}

                        def preserve(path, step=step, saved=saved, directory=directory,
                                     baseline=baseline, observed=observed, clock=clock,
                                     reset=pre["resets"][seed, step][index]):
                            saved.update(save_record(directory, f"future-{step}", {"candidate": path,
                                "baseline": baseline, "truth": observed, "clock": clock, "reset": reset}))

                        with counted_structure(scientific, counters):
                            path, checks = audited_future(models[seed], pre["resets"][seed, step][index],
                                clock, power, eq, step, budget, scientific, identity, preserve)
                        pair[step], physical[str(step)] = path, checks
                        candidate_paths[seed, step, index] = path
                        saved.update(save_record(directory, f"physical-{step}", checks))
                        ds[step] = rule_audit(history, observed, path, *common, inputs["history_time"][index], clock)
                        diagnoses[seed, step, index] = {"candidate": ds[step], "baseline": old_ds[step]}
                    fine_checks = refinement(pair[.005], pair[.0025], inputs["target_scale"])
                    rule_fine = compare_rule_pair(history, pair[.005], pair[.0025], ds[.005], *common,
                                                 ds[.005], ds[.0025])
                    old_rule_fine = compare_rule_pair(history, pre["controls"][seed, .005][a:b],
                        pre["controls"][seed, .0025][a:b], old_ds[.005], *common, old_ds[.005], old_ds[.0025])
                    valid = bool(fine_checks["qualified"] and pre["baseline_refinement"][str(seed)][index]["qualified"]
                        and rule_fine["qualified"] and old_rule_fine["qualified"]
                        and all(v["qualified"] for v in physical.values())
                        and all(d["margin_validity"]["qualified"] for d in (*ds.values(), *old_ds.values())))
                    slot.update(status="complete", V_pass=valid, refinement=fine_checks,
                        candidate_rule_refinement=rule_fine, baseline_rule_refinement=old_rule_fine)
                    all_valid = all_valid and valid
                except (np.linalg.LinAlgError, FloatingPointError, OverflowError, ArithmeticError) as error:
                    partial = getattr(error, "partial_record", None)
                    if partial is not None:
                        saved.update(save_record(directory, "partial_failure", partial))
                    slot.update(status="failed", V_pass=False, error_type=type(error).__name__, error=str(error))
                    all_valid = False
                saved.update(save_record(directory, "diagnostics", {"candidate": {str(k): v for k, v in ds.items()},
                                                                    "baseline": {str(k): v for k, v in old_ds.items()}}))
                slot["rhs_segments"] = budget.records[before:]
                write_json(directory / "slot.json", slot)
                saved["slot.json"] = sha256_file(directory / "slot.json")
                scientific.finish({"status": slot["status"], "artifact_hashes": saved,
                                   "rhs_segments": budget.records[before:]})
                if (index + 1) % 20 == 0 or index == 257:
                    print(f"A15 seed={seed}: {index + 1}/258 event slots retained", flush=True)
        complete = (len(candidate_paths) == 1548 and len(diagnoses) == 1548
                    and all(slot["status"] == "complete" for slot in slots))
        if complete:
            for seed in SEEDS:
                for step in STEPS:
                    key = f"{seed}-{step}"
                    comparisons[key], metric_valid = save_future_comparison(result_dir, key,
                        np.concatenate([candidate_paths[seed, step, i] for i in range(258)]),
                        pre["controls"][seed, step], truth["future"], truth["forecast_time"], offsets,
                        inputs["target_scale"], population["weights"])
                    all_valid = all_valid and metric_valid
                    rule_comparisons[key] = compare_rule_counts(
                        [diagnoses[seed, step, i]["candidate"] for i in range(258)],
                        [diagnoses[seed, step, i]["baseline"] for i in range(258)])
                    save_record(result_dir, f"rule-comparison-{key}", rule_comparisons[key])
        save_record(result_dir, "future_comparisons", comparisons)
        save_record(result_dir, "rule_comparisons", rule_comparisons)
        scientific.close()
        for key, item in indexed.items():
            integrity.append(bounded_integrity_hash(paths[key], item["size_bytes_stat_only"],
                item["sha256_from_sealed_manifest"], io, "after"))
        io.close()
        access, rhs = reader.snapshot(), budget.snapshot()
        actual = access["actual"]
        access_ok = (not access["failed"] and access["file_count"] == 1559
            and actual == {"file_contexts_returned": 1559, "members_header_returned": 3136,
                "members_payload_returned": 3136, "logical_elements_returned": 11213485,
                "i8_elements_returned": 1813} and access["truth_read"] and access["arrays_read_only"]
            and len(access["file_receipts"]) == 1559
            and all(item["status"] == "complete" for item in access["file_receipts"]))
        integrity_ok = (len(integrity) == 3118
            and sum(item["actual_bytes"] for item in integrity) == 199075034
            and all(item["status"] == "returned" for item in integrity)
            and all(sum(item["phase"] == phase for item in integrity) == 1559 for phase in ("before", "after")))
        updates = sum(s["successful_rk4_updates"] or 0 for s in budget.records)
        calls_ok = (rhs["attempted"] == rhs["returned"] == TOTAL_RHS and rhs["blocked"] == 0
            and rhs["rollout_calls_started"] == 1548 and updates == 1298646
            and rhs["integration_rhs"]["returned"] == INTEGRATION_RHS
            and rhs["physical_rhs"]["returned"] == PHYSICAL_RHS
            and counters == {"model_constructions": 3, "eigvalsh_attempted": 4653, "eigvalsh_returned": 4653})
        valid = all_valid and complete and access_ok and calls_ok and integrity_ok
        improvement = complete and all(v["I_pass"] for v in comparisons.values())
        diagnostic = complete and all(v["D_pass"] for v in rule_comparisons.values())
        summary = {"execution_completed": True, "scientific_verdict": "ACCEPT" if valid and improvement and diagnostic else "REJECT",
            "V_pass": valid, "I_pass": improvement, "D_pass": diagnostic, "complete_population": complete,
            "calls_exact": calls_ok, "access_checks_pass": access_ok, "integrity_checks_pass": integrity_ok,
            "npz_integrity_bytes": sum(item["actual_bytes"] for item in integrity),
            "rhs": rhs, "counters": counters,
            "successful_rk4_updates": updates, "allocated_train": 268, "eligible_train": 258,
            "seed_event_slots": 774, "completed_slots": sum(s["status"] == "complete" for s in slots),
            "source_commit": receipt["source_commit"], "claim_boundary": config["claim_boundary"],
            "fit_attempts": 0, "optimizer_updates": 0, "history_paths": 0, "baseline_future_paths": 0,
            "encoder_calls": 0, "expm": 0, "svd": 0, "ls": 0, "independent_force": 0,
            "candidate_future_paths": rhs["rollout_calls_started"], "label_reads": 0, "selected_model": None,
            "new_training_authorized": False, "val_evaluation_authorized": False, "model_promoted": False,
            "promotion_decision": "KEEP_BASELINE", "certified": False, "automatic_retry": False}
        write_json(result_dir / "slots.json", slots)
        write_json(result_dir / "summary.json", summary)
        write_json(result_dir / "access_ledger.json", access)
        write_json(result_dir / "rhs_budget.json", {"snapshot": rhs, "segments": budget.records})
        write_json(result_dir / "integrity_receipts.json", integrity)
        write_json(result_dir / "stage_evidence.json", stage_evidence(receipt, summary, config["claim_boundary"]))
        write_json(result_dir / "artifact_hashes.json", {p.relative_to(result_dir).as_posix(): sha256_file(p)
            for p in sorted(result_dir.rglob("*")) if p.is_file()})
        return summary
    except BaseException as error:
        try:
            write_json(result_dir / "failed_slots.json", slots)
            write_json(result_dir / "future_transfer_failure.json", {"error_type": type(error).__name__,
                "message": str(error), "automatic_retry": False, "execution_completed": False,
                "active_scope_status": "unknown_spent" if scientific.active else "closed",
                "counters_attempted": counters, "rhs_prefix_only": budget.snapshot()})
            if reader is not None:
                reader.close()
                write_json(result_dir / "failed_access_ledger.json", reader.snapshot())
            scientific.close()
            io.close()
        except BaseException as preservation_error:  # noqa: BLE001 - preserve the original fatal cause.
            error.add_note(f"Failure preservation also failed: {preservation_error!r}")
        raise


def read_source_metadata(root, inventory, io):
    """One bounded byte read per registered metadata document, after reservation."""
    result, seen, total = {}, set(), 0
    rows = inventory["metadata_files"]
    if len(rows) != 1563 or sum(row["size_bytes"] for row in rows) != 4173766:
        raise PermissionError("A15 metadata capacity differs")
    for row in rows:
        relative, size = row["path"], row["size_bytes"]
        path = root / relative
        if (relative in seen or row["max_reads"] != 1 or row["max_hash_passes"] != 1
                or row["max_decode_passes"] != int(row["mode"] == "json_metadata")
                or row["mode"] not in {"json_metadata", "integrity_only"}):
            raise PermissionError("A15 source metadata selector or capacity changed")
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
        seen.add(relative)
        io.append({"kind": "source_metadata", "transition": "attempted", "path": relative,
                   "planned_bytes": size, "sha256": row["sha256"], "mode": row["mode"]})
        try:
            with path.open("rb") as stream:
                before = os.fstat(stream.fileno())
                if before.st_size != size:
                    raise PermissionError("A15 metadata byte length changed")
                raw = stream.read(size)
                after = os.fstat(stream.fileno())
            if (len(raw) != size or hashlib.sha256(raw).hexdigest() != row["sha256"]
                    or (before.st_size, before.st_mtime_ns, before.st_ino)
                    != (after.st_size, after.st_mtime_ns, after.st_ino)):
                raise PermissionError("A15 source metadata identity changed")
            total += len(raw)
            if row["mode"] == "json_metadata":
                result[relative] = json.loads(raw.decode("utf-8"))
        except BaseException as error:
            io.append({"kind": "source_metadata", "transition": "failed", "path": relative,
                       "error_type": type(error).__name__})
            raise
        io.append({"kind": "source_metadata", "transition": "returned", "path": relative,
                   "actual_bytes": len(raw), "sha256": row["sha256"], "decoded": row["mode"] == "json_metadata"})
    if total != 4173766 or len(result) != 1559:
        raise PermissionError("A15 metadata actual totals differ")
    return result


def verify_metadata_semantics(metadata, plan, inventory):
    runs = inventory["source_runs"]
    def source(name, relative):
        return metadata[f"results/{runs[name]}/{relative}"]

    manifests = {name: source(name, "artifact_hashes.json") for name in runs}
    for item in inventory["npz_files"]:
        if manifests[item["source"]].get(item["relative_path"]) != item["sha256_from_sealed_manifest"]:
            raise PermissionError("A15 source file identity differs from sealed manifest")
    for item in inventory["metadata_files"]:
        for name, run in runs.items():
            prefix = f"results/{run}/"
            if (item["path"].startswith(prefix) and not item["path"].endswith("/artifact_hashes.json")
                    and manifests[name].get(item["path"][len(prefix):]) != item["sha256"]):
                raise PermissionError("A15 metadata identity differs from sealed manifest")
    population = plan["population"]
    original = source("A14", "allocated_population.json")
    if any(original[k] != population[k] for k in ("allocated_slots", "cache_rows", "weights", "seed_event_slots")):
        raise PermissionError("A15 population differs from the accepted A14 allocation")
    a10 = source("A10", "scientific_plan.json")
    if (a10["train_event_ids"] != [r["event_id"] for r in population["cache_rows"]]
            or a10["future_point_counts"] != population["future_point_counts"]
            or a10["event_weights"] != population["weights"]
            or any(a10["approval"][k] != v for k, v in plan["source_model_ancestry"].items())):
        raise PermissionError("A15 original event counts, weights or model ancestry differ")
    summary = source("A14", "summary.json")
    if (any(summary.get(k) is not True for k in ("execution_completed", "V_pass", "I_pass", "calls_exact", "access_checks_pass"))
            or summary["scientific_verdict"] != "ACCEPT" or summary["future_paths"] != 0 or summary["label_reads"] != 0
            or summary["completed_slots"] != 774):
        raise PermissionError("A15 requires the complete accepted A14 history mechanism")
    ancestry = source("A14", "source_members.json")
    for seed in SEEDS:
        parameters = source("A14", f"static-{seed}.json")["parameters"]
        if parameters["backbone"] != "sine":
            raise PermissionError("A15 backbone differs from A14")
        for key in ("mass", "edge_stiffness", "damping", "coupling", "latent_skew", "latent_dissipation"):
            old = ancestry[f"A06/checkpoint-240-{seed}/model.npz"][key]
            if any(parameters[key][field] != old[field] for field in ("shape", "dtype", "sha256_c_order_bytes")):
                raise PermissionError("A15 closure parameters differ from original A06 arrays")
    for key, value in source("A14", "permitted_inputs.json").items():
        old = ancestry["A10/real_train_arrays.npz"][key]
        if any(value[field] != old[field] for field in ("shape", "dtype", "sha256_c_order_bytes")):
            raise PermissionError("A15 causal inputs or scales differ from original A10 cache")
    for file in plan["npz_files"]:
        for member in file["allowed_members"]:
            descriptor_path = member["descriptor_metadata_path"]
            if descriptor_path is None:
                continue
            descriptor = metadata[descriptor_path]
            if member["name"].startswith("parameters__"):
                descriptor = descriptor["parameters"][member["name"].removeprefix("parameters__")]
            else:
                descriptor = descriptor[member["name"]]
            if (descriptor["array_member"] != member["name"] or descriptor["shape"] != member["shape"]
                    or descriptor["dtype"] != member["dtype"]
                    or descriptor["sha256_c_order_bytes"] != member["sha256_c_order_bytes"]):
                raise PermissionError("A15 permitted array descriptor differs from sealed metadata")


def verify_array_bindings(decoded, plan):
    indexed = {file["file_id"]: file for file in plan["npz_files"]}
    result = {}
    for file_id, arrays in decoded.items():
        file = indexed[file_id]
        if set(arrays) != {m["name"] for m in file["allowed_members"]}:
            raise PermissionError("A15 decoded member names differ from explicit plan")
        result[file_id] = {}
        for member in file["allowed_members"]:
            fingerprint = array_fingerprint(arrays[member["name"]])
            if (fingerprint["shape"] != member["shape"] or fingerprint["dtype"] != member["dtype"]
                    or (member["sha256_c_order_bytes"] is not None
                        and fingerprint["sha256_c_order_bytes"] != member["sha256_c_order_bytes"])):
                raise PermissionError("A15 decoded array identity differs from frozen source descriptor")
            result[file_id][member["name"]] = fingerprint
    return result


def run(*, root, experiment, stage, result_dir, run_id):
    if stage != STAGE:
        raise PermissionError("A15 permits only its registered train future transfer stage")
    config = tomllib.loads((root / "configs/experiments" / experiment["config"]).read_text(encoding="utf-8"))
    approval = authorization(root, experiment, config)
    plan, inventory, indexed = bound_metadata(root, config)
    budget_dir, receipt = reserve_budget(root, run_id, config)
    try:
        summary = _execute(root, result_dir, config, plan, inventory, indexed, receipt)
        authorization(root, experiment, config)
        if git_head(root) != receipt["source_commit"]:
            raise RuntimeError("A15 source HEAD changed during its one-use execution")
        write_json(budget_dir / "completed.json", {"run_id": run_id, "authorization_spent": True,
            "scientific_verdict": summary["scientific_verdict"],
            "artifact_manifest_sha256": sha256_file(result_dir / "artifact_hashes.json"),
            "scientific_review_sha256": approval["scientific_review_sha256"]})
    except BaseException as error:
        write_json(budget_dir / "failed.json", {"run_id": run_id, "authorization_spent": True,
            "automatic_retry": False, "error_type": type(error).__name__, "message": str(error)})
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
    parser = argparse.ArgumentParser(description="Execute one-use A15 train future transfer")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    print(execute_registered_audit(parser.parse_args().root))


if __name__ == "__main__":
    main()
