"""One-use A14: fixed models and whitelisted train-history caches only."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tomllib
from collections.abc import Mapping
from importlib.metadata import version
from pathlib import Path

import numpy as np
from filelock import FileLock

from kron_reduce.data.durable_audit_ledger import AppendOnlyAuditJournal, DurableAccessLedger
from kron_reduce.data.limited_npz_audit import NpzWhitelistReader, fixed_file_specs
from kron_reduce.evaluation.audit_budget import AuditRhsBudget
from kron_reduce.evaluation.history_cache_preflight import load_fixed_model, preflight
from kron_reduce.evaluation.history_mechanism import residual_statistics, weighted_residual_summary
from kron_reduce.evaluation.history_metrics import comparison, hierarchy_weights
from kron_reduce.evaluation.integral_initializer import Ledger, build, prepare_static
from kron_reduce.evaluation.scoped_audit_journal import ScopedAuditJournal
from kron_reduce.workflows.kiki_record20_input_audit import (
    array_fingerprint,
    bounded_integrity_hash,
    write_json,
    write_npz,
)
from project_harness.evidence import sha256_file
from project_harness.git_state import assert_clean, git_head, verify_bound_file
from project_harness.manifest import load_manifest
from project_harness.policy import load_policy, validate_run_policy
from project_harness.runner import _run_experiment, _verify_development_text_bindings

AUTH = "kiki_a14_development_20261024"
EXPERIMENT = "paper_kiki_a14_integral_history"
WORKFLOW = "kiki_a14_integral_history_v1"
STAGE = "history_mechanism"
MODULE = "kron_reduce.workflows.kiki_integral_history_audit"
MODE = "bounded_history_cache_audit_cli_only"
SEEDS = (324, 325, 326)
STEPS = (.005, .0025)
LIMITS = {"expm": 7740, "svd": 780, "force": 46440, "ls": 1548}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def population(allocation):
    """Restore provider order from immutable allocation, without new role selection."""
    all_rows = [r for source in sorted(allocation["source_bindings"])
                for r in allocation["rows"]
                if r["source_dataset"] == source and r["source_role"] == r["role"] == "train"]
    fields = ("event_id", "raw_id", "source_dataset", "fault_line", "source_role", "role", "declared_length")
    slots, eligible = [], []
    for row in all_rows:
        item = {key: row[key] for key in fields}
        item["cache_index"] = len(eligible) if row["declared_length"] >= 12 else None
        item["input_status"] = "eligible" if item["cache_index"] is not None else "original_contract_ineligible"
        slots.append(item)
        if item["cache_index"] is not None:
            eligible.append(item)
    if (len(slots), len(eligible), len({r["event_id"] for r in slots})) != (268, 258, 268):
        raise PermissionError("A14 frozen train population differs")
    return {"allocated_slots": slots, "cache_rows": eligible,
            "weights": hierarchy_weights(eligible).tolist(),
            "seed_event_slots": [{"seed": s, "cache_index": r["cache_index"], "event_id": r["event_id"]}
                                 for s in SEEDS for r in eligible]}


def authorization(root, experiment, config):
    validate_run_policy(root, EXPERIMENT, experiment)
    _verify_development_text_bindings(root, experiment)
    approval = load_policy(root).get("development_evaluations", {}).get(AUTH, {})
    expected = {"status": "user_approved", "authority": "explicit_user_reply",
                "scientific_design_approver": "Astra Ultra", "experiment": EXPERIMENT,
                "workflow": WORKFLOW, "stage": STAGE, "module": MODULE,
                "dataset": "kiki_ieee39_v1", "max_campaign_executions": 1, "retry_allowed": False}
    if any(type(approval.get(k)) is not type(v) or approval.get(k) != v for k, v in expected.items()):
        raise PermissionError("A14 lacks its independent active one-use authorization")
    if any(k not in approval or approval[k] != v for k, v in config.items()):
        raise PermissionError("A14 config differs from approved design")
    if (experiment != load_manifest(root)["experiments"].get(EXPERIMENT)
            or experiment.get("execution_mode") != MODE
            or config["input_contract"] != "measured_history_100ms_zoh_v1"):
        raise PermissionError("A14 execution identity mismatch")
    bindings = approval.get("implementation_bindings", {})
    if not bindings or "src/" + MODULE.replace(".", "/") + ".py" not in bindings:
        raise PermissionError("A14 lacks reviewed implementation bindings")
    for relative, digest in bindings.items():
        verify_bound_file(root, relative, digest)
    for pk, sk in (("read_plan_path", "read_plan_sha256"), ("inventory_path", "inventory_sha256"),
                   ("journal_protocol_path", "journal_protocol_sha256")):
        verify_bound_file(root, config[pk], config[sk])
    verify_bound_file(root, approval["scientific_review_path"], approval["scientific_review_sha256"])
    assert_clean(root)
    return approval


def bound_metadata(root, config):
    if sha256_file(root / config["allocation_path"]) != config["allocation_sha256"]:
        raise PermissionError("A14 frozen allocation metadata changed")
    plan = read_json(root / config["read_plan_path"])
    inventory = read_json(root / config["inventory_path"])
    if plan["population"] != population(read_json(root / config["allocation_path"])):
        raise PermissionError("A14 population/order/weight plan changed")
    expected = {s.key for s in fixed_file_specs()}
    indexed = {f'{r["source"]}/{r["relative_path"]}': r for r in inventory["files"]}
    if set(indexed) != expected or sum(r["size_bytes_stat_only"] for r in indexed.values()) != 87732555:
        raise PermissionError("A14 registered file inventory differs")
    return plan, indexed


def verify_source_manifests(root, inventory, io):
    """Read only sealed manifest metadata after the one-use budget is spent."""
    for source, info in inventory["manifests"].items():
        path = root / "results" / info["run_id"] / "artifact_hashes.json"
        io.append({"category": "source_manifest_metadata", "source": source, "transition": "attempted",
                   "sha256": info["manifest_sha256"], "size_bytes": info["manifest_bytes"]})
        if path.stat().st_size != info["manifest_bytes"] or sha256_file(path) != info["manifest_sha256"]:
            raise PermissionError("A14 source manifest changed")
        manifest = read_json(path)
        for row in inventory["files"]:
            if row["source"] == source and manifest.get(row["relative_path"]) != row["sha256_from_sealed_manifest"]:
                raise PermissionError("A14 NPZ identity differs from sealed manifest")
        io.append({"category": "source_manifest_metadata", "source": source, "transition": "returned"})


def reserve_budget(root, run_id, config):
    common = Path(subprocess.run(["git", "-C", str(root), "rev-parse", "--path-format=absolute",
        "--git-common-dir"], check=True, capture_output=True, text=True).stdout.strip()).resolve(strict=True)
    directory = common / "a14_history_budgets" / AUTH
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir(exist_ok=False)
    receipt = {"authorization": AUTH, "run_id": run_id, "source_commit": git_head(root),
               "spent": True, "automatic_retry": False, "read_plan_sha256": config["read_plan_sha256"]}
    write_json(directory / "reservation.json", receipt)
    return directory, receipt


def pack_record(value, prefix="", arrays=None):
    """Save every nested numerical intermediate without object arrays or JSON NaN."""
    if arrays is None:
        arrays = {}
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("object evidence is forbidden")
        arrays[prefix] = value
        return {"array_member": prefix, **array_fingerprint(value)}
    if isinstance(value, Mapping):
        return {k: pack_record(v, f"{prefix}__{k}" if prefix else k, arrays) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [pack_record(v, f"{prefix}__{i}", arrays) for i, v in enumerate(value)]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return {"value": None, "reason": "nonfinite_diagnostic", "representation": repr(value)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unhandled evidence type: {type(value).__name__}")


def save_record(directory, name, value):
    arrays = {}
    metadata = pack_record(value, arrays=arrays)
    npz, js = directory / f"{name}.npz", directory / f"{name}.json"
    write_npz(npz, arrays)
    write_json(js, metadata)
    return {npz.name: sha256_file(npz), js.name: sha256_file(js)}


def differences(coarse, fine, scale):
    visible = float(np.max(np.abs((coarse[:, :20] - fine[:, :20]) / scale)))
    hidden = float(np.max(np.linalg.norm(coarse[:, 20:] - fine[:, 20:], axis=1)))
    bound = float(1e-5 * (1 + np.max(np.linalg.norm(fine[:, 20:], axis=1))))
    return {"visible_max_normalized": visible, "hidden_max_norm": hidden,
            "hidden_bound": bound, "qualified": visible <= 1e-5 and hidden <= bound}


def audited_path(closure, q, v, z, clock, power, eq, step, budget, journal, identity, *, path_sink=None):
    """Original rollout/energy methods, with a counted instance RHS wrapper."""
    original = closure.rhs

    def counted(*args, **kwargs):
        def dispatch():
            phase = budget._active["phase"]
            sequence = budget._active["attempted"]
            journal.append({"category": "rhs", "phase": phase, "sequence": sequence,
                            "transition": "attempted", **identity})
            try:
                result = original(*args, **kwargs)
            except BaseException as error:
                journal.append({"category": "rhs", "phase": phase, "sequence": sequence,
                                "transition": "failed", "error_type": type(error).__name__, **identity})
                raise
            journal.append({"category": "rhs", "phase": phase, "sequence": sequence,
                            "transition": "returned", **identity})
            return result
        return budget.invoke(dispatch)

    updates = int(sum(max(1, int(np.ceil(h / step - 1e-12))) for h in np.diff(clock)))
    object.__setattr__(closure, "rhs", counted)
    try:
        with budget.segment(identity, "history_integration", 4 * updates):
            budget.note_rollout_start()
            journal.append({"category": "rollout", "transition": "attempted", **identity})
            try:
                path = closure.rollout(q, v, z, clock, power, eq, max_step=step)
            except BaseException as error:
                journal.append({"category": "rollout", "transition": "failed", **identity,
                                "error_type": type(error).__name__})
                raise
            journal.append({"category": "rollout", "transition": "returned", **identity})
        if path_sink is not None:
            path_sink(path)
        physical = []
        with budget.segment(identity, "physical_checks", 11):
            for state in path:
                check = closure.energy_identity(state, power, eq)
                check.update(q_mass_coi=abs(float(closure.mass @ state[:10])),
                             v_mass_coi=abs(float(closure.mass @ state[10:20])))
                physical.append(check)
        qualified = all(p["identity_error"] <= 1e-8 and p["q_mass_coi"] <= 1e-10
                        and p["v_mass_coi"] <= 1e-10 for p in physical)
        return path, {"points": physical, "qualified": qualified}
    finally:
        object.__delattr__(closure, "rhs")


def stage_evidence(receipt, verdict, boundary):
    return {"status": "PASS", "process_status": "PASS", "source_commit": receipt["source_commit"],
            "scientific_verdict": verdict, "promotion_decision": "KEEP_BASELINE",
            "data_policy": "kiki_original_train_only", "source_val_ids_read": [],
            "internal_val_ids_read": [], "test_ids_read": [], "hidden_training_targets_read": False,
            "level_6_claimed": False, "development_authorization_id": AUTH,
            "evidence_kind": "uncertified_train_history_mechanism_audit",
            "certified_envelope_established": False, "deployment_predictions_allowed": False,
            "model_promoted": False, "claim_boundary": boundary}


def prepare_seed_scope(model, seed, model_sha256, slots, result_dir, journal):
    """Close a numerical static failure locally; integrity failures propagate."""
    owner = f"A06-step240-seed{seed}"
    journal.begin(f"static-{seed}", {"svd": 2, "owner": owner, "model_sha256": model_sha256})
    ledger = Ledger({"svd": 2}, event_sink=journal.append, context=owner)
    try:
        static = prepare_static(model.closure, model.target_scale[10:], owner_key=owner, ledger=ledger)
    except (np.linalg.LinAlgError, FloatingPointError, OverflowError) as error:
        failed = {"seed": seed, "status": "failed", "error_type": type(error).__name__,
                  "message": str(error), "candidate_rollouts": "not_attempted", "automatic_retry": False}
        saved = save_record(result_dir, f"static-{seed}-failed",
                            {"failure": failed, "partial": getattr(error, "partial_record", None)})
        for slot in slots:
            if slot["seed"] == seed:
                slot.update(status="prerequisite_failed", V_pass=False,
                            candidate_rollouts="not_attempted", prerequisite=f"static-{seed}")
        journal.finish({"status": "failed", "operator_budget": ledger.snapshot(), "artifact_hashes": saved})
        return None, ledger.snapshot()
    saved = save_record(result_dir, f"static-{seed}", dict(static.data))
    journal.finish({"status": "complete", "operator_budget": ledger.snapshot(), "artifact_hashes": saved})
    return static, ledger.snapshot()


def _execute(root, result_dir, config, plan, inventory, receipt):
    slots = [{**r, "status": "not_attempted"} for r in plan["population"]["seed_event_slots"]]
    write_json(result_dir / "allocated_population.json", plan["population"])
    write_json(result_dir / "environment.json", {"python_executable": sys.executable,
        "python_version": sys.version, "versions": {name: version(name) for name in
            ("numpy", "scipy", "torch", "h5py", "filelock")}})
    reader = None
    io = AppendOnlyAuditJournal(result_dir / "integrity_and_opens.jsonl")
    scientific = ScopedAuditJournal(result_dir / "scientific_journal")
    budget = AuditRhsBudget(185760, 17028, 202788, 1548)
    ledgers, integrity, counters = [], [], {"model_loads": 0, "model_constructions": 0}
    try:
        verify_source_manifests(root, read_json(root / config["inventory_path"]), io)
        paths = {key: root / "results" / item["run_id"] / item["relative_path"]
                 for key, item in inventory.items()}
        for phase in ("before",):
            for key, item in inventory.items():
                integrity.append(bounded_integrity_hash(paths[key], item["size_bytes_stat_only"],
                    item["sha256_from_sealed_manifest"], io, phase))
            for item in plan["inherited_physics_certificates"]:
                integrity.append(bounded_integrity_hash(root / item["path"], item["size_bytes"], item["sha256"], io, phase))

        def ledger_factory(read_plan, **kwargs):
            return DurableAccessLedger(read_plan, journal_path=result_dir / "member_access.jsonl", **kwargs)

        import zipfile

        def open_zip(path, mode):
            io.append({"category": "npz_context", "path": str(path), "transition": "attempted"})
            value = zipfile.ZipFile(path, mode)
            try:
                io.append({"category": "npz_context", "path": str(path), "transition": "returned"})
            except BaseException:
                value.close()
                raise
            return value

        reader = NpzWhitelistReader(paths, ledger_factory=ledger_factory, zip_factory=open_zip)
        decoded = reader.read_all()
        reader.ledger.close_journal()
        write_json(result_dir / "source_members.json", pack_record(decoded, arrays={}))
        truth = decoded["A10/real_train_arrays.npz"]
        save_record(result_dir, "permitted_inputs", truth)
        controls = {(seed, step): decoded[
            f'A09/seed-{seed}-h005/paths.npz' if step == .005 else f'A10/seed-{seed}-h0025/paths.npz']["history"]
                    for seed in SEEDS for step in STEPS}
        save_record(result_dir, "all_cached_controls", {
            f"seed-{seed}-step-{step}": paths for (seed, step), paths in controls.items()})
        scientific.begin("preflight", {"model_loads": 3, "model_constructions": 3,
            "features": 774, "encoder": 774, "model_sources": {k: v["sha256_from_sealed_manifest"]
            for k, v in inventory.items() if k.startswith("A06/")}, "read_plan_sha256": config["read_plan_sha256"]})
        models = {}
        for seed in SEEDS:
            scientific.append({"category": "model_load_and_construction", "seed": seed, "transition": "attempted"})
            counters["model_loads"] += 1
            counters["model_constructions"] += 1
            models[seed] = load_fixed_model(decoded[f"A06/checkpoint-240-{seed}/model.npz"], seed)
            scientific.append({"category": "model_load_and_construction", "seed": seed, "transition": "returned"})
        pre = preflight(truth, models, controls, event_sink=scientific.append)
        pre_hashes = save_record(result_dir, "preflight", pre)
        scientific.finish({"status": "complete", "artifact_hashes": pre_hashes})
        all_valid, candidate_paths, comparisons = True, {}, {}
        for seed in SEEDS:
            model, owner = models[seed], f"A06-step240-seed{seed}"
            static, static_snapshot = prepare_seed_scope(model, seed,
                inventory[f"A06/checkpoint-240-{seed}/model.npz"]["sha256_from_sealed_manifest"],
                slots, result_dir, scientific)
            ledgers.append(static_snapshot)
            if static is None:
                all_valid = False
                continue
            for index, row in enumerate(plan["population"]["cache_rows"]):
                slot = slots[(seed - SEEDS[0]) * 258 + index]
                directory = result_dir / f"seed-{seed}-event-{index:03d}"
                directory.mkdir(exist_ok=False)
                scope = f"seed-{seed}-event-{index:03d}"
                scientific.begin(scope, {"slot": row, "owner": owner, "expm": 10, "svd": 1,
                    "force": 60, "ls": 2, "rollouts": 2, "rk4_updates": 60,
                    "integration_rhs": 240, "physical_rhs": 22,
                    "model_sha256": static.model_sha256,
                    "input_sha256": array_fingerprint(truth["history"][index])["sha256_c_order_bytes"]})
                slot["status"] = "attempted"
                ledger = Ledger({"expm": 10, "svd": 1, "force": 60, "ls": 2},
                                event_sink=scientific.append, context=scope)
                before = len(budget.records)
                saved = {}
                try:
                    observed = truth["history"][index]
                    eq, power = truth["equilibrium"][index], truth["constant_power"][index]
                    clock = truth["history_time"][index]
                    result = build(observed[:, :10], observed[:, 10:], clock, model.closure,
                                   power, eq, cache=static, owner_key=owner, ledger=ledger)
                    result["kinematic_normalized"] = result["kin"] / model.target_scale[:10]
                    for order in (2, 4):
                        residual = result[f"dynamic_residual{order}"]
                        result[f"dynamic_residual_normalized{order}"] = (
                            None if residual is None else residual.reshape(10, 10) * result["S"][None, :])
                    saved.update(save_record(directory, "integral", result))
                    slot["residual_statistics"] = residual_statistics(result, model.target_scale)
                    if result["status"] == "unresolved":
                        slot.update(status="unresolved", candidate_rollouts="not_attempted", V_pass=False)
                        all_valid = False
                    else:
                        delta_zh = float(np.linalg.norm(result["zh2"] - result["zh4"]))
                        delta_zt = float(np.linalg.norm(result["zTobs2"] - result["zTobs4"]))
                        zh_bound = float(1e-6 * (1 + np.linalg.norm(result["zh4"])))
                        zt_bound = float(1e-6 * (1 + np.linalg.norm(result["zTobs4"])))
                        quad = delta_zh <= zh_bound and delta_zt <= zt_bound
                        path_pair, physical = {}, {}
                        for step in STEPS:
                            identity = {"seed": seed, "event_id": row["event_id"], "step": step}

                            def preserve_path(path, step=step, saved=saved, directory=directory,
                                              baseline=controls[seed, step][index], observed=observed,
                                              clock=clock, zTobs=result["zTobs4"]):
                                saved.update(save_record(directory, f"history-{step}", {
                                    "candidate": path, "baseline": baseline,
                                    "reset": np.r_[observed[-1], path[-1, 20:]],
                                    "baseline_reset": np.r_[observed[-1], baseline[-1, 20:]],
                                    "closed_loop_minus_observed_driven_zT": path[-1, 20:] - zTobs,
                                    "observed": observed, "clock": clock}))

                            path, checks = audited_path(model.closure, observed[0, :10], observed[0, 10:],
                                result["candidate_zh"], clock - clock[0], power, eq, step, budget, scientific, identity,
                                path_sink=preserve_path)
                            path_pair[step], physical[str(step)] = path, checks
                            candidate_paths[seed, step, index] = path
                            saved.update(save_record(directory, f"physical-{step}", checks))
                        refinement = differences(path_pair[.005], path_pair[.0025], model.target_scale)
                        valid = bool(quad and refinement["qualified"] and all(p["qualified"] for p in physical.values()))
                        slot.update(status="complete", V_pass=valid, quadrature_pass=bool(quad),
                                    zh_quadrature_norm=delta_zh, zTobs_quadrature_norm=delta_zt,
                                    zh_quadrature_bound=zh_bound, zTobs_quadrature_bound=zt_bound,
                                    closed_loop_minus_observed_driven_zT_norms={str(step): float(np.linalg.norm(
                                        path_pair[step][-1, 20:] - result["zTobs4"])) for step in STEPS},
                                    refinement=refinement, physical=physical,
                                    hidden_energy_encoder=float(.5 * np.dot(controls[seed, .005][index, 0, 20:],
                                                                           controls[seed, .005][index, 0, 20:])))
                        energy_old = slot["hidden_energy_encoder"]
                        slot["hidden_energy_ratio"] = result["hidden_energy4"] / energy_old if energy_old > 0 else None
                        all_valid = all_valid and valid
                except (np.linalg.LinAlgError, FloatingPointError, OverflowError) as error:
                    partial = getattr(error, "partial_record", None)
                    if partial is not None:
                        saved.update(save_record(directory, "failed_partial_integral", partial))
                    slot.update(status="failed", V_pass=False, error_type=type(error).__name__, error=str(error))
                    all_valid = False
                slot["operators"] = ledger.snapshot()
                slot["rhs_segments"] = budget.records[before:]
                write_json(directory / "slot.json", slot)
                saved["slot.json"] = sha256_file(directory / "slot.json")
                scientific.finish({"status": slot["status"], "artifact_hashes": saved,
                                   "operator_budget": ledger.snapshot(), "rhs_segments": budget.records[before:]})
                ledgers.append(ledger.snapshot())
            print(f"A14 seed={seed}: all 258 allocated history slots retained", flush=True)
        complete = len(candidate_paths) == 1548 and all(s["status"] == "complete" for s in slots)
        write_json(result_dir / "mechanism_residual_summary.json", weighted_residual_summary(
            slots, plan["population"]["weights"]))
        if complete:
            for seed in SEEDS:
                for step in STEPS:
                    value = comparison(np.stack([candidate_paths[seed, step, i] for i in range(258)]),
                        controls[seed, step], truth["history"], truth["target_scale"], plan["population"]["weights"])
                    comparisons[f"{seed}-{step}"] = value
            save_record(result_dir, "history_comparisons", comparisons)
        scientific.close()
        for key, item in inventory.items():
            integrity.append(bounded_integrity_hash(paths[key], item["size_bytes_stat_only"],
                item["sha256_from_sealed_manifest"], io, "after"))
        for item in plan["inherited_physics_certificates"]:
            integrity.append(bounded_integrity_hash(root / item["path"], item["size_bytes"], item["sha256"], io, "after"))
        io.close()
        access = reader.snapshot()
        member_ledger = access["ledger"]
        totals = {category: sum(r["returned"].get(category, 0) for r in ledgers) for category in LIMITS}
        rhs = budget.snapshot()
        rhs["successful_rk4_updates"] = sum(r["successful_rk4_updates"] or 0 for r in budget.records)
        calls_ok = totals == LIMITS and rhs["attempted"] == rhs["returned"] == 202788 and rhs["blocked"] == 0 and rhs["rollout_calls_started"] == 1548 and rhs["successful_rk4_updates"] == 46440
        access_ok = (not access["reader_failed"] and access["read_member_count"] == 42
            and len(access["opened_file_keys"]) == 10 and not member_ledger["poisoned"]
            and not member_ledger["denied_attempts"] and not member_ledger["violations"]
            and member_ledger["actual_array_elements"]["returned"] == 440733
            and all(member_ledger["actual_counts"][key]["returned"] == planned
                    and member_ledger["actual_counts"][key]["failed"] == 0
                    for key, planned in member_ledger["planned_counts"].items()))
        valid = all_valid and complete and calls_ok and access_ok
        improvement = complete and all(c["I_pass"] for c in comparisons.values())
        summary = {"execution_completed": True, "scientific_verdict": "ACCEPT" if valid and improvement else "REJECT",
            "V_pass": valid, "I_pass": improvement, "comparison_complete": complete,
            "calls_exact": calls_ok, "access_checks_pass": access_ok, "operators_returned": totals,
            "rhs": rhs, "counters": counters, "encoder_call_counts": pre["call_counts"],
            "allocated_train": 268, "eligible_train": 258,
            "seed_event_slots": 774, "completed_slots": sum(s["status"] == "complete" for s in slots),
            "claim_boundary": config["claim_boundary"], "source_commit": receipt["source_commit"],
            "fit_attempts": 0, "optimizer_updates": 0, "future_paths": 0, "label_reads": 0,
            "new_training_authorized": False, "future_evaluation_authorized": False, "model_promoted": False,
            "promotion_decision": "KEEP_BASELINE", "certified": False, "automatic_retry": False}
        write_json(result_dir / "slots.json", slots)
        write_json(result_dir / "summary.json", summary)
        write_json(result_dir / "access_ledger.json", access)
        write_json(result_dir / "operator_ledgers.json", ledgers)
        write_json(result_dir / "rhs_budget.json", {"snapshot": rhs, "segments": budget.records})
        write_json(result_dir / "integrity_receipts.json", integrity)
        write_json(result_dir / "stage_evidence.json", stage_evidence(receipt, summary["scientific_verdict"], config["claim_boundary"]))
        write_json(result_dir / "artifact_hashes.json", {p.relative_to(result_dir).as_posix(): sha256_file(p)
            for p in sorted(result_dir.rglob("*")) if p.is_file()})
        return summary
    except BaseException as error:
        try:
            write_json(result_dir / "failed_slots.json", slots)
            write_json(result_dir / "history_audit_failure.json", {"error_type": type(error).__name__,
                "message": str(error), "automatic_retry": False, "execution_completed": False,
                "active_scope_status": "unknown_spent" if scientific.active else "closed",
                "counters_attempted": counters, "rhs_prefix_only": budget.snapshot()})
            if reader is not None:
                reader.ledger.close_journal()
                write_json(result_dir / "failed_access_ledger.json", reader.snapshot())
            scientific.close()
            io.close()
        except BaseException as preservation_error:  # noqa: BLE001 - preserve the original fatal cause.
            error.add_note(f"Failure preservation also failed: {preservation_error!r}")
        raise


def run(*, root, experiment, stage, result_dir, run_id):
    if stage != STAGE:
        raise PermissionError("A14 permits only its registered history mechanism stage")
    config = tomllib.loads((root / "configs/experiments" / experiment["config"]).read_text(encoding="utf-8"))
    approval = authorization(root, experiment, config)
    plan, inventory = bound_metadata(root, config)
    budget_dir, receipt = reserve_budget(root, run_id, config)
    try:
        summary = _execute(root, result_dir, config, plan, inventory, receipt)
        authorization(root, experiment, config)
        if git_head(root) != receipt["source_commit"]:
            raise RuntimeError("A14 source HEAD changed during the one-use execution")
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
    parser = argparse.ArgumentParser(description="Execute one-use A14 train-history mechanism audit")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    print(execute_registered_audit(parser.parse_args().root))


if __name__ == "__main__":
    main()
