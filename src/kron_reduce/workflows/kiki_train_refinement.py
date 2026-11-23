"""A10: all-train refinement of frozen models against saved A09 paths."""
from __future__ import annotations

import json
import time
import tomllib
from importlib.metadata import version

import numpy as np

from kron_reduce.data.kiki_history import KikiHistorySamples, resolve_source_root
from kron_reduce.evaluation.audit_budget import AuditRhsBudget
from kron_reduce.evaluation.recorded_error_audit import weights
from kron_reduce.evaluation.train_mechanism_audit import (
    _rule_and_bound,
    _trigger_times,
    audit_event,
)
from kron_reduce.evaluation.train_refinement_audit import compare_saved_replay, summarize_refinement
from kron_reduce.models.learned_latent import visible_scales
from kron_reduce.models.nonlinear_latent import LearnedNonlinearModel
from kron_reduce.workflows.kiki_short_history import shared_budget_directory
from kron_reduce.workflows.kiki_train_mechanism import (
    _npz,
    counted_model,
    integration_substeps,
    path_physics,
    weighted_trajectory_summary,
)
from kron_reduce.workflows.nonlinear_history_learning import _write
from project_harness.evidence import sha256_file, verify_artifact_directory
from project_harness.git_state import assert_clean, git_head, status_lines
from project_harness.report_storage import locate_report

AUTH = "kiki_a10_development_20261024"
SEEDS = (324, 325, 326)
STEPS = (0.005, 0.0025)
EXPERIMENT = "paper_kiki_a10_train_refinement"


def authorization(root, experiment, config):
    policy = json.loads((root/"configs/research_policy.json").read_text(encoding="utf-8"))
    approval = policy["development_evaluations"].get(AUTH, {})
    expected = {"status": "user_approved", "authority": "explicit_user_reply",
        "scientific_design_approver": "Astra Ultra", "experiment": EXPERIMENT,
        "workflow": "kiki_a10_train_refinement_v1", "stage": "train_refinement",
        "module": "kron_reduce.workflows.kiki_train_refinement", "dataset": "kiki_ieee39_v1",
        "max_fit_attempts": 0, "max_optimizer_updates": 0, "max_prediction_paths": 774,
        "max_rollout_calls": 1548, "max_rk4_updates": 896724,
        "max_integration_rhs": 3586896, "max_physical_rhs": 225729, "max_total_rhs": 3812625,
        "labels_allowed": False, "internal_val_payload_allowed": False,
        "source_val_payload_allowed": False, "source_test_payload_allowed": False,
        "model_promotion_allowed": False, "deployment_predictions_allowed": False}
    if any(type(approval.get(k)) is not type(v) or approval.get(k) != v for k, v in expected.items()):
        raise PermissionError("A10 requires its active fixed-model train-only authorization")
    if any(k not in approval or approval[k] != v for k, v in config.items()):
        raise PermissionError("A10 configuration differs from its approval")
    if (config["seeds"] != list(SEEDS) or config["max_steps"] != [0.0025]
            or config["input_contract"] != "measured_history_100ms_zoh_v1"
            or config["history_count"] != 11 or config["power_index"] != 10
            or config["persistence"] != 5 or config["total_future_points"] != 72147
            or config["allocated_train"] != 268 or config["eligible_train"] != 258):
        raise PermissionError("A10 fixed input, population or integration plan changed")
    manifest = json.loads((root/"configs/project_manifest.json").read_text(encoding="utf-8"))
    if manifest["experiments"].get(EXPERIMENT) != experiment:
        raise PermissionError("A10 experiment identity changed")
    for path_key, sha_key in (("development_protocol_path", "approval_protocol_sha256"),
                              ("design_path", "design_sha256")):
        if sha256_file(root/locate_report(root, config[path_key], config[sha_key])) != config[sha_key]:
            raise PermissionError("A10 approved document bytes changed")
    if sha256_file(root/locate_report(root, approval["scientific_review_path"], approval["scientific_review_sha256"])) != approval["scientific_review_sha256"]:
        raise PermissionError("A10 Astra scientific approval binding changed")
    return approval



def require_array_identity(saved, rebuilt):
    """Stop before integration if any saved input field or bit differs."""
    if set(saved) != set(rebuilt) or any(not np.array_equal(saved[k], v) for k,v in rebuilt.items()):
        raise ValueError("A10 train truth arrays differ from saved A09")


def compare_on_truth_windows(*, history, coarse, fine, truth_diagnostic,
                             equilibrium_delta, mass, omega_s, target_scale):
    """Compare two paths on all populations and the original truth's five points."""
    margin = audit_event(history=history, true_future=coarse, pred_future=fine,
        equilibrium_delta=equilibrium_delta, mass=mass, omega_s=omega_s, target_scale=target_scale)
    comparison = {k:margin[k] for k in ("history_rho","future_rho","full_rho")}
    representative = truth_diagnostic["representative_truth_window"]
    start, end = representative["start_index"], representative["end_index"]
    left, right = (np.concatenate((history, p[1:])) for p in (coarse, fine))
    fixed = _rule_and_bound(left[start:end+1],right[start:end+1],
        equilibrium_delta,mass,omega_s,min_end=4)
    for rule in (fixed["truth"], fixed["prediction"]):
        _trigger_times(rule,index_offset=start)
    comparison["representative_truth_window"] = {"start_index":start,"end_index":end,
        "coarse_margin_same_window":fixed["truth"]["rho"],
        "fine_margin_same_window":fixed["prediction"]["rho"],
        "fixed_five_point_audit":fixed}
    return comparison


def run(*, root, experiment, stage, result_dir, run_id):
    started = time.perf_counter()
    if stage != "train_refinement":
        raise ValueError("A10 has only its registered train-refinement stage")
    config_path = root/"configs/experiments"/experiment["config"]
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    approval = authorization(root, experiment, config)
    assert_clean(root)
    commit = git_head(root)
    runtime = {name: version(name) for name in ("numpy", "scipy", "torch", "h5py", "filelock")}
    if runtime != {"numpy": "2.5.2", "scipy": "1.18.0", "torch": "2.13.0+cpu",
                   "h5py": "3.16.0", "filelock": "3.32.2"}:
        raise RuntimeError("A10 requires the pinned A06/A08 CPU runtime")
    ledger = shared_budget_directory(root).parent/"a10_audit_budgets"/AUTH
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.mkdir(exist_ok=False)
    _write(ledger/"reserved.json", {"run_id": run_id, "source_commit": commit,
        "max_paths": 774, "fits": 0, "retries_allowed": False})
    source = root/"results"/config["source_run"]
    source_manifest = verify_artifact_directory(source, strict=True,
        expected_manifest_sha256=config["source_manifest_sha256"])
    prior = root/"results"/config["prior_audit_run"]
    prior_manifest = verify_artifact_directory(prior, strict=True,
        expected_manifest_sha256=config["prior_audit_manifest_sha256"])
    prior_plan = json.loads((prior/"scientific_plan.json").read_text(encoding="utf-8"))
    if sha256_file(prior/"scientific_plan.json") != config["prior_plan_sha256"]:
        raise ValueError("A10 fixed A09 plan changed")
    for rel, digest in config["prior_artifact_sha256"].items():
        if sha256_file(prior/rel) != digest:
            raise ValueError("A10 fixed A09 array changed")
    for rel in ("src/kron_reduce/data/kiki_history.py", "src/kron_reduce/data/history_contract.py",
                "src/kron_reduce/physics/coordinates.py", "src/kron_reduce/models/nonlinear_latent.py",
                "src/kron_reduce/models/causal_latent.py", "src/kron_reduce/models/learned_latent.py",
                "src/kron_reduce/evaluation/train_mechanism_audit.py"):
        if sha256_file(root/rel) != prior_plan["source_sha256"][rel]:
            raise ValueError("A10 source model/data conversion/audit code changed")
    allocation_path = root/config["frozen_allocation_path"]
    if sha256_file(allocation_path) != config["frozen_allocation_sha256"]:
        raise ValueError("A10 frozen allocation changed")
    allocation = json.loads(allocation_path.read_text(encoding="utf-8"))
    allocated = [r for r in allocation["rows"] if r["role"] == "train"]
    if len(allocated) != 268:
        raise ValueError("A10 requires all original train allocations")
    source_root = resolve_source_root(root, config["source_root"])
    bindings = allocation["source_bindings"]
    integrity = {"source_artifact_files": source_manifest, "source_artifact_root": source.as_posix(),
        "allocation_path": config["frozen_allocation_path"], "allocation_sha256": sha256_file(allocation_path),
        "raw_source_files": bindings, "raw_source_root": source_root.as_posix(),
        "prior_artifact_files": prior_manifest, "prior_artifact_root": prior.as_posix(),
        "purpose": "full-file SHA byte reads only; no holdout, label or future-input decoding"}
    for name, values in bindings.items():
        for relative, digest in values.items():
            if sha256_file(source_root/name/relative) != digest:
                raise ValueError("A10 source integrity mismatch before decoded access")
    provider = KikiHistorySamples(source_root, allocated, bindings, history_count=11, power_index=10)
    mass, samples = provider.load("train")
    if len(samples) != 258 or len(provider.coverage) != 268:
        raise ValueError("A10 eligible train queue differs")
    event_ids = [s.event_id for s in samples]
    metadata = {s.event_id: {"source_dataset": s.source_dataset, "fault_line": s.fault_line} for s in samples}
    counts = np.array([len(s.forecast_time)-1 for s in samples], dtype=np.int64)
    if int(counts.sum()) != 72147:
        raise ValueError("A10 total target-point budget changed")
    for s, count in zip(samples, counts, strict=True):
        if not 1 <= count <= 300:
            raise ValueError("A10 target count outside its real available range")
        if (integration_substeps(s.history.time-s.history.time[0], .0025) != 40
                or integration_substeps(s.forecast_time, .0025) != 4*count):
            raise ValueError("A10 original-clock substep budget changed")
    models = {}
    for index, seed in enumerate(SEEDS):
        checkpoint = source/config["source_checkpoint_paths"][index]
        if sha256_file(checkpoint) != config["source_checkpoint_sha256"][index]:
            raise ValueError("A10 fixed source checkpoint changed")
        model = LearnedNonlinearModel.load(checkpoint)
        if (model.settings.latent_dimension != 2 or model.settings.frames != 3
                or not np.array_equal(model.closure.mass, mass)
                or not np.array_equal(model.target_scale, visible_scales(samples))):
            raise ValueError("A10 original model, mass or scale differs")
        models[seed] = model
    offsets = np.r_[0, np.cumsum(counts+1)]
    arrays = {"history": np.stack([np.c_[s.history.delta, s.history.omega] for s in samples]),
        "future": np.concatenate([np.c_[s.target_delta, s.target_omega] for s in samples]),
        "offsets": offsets, "equilibrium": np.stack([s.equilibrium_delta for s in samples]),
        "mass": mass, "target_scale": models[SEEDS[0]].target_scale}
    with np.load(prior/"real_train_arrays.npz", allow_pickle=False) as saved:
        require_array_identity(saved, arrays)
    queue = [{"event_id": s.event_id, **metadata[s.event_id], "role": "train"} for s in samples]
    event_weights = weights(queue)
    if (event_ids != prior_plan["train_event_ids"] or counts.tolist() != prior_plan["future_point_counts"]
            or event_weights != prior_plan["event_weights"]):
        raise ValueError("A10 population order, counts or original weights changed")
    coverage = json.loads((prior/"train_coverage.json").read_text(encoding="utf-8"))["rows"]
    if provider.coverage != coverage:
        raise ValueError("A10 complete train allocation/coverage differs")
    arrays.update(constant_power=np.stack([s.history.constant_power for s in samples]),
        history_time=np.stack([s.history.time for s in samples]),
        forecast_time=np.concatenate([s.forecast_time for s in samples]),
        omega_s=np.array([provider.static[s.source_dataset]["omega_s"] for s in samples]))
    prior_summary = json.loads((prior/"audit_summary.json").read_text(encoding="utf-8"))
    old_comparisons = {(r["seed"],r["event_id"]): r for r in
        json.loads((prior/"refinement_checks.json").read_text(encoding="utf-8"))["rows"]}
    rows, coarse_histories, coarse_futures, source_checks, weighted_summaries = {}, {}, {}, {}, {}
    # Every old diagnostic and aggregate is reconstructed before any new RHS call.
    for seed in SEEDS:
        directory = prior/f"seed-{seed}-h005"
        with np.load(directory/"paths.npz", allow_pickle=False) as saved:
            hp, fp, op = saved["history"].copy(), saved["future"].copy(), saved["offsets"].copy()
        if (hp.shape != (258,11,22) or fp.shape != (int(offsets[-1]),22)
                or not np.array_equal(op,offsets) or not np.isfinite(hp).all() or not np.isfinite(fp).all()):
            raise ValueError("A10 saved coarse path shapes/order/finiteness differ")
        coarse_histories[seed] = hp
        coarse_futures[seed] = [fp[offsets[i]:offsets[i+1]] for i in range(258)]
        old_rows = json.loads((directory/"event_diagnostics.json").read_text(encoding="utf-8"))["rows"]
        if len(old_rows) != 258 or [r["event_id"] for r in old_rows] != event_ids:
            raise ValueError("A10 old diagnostic queue differs")
        for i, (s, old_row) in enumerate(zip(samples, old_rows, strict=True)):
            if (old_row["status"] != "ok" or old_row["seed"] != seed or old_row["max_step"] != .005
                    or not old_row["physical"]["qualified"]):
                raise ValueError("A10 saved A09 path/physics unavailable")
            diagnostic = audit_event(history=arrays["history"][i],
                true_future=arrays["future"][offsets[i]:offsets[i+1]],
                pred_future=coarse_futures[seed][i][:,:20], equilibrium_delta=s.equilibrium_delta,
                mass=mass, omega_s=arrays["omega_s"][i], target_scale=models[seed].target_scale)
            check = compare_saved_replay(diagnostic, old_row["diagnostic"])
            source_checks[f"{seed}/{s.event_id}"] = check
            if not check["qualified"]:
                _write(result_dir/"preflight_replay_failure.json", source_checks)
                raise ValueError("A10 old saved diagnostic reconstruction failed before new RHS")
            rows[seed,.005,s.event_id] = {**old_row, "diagnostic": diagnostic,
                "physical_evidence_source": "A09 saved checks; no old-state RHS recomputed"}
        aggregate = weighted_trajectory_summary(event_ids,event_weights,counts,
            {e: rows[seed,.005,e] for e in event_ids})
        check = compare_saved_replay(aggregate, prior_summary["weighted_trajectory_summaries"][f"{seed}-0.005"])
        source_checks[f"aggregate/{seed}"] = check
        if not check["qualified"]:
            _write(result_dir/"preflight_replay_failure.json", source_checks)
            raise ValueError("A10 original weighted aggregate reconstruction failed before new RHS")
        weighted_summaries[f"{seed}-0.005"] = aggregate
    code_paths = ["AGENTS.md", "configs/research_policy.json", "configs/project_manifest.json",
        config_path.relative_to(root).as_posix(), "configs/workflows/kiki_a10_train_refinement_v1.json",
        config["development_protocol_path"], config["design_path"], approval["scientific_review_path"],
        "configs/input_contracts.json", "src/kron_reduce/workflows/kiki_train_refinement.py",
        "src/kron_reduce/workflows/kiki_train_mechanism.py",
        "src/kron_reduce/evaluation/train_refinement_audit.py",
        "src/kron_reduce/evaluation/train_mechanism_audit.py", "src/kron_reduce/evaluation/audit_budget.py",
        "src/kron_reduce/models/nonlinear_latent.py", "src/kron_reduce/models/causal_latent.py",
        "src/kron_reduce/models/learned_latent.py", "src/kron_reduce/data/kiki_history.py",
        "src/kron_reduce/data/history_contract.py", "src/kron_reduce/physics/coordinates.py"]
    code_hashes = {p: sha256_file(root/p) for p in code_paths}
    plan = {"approval": approval, "source_commit": commit, "source_sha256": code_hashes,
        "runtime": runtime, "train_event_ids": event_ids, "future_point_counts": counts.tolist(),
        "event_weights": event_weights, "allocated_train": 268, "eligible_train": 258,
        "fits": 0, "optimizer_updates": 0, "max_prediction_paths": 774, "max_rollout_calls": 1548,
        "max_rk4_updates": 896724, "max_integration_rhs": 3586896,
        "max_physical_rhs": 225729, "max_total_rhs": 3812625,
        "coarse_reused_from_A09": True, "new_max_step": .0025, "labels_decoded": False}
    _write(result_dir/"scientific_plan.json",plan)
    _write(ledger/"plan.json",plan)
    _write(result_dir/"budget_binding.json", {"authorization_id":AUTH,
        "ledger_plan_sha256":sha256_file(ledger/"plan.json"),
        "scientific_plan_sha256":sha256_file(result_dir/"scientific_plan.json"), "retries_allowed":False})
    _write(result_dir/"input_and_saved_replay_checks.json", {"qualified":True,
        "array_equal_fields": ["history","future","equilibrium","mass","target_scale","offsets"],
        "event_order_counts_weights_coverage_equal":True,
        "causal_power_clock_binding":"same raw files and conversion code, not old array equality",
        "old_diagnostic_and_aggregate_checks":source_checks, "new_rhs_calls_before_checks":0})
    _write(result_dir/"integrity_file_reads.json",integrity)
    _write(result_dir/"train_coverage.json",{"rows":provider.coverage})
    _npz(result_dir/"real_train_arrays.npz",**arrays)
    budget = AuditRhsBudget(integration_limit=3586896, physical_limit=225729,
                           total_limit=3812625, rollout_limit=1548)
    refinements = {}
    for seed in SEEDS:
        model = counted_model(models[seed],budget)
        directory = result_dir/f"seed-{seed}-h0025"
        directory.mkdir()
        histories, futures, output = [], [], []
        for i, (s,count) in enumerate(zip(samples,counts,strict=True)):
            identity = {"seed":seed,"max_step":.0025,"event_id":s.event_id}
            slot = ledger/f"{seed}-h0025-{i:03d}"
            slot.mkdir(exist_ok=False)
            record = {**identity,"status":"failed","physical":{"qualified":False}}
            hp, fp = np.full((11,22),np.nan), np.full((int(count)+1,22),np.nan)
            before = len(budget.records)
            try:
                with budget.segment(identity,"history_integration",160):
                    z,hp = model.initialize(s.history,s.equilibrium_delta,max_step=.0025)
                with budget.segment(identity,"future_integration",16*int(count)):
                    fp = model.closure.rollout(s.history.delta[-1],s.history.omega[-1],z,
                        s.forecast_time,s.history.constant_power,s.equilibrium_delta,max_step=.0025)
                with budget.segment(identity,"physical_checks",int(count)+12):
                    physical = path_physics(model,hp,fp,s)
                diagnostic = audit_event(history=arrays["history"][i],
                    true_future=arrays["future"][offsets[i]:offsets[i+1]],pred_future=fp[:,:20],
                    equilibrium_delta=s.equilibrium_delta,mass=mass,omega_s=arrays["omega_s"][i],
                    target_scale=model.target_scale)
                record.update(status="ok",physical=physical,diagnostic=diagnostic)
            except (ValueError,ArithmeticError,RuntimeError) as error:
                record.update(exception_type=type(error).__name__,reason=str(error))
            record["rhs_segments"] = budget.records[before:]
            _write(slot/"finished.json",record)
            rows[seed,.0025,s.event_id] = record
            histories.append(hp)
            futures.append(fp)
            output.append(record)
            old_row = rows[seed,.005,s.event_id]
            refined = {"seed":seed,"event_id":s.event_id,"qualified":False}
            if record["status"] == "ok":
                try:
                    cp = coarse_futures[seed][i]
                    maximum = float(np.max(np.abs((cp[:,:20]-fp[:,:20])/model.target_scale)))
                    lr,rr = (r["diagnostic"]["future_rho"]["prediction"] for r in (old_row,record))
                    same = lr["triggered"] == rr["triggered"]
                    time_ok = not (same and lr["triggered"]) or abs(lr["first_trigger_end_index"]-rr["first_trigger_end_index"]) <= 2
                    margin = compare_on_truth_windows(history=arrays["history"][i],coarse=cp[:,:20],
                        fine=fp[:,:20],truth_diagnostic=record["diagnostic"],
                        equilibrium_delta=s.equilibrium_delta,mass=mass,
                        omega_s=arrays["omega_s"][i],target_scale=model.target_scale)
                    previous = old_comparisons[seed,s.event_id]["max_normalized_difference"]
                    refined.update(max_normalized_difference=maximum,same_rule_decision=same,
                        trigger_within_two_points=time_ok,coarse_rho=lr["rho"],fine_rho=rr["rho"],
                        absolute_rho_difference=abs(lr["rho"]-rr["rho"]),
                        history_initialization_max_normalized_difference=float(np.max(np.abs(
                            (coarse_histories[seed][i][:,:20]-hp[:,:20])/model.target_scale))),
                        coarse_fine_rule_margin_comparison=margin,
                        margin_bounds_verified=all(margin[k]["bound_verified"] is True
                            for k in ("history_rho","future_rho","full_rho"))
                            and margin["representative_truth_window"]["fixed_five_point_audit"]["bound_verified"] is True,
                        previous_A09_max_normalized_difference=previous,
                        ratio_new_to_previous=maximum/previous if previous > 1e-14 else None,
                        refinement_ratio_is_error_proof=False,
                        qualified=maximum <= 1e-5 and same and time_ok
                            and all(margin[k]["bound_verified"] is True for k in ("history_rho","future_rho","full_rho"))
                            and margin["representative_truth_window"]["fixed_five_point_audit"]["bound_verified"] is True)
                except (ValueError,ArithmeticError,RuntimeError) as error:
                    refined.update(exception_type=type(error).__name__,reason=str(error))
            refinements[seed,s.event_id] = refined
        _npz(directory/"paths.npz",history=np.stack(histories),future=np.concatenate(futures),offsets=offsets)
        _write(directory/"event_diagnostics.json",{"rows":output})
        weighted_summaries[f"{seed}-0.0025"] = weighted_trajectory_summary(event_ids,event_weights,counts,
            {e:rows[seed,.0025,e] for e in event_ids})
        print(f"A10 seed={seed}: 258 new fixed train paths audited",flush=True)
    snapshot = budget.snapshot()
    snapshot["successful_rk4_updates"] = sum(r["successful_rk4_updates"] or 0 for r in budget.records)
    complete = (len(refinements)==774 and len(budget.records)==2322
        and snapshot["integration_rhs"]["attempted"]==3586896
        and snapshot["physical_rhs"]["attempted"]==225729 and snapshot["rollout_calls_started"]==1548
        and snapshot["attempted"]==snapshot["returned"]==3812625 and snapshot["blocked"]==0
        and snapshot["successful_rk4_updates"]==896724
        and all(r["status"]=="complete" and r["attempted"]==r["returned"]==r["limit"] for r in budget.records))
    summary = summarize_refinement(event_ids,metadata,rows,refinements,source_checks,
        budget_complete=complete,event_weights=event_weights,counts=counts)
    summary.update(process_status="PASS",run_id=run_id,budget_complete=complete,budget=snapshot,
        allocated_train=268,eligible_train=258,prediction_paths_reserved=774,prediction_paths_finished=len(refinements),
        fits=0,optimizer_updates=0,old_saved_replay_checks=source_checks,
        weighted_trajectory_summaries=weighted_summaries,
        numerical_failures=[r for r in refinements.values() if not r["qualified"]],
        physical_or_path_failure_count=sum(r["status"]!="ok" or not r["physical"]["qualified"]
            for (seed,step,e),r in rows.items() if step==.0025),
        evidence_kind="uncertified_train_refinement_audit",delivery_predictions_issued=0,
        A09_verdict_unchanged="REJECT")
    _write(result_dir/"audit_summary.json",summary)
    _write(result_dir/"refinement_checks.json",{"rows":list(refinements.values())})
    _write(result_dir/"rhs_budget.json",{"snapshot":snapshot,"segments":budget.records})
    _write(result_dir/"data_access_audit.json", {**provider.access_audit(),
        "decoded_roles": ["train"], "decoded_labels_or_event_times": False,
        "allocate_roles_called": False, "truth_metadata_called": False,
        "split_metadata_reads": [f"{s}/splits.json" for s in sorted(bindings)],
        "static_h5_reads": ["inertia", "equilibrium/delta", "equilibrium/Pm", "attrs/omega_s",
            "attrs/mva_base", "attrs/raw_units", "attrs/state_mapping", "attrs/case_sha256"],
        "event_length_queries": [r["event_id"] for r in allocated],
        "file_integrity_reads_separately_recorded": "integrity_file_reads.json"})
    if git_head(root) != commit or status_lines(root) or any(sha256_file(root/p) != h for p, h in code_hashes.items()):
        raise RuntimeError("A10 implementation changed while the audit was running")
    if source_manifest != verify_artifact_directory(source, strict=True,
            expected_manifest_sha256=config["source_manifest_sha256"]):
        raise RuntimeError("A10 source evidence changed")
    if prior_manifest != verify_artifact_directory(prior, strict=True,
            expected_manifest_sha256=config["prior_audit_manifest_sha256"]):
        raise RuntimeError("A10 prior A09 evidence changed")
    for name, values in bindings.items():
        if any(sha256_file(source_root/name/p) != h for p, h in values.items()):
            raise RuntimeError("A10 raw source binding changed")
    _write(result_dir/"lineage.json", {"run_id": run_id, "source_commit": commit,
        "source_sha256": code_hashes, "wall_seconds": time.perf_counter()-started})
    _write(result_dir/"stage_evidence.json", {"source_commit": commit, "process_status": "PASS",
        "scientific_verdict": summary["scientific_verdict"], "promotion_decision": "KEEP_BASELINE",
        "data_policy": "kiki_original_train_only", "source_val_ids_read": [], "test_ids_read": [],
        "hidden_training_targets_read": False, "level_6_claimed": False,
        "development_authorization_id": AUTH, "evidence_kind": "uncertified_train_refinement_audit",
        "certified_envelope_established": False, "deployment_predictions_allowed": False,
        "model_promoted": False, "claim_boundary": config["claim_boundary"]})
    _write(result_dir/"artifact_hashes.json", {p.relative_to(result_dir).as_posix(): sha256_file(p)
        for p in sorted(result_dir.rglob("*")) if p.is_file() and p.name != "artifact_hashes.json"})
    verify_artifact_directory(result_dir, strict=True)
