"""A09: bounded, label-free trajectory-shape audit of three frozen A06 models."""
from __future__ import annotations

import json
import math
import time
import tomllib
from dataclasses import fields
from importlib.metadata import version

import numpy as np

from kron_reduce.data.kiki_history import KikiHistorySamples, resolve_source_root
from kron_reduce.evaluation.audit_budget import AuditRhsBudget
from kron_reduce.evaluation.recorded_error_audit import weights
from kron_reduce.evaluation.train_mechanism_audit import audit_event
from kron_reduce.models.learned_latent import visible_scales
from kron_reduce.models.nonlinear_latent import LearnedNonlinearModel, NonlinearPassiveClosure
from kron_reduce.workflows.kiki_short_history import shared_budget_directory
from kron_reduce.workflows.nonlinear_history_learning import _write
from project_harness.evidence import sha256_file, verify_artifact_directory
from project_harness.git_state import assert_clean, git_head, status_lines

AUTH = "kiki_a09_development_20261024"
SEEDS = (324, 325, 326)
STEPS = (0.01, 0.005)
EXPERIMENT = "paper_kiki_a09_train_mechanism"


def authorization(root, experiment, config):
    policy = json.loads((root/"configs/research_policy.json").read_text(encoding="utf-8"))
    approval = policy["development_evaluations"].get(AUTH, {})
    expected = {"status": "user_approved", "authority": "explicit_user_reply",
        "scientific_design_approver": "Astra Ultra", "experiment": EXPERIMENT,
        "workflow": "kiki_a09_train_mechanism_v1", "stage": "train_mechanism",
        "module": "kron_reduce.workflows.kiki_train_mechanism", "dataset": "kiki_ieee39_v1",
        "max_fit_attempts": 0, "max_optimizer_updates": 0, "max_prediction_paths": 1548,
        "max_rollout_calls": 3096, "max_rk4_updates": 672543,
        "max_integration_rhs": 2690172, "max_physical_rhs": 451458, "max_total_rhs": 3141630,
        "labels_allowed": False, "internal_val_payload_allowed": False,
        "source_val_payload_allowed": False, "source_test_payload_allowed": False,
        "model_promotion_allowed": False, "deployment_predictions_allowed": False}
    if any(type(approval.get(k)) is not type(v) or approval.get(k) != v for k, v in expected.items()):
        raise PermissionError("A09 requires its active fixed-model train-only authorization")
    if any(k not in approval or approval[k] != v for k, v in config.items()):
        raise PermissionError("A09 configuration differs from its approval")
    if (config["seeds"] != list(SEEDS) or config["max_steps"] != list(STEPS)
            or config["input_contract"] != "measured_history_100ms_zoh_v1"
            or config["history_count"] != 11 or config["power_index"] != 10
            or config["persistence"] != 5 or config["total_future_points"] != 72147
            or config["allocated_train"] != 268 or config["eligible_train"] != 258):
        raise PermissionError("A09 fixed input, population or integration plan changed")
    manifest = json.loads((root/"configs/project_manifest.json").read_text(encoding="utf-8"))
    if manifest["experiments"].get(EXPERIMENT) != experiment:
        raise PermissionError("A09 experiment identity changed")
    for path_key, sha_key in (("development_protocol_path", "approval_protocol_sha256"),
                              ("design_path", "design_sha256")):
        if sha256_file(root/config[path_key]) != config[sha_key]:
            raise PermissionError("A09 approved document bytes changed")
    if sha256_file(root/approval["scientific_review_path"]) != approval["scientific_review_sha256"]:
        raise PermissionError("A09 Astra scientific approval binding changed")
    return approval


def integration_substeps(clock, step):
    clock = np.asarray(clock, dtype=float)
    if clock.ndim != 1 or len(clock) < 2 or not np.isfinite(clock).all() or np.any(np.diff(clock) <= 0):
        raise ValueError("A09 clock must have real, increasing output points")
    return sum(max(1, int(np.ceil(interval/step-1e-12))) for interval in np.diff(clock))


class CountedClosure(NonlinearPassiveClosure):
    """Count calls around the original numerical methods without changing them."""
    def rhs(self, *args):
        return self._audit_budget.invoke(super().rhs, *args)

    def rollout(self, *args, **kwargs):
        self._audit_budget.note_rollout_start()
        return super().rollout(*args, **kwargs)


def counted_model(model, budget):
    closure = CountedClosure(**{f.name: getattr(model.closure, f.name) for f in fields(NonlinearPassiveClosure)})
    object.__setattr__(closure, "_audit_budget", budget)
    model.closure = closure
    return model


def path_physics(model, history_path, future_path, sample):
    n, mass = 10, model.closure.mass
    paths = (history_path, future_path)
    if any(not np.isfinite(p).all() for p in paths):
        raise FloatingPointError("A09 saved path is nonfinite")
    energy = max(model.closure.energy_identity(row, sample.history.constant_power, sample.equilibrium_delta)
                 ["identity_error"] for p in paths for row in p)
    coi = max(float(np.max(np.abs(p[:, :n]@mass))) for p in paths)
    coi = max(coi, *(float(np.max(np.abs(p[:, n:2*n]@mass))) for p in paths))
    if not math.isfinite(energy) or not math.isfinite(coi):
        raise FloatingPointError("A09 physical diagnostics are nonfinite")
    return {"energy_identity_max_error": energy, "coi_max_error": coi,
            "qualified": energy <= 1e-8 and coi <= 1e-10}


def compare_source_scores(diagnostics, source_rows):
    """Require the whole .01 s event queue and source values; never drop failures."""
    if len(diagnostics) != 258 or set(diagnostics) != set(source_rows):
        return {"qualified": False, "reason": "source event queue incomplete"}
    errors = []
    for event_id, expected in source_rows.items():
        actual = diagnostics[event_id]
        if actual.get("status") != "ok":
            errors.append({"event_id": event_id, "reason": "prediction failed"})
            continue
        for horizon in ("1.0", "3.0"):
            row = actual["diagnostic"]["horizon_scores"][horizon]
            for key, source_key in (("normalized_mse", "normalized_mse"),
                    ("normalized_delta_mse_contribution", "normalized_delta_mse_contribution"),
                    ("normalized_omega_mse_contribution", "normalized_omega_mse_contribution"),
                    ("physical_delta_mse_rad2", "delta_mse_rad2"),
                    ("physical_omega_mse_rad2_per_s2", "omega_mse_rad2_per_s2")):
                pair = (row.get(key), expected.get(horizon, {}).get(source_key))
                if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in pair):
                    errors.append({"event_id": event_id, "horizon": horizon, "metric": source_key,
                                   "reason": "missing or invalid source/prediction score"})
                    continue
                difference = abs(pair[0]-pair[1])
                if difference > 1e-12:
                    errors.append({"event_id": event_id, "horizon": horizon, "metric": source_key,
                                   "absolute_difference": difference})
    return {"qualified": not errors, "mismatches": errors, "absolute_tolerance": 1e-12}


def summarize_audit(event_ids, metadata, rows, refinements, source_checks, *, budget_complete):
    all_valid = (budget_complete and all(v["qualified"] for v in source_checks.values())
        and len(rows) == 1548 and len(refinements) == 774
        and all(r["status"] == "ok" and r["physical"]["qualified"] for r in rows.values())
        and all(r["qualified"] for r in refinements.values()))
    supported, per_seed = [], {}
    for seed in SEEDS:
        fine = [rows.get((seed, .005, event_id), {}) for event_id in event_ids]
        positive = [r for r in fine if r.get("status") == "ok" and r["diagnostic"]["observed_rule_hit"]]
        per_seed[str(seed)] = {"observed_positive_new_rule_count": len(positive),
            "observed_positive_not_preserved_event_ids": [r["event_id"] for r in positive
                if not r["diagnostic"]["prediction_rule_hit"]],
            "predicted_trigger_without_observed_trigger_count": sum(
                r.get("status") == "ok" and r["diagnostic"]["prediction_rule_hit"] is True
                and r["diagnostic"]["observed_rule_hit"] is False for r in fine)}
    for event_id in event_ids:
        qualifies = True
        for seed in SEEDS:
            for step in STEPS:
                row = rows.get((seed, step, event_id), {})
                if row.get("status") != "ok":
                    qualifies = False
                    continue
                d = row["diagnostic"]
                qualifies &= (d["history_rho"]["truth"]["triggered"] is False
                    and d["future_rho"]["truth"]["rho"] > 1e-12
                    and d["future_rho"]["prediction"]["triggered"] is False)
                if step == .005:
                    excess = d["representative_window_excess_over_event_mean_C_minus_pM"]
                    qualifies &= excess is not None and excess > 1e-12
        if qualifies:
            supported.append({"event_id": event_id, **metadata[event_id]})
    distinct_lines = sorted({r["fault_line"] for r in supported})
    return {"audit_all_checks_passed": bool(all_valid),
        "scientific_verdict": "ACCEPT" if all_valid else "REJECT",
        "promotion_decision": "KEEP_BASELINE", "model_promoted": False,
        "per_seed_fine_rule_diagnostics": per_seed,
        "time_local_error_evidence_events": supported, "distinct_physical_fault_lines": distinct_lines,
        "time_aggregation_proposal_supported": bool(all_valid and len(supported) >= 2 and len(distinct_lines) >= 2),
        "new_training_authorized": False, "nonlinear_memory_supported_by_this_audit": False,
        "classification_fsr_computed": False, "selected_candidate": None}


def weighted_trajectory_summary(event_ids, event_weights, counts, cohort):
    if any(cohort[e]["status"] != "ok" for e in event_ids):
        return {"complete_population": False, "failed_event_ids": [e for e in event_ids
            if cohort[e]["status"] != "ok"], "main_nrmse": None, "duration_buckets": None}
    main_mse = math.fsum(w*cohort[e]["diagnostic"]["event_mse"]
                        for w, e in zip(event_weights, event_ids, strict=True))
    horizons = {h: math.sqrt(math.fsum(w*cohort[e]["diagnostic"]["horizon_scores"][h]["normalized_mse"]
        for w, e in zip(event_weights, event_ids, strict=True))) for h in ("1.0", "3.0")}
    buckets = {}
    for name, predicate in (("through_1s", lambda n: n <= 100),
            ("between_1s_and_3s", lambda n: 100 < n < 300), ("scored_at_3s", lambda n: n == 300)):
        indices = [i for i, n in enumerate(counts) if predicate(n)]
        mass = math.fsum(event_weights[i] for i in indices)
        contribution = math.fsum(event_weights[i]*cohort[event_ids[i]]["diagnostic"]["event_mse"] for i in indices)
        buckets[name] = {"event_count": len(indices), "original_weight_mass": mass,
            "same_denominator_normalized_mse_contribution": contribution,
            "conditional_normalized_mse": contribution/mass if mass else None}
    if abs(math.fsum(b["same_denominator_normalized_mse_contribution"] for b in buckets.values())-main_mse) > 1e-12:
        raise ArithmeticError("A09 duration contributions do not reconstruct the original mean")
    return {"complete_population": True, "event_count": len(event_ids), "main_mse": main_mse,
            "main_nrmse": math.sqrt(main_mse), "horizon_nrmse": horizons, "duration_buckets": buckets}


def _npz(path, **arrays):
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def run(*, root, experiment, stage, result_dir, run_id):
    started = time.perf_counter()
    if stage != "train_mechanism":
        raise ValueError("A09 has only its registered train-mechanism stage")
    config_path = root/"configs/experiments"/experiment["config"]
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    approval = authorization(root, experiment, config)
    assert_clean(root)
    commit = git_head(root)
    runtime = {name: version(name) for name in ("numpy", "scipy", "torch", "h5py", "filelock")}
    if runtime != {"numpy": "2.5.2", "scipy": "1.18.0", "torch": "2.13.0+cpu",
                   "h5py": "3.16.0", "filelock": "3.32.2"}:
        raise RuntimeError("A09 requires the pinned A06/A08 CPU runtime")
    ledger = shared_budget_directory(root).parent/"a09_audit_budgets"/AUTH
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.mkdir(exist_ok=False)
    _write(ledger/"reserved.json", {"run_id": run_id, "source_commit": commit,
        "max_paths": 1548, "fits": 0, "retries_allowed": False})
    source = root/"results"/config["source_run"]
    source_manifest = verify_artifact_directory(source, strict=True,
        expected_manifest_sha256=config["source_manifest_sha256"])
    allocation_path = root/config["frozen_allocation_path"]
    if sha256_file(allocation_path) != config["frozen_allocation_sha256"]:
        raise ValueError("A09 frozen allocation changed")
    allocation = json.loads(allocation_path.read_text(encoding="utf-8"))
    allocated = [r for r in allocation["rows"] if r["role"] == "train"]
    if len(allocated) != 268:
        raise ValueError("A09 requires all original train allocations")
    source_root = resolve_source_root(root, config["source_root"])
    bindings = allocation["source_bindings"]
    integrity = {"source_artifact_files": source_manifest, "source_artifact_root": source.as_posix(),
        "allocation_path": config["frozen_allocation_path"], "allocation_sha256": sha256_file(allocation_path),
        "raw_source_files": bindings, "raw_source_root": source_root.as_posix(),
        "purpose": "full-file SHA byte reads only; no holdout, label or future-input decoding"}
    for name, values in bindings.items():
        for relative, digest in values.items():
            if sha256_file(source_root/name/relative) != digest:
                raise ValueError("A09 source integrity mismatch before decoded access")
    provider = KikiHistorySamples(source_root, allocated, bindings, history_count=11, power_index=10)
    mass, samples = provider.load("train")
    if len(samples) != 258 or len(provider.coverage) != 268:
        raise ValueError("A09 eligible train queue differs")
    event_ids = [s.event_id for s in samples]
    metadata = {s.event_id: {"source_dataset": s.source_dataset, "fault_line": s.fault_line} for s in samples}
    counts = np.array([len(s.forecast_time)-1 for s in samples], dtype=np.int64)
    if int(counts.sum()) != 72147:
        raise ValueError("A09 total target-point budget changed")
    for s, count in zip(samples, counts, strict=True):
        if not 1 <= count <= 300:
            raise ValueError("A09 target count outside its real available range")
        for step, multiplier in ((.01, 1), (.005, 2)):
            if (integration_substeps(s.history.time-s.history.time[0], step) != 10*multiplier
                    or integration_substeps(s.forecast_time, step) != count*multiplier):
                raise ValueError("A09 original-clock ceil substep count exceeds its frozen budget")
    models, source_rows, source_scores = {}, {}, {}
    for index, seed in enumerate(SEEDS):
        prefix = f"checkpoint-240-{seed}"
        if sha256_file(source/prefix/"model.npz") != config["source_checkpoint_sha256"][index]:
            raise ValueError("A09 source checkpoint changed")
        models[seed] = LearnedNonlinearModel.load(source/prefix/"model.npz")
        model = models[seed]
        if (model.settings.latent_dimension != 2 or model.settings.frames != 3
                or not np.array_equal(model.closure.mass, mass)
                or not np.array_equal(model.target_scale, visible_scales(samples))):
            raise ValueError("A09 source model, mass or frozen scale differs")
        score = json.loads((source/prefix/"train_scores.json").read_text(encoding="utf-8"))
        source_scores[seed] = score
        by_h = {h: {r["event_id"]: r for r in score["available"]["horizons"][h]["rows"]}
                for h in ("1.0", "3.0")}
        if any(list(by_h[h]) != event_ids for h in by_h):
            raise ValueError("A09 original train-score event order changed")
        source_rows[seed] = {event_id: {h: by_h[h][event_id] for h in by_h} for event_id in event_ids}
        for s, count in zip(samples, counts, strict=True):
            for h, expected_count in (("1.0", min(100, count)), ("3.0", count)):
                row = by_h[h][s.event_id]
                metric_keys = ("normalized_mse", "normalized_delta_mse_contribution",
                    "normalized_omega_mse_contribution", "delta_mse_rad2", "omega_mse_rad2_per_s2")
                if (row["status"] != "ok" or row["role"] != "train"
                        or row["source_dataset"] != s.source_dataset or row["fault_line"] != s.fault_line
                        or row["scored_real_future_points"] != expected_count
                        or abs(row["actual_duration_s"]-.01*expected_count) > 1e-12
                        or any(type(row.get(k)) not in (int, float) or not math.isfinite(row[k])
                            or row[k] < 0 for k in metric_keys)):
                    raise ValueError("A09 source real scored target clock differs")
    code_paths = ["AGENTS.md", "configs/research_policy.json", "configs/project_manifest.json",
        config_path.relative_to(root).as_posix(), "configs/workflows/kiki_a09_train_mechanism_v1.json",
        config["development_protocol_path"], config["design_path"], "configs/input_contracts.json",
        "src/kron_reduce/workflows/kiki_train_mechanism.py",
        "src/kron_reduce/evaluation/train_mechanism_audit.py", "src/kron_reduce/evaluation/audit_budget.py",
        "src/kron_reduce/models/nonlinear_latent.py", "src/kron_reduce/models/causal_latent.py",
        "src/kron_reduce/models/learned_latent.py", "src/kron_reduce/data/kiki_history.py",
        "src/kron_reduce/data/history_contract.py", "src/kron_reduce/physics/coordinates.py"]
    code_hashes = {p: sha256_file(root/p) for p in code_paths}
    queue_rows = [{"event_id": s.event_id, **metadata[s.event_id], "role": "train"} for s in samples]
    event_weights = weights(queue_rows)
    plan = {"approval": approval, "source_commit": commit, "source_sha256": code_hashes,
        "runtime": runtime, "train_event_ids": event_ids, "future_point_counts": counts.tolist(),
        "event_weights": event_weights, "allocated_train": 268, "eligible_train": 258,
        "fits": 0, "optimizer_updates": 0, "max_prediction_paths": 1548, "max_rollout_calls": 3096,
        "max_rk4_updates": 672543, "max_integration_rhs": 2690172,
        "max_physical_rhs": 451458, "max_total_rhs": 3141630,
        "representative_window": "truth maximum rho, earliest equal tie", "labels_decoded": False}
    _write(result_dir/"scientific_plan.json", plan)
    _write(ledger/"plan.json", plan)
    _write(result_dir/"budget_binding.json", {"authorization_id": AUTH, "ledger_plan_sha256": sha256_file(ledger/"plan.json"),
        "scientific_plan_sha256": sha256_file(result_dir/"scientific_plan.json"), "retries_allowed": False})
    _write(result_dir/"integrity_file_reads.json", integrity)
    _write(result_dir/"train_coverage.json", {"rows": provider.coverage})
    offsets = np.r_[0, np.cumsum(counts+1)]
    _npz(result_dir/"real_train_arrays.npz", history=np.stack([np.c_[s.history.delta, s.history.omega] for s in samples]),
        future=np.concatenate([np.c_[s.target_delta, s.target_omega] for s in samples]), offsets=offsets,
        equilibrium=np.stack([s.equilibrium_delta for s in samples]), mass=mass,
        target_scale=models[SEEDS[0]].target_scale)
    budget = AuditRhsBudget()
    rows, saved_paths, saved_histories, refinements, source_checks, weighted_summaries = {}, {}, {}, {}, {}, {}
    for seed in SEEDS:
        model = counted_model(models[seed], budget)
        for step in STEPS:
            key = "h010" if step == .01 else "h005"
            directory = result_dir/f"seed-{seed}-{key}"
            directory.mkdir()
            histories, futures, output = [], [], []
            for index, (s, count) in enumerate(zip(samples, counts, strict=True)):
                identity = {"seed": seed, "max_step": step, "event_id": s.event_id}
                slot = ledger/f"{seed}-{key}-{index:03d}"
                slot.mkdir(exist_ok=False)
                record = {**identity, "status": "failed", "physical": {"qualified": False}}
                history_path = np.full((11, 22), np.nan)
                future_path = np.full((int(count)+1, 22), np.nan)
                before = len(budget.records)
                multiplier = 1 if step == .01 else 2
                try:
                    with budget.segment(identity, "history_integration", 40*multiplier):
                        z, history_path = model.initialize(s.history, s.equilibrium_delta, max_step=step)
                    with budget.segment(identity, "future_integration", 4*int(count)*multiplier):
                        future_path = model.closure.rollout(s.history.delta[-1], s.history.omega[-1], z,
                            s.forecast_time, s.history.constant_power, s.equilibrium_delta, max_step=step)
                    with budget.segment(identity, "physical_checks", int(count)+12):
                        physical = path_physics(model, history_path, future_path, s)
                    diagnostic = audit_event(history=np.c_[s.history.delta, s.history.omega],
                        true_future=np.c_[s.target_delta, s.target_omega], pred_future=future_path[:, :20],
                        equilibrium_delta=s.equilibrium_delta, mass=mass,
                        omega_s=provider.static[s.source_dataset]["omega_s"], target_scale=model.target_scale)
                    record.update(status="ok", physical=physical, diagnostic=diagnostic)
                except (ValueError, ArithmeticError, RuntimeError) as error:
                    record.update(exception_type=type(error).__name__, reason=str(error))
                record["rhs_segments"] = budget.records[before:]
                _write(slot/"finished.json", record)
                rows[seed, step, s.event_id] = record
                histories.append(history_path)
                futures.append(future_path)
                output.append(record)
            _npz(directory/"paths.npz", history=np.stack(histories), future=np.concatenate(futures), offsets=offsets)
            _write(directory/"event_diagnostics.json", {"rows": output})
            saved_paths[seed, step] = futures
            saved_histories[seed, step] = histories
            weighted_summaries[f"{seed}-{step}"] = weighted_trajectory_summary(event_ids, event_weights, counts,
                {e: rows[seed, step, e] for e in event_ids})
            print(f"A09 seed={seed} step={step}: {len(output)} fixed train paths audited", flush=True)
        coarse = {event_id: rows[seed, .01, event_id] for event_id in event_ids}
        source_checks[str(seed)] = compare_source_scores(coarse, source_rows[seed])
        if source_checks[str(seed)]["qualified"]:
            main = math.fsum(w*coarse[e]["diagnostic"]["event_mse"] for w, e in zip(event_weights, event_ids, strict=True))
            expected = source_scores[seed]["available"]["multi_horizon_normalized_rmse"]**2
            source_checks[str(seed)].update(main_mse=main, source_main_mse=expected)
            reconstructed = weighted_summaries[f"{seed}-0.01"]
            if (abs(main-expected) > 1e-12
                    or abs(reconstructed["main_nrmse"]-math.sqrt(expected)) > 1e-12
                    or any(abs(reconstructed["horizon_nrmse"][h]
                        - source_scores[seed]["available"]["horizons"][h]["normalized_rmse"]) > 1e-12
                        for h in ("1.0", "3.0"))):
                source_checks[str(seed)]["qualified"] = False
        for index, event_id in enumerate(event_ids):
            left, right = rows[seed, .01, event_id], rows[seed, .005, event_id]
            refined = {"seed": seed, "event_id": event_id, "qualified": False}
            if left["status"] == right["status"] == "ok":
                lp, rp = saved_paths[seed, .01][index], saved_paths[seed, .005][index]
                maximum = float(np.max(np.abs((lp[:, :20]-rp[:, :20])/model.target_scale)))
                lr = left["diagnostic"]["future_rho"]["prediction"]
                rr = right["diagnostic"]["future_rho"]["prediction"]
                same = lr["triggered"] == rr["triggered"]
                time_ok = not (same and lr["triggered"]) or abs(lr["first_trigger_end_index"]-rr["first_trigger_end_index"]) <= 2
                sample = samples[index]
                margin_comparison = audit_event(history=np.c_[sample.history.delta, sample.history.omega],
                    true_future=lp[:, :20], pred_future=rp[:, :20], equilibrium_delta=sample.equilibrium_delta,
                    mass=mass, omega_s=provider.static[sample.source_dataset]["omega_s"],
                    target_scale=model.target_scale)["future_rho"]
                history_maximum = float(np.max(np.abs((saved_histories[seed, .01][index][:, :20]
                    - saved_histories[seed, .005][index][:, :20])/model.target_scale)))
                refined.update(max_normalized_difference=maximum, same_rule_decision=same,
                    trigger_within_two_points=time_ok, coarse_rho=lr["rho"], fine_rho=rr["rho"],
                    absolute_rho_difference=abs(lr["rho"]-rr["rho"]),
                    history_initialization_max_normalized_difference=history_maximum,
                    coarse_fine_rule_margin_comparison=margin_comparison,
                    qualified=maximum <= 1e-5 and same and time_ok)
            refinements[seed, event_id] = refined
    snapshot = budget.snapshot()
    snapshot["successful_rk4_updates"] = sum(r["successful_rk4_updates"] or 0 for r in budget.records)
    complete = (len(rows) == 1548 and snapshot["integration_rhs"]["attempted"] == 2690172
        and snapshot["physical_rhs"]["attempted"] == 451458 and snapshot["rollout_calls_started"] == 3096
        and snapshot["attempted"] == snapshot["returned"] == 3141630 and snapshot["blocked"] == 0
        and snapshot["successful_rk4_updates"] == 672543
        and all(r["status"] == "complete" and r["attempted"] == r["returned"] == r["limit"] for r in budget.records))
    summary = summarize_audit(event_ids, metadata, rows, refinements, source_checks, budget_complete=complete)
    summary.update(process_status="PASS", run_id=run_id, budget_complete=complete, budget=snapshot,
        allocated_train=268, eligible_train=258, prediction_paths_reserved=1548, prediction_paths_finished=len(rows),
        fits=0, optimizer_updates=0, source_score_reconstruction=source_checks,
        weighted_trajectory_summaries=weighted_summaries,
        numerical_failures=[r for r in refinements.values() if not r["qualified"]],
        physical_or_path_failure_count=sum(r["status"] != "ok" or not r["physical"]["qualified"] for r in rows.values()),
        evidence_kind="uncertified_train_mechanism_audit", delivery_predictions_issued=0)
    _write(result_dir/"audit_summary.json", summary)
    _write(result_dir/"refinement_checks.json", {"rows": list(refinements.values())})
    _write(result_dir/"rhs_budget.json", {"snapshot": snapshot, "segments": budget.records})
    _write(result_dir/"data_access_audit.json", {**provider.access_audit(),
        "decoded_roles": ["train"], "decoded_labels_or_event_times": False,
        "allocate_roles_called": False, "truth_metadata_called": False,
        "split_metadata_reads": [f"{s}/splits.json" for s in sorted(bindings)],
        "static_h5_reads": ["inertia", "equilibrium/delta", "equilibrium/Pm", "attrs/omega_s",
            "attrs/mva_base", "attrs/raw_units", "attrs/state_mapping", "attrs/case_sha256"],
        "event_length_queries": [r["event_id"] for r in allocated],
        "file_integrity_reads_separately_recorded": "integrity_file_reads.json"})
    if git_head(root) != commit or status_lines(root) or any(sha256_file(root/p) != h for p, h in code_hashes.items()):
        raise RuntimeError("A09 implementation changed while the audit was running")
    if source_manifest != verify_artifact_directory(source, strict=True,
            expected_manifest_sha256=config["source_manifest_sha256"]):
        raise RuntimeError("A09 source evidence changed")
    for name, values in bindings.items():
        if any(sha256_file(source_root/name/p) != h for p, h in values.items()):
            raise RuntimeError("A09 raw source binding changed")
    _write(result_dir/"lineage.json", {"run_id": run_id, "source_commit": commit,
        "source_sha256": code_hashes, "wall_seconds": time.perf_counter()-started})
    _write(result_dir/"stage_evidence.json", {"source_commit": commit, "process_status": "PASS",
        "scientific_verdict": summary["scientific_verdict"], "promotion_decision": "KEEP_BASELINE",
        "data_policy": "kiki_original_train_only", "source_val_ids_read": [], "test_ids_read": [],
        "hidden_training_targets_read": False, "level_6_claimed": False,
        "development_authorization_id": AUTH, "evidence_kind": "uncertified_train_mechanism_audit",
        "certified_envelope_established": False, "deployment_predictions_allowed": False,
        "model_promoted": False, "claim_boundary": config["claim_boundary"]})
    _write(result_dir/"artifact_hashes.json", {p.relative_to(result_dir).as_posix(): sha256_file(p)
        for p in sorted(result_dir.rglob("*")) if p.is_file() and p.name != "artifact_hashes.json"})
    verify_artifact_directory(result_dir, strict=True)
