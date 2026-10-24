"""Fixed A06 fresh-Adam refinement of complete frozen A05 models."""
from __future__ import annotations

import json
import time
import tomllib
from dataclasses import asdict, replace

import numpy as np

from kron_reduce.data.kiki_history import KikiHistorySamples, allocate_roles, resolve_source_root
from kron_reduce.evaluation.recorded_error_audit import join_horizons
from kron_reduce.models.learned_latent import visible_scales
from kron_reduce.models.nonlinear_latent import LearnedNonlinearModel, fit_nonlinear_model
from kron_reduce.workflows.kiki_paired_training import additive_channel_scores
from kron_reduce.workflows.kiki_short_history import (
    SEEDS,
    _digest,
    candidate_settings,
    numerical_population,
    population_scores,
    registered_contract,
    score_fixed_predictions,
    shared_budget_directory,
    stability_population,
    truth_metadata,
)
from kron_reduce.workflows.kiki_stability_comparison import summarize
from kron_reduce.workflows.nonlinear_history_learning import _write
from project_harness.evidence import sha256_file, verify_artifact_directory
from project_harness.git_state import git_head, status_lines

AUTH = "kiki_a06_development_20261024"
STEPS = (80, 240)
FULL_ARRAYS = ("mass", "edge_stiffness", "damping", "coupling", "latent_skew",
               "latent_dissipation", "encoder", "feature_scale", "target_scale")


class RefinementBudget:
    def __init__(self, root, plan, run_id, result_dir):
        self.directory = shared_budget_directory(root).parent / "a06_fit_budgets" / AUTH
        self.directory.parent.mkdir(parents=True, exist_ok=True)
        self.directory.mkdir(exist_ok=False)
        self.plan_hash, self.run_id = _digest(plan), run_id
        _write(self.directory / "plan.json", {"run_id": run_id, "plan": plan,
               "scientific_plan_sha256": self.plan_hash})
        _write(result_dir / "budget_binding.json", {"authorization_id": AUTH,
               "scientific_plan_sha256": self.plan_hash,
               "ledger_plan_sha256": sha256_file(self.directory / "plan.json"),
               "max_fit_attempts": 3, "max_optimizer_steps": 720, "retries_allowed": False})

    def slot(self, seed):
        if seed not in SEEDS:
            raise ValueError("unregistered refinement seed")
        return self.directory / str(seed)

    def claim(self, seed):
        self.slot(seed).mkdir(exist_ok=False)
        _write(self.slot(seed) / "started.json", {"run_id": self.run_id, "reserved_optimizer_steps": 240})

    def progress(self, seed, step):
        slot = self.slot(seed)
        if type(step) is not int or not 1 <= step <= 240 or (slot / "finished.json").exists():
            raise ValueError("invalid refinement progress")
        if step > 1 and not (slot / f"step-{step-1:03d}.json").exists():
            raise ValueError("nonsequential refinement progress")
        _write(slot / f"step-{step:03d}.json", {"completed_optimizer_steps": step})

    def finish(self, seed, path):
        _write(self.slot(seed) / "finished.json", {"record_sha256": sha256_file(path)})

    def snapshot(self):
        rows = [{"seed": s, "reserved_optimizer_steps": 240,
                 "confirmed_optimizer_steps": len(list(self.slot(s).glob("step-*.json"))),
                 "finished": (self.slot(s)/"finished.json").exists()}
                for s in SEEDS if self.slot(s).exists()]
        return {"slots": rows, "fit_attempts_reserved": len(rows),
                "optimizer_steps_reserved": 240*len(rows),
                "confirmed_optimizer_steps": sum(r["confirmed_optimizer_steps"] for r in rows),
                "scientific_plan_sha256": self.plan_hash}


def authorization(root, experiment, config):
    policy = json.loads((root / "configs/research_policy.json").read_text(encoding="utf-8"))
    approval = policy["development_evaluations"].get(AUTH, {})
    expected = {"status": "user_approved", "authority": "explicit_user_reply",
                "experiment": "paper_kiki_a06_refinement", "workflow": "kiki_a06_refinement_v1",
                "stage": "joint_refinement", "dataset": "kiki_ieee39_v1",
                "data_policy": "kiki_original_train_only", "input_contract": "measured_history_100ms_zoh_v1",
                "seeds": list(SEEDS), "optimizer_steps_per_fit": 240,
                "max_fit_attempts": 3, "max_optimizer_steps": 720,
                "source_val_payload_allowed": False, "source_test_payload_allowed": False,
                "deployment_predictions_allowed": False, "model_promotion_allowed": False}
    if config.get("development_authorization_id") != AUTH or any(
        type(approval.get(k)) is not type(v) or approval.get(k) != v for k, v in expected.items()
    ):
        raise PermissionError("A06 authorization does not bind this experiment")
    manifest = json.loads((root / "configs/project_manifest.json").read_text(encoding="utf-8"))
    if manifest["experiments"].get(expected["experiment"]) != experiment:
        raise PermissionError("A06 experiment identity mismatch")
    for path_key, sha_key in (("development_protocol_path", "approval_protocol_sha256"),
                              ("design_path", "design_sha256")):
        approval_key = "approval_protocol_path" if path_key == "development_protocol_path" else path_key
        if (config[path_key] != approval[approval_key] or config[sha_key] != approval[sha_key]
            or sha256_file(root / config[path_key]) != config[sha_key]):
            raise PermissionError("A06 protocol or design binding changed")
    fixed = {"updates_per_fit": 240, "checkpoints_at_updates": list(STEPS), "primary_checkpoint_update": 240,
             "learning_rate": .01, "encoder_penalty": .0001, "history_weight": 0.,
             "seeds": list(SEEDS), "split_seed": 324, "optimizer": "Adam", "optimizer_state": "fresh",
             "dtype": "float64", "gradient_clip": 10., "latent_dimension": 2, "encoder_frames": 3,
             "max_fit_attempts": 3, "max_optimizer_updates": 720,
             "source_checkpoint_paths": [f"candidate-J-{seed}/model.npz" for seed in SEEDS],
             "sources": ["ieee39_postfault", "ieee39_expansion"]}
    if any(config.get(k) != v for k, v in fixed.items()):
        raise ValueError("A06 fixed optimization plan changed")
    for key in ("source_run", "source_manifest_sha256", "frozen_allocation_sha256", "source_checkpoint_sha256"):
        if config[key] != approval[key]:
            raise PermissionError(f"A06 frozen source changed: {key}")
    return approval


def outcome(records, numerics, stability, baselines, *, fit_complete=False):
    terminal = {r["seed"]: r for r in records if r["checkpoint_step"] == 240}
    pairs = []
    for seed in SEEDS:
        row, baseline = terminal.get(seed, {}), baselines[seed]
        los = stability.get((240, seed), {}).get("summary", {})
        finite = row.get("val_main_nrmse") is not None and row.get("val_complete3_nrmse") is not None
        numeric = numerics.get((240, seed), {})
        replay_error = numeric.get("batched_direct_max_normalized_difference")
        all_trajectory = (len(numeric.get("rows", [])) == 88 and all(r["qualified"] for r in numeric["rows"])
                          and numeric.get("checkpoint_replay_exact") is True and replay_error is not None
                          and np.isfinite(replay_error) and replay_error <= 1e-10)
        all_los = len(stability.get((240, seed), {}).get("rows", [])) == 88 and all(
            r["qualified"] for r in stability[(240, seed)]["rows"])
        coverage = (los.get("eligible") == 88 and los.get("eligible_unstable") == 14
                    and los.get("eligible_stable") == 74 and los.get("numerical_unresolved") == 0
                    and los.get("development_evaluation_refusals") == 0)
        pairs.append({"seed": seed, "main_ratio": row["val_main_nrmse"]/baseline["main"] if finite else None,
                      "complete3_ratio": row["val_complete3_nrmse"]/baseline["complete3"] if finite else None,
                      "fn": los.get("fn"), "fp": los.get("fp"),
                      "baseline_fn": baseline["fn"], "baseline_fp": baseline["fp"],
                      "all_numerics": bool(all_trajectory and all_los), "full_eligible_coverage": coverage})
    valid = fit_complete and len(terminal) == 3 and all(p["main_ratio"] is not None for p in pairs)
    safeguards = valid and all(p["all_numerics"] and p["full_eligible_coverage"]
        and p["main_ratio"] <= 1. and p["complete3_ratio"] <= 1.05
        and p["fn"] <= p["baseline_fn"] and p["fp"] <= p["baseline_fp"] for p in pairs)
    zero = bool(safeguards and all(p["fn"] == 0 and p["fp"] == 0 for p in pairs))
    strict_fn_gain = bool(safeguards and all(p["fn"] < p["baseline_fn"] for p in pairs))
    return {"paired_results": pairs, "all_terminal_safeguards": bool(safeguards),
            "all_seed_strict_fn_gain": strict_fn_gain, "zero_missed_instability_on_eligible_val": zero,
            "zero_missed_instability_on_all_allocated_events": False,
            "scientific_verdict": "ACCEPT" if zero else "RETRY" if strict_fn_gain else "REJECT",
            "promotion_decision": "KEEP_BASELINE", "model_promoted": False,
            "primary_checkpoint_step": 240, "intermediate_checkpoint_selected": False}


def run(*, root, experiment, stage, result_dir, run_id):
    started, commit = time.perf_counter(), git_head(root)
    if stage != "joint_refinement" or status_lines(root):
        raise RuntimeError("A06 needs a clean committed checkout and registered stage")
    config_path = root / "configs/experiments" / experiment["config"]
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    approval = authorization(root, experiment, config)
    contract = registered_contract(root, config)
    source_run = root / "results" / config["source_run"]
    source_manifest = verify_artifact_directory(source_run, strict=True,
        expected_manifest_sha256=config["source_manifest_sha256"])
    source_plan = json.loads((source_run/"scientific_plan.json").read_text(encoding="utf-8"))
    allocation_path = root / config["frozen_allocation_path"]
    if sha256_file(allocation_path) != config["frozen_allocation_sha256"]:
        raise ValueError("A06 allocation changed")
    allocation = json.loads(allocation_path.read_text(encoding="utf-8"))
    starts, baselines = {}, {}
    for seed in SEEDS:
        prefix = f"candidate-J-{seed}"
        path = source_run/prefix/"model.npz"
        if sha256_file(path) != config["source_checkpoint_sha256"][list(SEEDS).index(seed)]:
            raise ValueError("A06 checkpoint binding changed")
        with np.load(path, allow_pickle=False) as data:
            starts[seed] = {key: data[key].copy() for key in FULL_ARRAYS}
        score = json.loads((source_run/prefix/"val_scores.json").read_text(encoding="utf-8"))
        los = json.loads((source_run/prefix/"stability.json").read_text(encoding="utf-8"))["summary"]
        baselines[seed] = {"main": score["available"]["multi_horizon_normalized_rmse"],
                           "complete3": score["complete3_common"]["multi_horizon_normalized_rmse"],
                           "fn": los["fn"], "fp": los["fp"]}
    paths = ["AGENTS.md", "configs/research_policy.json", "configs/project_manifest.json",
             config_path.relative_to(root).as_posix(), "configs/workflows/kiki_a06_refinement_v1.json",
             config["development_protocol_path"], config["design_path"], "configs/input_contracts.json",
             "src/kron_reduce/workflows/kiki_joint_refinement.py", "src/kron_reduce/workflows/kiki_paired_training.py",
             "src/kron_reduce/workflows/kiki_short_history.py", "src/kron_reduce/workflows/kiki_history_learning.py",
             "src/kron_reduce/workflows/kiki_stability_comparison.py", "src/kron_reduce/models/nonlinear_latent.py",
             "src/kron_reduce/models/learned_latent.py", "src/kron_reduce/models/causal_latent.py",
             "src/kron_reduce/data/kiki_history.py", "src/kron_reduce/data/history_contract.py",
             "src/kron_reduce/physics/coordinates.py"]
    code_hashes = {p: sha256_file(root/p) for p in paths}
    source_root = resolve_source_root(root, config["source_root"])
    rows, bindings = allocate_roles(source_root, config["sources"], 324)
    if rows != allocation["rows"] or bindings != allocation["source_bindings"]:
        raise ValueError("A06 source allocation changed")
    provider = KikiHistorySamples(source_root, rows, bindings, history_count=11, power_index=10)
    mass, train = provider.load("train")
    if len(train) != 258 or [s.event_id for s in train] != source_plan["train_event_ids"] or any(
        not np.array_equal(mass, a["mass"]) or not np.array_equal(visible_scales(train), a["target_scale"])
        for a in starts.values()):
        raise ValueError("A06 population, mass or scales changed")
    steps = 3*240*(258*10+sum(len(s.forecast_time)-1 for s in train))
    if steps != 53_803_440:
        raise ValueError("A06 exact training integration budget changed")
    plan = {"approval": approval, "source_commit": commit, "source_sha256": code_hashes,
            "train_event_ids": [s.event_id for s in train], "val_event_ids": source_plan["val_event_ids"],
            "source_bindings": bindings, "baselines": baselines,
            "source_manifest_sha256": config["source_manifest_sha256"],
            "training_rk4_event_steps": steps, "training_rhs_event_evaluations": 4*steps,
            "checkpoint_steps": list(STEPS), "primary_checkpoint_step": 240,
            "optimizer_restart": "fresh_Adam_not_exact_resume", "settings": {
                str(s): asdict(replace(candidate_settings("B", s), epochs=240)) for s in SEEDS}}
    _write(result_dir/"scientific_plan.json", plan)
    _write(result_dir/"role_allocation.json", allocation)
    ledger = RefinementBudget(root, plan, run_id, result_dir)
    fit_records, checkpoints = [], {}
    for seed in SEEDS:
        ledger.claim(seed)
        fit_dir = result_dir/f"fit-{seed}"
        fit_dir.mkdir()
        def checkpoint(step, model, state, seed=seed):
            if step not in STEPS or (step, seed) in checkpoints:
                raise ValueError("unregistered or duplicate checkpoint")
            fit = model.fit_record
            if (fit["actual_optimizer_steps"] != step or fit["final_checkpoint_optimizer_step"] != step
                or fit["planned_optimizer_steps"] != 240 or len(fit["loss_trace"]) != step
                or fit["parameter_count"] != 238 or fit["optimizer_updated_scalars"] != 238
                or fit.get("exact_optimizer_resume") is not False or fit.get("initial_optimizer_step") != 0
                or fit.get("initial_optimizer_moments_zero") is not True
                or max(fit["full_start_roundtrip_errors"].values()) > 1e-12
                or not np.array_equal(model.feature_scale, starts[seed]["feature_scale"])
                or not np.array_equal(model.target_scale, starts[seed]["target_scale"])
                or any(float(state[f"adam_step__{name}"]) != step for name in ("raw", "port", "skew", "decay", "encoder"))):
                raise ValueError("A06 snapshot violates frozen recovery, optimizer or step contract")
            directory = result_dir/f"checkpoint-{step}-{seed}"
            directory.mkdir()
            model.save(directory/"model.npz")
            with (directory/"training_state.npz").open("xb") as stream:
                np.savez_compressed(stream, **state)
            _write(directory/"fit_record.json", model.fit_record)
            checkpoints[(step, seed)] = sha256_file(directory/"model.npz")
        print(f"A06 {seed}: 240 additional fixed updates; 80 diagnostic, 240 primary", flush=True)
        try:
            model = fit_nonlinear_model(train, mass, replace(candidate_settings("B", seed), epochs=240),
                target_scale_samples=train, allow_partial_horizons=True, full_start=starts[seed],
                checkpoint_steps=STEPS, checkpoint_callback=checkpoint,
                progress_callback=lambda step, s=seed: ledger.progress(s, step))
            if model.fit_record["actual_optimizer_steps"] != 240 or any((p, seed) not in checkpoints for p in STEPS):
                raise ValueError("A06 incomplete final model or checkpoint snapshots")
            record = {"seed": seed, "status": "completed", "actual_optimizer_steps": 240,
                      "fit_record": model.fit_record}
        except (ValueError, FloatingPointError, OverflowError, RuntimeError) as error:
            record = {"seed": seed, "status": "failed", "reason": f"{type(error).__name__}: {error}",
                      "confirmed_optimizer_steps": len(list(ledger.slot(seed).glob("step-*.json")))}
        _write(fit_dir/"fit_record.json", record)
        ledger.finish(seed, fit_dir/"fit_record.json")
        fit_records.append(record)
    _write(result_dir/"budget_snapshot.json", ledger.snapshot())
    _write(result_dir/"training_freeze.json", {"fit_records": fit_records,
        "checkpoint_hashes": {f"{step}-{seed}": h for (step, seed), h in checkpoints.items()},
        "val_payloads_opened": False})
    _, val = provider.load("val")
    if len(val) != 88 or [s.event_id for s in val] != source_plan["val_event_ids"]:
        raise ValueError("A06 val population changed after training freeze")
    records, numerics, scores = [], {}, {}
    for (step, seed), checkpoint_sha in checkpoints.items():
        directory = result_dir/f"checkpoint-{step}-{seed}"
        model = LearnedNonlinearModel.load(directory/"model.npz")
        record = {"checkpoint_step": step, "seed": seed, "checkpoint_sha256": checkpoint_sha}
        for role, population in (("train", train), ("val", val)):
            try:
                score, predictions = population_scores(model, population, model.target_scale)
                additive_channel_scores(score, predictions, population, model.target_scale)
            except (ValueError, FloatingPointError, OverflowError, RuntimeError) as error:
                predictions = {s.event_id: str(error) for s in population}
                score = score_fixed_predictions(predictions, population, model.target_scale)
            _write(directory/f"{role}_scores.json", score)
            if role == "val":
                scores[(step, seed)] = score
                record.update(val_main_nrmse=score["available"]["multi_horizon_normalized_rmse"],
                              val_complete3_nrmse=score["complete3_common"]["multi_horizon_normalized_rmse"])
                try:
                    first = val[0]
                    direct = model.predict(first.history, first.forecast_time, first.equilibrium_delta)
                    restored = LearnedNonlinearModel.load(directory/"model.npz")
                    replay = restored.predict(first.history, first.forecast_time, first.equilibrium_delta)
                    diff = float(np.max(abs(direct[:, :20]-predictions[first.event_id])/model.target_scale))
                    if not np.isfinite(diff):
                        raise FloatingPointError("nonfinite replay")
                    numeric = {"rows": numerical_population(model, val, model.target_scale, contract),
                               "checkpoint_replay_exact": np.array_equal(direct, replay),
                               "batched_direct_max_normalized_difference": diff}
                except (ValueError, TypeError, FloatingPointError, OverflowError, RuntimeError) as error:
                    numeric = {"rows": [{"event_id": s.event_id, "qualified": False, "reason": str(error)} for s in val],
                               "checkpoint_replay_exact": False, "batched_direct_max_normalized_difference": None}
                numerics[(step, seed)] = numeric
                _write(directory/"trajectory_numerics.json", numeric)
        records.append(record)
    freeze = {"records": records, "fit_records": fit_records, "checkpoint_hashes": {
        f"{step}-{seed}": h for (step, seed), h in checkpoints.items()},
        "primary_checkpoint_step": 240, "selection_performed": False, "stability_evaluation_started": False}
    _write(result_dir/"comparison_freeze.json", freeze)
    freeze_sha = sha256_file(result_dir/"comparison_freeze.json")
    metadata = truth_metadata(provider)
    coverage = [{**r, **metadata[r["event_id"]], "prospective_eligible": r["history_eligible"] and (
        metadata[r["event_id"]]["true_event_time_record_s"] is None or
        metadata[r["event_id"]]["true_event_time_record_s"] > .1+1e-10)} for r in provider.coverage]
    _write(result_dir/"coverage.json", {"rows": coverage})
    labels = {s.event_id: metadata[s.event_id]["label"] for s in val}
    stability = {}
    for (step, seed), checkpoint_sha in checkpoints.items():
        directory = result_dir/f"checkpoint-{step}-{seed}"
        if sha256_file(directory/"model.npz") != checkpoint_sha:
            raise ValueError("A06 model changed after freeze")
        model = LearnedNonlinearModel.load(directory/"model.npz")
        score = scores[(step, seed)]
        if score["available"]["multi_horizon_normalized_rmse"] is not None:
            _, decomp = join_horizons(score, labels)
            _write(directory/"error_decomposition.json", decomp)
        try:
            los, arrays = stability_population(model, val, metadata, provider.static, contract)
        except (ValueError, FloatingPointError, OverflowError, RuntimeError) as error:
            unresolved = [{"event_id": s.event_id, **metadata[s.event_id], "qualified": False,
                "prospective_eligible": True, "predicted_los": False, "window_settled": False,
                "failure_reason": str(error)} for s in val]
            los, arrays = {"rows": unresolved, "summary": {**summarize(unresolved),
                "development_evaluation_refusals": 0}}, {}
        los.update(comparison_freeze_sha256=freeze_sha)
        los["summary"].update(allocated_events=101, allocated_unstable=27, allocated_stable=74,
                              unstable_eligibility_coverage=14/27,
                              assigned_unstable_detection_fraction=los["summary"]["tp"]/27)
        _write(directory/"stability.json", los)
        with (directory/"stability_paths.npz").open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        stability[(step, seed)] = los
        print(f"A06 step={step} seed={seed} val={score['available']['multi_horizon_normalized_rmse']} FN={los['summary']['fn']} FP={los['summary']['fp']}", flush=True)
    complete = (len(fit_records) == 3 and all(r["status"] == "completed" and r["actual_optimizer_steps"] == 240
                    for r in fit_records) and len(checkpoints) == 6
                and ledger.snapshot()["confirmed_optimizer_steps"] == 720)
    verdict = outcome(records, numerics, stability, baselines, fit_complete=complete)
    summary = {**verdict, "process_status": "PASS" if complete else "FAIL", "records": records,
               "fit_records": fit_records, "budget": ledger.snapshot(),
               "train_event_count": 258, "val_event_count": 88, "allocated_train": 268, "allocated_val": 101,
               "evidence_kind": "uncertified_development_evaluation", "delivery_predictions_issued": 0}
    _write(result_dir/"learning_summary.json", summary)
    _write(result_dir/"data_access_audit.json", {**provider.access_audit(),
        "labels_and_event_times_read_only_after_comparison_freeze": True})
    if (git_head(root) != commit or status_lines(root) or any(sha256_file(root/p) != h for p, h in code_hashes.items())
        or sha256_file(result_dir/"comparison_freeze.json") != freeze_sha
        or any(sha256_file(result_dir/f"checkpoint-{s}-{seed}/model.npz") != h for (s, seed), h in checkpoints.items())):
        raise RuntimeError("A06 implementation or checkpoint changed")
    for source, binding in bindings.items():
        if any(sha256_file(source_root/source/p) != h for p, h in binding.items()):
            raise RuntimeError("A06 source data changed")
    if any(sha256_file(source_run/p) != h for p, h in source_manifest.items()):
        raise RuntimeError("A06 inherited evidence changed")
    _write(result_dir/"lineage.json", {"run_id": run_id, "source_commit": commit,
        "source_sha256": code_hashes, "wall_seconds": time.perf_counter()-started})
    _write(result_dir/"stage_evidence.json", {"source_commit": commit, "process_status": summary["process_status"],
        "scientific_verdict": summary["scientific_verdict"], "promotion_decision": "KEEP_BASELINE",
        "data_policy": "kiki_original_train_only", "source_val_ids_read": [], "test_ids_read": [],
        "hidden_training_targets_read": False, "level_6_claimed": False, "development_authorization_id": AUTH,
        "evidence_kind": "uncertified_development_evaluation", "certified_envelope_established": False,
        "deployment_predictions_allowed": False, "model_promoted": False, "claim_boundary": config["claim_boundary"]})
    _write(result_dir/"artifact_hashes.json", {p.relative_to(result_dir).as_posix(): sha256_file(p)
        for p in sorted(result_dir.rglob("*")) if p.is_file() and p.name != "artifact_hashes.json"})
    verify_artifact_directory(result_dir, strict=True)
    if not complete:
        raise RuntimeError("A06 incomplete fit retained; no additional fits allowed")
