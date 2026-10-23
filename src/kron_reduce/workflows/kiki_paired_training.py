"""One authorized A05 paired intervention; no retries or model promotion."""
from __future__ import annotations

import json
import time
import tomllib
from dataclasses import asdict

import numpy as np

from kron_reduce.data.kiki_history import KikiHistorySamples, allocate_roles, resolve_source_root
from kron_reduce.evaluation.recorded_error_audit import join_horizons, paired_contributions
from kron_reduce.models.learned_latent import visible_scales
from kron_reduce.models.nonlinear_latent import LearnedNonlinearModel, fit_nonlinear_model
from kron_reduce.workflows.kiki_short_history import (
    SEEDS,
    _digest,
    candidate_settings,
    numerical_population,
    population_scores,
    real_evaluation_prerequisite,
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

AUTH = "kiki_a05_development_20261024"
GROUPS = ("F", "J")
PROPOSAL = "configs/proposals/kiki_a05_freeze_vs_joint.json"


def diagnostic_verdict(supported):
    return "ACCEPT" if supported else "REJECT"


class PairedBudget:
    """An exclusive campaign and exclusive fit slots survive process interruption."""

    def __init__(self, root, plan, run_id, result_dir):
        self.directory = shared_budget_directory(root).parent / "a05_fit_budgets" / AUTH
        self.directory.parent.mkdir(parents=True, exist_ok=True)
        # A previous campaign, including an interrupted campaign, cannot be replayed.
        self.directory.mkdir(exist_ok=False)
        self.plan_hash, self.run_id = _digest(plan), run_id
        _write(self.directory / "plan.json", {"run_id": run_id, "plan": plan,
               "scientific_plan_sha256": self.plan_hash})
        _write(result_dir / "budget_binding.json", {"authorization_id": AUTH,
               "scientific_plan_sha256": self.plan_hash,
               "ledger_plan_sha256": sha256_file(self.directory / "plan.json"),
               "max_fit_attempts": 6, "max_optimizer_steps": 480, "retries_allowed": False})

    def slot(self, group, seed):
        if group not in GROUPS or seed not in SEEDS:
            raise ValueError("unregistered paired fit slot")
        return self.directory / f"{group}-{seed}"

    def claim(self, group, seed):
        slot = self.slot(group, seed)
        slot.mkdir(exist_ok=False)
        _write(slot / "started.json", {"run_id": self.run_id, "reserved_optimizer_steps": 80})

    def progress(self, group, seed, epoch):
        slot = self.slot(group, seed)
        if type(epoch) is not int or not 1 <= epoch <= 80 or (slot / "finished.json").exists():
            raise ValueError("invalid optimizer progress")
        if epoch > 1 and not (slot / f"step-{epoch-1:03d}.json").exists():
            raise ValueError("nonsequential optimizer progress")
        _write(slot / f"step-{epoch:03d}.json", {"completed_optimizer_steps": epoch})

    def finish(self, group, seed, record_path):
        _write(self.slot(group, seed) / "finished.json", {
            "record_sha256": sha256_file(record_path), "run_id": self.run_id})

    def snapshot(self):
        rows = [{"group": g, "seed": s, "reserved_optimizer_steps": 80,
                 "confirmed_optimizer_steps": len(list(self.slot(g, s).glob("step-*.json"))),
                 "finished": (self.slot(g, s) / "finished.json").exists()}
                for g in GROUPS for s in SEEDS if self.slot(g, s).exists()]
        return {"slots": rows, "fit_attempts_reserved": len(rows),
                "optimizer_steps_reserved": 80 * len(rows),
                "confirmed_optimizer_steps": sum(r["confirmed_optimizer_steps"] for r in rows),
                "scientific_plan_sha256": self.plan_hash}


def authorization(root, experiment, config):
    policy = json.loads((root / "configs/research_policy.json").read_text(encoding="utf-8"))
    approval = policy["development_evaluations"].get(AUTH, {})
    expected = {"status": "user_approved", "authority": "explicit_user_reply",
                "experiment": "paper_kiki_a05_paired", "workflow": "kiki_a05_paired_v1",
                "stage": "paired_training", "dataset": "kiki_ieee39_v1",
                "data_policy": "kiki_original_train_only",
                "input_contract": "measured_history_100ms_zoh_v1", "split_seed": 324,
                "groups": list(GROUPS), "seeds": list(SEEDS),
                "optimizer_steps_per_fit": 80, "max_fit_attempts": 6, "max_optimizer_steps": 480,
                "source_val_payload_allowed": False, "source_test_payload_allowed": False,
                "deployment_predictions_allowed": False, "model_promotion_allowed": False,
                "independent_confirmation_required": True,
                "frozen_allocation_sha256": config["frozen_allocation_sha256"]}
    if config.get("development_authorization_id") != AUTH or any(
        type(approval.get(k)) is not type(v) or approval.get(k) != v for k, v in expected.items()
    ):
        raise PermissionError("A05 authorization does not bind this paired experiment")
    manifest = json.loads((root / "configs/project_manifest.json").read_text(encoding="utf-8"))
    if manifest["experiments"].get(expected["experiment"]) != experiment:
        raise PermissionError("A05 experiment identity mismatch")
    protocol = approval["approval_protocol_path"]
    if (protocol != config["development_protocol_path"]
        or sha256_file(root / protocol) != approval["approval_protocol_sha256"]
        or config["approval_protocol_sha256"] != approval["approval_protocol_sha256"]):
        raise PermissionError("A05 authorization protocol changed")
    proposal = json.loads((root / PROPOSAL).read_text(encoding="utf-8"))
    if (config["proposal_path"] != PROPOSAL or approval["proposal_path"] != PROPOSAL
        or sha256_file(root / PROPOSAL) != config["proposal_sha256"]
        or config["proposal_sha256"] != approval["proposal_sha256"]):
        raise PermissionError("A05 frozen proposal binding changed")
    for key in ("core_design_path", "core_design_sha256", "source_run", "source_manifest_sha256",
                "shared_checkpoint", "shared_checkpoint_sha256", "frozen_allocation_sha256"):
        if config[key] != proposal[key] or approval[key] != proposal[key]:
            raise PermissionError(f"A05 source binding changed: {key}")
    if sha256_file(root / proposal["core_design_path"]) != proposal["core_design_sha256"]:
        raise ValueError("Astra core design changed")
    if (config["epochs"], config["learning_rate"], config["encoder_penalty"],
        config["split_seed"], config["seeds"], config["sources"]) != (
        80, 0.01, 1e-4, 324, list(SEEDS), ["ieee39_postfault", "ieee39_expansion"]
    ):
        raise ValueError("A05 frozen configuration changed")
    extra = {"groups": list(GROUPS), "history_weight": 0., "optimizer": "Adam", "dtype": "float64",
             "gradient_clip": 10., "gradient_clip_population": "complete_parameter_vector_before_F_backbone_update_mask",
             "latent_dimension": 2, "encoder_frames": 3, "checkpoint_rule": "last_update_80",
             "max_fit_attempts": 6, "max_optimizer_steps": 480, "source_val_payload_allowed": False,
             "source_test_payload_allowed": False, "prediction_delivery_allowed": False,
             "model_promotion_allowed": False}
    if any(config.get(k) != v for k, v in extra.items()):
        raise ValueError("A05 training or evaluation boundary changed")
    return approval, proposal


def additive_channel_scores(scores, predictions, samples, scale):
    """Both channel contributions use the same 20-channel denominator."""
    sample_map = {s.event_id: s for s in samples}
    sections = [scores["complete1"], *[v for name in ("available", "complete3_common")
                                     for v in scores[name]["horizons"].values()]]
    for section in sections:
        for row in section["rows"]:
            if row["status"] != "ok":
                continue
            sample, count = sample_map[row["event_id"]], row["scored_real_future_points"]
            error = (predictions[sample.event_id][1:count+1] - np.c_[
                sample.target_delta[1:count+1], sample.target_omega[1:count+1]]) / scale
            row["normalized_delta_mse_contribution"] = float(np.mean(error[:, :10]**2) / 2)
            row["normalized_omega_mse_contribution"] = float(np.mean(error[:, 10:]**2) / 2)
            if not np.isclose(sum(row[k] for k in (
                "normalized_delta_mse_contribution", "normalized_omega_mse_contribution")),
                row["normalized_mse"], rtol=1e-12, atol=1e-12):
                raise ValueError("normalized channel contributions do not add to the joint MSE")


def judge(records, decompositions, numerics, stability):
    table = {(r["candidate"], r["seed"]): r for r in records}
    pairs = []
    for seed in SEEDS:
        f, j = table.get(("F", seed), {}), table.get(("J", seed), {})
        valid = all(r.get("status") == "completed" and r.get("actual_optimizer_steps") == 80
                    and r.get("val_main_nrmse") is not None for r in (f, j))
        pair = {"seed": seed, "valid": valid, "main_ratio_F_over_J": None,
                "complete3_ratio_F_over_J": None, "mechanism_direction": False,
                "fn_fp_nonworse": False}
        if valid and j["val_main_nrmse"] > 0 and j.get("val_complete3_nrmse", 0):
            pair["main_ratio_F_over_J"] = f["val_main_nrmse"] / j["val_main_nrmse"]
            pair["complete3_ratio_F_over_J"] = (
                f["val_complete3_nrmse"] / j["val_complete3_nrmse"]
                if f.get("val_complete3_nrmse") is not None else None)
        if ("F", seed) in decompositions and ("J", seed) in decompositions:
            fg, jg = [decompositions[(g, seed)]["groups"]["label"] for g in GROUPS]
            pair["mechanism_direction"] = bool(
                fg["stable"]["same_denominator_mse_contribution"] < jg["stable"]["same_denominator_mse_contribution"]
                and fg["unstable"]["same_denominator_mse_contribution"] <= jg["unstable"]["same_denominator_mse_contribution"])
        if all((g, seed) in stability for g in GROUPS):
            a, b = [stability[(g, seed)]["summary"] for g in GROUPS]
            pair["fn_fp_nonworse"] = bool(all(a[k] <= b[k] for k in ("fn", "fp")) and all(
                x["numerical_unresolved"] == 0 and x.get("development_evaluation_refusals") == 0
                and x.get("history_eligible") == 88 and x.get("eligible") == 88
                and x["tp"] + x["fn"] == 14 and x["tn"] + x["fp"] == 74
                for x in (a, b)))
        pairs.append(pair)
    ratios = [r["main_ratio_F_over_J"] for r in pairs]
    complete3 = [r["complete3_ratio_F_over_J"] for r in pairs]
    checks = {"all_six_fits_completed": len(table) == 6 and all(r["valid"] for r in pairs),
              "all_seed_mechanism_direction": all(r["mechanism_direction"] for r in pairs),
              "paired_main_gain": all(v is not None for v in ratios)
                  and max(ratios) <= 1 and float(np.median(ratios)) <= .90,
              "complete3_noninferiority": all(v is not None for v in complete3) and max(complete3) <= 1.05,
              "fn_fp_nonworse": all(r["fn_fp_nonworse"] for r in pairs),
              "all_trajectory_numerics": len(numerics) == 6 and all(
                  v["checkpoint_replay_exact"] and v["batched_direct_max_normalized_difference"] is not None
                  and np.isfinite(v["batched_direct_max_normalized_difference"])
                  and v["batched_direct_max_normalized_difference"] <= 1e-10
                  and len(v["rows"]) == 88 and all(r["qualified"] for r in v["rows"])
                  for v in numerics.values()),
              "all_8s_numerics": len(stability) == 6 and all(
                  len(v["rows"]) == 88 and all(r["qualified"] for r in v["rows"])
                  for v in stability.values())}
    return {"checks": checks, "paired_results": pairs,
            "mechanism_consistent_direction": checks["all_seed_mechanism_direction"],
            "next_stage_research_supported": all(checks.values()),
            "promotion_decision": "KEEP_BASELINE", "model_promoted": False,
            "certified_envelope_established": False, "deployment_predictions_allowed": False}


def run(*, root, experiment, stage, result_dir, run_id):
    started, commit = time.perf_counter(), git_head(root)
    if stage != "paired_training" or status_lines(root):
        raise RuntimeError("A05 requires its registered stage and a clean committed checkout")
    config_path = root / "configs/experiments" / experiment["config"]
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    approval, proposal = authorization(root, experiment, config)
    contract = registered_contract(root, config)
    source_run = root / "results" / proposal["source_run"]
    verify_artifact_directory(source_run, strict=True,
                              expected_manifest_sha256=proposal["source_manifest_sha256"])
    allocation_path = root / config["frozen_allocation_path"]
    if sha256_file(allocation_path) != proposal["frozen_allocation_sha256"]:
        raise ValueError("A05 allocation changed")
    allocation = json.loads(allocation_path.read_text(encoding="utf-8"))
    source_plan = json.loads((source_run / "scientific_plan.json").read_text(encoding="utf-8"))
    a_path, b_path = source_run / proposal["shared_checkpoint"], source_run / "candidate-B-324/model.npz"
    if sha256_file(a_path) != proposal["shared_checkpoint_sha256"] or sha256_file(b_path) != (
        "ac75db95c94588e1f09a2140a43df7aeef9161fa709800a07613cc450c027418"):
        raise ValueError("shared initialization checkpoint changed")
    with np.load(a_path, allow_pickle=False) as a, np.load(b_path, allow_pickle=False) as b:
        paired_start = {k: a[k].copy() for k in ("mass", "edge_stiffness", "damping", "target_scale")}
        paired_start["feature_scale"] = b["feature_scale"].copy()
        if not np.array_equal(a["target_scale"], b["target_scale"]):
            raise ValueError("inherited target scales differ")
    guard_paths = ["AGENTS.md", "configs/research_policy.json", "configs/project_manifest.json",
                   PROPOSAL, proposal["core_design_path"], config["development_protocol_path"],
                   config_path.relative_to(root).as_posix(), "configs/workflows/kiki_a05_paired_v1.json",
                   "configs/input_contracts.json", "src/kron_reduce/workflows/kiki_paired_training.py",
                   "src/kron_reduce/workflows/kiki_short_history.py",
                   "src/kron_reduce/workflows/kiki_history_learning.py",
                   "src/kron_reduce/workflows/kiki_stability_comparison.py",
                   "src/kron_reduce/evaluation/recorded_error_audit.py",
                   "src/kron_reduce/models/nonlinear_latent.py", "src/kron_reduce/models/learned_latent.py",
                   "src/kron_reduce/models/causal_latent.py", "src/kron_reduce/data/kiki_history.py",
                   "src/kron_reduce/data/history_contract.py", "src/kron_reduce/physics/coordinates.py"]
    code_hashes = {p: sha256_file(root / p) for p in guard_paths}
    _write(result_dir / "pre_execution_binding.json", {"source_commit": commit, "source_sha256": code_hashes,
           "source_manifest_sha256": proposal["source_manifest_sha256"], "approval": approval})
    source_root = resolve_source_root(root, config["source_root"])
    rows, bindings = allocate_roles(source_root, config["sources"], 324)
    if rows != allocation["rows"] or bindings != allocation["source_bindings"]:
        raise ValueError("A05 source allocation or bindings changed")
    provider = KikiHistorySamples(source_root, rows, bindings, history_count=11, power_index=10)
    mass, train = provider.load("train")
    _, val = provider.load("val")
    if (len(train), len(val)) != (258, 88) or any(
        [s.event_id for s in population] != source_plan[f"{role}_event_ids"]
        for role, population in (("train", train), ("val", val))
    ) or not np.array_equal(mass, paired_start["mass"]) or not np.array_equal(
        visible_scales(train), paired_start["target_scale"]
    ):
        raise ValueError("A05 eligible population, priors or train scales changed")
    event_steps = 6 * 80 * (len(train)*10 + sum(len(s.forecast_time)-1 for s in train))
    if event_steps != 35_868_960:
        raise ValueError("A05 exact integration budget changed")
    plan = {"approval": approval, "proposal": proposal, "source_sha256": code_hashes,
            "source_bindings": bindings, "train_event_ids": [s.event_id for s in train],
            "val_event_ids": [s.event_id for s in val], "source_commit": commit,
            "training_rk4_event_steps": event_steps, "training_rhs_event_evaluations": 4*event_steps,
            "historical_backbone_updates": 80, "historical_A04_total_updates": 960,
            "candidate_settings": {f"{g}-{s}": asdict(candidate_settings("B", s)) for g in GROUPS for s in SEEDS}}
    _write(result_dir / "scientific_plan.json", plan)
    _write(result_dir / "role_allocation.json", allocation)
    _write(result_dir / "delivery_prerequisite.json", real_evaluation_prerequisite())
    ledger = PairedBudget(root, plan, run_id, result_dir)
    models, records, scores, numerics, stability = {}, [], {}, {}, {}
    for seed in SEEDS:
        for group in GROUPS:
            directory = result_dir / f"candidate-{group}-{seed}"
            directory.mkdir()
            ledger.claim(group, seed)
            fit_start = time.perf_counter()
            print(f"A05 {group}-{seed}: 80 fixed updates", flush=True)
            try:
                model = fit_nonlinear_model(train, mass, candidate_settings("B", seed),
                    target_scale_samples=train, allow_partial_horizons=True,
                    paired_start=paired_start, freeze_backbone=group == "F",
                    progress_callback=lambda epoch, g=group, s=seed: ledger.progress(g, s, epoch))
                if (model.fit_record["actual_optimizer_steps"] != 80
                    or model.fit_record["parameter_count"] != 238
                    or model.fit_record["backbone_frozen"] != (group == "F")
                    or model.fit_record["optimizer_updated_scalars"] != (184 if group == "F" else 238)
                    or model.fit_record["final_checkpoint_optimizer_step"] != 80
                    or len(model.fit_record["loss_trace"]) != 80
                    or set(model.fit_record["initial_parameters"]) != {"raw", "port", "skew", "decay", "encoder"}
                    or not np.array_equal(model.feature_scale, paired_start["feature_scale"])
                    or not np.array_equal(model.target_scale, paired_start["target_scale"])):
                    raise ValueError("fitted model differs from the paired training plan")
                if group == "J" and ("F", seed) in models and model.fit_record["initial_parameter_sha256"] != (
                    models[("F", seed)][0].fit_record["initial_parameter_sha256"]
                ):
                    raise ValueError("F/J initial parameters differ")
                checkpoint = directory / "model.npz"
                model.save(checkpoint)
                restored = LearnedNonlinearModel.load(checkpoint)
                record = {"candidate": group, "seed": seed, "status": "completed",
                          "actual_optimizer_steps": 80, "fit_record": model.fit_record,
                          "checkpoint_sha256": sha256_file(checkpoint),
                          "fit_wall_seconds": time.perf_counter()-fit_start}
                models[(group, seed)] = (model, restored)
            except (ValueError, FloatingPointError, OverflowError, RuntimeError) as error:
                record = {"candidate": group, "seed": seed, "status": "failed",
                          "actual_optimizer_steps": None, "failure_type": type(error).__name__,
                          "reason": str(error), "val_main_nrmse": None,
                          "confirmed_optimizer_steps": len(list(ledger.slot(group, seed).glob("step-*.json")))}
            _write(directory / "fit_record.json", record)
            ledger.finish(group, seed, directory / "fit_record.json")
            records.append(record)
    _write(result_dir / "budget_snapshot.json", ledger.snapshot())
    # No val scoring, labels or LOS is inspected during any optimizer update.
    for record in records:
        group, seed = record["candidate"], record["seed"]
        if (group, seed) not in models:
            continue
        model, restored = models[(group, seed)]
        directory = result_dir / f"candidate-{group}-{seed}"
        for role, population in (("train", train), ("val", val)):
            try:
                score, predictions = population_scores(restored, population, restored.target_scale)
                additive_channel_scores(score, predictions, population, restored.target_scale)
            except (ValueError, FloatingPointError, OverflowError, RuntimeError) as error:
                predictions = {s.event_id: f"{type(error).__name__}: {error}" for s in population}
                score = score_fixed_predictions(predictions, population, restored.target_scale)
            _write(directory / f"{role}_scores.json", score)
            if role == "val":
                scores[(group, seed)] = score
                record.update(val_main_nrmse=score["available"]["multi_horizon_normalized_rmse"],
                              val_complete3_nrmse=score["complete3_common"]["multi_horizon_normalized_rmse"])
                first = val[0]
                try:
                    direct = restored.predict(first.history, first.forecast_time, first.equilibrium_delta)
                    replay = model.predict(first.history, first.forecast_time, first.equilibrium_delta)
                    batch_error = float(np.max(abs(direct[:, :20]-predictions[first.event_id])/restored.target_scale))
                    if not np.isfinite(batch_error):
                        raise FloatingPointError("nonfinite replay difference")
                    numerics[(group, seed)] = {
                        "rows": numerical_population(restored, val, restored.target_scale, contract),
                        "checkpoint_replay_exact": np.array_equal(direct, replay),
                        "batched_direct_max_normalized_difference": batch_error}
                except (ValueError, TypeError, FloatingPointError, OverflowError, RuntimeError) as error:
                    numerics[(group, seed)] = {
                        "rows": [{"event_id": s.event_id, "qualified": False,
                                  "failure_reason": str(error)} for s in val],
                        "checkpoint_replay_exact": False, "batched_direct_max_normalized_difference": None}
                _write(directory / "trajectory_numerics.json", numerics[(group, seed)])
    freeze = {"records": records, "scientific_plan_sha256": ledger.plan_hash,
              "checkpoint_hashes": {f"{g}-{s}": sha256_file(result_dir/f"candidate-{g}-{s}/model.npz")
                                    for g, s in models}, "stability_evaluation_started": False,
              "checkpoint_rule": "last_update_80", "selection_performed": False}
    _write(result_dir / "comparison_freeze.json", freeze)
    freeze_hash = sha256_file(result_dir / "comparison_freeze.json")
    metadata = truth_metadata(provider)
    coverage = []
    for row in provider.coverage:
        truth = metadata[row["event_id"]]
        already = truth["true_event_time_record_s"] is not None and truth["true_event_time_record_s"] <= .1+1e-10
        coverage.append({**row, **truth, "prospective_eligible": row["history_eligible"] and not already,
                         "exclusion_reason": "insufficient_real_history_plus_future" if not row["history_eligible"]
                         else "already_los_within_observation" if already else None})
    _write(result_dir / "coverage.json", {"rows": coverage})
    decompositions, joined_rows = {}, {}
    for (group, seed), (_, model) in models.items():
        directory = result_dir / f"candidate-{group}-{seed}"
        if sha256_file(directory/"model.npz") != freeze["checkpoint_hashes"][f"{group}-{seed}"]:
            raise RuntimeError("A05 checkpoint changed after comparison freeze")
        labels = {s.event_id: metadata[s.event_id]["label"] for s in val}
        score = scores[(group, seed)]
        if score["available"]["multi_horizon_normalized_rmse"] is not None:
            joined_rows[(group, seed)], decompositions[(group, seed)] = join_horizons(score, labels)
            _write(directory / "error_decomposition.json", decompositions[(group, seed)])
        try:
            result, arrays = stability_population(model, val, metadata, provider.static, contract)
        except (ValueError, FloatingPointError, OverflowError, RuntimeError) as error:
            unresolved = [{"event_id": s.event_id, "source_dataset": s.source_dataset,
                           "fault_line": s.fault_line, **metadata[s.event_id],
                           "prospective_eligible": metadata[s.event_id]["true_event_time_record_s"] is None
                           or metadata[s.event_id]["true_event_time_record_s"] > .1+1e-10,
                           "qualified": False, "predicted_los": False, "window_settled": False,
                           "failure_reason": f"{type(error).__name__}: {error}",
                           "prediction_delivery_status": "REFUSE_NO_CERTIFIED_ENVELOPE"} for s in val]
            result, arrays = {"rows": unresolved, "summary": {
                **summarize(unresolved), "development_evaluation_refusals": 0,
                "delivery_refusals": len(val)}}, {}
        allocated = [r for r in coverage if r["role"] == "val"]
        result["summary"].update(allocated_events=len(allocated),
            allocated_unstable=sum(r["label"] == "unstable" for r in allocated),
            allocated_stable=sum(r["label"] == "stable" for r in allocated),
            unstable_eligibility_coverage=result["summary"]["eligible_unstable"] / sum(r["label"] == "unstable" for r in allocated),
            assigned_unstable_detection_fraction=result["summary"]["tp"] / sum(r["label"] == "unstable" for r in allocated))
        result["comparison_freeze_sha256"] = freeze_hash
        _write(directory / "stability.json", result)
        with (directory / "stability_paths.npz").open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        stability[(group, seed)] = result
        print(f"A05 {group}-{seed} val={score['available']['multi_horizon_normalized_rmse']} LOS={result['summary']}", flush=True)
    paired = []
    for seed in SEEDS:
        if all((g, seed) in joined_rows for g in GROUPS):
            value = paired_contributions(joined_rows[("F", seed)], joined_rows[("J", seed)], labels)
            value.update(reference="same_seed_J", metric="F_minus_J_main_normalized_MSE", seed=seed)
            paired.append(value)
        else:
            paired.append({"seed": seed, "status": "UNDETERMINED", "reason": "failed_fit_or_score_retained"})
    _write(result_dir / "paired_decomposition.json", {"rows": paired})
    judgement = judge(records, decompositions, numerics, stability)
    summary = {**judgement, "records": records, "budget": ledger.snapshot(),
               "train_event_count": len(train), "val_event_count": len(val),
               "allocated_train": sum(r["role"] == "train" for r in rows),
               "allocated_val": sum(r["role"] == "val" for r in rows),
               "process_status": "PASS" if judgement["checks"]["all_six_fits_completed"] else "FAIL",
               "scientific_verdict": diagnostic_verdict(judgement["next_stage_research_supported"]),
               "evidence_kind": "uncertified_development_evaluation", "delivery_predictions_issued": 0,
               "all_allocated_events_retained": True}
    _write(result_dir / "learning_summary.json", summary)
    _write(result_dir / "data_access_audit.json", {**provider.access_audit(),
           "labels_and_event_times_read_only_after_comparison_freeze": True})
    if (git_head(root) != commit or status_lines(root)
        or any(sha256_file(root/p) != h for p, h in code_hashes.items())
        or any(sha256_file(result_dir/f"candidate-{name}/model.npz") != h
               for name, h in freeze["checkpoint_hashes"].items())
        or sha256_file(result_dir/"comparison_freeze.json") != freeze_hash):
        raise RuntimeError("A05 frozen implementation or comparison changed")
    for source, binding in bindings.items():
        if any(sha256_file(source_root/source/p) != h for p, h in binding.items()):
            raise RuntimeError("source data changed during A05")
    _write(result_dir / "lineage.json", {"source_commit": commit, "source_sha256": code_hashes,
           "run_id": run_id, "wall_seconds": time.perf_counter()-started})
    _write(result_dir / "stage_evidence.json", {"process_status": summary["process_status"],
           "scientific_verdict": summary["scientific_verdict"], "promotion_decision": "KEEP_BASELINE",
           "source_commit": commit, "data_policy": "kiki_original_train_only",
           "source_val_ids_read": [], "test_ids_read": [], "hidden_training_targets_read": False,
           "level_6_claimed": False, "claim_boundary": config["claim_boundary"],
           "development_authorization_id": AUTH, "evidence_kind": "uncertified_development_evaluation",
           "certified_envelope_established": False, "deployment_predictions_allowed": False,
           "model_promoted": False})
    _write(result_dir / "artifact_hashes.json", {p.relative_to(result_dir).as_posix(): sha256_file(p)
           for p in sorted(result_dir.rglob("*")) if p.is_file() and p.name != "artifact_hashes.json"})
    verify_artifact_directory(result_dir, strict=True)
    if summary["process_status"] != "PASS":
        raise RuntimeError("A05 failures retained; no additional fits permitted")
