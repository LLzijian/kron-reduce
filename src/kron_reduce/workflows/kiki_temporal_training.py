"""Frozen, paired temporal-objective comparison with explicit event-RHS budgets."""
from __future__ import annotations

import ast
import csv
import json
import math
import subprocess
import time
import tomllib
import traceback
from dataclasses import asdict, replace
from importlib.metadata import version

import numpy as np

from kron_reduce.data.kiki_history import KikiHistorySamples, resolve_source_root
from kron_reduce.evaluation.bounded_forecast import forecast_bounded
from kron_reduce.evaluation.event_rhs_budget import EventRhsBudget
from kron_reduce.evaluation.recorded_error_audit import weights
from kron_reduce.evaluation.temporal_comparison import outcome
from kron_reduce.evaluation.train_mechanism_audit import audit_event
from kron_reduce.evaluation.train_refinement_audit import (
    _coarse_fine_margin_failures,
    _diagnostic_margin_failures,
    compare_recursive,
)
from kron_reduce.models.learned_latent import visible_scales
from kron_reduce.models.nonlinear_latent import (
    LearnedNonlinearModel,
    _array_sha256,
    _features,
    _validated_full_start,
    fit_nonlinear_model,
)
from kron_reduce.workflows.kiki_paired_training import additive_channel_scores
from kron_reduce.workflows.kiki_short_history import (
    classify_los,
    registered_contract,
    score_fixed_predictions,
    shared_budget_directory,
    stability_refinement,
)
from kron_reduce.workflows.kiki_stability_comparison import summarize
from kron_reduce.workflows.kiki_train_mechanism import (
    _npz,
    counted_model,
    integration_substeps,
    path_physics,
)
from kron_reduce.workflows.kiki_train_refinement import (
    compare_on_truth_windows,
    require_array_identity,
)
from kron_reduce.workflows.nonlinear_history_learning import _write
from project_harness.evidence import sha256_file, verify_artifact_directory
from project_harness.git_state import assert_clean, git_head, status_lines

AUTH = "kiki_a11_development_20261024"
EXPERIMENT = "paper_kiki_a11_temporal"
GROUPS = ("mean", "time")
SEEDS = (324, 325, 326)
STEPS = (.005, .0025)
TAU = float.fromhex("0x1.f11fce9ec8e38p-5")
ARRAYS = ("mass", "edge_stiffness", "damping", "coupling", "latent_skew",
          "latent_dissipation", "encoder", "feature_scale", "target_scale")
FIXED = {"groups": list(GROUPS), "seeds": list(SEEDS), "updates_per_fit": 240,
    "max_fit_attempts": 6, "max_optimizer_updates": 1440, "max_prediction_paths": 4446,
    "max_rollout_calls": 8892, "max_training_rk4_updates": 215213760,
    "max_training_rhs": 860855040, "max_eval_rk4_updates": 6141372,
    "max_eval_integration_rhs": 24565488, "max_physical_rhs": 1984116,
    "max_eval_rhs": 26549604, "max_total_rhs": 887404644,
    "training_max_step": .005, "max_steps": list(STEPS), "temporal_temperature": TAU,
    "temporal_temperature_hex": TAU.hex(), "history_count": 11, "power_index": 10,
    "history_weight": 0., "encoder_penalty": .0001, "learning_rate": .01,
    "latent_dimension": 2, "encoder_frames": 3, "parameter_count": 238,
    "allocated_train": 268, "eligible_train": 258, "total_train_future_points": 72147,
    "allocated_val": 101, "eligible_val": 88, "total_val_future_points": 22734,
    "complete3_val": 74, "initial_optimizer_step": 0, "optimizer_state": "fresh_Adam",
    "source_val_payload_allowed": False, "source_test_payload_allowed": False,
    "model_promotion_allowed": False, "deployment_predictions_allowed": False,
    "labels_only_after_trajectory_freeze": True}


def authorization(root, experiment, config):
    policy = json.loads((root/"configs/research_policy.json").read_text(encoding="utf-8"))
    a = policy["development_evaluations"].get(AUTH, {})
    expected = {**FIXED, "status": "user_approved", "authority": "explicit_user_reply",
        "scientific_design_approver": "Astra Ultra", "experiment": EXPERIMENT,
        "workflow": "kiki_a11_temporal_v1", "stage": "temporal_training",
        "module": "kron_reduce.workflows.kiki_temporal_training", "dataset": "kiki_ieee39_v1"}
    if any(type(a.get(k)) is not type(v) or a[k] != v for k,v in expected.items()):
        raise PermissionError("A11 requires its active fixed-budget authorization")
    if any(a.get(k) != v for k,v in config.items()) or any(config.get(k) != v for k,v in FIXED.items()):
        raise PermissionError("A11 configuration differs from its approval")
    manifest = json.loads((root/"configs/project_manifest.json").read_text(encoding="utf-8"))
    if manifest["experiments"].get(EXPERIMENT) != experiment:
        raise PermissionError("A11 registered identity changed")
    for p,k in (("development_protocol_path","approval_protocol_sha256"),("design_path","design_sha256")):
        if sha256_file(root/config[p]) != config[k]:
            raise PermissionError("A11 approved document bytes changed")
    if sha256_file(root/a["scientific_review_path"]) != a["scientific_review_sha256"]:
        raise PermissionError("A11 implementation-review binding changed")
    return a


def local_statistics(event_ids, event_weights, counts, rows):
    """Preserve the original population denominator in each duration bucket."""
    if any(r["status"] != "ok" for r in rows):
        return {"complete": False, "metrics": None}
    if [r["event_id"] for r in rows] != event_ids:
        raise ValueError("local metric event order changed")
    values = []
    for r in rows:
        d = r["diagnostic"]
        if _diagnostic_margin_failures(d):
            raise ArithmeticError("recorded path margin evidence is incomplete")
        fixed = d["representative_truth_window"]["fixed_five_point_audit"]
        values.append((d["representative_window_error_density_C_over_p"],fixed["E_atom"],
            abs(fixed["truth"]["rho"]-fixed["prediction"]["rho"])))
    names = ("weighted_window_error_density","weighted_E_atom","weighted_absolute_rho_difference")
    metrics = {name: math.fsum(w*v[j] for w,v in zip(event_weights,values,strict=True))
               for j,name in enumerate(names)}
    buckets = {}
    for name, predicate in (("through_1s",lambda n:n<=100),
            ("between_1s_and_3s",lambda n:100<n<300),("scored_at_3s",lambda n:n==300)):
        indices = [i for i,n in enumerate(counts) if predicate(n)]
        mass = math.fsum(event_weights[i] for i in indices)
        contribution = {k:math.fsum(event_weights[i]*values[i][j] for i in indices) for j,k in enumerate(names)}
        buckets[name] = {"event_count":len(indices),"original_weight_mass":mass,
            "same_denominator_contribution":contribution,
            "conditional_mean":{k:v/mass if mass else None for k,v in contribution.items()}}
    if any(abs(math.fsum(b["same_denominator_contribution"][k] for b in buckets.values())-v)>1e-12
           for k,v in metrics.items()):
        raise ArithmeticError("local duration contributions do not reconstruct their total")
    diagnostics = []
    for row in rows:
        d = row["diagnostic"]
        if "normalized_point_e2" not in d:
            continue
        losses = np.asarray(d["normalized_point_e2"],dtype=float)
        n = len(losses)
        c = .5/n+.5/min(n,100)*(np.arange(n)<min(n,100))
        logits = np.log(c)+losses/TAU
        beta = np.exp(logits-np.max(logits))
        beta /= math.fsum(beta)
        window = d["representative_truth_window"]
        first,last = max(0,window["start_index"]-11),min(n,window["end_index"]-10)
        diagnostics.append({"event_id":row["event_id"],"max_beta":float(np.max(beta)),
            "effective_time_points":float(1/math.fsum(beta**2)),
            "fixed_truth_window_beta_mass":float(math.fsum(beta[first:last]))})
    return {"complete":True,"metrics":metrics,"duration_buckets":buckets,
            "fixed_temperature_time_weight_diagnostics":diagnostics}


def initial_risk(rows, ws, temperature):
    total = []
    for r in rows:
        losses = np.asarray(r["diagnostic"]["normalized_point_e2"])
        n = len(losses)
        c = .5/n + .5/min(n,100)*(np.arange(n)<min(n,100))
        logits = np.log(c)+losses/temperature
        top = float(np.max(logits))
        total.append(temperature*(top+math.log(math.fsum(np.exp(logits-top)))))
    return math.fsum(w*v for w,v in zip(ws,total,strict=True))


def val_truth_metadata(provider):
    """Filter registered val identities before interpreting any label or time."""
    wanted = [r for r in provider.rows if r["role"] == "val"]
    result = {}
    for source in sorted({r["source_dataset"] for r in wanted}):
        by_raw = {r["raw_id"]: r["event_id"] for r in wanted if r["source_dataset"] == source}
        with (provider.source_root/source/"dataset_index.csv").open(encoding="utf-8-sig",newline="") as stream:
            for row in csv.DictReader(stream):
                event_id = by_raw.get(row["event_id"])
                if event_id is None:
                    continue
                if event_id in result:
                    raise ValueError("duplicate val metadata identity")
                result[event_id] = {"label":row["label"],
                    "true_event_time_record_s":float(row["event_time_s"]) if row["event_time_s"] else None}
    if set(result) != {r["event_id"] for r in wanted}:
        raise ValueError("val metadata population incomplete")
    return result


def classification_counts(summary):
    """Adapt the established LOS summary without renaming unresolved evidence."""
    return {**{k:summary[k] for k in ("fn","fp","tp","tn")},
        "unresolved":summary["numerical_unresolved"],
        "eligible_events":summary["eligible"],
        "development_evaluation_refusals":summary["development_evaluation_refusals"]}


def replay_train_reference(sources, seed, train, mass, provider, target_scale, ids, ws, counts, result_dir):
    """Rebuild both source train baselines from sealed arrays, without RHS."""
    checks = []
    for name, suffix, step in (("a09","h005",.005),("a10","h0025",.0025)):
        directory = sources[name]/f"seed-{seed}-{suffix}"
        saved_rows = json.loads((directory/"event_diagnostics.json").read_text(encoding="utf-8"))["rows"]
        if [r["event_id"] for r in saved_rows] != ids:
            raise ValueError("source train reference order changed")
        with np.load(directory/"paths.npz",allow_pickle=False) as data:
            hp, fp, offsets = data["history"],data["future"],data["offsets"]
            if hp.shape != (258,11,22) or not np.array_equal(offsets,np.r_[0,np.cumsum(np.asarray(counts)+1)]):
                raise ValueError("source train path shapes changed")
            rebuilt = []
            for i,s in enumerate(train):
                diag = audit_event(history=np.c_[s.history.delta,s.history.omega],
                    true_future=np.c_[s.target_delta,s.target_omega],
                    pred_future=fp[offsets[i]:offsets[i+1],:20],equilibrium_delta=s.equilibrium_delta,
                    mass=mass,omega_s=provider.static[s.source_dataset]["omega_s"],target_scale=target_scale)
                mismatches = compare_recursive(saved_rows[i]["diagnostic"],diag)
                if mismatches or not saved_rows[i]["physical"]["qualified"]:
                    raise ValueError(f"source train reference diagnostic/physics mismatch: {s.event_id}")
                rebuilt.append({"event_id":s.event_id,"status":"ok","diagnostic":diag})
        stats = local_statistics(ids,ws,counts,rebuilt)
        mse = math.fsum(w*r["diagnostic"]["event_mse"] for w,r in zip(ws,rebuilt,strict=True))
        item = {"seed":seed,"max_step":step,"new_rhs":0,"path_sha256":sha256_file(directory/"paths.npz"),
            "diagnostics_sha256":sha256_file(directory/"event_diagnostics.json"),
            "all_event_diagnostics_rebuilt":True,"main_mse":mse,**stats}
        _write(result_dir/f"local-source-{seed}-{step}.json",item)
        checks.append(item)
    return checks


def verify_physical_code(root, source_commit):
    """Optimizer changes cannot silently alter the frozen model's equations."""
    path = "src/kron_reduce/models/nonlinear_latent.py"
    old = subprocess.check_output(["git","-C",str(root),"show",f"{source_commit}:{path}"]).decode("utf-8")
    new = (root/path).read_text(encoding="utf-8")
    names = {"NonlinearPassiveClosure","LearnedNonlinearModel","NonlinearFitSettings","_features","_validated_full_start"}
    def nodes(text):
        return {v.name:ast.dump(v,include_attributes=False) for v in ast.parse(text).body
                if isinstance(v,(ast.ClassDef,ast.FunctionDef)) and v.name in names}
    if nodes(old) != nodes(new) or set(nodes(new)) != names:
        raise ValueError("A11 source physical model/input/initialization code changed")


def run(*, root, experiment, stage, result_dir, run_id):
    """Any failed execution is sealed once, including every uncompleted slot."""
    state = {"stage": "preflight", "fit_records": [], "allocated": []}
    try:
        return _execute(root=root, experiment=experiment, stage=stage,
                        result_dir=result_dir, run_id=run_id, state=state)
    except Exception as error:
        preserve_failure(result_dir, run_id, state, error)
        raise


def preserve_failure(result_dir, run_id, state, error):
    """Never turn missing predictions or a partially written path into a pass."""
    budget = state.get("budget")
    snapshot = budget.snapshot() if budget is not None else None
    receipt = {"run_id": run_id, "status": "failed", "stage": state["stage"],
        "error_type": type(error).__name__, "reason": str(error),
        "traceback": traceback.format_exc(), "fit_records": state["fit_records"],
        "all_six_reserved_fits": [{"group": g, "seed": s} for g in GROUPS for s in SEEDS],
        "planned_prediction_paths": 4446, "planned_updates": 1440,
        "budget": snapshot, "retries_allowed": False, "scientific_verdict": "REJECT",
        "model_promoted": False, "selected_candidate": None,
        "prediction_delivery_count": 0,
        "allocated_events": [{"event_id": r["event_id"], "role": r["role"],
            "campaign_status": "unresolved_failed_campaign", "counted_as_correct": False}
            for r in state["allocated"]]}
    _write(result_dir/"execution_failure.json", receipt)
    if snapshot is not None and not (result_dir/"rhs_budget.json").exists():
        _write(result_dir/"rhs_budget.json", snapshot)
    _write(result_dir/"failure_artifact_hashes.json", {
        p.relative_to(result_dir).as_posix(): sha256_file(p)
        for p in sorted(result_dir.rglob("*")) if p.is_file() and p.name != "failure_artifact_hashes.json"})


def _execute(*, root, experiment, stage, result_dir, run_id, state):
    started = time.perf_counter()
    if stage != "temporal_training":
        raise ValueError("unregistered A11 stage")
    config_path = root/"configs/experiments"/experiment["config"]
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    approval = authorization(root,experiment,config)
    assert_clean(root)
    commit = git_head(root)
    runtime = {n:version(n) for n in ("numpy","scipy","torch","h5py","filelock")}
    if runtime != {"numpy":"2.5.2","scipy":"1.18.0","torch":"2.13.0+cpu","h5py":"3.16.0","filelock":"3.32.2"}:
        raise RuntimeError("A11 requires the pinned CPU runtime")
    ledger = shared_budget_directory(root).parent/"a11_temporal_budgets"/AUTH
    ledger.parent.mkdir(parents=True,exist_ok=True)
    ledger.mkdir(exist_ok=False)
    _write(ledger/"reserved.json",{"run_id":run_id,"source_commit":commit,"retries_allowed":False})
    sources = {name:root/"results"/config[key] for name,key in (("a06","source_run"),("a09","a09_run"),("a10","a10_run"))}
    manifests = {name:verify_artifact_directory(p,strict=True,expected_manifest_sha256=config[
        "source_manifest_sha256" if name=="a06" else f"{name}_manifest_sha256"]) for name,p in sources.items()}
    prior_plan = json.loads((sources["a10"]/"scientific_plan.json").read_text(encoding="utf-8"))
    prior_summary = json.loads((sources["a10"]/"audit_summary.json").read_text(encoding="utf-8"))
    if prior_summary["scientific_verdict"] != "ACCEPT" or not prior_summary["time_aggregation_proposal_supported"]:
        raise PermissionError("A11 requires the completed independent A10 support gate")
    verify_physical_code(root,prior_plan["source_commit"])
    for rel in ("src/kron_reduce/data/kiki_history.py","src/kron_reduce/data/history_contract.py",
                "src/kron_reduce/physics/coordinates.py","src/kron_reduce/models/learned_latent.py",
                "src/kron_reduce/models/causal_latent.py"):
        if sha256_file(root/rel) != prior_plan["source_sha256"][rel]:
            raise ValueError("A11 frozen data or model dependency changed")
    allocation_path = root/config["frozen_allocation_path"]
    if sha256_file(allocation_path) != config["frozen_allocation_sha256"]:
        raise ValueError("A11 allocation changed")
    allocation = json.loads(allocation_path.read_text(encoding="utf-8"))
    allocated = allocation["rows"]
    state["allocated"] = allocated
    if sum(r["role"]=="train" for r in allocated)!=268 or sum(r["role"]=="val" for r in allocated)!=101:
        raise ValueError("A11 allocation coverage changed")
    source_root = resolve_source_root(root,config["source_root"])
    bindings = allocation["source_bindings"]
    for source, files in bindings.items():
        if any(sha256_file(source_root/source/p)!=h for p,h in files.items()):
            raise ValueError("A11 raw source integrity changed")
    provider = KikiHistorySamples(source_root,allocated,bindings,history_count=11,power_index=10)
    mass, train = provider.load("train")
    event_ids = [s.event_id for s in train]
    counts = [len(s.forecast_time)-1 for s in train]
    ws = weights([{"event_id":s.event_id,"source_dataset":s.source_dataset,"fault_line":s.fault_line,"role":"train"} for s in train])
    if len(train)!=258 or sum(counts)!=72147 or event_ids!=prior_plan["train_event_ids"] or counts!=prior_plan["future_point_counts"] or ws!=prior_plan["event_weights"] or counts[0]!=300:
        raise ValueError("A11 frozen train population changed")
    offsets = np.r_[0,np.cumsum(np.asarray(counts)+1)]
    truth_arrays = {"history":np.stack([np.c_[s.history.delta,s.history.omega] for s in train]),
        "future":np.concatenate([np.c_[s.target_delta,s.target_omega] for s in train]),
        "offsets":offsets,"equilibrium":np.stack([s.equilibrium_delta for s in train]),
        "mass":mass,"target_scale":visible_scales(train),
        "constant_power":np.stack([s.history.constant_power for s in train]),
        "history_time":np.stack([s.history.time for s in train]),
        "forecast_time":np.concatenate([s.forecast_time for s in train]),
        "omega_s":np.array([provider.static[s.source_dataset]["omega_s"] for s in train])}
    with np.load(sources["a10"]/"real_train_arrays.npz",allow_pickle=False) as old:
        require_array_identity(old,truth_arrays)
    with np.load(sources["a09"]/"real_train_arrays.npz",allow_pickle=False) as old:
        require_array_identity(old,{k:truth_arrays[k] for k in old.files})
    for s,n in zip(train,counts,strict=True):
        for step,m in ((.005,2),(.0025,4)):
            if integration_substeps(s.history.time-s.history.time[0],step)!=10*m or integration_substeps(s.forecast_time,step)!=n*m:
                raise ValueError("A11 original-clock train substep plan changed")
    starts, models, cached_train, source_mse, start_bindings = {}, {}, {}, {}, {}
    features = np.stack([_features(s.history, s.equilibrium_delta, 3, "frames") for s in train])
    feature_scale = np.maximum(np.sqrt(np.asarray(ws) @ features**2), 1e-5)
    for i,seed in enumerate(SEEDS):
        cp = sources["a06"]/config["source_checkpoint_paths"][i]
        if sha256_file(cp)!=config["source_checkpoint_sha256"][i]:
            raise ValueError("A11 source checkpoint changed")
        models[seed] = LearnedNonlinearModel.load(cp)
        with np.load(cp,allow_pickle=False) as data:
            starts[seed] = {k:data[k].copy() for k in ARRAYS}
        settings = models[seed].settings
        if (settings.learning_rate!=.01 or settings.history_weight!=0 or settings.encoder_penalty!=.0001
                or settings.latent_dimension!=2 or settings.frames!=3 or settings.seed!=seed
                or settings.encoder_mode!="frames"):
            raise ValueError("A11 fixed source settings differ")
        raw_parameters, _, start_bindings[seed] = _validated_full_start(
            starts[seed],mass,feature_scale,truth_arrays["target_scale"],2)
        start_bindings[seed]["initial_parameter_sha256"] = {
            k:_array_sha256(v) for k,v in raw_parameters.items()}
        source_score = json.loads((sources["a06"]/f"checkpoint-240-{seed}/train_scores.json").read_text(encoding="utf-8"))
        hs = source_score["available"]["horizons"]
        rr = [{r["event_id"]:r for r in hs[h]["rows"]} for h in ("1.0","3.0")]
        source_mse[seed] = math.fsum(w*.5*(rr[0][e]["normalized_mse"]+rr[1][e]["normalized_mse"]) for w,e in zip(ws,event_ids,strict=True))
        cached_train[seed] = json.loads((sources["a09"]/f"seed-{seed}-h005/event_diagnostics.json").read_text(encoding="utf-8"))["rows"]
        if [r["event_id"] for r in cached_train[seed]] != event_ids:
            raise ValueError("A11 cached zero-step train order changed")
        replay_train_reference(sources,seed,train,mass,provider,models[seed].target_scale,
                               event_ids,ws,counts,result_dir)
    tau = math.fsum(source_mse.values())/3
    if tau.hex()!=TAU.hex() or config["temporal_temperature"].hex()!=TAU.hex():
        raise ValueError("A11 original train temperature reconstruction differs")
    code_paths = ["AGENTS.md","configs/research_policy.json","configs/project_manifest.json",
        config_path.relative_to(root).as_posix(),"configs/workflows/kiki_a11_temporal_v1.json",
        config["development_protocol_path"],config["design_path"],approval["scientific_review_path"],
        "src/kron_reduce/workflows/kiki_temporal_training.py","src/kron_reduce/models/nonlinear_latent.py",
        "src/kron_reduce/workflows/kiki_paired_training.py", "configs/input_contracts.json",
        "src/kron_reduce/models/temporal_risk.py","src/kron_reduce/evaluation/event_rhs_budget.py",
        "src/kron_reduce/evaluation/bounded_forecast.py","src/kron_reduce/evaluation/temporal_comparison.py",
        "src/kron_reduce/evaluation/train_mechanism_audit.py","src/kron_reduce/evaluation/train_refinement_audit.py",
        "src/kron_reduce/workflows/kiki_train_mechanism.py","src/kron_reduce/workflows/kiki_train_refinement.py",
        "src/kron_reduce/workflows/kiki_short_history.py","src/kron_reduce/workflows/kiki_stability_comparison.py",
        "src/kron_reduce/data/kiki_history.py","src/kron_reduce/data/history_contract.py",
        "src/kron_reduce/models/learned_latent.py","src/kron_reduce/models/causal_latent.py","src/kron_reduce/physics/coordinates.py"]
    code_hashes = {p:sha256_file(root/p) for p in code_paths}
    plan = {"approval":approval,"source_commit":commit,"source_sha256":code_hashes,"runtime":runtime,
        "train_event_ids":event_ids,"future_point_counts":counts,"event_weights":ws,
        "source_train_mse":source_mse,"source_start_bindings":start_bindings,
        "temporal_temperature":tau,"temporal_temperature_hex":tau.hex(),
        "fits":6,"updates_per_fit":240,"max_prediction_paths":4446,"max_rollout_calls":8892,
        "training_rhs":860855040,"eval_rhs":26549604,"max_total_rhs":887404644,
        "source_manifests":{k:sha256_file(p/"artifact_hashes.json") for k,p in sources.items()},
        "labels_decoded":False,"val_payloads_opened":False,"retries_allowed":False}
    _write(result_dir/"scientific_plan.json",plan)
    _write(ledger/"plan.json",plan)
    _write(result_dir/"budget_binding.json",{"authorization_id":AUTH,"scientific_plan_sha256":sha256_file(result_dir/"scientific_plan.json"),"ledger_plan_sha256":sha256_file(ledger/"plan.json")})
    _write(result_dir/"integrity_file_reads.json",{"raw_source_root":str(source_root),"raw_source_files":bindings,
        "source_artifacts":manifests,"purpose":"file-integrity byte reads separate from decoded payloads"})
    _npz(result_dir/"real_train_arrays.npz",**truth_arrays)
    budget = EventRhsBudget(integration_limit=885420528,physical_limit=1984116,total_limit=887404644,
        rollout_limit=8892,prediction_path_limit=4446,training_integration_limit=860855040,
        evaluation_integration_limit=24565488)
    state["budget"] = budget
    state["stage"] = "training"
    fit_records, checkpoints, final_models = state["fit_records"], {}, {}
    for seed in SEEDS:
        for group in GROUPS:
            slot = ledger/f"fit-{group}-{seed}"
            slot.mkdir(exist_ok=False)
            _npz(result_dir/f"initial-{group}-{seed}.npz",**starts[seed])
            settings = replace(models[seed].settings,epochs=240)
            directory = result_dir/f"candidate-{group}-{seed}"
            directory.mkdir()
            def progress(step,slot=slot):
                if step!=len(list(slot.glob("step-*.json")))+1 or not 1<=step<=240:
                    raise RuntimeError("A11 nonsequential optimizer progress")
                with (slot/f"step-{step:03d}.json").open("x",encoding="utf-8") as stream:
                    json.dump({"completed_optimizer_step":step},stream)
            def checkpoint(step,model,state,group=group,seed=seed,directory=directory,settings=settings):
                fit = model.fit_record
                first = fit["loss_trace"][0]
                baseline_mse = prior_summary["weighted_trajectory_summaries"][f"{seed}-0.005"]["main_mse"]
                expected_future = baseline_mse if group=="mean" else initial_risk(cached_train[seed],ws,TAU)
                if (step!=240 or (group,seed) in checkpoints or fit["actual_optimizer_steps"]!=240
                        or fit["parameter_count"]!=238 or len(fit["loss_trace"])!=240
                        or fit["total_optimizer_steps"]!=240 or fit["initial_optimizer_step"]!=0
                        or fit["initial_optimizer_moments_zero"] is not True or fit["exact_optimizer_resume"] is not False
                        or asdict(model.settings)!=asdict(settings) or max(fit["full_start_roundtrip_errors"].values())>1e-12
                        or abs(first["future_loss"]-baseline_mse)>1e-12
                        or abs(first["future_objective"]-expected_future)>1e-12
                        or abs(first["objective"]-first["future_objective"]-first["encoder_penalty"])>1e-12
                        or fit["training_rhs_event_evaluations"]!=143475840
                        or fit["training_max_step"]!=.005 or fit["temporal_temperature"]!=(TAU if group=="time" else None)
                        or fit["full_start_sha256"]!=start_bindings[seed]["full_start_sha256"]
                        or fit["initial_parameter_sha256"]!=start_bindings[seed]["initial_parameter_sha256"]
                        or fit["optimizer_updated_scalars"]!=238 or fit["backbone_frozen"]
                        or fit["new_nnls_performed"] or fit.get("risk_temperature") is not None
                        or any(not np.array_equal(a,b) for a,b in (
                            (model.closure.mass,starts[seed]["mass"]),
                            (model.feature_scale,starts[seed]["feature_scale"]),
                            (model.target_scale,starts[seed]["target_scale"])))
                        or any(float(state[f"adam_step__{name}"])!=240 for name in ("raw","port","skew","decay","encoder"))):
                    raise ValueError("A11 final checkpoint violates its frozen contract")
                model.save(directory/"model.npz")
                _npz(directory/"training_state.npz",**state)
                _write(directory/"fit_record.json",fit)
                checkpoints[group,seed] = sha256_file(directory/"model.npz")
            print(f"A11 {group}-{seed}: 240 fixed updates, two RK4 substeps",flush=True)
            try:
                with budget.segment({"group":group,"seed":seed},"training_integration",143475840):
                    model = fit_nonlinear_model(train,mass,settings,target_scale_samples=train,
                        allow_partial_horizons=True,full_start=starts[seed],checkpoint_steps=(240,),
                        checkpoint_callback=checkpoint,progress_callback=progress,training_max_step=.005,
                        temporal_temperature=TAU if group=="time" else None,rhs_audit=budget.invoke)
                if (group,seed) not in checkpoints:
                    raise ValueError("A11 missing final checkpoint")
                final_models[group,seed] = model
                record = {"group":group,"seed":seed,"status":"completed","actual_optimizer_steps":240,
                    "gradient_clipped_steps":model.fit_record["gradient_clipped_steps"],
                    "gradient_clipping_frequency":model.fit_record["gradient_clipped_steps"]/240}
            except Exception as error:
                checkpoints.pop((group,seed), None)
                record = {"group":group,"seed":seed,"status":"failed","reason":str(error),
                          "traceback":traceback.format_exc()}
            record["confirmed_optimizer_updates"] = len(list(slot.glob("step-*.json")))
            _write(slot/"finished.json",record)
            fit_records.append(record)
    if len(checkpoints)!=6:
        _write(result_dir/"training_freeze.json",{"fit_records":fit_records,"complete":False,"val_payloads_opened":False})
        raise RuntimeError("A11 fitted slots incomplete; no val decoding or substitute fits allowed")
    pair_checks = {str(s):final_models["mean",s].fit_record["initial_parameter_sha256"]==
        final_models["time",s].fit_record["initial_parameter_sha256"] for s in SEEDS}
    if not all(pair_checks.values()):
        raise ValueError("A11 actual mean/time raw initial parameter hashes differ")
    _write(result_dir/"training_freeze.json",{"fit_records":fit_records,"complete":True,
        "checkpoints":{f"{g}-{s}":h for (g,s),h in checkpoints.items()},
        "paired_initial_parameter_hashes_identical":pair_checks,"val_payloads_opened":False})
    state["stage"] = "val_evaluation"
    _, val = provider.load("val")
    val_ids = [s.event_id for s in val]
    if len(val)!=88 or sum(len(s.forecast_time)-1 for s in val)!=22734 or sum(len(s.forecast_time)==301 for s in val)!=74:
        raise ValueError("A11 fixed val population changed")
    source_plan = json.loads((sources["a06"]/"scientific_plan.json").read_text(encoding="utf-8"))
    if val_ids!=source_plan["val_event_ids"]:
        raise ValueError("A11 val order differs from source cache")
    val_counts = [len(s.forecast_time)-1 for s in val]
    _npz(result_dir/"real_val_arrays.npz",
        history=np.stack([np.c_[s.history.delta,s.history.omega] for s in val]),
        future=np.concatenate([np.c_[s.target_delta,s.target_omega] for s in val]),
        offsets=np.r_[0,np.cumsum(np.asarray(val_counts)+1)],
        equilibrium=np.stack([s.equilibrium_delta for s in val]),mass=mass,
        target_scale=truth_arrays["target_scale"],constant_power=np.stack([s.history.constant_power for s in val]),
        history_time=np.stack([s.history.time for s in val]),forecast_time=np.concatenate([s.forecast_time for s in val]),
        omega_s=np.array([provider.static[s.source_dataset]["omega_s"] for s in val]))
    _write(result_dir/"val_population_binding.json",{"event_ids":val_ids,"future_point_counts":val_counts,
        "event_metadata":[{"event_id":s.event_id,"source_dataset":s.source_dataset,"fault_line":s.fault_line,"role":"val"} for s in val],
        "training_freeze_sha256":sha256_file(result_dir/"training_freeze.json"),"labels_decoded":False})
    for s in val:
        for step,m in ((.005,2),(.0025,4)):
            if (integration_substeps(s.history.time-s.history.time[0],step)!=10*m
                    or integration_substeps(s.forecast_time,step)!=(len(s.forecast_time)-1)*m
                    or integration_substeps(np.arange(801)*.01,step)!=800*m):
                raise ValueError("A11 original-clock val substep plan changed")
    contract = registered_contract(root,config)
    cohorts, scores, comparisons, replay_checks = {}, {}, {}, []
    def score_and_diagnose(group,seed,step,population,hp,fp,physical,directory):
        predictions = {s.event_id:p[:len(s.forecast_time),:20] for s,p in zip(population,fp,strict=True)}
        score = score_fixed_predictions(predictions,population,models[seed].target_scale)
        additive_channel_scores(score,predictions,population,models[seed].target_scale)
        rows = []
        for i,s in enumerate(population):
            d = audit_event(history=np.c_[s.history.delta,s.history.omega],true_future=np.c_[s.target_delta,s.target_omega],
                pred_future=predictions[s.event_id],equilibrium_delta=s.equilibrium_delta,mass=mass,
                omega_s=provider.static[s.source_dataset]["omega_s"],target_scale=models[seed].target_scale)
            rows.append({"event_id":s.event_id,"status":"ok","physical":physical[i],"diagnostic":d})
        _write(directory/"scores.json",score)
        _write(directory/"event_diagnostics.json",{"rows":rows})
        role = population[0].role
        scores[group,seed,role,step] = score
        cohorts[group,seed,role,step] = {"history":hp,"future":fp,"rows":rows}
    def evaluate(group,seed,step,population,model):
        role = population[0].role
        ident = f"{group}-{seed}-{role}-{step}"
        slot = ledger/ident
        slot.mkdir(exist_ok=False)
        directory = result_dir/ident
        directory.mkdir()
        identity = {"group":group,"seed":seed,"role":role,"max_step":step}
        hp,fp = forecast_bounded(model,population,max_step=step,horizon_s=8. if role=="val" else None,budget=budget,identity=identity)
        with budget.segment(identity,"physical_checks",sum(len(h)+len(p) for h,p in zip(hp,fp,strict=True))):
            physical = [path_physics(model,h,p,s) for h,p,s in zip(hp,fp,population,strict=True)]
        _npz(directory/"paths.npz",history=hp,future=np.concatenate(fp),offsets=np.r_[0,np.cumsum([len(p) for p in fp])])
        score_and_diagnose(group,seed,step,population,hp,fp,physical,directory)
        _write(slot/"finished.json",{"reserved_paths":len(population),"status":"completed"})
    # Only existing arrays are decoded for the source .005 val reference.
    for seed in SEEDS:
        prefix = sources["a06"]/f"checkpoint-240-{seed}"
        original_score = json.loads((prefix/"val_scores.json").read_text(encoding="utf-8"))
        if [r["event_id"] for r in original_score["available"]["horizons"]["3.0"]["rows"]]!=val_ids:
            raise ValueError("A11 source val cache order is not bound")
        with np.load(prefix/"stability_paths.npz",allow_pickle=False) as data:
            source_paths = data["fine"].copy()
        if source_paths.shape!=(88,801,22) or not np.isfinite(source_paths).all():
            raise ValueError("A11 source .005 cache unavailable")
        directory = result_dir/f"source-{seed}-val-0.005"
        directory.mkdir()
        score_and_diagnose("source",seed,.005,val,None,list(source_paths),[{"qualified":False,"pending_saved_physics":True} for _ in val],directory)
        model = counted_model(LearnedNonlinearModel.load(prefix/"model.npz"),budget)
        evaluate("source",seed,.0025,val,model)
    for seed in SEEDS:
        for group in GROUPS:
            if (group,seed) not in checkpoints:
                continue
            path = result_dir/f"candidate-{group}-{seed}/model.npz"
            model = counted_model(LearnedNonlinearModel.load(path),budget)
            for population in (train,val):
                for step in STEPS:
                    evaluate(group,seed,step,population,model)
    # Exact direct/reload pairs: first train, or source first val, at fixed 3 s.
    for group in (*GROUPS,"source"):
        for seed in SEEDS:
            if group!="source" and (group,seed) not in checkpoints:
                continue
            sample = val[0] if group=="source" else train[0]
            path = sources["a06"]/f"checkpoint-240-{seed}/model.npz" if group=="source" else result_dir/f"candidate-{group}-{seed}/model.npz"
            for step in ((.0025,) if group=="source" else STEPS):
                saved = []
                for variant in ("direct","reload"):
                    identity = {"group":group,"seed":seed,"role":sample.role,"max_step":step,"check":variant}
                    ident = f"check-{group}-{seed}-{step}-{variant}"
                    slot = ledger/ident
                    slot.mkdir(exist_ok=False)
                    original = models[seed] if group=="source" else final_models[group,seed]
                    model = counted_model(original if variant=="direct" else LearnedNonlinearModel.load(path),budget)
                    mult = round(.01/step)
                    with budget.segment(identity,"history_integration",40*mult):
                        budget.note_prediction_paths(1)
                        z,hp = model.initialize(sample.history,sample.equilibrium_delta,max_step=step)
                    with budget.segment(identity,"future_integration",1200*mult):
                        fp = model.closure.rollout(sample.history.delta[-1],sample.history.omega[-1],z,
                            np.arange(301)*.01,sample.history.constant_power,sample.equilibrium_delta,max_step=step)
                    with budget.segment(identity,"physical_checks",312):
                        physical = path_physics(model,hp,fp,sample)
                    directory = result_dir/ident
                    directory.mkdir()
                    _npz(directory/"paths.npz",history=hp,future=fp)
                    _write(directory/"physical_checks.json",physical)
                    saved.append((hp,fp,physical))
                    _write(slot/"finished.json",{"status":"completed","reserved_paths":1})
                batch = cohorts[group,seed,sample.role,step]
                diff = float(np.max(np.abs((saved[0][1][:,:20]-batch["future"][0][:301,:20])/model.target_scale)))
                history_diff = (None if batch["history"] is None else float(np.max(np.abs((saved[0][0][:,:20]-batch["history"][0][:,:20])/model.target_scale))))
                exact = all(np.array_equal(a,b) for a,b in zip(saved[0][:2],saved[1][:2],strict=True))
                replay_checks.append({"group":group,"seed":seed,"step":step,"checkpoint_replay_exact":exact,
                    "batch_direct_max_normalized_difference":diff,"history_batch_direct_max_normalized_difference":history_diff,
                    "qualified":exact and diff<=1e-10 and (history_diff is None or history_diff<=1e-10) and all(x[2]["qualified"] for x in saved)})
    for key,left in list(cohorts.items()):
        group,seed,role,step = key
        if step!=.005 or (group,seed,role,.0025) not in cohorts:
            continue
        right = cohorts[group,seed,role,.0025]
        population = train if role=="train" else val
        table = []
        for i,s in enumerate(population):
            n = len(s.forecast_time)
            cp,fp = left["future"][i],right["future"][i]
            maximum = float(np.max(np.abs((cp[:n,:20]-fp[:n,:20])/models[seed].target_scale)))
            margin = compare_on_truth_windows(history=np.c_[s.history.delta,s.history.omega],coarse=cp[:n,:20],fine=fp[:n,:20],
                truth_diagnostic=right["rows"][i]["diagnostic"],equilibrium_delta=s.equilibrium_delta,mass=mass,
                omega_s=provider.static[s.source_dataset]["omega_s"],target_scale=models[seed].target_scale)
            margin_failures = _coarse_fine_margin_failures(
                {"coarse_fine_rule_margin_comparison":margin},
                left["rows"][i]["diagnostic"],right["rows"][i]["diagnostic"])
            qualified = maximum<=1e-5 and not margin_failures
            row = {"event_id":s.event_id,"max_normalized_difference":maximum,"coarse_fine_margin_comparison":margin}
            row["margin_bound_failures"] = margin_failures
            if role=="val":
                args = (s.equilibrium_delta,mass,provider.static[s.source_dataset]["omega_s"],contract)
                cc,fc = classify_los(s.history,cp[:,:20],*args),classify_los(s.history,fp[:,:20],*args)
                stability_numeric = stability_refinement(fp[:,:20],cp[:,:20],fc,cc,contract)
                row.update(coarse_classification=cc,fine_classification=fc,stability_numeric=stability_numeric)
                qualified &= stability_numeric["qualified"]
            row["qualified"] = bool(qualified)
            table.append(row)
        comparisons[group,seed,role] = table
    _write(result_dir/"refinement_checks.json",{"cohorts":[{"group":g,"seed":s,"role":r,"rows":v} for (g,s,r),v in comparisons.items()]})
    _write(result_dir/"replay_checks.json",{"rows":replay_checks})
    _write(result_dir/"trajectory_freeze.json",{"checkpoints":{f"{g}-{s}":h for (g,s),h in checkpoints.items()},
        "trajectory_evidence_files":{p.relative_to(result_dir).as_posix():sha256_file(p) for p in sorted(result_dir.rglob("*")) if p.is_file()},
        "labels_decoded":False,"selected_candidate":None})
    freeze_sha = sha256_file(result_dir/"trajectory_freeze.json")
    frozen_files = json.loads((result_dir/"trajectory_freeze.json").read_text(encoding="utf-8"))["trajectory_evidence_files"]
    state["stage"] = "post_freeze_label_diagnostics"
    metadata = val_truth_metadata(provider)
    _write(result_dir/"coverage.json",{"rows":[{**r,**metadata.get(r["event_id"],{})} for r in provider.coverage],
        "label_decoded_event_ids":sorted(metadata),"label_decoded_roles":["val"],
        "index_file_scans":"CSV rows scanned for IDs; label/time interpreted only for 101 registered internal val IDs"})
    records, baselines, stability = {}, {}, {}
    for seed in SEEDS:
        original = json.loads((sources["a06"]/f"checkpoint-240-{seed}/val_scores.json").read_text(encoding="utf-8"))
        old_physics = json.loads((sources["a06"]/f"checkpoint-240-{seed}/stability.json").read_text(encoding="utf-8"))["rows"]
        if [r["event_id"] for r in old_physics]!=val_ids:
            raise ValueError("A11 cached physical evidence order differs")
        saved_physics_rows = []
        for i,r in enumerate(old_physics):
            checks = [r["physical_checks"][j] for j in (0,2)]
            physical = {"qualified":all(c["qualified"] and math.isfinite(c["energy_identity_max_error"])
                and c["energy_identity_max_error"]<=1e-8 and math.isfinite(c["coi_weighted_sum_max_error"])
                and c["coi_weighted_sum_max_error"]<=1e-10 for c in checks),"source_saved_checks":checks}
            cohorts["source",seed,"val",.005]["rows"][i]["physical"] = physical
            saved_physics_rows.append({"event_id":r["event_id"],**physical})
        _write(result_dir/f"source-{seed}-val-0.005/saved_physics_citation.json", {
            "source_sha256":sha256_file(sources["a06"]/f"checkpoint-240-{seed}/stability.json"),
            "trajectory_freeze_sha256":freeze_sha,"new_rhs":0,"rows":saved_physics_rows})
        for group in (*GROUPS,"source"):
            if (group,seed,"val",.005) not in cohorts:
                continue
            table = []
            for c,s in zip(comparisons[group,seed,"val"],val,strict=True):
                table.append({"event_id":s.event_id,**metadata[s.event_id],**c["coarse_classification"],
                    "prospective_eligible":metadata[s.event_id]["true_event_time_record_s"] is None or metadata[s.event_id]["true_event_time_record_s"]>.1+1e-10,
                    "qualified":c["qualified"] and all(cohorts[group,seed,"val",step]["rows"][len(table)]["physical"]["qualified"] for step in STEPS),
                    "development_evaluation_refused":False})
            los = summarize(table)
            los.update(allocated_events=101,allocated_unstable=27,allocated_stable=74,eligible_events=los["eligible"],
                uncovered_unstable=13,unstable_eligibility_coverage=14/27,development_evaluation_refusals=0)
            stability[group,seed] = {"rows":table,"summary":los,"trajectory_freeze_sha256":freeze_sha}
            main = scores[group,seed,"val",.005]
            base = {"main_nrmse":main["available"]["multi_horizon_normalized_rmse"],
                "complete3_nrmse":main["complete3_common"]["multi_horizon_normalized_rmse"]}
            if group=="source":
                baselines[seed] = {**base,"original_main_nrmse":original["available"]["multi_horizon_normalized_rmse"],
                    "original_complete3_nrmse":original["complete3_common"]["multi_horizon_normalized_rmse"],
                    **classification_counts(los)}
                continue
            train_mse, local = {}, {}
            for step in STEPS:
                score = scores[group,seed,"train",step]
                train_mse[str(step)] = score["available"]["multi_horizon_normalized_rmse"]**2
                stats = local_statistics(event_ids,ws,counts,cohorts[group,seed,"train",step]["rows"])
                local[str(step)] = stats["metrics"]
                _write(result_dir/f"local-{group}-{seed}-{step}.json",stats)
            records[group,seed] = {"val_main_nrmse":base["main_nrmse"],"val_complete3_nrmse":base["complete3_nrmse"],
                **classification_counts(los),
                "train_main_mse_by_step":train_mse,"local_metrics_by_step":local}
            print(f"A11 {group}-{seed}: val={base['main_nrmse']}, FN={los['fn']}, FP={los['fp']}",flush=True)
    snapshot = budget.snapshot()
    complete = (snapshot["attempted"]==snapshot["returned"]==887404644 and snapshot["blocked"]==0
        and snapshot["phases"]["training_integration"]["returned"]==860855040
        and sum(snapshot["phases"][p]["returned"] for p in ("history_integration","future_integration"))==24565488
        and snapshot["physical_checks"]["returned"]==1984116 and snapshot["prediction_paths"]==4446
        and snapshot["rollouts_started"]==8892 and all(r["status"]=="complete" and r["attempted"]==r["returned"]==r["limit"] for r in budget.records))
    gates = {"budget_complete":complete,"all_six_fits_complete":len(checkpoints)==6 and all(r["status"]=="completed" and r["confirmed_optimizer_updates"]==240 for r in fit_records),
        "full_evaluation_population":len(cohorts)==30 and len(comparisons)==15,
        "all_trajectory_checks":all(r["qualified"] for t in comparisons.values() for r in t),
        "all_truth_path_margin_bounds":all(not _diagnostic_margin_failures(r.get("diagnostic"))
            for c in cohorts.values() for r in c["rows"]),
        "source_zero_population_complete":all(
            b["tp"]+b["fn"]==14 and b["tn"]+b["fp"]==74 and b["unresolved"]==0
            and b["eligible_events"]==88 and b["development_evaluation_refusals"]==0
            for b in baselines.values()) and len(baselines)==3,
        "all_physical_checks":all(r["physical"]["qualified"] for c in cohorts.values() for r in c["rows"]),
        "all_direct_replays":len(replay_checks)==15 and all(r["qualified"] for r in replay_checks),
        "all_models_frozen":all(sha256_file(result_dir/f"candidate-{g}-{s}/model.npz")==h for (g,s),h in checkpoints.items())}
    decision = outcome(records,baselines,validity_gates=gates)
    summary = {**decision,"scientific_verdict":decision["campaign_verdict"],"process_status":"PASS",
        "run_id":run_id,"validity_gates":gates,"records":[{"group":g,"seed":s,**v} for (g,s),v in records.items()],
        "baselines":baselines,"fit_records":fit_records,"budget":snapshot,"replay_checks":replay_checks,
        "comparison_freeze_sha256":freeze_sha,"model_promoted":False,"prediction_delivery_count":0,
        "evidence_kind":"uncertified_temporal_training_comparison","zero_scope":"14 eligible unstable val events; 13 of 27 assigned unstable events remain uncovered"}
    _write(result_dir/"learning_summary.json",summary)
    _write(result_dir/"rhs_budget.json",snapshot)
    _write(result_dir/"stability.json",{"cohorts":[{"group":g,"seed":s,**v} for (g,s),v in stability.items()]})
    _write(result_dir/"data_access_audit.json",{**provider.access_audit(),"allocate_roles_called":False,
        "labels_read_only_after_trajectory_freeze_sha256":freeze_sha,"decoded_roles":["train","val"],
        "label_decoded_roles":["val"],"label_decoded_event_ids":sorted(metadata),
        "file_integrity_reads_separately_recorded":"integrity_file_reads.json"})
    if git_head(root)!=commit or status_lines(root) or any(sha256_file(root/p)!=h for p,h in code_hashes.items()):
        raise RuntimeError("A11 implementation changed during execution")
    if (sha256_file(result_dir/"trajectory_freeze.json")!=freeze_sha
            or any(sha256_file(result_dir/p)!=h for p,h in frozen_files.items())):
        raise RuntimeError("A11 pre-label trajectory/model/score evidence changed after its freeze")
    for name,p in sources.items():
        if verify_artifact_directory(p,strict=True)!=manifests[name]:
            raise RuntimeError("A11 source artifacts changed")
    for source,files in bindings.items():
        if any(sha256_file(source_root/source/p)!=h for p,h in files.items()):
            raise RuntimeError("A11 raw source changed")
    _write(result_dir/"lineage.json",{"source_commit":commit,"source_sha256":code_hashes,"wall_seconds":time.perf_counter()-started})
    _write(result_dir/"stage_evidence.json",{"source_commit":commit,"process_status":"PASS",
        "scientific_verdict":summary["scientific_verdict"],"promotion_decision":"KEEP_BASELINE",
        "data_policy":"kiki_original_train_only","source_val_ids_read":[],"test_ids_read":[],
        "hidden_training_targets_read":False,"level_6_claimed":False,"development_authorization_id":AUTH,
        "evidence_kind":"uncertified_temporal_training_comparison","certified_envelope_established":False,
        "deployment_predictions_allowed":False,"model_promoted":False,"claim_boundary":config["claim_boundary"]})
    _write(result_dir/"artifact_hashes.json",{p.relative_to(result_dir).as_posix():sha256_file(p) for p in sorted(result_dir.rglob("*")) if p.is_file() and p.name!="artifact_hashes.json"})
    verify_artifact_directory(result_dir,strict=True)
