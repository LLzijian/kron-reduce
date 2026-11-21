from __future__ import annotations

import csv
import hashlib
import json
import tomllib
from pathlib import Path

import numpy as np

from kron_reduce.data import CanonicalStore, ModelDataRequest, adapt_event, load_split
from kron_reduce.data.audit import audit_model_sample
from kron_reduce.evaluation import evaluate_energy_gate, trajectory_metrics
from kron_reduce.evaluation.physics import numeric_physics_audit
from kron_reduce.models import AutoregressiveLSTMModel, DynamicM2AModel, EnergyM2AModel, GovernorM0Model, M0Model, M2AModel, PINNModel, RunContext, SINDYcModel
from kron_reduce.reporting import write_smoke_report
from kron_reduce.training.lineage import assert_fit_result_lineage, assert_optimizer_lineage


def _sha(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""): digest.update(block)
    return digest.hexdigest()


def _clone_m0(model: M0Model) -> M0Model:
    clone = M0Model(
        model.damping.copy(), model.coupling.copy(),
        selected_ridge=model.selected_ridge,
        max_internal_step=model.max_internal_step,
        diagnostics=dict(model.diagnostics),
    )
    clone.validation_scales = dict(model.validation_scales)
    return clone


def run(*, root: Path, experiment: dict, stage: str, result_dir: Path, run_id: str) -> None:
    if stage != "smoke":
        raise RuntimeError("paper_v1 full execution is intentionally not implemented before its approval gate")
    config_name = experiment.get("config", "paper_smoke.toml")
    config = tomllib.loads((root / "configs" / "experiments" / config_name).read_text(encoding="utf-8"))
    dataset_dir = root / "data" / "ieee39_postclear_canonical_v1"
    store = CanonicalStore.open(dataset_dir)
    split = load_split(dataset_dir, view=config["view"], split_name=config["split_name"], fold=config["fold"])
    assert_optimizer_lineage(split.train_ids, split.val_ids, split.test_ids)
    train_ids = [eid for eid in split.train_ids if store.metadata(eid).valid_length >= config["horizon_steps"]][:config["train_cases"]]
    val_ids = [eid for eid in split.val_ids if store.metadata(eid).valid_length >= config["horizon_steps"]][:config["val_cases"]]
    if len(train_ids) < config["train_cases"] or len(val_ids) < config["val_cases"]:
        raise RuntimeError("smoke data support is insufficient")
    request = ModelDataRequest.initial_state_rollout(pm_mode=config["pm_mode"])
    inertia = store.dataset_metadata.M_prior
    train = [adapt_event(store.load_event(eid), inertia_prior=inertia, request=request, representation=config["representation"]) for eid in train_ids]
    val = [adapt_event(store.load_event(eid), inertia_prior=inertia, request=request, representation=config["representation"]) for eid in val_ids]
    for sample in train + val: audit_model_sample(sample)
    context = RunContext(
        seed=config["seed"], epochs=config["epochs"], horizon_steps=config["horizon_steps"],
        max_cases=config["train_cases"], learning_rate=config.get("learning_rate", 1e-3),
        ridge_values=tuple(config.get("ridge_values", (1e-8, 1e-6, 1e-4, 1e-2))),
        integral_windows=tuple(config.get("integral_windows", (1, 5, 20))),
        sindy_thresholds=tuple(config.get("sindy_thresholds", (0.0, 1e-6, 1e-4, 1e-3, 1e-2))),
        rollout_refine=bool(config.get("rollout_refine", False)),
        max_internal_step=float(config.get("max_internal_step", 0.01)),
    )
    m0=M0Model(); fit_m0=m0.fit(train,val,context)
    available = [
        ("M0",m0,fit_m0,"npz"),
        ("M2A",M2AModel(_clone_m0(m0)),None,"npz"),
        ("M2A-dynamic",DynamicM2AModel(M2AModel(_clone_m0(m0))),None,"npz"),
        ("M0-governor",GovernorM0Model(_clone_m0(m0)),None,"npz"),
        ("M2A-energy",EnergyM2AModel(_clone_m0(m0)),None,"npz"),
        ("SINDYc",SINDYcModel(),None,"npz"),
        ("PINN",PINNModel(_clone_m0(m0)),None,"pt"),
        ("LSTM",AutoregressiveLSTMModel(),None,"pt"),
    ]
    methods = config.get("methods", [row[0] for row in available])
    models = [row for row in available if row[0] in methods]
    summaries=[]; physics=[]; loaded_models={}; model_dir=result_dir/"models"; model_dir.mkdir()
    loaders={"M0":M0Model,"M2A":M2AModel,"M2A-dynamic":DynamicM2AModel,"M0-governor":GovernorM0Model,"M2A-energy":EnergyM2AModel,"SINDYc":SINDYcModel,"PINN":PINNModel,"LSTM":AutoregressiveLSTMModel}
    for name,model,fit,extension in models:
        if fit is None: fit=model.fit(train,val,context)
        assert_fit_result_lineage(fit,split.train_ids,split.val_ids)
        path=model_dir/f"{name}.{extension}"; model.save(path); loaded=loaders[name].load(path); loaded_models[name]=loaded
        target=val[0]["training_context"]["targets"]; steps=config["horizon_steps"]
        try:
            dp,wp=loaded.rollout(val[0]["model_inputs"]["static"],target["t"][:steps],{"Pm":val[0]["model_inputs"]["exogenous"]["Pm"][:steps]})
            metrics=trajectory_metrics(target["delta"][:steps],target["omega"][:steps],dp,wp)
            status="completed" if np.isfinite(metrics["rmse"]) else "attempted_failed_with_evidence"
        except Exception as exc:
            metrics={"rmse":float("nan")}; status="attempted_failed_with_evidence"; metrics["error"]=repr(exc)
        summaries.append({"method":name,"status":status,"best_validation_loss":fit.best_validation_loss,**metrics})
        physics.append({"method":name,"declared":loaded.physics_audit()})
        if hasattr(loaded,"vector_field"):
            physics[-1]["numeric"]=numeric_physics_audit(loaded,val[0])
    if any(row["status"] != "completed" for row in summaries):
        raise RuntimeError("one or more required smoke models failed real closed-loop inference")
    energy=evaluate_energy_gate(loaded_models["M2A-energy"],train+val).to_dict() if "M2A-energy" in loaded_models else {"reported_name":"not_run"}
    if "M2A-energy" in loaded_models: loaded_models["M2A-energy"].save(model_dir/"M2A-energy.npz")
    for row in summaries:
        if row["method"]=="M2A-energy": row["reported_name"]=energy["reported_name"]
    (result_dir/"run_config.json").write_text(json.dumps({"run_id":run_id,"evidence_status":"smoke","config":config,"train_ids":train_ids,"val_ids":val_ids,"test_ids_read":[],"dataset_build_id":store.dataset_metadata.build_id,"input_contract":{"pm_mode":config["pm_mode"],"representation":config["representation"],"forbidden":["Pe","bus_v","bus_a","label","event_time_s","valid_length","valid_mask"]}},indent=2),encoding="utf-8")
    (result_dir/"model_summary.json").write_text(json.dumps(summaries,indent=2,allow_nan=True),encoding="utf-8")
    (result_dir/"physics_audit.json").write_text(json.dumps(physics,indent=2),encoding="utf-8")
    (result_dir/"energy_gate.json").write_text(json.dumps(energy,indent=2),encoding="utf-8")
    with (result_dir/"per_case_metrics.csv").open("w",newline="",encoding="utf-8") as handle:
        fields=sorted({key for row in summaries for key in row}); writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(summaries)
    audit={"status":"PASS","required_models":len(models),"completed_models":sum(row["status"]=="completed" for row in summaries),"test_ids_read":0,"full_run_locked":True}
    (result_dir/"audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    report_path=result_dir/"smoke_report.md"; write_smoke_report(report_path,run_id,summaries,energy)
    artifacts={path.relative_to(result_dir).as_posix():_sha(path) for path in result_dir.rglob("*") if path.is_file() and path.name!="artifact_hashes.json"}
    (result_dir/"artifact_hashes.json").write_text(json.dumps(artifacts,indent=2),encoding="utf-8")
