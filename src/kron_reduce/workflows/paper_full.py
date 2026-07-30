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
from kron_reduce.evaluation.metrics import aggregate_trajectory_metrics
from kron_reduce.evaluation.stability import classify_predicted_trajectory, classification_metrics
from kron_reduce.evaluation.statistics import fault_line_cluster_bootstrap
from kron_reduce.models import (
    AutoregressiveLSTMModel,
    DynamicM2AModel,
    EnergyM2AModel,
    DerivativeM0Model,
    GraphM0Model,
    GovernorM0Model,
    M0Model,
    M2AModel,
    PINNModel,
    RunContext,
    SINDYcModel,
    SwingSINDyModel,
    ThirdOrderM0Model,
    StructuredThirdOrderM0Model,
    LargeSignalThirdOrderM0Model,
    TransientEmfM0Model,
)
from kron_reduce.models.base import FitResult
from kron_reduce.training.lineage import (
    assert_fit_result_lineage,
    assert_optimizer_lineage,
)


LOADERS = {
    "M0": M0Model,
    "M2A": M2AModel,
    "M2A-energy": EnergyM2AModel,
    "M2A-dynamic": DynamicM2AModel,
    "M0-governor": GovernorM0Model,
    "M0-DD": DerivativeM0Model,
    "M0-GRAPH": GraphM0Model,
    "SINDYc": SINDYcModel,
    "Swing-SINDy": SwingSINDyModel,
    "M0-3": ThirdOrderM0Model,
    "M0-3S": StructuredThirdOrderM0Model,
    "M0-3L": LargeSignalThirdOrderM0Model,
    "M0-3E": TransientEmfM0Model,
    "PINN": PINNModel,
    "LSTM": AutoregressiveLSTMModel,
}
EXTENSIONS = {"PINN": "pt", "LSTM": "pt"}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
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


def _models(m0: M0Model, methods: list[str]) -> list[tuple[str, object]]:
    static_m2a = M2AModel(_clone_m0(m0))
    available = [
        ("M0", m0),
        ("M2A", static_m2a),
        ("M2A-dynamic", DynamicM2AModel(M2AModel(_clone_m0(m0)))),
        ("M0-governor", GovernorM0Model(_clone_m0(m0))),
        ("M0-DD", DerivativeM0Model()),
        ("M0-GRAPH", GraphM0Model()),
        ("M0-3", ThirdOrderM0Model()),
        ("M0-3S", StructuredThirdOrderM0Model()),
        ("M0-3L", LargeSignalThirdOrderM0Model()),
        ("M0-3E", TransientEmfM0Model()),
        ("M2A-energy", EnergyM2AModel(_clone_m0(m0))),
        ("SINDYc", SINDYcModel()),
        ("Swing-SINDy", SwingSINDyModel()),
        ("PINN", PINNModel(_clone_m0(m0))),
        ("LSTM", AutoregressiveLSTMModel()),
    ]
    return [(name, model) for name, model in available if name in methods]


def fixed_causal_classification_inputs(
    static: dict[str, np.ndarray], *, steps: int, dt_s: float
) -> tuple[np.ndarray, dict[str, np.ndarray | str]]:
    """Build a label- and truncation-independent causal stability rollout."""
    if steps < 2 or not np.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("classification rollout requires steps >= 2 and dt_s > 0")
    pm_clear = np.asarray(static["Pm_clear"], dtype=np.float64).reshape(1, -1)
    time_grid = np.arange(steps, dtype=np.float64) * float(dt_s)
    return time_grid, {
        "Pm": np.repeat(pm_clear, steps, axis=0),
        "interpolation": "zoh",
    }


def _load_frozen_m0(
    *, root: Path, config: dict, fold: str, seed: int,
    train: list[dict], val: list[dict], context: RunContext,
    split,
) -> tuple[M0Model, FitResult, dict[str, object]]:
    source_dir = root / str(config["frozen_m0_run"])
    checkpoint = source_dir / "checkpoints" / fold / str(seed) / "M0.npz"
    metadata = checkpoint.with_suffix(".json")
    artifact_manifest = json.loads(
        (source_dir / "artifact_hashes.json").read_text(encoding="utf-8")
    )
    checkpoint_key = checkpoint.relative_to(source_dir).as_posix()
    metadata_key = metadata.relative_to(source_dir).as_posix()
    for path, key in ((checkpoint, checkpoint_key), (metadata, metadata_key)):
        if key not in artifact_manifest or _sha(path) != artifact_manifest[key]:
            raise RuntimeError(f"frozen M0 artifact hash mismatch: {key}")
    records = json.loads(
        (source_dir / "selection_complete.json").read_text(encoding="utf-8")
    )
    source = next(
        (row for row in records if row["fold"] == fold and int(row["seed"]) == seed),
        None,
    )
    if source is None:
        raise RuntimeError(f"frozen M0 source lineage missing for {fold}/{seed}")
    source_train_ids = tuple(source["train_ids"])
    source_val_ids = tuple(source["val_ids"])
    assert_optimizer_lineage(source_train_ids, source_val_ids, split.test_ids)
    train_by_id = {sample["event_id"]: sample for sample in train}
    scale_samples = [train_by_id[event_id] for event_id in source_train_ids]
    model = M0Model.load(checkpoint)
    model.validation_scales = M0Model._training_scales(scale_samples, len(scale_samples))
    model.max_internal_step = context.max_internal_step
    audit = model.physics_audit()
    if audit.get("status") != "pass" or not audit.get("nonnegative_parameters"):
        raise RuntimeError(f"frozen M0 failed physical audit for {fold}/{seed}")
    validation = model.validation_summary(val, context)
    details = {
        "optimizer": "none_frozen_checkpoint",
        "solution_kind": "frozen_round1_M0",
        "validation_raw_mean_rmse": validation["raw_mean_rmse"],
        "validation_raw_median_rmse": validation["raw_median_rmse"],
        "maximum_validation_coi_drift": validation["maximum_coi_drift"],
        "source_run": str(config["frozen_m0_run"]),
        "source_checkpoint": checkpoint_key,
        "source_checkpoint_sha256": artifact_manifest[checkpoint_key],
        "source_train_ids": list(source_train_ids),
        "source_val_ids": list(source_val_ids),
    }
    result = FitResult(
        "M0", "completed", source_train_ids, source_val_ids,
        validation["standardized_loss"], details,
    )
    return model, result, details


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(*, root: Path, experiment: dict, stage: str, result_dir: Path, run_id: str) -> None:
    if stage not in {"validation_run", "full_run"}:
        raise RuntimeError(
            "paper full runner implements validation_run and gated full_run stages"
        )
    config_name = experiment.get("config", "paper_full.toml")
    config = tomllib.loads(
        (root / "configs" / "experiments" / config_name).read_text(
            encoding="utf-8"
        )
    )
    dataset_dir = root / "data" / "ieee39_postclear_canonical_v1"
    store = CanonicalStore.open(dataset_dir)
    inertia = store.dataset_metadata.M_prior
    main_request = ModelDataRequest.initial_state_rollout(config["main_pm_mode"])
    sensitivity_mode = config.get("sensitivity_pm_mode")
    sensitivity_request = (
        ModelDataRequest.initial_state_rollout(sensitivity_mode)
        if sensitivity_mode is not None
        else None
    )
    maximum_horizon = max(config["horizon_steps"])
    primary_analysis = str(
        config.get(
            "primary_analysis",
            "main_zoh"
            if config["main_pm_mode"] == "zoh"
            else "conditional_observed_pm",
        )
    )
    model_status: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    exclusion_rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []
    energy_rows: list[dict[str, object]] = []
    selection_records: list[dict[str, object]] = []
    frozen_m0_rows: list[dict[str, object]] = []
    validation_metric_rows: list[dict[str, object]] = []

    run_config = {
        "run_id": run_id,
        "evidence_status": "validation_only" if stage == "validation_run" else "full",
        "config": config,
        "dataset_revision": store.dataset_metadata.dataset_revision,
        "dataset_build_id": store.dataset_metadata.build_id,
        "test_access": "locked_until_all_fold_seed_selection_records_exist",
        "input_contract": {
            "main": (
                "initial_state_plus_Pm_clear_ZOH_causal"
                if config["main_pm_mode"] == "zoh"
                else "initial_state_plus_observed_Pm_conditional_noncausal"
            ),
            "sensitivity": (
                None
                if sensitivity_mode is None
                else "initial_state_plus_observed_Pm_conditional"
            ),
            "forbidden": [
                "Pe",
                "bus_v",
                "bus_a",
                "label",
                "event_time_s",
                "valid_length",
                "valid_mask",
                "tail_statistics",
            ],
        },
    }
    (result_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )
    checkpoint_root = result_dir / "checkpoints"
    checkpoint_root.mkdir()

    selected_models: dict[tuple[str, int], dict[str, object]] = {}
    fold_splits = {}
    for fold in config["folds"]:
        split = load_split(
            dataset_dir,
            view=config["view"],
            split_name=config["split_name"],
            fold=fold,
        )
        fold_splits[fold] = split
        assert_optimizer_lineage(split.train_ids, split.val_ids, split.test_ids)
        minimum_fit_steps = int(config.get("minimum_fit_steps", maximum_horizon))
        train_ids = [
            event_id
            for event_id in split.train_ids
            if store.metadata(event_id).valid_length >= minimum_fit_steps
        ]
        val_ids = [
            event_id
            for event_id in split.val_ids
            if store.metadata(event_id).valid_length >= minimum_fit_steps
        ]
        for role, all_ids, eligible_ids in (
            ("train", split.train_ids, train_ids),
            ("val", split.val_ids, val_ids),
        ):
            for event_id in sorted(set(all_ids) - set(eligible_ids)):
                exclusion_rows.append(
                    {
                        "fold": fold,
                        "role": role,
                        "event_id": event_id,
                        "reason": "insufficient_real_horizon",
                    }
                )
        if not train_ids or not val_ids:
            raise RuntimeError(f"{fold}: no eligible train or validation support")
        train = [
            adapt_event(
                store.load_event(event_id),
                inertia_prior=inertia,
                request=main_request,
                representation=config["representation"],
            )
            for event_id in train_ids
        ]
        val = [
            adapt_event(
                store.load_event(event_id),
                inertia_prior=inertia,
                request=main_request,
                representation=config["representation"],
            )
            for event_id in val_ids
        ]
        for sample in train + val:
            audit_model_sample(sample)
        for seed in config["seeds"]:
            context = RunContext(
                seed=seed,
                epochs=config["epochs"],
                horizon_steps=maximum_horizon,
                max_cases=max(len(train), len(val)),
                learning_rate=config["learning_rate"],
                ridge_values=tuple(config.get("ridge_values", (1e-8, 1e-6, 1e-4, 1e-2))),
                integral_windows=tuple(config.get("integral_windows", (1, 5, 20))),
                sindy_thresholds=tuple(config.get("sindy_thresholds", (0.0, 1e-6, 1e-4, 1e-3, 1e-2))),
                rollout_refine=bool(config.get("rollout_refine", False)),
                refinement_min_improvement=float(config.get("refinement_min_improvement", 0.02)),
                max_internal_step=float(config.get("max_internal_step", 0.01)),
                m2a_rollout_shortlist=int(config.get("m2a_rollout_shortlist", 4)),
                m2a_residual_scales=tuple(
                    config.get("m2a_residual_scales", (0.01, 0.02, 0.05, 0.1))
                ),
                m2a_validation_cvar_fraction=float(
                    config.get("m2a_validation_cvar_fraction", 0.1)
                ),
                m2a_max_cvar_degradation=float(
                    config.get("m2a_max_cvar_degradation", 0.0)
                ),
                m2a_enforce_dissipative_linear_terms=bool(
                    config.get("m2a_enforce_dissipative_linear_terms", False)
                ),
                graph_prune_thresholds=tuple(
                    config.get("graph_prune_thresholds", (0.0, 1e-3, 1e-2))
                ),
                graph_nonoverlap_windows=bool(
                    config.get("graph_nonoverlap_windows", True)
                ),
                fit_max_steps=int(config.get("fit_max_steps", 0)),
            )
            fold_seed_dir = checkpoint_root / fold / str(seed)
            fold_seed_dir.mkdir(parents=True)
            m0 = M0Model()
            try:
                if config.get("frozen_m0_run"):
                    m0, m0_fit, frozen_details = _load_frozen_m0(
                        root=root, config=config, fold=fold, seed=seed,
                        train=train, val=val, context=context, split=split,
                    )
                    frozen_m0_rows.append({"fold": fold, "seed": seed, **frozen_details})
                else:
                    m0_fit = m0.fit(train, val, context)
                assert_fit_result_lineage(m0_fit, split.train_ids, split.val_ids)
            except Exception as exc:
                m0_fit = None
                model_status.append(
                    {
                        "fold": fold,
                        "seed": seed,
                        "method": "M0",
                        "status": "failed",
                        "error": repr(exc),
                    }
                )
                if config.get("frozen_m0_run"):
                    raise RuntimeError(
                        f"{fold}/{seed}: frozen M0 load or lineage validation failed"
                    ) from exc
            if m0_fit is None:
                candidates = [("SINDYc", SINDYcModel()), ("LSTM", AutoregressiveLSTMModel())]
            else:
                candidates = _models(m0, config["methods"])
            frozen_damping = None if m0.damping is None else m0.damping.copy()
            frozen_coupling = None if m0.coupling is None else m0.coupling.copy()
            fitted: dict[str, object] = {}
            for name, model in candidates:
                try:
                    fit = m0_fit if name == "M0" else model.fit(train, val, context)
                    assert fit is not None
                    assert_fit_result_lineage(fit, split.train_ids, split.val_ids)
                    extension = EXTENSIONS.get(name, "npz")
                    checkpoint = fold_seed_dir / f"{name}.{extension}"
                    model.save(checkpoint)
                    loaded = LOADERS[name].load(checkpoint)
                    fitted[name] = loaded
                    physics_audit = loaded.physics_audit()
                    if frozen_damping is not None and (
                        not np.array_equal(m0.damping, frozen_damping)
                        or not np.array_equal(m0.coupling, frozen_coupling)
                    ):
                        raise RuntimeError("M2A candidate mutated the frozen M0 parameters")
                    model_status.append(
                        {
                            "fold": fold,
                            "seed": seed,
                            "method": name,
                            "status": "completed",
                            "best_validation_loss": fit.best_validation_loss,
                            "checkpoint": checkpoint.relative_to(result_dir).as_posix(),
                            "solution_kind": fit.details.get("solution_kind", ""),
                            "feature_version": fit.details.get("feature_version", ""),
                            "active_parameters": fit.details.get("active_parameters", ""),
                            "residual_scale": fit.details.get("residual_scale", ""),
                            "relative_validation_improvement": fit.details.get("relative_validation_improvement", fit.details.get("one_step_relative_improvement", "")),
                            "validation_raw_mean_rmse": fit.details.get("validation_raw_mean_rmse", ""),
                            "maximum_validation_coi_drift": fit.details.get("maximum_validation_coi_drift", ""),
                            "physics_audit_status": physics_audit.get("status", ""),
                            "selected_ridge": fit.details.get("selected_ridge", ""),
                            "selected_prune_threshold": fit.details.get("selected_prune_threshold", ""),
                            "minimum_node_damping": fit.details.get("minimum_node_damping", ""),
                            "minimum_edge_damping": fit.details.get("minimum_edge_damping", ""),
                            "minimum_local_stiffness": fit.details.get("minimum_local_stiffness", ""),
                            "minimum_quotient_damping_eigenvalue": fit.details.get("minimum_quotient_damping_eigenvalue", ""),
                            "minimum_quotient_stiffness_eigenvalue": fit.details.get("minimum_quotient_stiffness_eigenvalue", ""),
                            "maximum_linearized_mode_real_part": fit.details.get("maximum_linearized_mode_real_part", ""),
                            "active_edge_count": fit.details.get("active_edge_count", ""),
                            "matrix_condition_number": fit.details.get("matrix_condition_number", ""),
                            "effective_rank": fit.details.get("effective_rank", ""),
                        }
                    )
                except Exception as exc:
                    model_status.append(
                        {
                            "fold": fold,
                            "seed": seed,
                            "method": name,
                            "status": "failed",
                            "error": repr(exc),
                        }
                    )
            # Validation evidence is materialized before any test event is
            # opened.  It uses the same declared exogenous policy as fitting;
            # for round 3 this is conditional observed Pm with linear RK-stage
            # interpolation.  Labels are not read or used for selection.
            for method, model in fitted.items():
                for sample in val:
                    target = sample["training_context"]["targets"]
                    static = sample["model_inputs"]["static"]
                    claim_type = sample["training_context"]["claim_type"]
                    interpolation = (
                        "zoh" if claim_type == "causal_zoh_rollout" else "linear"
                    )
                    rollout_steps = min(len(target["t"]), maximum_horizon)
                    metadata = sample["training_context"]["metadata"]
                    try:
                        predicted_delta, predicted_omega = model.rollout(
                            static,
                            target["t"][:rollout_steps],
                            {
                                "Pm": sample["model_inputs"]["exogenous"]["Pm"][:rollout_steps],
                                "interpolation": interpolation,
                            },
                        )
                        for horizon in config["horizon_steps"]:
                            if len(target["t"]) < horizon:
                                exclusion_rows.append(
                                    {
                                        "fold": fold,
                                        "role": "val",
                                        "event_id": sample["event_id"],
                                        "method": method,
                                        "analysis": primary_analysis,
                                        "horizon_steps": horizon,
                                        "reason": "insufficient_real_horizon",
                                    }
                                )
                                continue
                            metrics = trajectory_metrics(
                                target["delta"][:horizon],
                                target["omega"][:horizon],
                                predicted_delta[:horizon],
                                predicted_omega[:horizon],
                            )
                            validation_metric_rows.append(
                                {
                                    "fold": fold,
                                    "seed": seed,
                                    "event_id": sample["event_id"],
                                    "fault_line": metadata["fault_line"],
                                    "method": method,
                                    "analysis": primary_analysis,
                                    "horizon_steps": horizon,
                                    "status": "completed",
                                    **metrics,
                                }
                            )
                    except Exception as exc:
                        validation_metric_rows.append(
                            {
                                "fold": fold,
                                "seed": seed,
                                "event_id": sample["event_id"],
                                "fault_line": metadata["fault_line"],
                                "method": method,
                                "analysis": primary_analysis,
                                "horizon_steps": maximum_horizon,
                                "status": "failed",
                                "error": repr(exc),
                            }
                        )
            if "M2A-energy" in fitted:
                gate = evaluate_energy_gate(fitted["M2A-energy"], train + val).to_dict()
                energy_rows.append({"fold": fold, "seed": seed, **gate})
                # Persist the gate-selected public name in the immutable run
                # checkpoint; the numerical parameters are unchanged.
                fitted["M2A-energy"].save(fold_seed_dir / "M2A-energy.npz")
            selection_records.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "train_ids": train_ids,
                    "val_ids": val_ids,
                    "test_ids_read": [],
                    "completed_methods": sorted(fitted),
                    "selection_status": "frozen_before_test_access",
                }
            )
            selected_models[(fold, seed)] = fitted

    governor_rows = [
        row for row in model_status
        if row.get("method") == "M0-governor" and row.get("status") == "completed"
        and isinstance(row.get("relative_validation_improvement"), (int, float))
    ]
    fold_improvements = {
        fold: float(np.mean([
            row["relative_validation_improvement"] for row in governor_rows
            if row["fold"] == fold
        ]))
        for fold in config["folds"]
        if any(row["fold"] == fold for row in governor_rows)
    }
    governor_gate = {
        "criterion": "H100_validation_mean_improvement_ge_2pct_in_at_least_4_of_5_folds_without_failure_increase",
        "fold_relative_improvements": fold_improvements,
        "same_direction_fold_count": sum(value > 0 for value in fold_improvements.values()),
        "mean_relative_improvement": float(np.mean(list(fold_improvements.values()))) if fold_improvements else float("nan"),
        "failure_count": sum(row.get("method") == "M0-governor" and row.get("status") == "failed" for row in model_status),
    }
    governor_gate["effective"] = bool(
        len(fold_improvements) == 5
        and governor_gate["same_direction_fold_count"] >= 4
        and governor_gate["mean_relative_improvement"] >= 0.02
        and governor_gate["failure_count"] == 0
    )
    (result_dir / "governor_ablation_gate.json").write_text(
        json.dumps(governor_gate, indent=2), encoding="utf-8"
    )
    m0_rows = [row for row in model_status if row.get("method") == "M0" and row.get("status") == "completed" and isinstance(row.get("validation_raw_mean_rmse"), (int, float))]
    m0_fold_rmse = {fold: float(np.mean([row["validation_raw_mean_rmse"] for row in m0_rows if row["fold"] == fold])) for fold in config["folds"] if any(row["fold"] == fold for row in m0_rows)}
    old_baseline = config.get("old_m0_h100_mean_rmse")
    fivefold_mean = float(np.mean(list(m0_fold_rmse.values()))) if m0_fold_rmse else float("nan")
    deterministic_gate = {
        "criterion": "F1_H100_mean_le_0.5_and_fivefold_mean_reduction_ge_80pct_and_COI_drift_lt_1e-10",
        "fold_validation_raw_mean_rmse": m0_fold_rmse,
        "fivefold_validation_raw_mean_rmse": fivefold_mean,
        "old_m0_h100_mean_rmse": old_baseline,
        "relative_reduction": (old_baseline-fivefold_mean)/old_baseline if old_baseline else None,
        "maximum_validation_coi_drift": max((row["maximum_validation_coi_drift"] for row in m0_rows), default=float("nan")),
    }
    deterministic_gate["passed"] = bool(len(m0_fold_rmse)==5 and m0_fold_rmse.get("F1",float("inf"))<=0.5 and old_baseline and deterministic_gate["relative_reduction"]>=0.8 and deterministic_gate["maximum_validation_coi_drift"]<1e-10)
    (result_dir / "deterministic_repair_gate.json").write_text(json.dumps(deterministic_gate,indent=2),encoding="utf-8")
    if frozen_m0_rows:
        (result_dir / "frozen_m0_lineage.json").write_text(
            json.dumps(frozen_m0_rows, indent=2), encoding="utf-8"
        )

    # Freeze one deployable physics model per fold/seed using validation only.
    # Test trajectories remain unopened until this mapping is durable.
    final_selections = []
    configured_final = list(config.get("final_candidate_methods", ["M0", "M2A", "M2A-dynamic", "M2A-energy"]))
    if governor_gate["effective"] and "M0-governor" not in configured_final:
        configured_final.append("M0-governor")
    for (fold, seed), fitted in selected_models.items():
        eligible = [
            row for row in model_status
            if row.get("fold") == fold and row.get("seed") == seed
            and row.get("method") in configured_final
            and row.get("status") == "completed"
            and row.get("method") in fitted
            and np.isfinite(float(row.get("best_validation_loss", np.inf)))
        ]
        if not eligible:
            raise RuntimeError(f"{fold}/{seed}: no validation-eligible final physics model")
        chosen = min(eligible, key=lambda row: float(row["best_validation_loss"]))
        source_method = str(chosen["method"])
        fitted["FINAL"] = fitted[source_method]
        selection = {
            "fold": fold,
            "seed": seed,
            "source_method": source_method,
            "best_validation_loss": float(chosen["best_validation_loss"]),
            "candidate_methods": sorted(row["method"] for row in eligible),
            "test_ids_read": [],
            "selection_status": "frozen_before_test_access",
        }
        final_selections.append(selection)
        for record in selection_records:
            if record["fold"] == fold and record["seed"] == seed:
                record["final_source_method"] = source_method
                record["final_validation_loss"] = selection["best_validation_loss"]
                break
    (result_dir / "final_model_selection.json").write_text(
        json.dumps(final_selections, indent=2), encoding="utf-8"
    )

    # This durable marker is written only after every fold/seed has completed
    # train/validation model selection. Test trajectories are first opened below.
    (result_dir / "selection_complete.json").write_text(
        json.dumps(selection_records, indent=2), encoding="utf-8"
    )

    # FINAL validation rows are exact copies of the already evaluated source
    # model selected within each fold.  No trajectory is recomputed and no
    # test identifier or test array is touched here.
    for selection in final_selections:
        source_rows = [
            row for row in validation_metric_rows
            if row["fold"] == selection["fold"]
            and row["seed"] == selection["seed"]
            and row["method"] == selection["source_method"]
        ]
        validation_metric_rows.extend(
            {**row, "method": "FINAL", "source_method": selection["source_method"]}
            for row in source_rows
        )

    validation_aggregate_rows: list[dict[str, object]] = []
    validation_groups = sorted({
        (row["method"], row["analysis"], row["horizon_steps"])
        for row in validation_metric_rows
    })
    for method, analysis, horizon in validation_groups:
        group = [
            row for row in validation_metric_rows
            if row["method"] == method
            and row["analysis"] == analysis
            and row["horizon_steps"] == horizon
        ]
        validation_aggregate_rows.append(
            {
                "method": method,
                "analysis": analysis,
                "horizon_steps": horizon,
                "population": len(group),
                **aggregate_trajectory_metrics(group),
            }
        )
    validation_final = next(
        (
            row for row in validation_aggregate_rows
            if row["method"] == "FINAL"
            and row["analysis"] == primary_analysis
            and row["horizon_steps"] == maximum_horizon
        ),
        None,
    )
    validation_m0 = next(
        (
            row for row in validation_aggregate_rows
            if row["method"] == "M0"
            and row["analysis"] == primary_analysis
            and row["horizon_steps"] == maximum_horizon
        ),
        None,
    )
    if validation_final is None or validation_m0 is None:
        raise RuntimeError("missing primary H100 validation evidence")
    validation_m0_index = {
        (row["fold"], row["seed"], row["event_id"]): row
        for row in validation_metric_rows
        if row["method"] == "M0"
        and row["analysis"] == primary_analysis
        and row["horizon_steps"] == maximum_horizon
        and row.get("status") == "completed"
    }
    validation_pairs = []
    for row in validation_metric_rows:
        if not (
            row["method"] == "FINAL"
            and row["analysis"] == primary_analysis
            and row["horizon_steps"] == maximum_horizon
            and row.get("status") == "completed"
        ):
            continue
        key = (row["fold"], row["seed"], row["event_id"])
        if key in validation_m0_index:
            validation_pairs.append(
                {
                    "fault_line": row["fault_line"],
                    "rmse_difference_FINAL_minus_M0": (
                        row["rmse"] - validation_m0_index[key]["rmse"]
                    ),
                }
            )
    validation_bootstrap = fault_line_cluster_bootstrap(
        validation_pairs,
        "rmse_difference_FINAL_minus_M0",
        n_bootstrap=config["bootstrap_draws"],
        seed=config["seeds"][0],
    )
    validation_relative_improvement = (
        float(validation_m0["mean_rmse"]) - float(validation_final["mean_rmse"])
    ) / max(float(validation_m0["mean_rmse"]), np.finfo(float).eps)
    validation_cvar_degradation = (
        float(validation_final["worst_10pct_cvar_rmse"])
        - float(validation_m0["worst_10pct_cvar_rmse"])
    ) / max(float(validation_m0["worst_10pct_cvar_rmse"]), np.finfo(float).eps)
    fold_relative_improvements = {}
    for fold in config["folds"]:
        final_fold = [
            float(row["rmse"]) for row in validation_metric_rows
            if row["fold"] == fold and row["method"] == "FINAL"
            and row["analysis"] == primary_analysis
            and row["horizon_steps"] == maximum_horizon
            and row.get("status") == "completed"
        ]
        m0_fold = [
            float(row["rmse"]) for row in validation_metric_rows
            if row["fold"] == fold and row["method"] == "M0"
            and row["analysis"] == primary_analysis
            and row["horizon_steps"] == maximum_horizon
            and row.get("status") == "completed"
        ]
        if final_fold and m0_fold:
            fold_relative_improvements[fold] = (
                float(np.mean(m0_fold)) - float(np.mean(final_fold))
            ) / max(float(np.mean(m0_fold)), np.finfo(float).eps)
    same_direction_fold_count = sum(value > 0.0 for value in fold_relative_improvements.values())
    required_source_method = config.get("validation_required_source_method")
    required_source_fold_count = sum(
        selection["source_method"] == required_source_method
        for selection in final_selections
    ) if required_source_method else 0
    graph_audit_rows = [
        row for row in model_status
        if row.get("method") == "M0-GRAPH" and row.get("status") == "completed"
    ]
    graph_physics_audits_passed = bool(
        graph_audit_rows
        and all(row.get("physics_audit_status") == "pass" for row in graph_audit_rows)
    )
    validation_gate = {
        "stage": "conditional_observed_pm_accuracy_before_governor_ode",
        "analysis": primary_analysis,
        "horizon_steps": maximum_horizon,
        "conditional_noncausal_upper_bound": config["main_pm_mode"] == "observed",
        "test_data_read": False,
        "omega_r2": float(validation_final["pooled_omega_r2"]),
        "omega_r2_target": float(config.get("target_omega_r2", 0.9)),
        "relative_mean_rmse_improvement_vs_M0": validation_relative_improvement,
        "minimum_relative_improvement": float(
            config.get("validation_min_improvement", 0.02)
        ),
        "failure_rate": float(validation_final["failure_rate"]),
        "worst_10pct_cvar_relative_degradation_vs_M0": validation_cvar_degradation,
        "maximum_cvar_relative_degradation": float(
            config.get("validation_max_cvar_degradation", np.inf)
        ),
        "fold_relative_improvements_vs_M0": fold_relative_improvements,
        "same_direction_fold_count": same_direction_fold_count,
        "minimum_same_direction_fold_count": int(
            config.get("validation_min_same_direction_folds", 0)
        ),
        "required_source_method": required_source_method,
        "required_source_fold_count": required_source_fold_count,
        "minimum_required_source_fold_count": int(
            config.get("validation_required_source_fold_count", 0)
        ),
        "graph_physics_audits_passed": graph_physics_audits_passed,
        "cluster_bootstrap": validation_bootstrap,
        "stability_labels_used_for_selection": False,
        "governor_ode_permitted_after_this_gate_only": True,
    }
    validation_gate["passed"] = bool(
        np.isfinite(validation_gate["omega_r2"])
        and validation_gate["omega_r2"] > validation_gate["omega_r2_target"]
        and validation_relative_improvement
        >= validation_gate["minimum_relative_improvement"]
        and validation_gate["failure_rate"] == 0.0
        and validation_bootstrap.get("status") == "completed"
        and float(validation_bootstrap.get("ci95_high", np.inf)) < 0.0
        and validation_cvar_degradation
        <= float(config.get("validation_max_cvar_degradation", np.inf))
        and same_direction_fold_count
        >= int(config.get("validation_min_same_direction_folds", 0))
        and (
            not required_source_method
            or required_source_fold_count
            >= int(config.get("validation_required_source_fold_count", 0))
        )
        and (
            not bool(config.get("validation_require_graph_physics_audits", False))
            or graph_physics_audits_passed
        )
    )
    _write_csv(result_dir / "validation_per_case_metrics.csv", validation_metric_rows)
    _write_csv(result_dir / "validation_aggregate_metrics.csv", validation_aggregate_rows)
    (result_dir / "validation_cluster_bootstrap.json").write_text(
        json.dumps(validation_bootstrap, indent=2), encoding="utf-8"
    )
    (result_dir / "validation_gate.json").write_text(
        json.dumps(validation_gate, indent=2), encoding="utf-8"
    )

    if stage == "validation_run":
        run_config["test_access"] = "remained_locked_validation_only"
        run_config["test_ids_read"] = []
        (result_dir / "run_config.json").write_text(
            json.dumps(run_config, indent=2), encoding="utf-8"
        )
        _write_csv(result_dir / "model_status.csv", model_status)
        _write_csv(result_dir / "exclusions.csv", exclusion_rows)
        _write_csv(result_dir / "energy_gate.csv", energy_rows)
        audit = {
            "status": "PASS",
            "validation_gate_passed": bool(validation_gate["passed"]),
            "test_opened": False,
            "test_ids_read": [],
            "stability_labels_used_for_selection": False,
            "completed_model_fits": sum(
                row["status"] == "completed" for row in model_status
            ),
            "failed_model_fits": sum(
                row["status"] == "failed" for row in model_status
            ),
        }
        (result_dir / "audit.json").write_text(
            json.dumps(audit, indent=2), encoding="utf-8"
        )
        artifacts = {
            path.relative_to(result_dir).as_posix(): _sha(path)
            for path in result_dir.rglob("*")
            if path.is_file() and path.name != "artifact_hashes.json"
        }
        (result_dir / "artifact_hashes.json").write_text(
            json.dumps(artifacts, indent=2), encoding="utf-8"
        )
        return

    if (
        bool(config.get("require_validation_gate_before_test", False))
        and not validation_gate["passed"]
    ):
        raise RuntimeError(
            "validation accuracy gate failed; test access remains locked and Pm ODE is not permitted"
        )

    classification_steps = int(config.get("classification_horizon_steps", maximum_horizon))
    classification_dt_s = float(config.get("classification_dt_s", 0.01))
    for fold in config["folds"]:
        split = fold_splits[fold]
        for event_id in split.test_ids:
            meta = store.metadata(event_id)
            if meta.valid_length < 2:
                exclusion_rows.append(
                    {
                        "fold": fold,
                        "role": "test",
                        "event_id": event_id,
                        "reason": "insufficient_real_horizon",
                    }
                )
                continue
            event = store.load_event(event_id)
            samples = {
                primary_analysis: adapt_event(
                    event,
                    inertia_prior=inertia,
                    request=main_request,
                    representation=config["representation"],
                )
            }
            if sensitivity_request is not None:
                sensitivity_analysis = str(
                    config.get("sensitivity_analysis", "conditional_observed_pm")
                )
                if sensitivity_analysis != primary_analysis:
                    samples[sensitivity_analysis] = adapt_event(
                        event,
                        inertia_prior=inertia,
                        request=sensitivity_request,
                        representation=config["representation"],
                    )
            for sample in samples.values():
                audit_model_sample(sample)
            for seed in config["seeds"]:
                fitted = selected_models[(fold, seed)]
                for method, model in fitted.items():
                    for analysis, sample in samples.items():
                        target = sample["training_context"]["targets"]
                        rollout_steps = min(len(target["t"]), maximum_horizon)
                        try:
                            predicted_delta, predicted_omega = model.rollout(
                                sample["model_inputs"]["static"],
                                target["t"][:rollout_steps],
                                {
                                    "Pm": sample["model_inputs"]["exogenous"]["Pm"][
                                        :rollout_steps
                                    ],
                                    "interpolation": (
                                        "zoh"
                                        if sample["training_context"]["claim_type"]
                                        == "causal_zoh_rollout"
                                        else "linear"
                                    ),
                                },
                            )
                            for horizon in config["horizon_steps"]:
                                if len(target["t"]) < horizon:
                                    exclusion_rows.append(
                                        {
                                            "fold": fold,
                                            "role": "test",
                                            "event_id": event_id,
                                            "method": method,
                                            "analysis": analysis,
                                            "horizon_steps": horizon,
                                            "reason": "insufficient_real_horizon",
                                        }
                                    )
                                    continue
                                metrics = trajectory_metrics(
                                    target["delta"][:horizon],
                                    target["omega"][:horizon],
                                    predicted_delta[:horizon],
                                    predicted_omega[:horizon],
                                )
                                metric_rows.append(
                                    {
                                        "fold": fold,
                                        "seed": seed,
                                        "event_id": event_id,
                                        "fault_line": meta.fault_line,
                                        "outcome_label": meta.outcome_label,
                                        "method": method,
                                        "analysis": analysis,
                                        "horizon_steps": horizon,
                                        "status": "completed",
                                        **metrics,
                                    }
                                )
                        except Exception as exc:
                            metric_rows.append(
                                {
                                    "fold": fold,
                                    "seed": seed,
                                    "event_id": event_id,
                                    "fault_line": meta.fault_line,
                                    "outcome_label": meta.outcome_label,
                                    "method": method,
                                    "analysis": analysis,
                                    "horizon_steps": maximum_horizon,
                                    "status": "failed",
                                    "error": repr(exc),
                                }
                            )
                    if not bool(config.get("evaluate_stability", True)):
                        continue
                    try:
                        stability_t, stability_exogenous = fixed_causal_classification_inputs(
                            samples[primary_analysis]["model_inputs"]["static"],
                            steps=classification_steps,
                            dt_s=classification_dt_s,
                        )
                        stability_delta, stability_omega = model.rollout(
                            samples[primary_analysis]["model_inputs"]["static"],
                            stability_t,
                            stability_exogenous,
                        )
                        stability_rows.append(
                            {
                                "fold": fold,
                                "seed": seed,
                                "event_id": event_id,
                                "method": method,
                                "true_label": meta.outcome_label,
                                "predicted_label": classify_predicted_trajectory(
                                    stability_delta, stability_omega
                                ),
                                "classification_horizon_steps": classification_steps,
                                "classification_dt_s": classification_dt_s,
                                "status": "completed",
                                "evidence_status": "exploratory_fixed_causal_rollout",
                            }
                        )
                    except Exception as exc:
                        stability_rows.append(
                            {
                                "fold": fold,
                                "seed": seed,
                                "event_id": event_id,
                                "method": method,
                                "true_label": meta.outcome_label,
                                "predicted_label": "",
                                "classification_horizon_steps": classification_steps,
                                "classification_dt_s": classification_dt_s,
                                "status": "failed",
                                "error": repr(exc),
                                "evidence_status": "exploratory_fixed_causal_rollout",
                            }
                        )

    aggregate_rows: list[dict[str, object]] = []
    completed_groups = sorted(
        {
            (row["method"], row["analysis"], row["horizon_steps"])
            for row in metric_rows
        }
    )
    for method, analysis, horizon in completed_groups:
        group = [
            row
            for row in metric_rows
            if row["method"] == method
            and row["analysis"] == analysis
            and row["horizon_steps"] == horizon
        ]
        aggregate_rows.append(
            {
                "method": method,
                "analysis": analysis,
                "horizon_steps": horizon,
                "population": len(group),
                **aggregate_trajectory_metrics(group),
            }
        )
    final_aggregate = next(
        (
            row for row in aggregate_rows
            if row["method"] == "FINAL"
            and row["analysis"] == primary_analysis
            and row["horizon_steps"] == maximum_horizon
        ),
        None,
    )
    m0_aggregate = next(
        (
            row for row in aggregate_rows
            if row["method"] == "M0"
            and row["analysis"] == primary_analysis
            and row["horizon_steps"] == maximum_horizon
        ),
        None,
    )
    final_stability_all = [row for row in stability_rows if row["method"] == "FINAL"]
    final_stability = [
        row for row in final_stability_all if row.get("status") == "completed"
    ]
    evaluate_stability = bool(config.get("evaluate_stability", True))
    if final_aggregate is None or m0_aggregate is None:
        raise RuntimeError("FINAL model did not produce complete acceptance evidence")
    if evaluate_stability and not final_stability:
        raise RuntimeError("FINAL model did not produce stability evidence")
    stability_metrics = (
        classification_metrics(
            [str(row["true_label"]) for row in final_stability],
            [str(row["predicted_label"]) for row in final_stability],
        )
        if evaluate_stability
        else {}
    )
    test_m0_index = {
        (row["fold"], row["seed"], row["event_id"]): row
        for row in metric_rows
        if row["method"] == "M0"
        and row["analysis"] == primary_analysis
        and row["horizon_steps"] == maximum_horizon
        and row.get("status") == "completed"
    }
    paired_rows: list[dict[str, object]] = []
    for row in metric_rows:
        if not (
            row["method"] == "FINAL"
            and row["analysis"] == primary_analysis
            and row["horizon_steps"] == maximum_horizon
            and row.get("status") == "completed"
        ):
            continue
        key = (row["fold"], row["seed"], row["event_id"])
        if key in test_m0_index:
            paired_rows.append(
                {
                    "fault_line": row["fault_line"],
                    "rmse_difference_FINAL_minus_M0": (
                        row["rmse"] - test_m0_index[key]["rmse"]
                    ),
                }
            )
    bootstrap = fault_line_cluster_bootstrap(
        paired_rows,
        "rmse_difference_FINAL_minus_M0",
        n_bootstrap=config["bootstrap_draws"],
        seed=config["seeds"][0],
    )
    omega_target = float(config.get("target_omega_r2", 0.9))
    fsr_target = float(config.get("target_instability_fsr", 0.2))
    omega_value = float(final_aggregate["pooled_omega_r2"])
    fsr_value = float(stability_metrics.get("FSR", np.nan))
    stability_failure_rate = (
        1.0 - len(final_stability) / len(final_stability_all)
        if evaluate_stability else float("nan")
    )
    relative_improvement = (
        float(m0_aggregate["mean_rmse"]) - float(final_aggregate["mean_rmse"])
    ) / max(float(m0_aggregate["mean_rmse"]), np.finfo(float).eps)
    accuracy_accepted = bool(
        np.isfinite(omega_value) and omega_value > omega_target
        and relative_improvement >= float(config.get("validation_min_improvement", 0.02))
        and float(final_aggregate["failure_rate"]) == 0.0
        and bootstrap.get("status") == "completed"
        and float(bootstrap.get("ci95_high", np.inf)) < 0.0
    )
    round_accepted = bool(
        accuracy_accepted
        and (
            not evaluate_stability
            or (
                np.isfinite(fsr_value) and fsr_value < fsr_target
                and stability_failure_rate == 0.0
            )
        )
    )
    acceptance = {
        "decision": "ACCEPT" if round_accepted else "REJECT",
        "method": "FINAL",
        "horizon_steps": maximum_horizon,
        "analysis": primary_analysis,
        "conditional_noncausal_upper_bound": config["main_pm_mode"] == "observed",
        "omega_r2": omega_value,
        "omega_r2_strict_target": omega_target,
        "omega_r2_gate_passed": bool(np.isfinite(omega_value) and omega_value > omega_target),
        "relative_mean_rmse_improvement_vs_M0": relative_improvement,
        "minimum_relative_improvement": float(
            config.get("validation_min_improvement", 0.02)
        ),
        "relative_improvement_gate_passed": bool(
            relative_improvement
            >= float(config.get("validation_min_improvement", 0.02))
        ),
        "cluster_bootstrap": bootstrap,
        "cluster_bootstrap_gate_passed": bool(
            bootstrap.get("status") == "completed"
            and float(bootstrap.get("ci95_high", np.inf)) < 0.0
        ),
        "stability_evaluated": evaluate_stability,
        "instability_FSR": fsr_value,
        "instability_FSR_strict_target": fsr_target,
        "instability_FSR_gate_passed": bool(
            not evaluate_stability
            or (np.isfinite(fsr_value) and fsr_value < fsr_target)
        ),
        "failure_rate": float(final_aggregate["failure_rate"]),
        "failure_gate_passed": float(final_aggregate["failure_rate"]) == 0.0,
        "classification_failure_rate": stability_failure_rate,
        "classification_failure_gate_passed": bool(
            not evaluate_stability or stability_failure_rate == 0.0
        ),
        "classification_horizon_steps": classification_steps,
        "classification_dt_s": classification_dt_s,
        "classification_is_independent_of_real_valid_length": (
            True if evaluate_stability else None
        ),
        "stability_is_derived_from_final_predicted_trajectories": True,
        "final_model_was_frozen_before_test_access": True,
        "classification_counts": stability_metrics,
    }
    (result_dir / "round_acceptance.json").write_text(
        json.dumps(acceptance, indent=2), encoding="utf-8"
    )
    completed_model_count = sum(row["status"] == "completed" for row in model_status)
    if completed_model_count == 0:
        raise RuntimeError("all full-run models failed; no eligible result exists")
    run_config["test_access"] = "unlocked_after_selection_complete"
    run_config["test_ids_read"] = sorted(
        {event_id for split in fold_splits.values() for event_id in split.test_ids}
    )
    (result_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )
    _write_csv(result_dir / "model_status.csv", model_status)
    _write_csv(result_dir / "per_case_metrics.csv", metric_rows)
    _write_csv(result_dir / "aggregate_metrics.csv", aggregate_rows)
    _write_csv(result_dir / "stability_exploratory.csv", stability_rows)
    _write_csv(result_dir / "exclusions.csv", exclusion_rows)
    _write_csv(result_dir / "energy_gate.csv", energy_rows)
    _write_csv(result_dir / "final_stability_metrics.csv", [{"method": "FINAL", **stability_metrics}])
    (result_dir / "final_minus_m0_cluster_bootstrap.json").write_text(
        json.dumps(bootstrap, indent=2), encoding="utf-8"
    )
    # Backward-compatible artifact name retained for existing postprocessors.
    (result_dir / "m2a_minus_m0_cluster_bootstrap.json").write_text(
        json.dumps(bootstrap, indent=2), encoding="utf-8"
    )
    audit = {
        "status": "PASS" if all(row["status"] == "completed" for row in model_status) else "COMPLETED_WITH_MODEL_FAILURES",
        "test_opened_after_selection_marker": True,
        "completed_model_fits": completed_model_count,
        "failed_model_fits": sum(row["status"] == "failed" for row in model_status),
        "zero_metrics_used_for_failures": False,
        "stability_is_exploratory": True,
    }
    (result_dir / "audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    artifacts = {
        path.relative_to(result_dir).as_posix(): _sha(path)
        for path in result_dir.rglob("*")
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    (result_dir / "artifact_hashes.json").write_text(
        json.dumps(artifacts, indent=2), encoding="utf-8"
    )
