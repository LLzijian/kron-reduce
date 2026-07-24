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
    GovernorM0Model,
    M0Model,
    M2AModel,
    PINNModel,
    RunContext,
    SINDYcModel,
)
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
    "SINDYc": SINDYcModel,
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
        ("M2A-energy", EnergyM2AModel(_clone_m0(m0))),
        ("SINDYc", SINDYcModel()),
        ("PINN", PINNModel(_clone_m0(m0))),
        ("LSTM", AutoregressiveLSTMModel()),
    ]
    return [(name, model) for name, model in available if name in methods]


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
    if stage != "full_run":
        raise RuntimeError("paper full runner only implements the gated full_run stage")
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
    sensitivity_request = ModelDataRequest.initial_state_rollout(
        config["sensitivity_pm_mode"]
    )
    maximum_horizon = max(config["horizon_steps"])
    model_status: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    exclusion_rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []
    energy_rows: list[dict[str, object]] = []
    selection_records: list[dict[str, object]] = []

    run_config = {
        "run_id": run_id,
        "evidence_status": "full",
        "config": config,
        "dataset_revision": store.dataset_metadata.dataset_revision,
        "dataset_build_id": store.dataset_metadata.build_id,
        "test_access": "locked_until_all_fold_seed_selection_records_exist",
        "input_contract": {
            "main": "initial_state_plus_Pm_tc_ZOH",
            "sensitivity": "initial_state_plus_observed_Pm_conditional",
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
        train_ids = [
            event_id
            for event_id in split.train_ids
            if store.metadata(event_id).valid_length >= maximum_horizon
        ]
        val_ids = [
            event_id
            for event_id in split.val_ids
            if store.metadata(event_id).valid_length >= maximum_horizon
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
            )
            fold_seed_dir = checkpoint_root / fold / str(seed)
            fold_seed_dir.mkdir(parents=True)
            m0 = M0Model()
            try:
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
            if m0_fit is None:
                candidates = [("SINDYc", SINDYcModel()), ("LSTM", AutoregressiveLSTMModel())]
            else:
                candidates = _models(m0, config["methods"])
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
                    model_status.append(
                        {
                            "fold": fold,
                            "seed": seed,
                            "method": name,
                            "status": "completed",
                            "best_validation_loss": fit.best_validation_loss,
                            "checkpoint": checkpoint.relative_to(result_dir).as_posix(),
                            "solution_kind": fit.details.get("solution_kind", ""),
                            "relative_validation_improvement": fit.details.get("relative_validation_improvement", fit.details.get("one_step_relative_improvement", "")),
                            "validation_raw_mean_rmse": fit.details.get("validation_raw_mean_rmse", ""),
                            "maximum_validation_coi_drift": fit.details.get("maximum_validation_coi_drift", ""),
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
                "main_zoh": adapt_event(
                    event,
                    inertia_prior=inertia,
                    request=main_request,
                    representation=config["representation"],
                ),
                "conditional_observed_pm": adapt_event(
                    event,
                    inertia_prior=inertia,
                    request=sensitivity_request,
                    representation=config["representation"],
                ),
            }
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
                                    ]
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
                            if analysis == "main_zoh":
                                stability_rows.append(
                                    {
                                        "fold": fold,
                                        "seed": seed,
                                        "event_id": event_id,
                                        "method": method,
                                        "true_label": meta.outcome_label,
                                        "predicted_label": classify_predicted_trajectory(
                                            predicted_delta, predicted_omega
                                        ),
                                        "evidence_status": "exploratory_from_rollout",
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
            and row["analysis"] == "main_zoh"
            and row["horizon_steps"] == maximum_horizon
        ),
        None,
    )
    final_stability = [row for row in stability_rows if row["method"] == "FINAL"]
    if final_aggregate is None or not final_stability:
        raise RuntimeError("FINAL model did not produce complete acceptance evidence")
    stability_metrics = classification_metrics(
        [str(row["true_label"]) for row in final_stability],
        [str(row["predicted_label"]) for row in final_stability],
    )
    omega_target = float(config.get("target_omega_r2", 0.9))
    fsr_target = float(config.get("target_instability_fsr", 0.2))
    omega_value = float(final_aggregate["pooled_omega_r2"])
    fsr_value = float(stability_metrics["FSR"])
    round_accepted = bool(
        np.isfinite(omega_value) and omega_value > omega_target
        and np.isfinite(fsr_value) and fsr_value < fsr_target
        and float(final_aggregate["failure_rate"]) == 0.0
    )
    acceptance = {
        "decision": "ACCEPT" if round_accepted else "REJECT",
        "method": "FINAL",
        "horizon_steps": maximum_horizon,
        "analysis": "main_zoh",
        "omega_r2": omega_value,
        "omega_r2_strict_target": omega_target,
        "omega_r2_gate_passed": bool(np.isfinite(omega_value) and omega_value > omega_target),
        "instability_FSR": fsr_value,
        "instability_FSR_strict_target": fsr_target,
        "instability_FSR_gate_passed": bool(np.isfinite(fsr_value) and fsr_value < fsr_target),
        "failure_rate": float(final_aggregate["failure_rate"]),
        "failure_gate_passed": float(final_aggregate["failure_rate"]) == 0.0,
        "stability_is_derived_from_final_predicted_trajectories": True,
        "final_model_was_frozen_before_test_access": True,
        "classification_counts": stability_metrics,
    }
    (result_dir / "round_acceptance.json").write_text(
        json.dumps(acceptance, indent=2), encoding="utf-8"
    )
    paired_rows: list[dict[str, object]] = []
    m0_index = {
        (row["fold"], row["seed"], row["event_id"], row["analysis"], row["horizon_steps"]): row
        for row in metric_rows
        if row["method"] == "M0" and row.get("status") == "completed"
    }
    for row in metric_rows:
        if row["method"] != "M2A" or row.get("status") != "completed":
            continue
        key = (row["fold"], row["seed"], row["event_id"], row["analysis"], row["horizon_steps"])
        if key in m0_index:
            paired_rows.append(
                {
                    "fault_line": row["fault_line"],
                    "rmse_difference_M2A_minus_M0": row["rmse"] - m0_index[key]["rmse"],
                }
            )
    bootstrap = fault_line_cluster_bootstrap(
        paired_rows,
        "rmse_difference_M2A_minus_M0",
        n_bootstrap=config["bootstrap_draws"],
        seed=config["seeds"][0],
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
