from __future__ import annotations

import csv
import hashlib
import json
import tomllib
from collections import defaultdict
from collections.abc import Iterable, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from kron_reduce.models import (
    M0Model,
    ModalResonantPortM0Model,
    PortStructuredDiscoveryM0Model,
    TransientEmfM0Model,
)
from kron_reduce.workflows.adaptive_fidelity_structure_discovery import _context
from kron_reduce.workflows.m0_afsd_v2_module_selection import (
    _module_mask,
    _paired_cluster_bootstrap,
    _residual_features,
)
from kron_reduce.workflows.m0_afsd_v6_full_model_subset_validation import (
    _detailed_aggregate,
    _evaluate_candidate,
    _load_v5_data,
    _modal_match_rows,
    _sign_flip_test,
    deterministic_fault_line_roles,
)


WORKFLOW = "m0_afsd_v7_modal_structure_discovery"
ROLLOUT_LABEL = "conditional rollout with observed Pm"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in materialized for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in materialized:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list, tuple)) else value
                for key, value in row.items()
            })


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_hashes(result_dir: Path) -> None:
    artifacts = {
        str(path.relative_to(result_dir)).replace("\\", "/"): _sha256(path)
        for path in sorted(result_dir.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    _json(result_dir/"artifact_hashes.json", artifacts)


def _verify_hashes(result_dir: Path) -> list[str]:
    expected = json.loads((result_dir/"artifact_hashes.json").read_text(encoding="utf-8"))
    return [
        relative for relative, digest in expected.items()
        if not (result_dir/relative).is_file() or _sha256(result_dir/relative) != digest
    ]


def _latest_result(root: Path, experiment: str, stage: str) -> Path:
    candidates = sorted(
        path for path in (root/"results").glob(f"{experiment}-{stage}-*")
        if path.is_dir() and not (path/"RUN_FAILED.json").exists()
    )
    if not candidates:
        raise FileNotFoundError(f"no completed v7 result for {experiment}/{stage}")
    return candidates[-1]


def _validate_scope(config: dict[str, Any]) -> None:
    expected = {
        "fold": "F1", "seed": 324, "pm_mode": "observed",
        "test_access": "forbidden", "other_folds_access": "forbidden",
        "canonical_mutation": "forbidden", "rollout_label": ROLLOUT_LABEL,
    }
    mismatch = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
    if mismatch:
        raise RuntimeError(f"v7 scope violation: {mismatch}")


def _branch(mode_pairs: tuple[tuple[float, float], ...]) -> str:
    return "SS-DM" if not mode_pairs else "SS-DM+"+"+".join(f"R{f:g}z{z:g}" for f, z in mode_pairs)


def _mode_candidates(config: dict[str, Any]) -> tuple[tuple[tuple[float, float], ...], ...]:
    return tuple(
        tuple((float(pair[0]), float(pair[1])) for pair in candidate)
        for candidate in config["resonance_mode_candidates"]
    )


def _hypothesis_registry(config: dict[str, Any], run_id: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "workflow": WORKFLOW,
        "registered_before_F1_validation_metrics": True,
        "primary_hypothesis": (
            "a passive second-order edge port with a genuine complex-conjugate pole can reduce "
            "the trajectory-frequency-damping tradeoff left by the accepted SS-DM model"
        ),
        "mechanism": {
            "q_dot": "p",
            "p_dot": "nu-omega0^2*q-2*zeta*omega0*p",
            "edge_force": "h*p, h>=0",
            "storage": "h/2*(p^2+omega0^2*q^2)",
            "supply_identity": "Hdot=h*p*nu-2*h*zeta*omega0*p^2",
        },
        "baseline": {
            "name": "SS-DM",
            "refit_from_canonical_train": True,
            "historical_checkpoint_read": False,
            "alpha": float(config["baseline_alpha"]),
            "tau": float(config["baseline_tau"]),
            "memory_taus": config["baseline_memory_taus"],
        },
        "candidate_grid": [[list(pair) for pair in candidate] for candidate in _mode_candidates(config)],
        "grid_is_train_pilot_informed": True,
        "selection_rule": (
            "development-selection 10 s RMSE improves >=2%; empirical frequency and damping do not "
            "worsen; MAC does not worsen; zero failures and physics pass; then minimum active state count"
        ),
        "confirmation_rule": (
            "fault-line-disjoint train audit and frozen F1 validation must retain >=2% 10 s improvement, "
            "positive 95% fault-line bootstrap lower bound, p<=0.05 sign flip, zero failures, RK4/COI/physics pass"
        ),
        "modal_evidence_policy": {
            "exact_DMD": "empirical forced-response descriptor only",
            "single_event_DMDc": "rejected in pre-registration pilot as unstable",
            "pooled_DMDc": "rejected because a common local A was unsupported on reference paths",
            "absolute_gate": {"frequency_relative_error": 0.10, "damping_absolute_error": 0.05, "MAC": 0.80},
        },
        "claim_boundaries": config["claim_boundaries"],
        "test_ids_read": [], "other_folds_read": [],
    }


def _run_hypothesis(config_path: Path, config: dict[str, Any], result_dir: Path, run_id: str) -> None:
    (result_dir/"frozen_config.toml").write_bytes(config_path.read_bytes())
    _json(result_dir/"hypothesis_registry.json", _hypothesis_registry(config, run_id))
    _json(result_dir/"lineage.json", {
        "run_id": run_id, "stage": "hypothesis_registration",
        "historical_result_artifacts_read": [], "test_ids_read": [],
    })
    _write_hashes(result_dir)


def _fit_m0(
    train: list[dict[str, Any]],
    selection: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    frozen_ridge: float | None = None,
) -> M0Model:
    context = _context(config, train)
    if frozen_ridge is None:
        model = M0Model(max_internal_step=float(config["max_internal_step"]))
        model.fit(train, selection, context)
        return model
    matrix, vector, details = M0Model.build_weak_system(
        train,
        windows=context.integral_windows,
        max_cases=context.max_cases,
        horizon_steps=context.horizon_steps,
        max_fit_steps=context.fit_max_steps,
        max_windows_per_case_horizon=context.weak_max_windows_per_case_horizon,
    )
    theta, solver = M0Model._solve_nonnegative_ridge(matrix, vector, float(frozen_ridge))
    n_nodes = len(np.asarray(train[0]["model_inputs"]["static"]["mass"]))
    model = M0Model(
        damping=theta[:n_nodes], coupling=theta[n_nodes:],
        selected_ridge=float(frozen_ridge),
        max_internal_step=float(config["max_internal_step"]),
        diagnostics={**details, **solver, "fixed_hyperparameter_refit": True},
    )
    model.validation_scales = M0Model._training_scales(train, context.max_cases)
    return model


def _fit_ssdm(
    train: list[dict[str, Any]],
    selection: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    frozen_ridge: float | None = None,
) -> PortStructuredDiscoveryM0Model:
    m0 = _fit_m0(train, selection, config, frozen_ridge=frozen_ridge)
    electrical = TransientEmfM0Model(
        anchor=deepcopy(m0),
        alpha=float(config["baseline_alpha"]),
        tau=float(config["baseline_tau"]),
        activation_threshold=0.0,
        activation_width=float(config["b3_activation_width"]),
        max_internal_step=float(config["max_internal_step"]),
        diagnostics={"maximum_equilibrium_residual": 0.0, "hyperparameters_frozen_from_v6": True},
    )
    taus = tuple(float(value) for value in config["baseline_memory_taus"])
    n_edges = len(np.asarray(m0.coupling))
    model = PortStructuredDiscoveryM0Model(
        backbone=electrical,
        static_coefficients=np.zeros((len(PortStructuredDiscoveryM0Model.STATIC_FEATURES), n_edges)),
        taus=np.asarray(taus),
        memory_coefficients=np.zeros((len(taus), n_edges)),
        max_internal_step=float(config["max_internal_step"]),
    )
    context = _context(config, train)
    prepared = model._prepare_weak_paths(
        train, horizon_steps=context.horizon_steps, max_fit_steps=context.fit_max_steps
    )
    stats = model._build_normal_equations(prepared, taus, windows=context.integral_windows)
    active = _module_mask(n_edges, len(taus), {"dissipative", "memory"})
    theta, _, solver = model._factorized_nonnegative_ridge(
        stats["gram"], stats["response"], stats["target_energy"], stats["effective_blocks"],
        float(config["baseline_psd_ridge"]), active,
    )
    model._set_candidate({"n_edges": n_edges, "fit_taus": taus, "theta": theta})
    storage = model._conservative_storage_audit(train+selection)
    model.diagnostics = {
        "enabled_modules": ["dissipative", "memory"],
        "selected_support": model._support_from_theta(theta, n_edges, len(taus)),
        "active_coefficients_after_refit": int(np.sum(theta > max(np.max(theta)*1e-12, np.finfo(float).eps))),
        "storage_audit": storage,
        "coefficient_refit_solver": solver,
        "fixed_fit_taus": list(taus),
        "fixed_ridge": float(config["baseline_psd_ridge"]),
    }
    return model


def _median_modal(rows: Sequence[dict[str, Any]]) -> dict[str, float | int]:
    if not rows:
        return {"matched_modes": 0, "frequency_error": float("inf"), "damping_error": float("inf"), "MAC": 0.0}
    return {
        "matched_modes": len(rows),
        "frequency_error": float(np.median([float(row["relative_frequency_error"]) for row in rows])),
        "damping_error": float(np.median([float(row["absolute_damping_ratio_error"]) for row in rows])),
        "MAC": float(np.median([float(row["mac"]) for row in rows])),
    }


def _candidate_evidence(
    baseline: PortStructuredDiscoveryM0Model,
    fit_samples: list[dict[str, Any]],
    evaluation_samples: list[dict[str, Any]],
    mode_pairs: tuple[tuple[float, float], ...],
    config: dict[str, Any],
    *,
    cohort: str,
) -> tuple[ModalResonantPortM0Model, list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    model = ModalResonantPortM0Model(
        backbone=deepcopy(baseline), mode_pairs=mode_pairs,
        max_internal_step=float(config["max_internal_step"]),
    )
    weak = model.fit_fixed_modes(
        fit_samples, mode_pairs, ridge=float(config["resonance_ridge"]),
        windows=tuple(int(value) for value in config["integral_windows"]),
        horizon_steps=int(config["selection_horizon_steps"]),
        max_fit_steps=int(config["fit_max_steps"]),
    )
    branch = _branch(mode_pairs)
    rows = _evaluate_candidate(
        model, "generator_10", evaluation_samples, evaluation_samples, None,
        tuple(int(value) for value in config["report_horizons"]),
        cohort=cohort, branch=branch,
    )
    aggregate = _detailed_aggregate(rows)
    modal_rows = _modal_match_rows(
        model, evaluation_samples, (1000,), branch=branch, population=cohort
    )
    summary = {
        "branch": branch, "mode_pairs": [list(pair) for pair in mode_pairs],
        "weak_fit": weak, "physics": model.physics_audit(),
        "rollout_10s": next(row for row in aggregate if int(row["horizon_steps"]) == 1000),
        "modal_10s": _median_modal(modal_rows),
    }
    return model, rows, summary, modal_rows


def _select_candidate(summaries: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    baseline = next(row for row in summaries if row["branch"] == "SS-DM")
    base_rmse = float(baseline["rollout_10s"]["mean_rmse"])
    base_modal = baseline["modal_10s"]
    eligible = []
    for row in summaries:
        rmse = float(row["rollout_10s"]["mean_rmse"])
        improvement = (base_rmse-rmse)/max(base_rmse, np.finfo(float).eps)
        modal = row["modal_10s"]
        gates = {
            "minimum_rollout_improvement": improvement >= float(config["minimum_model_improvement"]),
            "zero_failures": float(row["rollout_10s"]["failure_rate"]) == 0.0,
            "physics": row["physics"]["status"] == "pass",
            "frequency_not_worse": float(modal["frequency_error"]) <= float(base_modal["frequency_error"])+1e-12,
            "damping_not_worse": float(modal["damping_error"]) <= float(base_modal["damping_error"])+1e-12,
            "MAC_not_worse": float(modal["MAC"]) >= float(base_modal["MAC"])-float(config["maximum_mac_degradation"])-1e-12,
        }
        row["relative_rollout_improvement"] = improvement
        row["selection_gates"] = gates
        if all(gates.values()):
            eligible.append(row)
    selected = min(
        eligible,
        key=lambda row: (len(row["mode_pairs"]), float(row["rollout_10s"]["mean_rmse"]), row["branch"]),
    ) if eligible else baseline
    return {
        "status": "SELECTED" if eligible else "FALLBACK_SELECTED",
        "route": selected["branch"], "mode_pairs": selected["mode_pairs"],
        "baseline": baseline, "selected": selected,
        "eligible_routes": [row["branch"] for row in eligible],
    }


def _run_controlled(
    root: Path, config_path: Path, config: dict[str, Any], result_dir: Path, run_id: str
) -> None:
    hypothesis = _latest_result(root, str(config["hypothesis_experiment"]), "hypothesis_registration")
    if _verify_hashes(hypothesis):
        raise RuntimeError("v7 hypothesis artifacts changed before controlled experiment")
    registry = json.loads((hypothesis/"hypothesis_registry.json").read_text(encoding="utf-8"))
    if not registry["registered_before_F1_validation_metrics"]:
        raise RuntimeError("v7 hypothesis was not registered")
    (result_dir/"frozen_config.toml").write_bytes(config_path.read_bytes())
    data = _load_v5_data(root, config)
    qa_rows = [
        row for row in data["rows"]
        if row["role"] == "train" and bool(row["eligible_for_v5_selection"])
        and bool(row["eligible_h1000"])
    ]
    roles = deterministic_fault_line_roles(
        qa_rows, seed=int(config["seed"]),
        fit_fraction=float(config["development_fit_fraction"]),
        selection_fraction=float(config["development_selection_fraction"]),
    )
    row_by_id = {str(row["event_id"]): row for row in qa_rows}
    qa_ids = set(row_by_id)
    qa = [sample for sample in data["train"] if str(sample["event_id"]) in qa_ids]
    populations = {
        role: [sample for sample in qa if roles[str(row_by_id[str(sample["event_id"])] ["fault_line"])] == role]
        for role in ("fit", "selection", "audit")
    }
    baseline_dev = _fit_ssdm(populations["fit"], populations["selection"], config)
    dev_models, dev_rows, dev_summaries, dev_modal = {}, [], [], []
    for mode_pairs in _mode_candidates(config):
        model, rows, summary, modal = _candidate_evidence(
            baseline_dev, populations["fit"], populations["selection"], mode_pairs, config,
            cohort="development_selection",
        )
        dev_models[summary["branch"]] = model
        dev_rows.extend(rows); dev_summaries.append(summary); dev_modal.extend(modal)
    selection = _select_candidate(dev_summaries, config)
    selected_pairs = tuple(tuple(map(float, pair)) for pair in selection["mode_pairs"])
    audit_models, audit_rows, audit_summaries, audit_modal = {}, [], [], []
    for mode_pairs in ((), selected_pairs) if selected_pairs else ((),):
        model, rows, summary, modal = _candidate_evidence(
            baseline_dev, populations["fit"], populations["audit"], mode_pairs, config,
            cohort="development_audit",
        )
        audit_models[summary["branch"]] = model
        audit_rows.extend(rows); audit_summaries.append(summary); audit_modal.extend(modal)
    frozen_ridge = float(baseline_dev.backbone.anchor.selected_ridge)
    baseline_final = _fit_ssdm(qa, populations["selection"], config, frozen_ridge=frozen_ridge)
    final_models, validation_rows, validation_summaries, validation_modal = {}, [], [], []
    validation = list(data["validation"])
    for mode_pairs in ((), selected_pairs) if selected_pairs else ((),):
        model, rows, summary, modal = _candidate_evidence(
            baseline_final, qa, validation, mode_pairs, config,
            cohort="frozen_F1_validation",
        )
        final_models[summary["branch"]] = model
        validation_rows.extend(rows); validation_summaries.append(summary); validation_modal.extend(modal)
    selected_route = _branch(selected_pairs)
    selected_h1000 = [row for row in validation_rows if row["branch"] == selected_route and int(row["horizon_steps"]) == 1000]
    baseline_h1000 = [row for row in validation_rows if row["branch"] == "SS-DM" and int(row["horizon_steps"]) == 1000]
    statistics = {
        "bootstrap": _paired_cluster_bootstrap(
            selected_h1000, baseline_h1000,
            draws=int(config["performance_bootstrap_draws"]),
            confidence=float(config["bootstrap_confidence"]), seed=int(config["seed"]),
        ),
        "sign_flip": _sign_flip_test(selected_h1000, baseline_h1000, seed=int(config["seed"])),
    }
    convergence = {}
    selected_model = final_models[selected_route]
    for sample in validation[:3]:
        target = sample["training_context"]["targets"]
        steps = min(1000, len(target["t"]))
        convergence[str(sample["event_id"])] = selected_model.convergence_audit(
            sample["model_inputs"]["static"], np.asarray(target["t"][:steps]),
            {"Pm": np.asarray(sample["model_inputs"]["exogenous"]["Pm"][:steps]), "interpolation": "linear"},
        )
    residual_rows = []
    for route, model in final_models.items():
        for sample in validation:
            if len(sample["training_context"]["targets"]["t"]) >= 1000:
                residual_rows.append({"route": route, **_residual_features(model, sample, sample, None, 1000)})
    checkpoints = result_dir/"checkpoints"; checkpoints.mkdir()
    for route, model in final_models.items():
        model.save(checkpoints/f"final-{route}.npz")
    marker = {
        "run_id": run_id, "selected_route": selected_route,
        "selected_mode_pairs": [list(pair) for pair in selected_pairs],
        "development_selection": selection,
        "development_audit": audit_summaries,
        "validation_did_not_change_selection": True,
        "baseline_refit_from_canonical_train": True,
        "historical_checkpoint_read": False,
        "rollout_label": ROLLOUT_LABEL,
        "test_ids_read": [], "other_folds_read": [],
    }
    _json(result_dir/"selection_marker.json", marker)
    _json(result_dir/"candidate_summaries.json", {"selection": dev_summaries, "audit": audit_summaries, "validation": validation_summaries})
    _json(result_dir/"performance_statistics.json", statistics)
    _json(result_dir/"physics_audits.json", {route: model.physics_audit() for route, model in final_models.items()})
    _json(result_dir/"integration_convergence.json", convergence)
    _csv(result_dir/"data_qualification.csv", data["rows"])
    _csv(result_dir/"fault_line_development_roles.csv", [{"fault_line": line, "role": role} for line, role in sorted(roles.items())])
    _csv(result_dir/"development_per_case_metrics.csv", dev_rows+audit_rows)
    _csv(result_dir/"development_modal_matches.csv", dev_modal+audit_modal)
    _csv(result_dir/"validation_per_case_metrics.csv", validation_rows)
    _csv(result_dir/"validation_aggregate_metrics.csv", _detailed_aggregate(validation_rows))
    _csv(result_dir/"validation_modal_matches.csv", validation_modal)
    _csv(result_dir/"residual_per_case.csv", residual_rows)
    _json(result_dir/"lineage.json", {
        "run_id": run_id, "stage": "controlled_experiment",
        "hypothesis_result": hypothesis.name,
        "hypothesis_sha256": _sha256(hypothesis/"hypothesis_registry.json"),
        "historical_result_artifacts_read": [],
        "canonical_data_mutated": False, "test_ids_read": [], "other_folds_read": [],
    })
    _write_hashes(result_dir)


def _run_residual(root: Path, config: dict[str, Any], result_dir: Path, run_id: str) -> None:
    controlled = _latest_result(root, str(config["controlled_experiment"]), "controlled_experiment")
    if _verify_hashes(controlled):
        raise RuntimeError("controlled artifacts changed before v7 residual diagnosis")
    rows = _read_csv(controlled/"residual_per_case.csv")
    summary = {}
    for route in sorted({row["route"] for row in rows}):
        scoped = [row for row in rows if row["route"] == route and row.get("status") == "completed"]
        summary[route] = {
            "n_cases": len(scoped),
            "median_lag1": float(np.median([float(row["omega_residual_lag1"]) for row in scoped])) if scoped else None,
            "median_lag20": float(np.median([float(row["omega_residual_lag20"]) for row in scoped])) if scoped else None,
            "median_stable_rank": float(np.median([float(row["omega_residual_stable_rank"]) for row in scoped])) if scoped else None,
            "median_low_frequency_fraction": float(np.median([float(row["omega_residual_low_frequency_fraction"]) for row in scoped])) if scoped else None,
        }
    _json(result_dir/"residual_diagnosis.json", {
        "run_id": run_id, "routes": summary,
        "interpretation": "residual persistence is unresolved dynamics evidence, not permission to change the frozen branch",
        "test_ids_read": [],
    })
    _json(result_dir/"lineage.json", {"run_id": run_id, "stage": "residual_diagnosis", "controlled_result": controlled.name, "test_ids_read": []})
    _write_hashes(result_dir)


def _run_adversarial(root: Path, config: dict[str, Any], result_dir: Path, run_id: str) -> None:
    controlled = _latest_result(root, str(config["controlled_experiment"]), "controlled_experiment")
    residual = _latest_result(root, str(config["residual_experiment"]), "residual_diagnosis")
    if _verify_hashes(controlled) or _verify_hashes(residual):
        raise RuntimeError("v7 parent artifacts changed before adversarial review")
    marker = json.loads((controlled/"selection_marker.json").read_text(encoding="utf-8"))
    roles = _read_csv(controlled/"fault_line_development_roles.csv")
    by_line = defaultdict(set)
    for row in roles:
        by_line[row["fault_line"]].add(row["role"])
    checks = {
        "hypothesis_precedes_validation": json.loads((controlled/"lineage.json").read_text(encoding="utf-8"))["hypothesis_result"].startswith(str(config["hypothesis_experiment"])),
        "fault_line_roles_disjoint": all(len(value) == 1 for value in by_line.values()),
        "validation_frozen": bool(marker["validation_did_not_change_selection"]),
        "fresh_baseline_refit": bool(marker["baseline_refit_from_canonical_train"] and not marker["historical_checkpoint_read"]),
        "test_locked": marker["test_ids_read"] == [],
        "other_folds_locked": marker["other_folds_read"] == [],
        "observed_pm_named_conditional": marker["rollout_label"] == ROLLOUT_LABEL,
    }
    alternatives = [
        {"threat": "resonator only retunes stiffness", "disposition": "rejected algebraically: module has a complex pole and zero static gain from edge speed to force"},
        {"threat": "ordinary DMD confounds observed Pm forcing", "disposition": "retained; DMD is named empirical, while single-event and pooled DMDc failures are disclosed"},
        {"threat": "candidate grid was selected after a train pilot", "disposition": "retained and marked pilot-informed; frozen F1 validation is the first confirmation set"},
        {"threat": "edge resonator may represent node/controller physics", "disposition": "retained as a structure-equivalence class; no device semantic claim"},
        {"threat": "observed Pm hides deployment error", "disposition": f"retained; result is {ROLLOUT_LABEL}"},
        {"threat": "single F1 result is source-specific", "disposition": "retained; no other-fold or unknown-system claim"},
    ]
    review = {"run_id": run_id, "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "alternative_explanations": alternatives, "test_ids_read": []}
    _json(result_dir/"adversarial_review.json", review)
    _json(result_dir/"lineage.json", {"run_id": run_id, "stage": "adversarial_review", "controlled_result": controlled.name, "residual_result": residual.name, "test_ids_read": []})
    _write_hashes(result_dir)
    if review["status"] != "PASS":
        raise RuntimeError("v7 adversarial review failed")


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _write_report(result_dir: Path, controlled: Path, acceptance: dict[str, Any]) -> None:
    validation = _read_csv(controlled/"validation_aggregate_metrics.csv")
    selected = acceptance["selected_route"]
    lines = [
        "# M0-AFSD v7：被动复极点模块与模态识别改进报告", "",
        f"本报告仅覆盖 F1、seed 324、train/validation；所有结果均为 **{ROLLOUT_LABEL}**，未读取 test 或 F2–F5。", "",
        "## 方法", "",
        "v7 在从 canonical train 重新拟合的 SS-DM 上增加被动二阶边端口。每个状态满足 $\\dot q=p$、$\\dot p=\\nu-\\omega_0^2q-2\\zeta\\omega_0p$，边力为 $hp$ 且 $h\\ge0$。其储能导数与 swing 端口功率精确抵消，只留下非负耗散；固定频率和阻尼后，$h$ 由积分弱形式一次求解。", "",
        "候选由训练 pilot 限定，但开发 fit/selection/audit 按 fault line 分离；F1 validation 在分枝冻结后首次读取。普通 DMD 仅作为受迫轨迹的经验描述，DMDc 的不稳定结果作为不可辨识证据保留。", "",
        "## 冻结验证", "",
        "| 模型 | 视界 | mean RMSE | median RMSE | failure rate |", "|---|---:|---:|---:|---:|",
    ]
    for row in validation:
        if row["branch"] in {"SS-DM", selected}:
            lines.append(f"| {row['branch']} | {int(row['horizon_steps'])/100:.0f} s | {_float(row['mean_rmse']):.6f} | {_float(row['median_rmse']):.6f} | {_float(row['failure_rate']):.3f} |")
    modal = acceptance["modal_summary_10s"]
    lines.extend([
        "", "## 模态与命名", "",
        f"10秒经验模态的频率相对误差中位数为 {modal['frequency_error']:.3f}，阻尼绝对误差为 {modal['damping_error']:.3f}，MAC 为 {modal['MAC']:.3f}。绝对模态门为 **{acceptance['modal_absolute_gate']}**。",
        f"因此模型报告名为 **{acceptance['model_reporting_name']}**；它不等同于真实输电边或设备状态恢复。", "",
        "## 结论边界", "",
        f"Acceptance 为 **{acceptance['status']}**。当前结果不支持自主 Pm 预测、test 性能、跨 fold 泛化或唯一设备语义。统计、残差、物理和数值证书分别见同一 acceptance run 的索引文件。", "",
    ])
    (result_dir/"research_report_zh.md").write_text("\n".join(lines), encoding="utf-8")


def _run_acceptance(root: Path, config: dict[str, Any], result_dir: Path, run_id: str) -> None:
    controlled = _latest_result(root, str(config["controlled_experiment"]), "controlled_experiment")
    residual = _latest_result(root, str(config["residual_experiment"]), "residual_diagnosis")
    adversarial = _latest_result(root, str(config["adversarial_experiment"]), "adversarial_review")
    if _verify_hashes(controlled) or _verify_hashes(residual) or _verify_hashes(adversarial):
        raise RuntimeError("v7 parent artifacts changed before acceptance")
    marker = json.loads((controlled/"selection_marker.json").read_text(encoding="utf-8"))
    statistics = json.loads((controlled/"performance_statistics.json").read_text(encoding="utf-8"))
    physics = json.loads((controlled/"physics_audits.json").read_text(encoding="utf-8"))
    convergence = json.loads((controlled/"integration_convergence.json").read_text(encoding="utf-8"))
    aggregate = _read_csv(controlled/"validation_aggregate_metrics.csv")
    modal_rows = _read_csv(controlled/"validation_modal_matches.csv")
    selected_route = marker["selected_route"]
    selected_h1000 = next(row for row in aggregate if row["branch"] == selected_route and int(row["horizon_steps"]) == 1000)
    baseline_h1000 = next(row for row in aggregate if row["branch"] == "SS-DM" and int(row["horizon_steps"]) == 1000)
    improvement = (_float(baseline_h1000["mean_rmse"])-_float(selected_h1000["mean_rmse"]))/max(_float(baseline_h1000["mean_rmse"]), np.finfo(float).eps)
    selected_modal = _median_modal([row for row in modal_rows if row["branch"] == selected_route])
    baseline_modal = _median_modal([row for row in modal_rows if row["branch"] == "SS-DM"])
    modal_absolute = bool(
        selected_modal["matched_modes"] > 0 and selected_modal["frequency_error"] <= 0.10
        and selected_modal["damping_error"] <= 0.05 and selected_modal["MAC"] >= 0.80
    )
    bootstrap = statistics["bootstrap"]
    sign_flip = statistics["sign_flip"]
    gates = {
        "nonfallback_module_selected": selected_route != "SS-DM",
        "validation_10s_improvement_at_least_2pct": improvement >= float(config["minimum_model_improvement"]),
        "bootstrap_lower_bound_positive": bootstrap.get("status") == "completed" and float(bootstrap.get("ci_lower", -np.inf)) > 0,
        "sign_flip_p_at_most_0_05": sign_flip.get("status") == "completed" and float(sign_flip.get("two_sided_p_value", 1.0)) <= 0.05,
        "frequency_not_worse": float(selected_modal["frequency_error"]) <= float(baseline_modal["frequency_error"])+1e-12,
        "damping_not_worse": float(selected_modal["damping_error"]) <= float(baseline_modal["damping_error"])+1e-12,
        "MAC_not_worse": float(selected_modal["MAC"]) >= float(baseline_modal["MAC"])-1e-12,
        "zero_failures": _float(selected_h1000["failure_rate"]) == 0,
        "COI": _float(selected_h1000["maximum_coi_drift"]) <= float(config["maximum_coi_drift"]),
        "RK4": bool(convergence) and all(bool(row["convergent"]) for row in convergence.values()),
        "physics": physics[selected_route]["status"] == "pass",
        "test_locked": marker["test_ids_read"] == [], "other_folds_locked": marker["other_folds_read"] == [],
    }
    status = "PASS" if all(gates.values()) else "FAIL"
    acceptance = {
        "run_id": run_id, "status": status,
        "terminal_state": "MODAL_STRUCTURE_IMPROVED" if status == "PASS" else "MODEL_CLASS_INADEQUATE",
        "selected_route": selected_route,
        "model_reporting_name": "modal-preserving passive dynamic equivalent" if modal_absolute else "modal-improved rollout-accurate dynamic equivalent",
        "validation_relative_improvement_10s": improvement,
        "modal_summary_10s": selected_modal, "baseline_modal_summary_10s": baseline_modal,
        "modal_absolute_gate": "PASS" if modal_absolute else "FAIL",
        "gates": gates, "claim_boundary": ROLLOUT_LABEL,
        "test_ids_read": [], "other_folds_read": [], "merge_gate_required": True,
    }
    _json(result_dir/"acceptance.json", acceptance)
    _write_report(result_dir, controlled, acceptance)
    _json(result_dir/"lineage.json", {"run_id": run_id, "stage": "acceptance", "controlled_result": controlled.name, "residual_result": residual.name, "adversarial_result": adversarial.name, "test_ids_read": []})
    _write_hashes(result_dir)
    if status != "PASS":
        raise RuntimeError(f"v7 acceptance failed: {gates}")


def run(*, root: Path, experiment: dict[str, Any], stage: str, result_dir: Path, run_id: str) -> None:
    config_path = root/"configs"/"experiments"/str(experiment["config"])
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    _validate_scope(config)
    if stage == "hypothesis_registration":
        _run_hypothesis(config_path, config, result_dir, run_id)
    elif stage == "controlled_experiment":
        _run_controlled(root, config_path, config, result_dir, run_id)
    elif stage == "residual_diagnosis":
        _run_residual(root, config, result_dir, run_id)
    elif stage == "adversarial_review":
        _run_adversarial(root, config, result_dir, run_id)
    elif stage == "acceptance":
        _run_acceptance(root, config, result_dir, run_id)
    else:
        raise ValueError(f"unsupported v7 stage: {stage}")
