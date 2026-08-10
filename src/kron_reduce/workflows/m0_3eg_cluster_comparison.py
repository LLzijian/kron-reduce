from __future__ import annotations

import csv
import hashlib
import html
import json
import tomllib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from kron_reduce.data import CanonicalStore, ModelDataRequest, adapt_event, load_split
from kron_reduce.data.audit import audit_model_sample
from kron_reduce.models import M0Model, RunContext, TransientEmfM0Model
from kron_reduce.physics import CLUSTERED_7_PARTITION, aggregate_state, coi_project
from kron_reduce.workflows.physical_cluster_audit import (
    LEGACY_7_PARTITION,
    _aggregate_adapted_sample,
    _design_metrics,
    _discover_train_partition,
    _source,
    _trajectory_errors,
)

HIGHORDER_SOURCE = "ieee39_highorder_dataset"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({field for row in materialized for field in row})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    def coerce(value: str) -> Any:
        if value == "":
            return None
        if value.lower() in {"true", "false"}:
            return value.lower() == "true"
        try:
            return float(value)
        except ValueError:
            return value

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return [
            {key: coerce(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def _latest_result(root: Path, experiment_name: str, stage: str) -> Path:
    prefix = f"{experiment_name}-{stage}-"
    candidates = sorted(
        path
        for path in (root / "results").glob(f"{prefix}*")
        if path.is_dir() and not (path / "RUN_FAILED.json").exists()
    )
    if not candidates:
        raise FileNotFoundError(f"no completed result for {experiment_name}/{stage}")
    return candidates[-1]


def _verify_artifact_hashes(diagnostic_dir: Path) -> list[str]:
    expected = json.loads((diagnostic_dir / "artifact_hashes.json").read_text(encoding="utf-8"))
    return [
        relative
        for relative, digest in expected.items()
        if not (diagnostic_dir / relative).is_file()
        or _sha256(diagnostic_dir / relative) != digest
    ]


def _clearing_stress(sample: dict[str, Any]) -> float:
    static = sample["model_inputs"]["static"]
    incidence = np.asarray(static["incidence"], dtype=float)
    delta0 = np.asarray(static["delta0"], dtype=float).reshape(-1)
    equilibrium = np.asarray(static["equilibrium_delta"], dtype=float).reshape(-1)
    return float(
        np.max(np.abs(np.sin(incidence.T @ delta0) - np.sin(incidence.T @ equilibrium)))
    )


def _threshold_policy(train: list[dict[str, Any]]) -> dict[str, Any]:
    stress = np.asarray([_clearing_stress(sample) for sample in train], dtype=float)
    quantiles = np.quantile(stress, (0.25, 0.5, 0.75, 0.9))
    thresholds = tuple(sorted({0.0, *(float(value) for value in quantiles)}))
    return {
        "stress_definition": "max_abs_delta_sin_at_clearing",
        "quantiles": [0.25, 0.5, 0.75, 0.9],
        "quantile_values": quantiles.tolist(),
        "threshold_candidates": list(thresholds),
        "activation_width": max(0.02, float((quantiles[-1] - quantiles[0]) / 4.0)),
        "constructed_from": "train_only",
        "labels_used": False,
        "validation_or_test_used": False,
    }


def _flatten_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for original in rows:
        row = dict(original)
        sources = row.pop("source_standardized_loss", {})
        for source, value in sources.items():
            row[f"source_loss_{source}"] = value
        output.append(row)
    return output


def _fit_model(
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    config: dict[str, Any],
    checkpoint: Path,
    representation: str,
) -> tuple[TransientEmfM0Model, dict[str, Any], list[dict[str, Any]]]:
    policy = _threshold_policy(train)
    context = RunContext(
        seed=int(config["seed"]),
        epochs=0,
        horizon_steps=int(config["selection_horizon_steps"]),
        max_cases=max(len(train), len(val)),
        ridge_values=tuple(float(value) for value in config["ridge_values"]),
        integral_windows=tuple(int(value) for value in config["integral_windows"]),
        max_internal_step=float(config["max_internal_step"]),
    )
    model = TransientEmfM0Model(
        alpha_candidates=tuple(float(value) for value in config["alpha_candidates"]),
        tau_candidates=tuple(float(value) for value in config["tau_candidates"]),
        activation_threshold_candidates=tuple(policy["threshold_candidates"]),
        activation_width=float(policy["activation_width"]),
        max_source_loss_degradation=float(config["max_source_loss_degradation"]),
        max_internal_step=float(config["max_internal_step"]),
    )
    model.fit(train, val, context)
    if model.__class__ is not TransientEmfM0Model or model.method_name != "M0-3E":
        raise RuntimeError("comparison must use the basic TransientEmfM0Model only")
    if model.physics_audit().get("status") != "pass":
        raise RuntimeError(f"{representation} M0-3EG failed its physics audit")
    candidate_rows = _flatten_candidates(model.diagnostics.pop("candidate_table"))
    model.diagnostics["candidate_table_artifact"] = f"candidate_selection_{representation}.csv"
    model.save(checkpoint)

    highorder_train = [sample for sample in train if _source(sample) == HIGHORDER_SOURCE]
    matrix, _, weak = M0Model.build_weak_system(
        highorder_train,
        windows=context.integral_windows,
        max_cases=context.max_cases,
        horizon_steps=context.horizon_steps,
        max_fit_steps=context.fit_max_steps,
    )
    design = {
        "representation": representation,
        **_design_metrics(matrix),
        **weak,
        "anchor_selected_ridge": model.anchor.selected_ridge,
        "selected_alpha": model.alpha,
        "selected_tau": model.tau,
        "selected_activation_threshold": model.activation_threshold,
        "activation_width": model.activation_width,
        "selected_is_M0_fallback": model.diagnostics["selected_is_M0_fallback"],
        "eligible_candidate_count": model.diagnostics["eligible_candidate_count"],
        "selection_standardized_loss": model.diagnostics["selected_validation"][
            "standardized_loss"
        ],
        "selection_raw_mean_rmse": model.diagnostics["selected_validation"][
            "raw_mean_rmse"
        ],
    }
    return model, {"threshold_policy": policy, "design": design}, candidate_rows


def _project_prediction(
    values: np.ndarray,
    mass10: np.ndarray,
    mass7: np.ndarray,
    partition: tuple[tuple[int, ...], ...],
) -> np.ndarray:
    return coi_project(aggregate_state(values, mass10, partition), mass7)


def _paired_rollouts(
    model10: TransientEmfM0Model,
    models7: dict[str, TransientEmfM0Model],
    samples: dict[str, dict[str, list[dict[str, Any]]]],
    partitions: dict[str, tuple[tuple[int, ...], ...]],
    store: CanonicalStore,
    horizon_steps: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for representation, partition in partitions.items():
        for sample10, sample7 in zip(
            samples["generator_10"]["val"], samples[representation]["val"], strict=True
        ):
            target7 = sample7["training_context"]["targets"]
            steps = min(horizon_steps, len(target7["t"]))
            time_grid = np.asarray(target7["t"][:steps], dtype=float)
            truth_delta = np.asarray(target7["delta"][:steps], dtype=float)
            truth_omega = np.asarray(target7["omega"][:steps], dtype=float)
            incidence7 = np.asarray(sample7["model_inputs"]["static"]["incidence"], dtype=float)
            mass7 = np.asarray(sample7["model_inputs"]["static"]["mass"], dtype=float)
            mass10 = np.asarray(sample10["model_inputs"]["static"]["mass"], dtype=float)
            metadata = store.metadata(sample10["event_id"])
            common = {
                "representation": representation,
                "event_id": sample10["event_id"],
                "fault_line": metadata.fault_line,
                "source_dataset": metadata.source_dataset,
                "horizon_steps": steps,
                "horizon_s": float(time_grid[-1] - time_grid[0]),
            }
            try:
                pred10_delta, pred10_omega = model10.rollout(
                    sample10["model_inputs"]["static"],
                    time_grid,
                    {
                        "Pm": np.asarray(sample10["model_inputs"]["exogenous"]["Pm"][:steps]),
                        "interpolation": "linear",
                    },
                )
                projected_delta = _project_prediction(pred10_delta, mass10, mass7, partition)
                projected_omega = _project_prediction(pred10_omega, mass10, mass7, partition)
                rows.append(
                    {
                        **common,
                        "model_route": "generator_10_then_project",
                        "status": "completed",
                        **_trajectory_errors(
                            truth_delta,
                            truth_omega,
                            projected_delta,
                            projected_omega,
                            incidence7,
                            time_grid,
                        ),
                    }
                )
            except (FloatingPointError, ValueError, OverflowError, np.linalg.LinAlgError) as exc:
                rows.append(
                    {
                        **common,
                        "model_route": "generator_10_then_project",
                        "status": "failed",
                        "error": repr(exc),
                    }
                )
            try:
                pred7_delta, pred7_omega = models7[representation].rollout(
                    sample7["model_inputs"]["static"],
                    time_grid,
                    {
                        "Pm": np.asarray(sample7["model_inputs"]["exogenous"]["Pm"][:steps]),
                        "interpolation": "linear",
                    },
                )
                rows.append(
                    {
                        **common,
                        "model_route": "native_7",
                        "status": "completed",
                        **_trajectory_errors(
                            truth_delta,
                            truth_omega,
                            pred7_delta,
                            pred7_omega,
                            incidence7,
                            time_grid,
                        ),
                    }
                )
            except (FloatingPointError, ValueError, OverflowError, np.linalg.LinAlgError) as exc:
                rows.append(
                    {
                        **common,
                        "model_route": "native_7",
                        "status": "failed",
                        "error": repr(exc),
                    }
                )
    return rows


def _summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    keys = sorted({(row["representation"], row["model_route"]) for row in rows})
    for representation, route in keys:
        group = [
            row
            for row in rows
            if row["representation"] == representation and row["model_route"] == route
        ]
        complete = [row for row in group if row["status"] == "completed"]
        values = np.asarray([row["rmse"] for row in complete], dtype=float)
        omega = np.asarray([row["omega_rmse"] for row in complete], dtype=float)
        edge = np.asarray([row["edge_angle_rmse"] for row in complete], dtype=float)
        tail_count = max(1, int(np.ceil(0.1 * len(values)))) if len(values) else 0
        output.append(
            {
                "representation": representation,
                "model_route": route,
                "n_cases": len(group),
                "failure_rate": 1.0 - len(complete) / max(len(group), 1),
                "mean_rmse": float(np.mean(values)) if len(values) else float("inf"),
                "median_rmse": float(np.median(values)) if len(values) else float("inf"),
                "pooled_rmse": float(np.sqrt(np.mean(values**2))) if len(values) else float("inf"),
                "p90_rmse": float(np.quantile(values, 0.9)) if len(values) else float("inf"),
                "cvar10_rmse": (
                    float(np.mean(np.sort(values)[-tail_count:])) if len(values) else float("inf")
                ),
                "mean_omega_rmse": float(np.mean(omega)) if len(omega) else float("inf"),
                "mean_edge_angle_rmse": float(np.mean(edge)) if len(edge) else float("inf"),
            }
        )
    return output


def _cluster_bootstrap_difference(
    paired_rows: list[dict[str, Any]], draws: int, seed: int
) -> dict[str, Any]:
    by_event: dict[str, dict[str, dict[str, Any]]] = {}
    for row in paired_rows:
        if row["status"] == "completed":
            by_event.setdefault(str(row["event_id"]), {})[str(row["model_route"])] = row
    paired = [
        {
            "event_id": event_id,
            "fault_line": routes["native_7"]["fault_line"],
            "source_dataset": routes["native_7"]["source_dataset"],
            "difference": float(routes["native_7"]["rmse"] - routes["generator_10_then_project"]["rmse"]),
            "native_rmse": float(routes["native_7"]["rmse"]),
            "projected_rmse": float(routes["generator_10_then_project"]["rmse"]),
        }
        for event_id, routes in by_event.items()
        if {"native_7", "generator_10_then_project"}.issubset(routes)
    ]
    clusters = sorted({str(row["fault_line"]) for row in paired})
    if not paired or not clusters:
        return {
            "n_paired_cases": 0,
            "n_fault_line_clusters": 0,
            "mean_difference_native_minus_projected": float("nan"),
            "ci95_lower": float("nan"),
            "ci95_upper": float("nan"),
            "paired_rows": [],
        }
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(draws):
        sampled_clusters = rng.choice(clusters, size=len(clusters), replace=True)
        sampled = [
            row for cluster in sampled_clusters for row in paired if row["fault_line"] == cluster
        ]
        estimates.append(float(np.mean([row["difference"] for row in sampled])))
    native = float(np.mean([row["native_rmse"] for row in paired]))
    projected = float(np.mean([row["projected_rmse"] for row in paired]))
    return {
        "n_paired_cases": len(paired),
        "n_fault_line_clusters": len(clusters),
        "mean_native_rmse": native,
        "mean_projected_10_rmse": projected,
        "relative_improvement_native_7": (projected - native) / max(projected, 1e-15),
        "mean_difference_native_minus_projected": native - projected,
        "ci95_lower": float(np.quantile(estimates, 0.025)),
        "ci95_upper": float(np.quantile(estimates, 0.975)),
        "resampling_unit": "fault_line",
        "draws": draws,
        "paired_rows": paired,
    }


def _assess(
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    models: dict[str, TransientEmfM0Model],
    bootstrap: dict[str, dict[str, Any]],
    design_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    assessments = {}
    for representation, result in bootstrap.items():
        group = [row for row in rows if row["representation"] == representation]
        sources = sorted({str(row["source_dataset"]) for row in group})
        source_guard = {}
        for source in sources:
            source_rows = [row for row in group if row["source_dataset"] == source]
            by_route = {
                route: np.mean(
                    [
                        row["rmse"]
                        for row in source_rows
                        if row["model_route"] == route and row["status"] == "completed"
                    ]
                )
                for route in ("native_7", "generator_10_then_project")
            }
            source_guard[source] = {
                "native_mean_rmse": float(by_route["native_7"]),
                "projected_10_mean_rmse": float(by_route["generator_10_then_project"]),
                "native_not_worse": bool(by_route["native_7"] <= by_route["generator_10_then_project"]),
            }
        model = models[representation]
        gates = {
            "all_rollouts_completed": all(row["status"] == "completed" for row in group),
            "native_mean_lower_than_matched_projected_10": bool(
                result["mean_difference_native_minus_projected"] < 0.0
            ),
            "fault_line_cluster_bootstrap_ci_upper_below_zero": bool(
                result["ci95_upper"] < 0.0
            ),
            "both_sources_not_worse": all(row["native_not_worse"] for row in source_guard.values()),
            "nontrivial_third_state_selected": bool(
                model.alpha is not None
                and model.alpha > 0.0
                and model.activation_threshold > 0.0
                and not model.diagnostics["selected_is_M0_fallback"]
            ),
            "physics_audit_pass": model.physics_audit().get("status") == "pass",
        }
        assessments[representation] = {
            "status": "SUPPORTED" if all(gates.values()) else "NOT_SUPPORTED",
            "gates": gates,
            "source_guard": source_guard,
            "paired_comparison": {key: value for key, value in result.items() if key != "paired_rows"},
            "selected_alpha": model.alpha,
            "selected_tau": model.tau,
            "selected_activation_threshold": model.activation_threshold,
        }

    design = {row["representation"]: row for row in design_rows}
    primary = str(config["primary_partition"])
    efficiency = {
        representation: {
            "parameter_count_7": int(design[representation]["n_parameters"]),
            "parameter_count_10": int(design["generator_10"]["n_parameters"]),
            "condition_number_7": float(design[representation]["condition_number"]),
            "condition_number_10": float(design["generator_10"]["condition_number"]),
            "parameter_reduction_fraction": 1.0
            - float(design[representation]["n_parameters"])
            / float(design["generator_10"]["n_parameters"]),
            "condition_number_ratio_7_over_10": float(design[representation]["condition_number"])
            / float(design["generator_10"]["condition_number"]),
        }
        for representation in assessments
    }
    return {
        "question": "Does train-only seven-node clustering outperform direct ten-machine M0-3EG?",
        "primary_partition": primary,
        "primary_answer": assessments[primary]["status"],
        "interpretation": (
            "seven-node M0-3EG is superior on the predeclared matched-coordinate gate"
            if assessments[primary]["status"] == "SUPPORTED"
            else "seven-node superiority is not established on the predeclared matched-coordinate gate"
        ),
        "partition_assessments": assessments,
        "efficiency_and_identifiability": efficiency,
        "claim_boundary": {
            "fold": config["fold"],
            "seed": config["seed"],
            "test_read": False,
            "observed_Pm_is_conditional_upper_bound": True,
            "five_fold_generalization_claim": False,
            "representation_selected_on_validation": False,
        },
    }


def _make_figure(
    result_dir: Path,
    rows: list[dict[str, Any]],
    design_rows: list[dict[str, Any]],
    assessment: dict[str, Any],
) -> None:
    names = [
        "clustered_7_current",
        "clustered_7_legacy",
        "clustered_7_train_discovered",
    ]
    labels = ["current 7", "legacy 7", "train-only 7"]
    colors = {"native_7": "#B4473D", "generator_10_then_project": "#315A7D"}
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.8))
    positions = np.arange(len(names))
    for route, offset in (("generator_10_then_project", -0.17), ("native_7", 0.17)):
        values = [
            np.asarray(
                [
                    row["rmse"]
                    for row in rows
                    if row["representation"] == name
                    and row["model_route"] == route
                    and row["status"] == "completed"
                ]
            )
            for name in names
        ]
        axes[0].boxplot(
            values,
            positions=positions + offset,
            widths=0.28,
            patch_artist=True,
            showfliers=False,
            boxprops={"facecolor": colors[route], "alpha": 0.72, "linewidth": 0.8},
            medianprops={"color": "white", "linewidth": 1.5},
            whiskerprops={"linewidth": 0.8},
            capprops={"linewidth": 0.8},
        )
    axes[0].set_xticks(positions, labels, rotation=15)
    axes[0].set_ylabel("Matched 7-node RMSE")
    axes[0].set_title("a  Paired reintegration accuracy", loc="left", fontweight="bold")
    axes[0].plot([], [], color=colors["generator_10_then_project"], lw=7, label="10 → 7")
    axes[0].plot([], [], color=colors["native_7"], lw=7, label="native 7")
    axes[0].legend(frameon=False, fontsize=8)

    design = {row["representation"]: row for row in design_rows}
    all_names = ["generator_10", *names]
    condition = [float(design[name]["condition_number"]) for name in all_names]
    axes[1].bar(np.arange(4), condition, color=["#315A7D", "#8AA1B1", "#8AA1B1", "#B4473D"])
    axes[1].set_yscale("log")
    axes[1].set_xticks(np.arange(4), ["10", "current 7", "legacy 7", "train-only 7"], rotation=15)
    axes[1].set_ylabel("Weak-design condition number")
    axes[1].set_title("b  Identifiability", loc="left", fontweight="bold")

    differences = []
    lower = []
    upper = []
    for name in names:
        paired = assessment["partition_assessments"][name]["paired_comparison"]
        differences.append(float(paired["mean_difference_native_minus_projected"]))
        lower.append(float(paired["ci95_lower"]))
        upper.append(float(paired["ci95_upper"]))
    differences_array = np.asarray(differences)
    axes[2].errorbar(
        positions,
        differences,
        yerr=np.vstack((differences_array - np.asarray(lower), np.asarray(upper) - differences_array)),
        fmt="o",
        color="#B4473D",
        ecolor="#555555",
        capsize=3,
    )
    axes[2].axhline(0.0, color="black", lw=0.9, ls="--")
    axes[2].set_xticks(positions, labels, rotation=15)
    axes[2].set_ylabel("Native 7 − projected 10 RMSE")
    axes[2].set_title("c  Fault-line bootstrap 95% CI", loc="left", fontweight="bold")
    fig.tight_layout()
    figures = result_dir / "figures"
    figures.mkdir()
    fig.savefig(figures / "fig1_m0_3eg_7_vs_10.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures / "fig1_m0_3eg_7_vs_10.svg", bbox_inches="tight")
    plt.close(fig)


def _write_report(
    result_dir: Path,
    run_id: str,
    config: dict[str, Any],
    assessment: dict[str, Any],
) -> None:
    primary = str(config["primary_partition"])
    primary_row = assessment["partition_assessments"][primary]
    paired = primary_row["paired_comparison"]
    efficiency = assessment["efficiency_and_identifiability"][primary]
    answer = "是" if primary_row["status"] == "SUPPORTED" else "否"
    lines = [
        "# 基础 M0-3EG：七节点聚类与原始十机的严格对照",
        "",
        f"**运行标识：** `{run_id}`  ",
        f"**证据范围：** {config['fold']}、seed {config['seed']}、train/validation only；test 未读取；observed $P_m(t)$ 为条件上界。  ",
        "**模型范围：** 仅 `TransientEmfM0Model`（基础 M0-3EG）；无 GMD、DF、残差库、神经网络或重辨识。",
        "",
        "## 结论",
        "",
        f"对预注册的 train-only 七节点方法，答案是：**{answer}**（{primary_row['status']}）。",
        "",
        f"公平的同坐标比较中，原生七节点 mean RMSE={paired['mean_native_rmse']:.6g}，十机预测后投影到同一七节点坐标的 mean RMSE={paired['mean_projected_10_rmse']:.6g}；相对改进={100*paired['relative_improvement_native_7']:.2f}%。",
        f"配对差值（七节点−十机投影）为 {paired['mean_difference_native_minus_projected']:.6g}，按 fault line 聚类 bootstrap 的 95% CI 为 [{paired['ci95_lower']:.6g}, {paired['ci95_upper']:.6g}]。",
        "",
        "![M0-3EG 七节点与十机对照](figures/fig1_m0_3eg_7_vs_10.png)",
        "",
        "## 为什么必须这样比较",
        "",
        "七维原生 RMSE 与十维原生 RMSE维数不同、边数不同，不能直接排序。本轮先把十机预测作质量保持的 COI 聚合，得到与七节点模型完全相同的目标、节点数和边角误差，再做逐事件配对。这样比较的是模型闭环误差，而不是降维本身天然丢掉的机间细节。",
        "",
        "## 现象解释",
        "",
    ]
    if primary_row["status"] == "SUPPORTED":
        lines.extend(
            [
                "七节点的优势与‘更多数据’无关，而来自物理约束下更合适的辨识分辨率：",
                "",
                "- 相干机组的质量加权平均消除了簇内差模。对只含一个标量暂态状态 $q_i$ 的基础 M0-3EG，这些差模本来无法被正确表达，保留它们只会作为结构失配进入系数估计。",
                f"- M0 弱形式参数由 {efficiency['parameter_count_10']} 个降为 {efficiency['parameter_count_7']} 个，减少 {100*efficiency['parameter_reduction_fraction']:.1f}%；设计矩阵条件数之比（7/10）为 {efficiency['condition_number_ratio_7_over_10']:.3g}。这降低了高度相关边特征对 $D,K$ 的放大效应。",
                "- 聚类因此更准确地应称为相干子空间投影或经验时间尺度分离。它可呈现低通效果，但本轮没有构造图拉普拉斯频响证书，不能称为严格的图低通滤波器。",
            ]
        )
    else:
        lines.extend(
            [
                "参数减少和条件数改善并不自动转化为更低的闭环误差。当前预注册七节点分区未同时通过负差值、fault-line bootstrap、双数据源不退化与非平凡三阶状态门，因此不能声称七聚类优于十机。",
                "",
                "若某个历史分区看起来更好，只能说明优势依赖具体分区；若 95% CI 跨零，则还可能只是本 fold 的故障线路组成。此时把降维称为普遍低通增益会把表示偏差误当成算法提升。",
            ]
        )
    lines.extend(
        [
            "",
            "## 分区敏感性",
            "",
            "| 分区 | 判定 | 七节点 mean RMSE | 十机→七节点 mean RMSE | 相对改进 | 95% CI（差值） | alpha | tau | stress threshold |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, row in assessment["partition_assessments"].items():
        p = row["paired_comparison"]
        lines.append(
            f"| {name} | {row['status']} | {p['mean_native_rmse']:.6g} | {p['mean_projected_10_rmse']:.6g} | {100*p['relative_improvement_native_7']:.2f}% | [{p['ci95_lower']:.4g}, {p['ci95_upper']:.4g}] | {row['selected_alpha']:.4g} | {row['selected_tau']:.4g} | {row['selected_activation_threshold']:.4g} |"
        )
    lines.extend(
        [
            "",
            "## 可接受的主张边界",
            "",
            "- 这是单 fold、单 seed 的交叉折敏感性证据，不是五折总体显著性结论。",
            "- observed $P_m(t)$ 使用真实后续机械功率日程，只能解释为给定外源输入后的条件重积分能力，不能声称在线因果预测能力。",
            "- 分区由训练轨迹发现；验证集只选择每个表示内部的 M0-3EG 超参数，没有在三个分区中挑选赢家。",
            "- test 仍然封闭，标签未用于拟合、阈值或模型选择。",
        ]
    )
    markdown = "\n".join(lines) + "\n"
    (result_dir / "report.md").write_text(markdown, encoding="utf-8")
    escaped = html.escape(markdown)
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>基础 M0-3EG 七节点与十机对照</title><style>
body{{font-family:Arial,'Microsoft YaHei',sans-serif;margin:0;background:#f4f5f7;color:#202124;line-height:1.7}}
main{{max-width:1050px;margin:28px auto;background:white;padding:44px 54px;box-shadow:0 2px 16px #0001}}
h1{{border-bottom:3px solid #9d2f25;padding-bottom:12px}} img{{max-width:100%;border:1px solid #ddd}}
pre{{white-space:pre-wrap;font-family:Arial,'Microsoft YaHei',sans-serif}} @media(max-width:720px){{main{{margin:0;padding:24px}}}}
</style></head><body><main><h1>基础 M0-3EG：七节点与十机对照</h1>
<img src="figures/fig1_m0_3eg_7_vs_10.svg" alt="七节点与十机配对比较">
<pre>{escaped}</pre></main></body></html>"""
    (result_dir / "report.html").write_text(document, encoding="utf-8")


def _run_diagnostic(
    *, root: Path, config: dict[str, Any], config_path: Path, result_dir: Path, run_id: str
) -> None:
    if config.get("test_access") != "forbidden" or config.get("pm_mode") != "observed":
        raise RuntimeError("comparison requires forbidden test access and observed-Pm sensitivity")
    dataset_dir = root / "data" / str(config["dataset"])
    store = CanonicalStore.open(dataset_dir)
    split = load_split(
        dataset_dir,
        view=str(config["view"]),
        split_name=str(config["split_name"]),
        fold=str(config["fold"]),
    )
    train_ids = [
        event_id
        for event_id in split.train_ids
        if store.metadata(event_id).valid_length >= int(config["minimum_train_steps"])
    ]
    val_ids = [
        event_id
        for event_id in split.val_ids
        if store.metadata(event_id).valid_length >= int(config["selection_horizon_steps"])
    ]
    if not train_ids or not val_ids:
        raise RuntimeError("no eligible train/validation samples")
    request = ModelDataRequest.initial_state_rollout("observed")
    inertia = store.dataset_metadata.M_prior
    samples: dict[str, dict[str, list[dict[str, Any]]]] = {"generator_10": {}}
    for role, ids in (("train", train_ids), ("val", val_ids)):
        samples["generator_10"][role] = [
            adapt_event(
                store.load_event(event_id),
                inertia_prior=inertia,
                request=request,
                representation="generator_10",
            )
            for event_id in ids
        ]
        for sample in samples["generator_10"][role]:
            audit_model_sample(sample)

    discovered, pair_scores = _discover_train_partition(
        samples["generator_10"]["train"], float(config["partition_discovery_horizon_s"])
    )
    partitions = {
        "clustered_7_current": tuple(tuple(group) for group in CLUSTERED_7_PARTITION),
        "clustered_7_legacy": LEGACY_7_PARTITION,
        "clustered_7_train_discovered": discovered,
    }
    for representation, partition in partitions.items():
        samples[representation] = {}
        for role in ("train", "val"):
            samples[representation][role] = [
                _aggregate_adapted_sample(sample, partition, representation)
                for sample in samples["generator_10"][role]
            ]
            for sample in samples[representation][role]:
                audit_model_sample(sample)

    checkpoints = result_dir / "checkpoints"
    checkpoints.mkdir()
    models: dict[str, TransientEmfM0Model] = {}
    policies: dict[str, Any] = {}
    design_rows = []
    for representation in ("generator_10", *partitions):
        model, fit_audit, candidates = _fit_model(
            samples[representation]["train"],
            samples[representation]["val"],
            config,
            checkpoints / f"{representation}-M0-3EG.npz",
            representation,
        )
        models[representation] = model
        policies[representation] = fit_audit["threshold_policy"]
        design_rows.append(fit_audit["design"])
        _write_csv(result_dir / f"candidate_selection_{representation}.csv", candidates)

    selection_marker = {
        "run_id": run_id,
        "selection_complete": True,
        "fold": config["fold"],
        "seed": config["seed"],
        "model_family": "basic_M0-3EG_only",
        "representations_predeclared": ["generator_10", *partitions],
        "primary_partition_predeclared": config["primary_partition"],
        "test_ids_read": [],
        "labels_used": False,
        "partition_discovery_used_train_only": True,
        "selected_parameters": {
            name: {
                "alpha": model.alpha,
                "tau": model.tau,
                "activation_threshold": model.activation_threshold,
                "anchor_ridge": model.anchor.selected_ridge,
            }
            for name, model in models.items()
        },
    }
    _json(result_dir / "selection_marker.json", selection_marker)

    rows = _paired_rollouts(
        models["generator_10"],
        {name: models[name] for name in partitions},
        samples,
        partitions,
        store,
        int(config["selection_horizon_steps"]),
    )
    summary_rows = _summary(rows)
    bootstrap = {
        representation: _cluster_bootstrap_difference(
            [row for row in rows if row["representation"] == representation],
            int(config["bootstrap_draws"]),
            int(config["seed"]) + index,
        )
        for index, representation in enumerate(partitions)
    }
    assessment = _assess(config, rows, models, bootstrap, design_rows)

    convergence_rows = []
    for representation, model in models.items():
        sample = samples[representation]["val"][0]
        steps = int(config["selection_horizon_steps"])
        convergence_rows.append(
            {
                "representation": representation,
                "event_id": sample["event_id"],
                **model.convergence_audit(
                    sample["model_inputs"]["static"],
                    np.asarray(sample["model_inputs"]["exogenous"]["t"][:steps]),
                    {
                        "Pm": np.asarray(sample["model_inputs"]["exogenous"]["Pm"][:steps]),
                        "interpolation": "linear",
                    },
                ),
            }
        )

    dataset_rows = [
        {
            "role": role,
            "event_id": event_id,
            "source_dataset": store.metadata(event_id).source_dataset,
            "fault_line": store.metadata(event_id).fault_line,
            "valid_length": store.metadata(event_id).valid_length,
            "quality_status": store.metadata(event_id).quality_status,
        }
        for role, ids in (("train", train_ids), ("val", val_ids))
        for event_id in ids
    ]
    _write_csv(result_dir / "dataset_support.csv", dataset_rows)
    _write_csv(result_dir / "train_partition_pair_scores.csv", pair_scores)
    _write_csv(result_dir / "design_conditioning.csv", design_rows)
    _write_csv(result_dir / "paired_rollout_metrics.csv", rows)
    _write_csv(result_dir / "aggregate_metrics.csv", summary_rows)
    _write_csv(result_dir / "integration_convergence.csv", convergence_rows)
    _json(
        result_dir / "partition_definitions.json",
        {name: [list(group) for group in partition] for name, partition in partitions.items()},
    )
    _json(result_dir / "train_only_threshold_policies.json", policies)
    _json(
        result_dir / "paired_fault_line_bootstrap.json",
        {
            representation: {key: value for key, value in result.items() if key != "paired_rows"}
            for representation, result in bootstrap.items()
        },
    )
    _json(result_dir / "claim_assessment.json", assessment)
    _json(
        result_dir / "lineage.json",
        {
            "run_id": run_id,
            "dataset_revision": store.dataset_metadata.dataset_revision,
            "dataset_build_id": store.dataset_metadata.build_id,
            "fold": config["fold"],
            "seed": config["seed"],
            "train_ids_used": train_ids,
            "val_ids_used": val_ids,
            "test_ids_read": [],
            "labels_used_for_fit_selection_or_thresholds": False,
            "observed_Pm_is_conditional_noncausal_upper_bound": True,
            "partition_discovery_used_train_only": True,
            "validation_did_not_select_between_partitions": True,
            "primary_partition_predeclared": config["primary_partition"],
            "config_sha256": _sha256(config_path),
        },
    )
    _make_figure(result_dir, rows, design_rows, assessment)
    _write_report(result_dir, run_id, config, assessment)

    process_gates = {
        "test_data_unopened": True,
        "labels_unused": True,
        "observed_Pm_marked_conditional": True,
        "selection_marker_written_before_comparison": True,
        "basic_M0_3EG_class_only": all(
            model.__class__ is TransientEmfM0Model for model in models.values()
        ),
        "M0_anchor_frozen_after_fit": all(
            model.diagnostics["damping_frozen_after_anchor"]
            and model.diagnostics["coupling_frozen_after_anchor"]
            for model in models.values()
        ),
        "all_model_physics_audits_pass": all(
            model.physics_audit().get("status") == "pass" for model in models.values()
        ),
        "all_rollouts_completed": all(row["status"] == "completed" for row in rows),
        "all_integration_convergence_audits_pass": all(
            row["convergent"] for row in convergence_rows
        ),
        "three_partition_lineages_compared": len(partitions) == 3,
        "train_discovered_partition_is_primary": config["primary_partition"]
        == "clustered_7_train_discovered",
        "validation_did_not_select_partition": True,
        "fault_line_cluster_bootstrap_present": all(
            result["n_fault_line_clusters"] >= 2 for result in bootstrap.values()
        ),
        "scientific_result_may_be_negative": True,
    }
    process = {
        "status": "PASS"
        if all(value for key, value in process_gates.items() if key != "scientific_result_may_be_negative")
        else "FAIL",
        "gates": process_gates,
        "scientific_claim": assessment,
    }
    _json(result_dir / "diagnostic_process_acceptance.json", process)
    if process["status"] != "PASS":
        raise RuntimeError("M0-3EG cluster comparison process gate failed")
    artifacts = {
        path.relative_to(result_dir).as_posix(): _sha256(path)
        for path in result_dir.rglob("*")
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    _json(result_dir / "artifact_hashes.json", artifacts)


def _run_adversarial_audit(
    *, root: Path, config: dict[str, Any], result_dir: Path, run_id: str
) -> None:
    diagnostic_dir = _latest_result(root, str(config["diagnostic_experiment"]), "diagnostic")
    claims = json.loads((diagnostic_dir / "claim_assessment.json").read_text(encoding="utf-8"))
    process = json.loads(
        (diagnostic_dir / "diagnostic_process_acceptance.json").read_text(encoding="utf-8")
    )
    lineage = json.loads((diagnostic_dir / "lineage.json").read_text(encoding="utf-8"))
    rows = _read_csv(diagnostic_dir / "paired_rollout_metrics.csv")
    selection = json.loads((diagnostic_dir / "selection_marker.json").read_text(encoding="utf-8"))
    primary = str(config["primary_partition"])
    gates = {
        "artifact_hashes_match": not _verify_artifact_hashes(diagnostic_dir),
        "diagnostic_process_passed": process.get("status") == "PASS",
        "test_data_unopened": not lineage.get("test_ids_read"),
        "labels_unused": not lineage.get("labels_used_for_fit_selection_or_thresholds"),
        "observed_Pm_claim_bounded": bool(
            lineage.get("observed_Pm_is_conditional_noncausal_upper_bound")
        ),
        "primary_partition_predeclared": selection.get("primary_partition_predeclared")
        == primary,
        "partition_selection_not_validation_driven": bool(
            lineage.get("validation_did_not_select_between_partitions")
        ),
        "matched_coordinate_comparator_present": all(
            {
                row["model_route"]
                for row in rows
                if row["representation"] == representation
            }
            == {"native_7", "generator_10_then_project"}
            for representation in claims["partition_assessments"]
        ),
        "scientific_answer_follows_registered_gates": claims["primary_answer"]
        == claims["partition_assessments"][primary]["status"],
        "no_general_claim_from_nonprimary_partition": claims["claim_boundary"][
            "representation_selected_on_validation"
        ]
        is False,
        "single_fold_boundary_explicit": claims["claim_boundary"][
            "five_fold_generalization_claim"
        ]
        is False,
    }
    status = "PASS" if all(gates.values()) else "FAIL"
    review = {
        "status": status,
        "decision": "ACCEPTABLE_WITH_BOUNDED_SCIENTIFIC_CLAIM" if status == "PASS" else "REJECT",
        "run_id": run_id,
        "diagnostic_result": diagnostic_dir.name,
        "gates": gates,
        "adversarial_questions": [
            "Was native seven-node RMSE improperly compared with native ten-node RMSE?",
            "Was the winning partition chosen after validation inspection?",
            "Could a result caused only by one fault line be called superior?",
            "Was an M0 fallback mislabeled as nontrivial M0-3EG?",
            "Was observed Pm presented as a causal deployment result?",
        ],
    }
    _json(result_dir / "adversarial_review.json", review)
    (result_dir / "adversarial_review.md").write_text(
        "# M0-3EG 七节点对照的对抗性审查\n\n"
        f"**决定：{review['decision']}**\n\n"
        + "\n".join(f"- {'PASS' if value else 'FAIL'} `{key}`" for key, value in gates.items())
        + "\n",
        encoding="utf-8",
    )
    if status != "PASS":
        raise RuntimeError("adversarial audit rejected M0-3EG cluster comparison")


def _run_acceptance(
    *, root: Path, config: dict[str, Any], result_dir: Path, run_id: str
) -> None:
    diagnostic_dir = _latest_result(root, str(config["diagnostic_experiment"]), "diagnostic")
    adversarial_dir = _latest_result(
        root, str(config["adversarial_experiment"]), "adversarial_audit"
    )
    process = json.loads(
        (diagnostic_dir / "diagnostic_process_acceptance.json").read_text(encoding="utf-8")
    )
    adversarial = json.loads(
        (adversarial_dir / "adversarial_review.json").read_text(encoding="utf-8")
    )
    gates = {
        "diagnostic_process_passed": process.get("status") == "PASS",
        "diagnostic_hashes_match": not _verify_artifact_hashes(diagnostic_dir),
        "adversarial_review_passed": adversarial.get("status") == "PASS",
        "bounded_decision": adversarial.get("decision")
        == "ACCEPTABLE_WITH_BOUNDED_SCIENTIFIC_CLAIM",
    }
    status = "PASS" if all(gates.values()) else "FAIL"
    acceptance = {
        "status": status,
        "decision": "ACCEPT" if status == "PASS" else "REJECT",
        "scope": "F2 single-seed validation sensitivity; basic M0-3EG; observed-Pm conditional reintegration; test unopened",
        "run_id": run_id,
        "diagnostic_result": diagnostic_dir.name,
        "adversarial_result": adversarial_dir.name,
        "gates": gates,
    }
    _json(result_dir / "acceptance.json", acceptance)
    (result_dir / "acceptance.md").write_text(
        "# 最终验收\n\n"
        f"**决定：{acceptance['decision']}**\n\n"
        "接收范围仅限 F2、单 seed、observed-Pm 条件重积分下的基础 M0-3EG 七节点/十机公平对照；test 未读取。\n",
        encoding="utf-8",
    )
    if status != "PASS":
        raise RuntimeError("M0-3EG cluster comparison acceptance failed")


def run(*, root: Path, experiment: dict[str, Any], stage: str, result_dir: Path, run_id: str) -> None:
    config_path = root / "configs" / "experiments" / str(experiment["config"])
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    if stage == "diagnostic":
        _run_diagnostic(
            root=root,
            config=config,
            config_path=config_path,
            result_dir=result_dir,
            run_id=run_id,
        )
    elif stage == "adversarial_audit":
        _run_adversarial_audit(root=root, config=config, result_dir=result_dir, run_id=run_id)
    elif stage == "acceptance":
        _run_acceptance(root=root, config=config, result_dir=result_dir, run_id=run_id)
    else:
        raise RuntimeError(f"unsupported M0-3EG cluster-comparison stage: {stage}")
