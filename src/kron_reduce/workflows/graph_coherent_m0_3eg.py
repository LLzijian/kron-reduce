from __future__ import annotations

import json
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from kron_reduce.data import CanonicalStore, ModelDataRequest, adapt_event, load_split
from kron_reduce.data.audit import audit_model_sample
from kron_reduce.models import M0Model, TransientEmfM0Model
from kron_reduce.physics import (
    generalized_laplacian_spectrum,
    normalized_dynamic_pair_costs,
    normalized_forcing_pair_costs,
    optimal_aep_matching,
    robust_consensus_partition,
    synchronizing_laplacian,
)
from kron_reduce.workflows.m0_3eg_cluster_comparison import (
    HIGHORDER_SOURCE,
    _fit_model,
    _json,
    _latest_result,
    _paired_rollouts,
    _read_csv,
    _sha256,
    _summary,
    _verify_artifact_hashes,
    _write_csv,
)
from kron_reduce.workflows.physical_cluster_audit import (
    _aggregate_adapted_sample,
    _discover_train_partition,
    _source,
    _uniform_path,
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _case_costs(sample: dict[str, Any], maximum_time: float) -> tuple[np.ndarray, np.ndarray]:
    target = sample["training_context"]["targets"]
    _, delta = _uniform_path(
        np.asarray(target["t"]), np.asarray(target["delta"]), maximum_time=maximum_time
    )
    _, omega = _uniform_path(
        np.asarray(target["t"]), np.asarray(target["omega"]), maximum_time=maximum_time
    )
    _, pm = _uniform_path(
        np.asarray(target["t"]),
        np.asarray(sample["training_context"]["Pm_observed"]),
        maximum_time=maximum_time,
    )
    static = sample["model_inputs"]["static"]
    mass = np.asarray(static["mass"], dtype=float).reshape(-1)
    return (
        normalized_dynamic_pair_costs(delta, omega, mass),
        normalized_forcing_pair_costs(pm, static["Pm_equilibrium"], mass),
    )


def _discover_graph_partition(
    train: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[tuple[tuple[int, ...], ...], dict[str, Any], list[dict[str, Any]]]:
    highorder = [sample for sample in train if _source(sample) == HIGHORDER_SOURCE]
    if not highorder:
        raise RuntimeError("graph-coherent discovery requires high-order anchor cases")
    windows = tuple(int(value) for value in config["integral_windows"])
    matrix, vector, weak = M0Model.build_weak_system(
        highorder,
        windows=windows,
        max_cases=len(highorder),
        horizon_steps=int(config["selection_horizon_steps"]),
        max_fit_steps=int(config["graph_fit_max_steps"]),
    )
    graph_ridge = float(config["graph_anchor_ridge"])
    theta, solver = M0Model._solve_nonnegative_ridge(matrix, vector, graph_ridge)
    static = highorder[0]["model_inputs"]["static"]
    mass = np.asarray(static["mass"], dtype=float).reshape(-1)
    incidence = np.asarray(static["incidence"], dtype=float)
    equilibrium = np.asarray(static["equilibrium_delta"], dtype=float).reshape(-1)
    coupling = theta[mass.size :]
    laplacian, stiffness = synchronizing_laplacian(
        incidence, coupling, equilibrium
    )
    eigenvalues, eigenvectors = generalized_laplacian_spectrum(laplacian, mass)
    graph_matching, aep_rows = optimal_aep_matching(
        laplacian, mass, int(config["candidate_pair_merges"])
    )

    dynamic_by_source: dict[str, list[np.ndarray]] = defaultdict(list)
    forcing_by_source: dict[str, list[np.ndarray]] = defaultdict(list)
    for sample in train:
        dynamic, forcing = _case_costs(sample, float(config["coherency_horizon_s"]))
        source = _source(sample)
        dynamic_by_source[source].append(dynamic)
        forcing_by_source[source].append(forcing)
    dynamic_arrays = {
        source: np.stack(rows) for source, rows in dynamic_by_source.items()
    }
    forcing_arrays = {
        source: np.stack(rows) for source, rows in forcing_by_source.items()
    }
    partition, consensus = robust_consensus_partition(
        dynamic_costs_by_source=dynamic_arrays,
        forcing_costs_by_source=forcing_arrays,
        graph_matching=graph_matching,
        n_nodes=mass.size,
        candidate_pairs=int(config["candidate_pair_merges"]),
        bootstrap_draws=int(config["partition_bootstrap_draws"]),
        seed=int(config["seed"]),
        majority_lower_bound=float(config["bootstrap_majority_lower_bound"]),
    )
    diagnostics = {
        "method": "graph_constrained_multisource_consensus_coarsening",
        "partition": partition,
        "graph_anchor": {
            "training_source": HIGHORDER_SOURCE,
            "train_case_ids": [sample["event_id"] for sample in highorder],
            "validation_case_ids": [],
            "test_case_ids": [],
            "ridge_fixed_from_prior_fold": graph_ridge,
            "weak_system": {**weak, **solver},
            "damping": theta[: mass.size],
            "coupling": coupling,
        },
        "synchronizing_graph": {
            "stiffness_minimum": float(np.min(stiffness)),
            "stiffness_maximum": float(np.max(stiffness)),
            "generalized_eigenvalues": eigenvalues,
            "mass_orthonormality_error": float(
                np.max(np.abs(eigenvectors.T @ np.diag(mass) @ eigenvectors - np.eye(mass.size)))
            ),
            "graph_aep_matching": graph_matching,
            "minimum_aep_defect": float(aep_rows[0]["aep_defect"]),
        },
        "consensus": consensus,
        "quality_filter": {
            "minimum_train_steps": int(config["minimum_train_steps"]),
            "coherency_horizon_s": float(config["coherency_horizon_s"]),
            "short_cases_enter_partition_discovery": False,
        },
        "claims": {
            "physical_machine_parameter_recovery": False,
            "strict_low_pass_filter": False,
            "train_only_partition_discovery": True,
            "observed_Pm_is_conditional": True,
        },
    }
    top_aep = [
        {
            "rank": index + 1,
            "pairs": str(tuple(row["pairs"])),
            "partition": str(tuple(row["partition"])),
            "aep_defect": row["aep_defect"],
        }
        for index, row in enumerate(aep_rows[:25])
    ]
    return partition, diagnostics, top_aep


def _bootstrap_difference(
    rows: list[dict[str, Any]], draws: int, seed: int
) -> dict[str, Any]:
    by_event: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row["status"] == "completed":
            by_event.setdefault(str(row["event_id"]), {})[str(row["model_route"])] = row
    paired = []
    for event_id, routes in by_event.items():
        if {"native_reduced", "generator_10_then_project"}.issubset(routes):
            native = routes["native_reduced"]
            projected = routes["generator_10_then_project"]
            paired.append(
                {
                    "event_id": event_id,
                    "fault_line": native["fault_line"],
                    "source_dataset": native["source_dataset"],
                    "difference": float(native["rmse"] - projected["rmse"]),
                    "native_rmse": float(native["rmse"]),
                    "projected_rmse": float(projected["rmse"]),
                }
            )
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
        selected = rng.choice(clusters, size=len(clusters), replace=True)
        sample = [row for cluster in selected for row in paired if row["fault_line"] == cluster]
        estimates.append(float(np.mean([row["difference"] for row in sample])))
    native_mean = float(np.mean([row["native_rmse"] for row in paired]))
    projected_mean = float(np.mean([row["projected_rmse"] for row in paired]))
    return {
        "n_paired_cases": len(paired),
        "n_fault_line_clusters": len(clusters),
        "mean_native_rmse": native_mean,
        "mean_projected_10_rmse": projected_mean,
        "relative_improvement": (projected_mean - native_mean) / max(projected_mean, 1e-15),
        "mean_difference_native_minus_projected": native_mean - projected_mean,
        "ci95_lower": float(np.quantile(estimates, 0.025)),
        "ci95_upper": float(np.quantile(estimates, 0.975)),
        "resampling_unit": "fault_line",
        "draws": draws,
        "paired_rows": paired,
    }


def _assessment(
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    models: dict[str, TransientEmfM0Model],
    bootstrap: dict[str, dict[str, Any]],
    design_rows: list[dict[str, Any]],
    partitions: dict[str, tuple[tuple[int, ...], ...]],
) -> dict[str, Any]:
    assessments = {}
    for representation, result in bootstrap.items():
        group = [row for row in rows if row["representation"] == representation]
        source_guard = {}
        for source in sorted({str(row["source_dataset"]) for row in group}):
            source_rows = [row for row in group if row["source_dataset"] == source]
            native = np.mean(
                [row["rmse"] for row in source_rows if row["model_route"] == "native_reduced"]
            )
            projected = np.mean(
                [
                    row["rmse"]
                    for row in source_rows
                    if row["model_route"] == "generator_10_then_project"
                ]
            )
            source_guard[source] = {
                "native_mean_rmse": float(native),
                "projected_10_mean_rmse": float(projected),
                "native_not_worse": bool(native <= projected),
            }
        model = models[representation]
        gates = {
            "all_rollouts_completed": all(row["status"] == "completed" for row in group),
            "native_mean_lower_than_matched_projected_10": bool(
                result["mean_difference_native_minus_projected"] < 0.0
            ),
            "fault_line_bootstrap_ci_upper_below_zero": bool(result["ci95_upper"] < 0.0),
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
            "n_clusters": len(partitions[representation]),
            "status": "SUPPORTED" if all(gates.values()) else "NOT_SUPPORTED",
            "gates": gates,
            "source_guard": source_guard,
            "paired_comparison": {key: value for key, value in result.items() if key != "paired_rows"},
            "selected_alpha": model.alpha,
            "selected_tau": model.tau,
            "selected_activation_threshold": model.activation_threshold,
        }
    design = {row["representation"]: row for row in design_rows}
    efficiency = {}
    for representation, partition in partitions.items():
        efficiency[representation] = {
            "n_clusters": len(partition),
            "parameter_count": int(design[representation]["n_parameters"]),
            "parameter_count_10": int(design["generator_10"]["n_parameters"]),
            "parameter_reduction_fraction": 1.0
            - float(design[representation]["n_parameters"])
            / float(design["generator_10"]["n_parameters"]),
            "condition_number": float(design[representation]["condition_number"]),
            "condition_number_10": float(design["generator_10"]["condition_number"]),
            "condition_number_ratio": float(design[representation]["condition_number"])
            / float(design["generator_10"]["condition_number"]),
        }
    primary = str(config["primary_partition"])
    return {
        "question": (
            "Can graph-constrained, source-robust adaptive coarsening improve basic M0-3EG "
            "relative to direct ten-machine identification?"
        ),
        "primary_partition": primary,
        "primary_answer": assessments[primary]["status"],
        "primary_interpretation": (
            "graph-consensus coarsening is supported on the predeclared matched-coordinate gate"
            if assessments[primary]["status"] == "SUPPORTED"
            else "graph-consensus coarsening is not established on the predeclared matched-coordinate gate"
        ),
        "partition_assessments": assessments,
        "efficiency_and_identifiability": efficiency,
        "claim_boundary": {
            "fold": config["fold"],
            "seed": config["seed"],
            "test_read": False,
            "observed_Pm_is_conditional_upper_bound": True,
            "five_fold_generalization_claim": False,
            "partition_selected_on_validation": False,
            "fixed_seven_cluster_claim": False,
        },
    }


def _make_figure(
    result_dir: Path,
    rows: list[dict[str, Any]],
    assessment: dict[str, Any],
    design_rows: list[dict[str, Any]],
    graph_diagnostics: dict[str, Any],
) -> None:
    names = list(assessment["partition_assessments"])
    labels = [
        f"{assessment['partition_assessments'][name]['n_clusters']}-node\n{name.replace('clustered_', '')}"
        for name in names
    ]
    colors = {"generator_10_then_project": "#315A7D", "native_reduced": "#B4473D"}
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9))
    positions = np.arange(len(names))
    for route, offset in (("generator_10_then_project", -0.16), ("native_reduced", 0.16)):
        values = [
            [
                row["rmse"]
                for row in rows
                if row["representation"] == name
                and row["model_route"] == route
                and row["status"] == "completed"
            ]
            for name in names
        ]
        axes[0].boxplot(
            values,
            positions=positions + offset,
            widths=0.27,
            patch_artist=True,
            showfliers=False,
            boxprops={"facecolor": colors[route], "alpha": 0.75, "linewidth": 0.8},
            medianprops={"color": "white", "linewidth": 1.4},
        )
    axes[0].set_xticks(positions, labels)
    axes[0].set_ylabel("Matched-coordinate RMSE")
    axes[0].set_title("a  Conditional reintegration", loc="left", fontweight="bold")
    axes[0].plot([], [], color=colors["generator_10_then_project"], lw=7, label="10-node projected")
    axes[0].plot([], [], color=colors["native_reduced"], lw=7, label="native reduced")
    axes[0].legend(frameon=False, fontsize=8)

    design = {row["representation"]: row for row in design_rows}
    design_names = ["generator_10", *names]
    condition = [float(design[name]["condition_number"]) for name in design_names]
    parameters = [int(design[name]["n_parameters"]) for name in design_names]
    x = np.arange(len(design_names))
    axes[1].bar(x - 0.16, condition, width=0.32, color="#6A8CAF", label="condition number")
    twin = axes[1].twinx()
    twin.bar(x + 0.16, parameters, width=0.32, color="#D8A657", label="parameters")
    axes[1].set_yscale("log")
    axes[1].set_xticks(
        x,
        [
            "10-node",
            *[
                f"{assessment['partition_assessments'][name]['n_clusters']}-node"
                for name in names
            ],
        ],
    )
    axes[1].set_ylabel("Weak-design condition number")
    twin.set_ylabel("D + K parameter count")
    axes[1].set_title("b  Identifiability", loc="left", fontweight="bold")

    eigenvalues = np.asarray(
        graph_diagnostics["synchronizing_graph"]["generalized_eigenvalues"], dtype=float
    )
    axes[2].plot(np.arange(len(eigenvalues)), eigenvalues, "o-", color="#557A46", lw=1.3)
    axes[2].set_xlabel("Generalized mode index")
    axes[2].set_ylabel(r"$\lambda(L_s\phi=\lambda M\phi)$")
    axes[2].set_title("c  Train-only swing spectrum", loc="left", fontweight="bold")
    axes[2].grid(alpha=0.2)
    fig.tight_layout()
    figures = result_dir / "figures"
    figures.mkdir()
    fig.savefig(figures / "fig1_graph_coherent_m0_3eg.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures / "fig1_graph_coherent_m0_3eg.svg", bbox_inches="tight")
    plt.close(fig)


def _write_report(
    result_dir: Path,
    run_id: str,
    config: dict[str, Any],
    assessment: dict[str, Any],
    graph: dict[str, Any],
) -> None:
    primary = str(config["primary_partition"])
    primary_row = assessment["partition_assessments"][primary]
    paired = primary_row["paired_comparison"]
    efficiency = assessment["efficiency_and_identifiability"][primary]
    selected_pairs = graph["consensus"]["selected_pairs"]
    source_lines = []
    for source, row in primary_row["source_guard"].items():
        source_lines.append(
            f"| {source} | {row['native_mean_rmse']:.6f} | "
            f"{row['projected_10_mean_rmse']:.6f} | {row['native_not_worse']} |"
        )
    report = rf"""# 图约束跨源稳健粗化对基础 M0-3EG 的影响

**运行：** `{run_id}`<br>
**证据边界：** {config['fold']}、seed {config['seed']}、validation-only、test 未读取；observed $P_m(t)$ 仅为条件重积分上界。

## 结论

主候选为 {len(graph['partition'])} 节点图共识分区，训练阶段仅接受合并 {selected_pairs}。其 H=100 匹配坐标平均 RMSE 为 {paired['mean_native_rmse']:.6f}，十机模型投影后的平均 RMSE 为 {paired['mean_projected_10_rmse']:.6f}，相对变化为 {paired['relative_improvement']*100:.2f}%。故障线路聚类 bootstrap 的差值 95% CI 为 [{paired['ci95_lower']:.6f}, {paired['ci95_upper']:.6f}]，预声明门给出 **{primary_row['status']}**。

该结论不能外推为因果在线性能或五折普遍性。若状态为 NOT_SUPPORTED，仍接收本轮过程证据，但不宣称方法优于十机。

## 第一性原理方法

在平衡点附近，基础摆动骨架满足

$$M\ddot\delta + D\dot\delta + L_s\delta = \Delta P_m,\qquad
L_s=B\,\mathrm{{diag}}\!\left(K\cos(B^\top\delta_{{eq}})\right)B^\top.$$

一个聚类投影 $P$ 若要在不依赖具体轨迹的情况下保持簇一致运动，必须近似满足

$$ (I-P)M^{{-1}}L_sP\approx0. $$

本轮用其相对 Frobenius 范数作为 almost-equitable-partition (AEP) 缺陷，并解 $L_s\phi=\lambda M\phi$ 审计慢模。仅有图相似仍不够：被合并机组还必须在每个训练数据源内同时具有接近的 $\delta/\omega$ 响应和接近的 $M^{{-1}}\Delta P_m$ 输入。每个视角用 1000 次 case bootstrap 估计最优三对匹配的选择概率，要求 Wilson 95% 下界大于 0.5；最后取图 AEP、所有数据源、响应视角和输入视角的严格交集。分歧不会被强行压成七节点，而是自动减少合并数。

## 数据质量与泄漏控制

- 只有 valid length 不少于 {config['minimum_train_steps']} 的训练轨迹进入分区发现；短轨迹不会获得额外权重。
- 图锚点只由 high-order 训练轨迹以积分弱形式求解，固定 ridge={config['graph_anchor_ridge']}，不读取验证或 test。
- 分区在任何验证重积分之前冻结；验证仅在各表示内部选择基础 M0-3EG 的 ridge、$\alpha$、$\tau$ 与应力阈值。
- `Pe`、电压、标签、事件时刻和未来状态均未进入模型或分区发现。

## 结果

| 数据来源 | 原生降阶 RMSE | 十机投影 RMSE | 原生不劣 |
|---|---:|---:|---:|
{chr(10).join(source_lines)}

弱式参数数目从 {efficiency['parameter_count_10']} 降至 {efficiency['parameter_count']}（减少 {efficiency['parameter_reduction_fraction']*100:.1f}%）；条件数从 {efficiency['condition_number_10']:.4f} 变为 {efficiency['condition_number']:.4f}。这些量用于说明可辨识性，不等同于真实机组参数恢复。

![图共识 M0-3EG](figures/fig1_graph_coherent_m0_3eg.png)

## 物理解读

固定七聚类相当于无条件删除三个簇内差模。严格共识方法只删除在两个数据源、响应、机械输入和同步图上都稳定的差模；其余差模被保留，因为现有数据不能证明它们是同一个系统范围内可忽略的高频模。因而，这是一种受物理约束的保守图粗化，而不是把聚类泛称为低通滤波器。

## 审计边界

本轮只使用基础 `TransientEmfM0Model`；没有 GMD、DF、M2A 残差、神经网络或额外 ODE。报告中的 $D$、$K$ 与 $L_s$ 是降阶辨识参数及其线性化同步图，不宣称等于真实 Kron 网络或机组参数。

## 方法学依据

1. Chow, Cullum & Willoughby, *A sparse matrix approach to the analysis of power system dynamics*, IEEE Transactions on Power Apparatus and Systems (1984), DOI: [10.1109/TPAS.1984.318724](https://doi.org/10.1109/TPAS.1984.318724)。
2. Yusof, Alden & Rogers, *Slow coherency based network partitioning including load buses*, IEEE Transactions on Power Systems (1993), DOI: [10.1109/59.260855](https://doi.org/10.1109/59.260855)。
3. Tyuryukanov et al., *Controlled islanding of power networks based on graph reduction and spectral clustering*, IEEE Transactions on Power Systems (2021), DOI: [10.1109/TPWRS.2020.3009628](https://doi.org/10.1109/TPWRS.2020.3009628)。
4. Monshizadeh & van der Schaft, *Structure-preserving model reduction of physical network systems by clustering*，给出带节点/边权图的 almost-equitable partition 与二阶系统扩展：[arXiv:1403.4789](https://arxiv.org/abs/1403.4789)。
5. Cheng, Kawano & Scherpen, *Reduction of second-order network systems with structure preservation*，以节点动力学差异和图聚类构造保结构二阶降阶模型：[arXiv:1706.01751](https://arxiv.org/abs/1706.01751)。
6. Dörfler & Bullo, *Kron reduction of graphs with applications to electrical networks*，说明 Schur 补粗化对拓扑、谱与有效电阻的保持边界：[arXiv:1102.2950](https://arxiv.org/abs/1102.2950)。
"""
    (result_dir / "report.md").write_text(report, encoding="utf-8")


def _run_diagnostic(
    *, root: Path, config: dict[str, Any], config_path: Path, result_dir: Path, run_id: str
) -> None:
    store = CanonicalStore.open(root / "data" / str(config["dataset"]))
    split = load_split(
        root / "data" / str(config["dataset"]),
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
        raise RuntimeError("no eligible train/validation cases")
    request = ModelDataRequest.initial_state_rollout("observed")
    generator_samples = {
        role: [
            adapt_event(
                store.load_event(event_id),
                inertia_prior=store.dataset_metadata.M_prior,
                request=request,
                representation="generator_10",
            )
            for event_id in ids
        ]
        for role, ids in (("train", train_ids), ("val", val_ids))
    }
    for role_samples in generator_samples.values():
        for sample in role_samples:
            audit_model_sample(sample)

    graph_partition, graph_diagnostics, top_aep = _discover_graph_partition(
        generator_samples["train"], config
    )
    empirical_partition, empirical_pair_scores = _discover_train_partition(
        generator_samples["train"], float(config["coherency_horizon_s"])
    )
    partitions = {
        "clustered_graph_consensus": graph_partition,
        "clustered_7_trajectory": empirical_partition,
    }
    samples: dict[str, dict[str, list[dict[str, Any]]]] = {
        "generator_10": generator_samples
    }
    for representation, partition in partitions.items():
        samples[representation] = {
            role: [
                _aggregate_adapted_sample(sample, partition, representation)
                for sample in generator_samples[role]
            ]
            for role in ("train", "val")
        }
        for role in ("train", "val"):
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
        "primary_partition_groups": graph_partition,
        "primary_partition_n_clusters": len(graph_partition),
        "test_ids_read": [],
        "labels_used": False,
        "partition_discovery_used_train_only": True,
        "validation_selected_partition": False,
    }
    _json(result_dir / "selection_marker.json", _jsonable(selection_marker))

    rows = _paired_rollouts(
        models["generator_10"],
        {name: models[name] for name in partitions},
        samples,
        partitions,
        store,
        int(config["selection_horizon_steps"]),
    )
    for row in rows:
        if row["model_route"] == "native_7":
            row["model_route"] = "native_reduced"
    summary_rows = _summary(rows)
    bootstrap = {
        representation: _bootstrap_difference(
            [row for row in rows if row["representation"] == representation],
            int(config["comparison_bootstrap_draws"]),
            int(config["seed"]) + index,
        )
        for index, representation in enumerate(partitions)
    }
    assessment = _assessment(config, rows, models, bootstrap, design_rows, partitions)

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

    bootstrap_rows = []
    for source, source_row in graph_diagnostics["consensus"]["sources"].items():
        for view in ("dynamic", "forcing"):
            for row in source_row[f"{view}_bootstrap"]:
                bootstrap_rows.append(
                    {
                        "source_dataset": source,
                        "view": view,
                        "pair": f"{row['pair'][0]}-{row['pair'][1]}",
                        **{key: value for key, value in row.items() if key != "pair"},
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
    _write_csv(result_dir / "trajectory_partition_pair_scores.csv", empirical_pair_scores)
    _write_csv(result_dir / "coherency_bootstrap.csv", bootstrap_rows)
    _write_csv(result_dir / "aep_partition_candidates_top25.csv", top_aep)
    _write_csv(result_dir / "design_conditioning.csv", design_rows)
    _write_csv(result_dir / "paired_rollout_metrics.csv", rows)
    _write_csv(result_dir / "aggregate_metrics.csv", summary_rows)
    _write_csv(result_dir / "integration_convergence.csv", convergence_rows)
    _json(result_dir / "graph_partition_diagnostics.json", _jsonable(graph_diagnostics))
    _json(
        result_dir / "partition_definitions.json",
        {name: [list(group) for group in partition] for name, partition in partitions.items()},
    )
    _json(result_dir / "train_only_threshold_policies.json", policies)
    _json(
        result_dir / "paired_fault_line_bootstrap.json",
        {name: {key: value for key, value in row.items() if key != "paired_rows"} for name, row in bootstrap.items()},
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
            "validation_did_not_select_partition": True,
            "short_train_cases_excluded_by_predeclared_length_gate": True,
            "primary_partition_predeclared": config["primary_partition"],
            "config_sha256": _sha256(config_path),
        },
    )
    _make_figure(result_dir, rows, assessment, design_rows, graph_diagnostics)
    _write_report(result_dir, run_id, config, assessment, graph_diagnostics)

    selected_pairs = {
        tuple(pair) for pair in graph_diagnostics["consensus"]["selected_pairs"]
    }
    graph_pairs = {
        tuple(pair) for pair in graph_diagnostics["consensus"]["graph_matching"]
    }
    source_pair_sets = [
        {tuple(pair) for pair in row["source_supported_pairs"]}
        for row in graph_diagnostics["consensus"]["sources"].values()
    ]
    exact_consensus = set(graph_pairs)
    for pair_set in source_pair_sets:
        exact_consensus &= pair_set
    eigenvalues = np.asarray(
        graph_diagnostics["synchronizing_graph"]["generalized_eigenvalues"], dtype=float
    )
    process_gates = {
        "test_data_unopened": True,
        "labels_unused": True,
        "observed_Pm_marked_conditional": True,
        "selection_marker_written_before_comparison": True,
        "primary_partition_predeclared": config["primary_partition"] == "clustered_graph_consensus",
        "partition_selection_not_validation_driven": True,
        "short_case_quality_gate_predeclared": int(config["minimum_train_steps"]) >= 101,
        "graph_anchor_train_only": not graph_diagnostics["graph_anchor"]["validation_case_ids"],
        "synchronizing_stiffness_nonnegative": graph_diagnostics["synchronizing_graph"]["stiffness_minimum"] >= -1e-12,
        "generalized_laplacian_psd": float(np.min(eigenvalues)) >= -1e-10,
        "single_coi_zero_mode": int(np.sum(np.abs(eigenvalues) < 1e-10)) == 1,
        "selected_pairs_equal_exact_graph_source_intersection": selected_pairs
        == exact_consensus,
        "adaptive_cluster_count_not_forced_to_seven": len(graph_partition) != 7,
        "basic_M0_3EG_class_only": all(
            model.__class__ is TransientEmfM0Model for model in models.values()
        ),
        "all_model_physics_audits_pass": all(
            model.physics_audit().get("status") == "pass" for model in models.values()
        ),
        "all_rollouts_completed": all(row["status"] == "completed" for row in rows),
        "all_integration_convergence_audits_pass": all(row["convergent"] for row in convergence_rows),
        "fault_line_cluster_bootstrap_present": all(
            row["n_fault_line_clusters"] >= 2 for row in bootstrap.values()
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
        raise RuntimeError("graph-coherent M0-3EG diagnostic process gate failed")
    artifacts = {
        path.relative_to(result_dir).as_posix(): _sha256(path)
        for path in result_dir.rglob("*")
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    _json(result_dir / "artifact_hashes.json", artifacts)


def _run_adversarial_audit(
    *, root: Path, config: dict[str, Any], result_dir: Path, run_id: str
) -> None:
    diagnostic = _latest_result(root, str(config["diagnostic_experiment"]), "diagnostic")
    claims = json.loads((diagnostic / "claim_assessment.json").read_text(encoding="utf-8"))
    process = json.loads(
        (diagnostic / "diagnostic_process_acceptance.json").read_text(encoding="utf-8")
    )
    lineage = json.loads((diagnostic / "lineage.json").read_text(encoding="utf-8"))
    graph = json.loads((diagnostic / "graph_partition_diagnostics.json").read_text(encoding="utf-8"))
    selection = json.loads((diagnostic / "selection_marker.json").read_text(encoding="utf-8"))
    rows = _read_csv(diagnostic / "paired_rollout_metrics.csv")
    primary = str(config["primary_partition"])
    selected = {tuple(pair) for pair in graph["consensus"]["selected_pairs"]}
    graph_pairs = {tuple(pair) for pair in graph["consensus"]["graph_matching"]}
    supported = [
        {tuple(pair) for pair in row["source_supported_pairs"]}
        for row in graph["consensus"]["sources"].values()
    ]
    exact_consensus = set(graph_pairs)
    for source_pairs in supported:
        exact_consensus &= source_pairs
    primary_rows = [row for row in rows if row["representation"] == primary]
    gates = {
        "artifact_hashes_match": not _verify_artifact_hashes(diagnostic),
        "diagnostic_process_passed": process.get("status") == "PASS",
        "test_data_unopened": not lineage.get("test_ids_read"),
        "labels_unused": not lineage.get("labels_used_for_fit_selection_or_thresholds"),
        "observed_Pm_claim_bounded": bool(lineage.get("observed_Pm_is_conditional_noncausal_upper_bound")),
        "partition_frozen_before_validation_comparison": bool(selection.get("selection_complete"))
        and not selection.get("validation_selected_partition"),
        "strict_consensus_recomputed": selected == exact_consensus,
        "fixed_seven_was_not_forced": selection.get("primary_partition_n_clusters") != 7,
        "matched_coordinate_routes_present": {
            row["model_route"] for row in primary_rows
        }
        == {"native_reduced", "generator_10_then_project"},
        "fault_line_uncertainty_required_for_supported_claim": (
            claims["primary_answer"] != "SUPPORTED"
            or claims["partition_assessments"][primary]["gates"][
                "fault_line_bootstrap_ci_upper_below_zero"
            ]
        ),
        "scientific_answer_follows_registered_gates": claims["primary_answer"]
        == claims["partition_assessments"][primary]["status"],
        "single_fold_boundary_explicit": not claims["claim_boundary"]["five_fold_generalization_claim"],
    }
    status = "PASS" if all(gates.values()) else "FAIL"
    review = {
        "status": status,
        "decision": "ACCEPTABLE_WITH_BOUNDED_SCIENTIFIC_CLAIM" if status == "PASS" else "REJECT",
        "run_id": run_id,
        "diagnostic_result": diagnostic.name,
        "gates": gates,
        "adversarial_questions": [
            "Did short trajectories dominate partition discovery?",
            "Was a fixed seven-cluster result forced despite cross-source disagreement?",
            "Did the graph score use validation or test trajectories?",
            "Was native reduced error compared against an unmatched ten-node coordinate?",
            "Was observed Pm misrepresented as a causal deployment result?",
            "Was a negative scientific result silently promoted?",
        ],
    }
    _json(result_dir / "adversarial_review.json", review)
    (result_dir / "adversarial_review.md").write_text(
        "# 图共识 M0-3EG 对抗性审查\n\n"
        f"**决定：{review['decision']}**\n\n"
        + "\n".join(f"- {'PASS' if value else 'FAIL'} `{key}`" for key, value in gates.items())
        + "\n",
        encoding="utf-8",
    )
    if status != "PASS":
        raise RuntimeError("adversarial audit rejected graph-coherent M0-3EG")


def _run_acceptance(
    *, root: Path, config: dict[str, Any], result_dir: Path, run_id: str
) -> None:
    diagnostic = _latest_result(root, str(config["diagnostic_experiment"]), "diagnostic")
    adversarial = _latest_result(
        root, str(config["adversarial_experiment"]), "adversarial_audit"
    )
    process = json.loads(
        (diagnostic / "diagnostic_process_acceptance.json").read_text(encoding="utf-8")
    )
    review = json.loads(
        (adversarial / "adversarial_review.json").read_text(encoding="utf-8")
    )
    claims = json.loads((diagnostic / "claim_assessment.json").read_text(encoding="utf-8"))
    gates = {
        "diagnostic_process_passed": process.get("status") == "PASS",
        "diagnostic_hashes_match": not _verify_artifact_hashes(diagnostic),
        "adversarial_review_passed": review.get("status") == "PASS",
        "bounded_decision": review.get("decision") == "ACCEPTABLE_WITH_BOUNDED_SCIENTIFIC_CLAIM",
        "negative_result_would_remain_acceptable": claims["primary_answer"] in {"SUPPORTED", "NOT_SUPPORTED"},
    }
    status = "PASS" if all(gates.values()) else "FAIL"
    acceptance = {
        "status": status,
        "decision": "ACCEPT" if status == "PASS" else "REJECT",
        "scientific_answer": claims["primary_answer"],
        "scope": (
            "F3 single-seed validation; basic M0-3EG; graph-constrained source-robust adaptive "
            "coarsening; observed-Pm conditional reintegration; test unopened"
        ),
        "run_id": run_id,
        "diagnostic_result": diagnostic.name,
        "adversarial_result": adversarial.name,
        "gates": gates,
    }
    _json(result_dir / "acceptance.json", acceptance)
    (result_dir / "acceptance.md").write_text(
        "# 最终验收\n\n"
        f"**过程决定：{acceptance['decision']}**  \n"
        f"**科学回答：{acceptance['scientific_answer']}**\n\n"
        "接收范围仅限 F3、单 seed、observed-Pm 条件重积分下的基础 M0-3EG 图粗化证据；test 未读取。\n",
        encoding="utf-8",
    )
    if status != "PASS":
        raise RuntimeError("graph-coherent M0-3EG acceptance failed")


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
        raise RuntimeError(f"unsupported graph-coherent M0-3EG stage: {stage}")
