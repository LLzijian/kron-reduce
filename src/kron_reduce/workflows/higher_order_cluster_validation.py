"""Focused validation of the data-level clustering decision on higher-order data."""

from __future__ import annotations

import csv
import hashlib
import json
import tomllib
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from kron_reduce.data import (
    CanonicalStore,
    ModelDataRequest,
    adapt_event,
    load_split,
)
from kron_reduce.data.audit import audit_model_sample
from kron_reduce.physics import CLUSTERED_7_PARTITION
from kron_reduce.selection import (
    ClusterAuditResult,
    ClusterDecision,
    ClusterThresholds,
    audit_cluster_decision,
    decide_clustering,
)

WORKFLOW = "higher_order_cluster_validation_v1"
HIGHER_ORDER_SOURCE = "ieee39_highorder_dataset"
LEGACY_7_PARTITION = ((0, 7), (1, 2), (3,), (4,), (5, 6), (8,), (9,))
TRAIN_DISCOVERED_7_PARTITION = ((0, 7), (1, 2), (3, 6), (4,), (5,), (8,), (9,))
REFERENCE_PARTITIONS = {
    "clustered_7_current": tuple(tuple(group) for group in CLUSTERED_7_PARTITION),
    "clustered_7_legacy": LEGACY_7_PARTITION,
    "clustered_7_train_discovered": TRAIN_DISCOVERED_7_PARTITION,
}


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_plain(value), indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf-8",
    )


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
            writer.writerow(
                {
                    key: json.dumps(_plain(value), ensure_ascii=False)
                    if isinstance(value, (dict, list, tuple, np.ndarray))
                    else _plain(value)
                    for key, value in row.items()
                }
            )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_hashes(result_dir: Path) -> None:
    _json(
        result_dir / "artifact_hashes.json",
        {
            path.relative_to(result_dir).as_posix(): _file_sha256(path)
            for path in sorted(result_dir.rglob("*"))
            if path.is_file() and path.name != "artifact_hashes.json"
        },
    )


def _line_key(value: str) -> tuple[int, str]:
    try:
        return int(str(value).rsplit("_", 1)[-1]), str(value)
    except ValueError:
        return 10**9, str(value)


def _event_hash(event_id: str, seed: int) -> str:
    return hashlib.sha256(f"{int(seed)}:{event_id}".encode()).hexdigest()


def select_higher_order_subsets(
    rows: Sequence[Mapping[str, str]],
    *,
    seed: int = 324,
    main_per_line: int = 2,
) -> tuple[dict[str, tuple[str, ...]], list[dict[str, Any]]]:
    """Create line-balanced, response-blind subsets using stable event hashes."""

    if main_per_line < 1:
        raise ValueError("main_per_line must be positive")
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        event_id = str(row["event_id"])
        line = str(row["fault_line"])
        grouped[line].append(event_id)
    if not grouped:
        raise ValueError("higher-order subset selection requires nonempty rows")

    main: list[str] = []
    small: list[str] = []
    manifest: list[dict[str, Any]] = []
    for line in sorted(grouped, key=_line_key):
        ordered = sorted(grouped[line], key=lambda event_id: (_event_hash(event_id, seed), event_id))
        selected_main = set(ordered[:main_per_line])
        selected_small = {ordered[0]}
        main.extend(ordered[:main_per_line])
        small.append(ordered[0])
        for rank, event_id in enumerate(ordered, start=1):
            manifest.append(
                {
                    "event_id": event_id,
                    "fault_line": line,
                    "sha256_rank_within_line": rank,
                    "selection_hash": _event_hash(event_id, seed),
                    "in_main_31": event_id in selected_main,
                    "in_small_16": event_id in selected_small,
                    "in_full_58": True,
                    "selection_uses_response": False,
                    "selection_uses_label": False,
                }
            )
    full = tuple(row["event_id"] for row in manifest)
    return {
        "main": tuple(main),
        "small": tuple(small),
        "full": full,
    }, manifest


def _thresholds(config: Mapping[str, Any]) -> ClusterThresholds:
    return ClusterThresholds(
        information=float(config["cluster_information_budget"]),
        input=float(config["cluster_input_budget"]),
        mode=float(config["cluster_mode_budget"]),
        spectral_gap_ratio=float(config["cluster_spectral_gap_ratio"]),
        state=float(config["cluster_state_budget"]),
        coupling=float(config["cluster_coupling_budget"]),
        closure=float(config["cluster_closure_budget"]),
        bootstrap_support=float(config["cluster_bootstrap_support"]),
        confidence=float(config["cluster_confidence"]),
    )


def _adapt_samples(
    store: CanonicalStore,
    event_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    request = ModelDataRequest.initial_state_rollout("zoh")
    inertia = store.dataset_metadata.M_prior
    output: dict[str, dict[str, Any]] = {}
    for event_id in event_ids:
        sample = adapt_event(
            store.load_event(event_id),
            inertia_prior=inertia,
            request=request,
            representation="generator_10",
        )
        sample["training_context"].pop("label", None)
        metadata = sample["training_context"].get("metadata", {})
        sample["training_context"]["metadata"] = {
            "event_id": str(event_id),
            "fault_line": str(metadata["fault_line"]),
            "source_dataset": str(metadata["source_dataset"]),
        }
        audit_model_sample(sample)
        output[event_id] = sample
    return output


def _decision(
    samples: Sequence[dict[str, Any]],
    config: Mapping[str, Any],
    thresholds: ClusterThresholds,
    seed: int,
) -> ClusterDecision:
    return decide_clustering(
        samples,
        thresholds=thresholds,
        seed=int(seed),
        bootstrap_draws=int(config["cluster_bootstrap_draws"]),
        max_steps=int(config["cluster_max_steps"]),
        delays=int(config["cluster_delays"]),
    )


def validation_verdict(
    primary: ClusterDecision,
    candidate_rows: Sequence[Mapping[str, Any]],
    role_seed_decisions: Sequence[ClusterDecision],
    sensitivity_decisions: Sequence[ClusterDecision],
    *,
    required_seed_agreement: int = 4,
) -> dict[str, Any]:
    """Apply the predeclared accept/refuse validation decision table."""

    agreement = sum(
        decision.use_clustering == primary.use_clustering
        for decision in role_seed_decisions
    )
    seed_stable = agreement >= int(required_seed_agreement)
    sensitivity_stable = all(
        decision.use_clustering == primary.use_clustering
        for decision in sensitivity_decisions
    )
    passing = sorted(
        int(row["n_clusters"]) for row in candidate_rows if bool(row["passes_all"])
    )
    if not seed_stable or not sensitivity_stable:
        verdict = "INCONCLUSIVE"
    elif primary.use_clustering:
        selected = next(
            (
                row
                for row in candidate_rows
                if int(row["n_clusters"]) == int(primary.selected_rank)
            ),
            None,
        )
        if selected is None or not bool(selected["passes_all"]):
            verdict = "FAILED_FALSE_ACCEPTANCE"
        elif passing and int(primary.selected_rank) != passing[0]:
            verdict = "FAILED_RANK_SELECTION"
        else:
            verdict = "VALIDATED_ACCEPTANCE"
    elif passing:
        verdict = "FAILED_FALSE_REJECTION"
    else:
        verdict = "VALIDATED_REFUSAL"
    return {
        "verdict": verdict,
        "primary_use_clustering": bool(primary.use_clustering),
        "role_seed_agreement_count": int(agreement),
        "required_role_seed_agreement": int(required_seed_agreement),
        "role_seed_stable": bool(seed_stable),
        "sensitivity_stable": bool(sensitivity_stable),
        "audit_passing_candidate_ranks": passing,
        "selected_rank": int(primary.selected_rank),
    }


def _stability_row(
    variant: str,
    n_events: int,
    seed: int,
    decision: ClusterDecision,
) -> dict[str, Any]:
    return {
        "variant": variant,
        "n_events": int(n_events),
        "role_seed": int(seed),
        "use_clustering": bool(decision.use_clustering),
        "n_clusters": int(decision.n_clusters),
        "selected_rank": int(decision.selected_rank),
        "score": float(decision.score),
        "bootstrap_support": float(decision.bootstrap_support),
        "partition": [list(group) for group in decision.partition],
        "fit_groups": list(decision.roles["fit"]),
        "selection_groups": list(decision.roles["selection"]),
        "audit_groups": list(decision.roles["audit"]),
    }


def _plot_cluster_loss(decision: ClusterDecision, path: Path) -> None:
    rows = list(decision.candidate_scores)
    ranks = [int(row["n_clusters"]) for row in rows]
    point = [float(row["point_score"]) for row in rows]
    upper = [float(row["bootstrap_upper"]) for row in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(ranks, point, marker="o", label="point score")
    ax.plot(ranks, upper, marker="s", label="95% bootstrap upper")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="pass boundary")
    ax.set_xlabel("Number of clusters k")
    ax.set_ylabel("Normalized worst loss")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_partition(
    decision: ClusterDecision,
    audit: ClusterAuditResult,
    path: Path,
) -> None:
    if decision.use_clustering:
        partition = decision.partition
        label = f"selected k={decision.n_clusters}"
    else:
        closest = min(
            audit.candidate_audits,
            key=lambda row: (float(row["hard_score"]), int(row["n_clusters"])),
        )
        partition = tuple(tuple(group) for group in closest["partition"])
        label = f"closest audited candidate k={closest['n_clusters']} (not selected)"
    membership = np.zeros((decision.n_original, decision.n_original), dtype=float)
    for group in partition:
        membership[np.ix_(group, group)] = 1.0
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    image = ax.imshow(membership, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xlabel("Original node")
    ax.set_ylabel("Original node")
    ax.set_title(label)
    ax.set_xticks(range(decision.n_original))
    ax.set_yticks(range(decision.n_original))
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="same cluster")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _report(summary: Mapping[str, Any]) -> str:
    verdict = str(summary["verdict"])
    primary = summary["primary_decision"]
    return "\n".join(
        [
            "# Higher-order 聚类选择验证",
            "",
            f"- 最终判定：`{verdict}`",
            f"- 主子集：{summary['data']['main_events']} 条轨迹，覆盖 {summary['data']['fault_lines']} 条独立故障线路。",
            f"- 第一阶段选择：{'使用聚类' if primary['use_clustering'] else '保持十节点'}。",
            f"- 五种线路划分一致次数：{summary['role_seed_agreement_count']}/{summary['required_role_seed_agreement']}。",
            f"- 审计中全部通过的候选维数：{summary['audit_passing_candidate_ranks']}。",
            "- 所有轨迹和模态指标均在原始十节点坐标评价；审计数据没有返回训练或路由。",
            "- 未读取官方验证/测试性能，标签和离线通道未用于子集选择、拟合或判断。",
            "",
        ]
    )


def _validate_scope(config: Mapping[str, Any]) -> None:
    expected = {
        "dataset": "ieee39_postclear_canonical_v1",
        "view": "combined_dynamics",
        "split_name": "combined_line_5fold_v1",
        "fold": "F1",
        "seed": 324,
        "pm_mode": "zoh",
        "test_access": "forbidden",
        "validation_access": "forbidden",
        "other_folds_access": "forbidden",
        "source_dataset": HIGHER_ORDER_SOURCE,
    }
    mismatch = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"higher-order cluster validation scope violation: {mismatch}")


def _run_validation(
    root: Path,
    config_path: Path,
    config: Mapping[str, Any],
    result_dir: Path,
    run_id: str,
) -> None:
    if any(result_dir.iterdir()):
        raise RuntimeError("validation result directory must be newly created and empty")
    dataset_dir = root / "data" / str(config["dataset"])
    store = CanonicalStore.open(dataset_dir)
    split = load_split(
        dataset_dir,
        view=str(config["view"]),
        split_name=str(config["split_name"]),
        fold=str(config["fold"]),
    )
    rows = []
    for event_id in split.train_ids:
        metadata = store.metadata(event_id)
        if metadata.source_dataset == str(config["source_dataset"]):
            rows.append({"event_id": event_id, "fault_line": metadata.fault_line})
    subsets, subset_manifest = select_higher_order_subsets(
        rows,
        seed=int(config["seed"]),
        main_per_line=int(config["main_events_per_fault_line"]),
    )
    expected_counts = {
        "full": int(config["expected_full_events"]),
        "main": int(config["expected_main_events"]),
        "small": int(config["expected_fault_lines"]),
    }
    actual_counts = {name: len(values) for name, values in subsets.items()}
    if actual_counts != expected_counts:
        raise RuntimeError(
            f"unexpected higher-order subset counts: {actual_counts} != {expected_counts}"
        )
    if len({row["fault_line"] for row in rows}) != int(config["expected_fault_lines"]):
        raise RuntimeError("higher-order F1 train fault-line count changed")

    samples_by_id = _adapt_samples(store, subsets["full"])
    threshold = _thresholds(config)
    role_seeds = tuple(int(value) for value in config["role_seeds"])
    if role_seeds[0] != int(config["seed"]) or len(role_seeds) != 5:
        raise RuntimeError("role_seeds must contain the five predeclared seeds")
    main_samples = [samples_by_id[event_id] for event_id in subsets["main"]]
    role_decisions = [
        _decision(main_samples, config, threshold, seed) for seed in role_seeds
    ]
    primary = role_decisions[0]
    small_samples = [samples_by_id[event_id] for event_id in subsets["small"]]
    full_samples = [samples_by_id[event_id] for event_id in subsets["full"]]
    small_decision = _decision(small_samples, config, threshold, int(config["seed"]))
    full_decision = _decision(full_samples, config, threshold, int(config["seed"]))
    audit = audit_cluster_decision(
        main_samples,
        primary,
        thresholds=threshold,
        seed=int(config["seed"]),
        max_steps=int(config["cluster_max_steps"]),
        rollout_horizons_s=tuple(float(value) for value in config["rollout_horizons_s"]),
        rollout_budget=float(config["rollout_budget"]),
        cpig_bootstrap_draws=int(config["cpig_bootstrap_draws"]),
        baseline_partitions=REFERENCE_PARTITIONS,
    )
    verdict = validation_verdict(
        primary,
        audit.candidate_audits,
        role_decisions,
        (small_decision, full_decision),
        required_seed_agreement=int(config["required_seed_agreement"]),
    )
    stability_rows = [
        _stability_row("main_31", len(main_samples), seed, decision)
        for seed, decision in zip(role_seeds, role_decisions, strict=True)
    ]
    stability_rows.extend(
        [
            _stability_row("small_16", len(small_samples), int(config["seed"]), small_decision),
            _stability_row("full_58", len(full_samples), int(config["seed"]), full_decision),
        ]
    )
    summary = {
        "run_id": run_id,
        **verdict,
        "status": "COMPLETED",
        "workflow": WORKFLOW,
        "data": {
            "source_dataset": HIGHER_ORDER_SOURCE,
            "full_events": len(full_samples),
            "main_events": len(main_samples),
            "small_events": len(small_samples),
            "fault_lines": int(config["expected_fault_lines"]),
            "official_validation_ids_read": [],
            "test_ids_read": [],
            "other_folds_read": [],
            "labels_used": False,
            "offline_channels_used": [],
        },
        "primary_decision": primary.to_dict(),
        "thresholds": asdict(threshold),
        "rollout_horizons_s": list(audit.rollout_horizons_s),
        "rollout_budget": float(audit.rollout_budget),
        "decision_unchanged_during_audit": bool(audit.decision_unchanged),
        "reference_partitions_are_labels": False,
    }

    (result_dir / "frozen_config.toml").write_bytes(config_path.read_bytes())
    _csv(result_dir / "subset_manifest.csv", subset_manifest)
    _json(result_dir / "cluster_decision.json", primary.to_dict())
    _csv(result_dir / "candidate_validation.csv", audit.candidate_audits)
    _csv(result_dir / "decision_stability.csv", stability_rows)
    _csv(result_dir / "baseline_partitions.csv", audit.baseline_audits)
    _json(result_dir / "validation_summary.json", summary)
    _json(result_dir / "cluster_audit.json", audit.to_dict())
    _json(
        result_dir / "lineage.json",
        {
            "run_id": run_id,
            "workflow": WORKFLOW,
            "stage": "validation_run",
            "selection_corpus": "F1 train higher-order only",
            "official_validation_ids_read": [],
            "test_ids_read": [],
            "other_folds_read": [],
            "labels_used_for_subset_fit_selection_or_audit": False,
            "offline_channels_read": [],
            "audit_changed_decision": False,
        },
    )
    _plot_cluster_loss(primary, result_dir / "cluster_loss_profile.png")
    _plot_partition(primary, audit, result_dir / "partition_audit.png")
    (result_dir / "report.md").write_text(_report(summary), encoding="utf-8")
    _write_hashes(result_dir)


def run(
    *, root: Path, experiment: dict[str, Any], stage: str, result_dir: Path, run_id: str
) -> None:
    if stage != "validation_run":
        raise ValueError(f"unsupported higher-order cluster validation stage: {stage}")
    config_path = root / "configs" / "experiments" / str(experiment["config"])
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    _validate_scope(config)
    _run_validation(root, config_path, config, result_dir, run_id)
