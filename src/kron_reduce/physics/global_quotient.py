from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .aggregation import validate_partition


Partition = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class QuotientCoordinates:
    """Mass-orthonormal retained and discarded node coordinates."""

    retained: np.ndarray
    discarded: np.ndarray

    @property
    def full(self) -> np.ndarray:
        return np.column_stack((self.retained, self.discarded))


def mass_orthonormal_quotient_coordinates(
    mass: np.ndarray, partition: Partition
) -> QuotientCoordinates:
    """Construct a complete M-orthonormal basis adapted to ``partition``."""

    mass = np.asarray(mass, dtype=float).reshape(-1)
    if np.any(mass <= 0.0) or not np.all(np.isfinite(mass)):
        raise ValueError("mass must be finite and strictly positive")
    partition = validate_partition(partition, len(mass))
    membership = np.zeros((len(mass), len(partition)), dtype=float)
    for column, group in enumerate(partition):
        membership[np.asarray(group, dtype=int), column] = 1.0
    cluster_mass = membership.T @ mass
    retained = membership / np.sqrt(cluster_mass)[None, :]

    euclidean_retained = np.sqrt(mass)[:, None] * retained
    _, _, right = np.linalg.svd(euclidean_retained.T, full_matrices=True)
    discarded_euclidean = right[len(partition) :].T
    discarded = discarded_euclidean / np.sqrt(mass)[:, None]
    return QuotientCoordinates(retained=retained, discarded=discarded)


def _coordinates(values: np.ndarray, basis: np.ndarray, mass: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float) @ (np.asarray(mass)[:, None] * basis)


def _ridge_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    evaluation_x: np.ndarray,
    *,
    ridge: float,
    rank_cap: int,
) -> np.ndarray:
    center = np.mean(train_x, axis=0)
    scale = np.std(train_x, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    left = (train_x - center) / scale
    right = (evaluation_x - center) / scale
    target_center = np.mean(train_y, axis=0)
    target = train_y - target_center
    u, singular, vt = np.linalg.svd(left, full_matrices=False)
    effective = int(np.sum(singular > max(float(singular[0]) * 1e-10, 1e-12)))
    use = min(max(effective, 1), int(rank_cap))
    gain = singular[:use] / (singular[:use] ** 2 + float(ridge))
    coefficients = vt[:use].T @ (gain[:, None] * (u[:, :use].T @ target))
    return right @ coefficients + target_center


def _path_rows(
    path: dict[str, Any],
    mass: np.ndarray,
    basis: QuotientCoordinates,
    *,
    horizon: int,
    delays: int,
    max_rows: int,
) -> list[dict[str, Any]]:
    delta = np.asarray(path["delta"], dtype=float)
    omega = np.asarray(path["omega"], dtype=float)
    forcing = np.asarray(path["u"], dtype=float)
    if not (len(delta) == len(omega) == len(forcing)):
        raise ValueError("delta, omega and u must have the same time dimension")
    if delta.shape[1] != len(mass):
        raise ValueError("path node dimension does not match mass")
    if len(delta) <= horizon + delays - 1:
        return []

    retained_delta = _coordinates(delta, basis.retained, mass)
    retained_omega = _coordinates(omega, basis.retained, mass)
    discarded_delta = _coordinates(delta, basis.discarded, mass)
    discarded_omega = _coordinates(omega, basis.discarded, mass)
    full_delta = _coordinates(delta, basis.full, mass)
    full_omega = _coordinates(omega, basis.full, mass)
    acceleration = forcing / mass[None, :]
    retained_input = _coordinates(acceleration, basis.retained, mass)
    full_input = _coordinates(acceleration, basis.full, mass)

    starts = np.arange(delays - 1, len(delta) - int(horizon))
    if len(starts) > int(max_rows):
        starts = starts[
            np.unique(np.linspace(0, len(starts) - 1, int(max_rows)).round().astype(int))
        ]
    rows: list[dict[str, Any]] = []
    for stop in starts:
        window = slice(int(stop) - delays + 1, int(stop) + 1)
        retained_state = np.concatenate((retained_delta, retained_omega), axis=1)
        full_state = np.concatenate((full_delta, full_omega), axis=1)
        base = np.concatenate(
            (retained_state[window].reshape(-1), retained_input[window].reshape(-1))
        )
        full = np.concatenate((full_state[window].reshape(-1), full_input[window].reshape(-1)))
        future = int(stop) + int(horizon)
        retained_increment = retained_state[future] - retained_state[int(stop)]
        full_increment = full_state[future] - full_state[int(stop)]
        hidden_increment = np.concatenate(
            (
                discarded_delta[future] - discarded_delta[int(stop)],
                discarded_omega[future] - discarded_omega[int(stop)],
            )
        )
        rows.append(
            {
                "base": base,
                "full": full,
                "target": retained_increment,
                "retained_energy": float(np.mean(retained_increment**2)),
                "full_energy": float(np.sum(full_increment**2)),
                "hidden_energy": float(np.sum(hidden_increment**2))
                if len(hidden_increment)
                else 0.0,
                "fault_line": str(path["fault_line"]),
            }
        )
    return rows


def _stack_equal_fault_line_rows(
    paths: Sequence[dict[str, Any]],
    mass: np.ndarray,
    basis: QuotientCoordinates,
    *,
    horizon: int,
    delays: int,
    max_rows_per_fault_line: int,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(paths, key=lambda value: str(value.get("event_id", ""))):
        for row in _path_rows(
            path,
            mass,
            basis,
            horizon=horizon,
            delays=delays,
            max_rows=max_rows_per_fault_line,
        ):
            grouped[row["fault_line"]].append(row)
    selected: list[dict[str, Any]] = []
    for line in sorted(grouped):
        rows = grouped[line]
        if len(rows) > int(max_rows_per_fault_line):
            indices = np.unique(
                np.linspace(0, len(rows) - 1, int(max_rows_per_fault_line))
                .round()
                .astype(int)
            )
            rows = [rows[int(index)] for index in indices]
        selected.extend(rows)
    if not selected:
        raise ValueError("no eligible direct-horizon rows")
    return {
        "base": np.stack([row["base"] for row in selected]),
        "full": np.stack([row["full"] for row in selected]),
        "target": np.stack([row["target"] for row in selected]),
        "retained_energy": np.asarray([row["retained_energy"] for row in selected]),
        "full_energy": np.asarray([row["full_energy"] for row in selected]),
        "hidden_energy": np.asarray([row["hidden_energy"] for row in selected]),
        "fault_line": [str(row["fault_line"]) for row in selected],
    }


def _fold_assignment(paths: Sequence[dict[str, Any]], folds: int, seed: int) -> dict[str, int]:
    lines = sorted({str(path["fault_line"]) for path in paths})
    if len(lines) < int(folds):
        raise ValueError("fewer fault-line clusters than cross-fitting folds")
    rng = np.random.default_rng(seed)
    order = np.asarray(lines, dtype=object)[rng.permutation(len(lines))].tolist()
    return {str(line): index % int(folds) for index, line in enumerate(order)}


def _cvar(values: np.ndarray, tail_fraction: float) -> float:
    values = np.sort(np.asarray(values, dtype=float))
    count = max(1, int(np.ceil(float(tail_fraction) * len(values))))
    return float(np.mean(values[-count:]))


def _bootstrap_cvar(
    values: np.ndarray,
    *,
    tail_fraction: float,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return {"status": "INSUFFICIENT_INFORMATION", "n_fault_lines": len(values)}
    rng = np.random.default_rng(seed)
    sampled = values[
        rng.integers(0, len(values), size=(int(draws), len(values)))
    ]
    statistics = np.asarray([_cvar(row, tail_fraction) for row in sampled])
    return {
        "status": "completed",
        "estimate": _cvar(values, tail_fraction),
        "ci_lower": float(np.quantile(statistics, 0.025)),
        "ci_upper": float(np.quantile(statistics, 0.975)),
        "bootstrap_draws": int(draws),
        "n_fault_lines": len(values),
        "tail_fraction": float(tail_fraction),
    }


def _line_gap_rows(
    evaluation_rows: dict[str, Any],
    base_prediction: np.ndarray,
    full_prediction: np.ndarray,
) -> list[dict[str, Any]]:
    base_loss = np.mean((evaluation_rows["target"] - base_prediction) ** 2, axis=1)
    full_loss = np.mean((evaluation_rows["target"] - full_prediction) ** 2, axis=1)
    by_line: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for index, line in enumerate(evaluation_rows["fault_line"]):
        bucket = by_line[str(line)]
        bucket["base_loss"].append(float(base_loss[index]))
        bucket["full_loss"].append(float(full_loss[index]))
        bucket["retained_energy"].append(float(evaluation_rows["retained_energy"][index]))
        bucket["full_energy"].append(float(evaluation_rows["full_energy"][index]))
        bucket["hidden_energy"].append(float(evaluation_rows["hidden_energy"][index]))
    output = []
    for line in sorted(by_line):
        bucket = by_line[line]
        retained_energy = float(np.mean(bucket["retained_energy"]))
        full_energy = float(np.mean(bucket["full_energy"]))
        if full_energy <= 1e-14:
            continue
        base_error = float(np.mean(bucket["base_loss"]))
        full_error = float(np.mean(bucket["full_loss"]))
        representation_loss = float(np.mean(bucket["hidden_energy"])) / full_energy
        closure_gain = max(0.0, base_error - full_error) / max(retained_energy, 1e-14)
        probe_skill = 1.0 - min(base_error, full_error) / max(retained_energy, 1e-14)
        output.append(
            {
                "fault_line": line,
                "representation_loss": representation_loss,
                "conditional_closure_gain": closure_gain,
                "uqg_local": max(representation_loss, closure_gain),
                "probe_skill": probe_skill,
            }
        )
    return output


def _summarize_horizon(
    line_rows: list[dict[str, Any]],
    *,
    horizon: int,
    folds: int,
    tail_fraction: float,
    bootstrap_draws: int,
    seed: int,
) -> dict[str, Any]:
    if len(line_rows) < int(folds):
        return {
            "horizon_steps": int(horizon),
            "status": "INSUFFICIENT_INFORMATION",
            "n_fault_lines": len(line_rows),
        }
    values = np.asarray([row["uqg_local"] for row in line_rows])
    interval = _bootstrap_cvar(
        values,
        tail_fraction=tail_fraction,
        draws=bootstrap_draws,
        seed=seed + int(horizon),
    )
    tail_count = max(1, int(np.ceil(tail_fraction * len(line_rows))))
    top_lines = sorted(
        line_rows, key=lambda row: (-float(row["uqg_local"]), row["fault_line"])
    )[:tail_count]
    return {
        "horizon_steps": int(horizon),
        "status": "completed",
        "uqg": interval["estimate"],
        "uqg_ci_lower": interval["ci_lower"],
        "uqg_ci_upper": interval["ci_upper"],
        "mean_probe_skill": float(np.mean([row["probe_skill"] for row in line_rows])),
        "n_fault_lines": len(line_rows),
        "tail_fault_lines": top_lines,
        "fault_line_rows": line_rows,
    }


def _decision(
    horizon_results: list[dict[str, Any]],
    horizons: Sequence[int],
    *,
    tolerance: float,
    minimum_probe_skill: float,
) -> tuple[str, str, float, float]:
    completed = [row for row in horizon_results if row["status"] == "completed"]
    informative = len(completed) == len(tuple(horizons)) and all(
        float(row["mean_probe_skill"]) >= float(minimum_probe_skill) for row in completed
    )
    if not informative:
        return "INSUFFICIENT_INFORMATION", "NO_PARTITION_DECISION", float("nan"), float("nan")
    point = max(float(row["uqg"]) for row in completed)
    lower = max(float(row["uqg_ci_lower"]) for row in completed)
    upper = max(float(row["uqg_ci_upper"]) for row in completed)
    if upper <= float(tolerance):
        return "EMPIRICALLY_SUFFICIENT", "ACCEPT_FOR_OBSERVED_ENVELOPE", point, upper
    if lower > float(tolerance):
        return "PARTITION_REJECTED", "DISCARDED_COORDINATES_REQUIRED", point, upper
    return (
        "INSUFFICIENT_INFORMATION",
        "EQUIVALENCE_INTERVAL_CROSSES_TOLERANCE",
        point,
        upper,
    )


def global_quotient_gap(
    paths: Sequence[dict[str, Any]],
    mass: np.ndarray,
    partition: Partition,
    *,
    horizons: Sequence[int] = (100, 500, 1000),
    folds: int = 5,
    delays: int = 1,
    ridge: float = 1e-6,
    rank_cap: int = 12,
    max_rows_per_fault_line: int = 64,
    tail_fraction: float = 0.10,
    tolerance: float = 0.02,
    bootstrap_draws: int = 500,
    minimum_probe_skill: float = 0.0,
    seed: int = 324,
) -> dict[str, Any]:
    """Cross-fitted Unified Quotient Gap (UQG) over all observed regimes.

    UQG is the worst-horizon fault-line tail risk of two non-compensating
    losses: discarded output energy and the conditional predictive gain from
    adding discarded state/input coordinates.  Source labels are deliberately
    absent.  This answers whether one quotient represents the union of observed
    excitations; it does not identify a causal effect of disturbance level.
    """

    mass = np.asarray(mass, dtype=float).reshape(-1)
    partition = validate_partition(partition, len(mass))
    basis = mass_orthonormal_quotient_coordinates(mass, partition)
    if basis.discarded.shape[1] == 0:
        return {
            "status": "EMPIRICALLY_SUFFICIENT",
            "decision": "FULL_RESOLUTION_REFERENCE",
            "partition": [list(group) for group in partition],
            "spatial_coordinates": len(partition),
            "uqg": 0.0,
            "uqg_ci_upper": 0.0,
            "tolerance": float(tolerance),
            "horizons": [],
            "source_labels_used": False,
            "fault_line_equal_weight": True,
        }

    assignment = _fold_assignment(paths, folds, seed)
    horizon_results = []
    for horizon in tuple(int(value) for value in horizons):
        line_rows: list[dict[str, Any]] = []
        for fold in range(int(folds)):
            train = [
                path
                for path in paths
                if assignment[str(path["fault_line"])] != fold
            ]
            evaluation = [
                path
                for path in paths
                if assignment[str(path["fault_line"])] == fold
            ]
            try:
                train_rows = _stack_equal_fault_line_rows(
                    train,
                    mass,
                    basis,
                    horizon=horizon,
                    delays=delays,
                    max_rows_per_fault_line=max_rows_per_fault_line,
                )
                eval_rows = _stack_equal_fault_line_rows(
                    evaluation,
                    mass,
                    basis,
                    horizon=horizon,
                    delays=delays,
                    max_rows_per_fault_line=max_rows_per_fault_line,
                )
            except ValueError:
                continue
            base_prediction = _ridge_predict(
                train_rows["base"],
                train_rows["target"],
                eval_rows["base"],
                ridge=ridge,
                rank_cap=rank_cap,
            )
            full_prediction = _ridge_predict(
                train_rows["full"],
                train_rows["target"],
                eval_rows["full"],
                ridge=ridge,
                rank_cap=rank_cap,
            )
            line_rows.extend(_line_gap_rows(eval_rows, base_prediction, full_prediction))
        horizon_results.append(
            _summarize_horizon(
                line_rows,
                horizon=horizon,
                folds=folds,
                tail_fraction=tail_fraction,
                bootstrap_draws=bootstrap_draws,
                seed=seed,
            )
        )

    status, decision, point, upper = _decision(
        horizon_results,
        horizons,
        tolerance=tolerance,
        minimum_probe_skill=minimum_probe_skill,
    )
    return {
        "status": status,
        "decision": decision,
        "partition": [list(group) for group in partition],
        "spatial_coordinates": len(partition),
        "discarded_coordinates": basis.discarded.shape[1],
        "uqg": point,
        "uqg_ci_upper": upper,
        "tolerance": float(tolerance),
        "tail_fraction": float(tail_fraction),
        "horizons": horizon_results,
        "source_labels_used": False,
        "fault_line_equal_weight": True,
        "scope": "union of observed Q-A train excitations",
        "nonclaim": "unselected physical modes need not be absent outside the observed envelope",
    }


def holdout_global_quotient_gap(
    train_paths: Sequence[dict[str, Any]],
    evaluation_paths: Sequence[dict[str, Any]],
    mass: np.ndarray,
    partition: Partition,
    *,
    horizons: Sequence[int] = (100, 500, 1000),
    delays: int = 1,
    ridge: float = 1e-6,
    rank_cap: int = 12,
    max_rows_per_fault_line: int = 64,
    tail_fraction: float = 0.10,
    tolerance: float = 0.02,
    bootstrap_draws: int = 500,
    minimum_probe_skill: float = 0.0,
    seed: int = 324,
) -> dict[str, Any]:
    """Confirm a frozen quotient on a fault-line-disjoint holdout population."""

    mass = np.asarray(mass, dtype=float).reshape(-1)
    partition = validate_partition(partition, len(mass))
    train_lines = {str(path["fault_line"]) for path in train_paths}
    evaluation_lines = {str(path["fault_line"]) for path in evaluation_paths}
    if train_lines & evaluation_lines:
        raise ValueError("train and holdout fault lines must be disjoint")
    basis = mass_orthonormal_quotient_coordinates(mass, partition)
    horizon_results = []
    for horizon in tuple(int(value) for value in horizons):
        try:
            train_rows = _stack_equal_fault_line_rows(
                train_paths,
                mass,
                basis,
                horizon=horizon,
                delays=delays,
                max_rows_per_fault_line=max_rows_per_fault_line,
            )
            evaluation_rows = _stack_equal_fault_line_rows(
                evaluation_paths,
                mass,
                basis,
                horizon=horizon,
                delays=delays,
                max_rows_per_fault_line=max_rows_per_fault_line,
            )
        except ValueError:
            horizon_results.append(
                {"horizon_steps": horizon, "status": "INSUFFICIENT_INFORMATION"}
            )
            continue
        base_prediction = _ridge_predict(
            train_rows["base"],
            train_rows["target"],
            evaluation_rows["base"],
            ridge=ridge,
            rank_cap=rank_cap,
        )
        full_prediction = _ridge_predict(
            train_rows["full"],
            train_rows["target"],
            evaluation_rows["full"],
            ridge=ridge,
            rank_cap=rank_cap,
        )
        line_rows = _line_gap_rows(evaluation_rows, base_prediction, full_prediction)
        horizon_results.append(
            _summarize_horizon(
                line_rows,
                horizon=horizon,
                folds=2,
                tail_fraction=tail_fraction,
                bootstrap_draws=bootstrap_draws,
                seed=seed,
            )
        )
    status, decision, point, upper = _decision(
        horizon_results,
        horizons,
        tolerance=tolerance,
        minimum_probe_skill=minimum_probe_skill,
    )
    return {
        "status": status,
        "decision": decision,
        "partition": [list(group) for group in partition],
        "uqg": point,
        "uqg_ci_upper": upper,
        "tolerance": float(tolerance),
        "horizons": horizon_results,
        "source_labels_used": False,
        "fault_line_equal_weight": True,
        "selection_frozen_before_holdout": True,
    }


def merge_partition(partition: Partition, left: int, right: int) -> Partition:
    partition = tuple(tuple(group) for group in partition)
    if left >= right or left < 0 or right >= len(partition):
        raise ValueError("invalid merge indices")
    merged = tuple(sorted((*partition[left], *partition[right])))
    output = [group for index, group in enumerate(partition) if index not in {left, right}]
    output.append(merged)
    return tuple(sorted(output, key=lambda group: (group[0], len(group), group)))


def unified_quotient_agglomeration(
    paths: Sequence[dict[str, Any]],
    mass: np.ndarray,
    *,
    horizons: Sequence[int] = (100, 500, 1000),
    screen_horizon: int = 100,
    minimum_clusters: int = 2,
    folds: int = 5,
    delays: int = 1,
    ridge: float = 1e-6,
    rank_cap: int = 12,
    max_rows_per_fault_line: int = 64,
    tail_fraction: float = 0.10,
    tolerance: float = 0.02,
    screen_bootstrap_draws: int = 200,
    final_bootstrap_draws: int = 500,
    seed: int = 324,
) -> dict[str, Any]:
    """Agglomerate nodes only when the whole-data UQG certificate passes."""

    mass = np.asarray(mass, dtype=float).reshape(-1)
    current: Partition = tuple((node,) for node in range(len(mass)))
    lineage: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    full_evaluations = 0
    unresolved_stop = False
    while len(current) > int(minimum_clusters):
        screened = []
        for left in range(len(current)):
            for right in range(left + 1, len(current)):
                child = merge_partition(current, left, right)
                screen = global_quotient_gap(
                    paths,
                    mass,
                    child,
                    horizons=(int(screen_horizon),),
                    folds=folds,
                    delays=delays,
                    ridge=ridge,
                    rank_cap=rank_cap,
                    max_rows_per_fault_line=max_rows_per_fault_line,
                    tail_fraction=tail_fraction,
                    tolerance=tolerance,
                    bootstrap_draws=screen_bootstrap_draws,
                    seed=seed + len(current),
                )
                row = {
                    "round": len(mass) - len(current) + 1,
                    "parent": [list(group) for group in current],
                    "child": [list(group) for group in child],
                    "merge": [list(current[left]), list(current[right])],
                    "screen": screen,
                    "screen_pass": screen["status"] == "EMPIRICALLY_SUFFICIENT",
                    "full": None,
                    "selected": False,
                }
                candidates.append(row)
                if row["screen_pass"]:
                    screened.append(row)
        if not screened:
            current_round = [
                row for row in candidates if int(row["round"]) == len(mass) - len(current) + 1
            ]
            unresolved_stop = any(
                row["screen"]["status"] != "PARTITION_REJECTED" for row in current_round
            )
            break
        passed = []
        for row in sorted(
            screened,
            key=lambda value: (
                float(value["screen"]["uqg_ci_upper"]),
                value["child"],
            ),
        ):
            child = tuple(tuple(group) for group in row["child"])
            full = global_quotient_gap(
                paths,
                mass,
                child,
                horizons=tuple(int(value) for value in horizons),
                folds=folds,
                delays=delays,
                ridge=ridge,
                rank_cap=rank_cap,
                max_rows_per_fault_line=max_rows_per_fault_line,
                tail_fraction=tail_fraction,
                tolerance=tolerance,
                bootstrap_draws=final_bootstrap_draws,
                seed=seed + 1000 + len(current),
            )
            row["full"] = full
            full_evaluations += 1
            if full["status"] == "EMPIRICALLY_SUFFICIENT":
                passed.append(row)
        if not passed:
            unresolved_stop = any(
                row["full"] is not None
                and row["full"]["status"] != "PARTITION_REJECTED"
                for row in screened
            )
            break
        selected = min(
            passed,
            key=lambda value: (
                float(value["full"]["uqg_ci_upper"]),
                float(value["full"]["uqg"]),
                value["child"],
            ),
        )
        selected["selected"] = True
        lineage.append(
            {
                "partition": selected["child"],
                "merge": selected["merge"],
                "uqg": selected["full"]["uqg"],
                "uqg_ci_upper": selected["full"]["uqg_ci_upper"],
            }
        )
        current = tuple(tuple(group) for group in selected["child"])

    status = (
        "INSUFFICIENT_INFORMATION"
        if unresolved_stop
        else "FULL_RESOLUTION_REQUIRED"
        if len(current) == len(mass)
        else "HARD_CLUSTER_ADMISSIBLE"
    )
    return {
        "status": status,
        "selected_partition": [list(group) for group in current],
        "n_clusters": len(current),
        "selected_path": lineage,
        "candidate_evaluations": candidates,
        "screen_evaluations": len(candidates),
        "full_evaluations": full_evaluations,
        "selection_rule": "minimum UQG upper bound among certificate-passing merges",
        "source_overlap_required": False,
        "all_QA_fault_lines_jointly_used": True,
        "unresolved_candidate_prevented_minimality_claim": unresolved_stop,
    }
