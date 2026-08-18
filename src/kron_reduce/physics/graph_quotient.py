from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from kron_reduce.evaluation.selection_validity import equal_fault_line_paired_inference

from .aggregation import validate_partition

Partition = tuple[tuple[int, ...], ...]


def partition_membership(partition: Partition, n_nodes: int) -> np.ndarray:
    """Return the binary node-to-cluster matrix S_C."""

    canonical = validate_partition(partition, n_nodes)
    membership = np.zeros((n_nodes, len(canonical)), dtype=float)
    for cluster, members in enumerate(canonical):
        membership[np.asarray(members, dtype=int), cluster] = 1.0
    return membership


def mass_orthogonal_projectors(
    mass: np.ndarray, partition: Partition
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct S_C, P_C and Q_C for the mass inner product."""

    mass = np.asarray(mass, dtype=float).reshape(-1)
    if not np.all(np.isfinite(mass)) or np.any(mass <= 0.0):
        raise ValueError("mass must be finite and strictly positive")
    membership = partition_membership(partition, len(mass))
    metric_membership = mass[:, None] * membership
    gram = membership.T @ metric_membership
    projector = membership @ np.linalg.solve(gram, metric_membership.T)
    complement = np.eye(len(mass)) - projector
    return membership, projector, complement


def projection_certificate(
    mass: np.ndarray, partition: Partition, *, tolerance: float = 1e-12
) -> dict[str, Any]:
    mass = np.asarray(mass, dtype=float).reshape(-1)
    membership, projector, complement = mass_orthogonal_projectors(mass, partition)
    metric = np.diag(mass)
    cluster_mass = membership.T @ mass
    return {
        "status": "PASS"
        if max(
            np.max(np.abs(projector @ projector - projector)),
            np.max(np.abs(projector.T @ metric - metric @ projector)),
            np.max(np.abs(projector @ complement)),
        )
        <= tolerance
        else "FAIL",
        "idempotence_error": float(np.max(np.abs(projector @ projector - projector))),
        "mass_self_adjoint_error": float(np.max(np.abs(projector.T @ metric - metric @ projector))),
        "orthogonality_error": float(np.max(np.abs(projector @ complement))),
        "mass_conservation_error": float(abs(np.sum(cluster_mass) - np.sum(mass))),
        "cluster_mass": cluster_mass.tolist(),
        "full_resolution_exact": bool(
            len(partition) == len(mass) and np.array_equal(projector, np.eye(len(mass)))
        ),
    }


def _mass_energy(values: np.ndarray, mass: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    mass = np.asarray(mass, dtype=float).reshape(-1)
    return float(np.sum(values**2 * mass[None, :]))


def intra_cluster_energy(
    paths: Sequence[dict[str, Any]], mass: np.ndarray, partition: Partition
) -> dict[str, float]:
    _, _, complement = mass_orthogonal_projectors(mass, partition)
    totals = defaultdict(float)
    discarded = defaultdict(float)
    for path in paths:
        for signal in ("delta", "omega"):
            values = np.asarray(path[signal], dtype=float)
            if signal == "delta":
                values = values - values[[0]]
            hidden = values @ complement.T
            totals[signal] += _mass_energy(values, mass)
            discarded[signal] += _mass_energy(hidden, mass)
    return {
        signal: discarded[signal] / max(totals[signal], np.finfo(float).eps)
        for signal in ("delta", "omega")
    }


def input_heterogeneity(
    paths: Sequence[dict[str, Any]], mass: np.ndarray, partition: Partition
) -> float:
    _, _, complement = mass_orthogonal_projectors(mass, partition)
    mass = np.asarray(mass, dtype=float).reshape(-1)
    total = 0.0
    hidden = 0.0
    for path in paths:
        forcing = np.asarray(path["u"], dtype=float) / mass[None, :]
        total += _mass_energy(forcing, mass)
        hidden += _mass_energy(forcing @ complement.T, mass)
    return hidden / max(total, np.finfo(float).eps)


def modal_loss(active_modes: np.ndarray | None, mass: np.ndarray, partition: Partition) -> float:
    if active_modes is None or np.asarray(active_modes).size == 0:
        return float("nan")
    modes = np.asarray(active_modes, dtype=complex)
    if modes.ndim == 1:
        modes = modes[:, None]
    _, _, complement = mass_orthogonal_projectors(mass, partition)
    mass = np.asarray(mass, dtype=float).reshape(-1)
    numerator = np.sum(np.abs(complement @ modes) ** 2 * mass[:, None])
    denominator = np.sum(np.abs(modes) ** 2 * mass[:, None])
    return float(numerator / max(float(denominator), np.finfo(float).eps))


def _stack_horizon_rows(
    paths: Sequence[dict[str, Any]],
    projector: np.ndarray,
    complement: np.ndarray,
    *,
    horizon: int,
    delays: int,
    max_rows_per_path: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    base_rows: list[np.ndarray] = []
    full_rows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    lines: list[str] = []
    for path in paths:
        delta = np.asarray(path["delta"], dtype=float)
        omega = np.asarray(path["omega"], dtype=float)
        forcing = np.asarray(path["u"], dtype=float)
        retained = np.concatenate((delta @ projector.T, omega @ projector.T), axis=1)
        hidden = np.concatenate((delta @ complement.T, omega @ complement.T), axis=1)
        starts = np.arange(delays - 1, len(delta) - horizon)
        if len(starts) > max_rows_per_path:
            starts = np.unique(
                np.linspace(0, len(starts) - 1, max_rows_per_path).round().astype(int)
            )
            starts = starts + delays - 1
        for stop in starts:
            window = slice(stop - delays + 1, stop + 1)
            base = np.concatenate((retained[window].reshape(-1), forcing[stop]))
            extra = hidden[window].reshape(-1)
            base_rows.append(base)
            full_rows.append(np.concatenate((base, extra)))
            targets.append(retained[stop + horizon])
            lines.append(str(path.get("fault_line", path.get("event_id", "unknown"))))
    if not base_rows:
        raise ValueError("no eligible horizon rows")
    return np.stack(base_rows), np.stack(full_rows), np.stack(targets), lines


def _ridge_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    *,
    ridge: float,
    rank_cap: int,
) -> np.ndarray:
    center = np.mean(train_x, axis=0)
    scale = np.std(train_x, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    left = (train_x - center) / scale
    right = (eval_x - center) / scale
    y_center = np.mean(train_y, axis=0)
    target = train_y - y_center
    u, singular, vt = np.linalg.svd(left, full_matrices=False)
    rank = max(1, min(int(rank_cap), len(singular)))
    u = u[:, :rank]
    singular = singular[:rank]
    vt = vt[:rank]
    coefficients = vt.T @ ((singular / (singular**2 + float(ridge)))[:, None] * (u.T @ target))
    return right @ coefficients + y_center


def conditional_predictive_information_gain(
    train_paths: Sequence[dict[str, Any]],
    evaluation_paths: Sequence[dict[str, Any]],
    mass: np.ndarray,
    partition: Partition,
    *,
    horizons: tuple[int, ...] = (1, 5, 20),
    delays: int = 2,
    ridge: float = 1e-6,
    rank_cap: int = 12,
    max_rows_per_path: int = 128,
    bootstrap_draws: int = 500,
    seed: int = 324,
) -> list[dict[str, Any]]:
    """Estimate CPIG with train-only ridge-Hankel probes and held-out paths."""

    _, projector, complement = mass_orthogonal_projectors(mass, partition)
    output: list[dict[str, Any]] = []
    for horizon in horizons:
        try:
            train_base, train_full, train_target, _ = _stack_horizon_rows(
                train_paths,
                projector,
                complement,
                horizon=int(horizon),
                delays=int(delays),
                max_rows_per_path=int(max_rows_per_path),
            )
            eval_base, eval_full, eval_target, lines = _stack_horizon_rows(
                evaluation_paths,
                projector,
                complement,
                horizon=int(horizon),
                delays=int(delays),
                max_rows_per_path=int(max_rows_per_path),
            )
        except ValueError:
            output.append({"horizon_steps": int(horizon), "status": "INSUFFICIENT_EXCITATION"})
            continue
        variance = float(np.mean((eval_target - np.mean(eval_target, axis=0)) ** 2))
        if variance <= 1e-14:
            output.append({"horizon_steps": int(horizon), "status": "INSUFFICIENT_EXCITATION"})
            continue
        base_prediction = _ridge_predict(
            train_base, train_target, eval_base, ridge=ridge, rank_cap=rank_cap
        )
        full_prediction = _ridge_predict(
            train_full, train_target, eval_full, ridge=ridge, rank_cap=rank_cap
        )
        base_error = np.mean((eval_target - base_prediction) ** 2, axis=1)
        full_error = np.mean((eval_target - full_prediction) ** 2, axis=1)
        inference = equal_fault_line_paired_inference(
            [
                {
                    "fault_line": line,
                    "comparator_minus_reference": float(base - full),
                }
                for line, base, full in zip(lines, base_error, full_error, strict=True)
            ],
            bootstrap_draws=int(bootstrap_draws),
            seed=int(seed + horizon),
        )
        output.append(
            {
                "horizon_steps": int(horizon),
                "status": "completed",
                "base_error": float(np.mean(base_error)),
                "full_error": float(np.mean(full_error)),
                "cpig": float(
                    (np.mean(base_error) - np.mean(full_error))
                    / max(np.mean(base_error), np.finfo(float).eps)
                ),
                "fault_line_inference": inference,
                "probe": "equal-rank-cap ridge-Hankel direct-horizon predictor",
                "rank_cap": int(rank_cap),
            }
        )
    return output


def repeatable_dmd_embedding(
    paths: Sequence[dict[str, Any]], mass: np.ndarray, *, rank: int = 4
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a deterministic mass-normalized DMD node embedding."""

    mass = np.asarray(mass, dtype=float).reshape(-1)
    root_mass = np.sqrt(mass)
    left_blocks = []
    right_blocks = []
    for path in paths:
        state = np.concatenate(
            (
                np.asarray(path["delta"], dtype=float) * root_mass[None, :],
                np.asarray(path["omega"], dtype=float) * root_mass[None, :],
            ),
            axis=1,
        )
        if len(state) >= 3:
            left_blocks.append(state[:-1].T)
            right_blocks.append(state[1:].T)
    if not left_blocks:
        return np.empty((len(mass), 0)), {"status": "INSUFFICIENT_EXCITATION"}
    x = np.concatenate(left_blocks, axis=1)
    y = np.concatenate(right_blocks, axis=1)
    u, singular, vt = np.linalg.svd(x, full_matrices=False)
    effective = int(np.sum(singular > max(singular[0] * 1e-8, 1e-12)))
    use = min(max(1, int(rank)), effective)
    reduced = u[:, :use].T @ y @ vt[:use].T @ np.diag(1.0 / singular[:use])
    eigenvalues, vectors = np.linalg.eig(reduced)
    modes = y @ vt[:use].T @ np.diag(1.0 / singular[:use]) @ vectors
    order = np.argsort(-np.abs(eigenvalues))[:use]
    modes = modes[:, order]
    node_embedding = np.concatenate(
        (np.abs(modes[: len(mass)]), np.abs(modes[len(mass) :])), axis=1
    )
    norms = np.linalg.norm(node_embedding, axis=1, keepdims=True)
    node_embedding = node_embedding / np.where(norms > 0, norms, 1.0)
    return node_embedding.real, {
        "status": "completed",
        "requested_rank": int(rank),
        "effective_rank": effective,
        "used_rank": use,
        "eigenvalues_real": np.real(eigenvalues[order]).tolist(),
        "eigenvalues_imag": np.imag(eigenvalues[order]).tolist(),
    }


def dynamic_distance_matrix(
    paths: Sequence[dict[str, Any]],
    mass: np.ndarray,
    *,
    modal_weight: float = 1.0,
    delta_weight: float = 1.0,
    omega_weight: float = 1.0,
    input_weight: float = 1.0,
    modal_rank: int = 4,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Construct the v4 dynamic similarity distance using train paths only."""

    mass = np.asarray(mass, dtype=float).reshape(-1)
    modal, modal_diagnostics = repeatable_dmd_embedding(paths, mass, rank=modal_rank)
    n_nodes = len(mass)
    distance = np.zeros((n_nodes, n_nodes), dtype=float)
    components = defaultdict(lambda: np.zeros((n_nodes, n_nodes), dtype=float))
    for left in range(n_nodes):
        for right in range(left + 1, n_nodes):
            modal_cost = float(np.sum((modal[left] - modal[right]) ** 2)) if modal.size else 0.0
            delta_cost = np.mean(
                [
                    np.mean(
                        (
                            np.asarray(path["delta"])[:, left] / np.sqrt(mass[left])
                            - np.asarray(path["delta"])[:, right] / np.sqrt(mass[right])
                        )
                        ** 2
                    )
                    for path in paths
                ]
            )
            omega_cost = np.mean(
                [
                    np.mean(
                        (
                            np.asarray(path["omega"])[:, left] / np.sqrt(mass[left])
                            - np.asarray(path["omega"])[:, right] / np.sqrt(mass[right])
                        )
                        ** 2
                    )
                    for path in paths
                ]
            )
            input_cost = np.mean(
                [
                    np.mean(
                        (
                            np.asarray(path["u"])[:, left] / mass[left]
                            - np.asarray(path["u"])[:, right] / mass[right]
                        )
                        ** 2
                    )
                    for path in paths
                ]
            )
            values = {
                "modal": modal_cost,
                "delta": float(delta_cost),
                "omega": float(omega_cost),
                "input": float(input_cost),
            }
            for name, value in values.items():
                components[name][left, right] = components[name][right, left] = value
            total = (
                modal_weight * modal_cost
                + delta_weight * delta_cost
                + omega_weight * omega_cost
                + input_weight * input_cost
            )
            distance[left, right] = distance[right, left] = total
    return distance, {
        "status": "completed" if paths else "INSUFFICIENT_EXCITATION",
        "modal": modal_diagnostics,
        "weights": {
            "modal": float(modal_weight),
            "delta": float(delta_weight),
            "omega": float(omega_weight),
            "input": float(input_weight),
        },
        "components": {name: value.tolist() for name, value in components.items()},
        "validation_or_test_used": False,
    }


def internal_mode_basis(
    paths: Sequence[dict[str, Any]],
    mass: np.ndarray,
    partition: Partition,
    rank: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Find mass-orthonormal modes in Q_C from train trajectories."""

    if rank < 0:
        raise ValueError("rank must be nonnegative")
    mass = np.asarray(mass, dtype=float).reshape(-1)
    membership, _, complement = mass_orthogonal_projectors(mass, partition)
    if rank == 0:
        return np.empty((len(mass), 0)), {"status": "completed", "rank": 0}
    residuals = []
    for path in paths:
        for signal in ("delta", "omega"):
            values = np.asarray(path[signal], dtype=float)
            residuals.append(values @ complement.T)
    residual = np.concatenate(residuals, axis=0)
    weighted = residual * np.sqrt(mass)[None, :]
    _, singular, vt = np.linalg.svd(weighted, full_matrices=False)
    available = int(np.sum(singular > max(singular[0] * 1e-10, 1e-12))) if len(singular) else 0
    use = min(int(rank), available, len(mass) - len(partition))
    if use < rank:
        return np.empty((len(mass), 0)), {
            "status": "INSUFFICIENT_EXCITATION",
            "requested_rank": int(rank),
            "available_rank": available,
        }
    basis = vt[:use].T / np.sqrt(mass)[:, None]
    basis = complement @ basis
    gram = basis.T @ (mass[:, None] * basis)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    basis = basis @ eigenvectors @ np.diag(1.0 / np.sqrt(np.maximum(eigenvalues, 1e-15)))
    return basis, {
        "status": "completed",
        "rank": use,
        "mass_orthogonality_error": float(
            np.max(np.abs(basis.T @ (mass[:, None] * basis) - np.eye(use)))
        ),
        "cluster_orthogonality_error": float(
            np.max(np.abs(basis.T @ (mass[:, None] * membership)))
        ),
        "singular_values": singular[:use].tolist(),
    }


def merge_partition(partition: Partition, left: int, right: int) -> Partition:
    canonical = tuple(tuple(group) for group in partition)
    if left == right or min(left, right) < 0 or max(left, right) >= len(canonical):
        raise ValueError("invalid merge indices")
    merged = tuple(sorted((*canonical[left], *canonical[right])))
    result = [group for index, group in enumerate(canonical) if index not in {left, right}]
    result.append(merged)
    return tuple(sorted(result, key=lambda group: (group[0], len(group), group)))


@dataclass(frozen=True)
class ContractionCandidate:
    partition: Partition
    score: float
    certificate: dict[str, Any]
    lineage: tuple[dict[str, Any], ...]


def beam_graph_contraction(
    distance: np.ndarray,
    certificate_fn: Callable[[Partition], dict[str, Any]],
    *,
    beam_width: int = 5,
    minimum_clusters: int = 2,
) -> dict[str, Any]:
    """Deterministic certificate-gated agglomeration from full resolution."""

    distance = np.asarray(distance, dtype=float)
    if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise ValueError("distance must be square")
    n_nodes = len(distance)
    initial: Partition = tuple((node,) for node in range(n_nodes))
    beam = [ContractionCandidate(initial, 0.0, certificate_fn(initial), ())]
    tree: list[dict[str, Any]] = []
    accepted = list(beam)
    while beam and len(beam[0].partition) > minimum_clusters:
        proposed: list[ContractionCandidate] = []
        seen: set[Partition] = set()
        for parent in beam:
            partition = parent.partition
            for left in range(len(partition)):
                for right in range(left + 1, len(partition)):
                    child = merge_partition(partition, left, right)
                    if child in seen:
                        continue
                    seen.add(child)
                    cost = float(np.mean(distance[np.ix_(partition[left], partition[right])]))
                    certificate = certificate_fn(child)
                    admissible = certificate.get("status") == "HARD_CLUSTER_ADMISSIBLE"
                    row = {
                        "parent": [list(group) for group in partition],
                        "child": [list(group) for group in child],
                        "merge": [list(partition[left]), list(partition[right])],
                        "dynamic_distance": cost,
                        "certificate_status": certificate.get("status"),
                        "admissible": admissible,
                    }
                    tree.append(row)
                    if admissible:
                        proposed.append(
                            ContractionCandidate(
                                child,
                                parent.score + cost,
                                certificate,
                                (*parent.lineage, row),
                            )
                        )
        if not proposed:
            break
        beam = sorted(
            proposed,
            key=lambda item: (item.score, len(item.partition), item.partition),
        )[: int(beam_width)]
        accepted.extend(beam)
    best = min(accepted, key=lambda item: (len(item.partition), item.score, item.partition))
    return {
        "status": "completed",
        "beam_width": int(beam_width),
        "best_partition": [list(group) for group in best.partition],
        "best_score": float(best.score),
        "best_certificate": best.certificate,
        "selected_path": list(best.lineage),
        "tree": tree,
        "evaluated_candidates": len(tree) + 1,
    }
