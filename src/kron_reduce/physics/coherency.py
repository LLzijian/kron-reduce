from __future__ import annotations

import itertools
from typing import Any, TypeAlias

import numpy as np

from .aggregation import cluster_aggregation_matrices, validate_partition
from .coordinates import coi_project, coi_project_power

Pair: TypeAlias = tuple[int, int]
Partition: TypeAlias = tuple[tuple[int, ...], ...]


def node_pairs(n_nodes: int) -> tuple[Pair, ...]:
    if n_nodes < 2:
        raise ValueError("at least two nodes are required")
    return tuple(itertools.combinations(range(n_nodes), 2))


def disjoint_pair_matchings(n_nodes: int, n_pairs: int) -> tuple[tuple[Pair, ...], ...]:
    """Enumerate deterministic matchings containing exactly ``n_pairs`` edges."""
    if n_pairs < 0 or 2 * n_pairs > n_nodes:
        raise ValueError("n_pairs must define a feasible node matching")
    if n_pairs == 0:
        return ((),)
    candidates = []
    for chosen in itertools.combinations(node_pairs(n_nodes), n_pairs):
        if len({node for pair in chosen for node in pair}) == 2 * n_pairs:
            candidates.append(tuple(sorted(chosen)))
    return tuple(sorted(set(candidates)))


def partition_from_pairs(n_nodes: int, pairs: tuple[Pair, ...]) -> Partition:
    canonical_pairs = tuple(tuple(sorted((int(left), int(right)))) for left, right in pairs)
    used = {node for pair in canonical_pairs for node in pair}
    if len(used) != 2 * len(canonical_pairs):
        raise ValueError("pair merges must be disjoint")
    partition = canonical_pairs + tuple((node,) for node in range(n_nodes) if node not in used)
    return validate_partition(partition, n_nodes)


def _matching_indicator(n_nodes: int, n_pairs: int) -> tuple[tuple[tuple[Pair, ...], ...], np.ndarray]:
    pairs = node_pairs(n_nodes)
    pair_index = {pair: index for index, pair in enumerate(pairs)}
    matchings = disjoint_pair_matchings(n_nodes, n_pairs)
    indicator = np.zeros((len(matchings), len(pairs)), dtype=np.float64)
    for row, matching in enumerate(matchings):
        indicator[row, [pair_index[pair] for pair in matching]] = 1.0
    return matchings, indicator


def optimal_pair_matching(pair_cost: np.ndarray, n_nodes: int, n_pairs: int) -> tuple[Pair, ...]:
    """Return the minimum-cost disjoint matching, breaking ties lexicographically."""
    pair_cost = np.asarray(pair_cost, dtype=np.float64).reshape(-1)
    expected = len(node_pairs(n_nodes))
    if pair_cost.size != expected or not np.all(np.isfinite(pair_cost)):
        raise ValueError("pair_cost must be finite and aligned with the complete node graph")
    matchings, indicator = _matching_indicator(n_nodes, n_pairs)
    return matchings[int(np.argmin(indicator @ pair_cost))]


def synchronizing_laplacian(
    incidence: np.ndarray,
    coupling: np.ndarray,
    equilibrium_delta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearized swing stiffness ``L_s = B diag(K cos(eta_eq)) B.T``."""
    incidence = np.asarray(incidence, dtype=np.float64)
    coupling = np.asarray(coupling, dtype=np.float64).reshape(-1)
    equilibrium_delta = np.asarray(equilibrium_delta, dtype=np.float64).reshape(-1)
    if incidence.ndim != 2 or incidence.shape[0] != equilibrium_delta.size:
        raise ValueError("incidence and equilibrium angle shapes are inconsistent")
    if incidence.shape[1] != coupling.size or np.any(coupling < 0.0):
        raise ValueError("coupling must be nonnegative and edge aligned")
    stiffness = coupling * np.cos(incidence.T @ equilibrium_delta)
    if np.any(stiffness < -1e-12):
        raise ValueError("the fitted operating point has negative synchronizing stiffness")
    stiffness = np.maximum(stiffness, 0.0)
    laplacian = incidence @ np.diag(stiffness) @ incidence.T
    return 0.5 * (laplacian + laplacian.T), stiffness


def generalized_laplacian_spectrum(
    laplacian: np.ndarray, mass: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Solve ``L phi = lambda M phi`` with mass-orthonormal eigenvectors."""
    laplacian = np.asarray(laplacian, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64).reshape(-1)
    if laplacian.shape != (mass.size, mass.size) or np.any(mass <= 0.0):
        raise ValueError("a square Laplacian and positive aligned mass are required")
    inverse_sqrt = 1.0 / np.sqrt(mass)
    normalized = inverse_sqrt[:, None] * laplacian * inverse_sqrt[None, :]
    eigenvalues, normalized_vectors = np.linalg.eigh(0.5 * (normalized + normalized.T))
    eigenvalues[np.abs(eigenvalues) < 1e-12] = 0.0
    eigenvectors = inverse_sqrt[:, None] * normalized_vectors
    return eigenvalues, eigenvectors


def almost_equitable_defect(
    laplacian: np.ndarray, mass: np.ndarray, partition: Partition
) -> float:
    """Relative failure of the cluster-consensus subspace to be invariant.

    For the linearized swing operator ``A = M^-1 L``, an exact weighted
    equitable partition satisfies ``(I-P) A P = 0``.  The returned Frobenius
    ratio is zero exactly in that case.
    """
    laplacian = np.asarray(laplacian, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64).reshape(-1)
    canonical = validate_partition(partition, mass.size)
    _, _, projection = cluster_aggregation_matrices(mass, canonical)
    operator = laplacian / mass[:, None]
    residual = (np.eye(mass.size) - projection) @ operator @ projection
    return float(np.linalg.norm(residual, ord="fro") / max(np.linalg.norm(operator, ord="fro"), 1e-15))


def optimal_aep_matching(
    laplacian: np.ndarray, mass: np.ndarray, n_pairs: int
) -> tuple[tuple[Pair, ...], list[dict[str, Any]]]:
    """Select the matching with the smallest weighted AEP invariance defect."""
    mass = np.asarray(mass, dtype=np.float64).reshape(-1)
    rows = []
    for matching in disjoint_pair_matchings(mass.size, n_pairs):
        partition = partition_from_pairs(mass.size, matching)
        rows.append(
            {
                "pairs": matching,
                "partition": partition,
                "aep_defect": almost_equitable_defect(laplacian, mass, partition),
            }
        )
    selected = min(rows, key=lambda row: (row["aep_defect"], row["pairs"]))
    return selected["pairs"], sorted(rows, key=lambda row: (row["aep_defect"], row["pairs"]))


def normalized_dynamic_pair_costs(
    delta: np.ndarray, omega: np.ndarray, mass: np.ndarray
) -> np.ndarray:
    """Pairwise electromechanical dissimilarity invariant to common-angle shifts."""
    delta = np.asarray(delta, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64).reshape(-1)
    if delta.ndim != 2 or omega.shape != delta.shape or delta.shape[1] != mass.size:
        raise ValueError("delta, omega and mass shapes are inconsistent")
    delta = coi_project(delta, mass)
    omega = coi_project(omega, mass)
    delta = delta - delta[[0]]
    delta_scale = max(float(np.sqrt(np.mean(delta**2))), 1e-12)
    omega_scale = max(float(np.sqrt(np.mean(omega**2))), 1e-12)
    return np.asarray(
        [
            0.5
            * (
                float(np.sqrt(np.mean((delta[:, left] - delta[:, right]) ** 2))) / delta_scale
                + float(np.sqrt(np.mean((omega[:, left] - omega[:, right]) ** 2)))
                / omega_scale
            )
            for left, right in node_pairs(mass.size)
        ],
        dtype=np.float64,
    )


def normalized_forcing_pair_costs(
    pm: np.ndarray, pm_equilibrium: np.ndarray, mass: np.ndarray
) -> np.ndarray:
    """Pairwise mismatch of accelerating-power inputs in the COI quotient."""
    pm = np.asarray(pm, dtype=np.float64)
    equilibrium = np.asarray(pm_equilibrium, dtype=np.float64).reshape(-1)
    mass = np.asarray(mass, dtype=np.float64).reshape(-1)
    if pm.ndim != 2 or pm.shape[1] != mass.size or equilibrium.size != mass.size:
        raise ValueError("Pm, equilibrium and mass shapes are inconsistent")
    accelerating_power = coi_project_power(pm - equilibrium, mass) / mass
    scale = max(float(np.sqrt(np.mean(accelerating_power**2))), 1e-12)
    return np.asarray(
        [
            float(
                np.sqrt(np.mean((accelerating_power[:, left] - accelerating_power[:, right]) ** 2))
                / scale
            )
            for left, right in node_pairs(mass.size)
        ],
        dtype=np.float64,
    )


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("invalid binomial counts")
    probability = successes / trials
    denominator = 1.0 + z * z / trials
    center = (probability + z * z / (2.0 * trials)) / denominator
    radius = z * np.sqrt(
        probability * (1.0 - probability) / trials + z * z / (4.0 * trials * trials)
    ) / denominator
    return float(max(0.0, center - radius)), float(min(1.0, center + radius))


def bootstrap_matching_stability(
    case_pair_costs: np.ndarray,
    *,
    n_nodes: int,
    n_pairs: int,
    draws: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Case-bootstrap selection frequency for each edge in an optimal matching."""
    costs = np.asarray(case_pair_costs, dtype=np.float64)
    if costs.ndim != 2 or costs.shape[1] != len(node_pairs(n_nodes)):
        raise ValueError("case_pair_costs must have one complete-graph row per case")
    if costs.shape[0] == 0 or draws <= 0 or not np.all(np.isfinite(costs)):
        raise ValueError("finite nonempty costs and positive bootstrap draws are required")
    matchings, indicator = _matching_indicator(n_nodes, n_pairs)
    pair_index = {pair: index for index, pair in enumerate(node_pairs(n_nodes))}
    counts = np.zeros(len(pair_index), dtype=int)
    rng = np.random.default_rng(seed)
    for _ in range(draws):
        sampled = rng.integers(0, costs.shape[0], size=costs.shape[0])
        selected = matchings[int(np.argmin(indicator @ np.median(costs[sampled], axis=0)))]
        for pair in selected:
            counts[pair_index[pair]] += 1
    rows = []
    for pair, index in pair_index.items():
        lower, upper = wilson_interval(int(counts[index]), draws)
        rows.append(
            {
                "pair": pair,
                "selection_count": int(counts[index]),
                "selection_probability": float(counts[index] / draws),
                "wilson95_lower": lower,
                "wilson95_upper": upper,
            }
        )
    return sorted(rows, key=lambda row: (-row["selection_probability"], row["pair"]))


def robust_consensus_partition(
    *,
    dynamic_costs_by_source: dict[str, np.ndarray],
    forcing_costs_by_source: dict[str, np.ndarray],
    graph_matching: tuple[Pair, ...],
    n_nodes: int,
    candidate_pairs: int,
    bootstrap_draws: int,
    seed: int,
    majority_lower_bound: float = 0.5,
) -> tuple[Partition, dict[str, Any]]:
    """Conservative multi-view coarsening with an adaptive cluster count.

    A pair is merged only when it belongs to the optimal response and forcing
    matchings for every training source, has a bootstrap Wilson lower bound
    above ``majority_lower_bound`` in every view/source, and belongs to the
    graph-theoretic AEP matching.  Therefore disagreement reduces the number
    of merges instead of forcing an unsupported fixed cluster count.
    """
    sources = sorted(dynamic_costs_by_source)
    if not sources or sources != sorted(forcing_costs_by_source):
        raise ValueError("dynamic and forcing costs must contain the same sources")
    graph_pairs = {tuple(pair) for pair in graph_matching}
    diagnostics: dict[str, Any] = {"sources": {}, "graph_matching": tuple(graph_matching)}
    accepted = set(graph_pairs)
    pairs = node_pairs(n_nodes)
    for source_index, source in enumerate(sources):
        dynamic = np.asarray(dynamic_costs_by_source[source], dtype=np.float64)
        forcing = np.asarray(forcing_costs_by_source[source], dtype=np.float64)
        if dynamic.shape != forcing.shape or dynamic.ndim != 2 or dynamic.shape[1] != len(pairs):
            raise ValueError(f"invalid case-pair cost matrices for source {source}")
        dynamic_point = optimal_pair_matching(np.median(dynamic, axis=0), n_nodes, candidate_pairs)
        forcing_point = optimal_pair_matching(np.median(forcing, axis=0), n_nodes, candidate_pairs)
        dynamic_bootstrap = bootstrap_matching_stability(
            dynamic,
            n_nodes=n_nodes,
            n_pairs=candidate_pairs,
            draws=bootstrap_draws,
            seed=seed + 2 * source_index,
        )
        forcing_bootstrap = bootstrap_matching_stability(
            forcing,
            n_nodes=n_nodes,
            n_pairs=candidate_pairs,
            draws=bootstrap_draws,
            seed=seed + 2 * source_index + 1,
        )
        dynamic_stable = {
            row["pair"] for row in dynamic_bootstrap if row["wilson95_lower"] > majority_lower_bound
        }
        forcing_stable = {
            row["pair"] for row in forcing_bootstrap if row["wilson95_lower"] > majority_lower_bound
        }
        source_supported = (
            set(dynamic_point) & set(forcing_point) & dynamic_stable & forcing_stable
        )
        accepted &= source_supported
        diagnostics["sources"][source] = {
            "n_cases": int(dynamic.shape[0]),
            "dynamic_point_matching": dynamic_point,
            "forcing_point_matching": forcing_point,
            "source_supported_pairs": tuple(sorted(source_supported)),
            "dynamic_bootstrap": dynamic_bootstrap,
            "forcing_bootstrap": forcing_bootstrap,
        }
    selected = tuple(sorted(accepted))
    partition = partition_from_pairs(n_nodes, selected)
    diagnostics.update(
        {
            "selected_pairs": selected,
            "n_selected_pairs": len(selected),
            "n_clusters": len(partition),
            "majority_lower_bound": majority_lower_bound,
            "selection_rule": (
                "intersection(graph AEP matching, all-source response matching, "
                "all-source accelerating-power matching, Wilson95 lower > 0.5)"
            ),
        }
    )
    return partition, diagnostics
