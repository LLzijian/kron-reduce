from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from .aggregation import (
    aggregate_event_arrays,
    project_cluster_consensus,
    validate_partition,
)
from .coherency import (
    node_pairs,
    normalized_dynamic_pair_costs,
    normalized_forcing_pair_costs,
)
from .coordinates import coi_project, coi_project_power
from .graph import complete_incidence

Partition = tuple[tuple[int, ...], ...]


def _source(sample: dict[str, Any]) -> str:
    return str(sample["training_context"]["metadata"].get("source_dataset", "all"))


def _pair_matrix(values: np.ndarray, n_nodes: int) -> np.ndarray:
    matrix = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for value, (left, right) in zip(values, node_pairs(n_nodes), strict=True):
        matrix[left, right] = matrix[right, left] = float(value)
    return matrix


def discover_nested_partitions(
    train_samples: list[dict[str, Any]],
    *,
    cluster_counts: tuple[int, ...] = (5, 6, 7, 8, 9),
    maximum_steps: int = 1001,
) -> tuple[dict[int, Partition], dict[str, Any]]:
    """Discover a train-only, source-balanced nested spatial hierarchy.

    Pair costs combine relative electromechanical response and accelerating-
    power mismatch.  Agglomeration is deterministic average linkage, so moving
    from k to k+1 retains one additional internal spatial degree of freedom.
    """

    if not train_samples:
        raise ValueError("partition discovery requires nonempty train samples")
    n_nodes = int(np.asarray(train_samples[0]["model_inputs"]["static"]["mass"]).size)
    counts = tuple(sorted({int(value) for value in cluster_counts}))
    if not counts or counts[0] < 2 or counts[-1] >= n_nodes:
        raise ValueError("cluster counts must lie between two and n_nodes-1")
    by_source: dict[str, list[np.ndarray]] = {}
    component_rows: list[dict[str, Any]] = []
    for sample in train_samples:
        target = sample["training_context"]["targets"]
        static = sample["model_inputs"]["static"]
        steps = min(len(target["t"]), maximum_steps)
        dynamic = normalized_dynamic_pair_costs(
            np.asarray(target["delta"][:steps]),
            np.asarray(target["omega"][:steps]),
            np.asarray(static["mass"]),
        )
        forcing = normalized_forcing_pair_costs(
            np.asarray(sample["training_context"]["Pm_observed"][:steps]),
            np.asarray(static["Pm_equilibrium"]),
            np.asarray(static["mass"]),
        )
        combined = 0.5 * (dynamic + forcing)
        source = _source(sample)
        by_source.setdefault(source, []).append(combined)
        component_rows.append(
            {
                "event_id": sample["event_id"],
                "source_dataset": source,
                "median_dynamic_cost": float(np.median(dynamic)),
                "median_forcing_cost": float(np.median(forcing)),
            }
        )
    source_medians = {
        source: np.median(np.stack(rows), axis=0) for source, rows in by_source.items()
    }
    balanced = np.mean(np.stack(list(source_medians.values())), axis=0)
    distance = _pair_matrix(balanced, n_nodes)

    clusters: list[tuple[int, ...]] = [(node,) for node in range(n_nodes)]
    snapshots: dict[int, Partition] = {}
    merges: list[dict[str, Any]] = []
    while len(clusters) > min(counts):
        candidates = []
        for left in range(len(clusters)):
            for right in range(left + 1, len(clusters)):
                cross = distance[np.ix_(clusters[left], clusters[right])]
                candidates.append(
                    (float(np.mean(cross)), clusters[left], clusters[right], left, right)
                )
        score, left_group, right_group, left, right = min(
            candidates, key=lambda row: (row[0], row[1], row[2])
        )
        merged = tuple(sorted((*left_group, *right_group)))
        clusters = [
            group for index, group in enumerate(clusters) if index not in {left, right}
        ]
        clusters.append(merged)
        clusters.sort(key=lambda group: (group[0], len(group), group))
        merges.append(
            {
                "from": [list(left_group), list(right_group)],
                "to": list(merged),
                "source_balanced_cost": score,
                "n_clusters_after_merge": len(clusters),
            }
        )
        if len(clusters) in counts:
            snapshots[len(clusters)] = validate_partition(tuple(clusters), n_nodes)
    missing = set(counts) - set(snapshots)
    if missing:
        raise RuntimeError(f"hierarchy did not produce requested counts: {sorted(missing)}")
    return {count: snapshots[count] for count in counts}, {
        "method": "train_only_source_balanced_response_plus_forcing_average_linkage",
        "sources": {source: len(rows) for source, rows in by_source.items()},
        "source_median_pair_costs": {
            source: values.tolist() for source, values in source_medians.items()
        },
        "source_balanced_pair_costs": balanced.tolist(),
        "merges": merges,
        "component_rows": component_rows,
        "labels_used": False,
        "validation_or_test_used": False,
    }


def aggregate_model_sample(sample: dict[str, Any], partition: Partition, name: str) -> dict[str, Any]:
    """Create a reduced scientific view from a generator-10 model sample."""

    result = deepcopy(sample)
    target = sample["training_context"]["targets"]
    static = sample["model_inputs"]["static"]
    mass = np.asarray(static["mass"], dtype=np.float64)
    partition = validate_partition(partition, len(mass))
    arrays = aggregate_event_arrays(
        np.asarray(target["delta"]),
        np.asarray(target["omega"]),
        np.asarray(sample["training_context"]["Pm_observed"]),
        np.asarray(static["equilibrium_delta"]),
        mass,
        partition,
    )
    equilibrium_pm = np.asarray(
        [np.sum(np.asarray(static["Pm_equilibrium"])[list(group)]) for group in partition]
    )
    pm_schedule = np.asarray(
        [
            np.sum(np.asarray(sample["model_inputs"]["exogenous"]["Pm"])[..., list(group)], axis=-1)
            for group in partition
        ]
    ).T
    incidence, edges = complete_incidence(len(partition))
    reduced_mass = arrays["mass"]
    arrays["delta"] = coi_project(arrays["delta"], reduced_mass)
    arrays["omega"] = coi_project(arrays["omega"], reduced_mass)
    arrays["Pm"] = coi_project_power(arrays["Pm"], reduced_mass)
    equilibrium_pm = coi_project_power(equilibrium_pm, reduced_mass)
    pm_schedule = coi_project_power(pm_schedule, reduced_mass)
    reduced_static = {
        "delta0": arrays["delta"][[0]].copy(),
        "omega0": arrays["omega"][[0]].copy(),
        "Pm_equilibrium": equilibrium_pm.copy(),
        "Pm_clear": pm_schedule[[0]].copy(),
        "Pm0": pm_schedule[[0]].copy(),
        "equilibrium_delta": coi_project(arrays["equilibrium_delta"], reduced_mass),
        "mass": reduced_mass.copy(),
        "incidence": incidence,
    }
    result["model_inputs"]["static"] = reduced_static
    result["model_inputs"]["exogenous"]["Pm"] = pm_schedule
    result["training_context"]["targets"]["delta"] = arrays["delta"]
    result["training_context"]["targets"]["omega"] = arrays["omega"]
    result["training_context"]["Pm_observed"] = arrays["Pm"]
    result["training_context"]["Pm_equilibrium"] = equilibrium_pm.copy()
    result["training_context"]["representation"] = name
    result["training_context"]["edges"] = edges
    result["training_context"]["aggregation_partition"] = partition
    return result


def spatial_projection_audit(
    samples: list[dict[str, Any]], partition: Partition, *, maximum_steps: int = 1001
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        target = sample["training_context"]["targets"]
        mass = np.asarray(sample["model_inputs"]["static"]["mass"], dtype=np.float64)
        steps = min(len(target["t"]), maximum_steps)
        for signal, values in (
            ("delta", np.asarray(target["delta"][:steps]) - np.asarray(target["delta"][[0]])),
            ("omega", np.asarray(target["omega"][:steps])),
        ):
            low = project_cluster_consensus(values, mass, partition)
            high = values - low
            total = float(np.sum(values**2 * mass[None, :]))
            discarded = float(np.sum(high**2 * mass[None, :]))
            rows.append(
                {
                    "event_id": sample["event_id"],
                    "source_dataset": _source(sample),
                    "signal": signal,
                    "n_clusters": len(partition),
                    "discarded_mass_energy_fraction": discarded / max(total, 1e-15),
                    "retained_mass_energy_fraction": 1.0 - discarded / max(total, 1e-15),
                }
            )
    return rows


def reconstruct_model_state(
    values: np.ndarray, mass: np.ndarray, partition: Partition | None
) -> np.ndarray:
    if partition is None:
        return np.asarray(values, dtype=np.float64)
    from .aggregation import reconstruct_cluster_state

    return reconstruct_cluster_state(values, mass, partition)
