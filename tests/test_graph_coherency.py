from __future__ import annotations

import numpy as np

from kron_reduce.physics import (
    almost_equitable_defect,
    generalized_laplacian_spectrum,
    node_pairs,
    normalized_dynamic_pair_costs,
    normalized_forcing_pair_costs,
    optimal_aep_matching,
    robust_consensus_partition,
    synchronizing_laplacian,
)


def _twin_laplacian() -> np.ndarray:
    adjacency = np.zeros((5, 5), dtype=float)
    for node, weight in ((2, 1.0), (3, 2.0), (4, 3.0)):
        adjacency[0, node] = adjacency[node, 0] = weight
        adjacency[1, node] = adjacency[node, 1] = weight
    adjacency[2, 3] = adjacency[3, 2] = 0.7
    adjacency[3, 4] = adjacency[4, 3] = 1.3
    return np.diag(np.sum(adjacency, axis=1)) - adjacency


def test_weighted_aep_detects_exact_graph_twins() -> None:
    mass = np.ones(5)
    laplacian = _twin_laplacian()
    matching, rows = optimal_aep_matching(laplacian, mass, n_pairs=1)
    assert matching == ((0, 1),)
    assert rows[0]["aep_defect"] < 1e-14
    assert almost_equitable_defect(laplacian, mass, rows[0]["partition"]) < 1e-14


def test_generalized_laplacian_spectrum_is_mass_orthonormal() -> None:
    mass = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    laplacian = _twin_laplacian()
    eigenvalues, eigenvectors = generalized_laplacian_spectrum(laplacian, mass)
    assert abs(eigenvalues[0]) < 1e-12
    assert np.all(eigenvalues[1:] > 0.0)
    assert np.allclose(eigenvectors.T @ np.diag(mass) @ eigenvectors, np.eye(5), atol=1e-12)


def test_pair_costs_are_coi_shift_invariant() -> None:
    rng = np.random.default_rng(11)
    mass = np.asarray([1.0, 2.0, 3.0])
    delta = rng.normal(size=(20, 3))
    omega = rng.normal(size=(20, 3))
    common = np.linspace(-0.4, 0.6, 20)[:, None]
    reference = normalized_dynamic_pair_costs(delta, omega, mass)
    shifted = normalized_dynamic_pair_costs(delta + common, omega - 2.0 * common, mass)
    assert np.allclose(reference, shifted, atol=1e-12)

    pm_equilibrium = np.asarray([0.1, -0.2, 0.1])
    pm = rng.normal(size=(20, 3))
    forcing_reference = normalized_forcing_pair_costs(pm, pm_equilibrium, mass)
    forcing_shifted = normalized_forcing_pair_costs(
        pm + common * mass[None, :], pm_equilibrium, mass
    )
    assert np.allclose(forcing_reference, forcing_shifted, atol=1e-12)


def test_strict_multisource_consensus_adapts_cluster_count() -> None:
    pairs = node_pairs(4)
    index = {pair: position for position, pair in enumerate(pairs)}
    rng = np.random.default_rng(22)

    def costs(best_pair: tuple[int, int]) -> np.ndarray:
        values = 1.0 + 0.01 * rng.normal(size=(24, len(pairs)))
        values[:, index[best_pair]] = 0.01 + 0.001 * rng.random(24)
        return values

    source_a = costs((0, 1))
    source_b = costs((0, 1))
    partition, diagnostics = robust_consensus_partition(
        dynamic_costs_by_source={"a": source_a, "b": source_b},
        forcing_costs_by_source={"a": source_a, "b": source_b},
        graph_matching=((0, 1),),
        n_nodes=4,
        candidate_pairs=1,
        bootstrap_draws=200,
        seed=3,
    )
    assert partition == ((0, 1), (2,), (3,))
    assert diagnostics["n_clusters"] == 3

    source_disagrees = costs((2, 3))
    identity, diagnostics = robust_consensus_partition(
        dynamic_costs_by_source={"a": source_a, "b": source_disagrees},
        forcing_costs_by_source={"a": source_a, "b": source_disagrees},
        graph_matching=((0, 1),),
        n_nodes=4,
        candidate_pairs=1,
        bootstrap_draws=200,
        seed=4,
    )
    assert identity == ((0,), (1,), (2,), (3,))
    assert diagnostics["n_selected_pairs"] == 0


def test_synchronizing_laplacian_matches_edge_quadratic_form() -> None:
    incidence = np.asarray([[1.0, 0.0], [-1.0, 1.0], [0.0, -1.0]])
    coupling = np.asarray([2.0, 3.0])
    equilibrium = np.asarray([0.1, -0.2, 0.1])
    laplacian, stiffness = synchronizing_laplacian(incidence, coupling, equilibrium)
    vector = np.asarray([0.4, -0.1, -0.3])
    assert np.isclose(
        vector @ laplacian @ vector,
        np.sum(stiffness * (incidence.T @ vector) ** 2),
        atol=1e-13,
    )
