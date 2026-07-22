from __future__ import annotations

import numpy as np


def complete_incidence(n_nodes: int) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
    edges = tuple((i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes))
    incidence = np.zeros((n_nodes, len(edges)), dtype=np.float64)
    for column, (i, j) in enumerate(edges):
        incidence[i, column] = 1.0
        incidence[j, column] = -1.0
    return incidence, edges
