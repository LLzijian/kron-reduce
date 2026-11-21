import json
from math import log

import numpy as np
import pytest
import torch
from PIL import Image

from kron_reduce.evaluation.postfault_m4_graph import evaluate_graph, plot_topology
from kron_reduce.models.machine_local_topology import TopologyMemoryModel
from kron_reduce.models.nonlinear_latent import NonlinearPassiveClosure
from kron_reduce.models.unified_discovery_m3 import UnifiedDiscoveryModel
from kron_reduce.physics.graph import complete_incidence


def test_coherency_and_effective_edges_share_the_known_cut(tmp_path):
    incidence, edges = complete_incidence(10)
    stiffness = np.array([.025 if 9 in pair else .2 for pair in edges])
    mass, port, tau = np.ones(10), np.full(10, .5), np.full(10, .6)
    damping = .5 * (np.eye(10) - np.ones((10, 10)) / 10.)
    empty = np.empty((0, 0))
    backbone = NonlinearPassiveClosure(mass, stiffness, damping, np.empty((10, 0)), empty, empty)
    hard = TopologyMemoryModel(backbone, topology=False, port=port, tau=tau, train_scales=True)
    m3 = UnifiedDiscoveryModel(backbone, port=port, tau=tau, support=("G2",))
    with torch.no_grad():
        hard.log_stiffness_factor.fill_(log(1.1))
        m3.edge_log_factor[0] = log(1.25)
        m3.edge_log_factor[8] = log(.8)

    wave = np.sin(np.arange(811) * .06)[:, None] * np.r_[np.ones(9), -9.]
    offsets = np.linspace(-.1, .1, 10)
    events = [{"role": "val", "label": "stable", "id": f"synthetic-{i}",
               "x": np.c_[np.zeros((811, 10)), amplitude * wave + offsets]}
              for i, amplitude in enumerate((.003, .03, 0.))]
    result = evaluate_graph({"hard_r": hard, "m3": m3}, events)
    json.dumps(result, allow_nan=False)
    assert result["stable_val"]["events_used"] == 2
    assert result["stable_val"]["uninformative_events"] == ["synthetic-2"]
    assert result["stable_val"]["spectrum"]["groups"] == [list(range(9)), [9]]
    assert result["stable_val"]["distance_by_edge"][8] == pytest.approx(100. / 9.)
    assert result["comparisons"]["m3"]["stable_val"]["matched_pairs"] == 45
    assert result["comparisons"]["m3"]["stable_val"]["adjusted_rand_index"] == 1.
    assert result["comparisons"]["hard_r"]["g10_vs_remaining"]["same_unordered_cut"]
    assert result["graphs"]["m3"]["fiedler_gap"] > 0.
    assert len(result["edge_rows"]) == 45
    first, second = result["top_10_by_absolute_change_from_a06"][:2]
    assert [first["machine_indices"], second["machine_indices"]] == [[0, 1], [0, 9]]
    assert first["relative_change"]["m3_vs_hard_r"] == pytest.approx(1.25 / 1.1 - 1.)
    assert second["relative_change"]["m3_vs_a06"] == pytest.approx(-.2)
    np.testing.assert_allclose(result["graphs"]["hard_r"]["laplacian"], (incidence * (1.1 * stiffness)) @ incidence.T)

    png, svg = plot_topology(result, tmp_path / "effective_topology")
    with Image.open(png) as raster:
        assert raster.info["dpi"] == pytest.approx((600., 600.), abs=.01)
    vector = svg.read_bytes()
    assert b"G10" in vector and b"vs Hard-R" in vector and b"<text" in vector
    assert b"\r" not in vector and all(line == line.rstrip() for line in vector.splitlines())
