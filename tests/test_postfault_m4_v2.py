import json

import numpy as np
import pytest

from kron_reduce.evaluation.postfault_m4_graph import _partition
from kron_reduce.evaluation.postfault_m4_v2 import (
    plot_energy_decomposition,
    plot_slow_coherency,
    separate_storage,
    slow_coherency,
)
from kron_reduce.physics.graph import complete_incidence


def _graph():
    incidence, edges = complete_incidence(10)
    left, right = np.array(edges).T
    scale = {"a06": 1., "hard_r": 2., "m3": .8}
    return {"edge_rows": [{"edge_index": i, "machine_indices": list(pair), "stiffness": scale}
                          for i, pair in enumerate(edges)],
            "graphs": {k: _partition(incidence, np.full(len(edges), v), left, right) for k, v in scale.items()},
            "stable_val": {"role": "val", "truth_label": "stable", "events_used": 95, "events_allocated": 95,
                           "forecast_tail_s": [6., 8.], "spectrum": {"groups": [list(range(9)), [9]]}}}


def _profiles():
    profiles = {}
    for role in ("train", "val"):
        profiles[role] = {}
        for step in ("0.0025", "0.00125"):
            profiles[role][step] = {}
            for model in ("hard_r", "soft_2r", "m3"):
                rows = []
                for i, kinetic in enumerate((1., 2., 2., 4.)):
                    memory = 100. if i < 2 and model == "soft_2r" else 0.
                    potential = .2*i if model == "m3" else 0.
                    rows.append({"id": f"{role}/{i}", "label": "stable" if i < 2 else "unstable",
                                 "kinetic": kinetic, "potential": potential, "memory": memory,
                                 "total": kinetic+potential+memory})
                profiles[role][step][model] = {"rows": rows}
    return profiles


def test_complete_graph_with_known_mass_weighted_mode_and_degenerate_ordinary_spectrum():
    mass = np.r_[np.ones(9), 4.]
    result = slow_coherency(_graph(), mass)
    for name, scale in (("a06", 1.), ("hard_r", 2.), ("m3", .8)):
        graph = result["graphs"][name]
        general = graph["inertia_weighted"]
        assert general["eigenvalues"][1] == pytest.approx(scale*(1.+9./4.))
        assert general["lambda3_over_lambda2"] == pytest.approx(10./(1.+9./4.))
        assert general["groups"] == [list(range(9)), [9]]
        assert general["reference_comparison"]["matched_pairs"] == 45
        assert general["reference_comparison"]["adjusted_rand_index"] == 1.
        vectors = np.array(general["physical_eigenvectors_by_column"])
        np.testing.assert_allclose(vectors.T @ np.diag(mass) @ vectors, np.eye(10), atol=1e-12)
        assert general["relative_residual_frobenius"] < 1e-12
        assert graph["ordinary"]["eigenspace_dimension"] == 9
        assert graph["ordinary"]["reference_comparison"] is None
    json.dumps(result, allow_nan=False)


def test_matched_storage_ranking_ties_ratios_and_control():
    result = separate_storage(_profiles())
    for steps in result["splits"].values():
        for data in steps.values():
            hard = data["models"]["hard_r"]["components"]
            assert hard["kinetic"]["auroc"] == hard["mechanical"]["auroc"] == .875
            assert hard["mechanical"]["unstable_to_stable_median_ratio"] == 2.
            assert hard["potential"]["auroc"] == .5 and hard["potential"]["unstable_to_stable_median_ratio"] is None
            assert hard["mechanical"]["stable_q25_median_q75"] == [1.25, 1.5, 1.75]
            assert data["paired_v4"]["same_mechanical_each_event"]
            assert data["paired_v4"]["max_abs_total_minus_memory_difference"] == 0.
            assert data["paired_v4"]["soft_minus_hard_total_auroc"] == -.875
    json.dumps(result, allow_nan=False)
    bad = _profiles()
    bad["val"]["0.0025"]["hard_r"]["rows"][0]["total"] = 20.
    with pytest.raises(ValueError, match="inconsistent total"):
        separate_storage(bad)


def test_figures_are_600_dpi_and_vector_text(tmp_path):
    from PIL import Image

    graph = slow_coherency(_graph(), np.r_[np.ones(9), 4.])
    storage = separate_storage(_profiles())
    for plot, data, label in ((plot_slow_coherency, graph, "slow"), (plot_energy_decomposition, storage, "energy")):
        png, svg = plot(data, tmp_path/label)
        with Image.open(png) as image:
            assert image.info["dpi"] == pytest.approx((600, 600), abs=.1)
        source = svg.read_bytes()
        assert b"<text" in source and b"<image" not in source and b"\r" not in source
        assert all(line.rstrip() == line for line in source.splitlines())
