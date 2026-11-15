import numpy as np
import pytest
import torch

from kron_reduce.evaluation.machine_local_topology import make_model
from kron_reduce.evaluation.machine_local_topology_margin import fit_counts, save_equation
from kron_reduce.evaluation.machine_local_topology_truth import parameters


def test_fit_counts_uses_exact_visible_future_support():
    rows = [
        {"label": "stable", "trigger_index": None},
        {"label": "unstable", "trigger_index": 11},
        {"label": "unstable", "trigger_index": 12},
        {"label": "unstable", "trigger_index": 310},
    ]

    counts = fit_counts(rows)

    assert isinstance(counts, np.ndarray)
    assert np.issubdtype(counts.dtype, np.integer)
    np.testing.assert_array_equal(counts, [300, 1, 2, 300])


@pytest.mark.parametrize("row", [
    {"label": "unknown", "trigger_index": None},
    {"label": "stable", "trigger_index": 11},
    {"label": "unstable", "trigger_index": None},
    {"label": "unstable"},
    {"label": "unstable", "trigger_index": "11"},
    {"label": "unstable", "trigger_index": 11.0},
    {"label": "unstable", "trigger_index": True},
    {"label": "unstable", "trigger_index": 10},
    {"label": "unstable", "trigger_index": 311},
])
def test_fit_counts_rejects_invalid_or_unusable_rows(row):
    with pytest.raises(ValueError):
        fit_counts([row])


def test_save_frozen_equation_from_evaluation_device(tmp_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_model(parameters(), "T0M1", device, train_scales=True).requires_grad_(False)
    save_equation(model, tmp_path, "frozen")
    with np.load(tmp_path / "frozen-equation.npz", allow_pickle=False) as saved:
        assert set(saved.files) == set(model.state_dict())
        for key, value in model.state_dict().items():
            np.testing.assert_array_equal(saved[key], value.cpu().numpy())
