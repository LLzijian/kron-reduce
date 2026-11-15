import numpy as np
import pytest

from kron_reduce.evaluation.machine_local_topology_margin import fit_counts


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
