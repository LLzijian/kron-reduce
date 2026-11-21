import io
from zipfile import ZipFile

import numpy as np
import pytest

from kron_reduce.workflows.task3_unsettled_15s import classify, history_member


def test_history_reads_first_eleven_rows():
    x = np.arange(811 * 20, dtype=float).reshape(811, 20)
    file = io.BytesIO()
    np.savez_compressed(file, x=x)
    with ZipFile(file) as archive:
        np.testing.assert_array_equal(history_member(archive, "x"), x[:11])


def test_extension_uses_full_trigger_and_new_settlement_window():
    pytest.importorskip("task3.protocol")
    event = {"id": "synthetic", "x": np.zeros((11, 20)), "mass": np.ones(10),
             "equilibrium_delta": np.zeros(10), "omega_s": 2 * np.pi * 60,
             "label": "unstable", "event_time_s": None}
    path = np.zeros((1501, 20))
    path[700:801, 10] = 3
    path[800:805, 0] = 4
    path[800:805, 10] = 12
    first = classify([path, path], event, [.0025, .00125], 8, match_settlement=True)
    assert first["status"] == "window_no_los_unsettled"
    extended = classify([path, path], event, [.0025, .00125], 15, match_settlement=True)
    assert extended["status"] == "predicted_los"
    assert extended["predicted_los_time_forecast_s"] == 8.04
    path[800:805] = 0
    assert classify([path], event, [.01], 15, match_settlement=False)["status"] == "window_settled"
