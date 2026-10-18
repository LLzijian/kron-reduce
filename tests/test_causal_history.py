from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from kron_reduce.data.causal_history import read_measured_history


def test_history_reads_no_sample_after_forecast_origin(tmp_path: Path) -> None:
    path = tmp_path / "history.h5"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("postclear")
        # A shifted first sample is followed by the actual 0.01 s source clock.
        time = np.r_[0.0, 0.003 + np.arange(100) * 0.01]
        values = np.zeros((101, 10))
        values[21:] = np.nan  # These future values must never reach the input.
        group.create_dataset("t", data=time)
        group.create_dataset("delta", data=values)
        group.create_dataset("omega", data=values)
        window = read_measured_history(
            handle,
            sample_count=21,
            mass=np.ones(10),
            constant_power=np.zeros(10),
            equilibrium_delta=np.zeros(10),
        )
        assert np.isclose(window.forecast_origin_after_clearing_s, 0.193)
        assert window.history.time[-1] == 0.0
        assert np.isfinite(window.history.delta).all()
        assert {row["dataset"] for row in window.access_log} == {
            "postclear/t",
            "postclear/delta",
            "postclear/omega",
        }
        assert all(row["stop_exclusive"] == 21 for row in window.access_log)


def test_history_does_not_pad_short_observations(tmp_path: Path) -> None:
    with h5py.File(tmp_path / "short.h5", "w") as handle:
        group = handle.create_group("postclear")
        group.create_dataset("t", data=np.arange(8) * 0.01)
        with pytest.raises(ValueError, match="real registered samples"):
            read_measured_history(
                handle,
                sample_count=11,
                mass=np.ones(10),
                constant_power=np.zeros(10),
                equilibrium_delta=np.zeros(10),
            )
