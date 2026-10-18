from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from kron_reduce.data.p2_roles import P2RoleRow
from kron_reduce.models.causal_latent import PassiveLatentClosure
from kron_reduce.models.learned_latent import LatentFitSettings, LearnedLatentModel
from kron_reduce.workflows.ieee39_history_learning import AuthorizedHistorySamples
from project_harness.evidence import sha256_file


def _provider(tmp_path: Path):
    mass = np.ones(10)
    rows = tuple(
        P2RoleRow(f"{role}-event", "synthetic", role, role, "train", 1001)
        for role in ("fit", "selection", "internal_audit")
    )
    with h5py.File(tmp_path / "ieee39_postclear_canonical_v1.h5", "w") as handle:
        for row in rows:
            group = handle.create_group(f"events/{row.event_id}")
            equilibrium = np.linspace(-0.1, 0.1, 10)
            time = np.r_[0.0, 0.003 + np.arange(1000) * 0.01]
            pattern = np.linspace(-0.01, 0.01, 10)
            group.create_dataset("postclear/t", data=time)
            group.create_dataset(
                "postclear/delta", data=equilibrium[None, :] + time[:, None] * pattern[None, :]
            )
            group.create_dataset("postclear/omega", data=np.broadcast_to(pattern, (len(time), 10)))
            pm = np.full((len(time), 10), np.nan)
            pm[0] = 0.0
            group.create_dataset("postclear/Pm", data=pm)
            group.create_dataset("equilibrium/delta0", data=equilibrium)
            group.create_dataset("equilibrium/Pm_equilibrium", data=np.zeros(10))
    return AuthorizedHistorySamples(tmp_path, rows, mass, tmp_path)


def test_f1_history_provider_enforces_freeze_and_future_channel_boundary(tmp_path: Path) -> None:
    provider = _provider(tmp_path)
    config = {
        "history_durations_s": [0.2],
        "history_sample_counts": [21],
        "forecast_duration_s": 1.0,
        "audit_forecast_duration_s": 9.0,
    }
    with pytest.raises(FileNotFoundError):
        provider(config, "audit", 0.2)
    assert provider.accesses == []
    _, train = provider(config, "train", 0.2)
    assert len(train) == 1
    assert np.isclose(-train[0].history.time[0], 0.193)
    assert np.max(np.abs(train[0].history.delta[0])) < 1e-14  # equilibrium is removed
    assert np.isfinite(train[0].history.constant_power).all()
    laplacian = np.eye(10) - np.ones((10, 10)) / 10
    closure = PassiveLatentClosure(
        np.ones(10),
        laplacian,
        0.1 * laplacian,
        np.zeros((10, 0)),
        np.zeros((0, 0)),
        np.zeros((0, 0)),
    )
    model = LearnedLatentModel(
        closure,
        np.zeros((0, 110)),
        np.ones(110),
        np.ones(20),
        LatentFitSettings(latent_dimension=0),
        {"fitted_event_ids": ["fit-event", "selection-event"]},
    )
    for name in ("selected-final.npz", "memoryless-final.npz"):
        model.save(tmp_path / name)
    (tmp_path / "selection_freeze.json").write_text(
        json.dumps(
            {
                "final_checkpoint_sha256": sha256_file(tmp_path / "selected-final.npz"),
                "memoryless_checkpoint_sha256": sha256_file(tmp_path / "memoryless-final.npz"),
            }
        ),
        encoding="utf-8",
    )
    _, audit = provider(config, "audit", 0.2)
    assert len(audit[0].forecast_time) == 901
    assert audit[0].forecast_time[-1] == 9.0
    with pytest.raises(RuntimeError, match="only be opened once"):
        provider(config, "audit", 0.2)
    access = provider.access_audit()
    assert access["future_pm_reads"] == []
    assert access["test_ids_read"] == []
    assert access["audit_open_count"] == 1
    assert all(
        row["selection"] == "0" for row in access["accesses"] if row["dataset"] == "postclear/Pm"
    )
