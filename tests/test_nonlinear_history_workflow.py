import json
import tomllib

import h5py
import numpy as np
import pytest

from kron_reduce.data.p2_roles import P2RoleRow
from kron_reduce.models.nonlinear_latent import EquilibriumLearningSample
from kron_reduce.workflows import nonlinear_history_learning as workflow
from project_harness.evidence import sha256_file, verify_artifact_directory


def test_synthetic_freeze_precedes_independent_population(root, tmp_path, monkeypatch):
    config_path = root / "configs/experiments/nonlinear_history_synthetic_a03_v1.toml"
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    config.update(epochs=1, train_lines=2, val_lines=2, audit_lines=2, pairs_per_line=1)
    config["candidates"] = config["candidates"][:2]
    result = tmp_path / "result"
    result.mkdir()
    original = workflow.synthetic_samples
    roles = []

    def monitored(config, role):
        if role == "audit":
            freeze = json.loads((result / "selection_freeze.json").read_text(encoding="utf-8"))
            for candidate in freeze["candidate_table"]:
                assert (
                    sha256_file(result / f"candidate-{candidate['name']}.npz")
                    == candidate["checkpoint_sha256"]
                )
            assert freeze["ieee39_audit_authorized"] is False
        roles.append(role)
        return original(config, role)

    def refuse_hdf(*args, **kwargs):
        raise AssertionError("synthetic learning must not read canonical HDF")

    monkeypatch.setattr(workflow, "synthetic_samples", monitored)
    monkeypatch.setattr(h5py, "File", refuse_hdf)
    workflow.execute_study(
        root=root,
        config=config,
        config_path=config_path,
        result_dir=result,
        run_id="unit-nonlinear",
    )
    assert roles == ["train", "val", "audit"]
    summary = json.loads((result / "learning_summary.json").read_text(encoding="utf-8"))
    assert summary["ieee39_audit_event_count"] == 0
    assert summary["hidden_training_targets_read"] is False
    assert summary["checkpoint_replay_exact"] is True
    verify_artifact_directory(result, strict=True)


def test_f1_reader_retains_equilibrium_and_never_opens_audit(tmp_path):
    mass = np.ones(10)
    rows = tuple(
        P2RoleRow(f"{role}-event", "synthetic", role, role, "train", 1001)
        for role in ("fit", "selection", "internal_audit")
    )
    eq = np.linspace(-0.1, 0.1, 10)
    with h5py.File(tmp_path / "ieee39_postclear_canonical_v1.h5", "w") as handle:
        for row in rows[:2]:
            group = handle.create_group(f"events/{row.event_id}")
            time = np.r_[0.0, 0.003 + np.arange(320) * 0.01]
            pattern = np.linspace(-0.01, 0.01, 10)
            group.create_dataset("postclear/t", data=time)
            group.create_dataset("postclear/delta", data=eq + time[:, None] * pattern)
            group.create_dataset("postclear/omega", data=np.broadcast_to(pattern, (len(time), 10)))
            power = np.full((len(time), 10), np.nan)
            power[0] = 0.0
            group.create_dataset("postclear/Pm", data=power)
            group.create_dataset("equilibrium/delta0", data=eq)
            group.create_dataset("equilibrium/Pm_equilibrium", data=np.zeros(10))
    provider = workflow.TrainValNonlinearSamples(tmp_path, rows, mass, tmp_path)
    config = {
        "history_durations_s": [0.2],
        "history_sample_counts": [21],
        "forecast_duration_s": 3.0,
    }
    for role in ("train", "val"):
        _, samples = provider(config, role, 0.2)
        assert isinstance(samples[0], EquilibriumLearningSample)
        assert np.allclose(samples[0].equilibrium_delta, eq)
        assert np.isclose(samples[0].history.time[0], -0.193)
        assert samples[0].forecast_time[-1] == 3.0
    accesses = len(provider.accesses)
    with pytest.raises(RuntimeError, match="never authorizes"):
        provider(config, "audit", 0.2)
    assert len(provider.accesses) == accesses
    assert provider.access_audit()["audit_open_count"] == 0
    assert provider.access_audit()["future_pm_reads"] == []
