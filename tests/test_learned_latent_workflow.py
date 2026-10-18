from __future__ import annotations

import json
from pathlib import Path

import h5py
import pytest

from kron_reduce.workflows import learned_causal_latent as workflow
from project_harness.evidence import sha256_file, verify_artifact_directory


def test_learning_pipeline_freezes_before_opening_audit(
    root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_name = "learned_causal_latent_synthetic_v1.toml"
    config = (root / "configs" / "experiments" / config_name).read_text(encoding="utf-8")
    config = (
        config.replace("epochs = 400", "epochs = 2")
        .replace("latent_dimensions = [0, 2, 4]", "latent_dimensions = [0, 2]")
        .replace("history_durations_s = [0.1, 0.2, 0.5]", "history_durations_s = [0.2]")
        .replace("train_lines_per_source = 3", "train_lines_per_source = 1")
        .replace("val_lines_per_source = 2", "val_lines_per_source = 1")
    )
    config_path = tmp_path / "configs" / "experiments" / config_name
    config_path.parent.mkdir(parents=True)
    config_path.write_text(config, encoding="utf-8")
    for relative in (
        "src/kron_reduce/models/learned_latent.py",
        "src/kron_reduce/models/causal_latent.py",
        "src/kron_reduce/data/history_contract.py",
        "src/kron_reduce/workflows/learned_causal_latent.py",
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((root / relative).read_bytes())
    result = tmp_path / "result"
    result.mkdir()
    original = workflow.make_visible_samples
    roles = []

    def monitored_samples(config, role, duration):
        if role == "audit":
            freeze = json.loads((result / "selection_freeze.json").read_text(encoding="utf-8"))
            assert sha256_file(result / "selected-final.npz") == freeze["final_checkpoint_sha256"]
            assert freeze["audit_generated"] is False
        roles.append(role)
        return original(config, role, duration)

    def refuse_hdf(*args, **kwargs):
        raise AssertionError("synthetic learning may not read canonical data")

    monkeypatch.setattr(workflow, "make_visible_samples", monitored_samples)
    monkeypatch.setattr(h5py, "File", refuse_hdf)
    workflow.run(
        root=tmp_path,
        experiment={"config": config_name, "data_policy": "analytic_synthetic_only"},
        stage="learning_audit",
        result_dir=result,
        run_id="unit-visible-learning",
    )
    assert roles == ["train", "val", "audit"]
    summary = json.loads((result / "learning_summary.json").read_text(encoding="utf-8"))
    assert summary["process_status"] == "PASS"
    assert summary["hidden_training_targets_read"] is False
    assert summary["audit_opened_after_checkpoint_freeze"] is True
    verify_artifact_directory(result, strict=True)
