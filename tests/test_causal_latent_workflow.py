from __future__ import annotations

import json
from pathlib import Path

import h5py
import pytest

from kron_reduce.workflows.causal_latent_initialization import run
from project_harness.evidence import verify_artifact_directory


def test_registered_synthetic_workflow_has_no_hdf_access(
    root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse_dataset(*args, **kwargs):
        raise AssertionError("the analytic audit may not read canonical HDF data")

    monkeypatch.setattr(h5py, "File", refuse_dataset)
    experiment = {
        "config": "causal_latent_initialization_synthetic_v1.toml",
        "data_policy": "analytic_synthetic_only",
    }
    run(
        root=root,
        experiment=experiment,
        stage="synthetic_audit",
        result_dir=tmp_path,
        run_id="unit-synthetic-only",
    )
    verify_artifact_directory(tmp_path, strict=True)
    evidence = json.loads((tmp_path / "stage_evidence.json").read_text(encoding="utf-8"))
    assert evidence["process_status"] == "PASS"
    assert evidence["ieee39_trajectory_reads"] == []
    assert evidence["unobservable_refusal_passed"] is True
    ablations = json.loads(
        (tmp_path / "history_initialization_ablations.json").read_text(encoding="utf-8")
    )
    assert ablations["forecast_origin_after_clearing_s"] == 0.2
    assert ablations["latent_history_initialized"]["omega_rmse_rad_per_s"] < 1e-8
    assert ablations["latent_zero_initialized"]["omega_rmse_rad_per_s"] > 1e-3
