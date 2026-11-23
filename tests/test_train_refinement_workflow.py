from __future__ import annotations

import json
import shutil
import tomllib
from pathlib import Path

import numpy as np
import pytest

from kron_reduce.evaluation.train_mechanism_audit import audit_event
from kron_reduce.workflows import kiki_train_refinement as workflow
from project_harness.report_storage import locate_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUTH = workflow.AUTH
EXPERIMENT = workflow.EXPERIMENT
CONFIG_REL = Path("configs/experiments/paper_kiki_a10_train_refinement.toml")


def _copy_approval_fixture(destination: Path):
    policy_path = PROJECT_ROOT / "configs/research_policy.json"
    manifest_path = PROJECT_ROOT / "configs/project_manifest.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    approval = policy["development_evaluations"][AUTH]
    relatives = [
        CONFIG_REL,
        Path("configs/research_policy.json"),
        Path("configs/project_manifest.json"),
        Path(approval["development_protocol_path"]),
        Path(approval["design_path"]),
        Path(approval["scientific_review_path"]),
    ]
    for relative in relatives:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        source = locate_report(PROJECT_ROOT, relative.as_posix())
        shutil.copyfile(PROJECT_ROOT / source, target)

    # Reconstruct the pre-run approval only in this temporary fixture.
    policy["development_evaluations"][AUTH]["status"] = "user_approved"
    (destination / "configs/research_policy.json").write_text(
        json.dumps(policy, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    config = tomllib.loads((destination / CONFIG_REL).read_text(encoding="utf-8"))
    copied_manifest = json.loads(
        (destination / "configs/project_manifest.json").read_text(encoding="utf-8"))
    experiment = copied_manifest["experiments"][EXPERIMENT]
    assert experiment == manifest["experiments"][EXPERIMENT]
    return destination, config, experiment


def test_a10_authorization_accepts_registered_config_manifest_and_all_bound_docs(tmp_path):
    root, config, experiment = _copy_approval_fixture(tmp_path / "valid")
    approval = workflow.authorization(root, experiment, config)
    assert approval["status"] == "user_approved"
    assert approval["scientific_review_path"]
    assert approval["scientific_review_sha256"]
    assert config["approval_protocol_sha256"]
    assert config["design_sha256"]


def test_real_registry_keeps_a09_consumed_while_a10_fixture_is_reconstructed(tmp_path):
    policy = json.loads((PROJECT_ROOT / "configs/research_policy.json").read_text(encoding="utf-8"))
    assert policy["development_evaluations"]["kiki_a09_development_20261024"]["status"] == "consumed"
    _, _, _ = _copy_approval_fixture(tmp_path / "approved-copy")
    reread = json.loads((PROJECT_ROOT / "configs/research_policy.json").read_text(encoding="utf-8"))
    assert reread["development_evaluations"]["kiki_a09_development_20261024"]["status"] == "consumed"
    assert reread["development_evaluations"][AUTH]["status"] == policy["development_evaluations"][AUTH]["status"]


def test_a10_authorization_rejects_changed_budget_and_consumed_status(tmp_path):
    root, config, experiment = _copy_approval_fixture(tmp_path / "budget")
    changed = dict(config)
    changed["max_total_rhs"] += 1
    with pytest.raises(PermissionError, match="differs from its approval"):
        workflow.authorization(root, experiment, changed)

    consumed_root, consumed_config, consumed_experiment = _copy_approval_fixture(tmp_path / "consumed")
    policy_path = consumed_root / "configs/research_policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["development_evaluations"][AUTH]["status"] = "consumed"
    policy_path.write_text(json.dumps(policy), encoding="utf-8", newline="\n")
    with pytest.raises(PermissionError, match="active fixed-model"):
        workflow.authorization(consumed_root, consumed_experiment, consumed_config)


@pytest.mark.parametrize(
    ("relative_key", "error_match"),
    [
        ("development_protocol_path", "approved document bytes changed"),
        ("design_path", "approved document bytes changed"),
        ("scientific_review_path", "Astra scientific approval binding changed"),
    ],
)
def test_a10_authorization_rejects_tampered_bound_documents(tmp_path, relative_key, error_match):
    root, config, experiment = _copy_approval_fixture(tmp_path / relative_key.replace("/", "_"))
    relative = config[relative_key] if relative_key in config else json.loads(
        (root / "configs/research_policy.json").read_text(encoding="utf-8")
    )["development_evaluations"][AUTH][relative_key]
    path = root / relative
    path.write_bytes(path.read_bytes() + b"tampered\n")
    with pytest.raises(PermissionError, match=error_match):
        workflow.authorization(root, experiment, config)


def test_a10_authorization_rejects_manifest_identity_change(tmp_path):
    root, config, experiment = _copy_approval_fixture(tmp_path / "manifest")
    changed = dict(experiment)
    changed["stage"] = "other_stage"
    with pytest.raises(PermissionError, match="experiment identity changed"):
        workflow.authorization(root, changed, config)


def test_a10_saved_array_guard_rejects_one_bit_synthetic_perturbation():
    base = {
        "history": np.arange(33, dtype=np.float64).reshape(11, 3),
        "future": np.array([0.0, 0.5, 1.0], dtype=np.float64),
        "offsets": np.array([0, 3], dtype=np.int64),
    }
    workflow.require_array_identity({key: value.copy() for key, value in base.items()}, base)

    altered = {key: value.copy() for key, value in base.items()}
    altered["history"][7, 1] = np.nextafter(altered["history"][7, 1], np.inf)
    with pytest.raises(ValueError, match="differ"):
        workflow.require_array_identity(base, altered)

    missing = dict(base)
    missing.pop("future")
    with pytest.raises(ValueError, match="differ"):
        workflow.require_array_identity(base, missing)


def test_coarse_fine_bound_uses_original_truth_window_even_if_coarse_peak_moves():
    history = np.zeros((11,20))
    truth,coarse,fine = (np.zeros((13,20)) for _ in range(3))
    truth[1:6,0], truth[1:6,10] = 5.0,20.0
    coarse[1:6,0],coarse[1:6,10] = 1.0,20.0
    fine[1:6,0],fine[1:6,10] = 1.2,20.0
    coarse[8:13,0],coarse[8:13,10] = 8.0,20.0
    fine[8:13,0],fine[8:13,10] = 7.0,20.0
    options = {"equilibrium_delta":np.zeros(10),"mass":np.ones(10),
               "omega_s":120*np.pi,"target_scale":np.ones(20)}
    actual = audit_event(history=history,true_future=truth,pred_future=fine,**options)
    comparison = workflow.compare_on_truth_windows(history=history,coarse=coarse,fine=fine,
        truth_diagnostic=actual,**options)
    rep = comparison["representative_truth_window"]
    assert (rep["start_index"],rep["end_index"]) == (11,15)
    assert comparison["future_rho"]["truth"]["best_window_start_index"] == 18
    assert comparison["future_rho"]["truth"]["rho"] > 0
    assert rep["coarse_margin_same_window"] < 0
    fixed = rep["fixed_five_point_audit"]
    assert fixed["bound_verified"] is True
    assert fixed["truth"]["index_offset_to_record_grid"] == 11
    assert abs(fixed["truth"]["rho"]-fixed["prediction"]["rho"]) <= fixed["E_atom"]+1e-12
