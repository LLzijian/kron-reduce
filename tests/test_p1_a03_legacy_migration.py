"""Legacy format tests, including read-only input from the real failed baseline.

These are metadata-migration tests, not model-fitting or performance evidence.
All mutated artifacts live in isolated pytest directories, never real results.
"""
from __future__ import annotations

import copy
import json
import shutil
import tomllib
from pathlib import Path

import pytest

import kron_reduce.workflows.ieee39_modular_energy_p1_diagnosis as diagnosis
from kron_reduce.workflows.ieee39_modular_energy_p1_diagnosis import (
    CONFIG_FILE_BY_WORKFLOW,
    EXPECTED_CONFIG_A02,
    EXPECTED_CONFIG_A03,
    WORKFLOW_A03,
    _find_registered_plan,
    _json,
    _legacy_migration_records,
    _run_candidate,
    _run_plan,
    _sha256,
    _validate_config,
    _write_hashes,
    _write_legacy_migration,
    validate_p1_candidate_contract,
)
from project_harness.report_storage import locate_report


def _strict_json(path: Path):
    def reject(token):
        raise ValueError(f"nonfinite token in new JSON: {token}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


@pytest.fixture
def legacy(tmp_path):
    baseline, output = tmp_path / "b", tmp_path / "c"
    baseline.mkdir()
    output.mkdir()
    declarations = {"model_count": 1, "source_routing": False}
    for name in ("validation_summary.json", "lineage.json"):
        _json(baseline / name, declarations)
    # Simulate a historical permissive writer, not the strict new artifact writer.
    payload = {"a~b/c": [{"margin": float("nan")}, float("inf"), -float("inf")], "text": "NaN"}
    (baseline / "structure_selection_result.json").write_text(
        json.dumps(payload, allow_nan=True), encoding="utf-8"
    )
    return baseline, output


def test_nonfinite_legacy_payload_is_preserved_not_zeroed(legacy):
    baseline, output = legacy
    original = (baseline / "structure_selection_result.json").read_bytes()
    _write_legacy_migration(baseline, output)
    assert (output / "legacy_structure_selection_result.json").read_bytes() == original
    assert (baseline / "structure_selection_result.json").read_bytes() == original
    envelope = _strict_json(output / "candidate_structure_selection_explicit.json")
    assert "a~b/c" not in envelope  # No scientific payload in the new metadata.
    assert envelope["legacy_structure_artifact"]["sha256"] == _sha256(baseline / "structure_selection_result.json")
    audit = _strict_json(output / "legacy_format_audit.json")
    assert audit["nonfinite_value_count"] == 3
    assert [item["json_pointer"] for item in audit["nonfinite_values"]] == [
        "/a~0b~1c/0/margin", "/a~0b~1c/1", "/a~0b~1c/2"
    ]
    assert [item["value_kind"] for item in audit["nonfinite_values"]] == ["NaN", "Infinity", "-Infinity"]
    assert audit["scientific_values_rewritten"] is False
    assert audit["scientific_acceptance"] == "NOT_EVALUATED"


@pytest.mark.parametrize("file,field,value", [
    ("validation_summary.json", "model_count", True),
    ("lineage.json", "model_count", 2),
    ("lineage.json", "source_routing", True),
    ("lineage.json", "routing_keys", ["source"]),
    ("structure_selection_result.json", "model_count", 2),
    ("structure_selection_result.json", "source_routing", True),
    ("structure_selection_result.json", "routing_keys", ["fault"]),
])
def test_legacy_declaration_conflicts_refuse_before_copy(legacy, file, field, value):
    baseline, output = legacy
    record = json.loads((baseline / file).read_text(encoding="utf-8"))
    record[field] = value
    (baseline / file).write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(RuntimeError):
        _write_legacy_migration(baseline, output)
    assert list(output.iterdir()) == []


def test_legacy_duplicate_keys_and_overwrite_are_refused(legacy):
    baseline, output = legacy
    path = baseline / "structure_selection_result.json"
    original = path.read_bytes()
    path.write_text('{"margin":0,"margin":NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        _write_legacy_migration(baseline, output)
    assert list(output.iterdir()) == []
    path.write_bytes(original)
    _write_legacy_migration(baseline, output)
    preserved = {file.name: file.read_bytes() for file in output.iterdir()}
    with pytest.raises(FileExistsError):
        _write_legacy_migration(baseline, output)
    assert {file.name: file.read_bytes() for file in output.iterdir()} == preserved


def test_a03_registration_changes_attempt_not_scientific_thresholds():
    repo = Path(__file__).resolve().parents[1]
    config = tomllib.loads((repo / "configs/experiments" / CONFIG_FILE_BY_WORKFLOW[WORKFLOW_A03]).read_text(encoding="utf-8"))
    _validate_config(config, "codex/loop-p1-a03")
    changed = {key for key in config if config[key] != EXPECTED_CONFIG_A02[key]}
    assert changed == {"protocol_revision", "attempt", "branch", "previous_attempt_workflow", "previous_retry_evidence", "previous_retry_evidence_sha256"}
    manifest = json.loads((repo / "configs/project_manifest.json").read_text(encoding="utf-8"))
    entries = {key: value for key, value in manifest["experiments"].items() if "p1_a03_" in key}
    assert len(entries) == 9
    assert all(value["workflow"] == WORKFLOW_A03 and value["reads_test_performance"] is False for value in entries.values())


@pytest.fixture
def real_saved_candidate(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    config = copy.deepcopy(EXPECTED_CONFIG_A03)
    source = repo / "results" / config["baseline_run_id"]
    assert source.is_dir(), "This regression requires the registered real legacy baseline"
    baseline = tmp_path / "results/b"
    shutil.copytree(source, baseline)
    config["baseline_run_id"] = "b"  # Short fixture path avoids the Windows path-length limit.
    relative_config = Path("configs/experiments") / CONFIG_FILE_BY_WORKFLOW[WORKFLOW_A03]
    config_path = tmp_path / relative_config
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        (repo / relative_config).read_text(encoding="utf-8").replace(EXPECTED_CONFIG_A03["baseline_run_id"], "b"),
        encoding="utf-8",
    )
    for relative in (
        config["previous_retry_evidence"],
        f"configs/workflows/{WORKFLOW_A03}.json",
        "docs/protocols/ieee39_modular_energy_p1_a03_design_zh.md",
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / locate_report(repo, relative), destination)
    plan = tmp_path / "results/p"
    _run_plan(tmp_path, config_path, config, plan, plan.name)
    state_path = tmp_path / ".project_state" / f"{WORKFLOW_A03}_state.json"
    state_path.parent.mkdir()
    _json(state_path, {
        "workflow": WORKFLOW_A03, "current_stage": "plan_registered", "approvals": {},
        "history": [{"action": "advance", "from": "scaffold", "to": "plan_registered", "evidence": {
            "status": "PASS", "result_dir": "p", "artifact_manifest": {
                "run_id": "p", "path": "artifact_hashes.json", "sha256": _sha256(plan / "artifact_hashes.json")
            }
        }}],
    })
    # Snapshot mechanics have their own Git regression; this fixture tests real saved data.
    monkeypatch.setattr(diagnosis, "_source_snapshot", lambda *_: {
        "branch": config["branch"], "reproducible_from_clean_commit": False,
    })
    candidate = tmp_path / "results/c"
    _run_candidate(tmp_path, config_path, config, candidate, candidate.name)
    return tmp_path, config, config_path, baseline, candidate


def test_real_legacy_candidate_acquisition_and_metadata_check(real_saved_candidate):
    root, config, _, baseline, candidate = real_saved_candidate
    result = validate_p1_candidate_contract(root, candidate, config, expected_manifest_sha256=_sha256(candidate / "artifact_hashes.json"))
    assert result["status"] == "PASS"
    assert result["full_model_unit_gate_passed"] is False
    assert result["level_6_claimed"] is False
    assert _sha256(candidate / "legacy_structure_selection_result.json") == _sha256(baseline / "structure_selection_result.json")
    envelope, expected_audit = _legacy_migration_records(baseline)
    assert _strict_json(candidate / "candidate_structure_selection_explicit.json") == envelope
    audit = _strict_json(candidate / "legacy_format_audit.json")
    assert audit == expected_audit and audit["nonfinite_value_count"] > 0


@pytest.mark.parametrize("attack", ["zero_missing_margin", "falsify_audit", "new_scientific_claim", "future_pm", "test", "other_fold"])
def test_rehashed_real_candidate_still_rejects_semantic_corruption(real_saved_candidate, attack):
    root, config, _, _, candidate = real_saved_candidate
    if attack == "zero_missing_margin":
        path = candidate / "legacy_structure_selection_result.json"
        path.write_text(path.read_text(encoding="utf-8").replace('"margin": NaN', '"margin": 0.0'), encoding="utf-8")
    elif attack == "falsify_audit":
        path = candidate / "legacy_format_audit.json"
        value = _strict_json(path)
        value["nonfinite_value_count"] = 0
        _json(path, value)
    elif attack == "new_scientific_claim":
        path = candidate / "candidate_structure_selection_explicit.json"
        value = _strict_json(path)
        value["level_6_claimed"] = True
        _json(path, value)
    else:
        path = candidate / "data_access_audit.json"
        value = _strict_json(path)
        field, changed = {
            "future_pm": ("future_observed_pm_read", True),
            "test": ("test_ids_read", ["test-example"]),
            "other_fold": ("other_folds_read", ["F2"]),
        }[attack]
        value[field] = changed
        _json(path, value)
    _write_hashes(candidate)
    with pytest.raises(RuntimeError):
        validate_p1_candidate_contract(root, candidate, config, expected_manifest_sha256=_sha256(candidate / "artifact_hashes.json"))


def test_frozen_a03_design_cannot_be_changed_after_registration(real_saved_candidate):
    root, config, config_path, _, _ = real_saved_candidate
    (root / "docs/protocols/ieee39_modular_energy_p1_a03_design_zh.md").write_text("different contract", encoding="utf-8")
    with pytest.raises(RuntimeError, match="current protocol differs"):
        _find_registered_plan(root, config_path, config)


def test_candidate_stage_must_call_contract_validator(real_saved_candidate, monkeypatch):
    root, config, config_path, _, _ = real_saved_candidate

    def refuse(*args, **kwargs):
        raise RuntimeError("injected contract refusal")

    monkeypatch.setattr(diagnosis, "validate_p1_candidate_contract", refuse)
    with pytest.raises(RuntimeError, match="injected contract refusal"):
        _run_candidate(root, config_path, config, root / "results/c2", "c2")
