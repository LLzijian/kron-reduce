from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from project_harness.evidence import (
    EvidenceContext,
    validate_stage_evidence,
    verify_artifact_directory,
)
from kron_reduce.workflows import p1_a03_real_f1_controlled_chain as chain


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _instance(instance_id: str, family: str) -> dict:
    factor = [] if family == "NODE_HIGH_ORDER" else [[1.0], [0.0]]
    return {
        "instance_id": instance_id,
        "family": family,
        "time_constant": 0.2,
        "frequency_hz": 0.8 if family == "RESONANT_MEMORY" else None,
        "damping_ratio": 0.1 if family == "RESONANT_MEMORY" else None,
        "factor": factor,
        "port_rank": 0 if family == "NODE_HIGH_ORDER" else 1,
        "source_support": {"source_a": 1.0},
        "energy_constraint": "reciprocal_storage",
        "removable": True,
    }


def _legacy_structure(path: Path) -> str:
    instances = [_instance("node", "NODE_HIGH_ORDER")]
    instances += [_instance(f"edge-{index}", "EDGE_MEMORY") for index in range(3)]
    instances += [_instance(f"resonant-{index}", "RESONANT_MEMORY") for index in range(3)]
    value = {
        "format_version": 2,
        "active_block_instances": instances,
        "memory_taus": [0.12, 0.25, 0.44],
        "resonances": [[0.2, 0.49], [0.8, 0.125], [1.4, 0.1]],
        "node_tau": 0.2,
        "potential_scale": 0.25,
        "max_internal_step": 0.01,
        "node_damping": [999.0],
        "coupling": [999.0],
    }
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    return _sha(path)


def _projection_contract() -> dict:
    return {
        "source_format_version": 2,
        "target_format_version": 3,
        "node_tau": 0.2,
        "potential_scale": 0.25,
        "max_internal_step": 0.01,
        "expected_node_high_order_instances": 1,
        "expected_edge_memory_instances": 3,
        "expected_resonant_memory_instances": 3,
        "numerical_coefficients_inherited": False,
    }


def test_frozen_execution_config_and_manifest_registration() -> None:
    root = Path(__file__).resolve().parents[1]
    config_path, config = chain._load_execution_config(
        root, {"config": chain.CONFIG_NAME}
    )
    assert _sha(config_path) == chain.CONFIG_SHA256
    assert _sha(root / chain.PROTOCOL_PATH) == chain.PROTOCOL_SHA256
    assert config["execution"]["child_directories"] == list(chain.EXPECTED_CHILDREN)
    project = json.loads((root / "configs/project_manifest.json").read_text(encoding="utf-8"))
    for name in ("model_unit_gate", "controlled_experiment"):
        entry = project["experiments"][
            f"paper_ieee39_modular_energy_p1_a03_{name}_f1_seed324"
        ]
        assert entry["status"] == "gated"
        assert entry["module"].endswith("p1_a03_real_f1_controlled_chain")
        assert entry["config"] == chain.CONFIG_NAME


def test_structure_projection_strips_routing_and_fitted_arrays(tmp_path: Path) -> None:
    source = tmp_path / "legacy.json"
    digest = _legacy_structure(source)
    output = tmp_path / "structure.json"
    projected = chain.project_structure_prior(
        source, digest, output, _projection_contract()
    )
    assert projected["format_version"] == 3
    assert projected["model_count"] == 1
    assert projected["source_routing"] is False
    assert projected["numerical_coefficients_inherited"] is False
    assert projected["source_support_removed"] is True
    assert len(projected["active_block_instances"]) == 7
    assert all(row["source_support"] == {} for row in projected["active_block_instances"])
    assert "node_damping" not in projected and "coupling" not in projected
    with pytest.raises(FileExistsError):
        chain.project_structure_prior(source, digest, output, _projection_contract())


def test_structure_projection_rejects_changed_union_and_source_hash(tmp_path: Path) -> None:
    source = tmp_path / "legacy.json"
    digest = _legacy_structure(source)
    value = json.loads(source.read_text(encoding="utf-8"))
    value["active_block_instances"].pop()
    source.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="hash-mismatched"):
        chain.project_structure_prior(
            source, digest, tmp_path / "out-a.json", _projection_contract()
        )
    changed_digest = _sha(source)
    with pytest.raises(ValueError, match="instance counts"):
        chain.project_structure_prior(
            source, changed_digest, tmp_path / "out-b.json", _projection_contract()
        )


def test_source_commit_must_be_unique_current_head(monkeypatch: pytest.MonkeyPatch) -> None:
    commit = "a" * 40
    state = {
        "current_stage": "source_commit",
        "history": [
            {"action": "advance", "to": "source_commit", "evidence": {"git_commit": commit}}
        ],
    }
    monkeypatch.setattr(chain, "load_state", lambda _root, _workflow: state)
    monkeypatch.setattr(chain, "git_head", lambda _root: commit)
    assert chain._source_commit_evidence(Path("."))["git_commit"] == commit
    monkeypatch.setattr(chain, "git_head", lambda _root: "b" * 40)
    with pytest.raises(RuntimeError, match="differs"):
        chain._source_commit_evidence(Path("."))


def test_model_gate_scratch_path_is_short_unique_and_run_bound(tmp_path: Path) -> None:
    long_prefix = "paper_ieee39_modular_energy_p1_a03_model_unit_gate_f1_seed324-"
    first = chain._scratch_run_directory(tmp_path, long_prefix + "run-a" * 20)
    second = chain._scratch_run_directory(tmp_path, long_prefix + "run-b" * 20)
    assert first.parent == tmp_path / "scratch" / "one_off"
    assert len(first.name) == len("p1-model-gate-") + 16
    assert first != second
    with pytest.raises(ValueError, match="unsafe run ID"):
        chain._scratch_run_directory(tmp_path, "../escape")


def test_junit_parser_sums_child_suites_and_rejects_failures(tmp_path: Path) -> None:
    report = tmp_path / "report.xml"
    report.write_text(
        '<testsuites><testsuite tests="2" failures="0" errors="0" skipped="1"/>'
        '<testsuite tests="3" failures="0" errors="0" skipped="0"/></testsuites>',
        encoding="utf-8",
    )
    assert chain._parse_junit(report) == {
        "tests": 5,
        "failures": 0,
        "errors": 0,
        "skipped": 1,
    }
    report.write_text(
        '<testsuite tests="1" failures="1" errors="0" skipped="0"/>',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="did not pass"):
        chain._parse_junit(report)


def test_model_gate_writes_nonpromotional_strict_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path
    candidate = root / "results" / "candidate"
    candidate.mkdir(parents=True)
    digest = _legacy_structure(candidate / "frozen_model.json")
    dataset = root / "data" / "canonical"
    dataset.mkdir(parents=True)
    config_path = root / "configs" / "chain.toml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("x=1\n", encoding="utf-8")
    result = root / "results" / "gate"
    result.mkdir()
    config = {
        "candidate_structure_file": "frozen_model.json",
        "candidate_structure_sha256": digest,
        "candidate_artifact_manifest_sha256": "c" * 64,
        "structure_projection": _projection_contract(),
    }
    monkeypatch.setattr(
        chain, "_candidate_context", lambda _root, _config: (candidate, {"status": "PASS"})
    )
    monkeypatch.setattr(chain, "_verify_dataset_inputs", lambda _root, _config: dataset)
    monkeypatch.setattr(
        chain, "_run_full_pytest", lambda _root, _result, _run_id: {"tests": 407, "failures": 0, "errors": 0, "skipped": 0}
    )
    monkeypatch.setattr(chain, "audit_project", lambda _root: {"status": "PASS", "findings": []})
    monkeypatch.setattr(chain, "verify_data", lambda _root: {"status": "PASS", "findings": []})
    monkeypatch.setattr(chain, "status_lines", lambda _root: [])
    monkeypatch.setattr(chain, "git_head", lambda _root: "d" * 40)
    chain._run_model_unit_gate(root, config_path, config, result, "gate")
    manifest = verify_artifact_directory(result, strict=True)
    assert "stage_evidence.json" in manifest
    evidence = json.loads((result / "stage_evidence.json").read_text(encoding="utf-8"))
    assert evidence["unit_test_count"] == 407
    assert "scientific_verdict" not in evidence
    assert evidence["promotion_decision"] == "KEEP_BASELINE"
    assert evidence["level_6_claimed"] is False
    workflow = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs/workflows/ieee39_modular_energy_p1_a03_v1.json"
        ).read_text(encoding="utf-8")
    )
    binding = {"run_id": result.name, "path": "artifact_hashes.json", "sha256": _sha(result / "artifact_hashes.json")}
    validate_stage_evidence(
        EvidenceContext(root=root, workflow=workflow, target="model_unit_gate", predecessor="candidate_acquired"),
        {"status": "PASS", "result_dir": result.name, "artifact_manifest": binding, **evidence},
    )


def _fake_child(directory: Path, name: str) -> str:
    directory.mkdir(parents=True, exist_ok=False)
    payload = directory / "payload.json"
    payload.write_text(json.dumps({"name": name}), encoding="utf-8")
    manifest = directory / "artifact_hashes.json"
    manifest.write_text(json.dumps({"payload.json": _sha(payload)}), encoding="utf-8")
    return _sha(manifest)


def test_controlled_parent_orders_freeze_before_conditional_pm_and_seals_children(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path
    candidate = root / "results" / "candidate"
    candidate.mkdir(parents=True)
    digest = _legacy_structure(candidate / "frozen_model.json")
    (candidate / "roles.csv").write_text("unused\n", encoding="utf-8")
    dataset = root / "data" / "canonical"
    dataset.mkdir(parents=True)
    config_path = root / "configs" / "chain.toml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("x=1\n", encoding="utf-8")
    result = root / "results" / "controlled"
    result.mkdir()
    commit = "e" * 40
    config = {
        "candidate_structure_file": "frozen_model.json",
        "candidate_structure_sha256": digest,
        "candidate_artifact_manifest_sha256": "f" * 64,
        "role_manifest_file": "roles.csv",
        "role_manifest_sha256": "1" * 64,
        "fold": "F1",
        "structure_projection": _projection_contract(),
        "calibration": {"max_nfev": 16},
        "dataset_h5_sha256": "2" * 64,
    }
    order: list[str] = []

    class FakeManifest:
        source_sha256 = "1" * 64
        role_id_hashes = {"fit": "a", "selection": "b", "internal_audit": "c"}

        def ids(self, role: str) -> tuple[str, ...]:
            return {"fit": ("fit",), "selection": ("selection",), "internal_audit": ("internal",)}[role]

    class FakeManifestFactory:
        @staticmethod
        def from_csv(*_args, **_kwargs):
            return FakeManifest()

    class FakeReader:
        def __init__(self, *_args):
            self.accesses = []
            self.time_grid_audits = []

        def read_role(self, role: str):
            order.append(f"read_{role}")
            self.accesses.append({"role": role})
            return [role]

    monkeypatch.setattr(chain, "_source_commit_evidence", lambda _root, _workflow=chain.WORKFLOW: {"git_commit": commit})
    monkeypatch.setattr(chain, "_candidate_context", lambda _root, _config: (candidate, {"status": "PASS"}))
    monkeypatch.setattr(chain, "_verify_dataset_inputs", lambda _root, _config: dataset)
    monkeypatch.setattr(chain, "P1RoleManifest", FakeManifestFactory)
    monkeypatch.setattr(chain, "P1CausalReader", FakeReader)
    monkeypatch.setattr(
        chain,
        "CanonicalStore",
        SimpleNamespace(open=lambda _path: SimpleNamespace(h5_path=dataset / "data.h5", dataset_metadata=SimpleNamespace(M_prior=[1] * 10))),
    )

    def fake_pm(*_args, role: str, purpose, **kwargs):
        order.append(f"pm_{role}_{purpose.value}")
        if role == "internal_audit":
            assert "fit_five" in order
            assert kwargs["expected_core_sha256"] == "core-sha"
        return SimpleNamespace(
            accesses=({"role": role},),
            core_checkpoint_sha256=kwargs.get("expected_core_sha256"),
            time_grid_audits=(),
        )

    monkeypatch.setattr(chain, "read_pm_batch", fake_pm)

    def fake_base(directory: Path, *_args, **_kwargs):
        order.append("fit_five")
        child_sha = _fake_child(directory, "base")
        return {"directory": str(directory.resolve()), "artifact_manifest_sha256": child_sha}

    monkeypatch.setattr(chain, "fit_and_save_five_arms", fake_base)
    monkeypatch.setattr(chain, "verify_five_arm_bundle", lambda *_args: {"arms": [{}, {}, {"checkpoint_hash": "core-sha"}]})

    def fake_five_eval(*args):
        directory = args[-1]
        order.append("evaluate_five")
        _fake_child(directory, "five-eval")
        return {"arms": [{}, {}, {}, {}, {}]}

    monkeypatch.setattr(chain, "evaluate_saved_five_arms", fake_five_eval)

    def fake_counter(*args):
        directory = args[-1]
        order.append("fit_counter")
        child_sha = _fake_child(directory, "counter")
        return {"directory": str(directory.resolve()), "artifact_manifest_sha256": child_sha}

    monkeypatch.setattr(chain, "fit_and_save_counterfactuals", fake_counter)

    def fake_counter_eval(*args):
        directory = args[-1]
        order.append("evaluate_counter")
        child_sha = _fake_child(directory, "counter-eval")
        return {"directory": str(directory.resolve()), "artifact_manifest_sha256": child_sha, "variant_count": 8}

    monkeypatch.setattr(chain, "evaluate_saved_counterfactuals", fake_counter_eval)

    def fake_support(*args):
        directory = args[-1]
        order.append("support")
        child_sha = _fake_child(directory, "support")
        return {"directory": str(directory.resolve()), "artifact_manifest_sha256": child_sha, "module_count": 7, "module_statuses": {"node": "UNRESOLVED"}}

    monkeypatch.setattr(chain, "evaluate_saved_module_support", fake_support)

    def fake_modal(*args):
        directory = args[-1]
        order.append("modal")
        child_sha = _fake_child(directory, "modal")
        return {"directory": str(directory.resolve()), "artifact_manifest_sha256": child_sha, "view_status_counts": {"UNDETERMINED": 1}}

    monkeypatch.setattr(chain, "evaluate_saved_modal_observability", fake_modal)
    monkeypatch.setattr(chain, "verify_counterfactual_artifacts", lambda *_args: {})
    monkeypatch.setattr(chain, "verify_module_support_artifacts", lambda *_args: {})
    monkeypatch.setattr(chain, "verify_modal_observability_artifacts", lambda *_args: {})

    chain._run_controlled_experiment(root, config_path, config, result, "controlled")
    verify_artifact_directory(result, strict=True)
    assert order == [
        "read_fit",
        "read_selection",
        "read_internal_audit",
        "pm_fit_fit_closure_target",
        "fit_five",
        "pm_internal_audit_conditional_diagnostic",
        "evaluate_five",
        "fit_counter",
        "evaluate_counter",
        "support",
        "modal",
    ]
    summary = json.loads((result / "controlled_experiment_summary.json").read_text(encoding="utf-8"))
    assert summary["scientific_verdict"] is None
    assert summary["promotion_decision"] == "KEEP_BASELINE"
    assert summary["level_6_claimed"] is False
    assert set(json.loads((result / "lineage.json").read_text(encoding="utf-8"))["child_artifacts"]) == set(chain.EXPECTED_CHILDREN)
    stage_evidence = json.loads((result / "stage_evidence.json").read_text(encoding="utf-8"))
    assert "scientific_verdict" not in stage_evidence
    workflow = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs/workflows/ieee39_modular_energy_p1_a03_v1.json"
        ).read_text(encoding="utf-8")
    )
    binding = {"run_id": result.name, "path": "artifact_hashes.json", "sha256": _sha(result / "artifact_hashes.json")}
    validate_stage_evidence(
        EvidenceContext(root=root, workflow=workflow, target="controlled_experiment", predecessor="source_commit"),
        {"status": "PASS", "result_dir": result.name, "artifact_manifest": binding, **stage_evidence},
    )
