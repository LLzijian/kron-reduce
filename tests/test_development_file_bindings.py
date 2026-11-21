from __future__ import annotations

import hashlib
import json
import subprocess

import pytest

from project_harness import runner
from project_harness.git_state import verify_bound_file


def _git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def _repo(root, attributes=None):
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "binding-test@example.invalid")
    _git(root, "config", "user.name", "Binding Test")
    if attributes is not None:
        (root / ".gitattributes").write_text(attributes, encoding="ascii", newline="\n")
    return root


def _commit(root, *paths):
    _git(root, "add", "--", *paths)
    _git(root, "commit", "-qm", "fixture")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def test_text_attribute_normalizing_crlf_is_rejected(tmp_path):
    root = _repo(tmp_path / "repo", "*.md text eol=lf\n")
    content = b"line one\r\nline two\r\n"
    (root / "design.md").write_bytes(content)
    _commit(root, ".gitattributes", "design.md")
    blob = _git(root, "show", "HEAD:design.md").stdout
    assert _sha(blob) != _sha(content)
    with pytest.raises(RuntimeError, match="BOUND_FILE_HEAD_MISMATCH"):
        verify_bound_file(root, "design.md", _sha(content))


def test_text_minus_preserves_crlf_blob_and_passes(tmp_path):
    root = _repo(tmp_path / "repo", "design.md -text\n")
    content = b"line one\r\nline two\r\n"
    (root / "design.md").write_bytes(content)
    _commit(root, ".gitattributes", "design.md")
    verify_bound_file(root, "design.md", _sha(content))


def test_normal_lf_blob_passes(tmp_path):
    root = _repo(tmp_path / "repo")
    content = b"line one\nline two\n"
    (root / "design.md").write_bytes(content)
    _commit(root, "design.md")
    verify_bound_file(root, "design.md", _sha(content))


def test_relocated_bound_report_verifies_sealed_original_bytes(tmp_path):
    root = _repo(tmp_path / "repo")
    original = b"approved historical report\n"
    digest = _sha(original)
    sealed = root / "docs" / "reports" / "sealed" / "report.md"
    view = root / "docs" / "reports" / "topic" / "report.md"
    sealed.parent.mkdir(parents=True)
    view.parent.mkdir(parents=True)
    sealed.write_bytes(original)
    view.write_text("current navigable view\n", encoding="utf-8", newline="\n")
    registry = root / "configs" / "report_relocations.json"
    registry.parent.mkdir()
    registry.write_text(json.dumps({"docs/reports/report.md": {
        "path": "docs/reports/topic/report.md",
        "original_path": "docs/reports/sealed/report.md",
        "sha256": digest,
    }}), encoding="utf-8")
    _commit(root, "configs/report_relocations.json", "docs/reports/sealed/report.md",
            "docs/reports/topic/report.md")

    verify_bound_file(root, "docs/reports/report.md", digest)


@pytest.mark.parametrize("damage", ["sealed_bytes", "unsafe_alias"])
def test_relocated_bound_report_refuses_tampering_or_unsafe_alias(tmp_path, damage):
    root = _repo(tmp_path / "repo")
    original = b"approved historical report\n"
    digest = _sha(original)
    sealed = root / "docs" / "reports" / "sealed" / "report.md"
    view = root / "docs" / "reports" / "topic" / "report.md"
    sealed.parent.mkdir(parents=True)
    view.parent.mkdir(parents=True)
    sealed.write_bytes(original)
    view.write_text("current navigable view\n", encoding="utf-8", newline="\n")
    relocation = {
        "path": "../escape/report.md" if damage == "unsafe_alias" else "docs/reports/topic/report.md",
        "original_path": "docs/reports/sealed/report.md",
        "sha256": digest,
    }
    registry = root / "configs" / "report_relocations.json"
    registry.parent.mkdir()
    registry.write_text(json.dumps({"docs/reports/report.md": relocation}), encoding="utf-8")
    _commit(root, "configs/report_relocations.json", "docs/reports/sealed/report.md",
            "docs/reports/topic/report.md")
    if damage == "sealed_bytes":
        sealed.write_bytes(b"tampered historical report\n")
        bound_digest = digest
    else:
        bound_digest = _sha(view.read_bytes())

    expected_error = "REPORT_RELOCATION: invalid destination" if damage == "unsafe_alias" else "BOUND_FILE_WORKTREE_MISMATCH"
    with pytest.raises(RuntimeError, match=expected_error):
        verify_bound_file(root, "docs/reports/report.md", bound_digest)


def test_worktree_edit_is_rejected_when_head_blob_matches(tmp_path):
    root = _repo(tmp_path / "repo", "design.md -text\n")
    committed = b"original\r\n"
    (root / "design.md").write_bytes(committed)
    _commit(root, ".gitattributes", "design.md")
    (root / "design.md").write_bytes(b"edited\r\n")
    with pytest.raises(RuntimeError, match="BOUND_FILE_WORKTREE_MISMATCH"):
        verify_bound_file(root, "design.md", _sha(committed))


def test_runner_checks_policy_then_binding_before_data_or_workflow(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "load_manifest", lambda _root: {"experiments": {
        "authorized": {"development_authorization_id": "auth", "workflow": "wf",
                       "dataset": "kiki_ieee39_v1", "module": "missing.module"}}})
    monkeypatch.setattr(runner, "validate_run_policy", lambda *a, **k: calls.append("policy"))

    def reject(*_args, **_kwargs):
        calls.append("binding")
        raise RuntimeError("BOUND_FILE_HEAD_MISMATCH")

    monkeypatch.setattr(runner, "_verify_development_text_bindings", reject)
    monkeypatch.setattr(runner, "verify_data", lambda *a, **k: pytest.fail("data validation ran"))
    monkeypatch.setattr(runner.importlib, "import_module", lambda *a, **k: pytest.fail("workflow imported"))
    with pytest.raises(RuntimeError, match="BOUND_FILE_HEAD_MISMATCH"):
        runner.run_experiment(tmp_path, "authorized", "fit")
    assert calls == ["policy", "binding"]
    assert not (tmp_path / ".project_state").exists()
    assert not (tmp_path / "results").exists()


def test_missing_authorization_and_historical_reproduction_are_noops(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "load_policy", lambda _root: pytest.fail("policy should not be read"))
    runner._verify_development_text_bindings(tmp_path, {"config": "legacy.toml"})
    runner._verify_development_text_bindings(
        tmp_path, {"development_authorization_id": "legacy-auth", "config": "legacy.toml"},
        historical_reproduction=True,
    )


def test_config_policy_binding_mismatch_is_rejected_before_file_check(tmp_path, monkeypatch):
    config_dir = tmp_path / "configs" / "experiments"
    config_dir.mkdir(parents=True)
    protocol = tmp_path / "docs" / "protocol.md"
    protocol.parent.mkdir()
    protocol.write_text("approval\n", encoding="utf-8", newline="\n")
    design = tmp_path / "docs" / "design.md"
    design.write_text("design\n", encoding="utf-8", newline="\n")
    protocol_sha = _sha(protocol.read_bytes())
    design_sha = _sha(design.read_bytes())
    (config_dir / "authorized.toml").write_text(
        "development_authorization_id = 'auth'\n"
        "development_protocol_path = 'docs/protocol.md'\n"
        f"approval_protocol_sha256 = '{protocol_sha}'\n"
        "design_path = 'docs/design.md'\n"
        f"design_sha256 = '{design_sha}'\n",
        encoding="utf-8", newline="\n",
    )
    monkeypatch.setattr(runner, "load_policy", lambda _root: {
        "development_evaluations": {"auth": {
            "status": "user_approved",
            "approval_protocol_path": "docs/protocol.md",
            "approval_protocol_sha256": "0" * 64,
            "design_path": "docs/design.md",
            "design_sha256": design_sha,
        }}})
    monkeypatch.setattr(runner, "verify_bound_file", lambda *a: pytest.fail("file check ran"))
    with pytest.raises(RuntimeError, match="DEVELOPMENT_BINDING_POLICY_MISMATCH"):
        runner._verify_development_text_bindings(
            tmp_path, {"development_authorization_id": "auth", "config": "authorized.toml"})


def test_registered_approved_protocol_and_design_bindings_pass(tmp_path, monkeypatch):
    root = _repo(tmp_path / "repo")
    config_dir = root / "configs" / "experiments"
    config_dir.mkdir(parents=True)
    protocol = root / "docs" / "protocol.md"
    protocol.parent.mkdir()
    protocol.write_text("approved protocol\n", encoding="utf-8", newline="\n")
    design = root / "docs" / "design.md"
    design.write_text("frozen design\n", encoding="utf-8", newline="\n")
    protocol_sha, design_sha = _sha(protocol.read_bytes()), _sha(design.read_bytes())
    config = config_dir / "authorized.toml"
    config.write_text(
        "development_authorization_id = 'auth'\n"
        "development_protocol_path = 'docs/protocol.md'\n"
        f"approval_protocol_sha256 = '{protocol_sha}'\n"
        "design_path = 'docs/design.md'\n"
        f"design_sha256 = '{design_sha}'\n",
        encoding="utf-8", newline="\n",
    )
    _commit(root, "configs/experiments/authorized.toml", "docs/protocol.md", "docs/design.md")
    monkeypatch.setattr(runner, "load_policy", lambda _root: {
        "development_evaluations": {"auth": {
            "status": "user_approved",
            "approval_protocol_path": "docs/protocol.md",
            "approval_protocol_sha256": protocol_sha,
            "design_path": "docs/design.md",
            "design_sha256": design_sha,
        }}})
    runner._verify_development_text_bindings(
        root, {"development_authorization_id": "auth", "config": "authorized.toml"})


def test_non_active_approval_is_rejected(tmp_path, monkeypatch):
    config_dir = tmp_path / "configs" / "experiments"
    config_dir.mkdir(parents=True)
    (config_dir / "authorized.toml").write_text(
        "development_authorization_id = 'auth'\n", encoding="utf-8")
    monkeypatch.setattr(runner, "load_policy", lambda _root: {
        "development_evaluations": {"auth": {"status": "consumed"}}})
    with pytest.raises(RuntimeError, match="DEVELOPMENT_AUTHORIZATION_NOT_ACTIVE_USER_APPROVED"):
        runner._verify_development_text_bindings(
            tmp_path, {"development_authorization_id": "auth", "config": "authorized.toml"})
