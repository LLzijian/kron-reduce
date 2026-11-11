import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts import audit_state_port_train_saved as saved_audit


def _metadata_only_authorization(monkeypatch, tmp_path, *, active):
    monkeypatch.setattr(saved_audit, "PLAN", "plan.json")
    monkeypatch.setattr(saved_audit, "SOURCES", "sources.json")
    monkeypatch.setattr(saved_audit, "PROTOCOL", "protocol.md")
    monkeypatch.setattr(saved_audit, "SCRIPT", "audit.py")
    monkeypatch.setattr(saved_audit, "sha256", lambda _path: "bound-hash")
    plan = {
        "scope": {},
        "postrun_saved_arithmetic_audit": {
            "npz_decode_limit_per_file": 1,
            "model_calls": 0,
            "rhs_calls": 0,
            "autograd_calls": 0,
            "optimizer_calls": 0,
            "output_path": saved_audit.OUTPUT,
            "script_path": "audit.py",
            "script_sha256": "bound-hash",
            "same_run_json_and_artifact_hash_checks": True,
            "allowed_records": ["common-initial-state", "fixed-targets", "*-*/update-*",
                                "*-*/final", "paths-*", "numpy-*", "scores-*"],
        },
        "sources_path": "sources.json",
        "sources_sha256": "bound-hash",
        "protocol_path": "protocol.md",
        "protocol_sha256": "bound-hash",
    }
    sources = {"schema": "kiki_state_port_train_sources_v1"}
    approval = {
        "status": "user_approved" if active else "inactive",
        "authorization": saved_audit.AUTH,
        "plan_path": "plan.json",
        "plan_sha256": "bound-hash",
        "protocol_path": "protocol.md",
        "protocol_sha256": "bound-hash",
        "sources_path": "sources.json",
        "sources_sha256": "bound-hash",
        "script_path": "audit.py",
        "script_sha256": "bound-hash",
        "same_run_json_and_artifact_hash_checks": True,
        "allowed_records": plan["postrun_saved_arithmetic_audit"]["allowed_records"],
        "npz_decode_limit_per_file": 1,
        "model_calls": 0,
        "rhs_calls": 0,
        "autograd_calls": 0,
        "optimizer_calls": 0,
        "output_path": saved_audit.OUTPUT,
        "implementation_bindings": {},
    }
    policy = {"development_evaluations": {saved_audit.AUTH: approval}}
    payloads = {
        "plan.json": plan,
        "sources.json": sources,
        "configs/research_policy.json": policy,
    }
    reads = []

    def read_json(path):
        relative = path.relative_to(tmp_path).as_posix()
        reads.append(relative)
        if relative not in payloads:
            pytest.fail(f"unexpected metadata/payload read: {relative}")
        return payloads[relative]

    monkeypatch.setattr(saved_audit, "read_json", read_json)
    monkeypatch.setattr(
        saved_audit.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="", returncode=0),
    )
    return reads


def test_dirty_checkout_rejected_before_metadata_or_npz_reads(monkeypatch, tmp_path):
    calls = []

    def dirty(*_args, **_kwargs):
        calls.append("git-status")
        return SimpleNamespace(stdout=" M tracked-file.py\n", returncode=0)

    monkeypatch.setattr(saved_audit.subprocess, "run", dirty)
    monkeypatch.setattr(saved_audit, "read_json", lambda *_args: pytest.fail("metadata read while dirty"))
    monkeypatch.setattr(saved_audit, "decode_record", lambda *_args: pytest.fail("NPZ decoded while dirty"))
    with pytest.raises(RuntimeError, match="clean checkout"):
        saved_audit.authorize_and_reserve(tmp_path)
    assert calls == ["git-status"]


@pytest.mark.parametrize("active", [False, True], ids=["inactive-policy", "missing-binding"])
def test_policy_or_exact_binding_rejected_before_result_payload(monkeypatch, tmp_path, active):
    reads = _metadata_only_authorization(monkeypatch, tmp_path, active=active)
    monkeypatch.setattr(saved_audit, "decode_record", lambda *_args: pytest.fail("NPZ decoded before authorization"))
    expected = "approval or source bindings mismatch" if not active else "script or source manifest is not bound"
    with pytest.raises(RuntimeError, match=expected):
        saved_audit.authorize_and_reserve(tmp_path)
    assert reads == ["plan.json", "sources.json", "configs/research_policy.json"]


@pytest.mark.parametrize("bad_hash", [True, False], ids=["hash-mismatch", "npz-roster-mismatch"])
def test_artifact_manifest_rejected_before_record_decode(monkeypatch, tmp_path, bad_hash):
    (tmp_path / "sentinel.txt").write_text("toy artifact", encoding="utf-8")
    digest = hashlib.sha256(b"toy artifact").hexdigest()
    manifest = {"sentinel.txt": "wrong" if bad_hash else digest}
    (tmp_path / "artifact_hashes.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(saved_audit, "decode_record", lambda *_args: pytest.fail("decoded before manifest gate"))
    with pytest.raises(RuntimeError, match="manifest or NPZ roster failed before any decode"):
        saved_audit.audit_records(tmp_path.parent, tmp_path, {}, {}, saved_audit.Audit())

