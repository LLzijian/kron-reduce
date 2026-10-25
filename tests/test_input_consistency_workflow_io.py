"""Synthetic _execute integration across OutputIO, scope journals and sealing."""

from __future__ import annotations

import hashlib
import json
import os

import pytest
from test_input_consistency_workflow import synthetic_campaign

from kron_reduce.evaluation.input_consistency_output_io import OutputIO
from kron_reduce.evaluation.input_consistency_result_audit import (
    SealedRun,
    audit_output_io,
    audit_saved_scientific_values,
    audit_scope_store,
)
from kron_reduce.evaluation.input_consistency_seal import SealLedger
from kron_reduce.workflows import kiki_input_consistency as workflow


class _ManifestHashFailurePath:
    def __init__(self, target, state):
        self.target = target
        self.state = state

    def __getattr__(self, name):
        return getattr(self.target, name)

    def open(self, mode="r", *args, **kwargs):
        if mode == "rb":
            self.state["triggered"] = True
            raise OSError("injected manifest hash open failure")
        return self.target.open(mode, *args, **kwargs)


def _prepared_run(tmp_path, monkeypatch):
    root = tmp_path / "control-root"
    controls = root / "controls"
    requirements = root / "requirements"
    controls.mkdir(parents=True)
    requirements.mkdir()
    (requirements / "constraints-py312.txt").write_text(
        "synthetic constraints fixture\n", encoding="utf-8"
    )
    result_dir = tmp_path / "runs" / "synthetic-a17-run"
    result_dir.mkdir(parents=True)
    plan, compute, receipt = synthetic_campaign(result_dir, monkeypatch, make=False)
    receipt["run_id"] = result_dir.name
    plan_path, compute_path = controls / "read_plan.json", controls / "compute_plan.json"
    plan_path.write_text(json.dumps(plan, allow_nan=False), encoding="utf-8")
    compute_path.write_text(json.dumps(compute, allow_nan=False), encoding="utf-8")
    config = {
        "read_plan_path": "controls/read_plan.json",
        "compute_plan_path": "controls/compute_plan.json",
    }
    return root, result_dir, config, plan, compute, receipt


def _inject_fsync_failure(monkeypatch, target_path):
    enter = OutputIO.__enter__
    exit_context = OutputIO.__exit__
    state = {"triggered": False, "path": target_path}

    def enter_with_fault(self):
        result = enter(self)
        native_fsync = self._original_fsync

        def selected_fsync(file_descriptor):
            record = self._fd_handles.get(file_descriptor)
            if record is not None and record["path"] == target_path and not state["triggered"]:
                state["triggered"] = True
                raise OSError(f"injected output fsync failure for {target_path}")
            return native_fsync(file_descriptor)

        self._original_fsync = selected_fsync
        self._restore_native_fsync = native_fsync
        return result

    def exit_with_restored_fsync(self, exc_type, exc, traceback):
        self._original_fsync = self._restore_native_fsync
        return exit_context(self, exc_type, exc, traceback)

    monkeypatch.setattr(OutputIO, "__enter__", enter_with_fault)
    monkeypatch.setattr(OutputIO, "__exit__", exit_with_restored_fsync)
    return state


def _inject_seal_failure(monkeypatch, failure_stage):
    state = {"triggered": False}
    if failure_stage == "manifest_hash":
        native_safe_path = SealLedger._safe_path

        def fail_manifest_hash(ledger, name):
            target = native_safe_path(ledger, name)
            if name == "artifact_hashes.json":
                return _ManifestHashFailurePath(target, state)
            return target

        monkeypatch.setattr(SealLedger, "_safe_path", fail_manifest_hash)
        return state
    if failure_stage == "manifest_write":
        native_write = SealLedger.write_json
        native_fsync = os.fsync

        def fail_manifest_fsync(ledger, name, value):
            if name != "artifact_hashes.json":
                return native_write(ledger, name, value)

            def fail_once(file_descriptor):
                if not state["triggered"]:
                    state["triggered"] = True
                    raise OSError("injected manifest write fsync failure")
                return native_fsync(file_descriptor)

            os.fsync = fail_once
            try:
                return native_write(ledger, name, value)
            finally:
                os.fsync = native_fsync

        monkeypatch.setattr(SealLedger, "write_json", fail_manifest_fsync)
        return state
    raise AssertionError(f"unknown synthetic seal failure stage {failure_stage}")


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_manifest(result_dir):
    manifest = _read_json(result_dir / "artifact_hashes.json")
    actual_paths = {
        path.relative_to(result_dir).as_posix()
        for path in result_dir.rglob("*")
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    assert set(manifest) == actual_paths
    for relative, expected in manifest.items():
        digest = hashlib.sha256((result_dir / relative).read_bytes()).hexdigest()
        assert digest == expected
    return manifest


def _assert_source_journal(result_dir, expected_contexts):
    rows = [
        _read_json_line(line)
        for line in (result_dir / "source_access.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    attempted = [row for row in rows if row["transition"] == "attempted"]
    returned = [row for row in rows if row["transition"] == "returned"]
    failed = [row for row in rows if row["transition"] == "failed"]
    assert len(attempted) == len(returned) and not failed
    context_attempts = [row["identity"] for row in attempted if row["operation"] == "read_context"]
    assert len(context_attempts) == len(set(context_attempts)) == expected_contexts
    assert {row["identity"] for row in returned if row["operation"] == "read_context"} == set(
        context_attempts
    )
    return rows


def _read_json_line(line):
    return json.loads(line)


def test_execute_native_fsync_seals_complete_synthetic_campaign(tmp_path, monkeypatch):
    root, result_dir, config, plan, compute, receipt = _prepared_run(tmp_path, monkeypatch)
    summary, tail = workflow._execute(root, result_dir, config, plan, compute, receipt)

    assert summary["scientific_verdict"] == "REJECT"
    assert summary["promotion_decision"] == "KEEP_BASELINE"
    assert summary["scope_count"] == len(compute["scope_order"])
    assert summary["allocated_train"] == 268
    assert summary["eligible_train"] == 1
    assert (result_dir / "summary.json").is_file()
    assert _read_json(result_dir / "summary.json")["promotion_decision"] == "KEEP_BASELINE"
    assert _read_json(result_dir / "allocated_slots.json") == plan["population"]
    assert tail["manifest_final_hash_is_external_completion_action"] is True

    manifest = _verify_manifest(result_dir)
    assert "output_io.jsonl" in manifest
    assert "output_io_summary.json" in manifest
    assert "scientific_scopes.json" in manifest
    scopes = _read_json(result_dir / "scientific_scopes.json")
    assert [row["scope_id"] for row in scopes] == compute["scope_order"]
    assert all(row["durably_closed"] for row in scopes)
    assert sum(row["status"] != "complete" for row in scopes) == 0

    source_rows = _assert_source_journal(result_dir, len(plan["contexts"]))
    assert any(
        row["operation"] == "advance" and row["identity"] == "origins_freeze.json"
        for row in source_rows
    )
    assert any(
        row["operation"] == "advance" and row["identity"] == "futures_freeze.json"
        for row in source_rows
    )
    output_rows = [
        json.loads(line)
        for line in (result_dir / "output_io.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    io_summary = _read_json(result_dir / "output_io_summary.json")
    captured = io_summary["captured_run_io"]
    assert len(output_rows) == captured["records"]
    assert (
        len(output_rows)
        == tail["output_transitions_including_seal"]
        - 2 * (io_summary["sealing"]["actions_including_last_writes"])
    )
    assert captured["poisoned"] is False
    assert captured["records"] == len(output_rows)
    assert captured["fsync_calls_returned"] > 0
    assert captured["active_open_ids"] == []

    sealed = SealedRun(result_dir, tail["artifact_manifest_sha256"])
    scope_audit = audit_scope_store(sealed, compute)
    copies = {
        1: sealed.record("causal_inputs"),
        2: sealed.record("forecast_clocks"),
        3: sealed.record("comparison_inputs"),
    }
    replay_counts, replay_verdict = audit_saved_scientific_values(
        sealed, plan, scope_audit["statuses"], copies
    )
    assert replay_counts["history_paths_verified"] == 6
    assert replay_counts["future_paths_verified"] == 6
    assert replay_verdict["scientific_verdict"] == summary["scientific_verdict"]
    output_audit = audit_output_io(sealed, len(source_rows), compute=compute)
    assert output_audit["freeze_output_lifecycle_order_verified"] is True
    assert (
        output_audit["output_transitions_including_seal"]
        == tail["output_transitions_including_seal"]
    )


@pytest.mark.parametrize(
    ("target", "expected_unclosed"),
    [("hist-324-000-h005.npz", "hist-324-000-h005"), ("summary.json", None)],
)
def test_execute_fsync_failure_preserves_fatal_prefix_without_recomputation(
    tmp_path, monkeypatch, target, expected_unclosed
):
    root, result_dir, config, plan, compute, receipt = _prepared_run(tmp_path, monkeypatch)
    injection = _inject_fsync_failure(monkeypatch, target)

    with pytest.raises(OSError, match="injected output fsync failure"):
        workflow._execute(root, result_dir, config, plan, compute, receipt)
    assert injection["triggered"]

    partial = _read_json(result_dir / "fatal_scientific_partial.json")
    fatal = _read_json(result_dir / "fatal_prefix.json")
    assert partial["error_type"] == "OSError"
    assert fatal["retry_allowed"] is False
    if target == "hist-324-000-h005.npz":
        reader_cleanup = next(row for row in fatal["cleanup_errors"] if row["resource"] == "reader")
        assert reader_cleanup["error_type"] == "OutputIOError"
        assert "poisoned" in reader_cleanup["message"]
    assert fatal["output_io"]["poisoned"] is True
    assert fatal["unclosed_scope_unknown_spent"] == expected_unclosed
    assert fatal["unvisited_scopes"] == (
        compute["scope_order"][compute["scope_order"].index(expected_unclosed) + 1 :]
        if expected_unclosed is not None
        else []
    )
    assert not (result_dir / "artifact_hashes.json").exists()
    assert not (result_dir / "output_io_summary.json").exists()

    read_prefix = fatal["read_prefix"]
    used_contexts = read_prefix["used"]
    assert len(used_contexts) == len(set(used_contexts))
    assert (
        len(used_contexts) == len(plan["contexts"])
        if target == "summary.json"
        else len(used_contexts) < len(plan["contexts"])
    )
    source_rows = [
        _read_json_line(line)
        for line in (result_dir / "source_access.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    reads = [
        row["identity"]
        for row in source_rows
        if row["operation"] == "read_context" and row["transition"] == "returned"
    ]
    assert len(reads) == len(set(reads))
    assert (
        len(reads) == len(plan["contexts"])
        if target == "summary.json"
        else len(reads) < len(plan["contexts"])
    )

    output_rows = [
        json.loads(line)
        for line in (result_dir / "output_io.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    failed = [
        row for row in output_rows if row.get("path") == target and row["transition"] == "failed"
    ]
    assert len(failed) == 1 and failed[0]["fsync_calls_failed"] == 1
    assert fatal["output_io"]["unknown_operations"]["fsync"] == 1
    if expected_unclosed is not None:
        call_rows = [
            json.loads(line)
            for line in (result_dir / "operations" / "calls.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert {row["scope_id"] for row in call_rows} == set(
            compute["scope_order"][: compute["scope_order"].index(expected_unclosed) + 1]
        )
        assert any(
            row.get("category") == "integration_rhs" and row["transition"] == "returned"
            for row in call_rows
            if row["scope_id"] == expected_unclosed
        )


@pytest.mark.parametrize("failure_stage", ["manifest_write", "manifest_hash"])
def test_execute_seal_failure_saves_seal_prefix_without_repeating_science(
    tmp_path, monkeypatch, failure_stage
):
    root, result_dir, config, plan, compute, receipt = _prepared_run(tmp_path, monkeypatch)
    injection = _inject_seal_failure(monkeypatch, failure_stage)

    with pytest.raises(OSError, match="injected manifest"):
        workflow._execute(root, result_dir, config, plan, compute, receipt)
    assert injection["triggered"]

    partial = _read_json(result_dir / "fatal_scientific_partial.json")
    fatal = _read_json(result_dir / "fatal_prefix.json")
    assert partial["error_type"] == "OSError"
    assert "injected manifest" in partial["message"]
    assert partial["exception_chain"] == []
    seal_prefix = partial["input_consistency_seal_partial_record"]
    assert seal_prefix["poisoned"] is True
    assert seal_prefix["first_error"]["type"] == "OSError"
    assert "injected manifest" in seal_prefix["first_error"]["message"]
    assert seal_prefix["action_count"] > 0
    assert seal_prefix["transition_records"] == 2 * seal_prefix["action_count"]
    assert seal_prefix["actions"][-1]["status"] == "failed"
    assert seal_prefix["actions"][-1]["operation"] == (
        "hash_file" if failure_stage == "manifest_hash" else "write_json"
    )
    assert seal_prefix["actions"][-1]["name"] == "artifact_hashes.json"
    assert seal_prefix["actions"][-1]["stage"] == (
        "open" if failure_stage == "manifest_hash" else "fsync"
    )
    if failure_stage == "manifest_hash":
        assert seal_prefix["actions"][-1]["read_bytes_returned"] == 0
        assert seal_prefix["actions"][-1]["unknown_operations"] == 0
    else:
        assert seal_prefix["actions"][-1]["write_bytes_returned"] > 0
        assert seal_prefix["actions"][-1]["durability_unknown"] is True
    assert fatal["retry_allowed"] is False
    assert fatal["unclosed_scope_unknown_spent"] is None
    assert fatal["unvisited_scopes"] == []
    assert len(fatal["closed_scopes_prefix"]) == len(compute["scope_order"])
    assert (result_dir / "output_io_summary.json").is_file()
    assert (result_dir / "artifact_hashes.json").is_file()

    source_rows = [
        _read_json_line(line)
        for line in (result_dir / "source_access.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    returned_contexts = [
        row["identity"]
        for row in source_rows
        if row["operation"] == "read_context" and row["transition"] == "returned"
    ]
    assert len(returned_contexts) == len(set(returned_contexts)) == len(plan["contexts"])
    scope_rows = [
        _read_json_line(line)
        for line in (result_dir / "operations" / "calls.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {row["scope_id"] for row in scope_rows} == set(compute["scope_order"])
