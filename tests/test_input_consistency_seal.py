"""Pure synthetic tests for the post-OutputIO seal action ledger."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from kron_reduce.evaluation.input_consistency_seal import SealLedger, SealLedgerError


class _ReadPrefixThenFail:
    def __init__(self, stream):
        self.stream = stream
        self.calls = 0

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def read(self, size=-1):
        self.calls += 1
        if self.calls == 1:
            return self.stream.read(5)
        raise OSError("synthetic hash read failure after returned prefix")


class _EarlyEOF:
    def __init__(self, stream):
        self.stream = stream

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def read(self, _size=-1):
        return b""


class _ChangedSizePath:
    def __init__(self, path):
        self.path = path
        self.stat_calls = 0

    def stat(self):
        self.stat_calls += 1
        value = self.path.stat()
        if self.stat_calls == 2:
            return type("ChangedStat", (), {"st_size": value.st_size + 1})()
        return value

    def open(self, mode):
        return self.path.open(mode)


class _ShortWriter:
    def __init__(self, stream):
        self.stream = stream

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def write(self, data):
        return self.stream.write(data[:3])


class _FailFlush:
    def __init__(self, stream):
        self.stream = stream

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def flush(self):
        raise OSError("synthetic flush failure")


class _FailClose:
    def __init__(self, stream):
        self.stream = stream

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def close(self):
        self.stream.close()
        raise OSError("synthetic close failure")


def _run_dir(tmp_path):
    directory = tmp_path / "result-run"
    directory.mkdir(parents=True)
    return directory


def _expect_partial(error, ledger, expected_stage):
    assert error.input_consistency_seal_partial_record == ledger.snapshot()
    assert ledger.poisoned
    action = ledger.snapshot()["actions"][-1]
    assert action["status"] == "failed"
    assert action["stage"] == expected_stage
    return action


def test_native_hash_count_and_exclusive_json_write_account_exact_bytes(tmp_path):
    directory = _run_dir(tmp_path)
    source = directory / "source_access.jsonl"
    payload = b"first\nsecond\nlast\n"
    source.write_bytes(payload)
    original_fsync = os.fsync

    ledger = SealLedger(directory, max_actions=4)
    digest = ledger.hash_file("source_access.jsonl")
    assert digest == hashlib.sha256(payload).hexdigest()
    assert ledger.count_lines("source_access.jsonl") == 3
    written = ledger.write_json("summary.json", {"status": "synthetic", "rows": 3})
    encoded = (
        json.dumps(
            {"status": "synthetic", "rows": 3}, ensure_ascii=False, allow_nan=False, indent=2
        )
        + "\n"
    ).encode("utf-8")
    assert written == len(encoded)
    assert (directory / "summary.json").read_bytes() == encoded
    snapshot = ledger.snapshot()
    assert os.fsync is original_fsync
    assert snapshot["action_count"] == 3
    assert snapshot["transition_records"] == 6
    assert snapshot["actual_read_bytes"] == 2 * len(payload)
    assert snapshot["actual_write_bytes"] == len(encoded)
    assert snapshot["unknown_operations"] == 0
    assert snapshot["poisoned"] is False
    assert [row["status"] for row in snapshot["actions"]] == ["returned"] * 3
    assert snapshot["actions"][0]["size_bytes_before"] == len(payload)
    assert snapshot["actions"][0]["size_bytes_after"] == len(payload)
    assert snapshot["actions"][1]["ends_with_newline"] is True

    snapshot["actions"][0]["status"] = "mutated-copy"
    assert ledger.snapshot()["actions"][0]["status"] == "returned"


def test_hash_failure_retains_returned_prefix_and_unknown_read(tmp_path, monkeypatch):
    directory = _run_dir(tmp_path)
    target = directory / "manifest.json"
    target.write_bytes(b"manifest bytes beyond a short returned prefix")
    original_open = Path.open

    def open_with_prefix_failure(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        if Path(path) == target and (args and args[0] == "rb"):
            return _ReadPrefixThenFail(stream)
        return stream

    monkeypatch.setattr(Path, "open", open_with_prefix_failure)
    ledger = SealLedger(directory, max_actions=3)
    with pytest.raises(OSError, match="returned prefix") as caught:
        ledger.hash_file("manifest.json")
    action = _expect_partial(caught.value, ledger, "read")
    assert action["read_bytes_returned"] == 5
    assert action["unknown_operations"] == 1
    assert action["hash_prefix_sha256"] == hashlib.sha256(b"manif").hexdigest()
    assert ledger.snapshot()["actual_read_bytes"] == 5
    before_rejection = ledger.snapshot()
    with pytest.raises(SealLedgerError, match="permanently poisoned") as rejected:
        ledger.hash_file("manifest.json")
    assert rejected.value.input_consistency_seal_partial_record == before_rejection
    assert ledger.snapshot()["action_count"] == 1


def test_known_early_eof_and_postread_size_change_fail_without_unknown_bytes(tmp_path, monkeypatch):
    directory = _run_dir(tmp_path)
    target = directory / "body.json"
    target.write_bytes(b"complete fixture body")
    original_open = Path.open

    def early_eof_open(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        if Path(path) == target and args and args[0] == "rb":
            return _EarlyEOF(stream)
        return stream

    monkeypatch.setattr(Path, "open", early_eof_open)
    early = SealLedger(directory, max_actions=2)
    with pytest.raises(OSError, match="byte count or file size changed") as caught:
        early.hash_file("body.json")
    early_action = _expect_partial(caught.value, early, "size_validation")
    assert early_action["size_bytes_before"] == len(b"complete fixture body")
    assert early_action["size_bytes_after"] == len(b"complete fixture body")
    assert early_action["read_bytes_returned"] == 0
    assert early_action["unknown_operations"] == 0
    assert early.snapshot()["unknown_operations"] == 0

    monkeypatch.setattr(Path, "open", original_open)
    changed = SealLedger(directory, max_actions=2)
    changed_path = _ChangedSizePath(target)
    monkeypatch.setattr(changed, "_safe_path", lambda _name: changed_path)
    with pytest.raises(OSError, match="byte count or file size changed") as changed_error:
        changed.hash_file("body.json")
    changed_action = _expect_partial(changed_error.value, changed, "size_validation")
    assert changed_action["size_bytes_before"] == len(b"complete fixture body")
    assert changed_action["size_bytes_after"] == len(b"complete fixture body") + 1
    assert changed_action["read_bytes_returned"] == len(b"complete fixture body")
    assert changed_action["unknown_operations"] == 0


def test_known_short_write_is_not_counted_as_unknown_and_no_retry(tmp_path, monkeypatch):
    directory = _run_dir(tmp_path)
    original_open = Path.open

    def short_write_open(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        if Path(path).name == "summary.json" and args and args[0] == "xb":
            return _ShortWriter(stream)
        return stream

    monkeypatch.setattr(Path, "open", short_write_open)
    ledger = SealLedger(directory, max_actions=3)
    with pytest.raises(OSError, match="short write") as caught:
        ledger.write_json("summary.json", {"summary": "synthetic"})
    action = _expect_partial(caught.value, ledger, "write")
    assert action["write_bytes_returned"] == 3
    assert action["short_writes"] == 1
    assert action["unknown_operations"] == 0
    assert ledger.snapshot()["actual_write_bytes"] == 3
    assert (directory / "summary.json").read_bytes() == b"{\n "
    with pytest.raises(SealLedgerError, match="permanently poisoned"):
        ledger.write_json("summary.json", {"retry": True})
    assert ledger.snapshot()["action_count"] == 1


def test_write_fsync_failure_marks_durability_unknown_and_preserves_bytes(tmp_path, monkeypatch):
    directory = _run_dir(tmp_path)

    def failed_fsync(_file_descriptor):
        raise OSError("synthetic fsync failure")

    monkeypatch.setattr(os, "fsync", failed_fsync)
    ledger = SealLedger(directory, max_actions=2)
    with pytest.raises(OSError, match="fsync failure") as caught:
        ledger.write_json("summary.json", {"saved": "before fsync"})
    action = _expect_partial(caught.value, ledger, "fsync")
    assert action["write_bytes_returned"] == (directory / "summary.json").stat().st_size
    assert action["unknown_operations"] == 0
    assert action["durability_unknown"] is True
    assert action["error_type"] == "OSError"


@pytest.mark.parametrize(
    ("failure_proxy", "expected_stage", "expected_message"),
    [(_FailFlush, "flush", "flush failure"), (_FailClose, "close", "close failure")],
)
def test_write_flush_and_close_failures_capture_durability_uncertainty(
    tmp_path, monkeypatch, failure_proxy, expected_stage, expected_message
):
    directory = _run_dir(tmp_path)
    original_open = Path.open

    def failing_stream_open(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        if Path(path).name == "summary.json" and args and args[0] == "xb":
            return failure_proxy(stream)
        return stream

    monkeypatch.setattr(Path, "open", failing_stream_open)
    ledger = SealLedger(directory, max_actions=2)
    with pytest.raises(OSError, match=expected_message) as caught:
        ledger.write_json("summary.json", {"saved": "all returned bytes"})
    action = _expect_partial(caught.value, ledger, expected_stage)
    assert action["write_bytes_returned"] == (directory / "summary.json").stat().st_size
    assert action["durability_unknown"] is True


def test_manifest_hash_open_failure_and_link_rejection_poison_before_retry(tmp_path, monkeypatch):
    directory = _run_dir(tmp_path)
    ledger = SealLedger(directory, max_actions=3)
    ledger.write_json("artifact_hashes.json", {"synthetic.json": "0" * 64})
    original_open = Path.open
    target = directory / "artifact_hashes.json"

    def fail_manifest_open(path, *args, **kwargs):
        if Path(path) == target and args and args[0] == "rb":
            raise OSError("synthetic manifest hash open failure")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_manifest_open)
    with pytest.raises(OSError, match="manifest hash open failure") as caught:
        ledger.hash_file("artifact_hashes.json")
    action = _expect_partial(caught.value, ledger, "open")
    assert action["read_bytes_returned"] == 0
    assert action["unknown_operations"] == 0
    assert ledger.snapshot()["action_count"] == 2

    linked = _run_dir(tmp_path / "link-case")
    (linked / "junction").mkdir()
    (linked / "junction" / "member.json").write_text("{}", encoding="utf-8")
    link_ledger = SealLedger(linked, max_actions=2)
    monkeypatch.setattr(link_ledger, "_is_junction", lambda path: path.name == "junction")
    with pytest.raises(PermissionError, match="link or junction") as link_error:
        link_ledger.hash_file("junction/member.json")
    link_action = _expect_partial(link_error.value, link_ledger, "path_validation")
    assert link_action["read_operations_attempted"] == 0
    assert link_ledger.snapshot()["actual_read_bytes"] == 0


def test_cap_duplicate_identity_and_unsafe_names_fail_closed(tmp_path):
    directory = _run_dir(tmp_path)
    (directory / "one.json").write_text("{}", encoding="utf-8")
    cap_ledger = SealLedger(directory, max_actions=1)
    assert len(cap_ledger.hash_file("one.json")) == 64
    with pytest.raises(SealLedgerError, match="action cap") as cap_error:
        cap_ledger.count_lines("one.json")
    assert cap_error.value.input_consistency_seal_partial_record["action_count"] == 1
    assert cap_ledger.snapshot()["transition_records"] == 2
    assert cap_ledger.poisoned

    duplicate = SealLedger(directory, max_actions=3)
    duplicate.hash_file("one.json")
    with pytest.raises(SealLedgerError, match="duplicate seal action"):
        duplicate.hash_file("one.json")
    assert duplicate.snapshot()["action_count"] == 2
    assert duplicate.poisoned

    for unsafe_name in ("C:alternate", "nested//file.json", "nested/../file.json"):
        isolated = _run_dir(tmp_path / unsafe_name.replace("/", "_").replace(":", "_"))
        poisoned = SealLedger(isolated, max_actions=2)
        with pytest.raises(PermissionError, match="normalized relative path"):
            poisoned.hash_file(unsafe_name)
        assert poisoned.snapshot()["actions"][0]["stage"] == "path_validation"
        assert poisoned.snapshot()["actual_read_bytes"] == 0
