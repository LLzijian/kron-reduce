"""Synthetic tests for bounded run-directory Path.open accounting."""

from __future__ import annotations

import io
import json
import os
import zipfile
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest

from kron_reduce.data.durable_audit_ledger import AppendOnlyAuditJournal
from kron_reduce.evaluation.input_consistency_journal import InputConsistencyJournal
from kron_reduce.evaluation.input_consistency_output_io import OutputIO, OutputIOError


class _PrefixReadFailure:
    def __init__(self, stream):
        self.stream = stream

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def read(self, size=-1):
        self.stream.read(2)
        raise OSError("synthetic read failure after an unknown prefix")


class _PrefixWriteFailure:
    def __init__(self, stream):
        self.stream = stream

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def write(self, data):
        self.stream.write(data[:2])
        raise OSError("synthetic write failure after an unknown prefix")


class _CloseFailure:
    def __init__(self, stream):
        self.stream = stream

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def close(self):
        self.stream.close()
        raise OSError("synthetic close failure")


def _run_directory(tmp_path: Path) -> Path:
    directory = tmp_path / "run"
    directory.mkdir(parents=True)
    return directory


def test_text_binary_and_numpy_zip_io_are_measured_but_external_source_passes_through(tmp_path):
    run = _run_directory(tmp_path)
    external = tmp_path / "source.bin"
    external.write_bytes(b"source-only")
    original_open, original_fsync = Path.open, os.fsync

    with OutputIO(run) as meter:
        with (run / "text.json").open("w", encoding="utf-8", newline="\n") as stream:
            assert stream.write("ábc\n") == 4
        with (run / "text.json").open("r", encoding="utf-8", newline="\n") as stream:
            assert stream.read(1) == "á"
            assert stream.readline() == "bc\n"
            assert stream.read(100) == ""

        with (run / "raw.bin").open("wb+") as stream:
            assert stream.write(b"abcdef") == 6
            stream.seek(2)
            assert stream.write(b"XY") == 2
            stream.seek(0)
            assert stream.read(6) == b"abXYef"

        with (run / "values.npz").open("wb") as stream:
            np.savez_compressed(stream, values=np.arange(7, dtype=np.float64))
        with (run / "values.npz").open("rb") as stream, zipfile.ZipFile(stream, "r") as archive:
            values = np.load(io.BytesIO(archive.read("values.npy")), allow_pickle=False)
        assert np.array_equal(values, np.arange(7, dtype=np.float64))

        with external.open("rb") as stream:
            assert stream.read() == b"source-only"
        with external.open("ab") as stream:
            os.fsync(stream.fileno())
        inside = meter.snapshot()
        assert inside["open_attempted"] == 5
        assert inside["records"] == 10
        assert inside["write_bytes_returned"] >= 5 + 8
        assert inside["read_bytes_returned"] >= 1 + 3 + 6
        assert inside["seek_calls_returned"] >= 2
        assert inside["fsync_calls_returned"] == 0

    assert Path.open is original_open
    assert os.fsync is original_fsync
    final = meter.snapshot()
    assert final["poisoned"] is False
    assert final["active_open_ids"] == []
    assert final["read_bytes_returned"] >= inside["read_bytes_returned"]

    rows = [
        json.loads(line)
        for line in (run / "output_io.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == final["records"] == 10
    assert sum(row["transition"] == "attempted" for row in rows) == 5
    assert sum(row["transition"] == "returned" for row in rows) == 5
    assert all(row["path"] != "output_io.jsonl" for row in rows)
    terminal = next(
        row for row in rows if row["path"] == "text.json" and row["transition"] == "returned"
    )
    assert terminal["write_bytes_returned"] == 5
    assert terminal["read_bytes_returned"] == 0


def test_scope_range_reader_and_reader_journal_are_measured(tmp_path):
    run = _run_directory(tmp_path)
    with OutputIO(run) as meter:
        journal = InputConsistencyJournal(run / "scope")
        journal.begin("scope-one", {"ls": 1})
        journal.call("ls", lambda: 7)
        journal.persist_and_finish(lambda: {"saved.json": "a" * 64}, status="complete")
        journal.close()

        reader_journal = AppendOnlyAuditJournal(run / "reader" / "source_access.jsonl")
        reader_journal.append({"key": "synthetic-key", "operation": "attempted", "phase": 1})
        reader_journal.append({"key": "synthetic-key", "operation": "returned", "phase": 1})
        reader_journal.close()

        snapshot = meter.snapshot()
        assert snapshot["open_attempted"] >= 5
        assert snapshot["read_bytes_returned"] > 0
        assert snapshot["write_bytes_returned"] > 0
        assert snapshot["fsync_calls_returned"] >= 3

    assert meter.snapshot()["records"] == 2 * meter.snapshot()["open_closed_returned"]
    assert (run / "scope" / "scope-one-closed.json").exists()
    assert (run / "reader" / "source_access.jsonl").exists()
    assert Path.open is not None


def test_failed_read_and_write_prefixes_are_unknown_and_poison(tmp_path):
    run = _run_directory(tmp_path)
    read_target = run / "read.bin"
    read_target.write_bytes(b"0123456789")
    with OutputIO(run) as meter:
        with pytest.raises(OSError, match="unknown prefix"), read_target.open("rb") as stream:
            stream._stream = _PrefixReadFailure(stream._stream)
            stream.read(10)
        snapshot = meter.snapshot()
        assert snapshot["poisoned"] is True
        assert snapshot["read_bytes_returned"] == 0
        assert snapshot["read_bytes_unknown_operations"] == 1
        with pytest.raises(OutputIOError, match="rejected after poison"):
            (run / "blocked.bin").open("wb")

    terminal = [json.loads(line) for line in (run / "output_io.jsonl").read_text().splitlines()]
    read_terminal = next(
        row for row in terminal if row.get("path") == "read.bin" and row["transition"] == "failed"
    )
    assert read_terminal["read_bytes_unknown_operations"] == 1

    second = _run_directory(tmp_path / "write-case")
    target = second / "write.bin"
    with OutputIO(second) as meter2:
        with pytest.raises(OSError, match="unknown prefix"), target.open("wb") as stream:
            stream._stream = _PrefixWriteFailure(stream._stream)
            stream.write(b"abcdef")
        assert meter2.snapshot()["write_bytes_returned"] == 0
        assert meter2.snapshot()["write_bytes_unknown_operations"] == 1
        assert meter2.poisoned


def test_open_symlink_fsync_and_close_failures_poison_and_restore_process_hooks(
    tmp_path, monkeypatch
):
    run = _run_directory(tmp_path)
    open_before, fsync_before = Path.open, os.fsync
    with OutputIO(run) as meter:
        with pytest.raises(FileNotFoundError):
            (run / "missing.bin").open("rb")
        assert meter.poisoned
        with pytest.raises(OutputIOError):
            (run / "later.bin").open("wb")
    assert Path.open is open_before and os.fsync is fsync_before
    rows = [json.loads(line) for line in (run / "output_io.jsonl").read_text().splitlines()]
    assert [row["transition"] for row in rows] == ["attempted", "failed", "attempted", "failed"]
    assert all(row["opened"] is False for row in rows if row["transition"] == "failed")

    linked = _run_directory(tmp_path / "link-case")
    manager = OutputIO(linked)
    monkeypatch.setattr(manager, "_is_junction", lambda path: path.name == "junction")
    with manager:
        with pytest.raises(OutputIOError, match="unsafe path"):
            (linked / "junction" / "target.json").open("wb")
        assert manager.poisoned
    linked_rows = [
        json.loads(line) for line in (linked / "output_io.jsonl").read_text().splitlines()
    ]
    assert linked_rows[0]["transition"] == "attempted" and linked_rows[1]["transition"] == "failed"

    fsync_run = _run_directory(tmp_path / "fsync-case")
    native_fsync = os.fsync

    def fail_fsync(_fd):
        raise OSError("synthetic fsync failure")

    monkeypatch.setattr(os, "fsync", fail_fsync)
    with (
        pytest.raises(OutputIOError, match="cleanup failed"),
        OutputIO(fsync_run) as fsync_meter,
        (fsync_run / "durable.bin").open("wb") as stream,
    ):
        stream.write(b"prefix")
        with pytest.raises(OSError, match="synthetic fsync"):
            os.fsync(stream.fileno())
        assert fsync_meter.poisoned
        with pytest.raises(OutputIOError, match="rejected after poison"):
            (fsync_run / "after.bin").open("wb")
    assert fsync_meter.snapshot()["journal_closed"]
    assert not fsync_meter.snapshot()["journal_durably_synced"]
    assert os.fsync is fail_fsync
    monkeypatch.setattr(os, "fsync", native_fsync)
    fsync_rows = [
        json.loads(line) for line in (fsync_run / "output_io.jsonl").read_text().splitlines()
    ]
    failed_fsync_stream = next(
        row
        for row in fsync_rows
        if row.get("path") == "durable.bin" and row["transition"] == "failed"
    )
    assert failed_fsync_stream["fsync_calls_failed"] == 1

    close_run = _run_directory(tmp_path / "close-case")
    close_manager = OutputIO(close_run)
    close_open, close_fsync = Path.open, os.fsync
    with close_manager:
        stream = (close_run / "close.bin").open("wb")
        stream._stream = _CloseFailure(stream._stream)
        stream.write(b"close-prefix")
        with pytest.raises(OSError, match="close failure"):
            stream.close()
        assert close_manager.poisoned
    assert Path.open is close_open and os.fsync is close_fsync
    close_rows = [
        json.loads(line) for line in (close_run / "output_io.jsonl").read_text().splitlines()
    ]
    assert close_rows[-1]["transition"] == "failed"


def test_transition_cap_stops_before_the_next_underlying_open(tmp_path):
    run = _run_directory(tmp_path)
    with OutputIO(run, max_records=2) as meter:
        with (run / "one.bin").open("wb") as stream:
            stream.write(b"one")
        with pytest.raises(OutputIOError, match="transition cap"):
            (run / "two.bin").open("wb")
        assert meter.snapshot()["records"] == 2
        assert meter.poisoned
        assert not (run / "two.bin").exists()


@pytest.mark.parametrize("value", [b"abcdef", "甲乙丙丁"])
def test_known_short_write_stops_with_actual_prefix_and_no_unknown_bytes(tmp_path, value):
    run = _run_directory(tmp_path)

    class ShortWriter:
        def __init__(self, stream):
            self.stream = stream

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def write(self, data):
            return self.stream.write(data[:2])

    with OutputIO(run) as meter:
        mode = "wb" if isinstance(value, bytes) else "w"
        kwargs = {} if isinstance(value, bytes) else {"encoding": "utf-8", "newline": "\n"}
        with (run / "short.bin").open(mode, **kwargs) as stream:
            stream._stream = ShortWriter(stream._stream)
            with pytest.raises(OSError, match="short or invalid count"):
                stream.write(value)
        assert meter.poisoned
        expected_bytes = 2 if isinstance(value, bytes) else len(value[:2].encode("utf-8"))
        assert meter.snapshot()["write_bytes_returned"] == expected_bytes
        assert meter.snapshot()["write_bytes_unknown_operations"] == 0
        with pytest.raises(OutputIOError):
            (run / "must-not-exist").open("wb")
    terminal = [json.loads(line) for line in (run / "output_io.jsonl").read_text().splitlines()
                if '"transition":"failed"' in line]
    assert terminal[0]["short_writes"] == 1
    assert terminal[0]["write_bytes_unknown_operations"] == 0


@pytest.mark.parametrize("own_slot", [False, True])
@pytest.mark.parametrize("fail_context", [False, True])
def test_restore_exact_class_inheritance_and_later_parent_hook(tmp_path, monkeypatch, own_slot, fail_context):
    concrete = type(Path())
    native_parent = Path.open
    inherited = object()
    calls = []

    def original_own(path, *args, **kwargs):
        return native_parent(path, *args, **kwargs)

    if own_slot:
        monkeypatch.setattr(concrete, "open", original_own)
    else:
        assert "open" not in concrete.__dict__
    before = concrete.__dict__.get("open", inherited)
    run = _run_directory(tmp_path)
    with (
        pytest.raises(RuntimeError, match="synthetic context failure") if fail_context else nullcontext(),
        OutputIO(run),
    ):
        with (run / "body.bin").open("wb") as stream:
            stream.write(b"saved")
        if fail_context:
            raise RuntimeError("synthetic context failure")
    assert concrete.__dict__.get("open", inherited) is before

    def later_parent_hook(path, *args, **kwargs):
        calls.append(path)
        return native_parent(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", later_parent_hook)
    with (run / "body.bin").open("rb") as stream:
        assert stream.read() == b"saved"
    assert bool(calls) is (not own_slot)


