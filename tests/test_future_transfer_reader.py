from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile

import numpy as np
import pytest

from kron_reduce.data.future_transfer_reader import (
    EXPECTED_ELEMENTS,
    EXPECTED_FILES,
    EXPECTED_I8_ELEMENTS,
    EXPECTED_MEMBERS,
    FileSpec,
    FutureTransferAccessError,
    FutureTransferReader,
    MemberSpec,
    _FileJournal,
    _read_header,
    fixed_file_specs,
)


def _small_specs():
    return (
        FileSpec(
            "A14/test-input.npz",
            "prerequisite",
            (MemberSpec("values", (3,)), MemberSpec("offsets", (2,), "i8")),
        ),
        FileSpec(
            "A10/test-truth.npz",
            "truth",
            (MemberSpec("future", (4, 2)), MemberSpec("forecast_time", (4,))),
        ),
    )


def _archives(root, *, wrong_dtype=False):
    files = {"A14/test-input.npz": root / "input.npz", "A10/test-truth.npz": root / "truth.npz"}
    np.savez(
        files["A14/test-input.npz"],
        values=np.arange(3, dtype=np.float64),
        offsets=np.array([0, 2], dtype=np.int64),
        future_sentinel=np.ones(1),
    )
    future = np.arange(8, dtype=np.float32 if wrong_dtype else np.float64).reshape(4, 2)
    np.savez(
        files["A10/test-truth.npz"],
        future=future,
        forecast_time=np.arange(4, dtype=np.float64),
        test_sentinel=np.ones(1),
    )
    return files


class _SpyZip:
    def __init__(self, counts, path, mode):
        self._zip = zipfile.ZipFile(path, mode)
        self.counts = counts

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return self._zip.__exit__(*args)

    def infolist(self):
        return self._zip.infolist()

    def open(self, info, mode="r"):
        self.counts.append(info.filename)
        if "sentinel" in info.filename:
            raise AssertionError("unregistered member opened")
        return self._zip.open(info, mode)


class _ReadSpyStream:
    def __init__(self, stream, reads):
        self.stream = stream
        self.reads = reads

    def read(self, size=-1):
        self.reads.append(size)
        return self.stream.read(size)

    def close(self):
        return self.stream.close()


class _ReadSpyZip(_SpyZip):
    def __init__(self, counts, path, mode, reads):
        super().__init__(counts, path, mode)
        self.reads = reads

    def open(self, info, mode="r"):
        return _ReadSpyStream(super().open(info, mode), self.reads)


def _reader(tmp_path, *, wrong_dtype=False):
    paths = _archives(tmp_path, wrong_dtype=wrong_dtype)
    opened = []
    reader = FutureTransferReader._for_tests(
        paths,
        journal_path=tmp_path / "journal.jsonl",
        specs=_small_specs(),
        zip_factory=lambda path, mode: _SpyZip(opened, path, mode),
    )
    return reader, opened, paths


def test_production_plan_has_exact_a15_capacity():
    specs = fixed_file_specs()
    members, elements, i8 = __import__(
        "kron_reduce.data.future_transfer_reader", fromlist=["_validate_specs"]
    )._validate_specs(specs)
    assert (len(specs), members, elements, i8) == (
        EXPECTED_FILES,
        EXPECTED_MEMBERS,
        EXPECTED_ELEMENTS,
        EXPECTED_I8_ELEMENTS,
    )
    assert sum(f.phase == "prerequisite" for f in specs) == 1558
    assert sum(f.phase == "truth" for f in specs) == 1


def test_production_constructor_rejects_reduced_plan(tmp_path):
    specs = _small_specs()
    paths = {f.key: tmp_path / f"{i}.npz" for i, f in enumerate(specs)}
    with pytest.raises(TypeError):
        FutureTransferReader(paths, journal_path=tmp_path / "j.jsonl", specs=specs)


def test_phase_order_readonly_arrays_and_unregistered_sentinel(tmp_path):
    reader, opened, _ = _reader(tmp_path)
    with pytest.raises(FutureTransferAccessError, match="freeze"):
        reader.read_truth()
    prerequisite = reader.read_prerequisite()
    values = prerequisite["A14/test-input.npz"]
    assert np.array_equal(values["values"], np.arange(3, dtype=np.float64))
    assert values["values"].flags.writeable is False
    assert set(opened) == {"values.npy", "offsets.npy"}
    with pytest.raises(FutureTransferAccessError, match="freeze"):
        reader.read_truth()
    reader.seal_resets("a" * 64)
    truth = reader.read_truth()["A10/test-truth.npz"]
    assert truth["future"].shape == (4, 2) and truth["future"].flags.writeable is False
    assert set(opened) == {"values.npy", "offsets.npy", "future.npy", "forecast_time.npy"}
    snap = reader.snapshot()
    assert snap["file_count"] == 2 and snap["actual"]["members_payload_returned"] == 4
    assert snap["actual"]["logical_elements_returned"] == 17 and snap["arrays_read_only"]
    reader.close()


def test_header_type_failure_spends_file_and_never_reads_payload(tmp_path):
    reader, opened, _ = _reader(tmp_path, wrong_dtype=True)
    reader.read_prerequisite()
    reader.seal_resets("d" * 64)
    with pytest.raises(ValueError, match="registered little-endian"):
        reader.read_truth()
    assert reader.failed is True
    assert opened[-1] == "future.npy" and "forecast_time.npy" not in opened
    with pytest.raises(FutureTransferAccessError, match="poisoned"):
        reader.read_prerequisite()
    reader.close()


def test_each_file_has_durable_reservation_calls_and_range_sha(tmp_path):
    reader, _, _ = _reader(tmp_path)
    reader.read_prerequisite()
    reader.seal_resets("b" * 64)
    reader.read_truth()
    reader.close()
    rows = [
        json.loads(x) for x in (tmp_path / "journal.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    reserves = [r for r in rows if r["kind"] == "file_reserved"]
    closes = [r for r in rows if r["kind"] == "file_closed"]
    assert len(reserves) == len(closes) == 2
    for marker in closes:
        assert marker["status"] == "complete"
        segment = (tmp_path / "journal.jsonl").read_bytes()[marker["start"] : marker["end"]]
        assert hashlib.sha256(segment).hexdigest() == marker["range_sha256"]
        assert marker["actual"]["zip_contexts"] == 1
        assert marker["actual"]["headers_returned"] == marker["actual"]["payloads_returned"]


def test_truth_failure_is_spent_and_cannot_be_retried(tmp_path):
    reader, opened, _ = _reader(tmp_path, wrong_dtype=True)
    reader.read_prerequisite()
    reader.seal_resets("c" * 64)
    with pytest.raises(ValueError, match="registered little-endian"):
        reader.read_truth()
    with pytest.raises(FutureTransferAccessError, match="poisoned"):
        reader.read_truth()
    assert opened.count("future.npy") == 1
    reader.close()


def test_freeze_sha_once_and_calls_after_close_are_rejected(tmp_path):
    reader, _, _ = _reader(tmp_path)
    reader.read_prerequisite()
    with pytest.raises(FutureTransferAccessError):
        reader.seal_resets("１２" + "a" * 62)
    reader.seal_resets("a" * 64)
    with pytest.raises(FutureTransferAccessError, match="only once"):
        reader.seal_resets("b" * 64)
    reader.close()
    with pytest.raises(FutureTransferAccessError, match="closed"):
        reader.read_truth()


def test_header_length_limit_short_header_and_v3_rejection():
    with pytest.raises(ValueError, match="safety limit"):
        _read_header(io.BytesIO(b"\x93NUMPY\x02\x00" + (10001).to_bytes(4, "little")))
    with pytest.raises(EOFError, match="short NPY header"):
        _read_header(io.BytesIO(b"\x93NUMPY\x01\x00\x0a\x00x"))
    with pytest.raises(ValueError, match="unsupported NPY version"):
        _read_header(io.BytesIO(b"\x93NUMPY\x03\x00"))


def _finish_journal_file(journal, spec):
    journal.reserve(spec)
    journal.transition(spec.key, "zip_context", "attempted")
    journal.transition(spec.key, "zip_context", "returned")
    for member in spec.members:
        selector = f"{spec.key}::{member.name}"
        metadata = {
            "shape": member.shape,
            "dtype": "<i8" if member.dtype == "i8" else "<f8",
            "payload_bytes": member.elements * 8,
        }
        journal.transition(selector, "header", "attempted")
        journal.transition(selector, "header", "returned", metadata)
        journal.transition(selector, "payload", "attempted", {"bytes": member.elements * 8})
        journal.transition(
            selector, "payload", "returned", {"elements": member.elements, "dtype": member.dtype}
        )


@pytest.mark.parametrize("corruption", ["delete_terminal", "alter_selector"])
def test_durable_range_replay_rejects_deleted_or_tampered_rows(tmp_path, corruption):
    spec = _small_specs()[0]
    journal = _FileJournal(tmp_path / "replay.jsonl")
    _finish_journal_file(journal, spec)
    journal.stream.flush()
    lines = (tmp_path / "replay.jsonl").read_text(encoding="utf-8").splitlines()
    if corruption == "delete_terminal":
        lines.pop(-1)
    else:
        row = json.loads(lines[-1])
        row["selector"] = "unknown::member"
        lines[-1] = json.dumps(row, separators=(",", ":"))
    with (tmp_path / "replay.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines) + "\n")
    journal.stream.seek(0, os.SEEK_END)
    with pytest.raises(FutureTransferAccessError):
        journal.close_file("complete")
    assert not any(
        row.get("kind") == "file_closed"
        for row in map(
            json.loads, (tmp_path / "replay.jsonl").read_text(encoding="utf-8").splitlines()
        )
    )
    journal.close()


def test_zip_open_failure_is_spent_once(tmp_path):
    paths = _archives(tmp_path)
    calls = []

    def fail_open(path, mode):
        calls.append((path, mode))
        raise OSError("synthetic open failure")

    reader = FutureTransferReader._for_tests(
        paths,
        journal_path=tmp_path / "open-fail.jsonl",
        specs=_small_specs(),
        zip_factory=fail_open,
    )
    with pytest.raises(OSError, match="synthetic"):
        reader.read_prerequisite()
    with pytest.raises(FutureTransferAccessError, match="poisoned"):
        reader.read_prerequisite()
    assert len(calls) == 1
    reader.close()


def test_duplicate_zip_names_are_rejected_before_member_open(tmp_path):
    path = tmp_path / "duplicate.npz"
    array = io.BytesIO()
    np.save(array, np.arange(3, dtype=np.float64))
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("values.npy", array.getvalue())
        archive.writestr("values.npy", array.getvalue())
        array2 = io.BytesIO()
        np.save(array2, np.array([0, 2], dtype=np.int64))
        archive.writestr("offsets.npy", array2.getvalue())
    paths = {"A14/test-input.npz": path}
    opened = []
    reader = FutureTransferReader._for_tests(
        paths,
        journal_path=tmp_path / "duplicates.jsonl",
        specs=(_small_specs()[0],),
        zip_factory=lambda p, m: _SpyZip(opened, p, m),
    )
    with pytest.raises(ValueError, match="duplicate ZIP"):
        reader.read_prerequisite()
    assert opened == []
    reader.close()


def test_crc_failure_is_recorded_and_file_cannot_be_reopened(tmp_path):
    reader, _, paths = _reader(tmp_path)
    source = paths["A14/test-input.npz"]
    with zipfile.ZipFile(source, "r") as archive:
        entries = {info.filename: archive.read(info.filename) for info in archive.infolist()}
    with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    with zipfile.ZipFile(source, "r") as archive:
        info = archive.getinfo("values.npy")
        offset = info.header_offset + 30 + len(info.filename.encode()) + len(info.extra)
    raw = bytearray(source.read_bytes())
    raw[offset + len(entries["values.npy"]) - 1] ^= 0x01
    source.write_bytes(raw)
    with pytest.raises(zipfile.BadZipFile):
        reader.read_prerequisite()
    with pytest.raises(FutureTransferAccessError, match="poisoned"):
        reader.read_prerequisite()
    assert reader.opened == {"A14/test-input.npz"}
    reader.close()


def test_journal_attempt_failure_happens_before_source_callback(tmp_path, monkeypatch):
    paths = _archives(tmp_path)
    calls = []
    reader = FutureTransferReader._for_tests(
        paths,
        journal_path=tmp_path / "write-fail.jsonl",
        specs=_small_specs(),
        zip_factory=lambda *args: calls.append("zip") or zipfile.ZipFile(*args),
    )
    original = reader.journal._write

    def fail_before_attempt(row):
        if row.get("kind") == "member_transition" and row.get("operation") == "zip_context":
            raise OSError("journal fsync failed")
        return original(row)

    monkeypatch.setattr(reader.journal, "_write", fail_before_attempt)
    with pytest.raises(OSError, match="fsync"):
        reader.read_prerequisite()
    assert calls == []
    reader.close()


@pytest.mark.parametrize("operation", ["zip_context", "header", "payload"])
def test_durable_attempt_write_failure_precedes_source_operation(tmp_path, monkeypatch, operation):
    paths = _archives(tmp_path)
    zip_calls = []
    member_opens = []
    reads = []

    def factory(path, mode):
        zip_calls.append(path)
        return _ReadSpyZip(member_opens, path, mode, reads)

    reader = FutureTransferReader._for_tests(
        {"A14/test-input.npz": paths["A14/test-input.npz"]},
        journal_path=tmp_path / f"{operation}-failure.jsonl",
        specs=(_small_specs()[0],),
        zip_factory=factory,
    )
    original = reader.journal._write

    def fail_attempt(row):
        if (
            row.get("kind") == "member_transition"
            and row.get("operation") == operation
            and row.get("transition") == "attempted"
        ):
            raise OSError(f"{operation} journal fsync failed")
        return original(row)

    monkeypatch.setattr(reader.journal, "_write", fail_attempt)
    with pytest.raises(OSError, match="journal fsync"):
        reader.read_prerequisite()
    if operation == "zip_context":
        assert zip_calls == [] and member_opens == [] and reads == []
    elif operation == "header":
        assert len(zip_calls) == 1 and member_opens == [] and reads == []
    else:
        assert len(zip_calls) == 1 and member_opens == ["values.npy"]
        assert 24 not in reads  # values payload has 3 float64 elements
    reader.close()


def test_before_open_callback_runs_once_after_reservation_and_before_zip(tmp_path):
    paths = _archives(tmp_path)
    events = []

    def callback(file_key, path):
        events.append(("callback", file_key, path))

    def factory(path, mode):
        events.append(("zip", path))
        return zipfile.ZipFile(path, mode)

    reader = FutureTransferReader._for_tests(
        paths,
        journal_path=tmp_path / "callback-order.jsonl",
        specs=_small_specs(),
        zip_factory=factory,
        before_open=callback,
    )
    reader.read_prerequisite()
    assert [event[0] for event in events] == ["callback", "zip"]
    assert reader.before_open_calls == ["A14/test-input.npz"]
    reader.seal_resets("f" * 64)
    reader.read_truth()
    assert [event[0] for event in events] == ["callback", "zip", "callback", "zip"]
    assert reader.before_open_calls == ["A14/test-input.npz", "A10/test-truth.npz"]
    reader.close()


def test_before_open_truth_callback_occurs_only_after_freeze(tmp_path):
    paths = _archives(tmp_path)
    reader_holder = {}
    observed_freeze = []

    def callback(file_key, _path):
        if file_key == "A10/test-truth.npz":
            observed_freeze.append(reader_holder["reader"].freeze_sha256)

    reader = FutureTransferReader._for_tests(
        paths,
        journal_path=tmp_path / "truth-callback.jsonl",
        specs=_small_specs(),
        before_open=callback,
    )
    reader_holder["reader"] = reader
    reader.read_prerequisite()
    assert observed_freeze == []
    reader.seal_resets("e" * 64)
    reader.read_truth()
    assert observed_freeze == ["e" * 64]
    reader.close()


def test_reservation_or_callback_failure_spends_without_opening_zip(tmp_path, monkeypatch):
    paths = _archives(tmp_path)
    calls = []
    reader = FutureTransferReader._for_tests(
        paths,
        journal_path=tmp_path / "reserve-failure.jsonl",
        specs=_small_specs(),
        zip_factory=lambda *args: calls.append("zip") or zipfile.ZipFile(*args),
        before_open=lambda *_args: calls.append("callback"),
    )

    def fail_reservation(_spec):
        raise OSError("reserve fsync failed")

    monkeypatch.setattr(reader.journal, "reserve", fail_reservation)
    with pytest.raises(OSError, match="reserve fsync"):
        reader.read_prerequisite()
    assert calls == []
    reader.close()

    callback_calls = []
    reader = FutureTransferReader._for_tests(
        paths,
        journal_path=tmp_path / "callback-failure.jsonl",
        specs=_small_specs(),
        zip_factory=lambda *args: calls.append("zip") or zipfile.ZipFile(*args),
        before_open=lambda key, _path: (
            callback_calls.append(key),
            (_ for _ in ()).throw(OSError("hash failed")),
        ),
    )
    with pytest.raises(OSError, match="hash failed"):
        reader.read_prerequisite()
    assert callback_calls == ["A14/test-input.npz"]
    assert calls == []
    with pytest.raises(FutureTransferAccessError, match="poisoned"):
        reader.read_prerequisite()
    assert callback_calls == ["A14/test-input.npz"]
    assert reader.journal.closed_files[0]["status"] == "failed"
    reader.close()
