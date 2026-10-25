"""Synthetic fail-closed tests for the staged A17 saved-array reader."""
from __future__ import annotations

import hashlib
import io
import json
import os
import tokenize
import zipfile
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from kron_reduce.data.input_consistency_reader import (
    AUTH,
    MAX_HEADER_BYTES,
    RUNS,
    InputConsistencyReader,
    _BoundedHeaderReader,
    safe_path,
    validate_saved_plan,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_metadata_envelope_uses_actual_first_read_bytes_and_is_immutable(tmp_path):
    root, plan, _ = _synthetic_plan(tmp_path)
    instance = InputConsistencyReader(root, plan, tmp_path / "audit-envelope")
    documents = instance.read_metadata()
    relative = plan["metadata_files"][0]["path"]
    documents[relative]["synthetic"] = False
    envelopes = instance.metadata_envelopes()
    actual = (root / relative).read_bytes()
    assert envelopes[relative]["sha256"] == _sha(actual)
    assert envelopes[relative]["size_bytes"] == len(actual)
    assert envelopes[relative]["document"]["synthetic"] is True
    with pytest.raises(TypeError):
        envelopes[relative]["document"]["synthetic"] = False
    assert envelopes[relative]["document"]["synthetic"] is True
    instance.close()
    before = dict(instance.used)
    with pytest.raises(PermissionError, match="closed"):
        instance.metadata_envelopes()
    assert instance.used == before


def _npy_bytes(value: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.save(stream, value, allow_pickle=False)
    return stream.getvalue()


def _archive_bytes(members: dict[str, np.ndarray]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, value in members.items():
            archive.writestr(name + ".npy", _npy_bytes(value))
    return stream.getvalue()


def _synthetic_plan(tmp_path: Path, *, archive_bytes: bytes | None = None,
                    member_dtype: str = "<f8", member_shape: list[int] | None = None,
                    expected_array_sha: str | None = None,
                    two_origin_members: bool = False,
                    bad_second_array_sha: bool = False,
                    metadata_count: int = 1,
                    origin_array: np.ndarray | None = None) -> tuple[Path, dict, dict[str, np.ndarray]]:
    root = tmp_path
    a14 = RUNS["A14"]
    archive_path = f"results/{a14}/synthetic.npz"
    values = {
        "origin": np.array([[1.0, 2.0]], dtype=np.float64),
        "future": np.array([[3.0, 4.0]], dtype=np.float64),
    }
    if origin_array is not None:
        values["origin"] = np.asarray(origin_array, dtype=np.float64)
    if two_origin_members:
        values["origin2"] = np.array([[5.0, 6.0]], dtype=np.float64)
    data = archive_bytes if archive_bytes is not None else _archive_bytes(values)
    physical = root / archive_path
    physical.parent.mkdir(parents=True, exist_ok=True)
    physical.write_bytes(data)
    metadata_rows = []
    for index in range(metadata_count):
        metadata_path = f"results/{a14}/metadata-{index}.json"
        metadata = json.dumps({"synthetic": True, "index": index}).encode("utf-8") + b"\n"
        (root / metadata_path).write_bytes(metadata)
        metadata_rows.append({"path": metadata_path, "size_bytes": len(metadata),
                              "sha256": _sha(metadata)})
    def descriptor(name: str, array: np.ndarray, *, bad_sha: bool = False) -> dict:
        return {
            "name": name,
            "dtype": member_dtype,
            "shape": member_shape if member_shape is not None else list(array.shape),
            "sha256_c_order_bytes": ("0" * 64 if bad_sha else
                                     expected_array_sha or _sha(array.tobytes(order="C"))),
        }

    origin_members = [descriptor("origin", values["origin"])]
    if two_origin_members:
        origin_members.append(descriptor("origin2", values["origin2"], bad_sha=bad_second_array_sha))

    plan = {
        "authorization": AUTH,
        "metadata_files": metadata_rows,
        "contexts": [
            {
                "context_id": "origin-context",
                "source": "A14",
                "phase": 1,
                "path": archive_path,
                "size_bytes": len(data),
                "sha256": _sha(data),
                "members": origin_members,
            },
            {
                "context_id": "future-context",
                "source": "A14",
                "phase": 2,
                "path": archive_path,
                "size_bytes": len(data),
                "sha256": _sha(data),
                "members": [descriptor("future", values["future"])],
            },
        ],
        "freeze_contract": {
            "max_bytes": 4096,
            "population_sha256": "a" * 64,
            "slot_ids": ["synthetic-0"],
        },
    }
    # Validate the reusable archive identity and member descriptions now.
    validate_saved_plan(plan)
    return root, plan, values


def _reader(tmp_path: Path, *, archive_bytes: bytes | None = None,
            member_dtype: str = "<f8", member_shape: list[int] | None = None,
            expected_array_sha: str | None = None) -> InputConsistencyReader:
    root, plan, _ = _synthetic_plan(
        tmp_path,
        archive_bytes=archive_bytes,
        member_dtype=member_dtype,
        member_shape=member_shape,
        expected_array_sha=expected_array_sha,
    )
    return InputConsistencyReader(root, plan, tmp_path / "audit")


def _begin(reader: InputConsistencyReader, tmp_path: Path) -> None:
    reader.read_metadata()
    reader.integrity("before")


def _freeze(reader: InputConsistencyReader, phase: int, *, status: str = "complete",
            run_id: str | None = None, kind: str | None = None,
            records: list | None = None) -> tuple[Path, str, int]:
    name = "origins_freeze.json" if phase == 2 else "futures_freeze.json"
    marker_kind = "a17_origins_freeze" if phase == 2 else "a17_futures_freeze"
    record_name = "origin-record.json" if phase == 2 else "future-record.json"
    record_bytes = json.dumps({"synthetic": phase}, separators=(",", ":")).encode("utf-8")
    (reader.directory / record_name).write_bytes(record_bytes)
    row = {"slot_id": "synthetic-0", "status": status, "record_path": record_name,
           "record_sha256": _sha(record_bytes)}
    marker = {
        "authorization": AUTH,
        "kind": kind or marker_kind,
        "run_id": run_id or reader.directory.name,
        "population_sha256": reader.plan["freeze_contract"]["population_sha256"],
        "records": records if records is not None else [row],
    }
    payload = (json.dumps(marker, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    path = reader.directory / name
    path.write_bytes(payload)
    return path, _sha(payload), len(payload)


def _ready_phase1(instance: InputConsistencyReader) -> None:
    instance.read_metadata()
    instance.integrity("before")
    instance.read_context("origin-context")


def test_three_phase_gates_and_shared_archive_members_are_read_once(tmp_path: Path) -> None:
    root, plan, values = _synthetic_plan(tmp_path)
    audit_dir = tmp_path / "audit"
    reader = InputConsistencyReader(root, plan, audit_dir)

    with pytest.raises(PermissionError, match="before-integrity"):
        reader.read_context("origin-context")
    assert reader.failed
    reader.close()

    # A fresh reader completes the full staged protocol.
    reader = InputConsistencyReader(root, plan, tmp_path / "audit-complete")
    documents = reader.read_metadata()
    assert documents[plan["metadata_files"][0]["path"]] == {"synthetic": True, "index": 0}
    with pytest.raises(PermissionError, match="cannot advance"):
        reader.advance(2, *_freeze(reader, 2))
    assert reader.failed
    reader.close()

    reader = InputConsistencyReader(root, plan, tmp_path / "audit-success")
    reader.read_metadata()
    reader.integrity("before")
    first = reader.read_context("origin-context")
    assert np.array_equal(first["origin"], values["origin"])
    reader.advance(2, *_freeze(reader, 2))
    second = reader.read_context("future-context")
    assert np.array_equal(second["future"], values["future"])
    reader.advance(3, *_freeze(reader, 3))
    reader.integrity("after")
    snapshot = reader.finish()
    assert snapshot["failed"] is False
    assert snapshot["array_reads"] == snapshot["descriptors"] == 2
    assert snapshot["array_elements"] == 4
    assert 0 < snapshot["header_bytes"] <= 2 * MAX_HEADER_BYTES
    assert snapshot["opaque_passes"] == ["after", "before"]
    transitions = [json.loads(line) for line in (tmp_path / "audit-success" / "source_access.jsonl").read_text().splitlines()]
    assert sum(row["operation"] == "opaque_hash" and row["transition"] == "returned" for row in transitions) == 2
    assert sum(row["operation"] == "array" and row["transition"] == "returned" for row in transitions) == 2


@pytest.mark.parametrize("action", ["unknown", "repeat", "wrong_phase", "repeat_hash"])
def test_unlisted_repeated_or_wrong_phase_access_poison_reader(tmp_path: Path, action: str) -> None:
    root, plan, _ = _synthetic_plan(tmp_path)
    instance = InputConsistencyReader(root, plan, tmp_path / f"audit-{action}")
    if action == "repeat_hash":
        instance.read_metadata()
    else:
        _begin(instance, tmp_path)
    if action == "unknown":
        with pytest.raises(PermissionError, match="unknown context"):
            instance.read_context("not-registered")
    elif action == "wrong_phase":
        with pytest.raises(PermissionError, match="unknown context"):
            instance.read_context("future-context")
    elif action == "repeat_hash":
        instance.integrity("before")
        with pytest.raises(PermissionError, match="invalid or repeated"):
            instance.integrity("before")
    else:
        instance.read_context("origin-context")
        with pytest.raises(PermissionError, match="already spent"):
            instance.read_context("origin-context")
    assert instance.failed
    with pytest.raises(PermissionError, match="permanently poisoned"):
        instance.read_metadata()
    instance.close()


@pytest.mark.parametrize("relative", [
    "C:/outside.npz", "results//x/file.npz", "results/./x/file.npz",
    "results/../outside.npz", "/absolute/file.npz", "results\\x\\file.npz",
])
def test_windows_and_noncanonical_paths_are_rejected(tmp_path: Path, relative: str) -> None:
    with pytest.raises(PermissionError, match="unsafe bounded source path"):
        safe_path(tmp_path, relative)


def test_linked_parent_is_rejected_before_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path.resolve()
    target = root / "results" / RUNS["A14"] / "payload.json"
    target.parent.mkdir(parents=True)
    target.write_text("{}", encoding="utf-8")
    linked_parent = target.parent
    original = Path.is_symlink

    def simulated_link(path: Path) -> bool:
        return path == linked_parent or original(path)

    monkeypatch.setattr(Path, "is_symlink", simulated_link)
    with pytest.raises(PermissionError, match="linked bounded source path"):
        safe_path(root, f"results/{RUNS['A14']}/payload.json")


def test_junction_parent_is_rejected_before_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path.resolve()
    target = root / "results" / RUNS["A14"] / "payload.json"
    target.parent.mkdir(parents=True)
    target.write_text("{}", encoding="utf-8")
    junction_parent = target.parent
    monkeypatch.setattr(Path, "is_junction", lambda path: path == junction_parent, raising=False)
    with pytest.raises(PermissionError, match="linked bounded source path"):
        safe_path(root, f"results/{RUNS['A14']}/payload.json")


@pytest.mark.parametrize("corruption", ["dtype", "shape", "sha", "truncated", "oversized", "object"])
def test_npy_descriptor_and_payload_corruption_fail_closed(tmp_path: Path, corruption: str) -> None:
    arrays = {"origin": np.array([[1.0, 2.0]]), "future": np.array([[3.0, 4.0]])}
    if corruption == "truncated":
        data = b"\x93NUMPY\x01\x00\x16\x00{'descr': '<f8', 'fortran_order': False, 'shape': (1, 2), }\n" + b"\x00" * 8
        malformed = io.BytesIO()
        with zipfile.ZipFile(malformed, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr("origin.npy", data)
            archive.writestr("future.npy", _npy_bytes(arrays["future"]))
        archive_bytes = malformed.getvalue()
        shape, dtype, expected_sha = [1, 2], "<f8", _sha(arrays["origin"].tobytes())
    elif corruption == "oversized":
        data = _archive_bytes({"origin": np.ones((100, 100)), "future": arrays["future"]})
        archive_bytes = data
        shape, dtype, expected_sha = [1, 2], "<f8", _sha(arrays["origin"].tobytes())
    elif corruption == "object":
        with pytest.raises(PermissionError, match="invalid or repeated member selector"):
            _synthetic_plan(tmp_path, member_dtype="|O")
        return
    else:
        archive_bytes = _archive_bytes(arrays)
        shape = [1, 3] if corruption == "shape" else [1, 2]
        dtype = "<i8" if corruption == "dtype" else "<f8"
        expected_sha = "0" * 64 if corruption == "sha" else _sha(arrays["origin"].tobytes())

    root, plan, _ = _synthetic_plan(
        tmp_path,
        archive_bytes=archive_bytes,
        member_dtype=dtype,
        member_shape=shape,
        expected_array_sha=expected_sha,
    )
    instance = InputConsistencyReader(root, plan, tmp_path / f"audit-{corruption}")
    _begin(instance, tmp_path)
    with pytest.raises((PermissionError, ValueError, EOFError, zipfile.BadZipFile, tokenize.TokenError)):
        instance.read_context("origin-context")
    assert instance.failed
    assert instance.array_reads == 0
    if corruption in {"dtype", "shape", "truncated"}:
        context = instance.partial_record["contexts"].get("origin-context", {})
        header = context.get("headers", {}).get("origin", {})
        assert instance.header_bytes > 0
        assert header.get("bytes_read", 0) > 0
    with pytest.raises(PermissionError, match="permanently poisoned"):
        instance.read_context("origin-context")
    instance.close()


def test_missing_source_poisoned_before_array_return(tmp_path: Path) -> None:
    root, plan, _ = _synthetic_plan(tmp_path)
    (root / plan["contexts"][0]["path"]).unlink()
    instance = InputConsistencyReader(root, plan, tmp_path / "audit-missing")
    instance.read_metadata()
    with pytest.raises(OSError):
        instance.integrity("before")
    assert instance.failed
    assert instance.array_reads == 0
    instance.close()


def test_opaque_archive_sha_mismatch_stops_before_any_member_decode(tmp_path: Path) -> None:
    root, plan, _ = _synthetic_plan(tmp_path)
    archive_path = root / plan["contexts"][0]["path"]
    raw = bytearray(archive_path.read_bytes())
    raw[-1] ^= 1
    archive_path.write_bytes(raw)
    instance = InputConsistencyReader(root, plan, tmp_path / "audit-source-sha")
    instance.read_metadata()
    with pytest.raises(PermissionError, match="opaque archive digest differs"):
        instance.integrity("before")
    assert instance.failed
    assert instance.array_reads == instance.decoded_array_elements == instance.payload_bytes_read == 0
    instance.close()


@pytest.mark.parametrize("failed_operation", ["metadata", "array", "after_hash", "close"])
def test_fsync_failures_permanently_poison_and_prevent_followup_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_operation: str,
) -> None:
    root, plan, _ = _synthetic_plan(tmp_path)
    instance = InputConsistencyReader(root, plan, tmp_path / f"audit-fsync-{failed_operation}")
    native_fsync = os.fsync
    calls = 0
    fail_at: int | None = None

    def selective_fsync(fd: int) -> None:
        nonlocal calls
        calls += 1
        if fail_at is not None and calls == fail_at:
            raise OSError("injected one-shot fsync failure")
        native_fsync(fd)

    monkeypatch.setattr(os, "fsync", selective_fsync)
    if failed_operation == "metadata":
        fail_at = 1
        with pytest.raises(OSError):
            instance.read_metadata()
    else:
        instance.read_metadata()
        if failed_operation == "close":
            fail_at = calls + 1
            with pytest.raises(OSError):
                instance.close()
            assert instance.failed
            with pytest.raises(PermissionError, match="closed|poisoned"):
                instance.read_context("origin-context")
            return
        instance.integrity("before")
        if failed_operation == "array":
            fail_at = calls + 1
            with pytest.raises(OSError):
                instance.read_context("origin-context")
            assert instance.array_reads == 0
        else:
            instance.read_context("origin-context")
            instance.advance(2, *_freeze(instance, 2))
            instance.read_context("future-context")
            instance.advance(3, *_freeze(instance, 3))
            fail_at = calls + 1
            with pytest.raises(OSError):
                instance.integrity("after")
    assert instance.failed
    with pytest.raises(PermissionError, match="permanently poisoned"):
        instance.read_context("origin-context")
    instance.close()


@pytest.mark.parametrize("position", ["after_before", "after_contexts", "before_phase3"])
def test_after_integrity_is_strictly_after_phase3_and_all_contexts(
    tmp_path: Path, position: str,
) -> None:
    root, plan, _ = _synthetic_plan(tmp_path)
    instance = InputConsistencyReader(root, plan, tmp_path / f"audit-order-{position}")
    if position == "after_before":
        instance.read_metadata()
        with pytest.raises(PermissionError, match="after-integrity requires"):
            instance.integrity("after")
        assert not any(key.startswith("hash::after") for key in instance.used)
    elif position == "after_contexts":
        instance.read_metadata()
        instance.integrity("before")
        instance.read_context("origin-context")
        instance.advance(2, *_freeze(instance, 2))
        instance.read_context("future-context")
        with pytest.raises(PermissionError, match="after-integrity requires"):
            instance.integrity("after")
        assert not any(key.startswith("hash::after") for key in instance.used)
    else:
        _ready_phase1(instance)
        instance.advance(2, *_freeze(instance, 2))
        instance.read_context("future-context")
        with pytest.raises(PermissionError, match="after-integrity requires"):
            instance.integrity("after")
        assert instance.failed
        assert not any(key.startswith("hash::after") for key in instance.used)
    assert instance.failed
    instance.close()


def test_after_integrity_rejects_repeat_and_succeeds_only_after_phase3(tmp_path: Path) -> None:
    root, plan, _ = _synthetic_plan(tmp_path)
    instance = InputConsistencyReader(root, plan, tmp_path / "audit-after-order")
    _ready_phase1(instance)
    instance.advance(2, *_freeze(instance, 2))
    instance.read_context("future-context")
    instance.advance(3, *_freeze(instance, 3))
    instance.integrity("after")
    with pytest.raises(PermissionError, match="invalid or repeated"):
        instance.integrity("after")
    assert instance.failed
    instance.close()


def test_metadata_oversize_is_rejected_by_stat_before_any_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, plan, _ = _synthetic_plan(tmp_path)
    row = plan["metadata_files"][0]
    target = root / row["path"]
    target.write_bytes(target.read_bytes() + b"excess")
    instance = InputConsistencyReader(root, plan, tmp_path / "audit-metadata-over")
    original = Path.open
    reads = []

    def monitored_open(path: Path, *args, **kwargs):
        if path == target:
            reads.append((args, kwargs))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", monitored_open)
    with pytest.raises(PermissionError, match="byte length differs before read"):
        instance.read_metadata()
    assert reads == []
    assert instance.failed
    instance.close()


def test_metadata_read_request_never_exceeds_registered_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, plan, _ = _synthetic_plan(tmp_path)
    row = plan["metadata_files"][0]
    target = root / row["path"]
    instance = InputConsistencyReader(root, plan, tmp_path / "audit-metadata-cap")
    original = Path.open
    requests = []

    class ReadMonitor:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def read(self, size=-1):
            requests.append(size)
            assert 0 <= size <= row["size_bytes"]
            return self.stream.read(size)

    def monitored_open(path: Path, *args, **kwargs):
        stream = original(path, *args, **kwargs)
        return ReadMonitor(stream) if path == target else stream

    monkeypatch.setattr(Path, "open", monitored_open)
    instance.read_metadata()
    assert requests == [row["size_bytes"]]
    instance.close()


@pytest.mark.parametrize("corruption", ["run_id", "kind", "population", "empty", "duplicate", "slot_id", "status", "missing_record"])
def test_phase_freeze_semantics_are_bound_to_run_population_and_order(
    tmp_path: Path, corruption: str,
) -> None:
    root, plan, _ = _synthetic_plan(tmp_path)
    instance = InputConsistencyReader(root, plan, tmp_path / f"audit-freeze-{corruption}")
    _ready_phase1(instance)
    marker_path, _, _ = _freeze(instance, 2)
    marker = json.loads(marker_path.read_bytes())
    if corruption == "run_id":
        marker["run_id"] = "some-old-run"
    elif corruption == "kind":
        marker["kind"] = "a17_futures_freeze"
    elif corruption == "population":
        marker["population_sha256"] = "b" * 64
    elif corruption == "empty":
        marker["records"] = []
    elif corruption == "duplicate":
        marker["records"] *= 2
    elif corruption == "slot_id":
        marker["records"][0]["slot_id"] = "unregistered-slot"
    elif corruption == "status":
        marker["records"][0]["status"] = "skipped"
    elif corruption == "missing_record":
        marker["records"][0]["record_path"] = "absent.json"
    payload = (json.dumps(marker, separators=(",", ":")) + "\n").encode()
    marker_path.write_bytes(payload)
    with pytest.raises((PermissionError, OSError)):
        instance.advance(2, marker_path, _sha(payload), len(payload))
    assert instance.failed and instance.phase == 1
    instance.close()


@pytest.mark.parametrize("status", ["numerical_failure", "prerequisite_failed"])
def test_freeze_records_retain_allowed_failure_states(tmp_path: Path, status: str) -> None:
    root, plan, _ = _synthetic_plan(tmp_path)
    instance = InputConsistencyReader(root, plan, tmp_path / f"audit-freeze-{status}")
    _ready_phase1(instance)
    marker_path, digest, size = _freeze(instance, 2, status=status)
    instance.advance(2, marker_path, digest, size)
    assert instance.phase == 2 and not instance.failed
    instance.close()


def test_freeze_size_and_path_are_checked_before_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, plan, _ = _synthetic_plan(tmp_path)
    instance = InputConsistencyReader(root, plan, tmp_path / "audit-freeze-bound")
    _ready_phase1(instance)
    marker_path, digest, size = _freeze(instance, 2)
    original = Path.open
    reads = []

    def monitored_open(path: Path, *args, **kwargs):
        if path == marker_path:
            reads.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", monitored_open)
    with pytest.raises(PermissionError, match="byte length differs before read"):
        instance.advance(2, marker_path, digest, size - 1)
    assert reads == []
    assert instance.failed and instance.phase == 1
    instance.close()


def test_freeze_rejects_any_noncanonical_marker_path(tmp_path: Path) -> None:
    root, plan, _ = _synthetic_plan(tmp_path)
    instance = InputConsistencyReader(root, plan, tmp_path / "audit-freeze-path")
    _ready_phase1(instance)
    other = instance.directory / "arbitrary.json"
    other.write_text("{}", encoding="utf-8")
    with pytest.raises(PermissionError, match="phase-specific run marker"):
        instance.advance(2, other, "0" * 64, 2)
    assert instance.failed and instance.phase == 1
    instance.close()


def test_header_proxy_rejects_over_budget_request_before_underlying_read() -> None:
    class CountedStream:
        def __init__(self):
            self.reads = []

        def read(self, size=-1):
            self.reads.append(size)
            return b"x" * size

    base = CountedStream()
    bounded = _BoundedHeaderReader(base)
    assert bounded.read(MAX_HEADER_BYTES - 1) == b"x" * (MAX_HEADER_BYTES - 1)
    with pytest.raises(PermissionError, match="fixed byte cap"):
        bounded.read(2)
    assert base.reads == [MAX_HEADER_BYTES - 1]


def test_zip_header_length_beyond_cap_is_rejected_before_header_reads(tmp_path: Path) -> None:
    # NPY v1 header length is read from its prefix; ZIP metadata advertises a
    # member larger than the registered payload plus the absolute header cap.
    header_length = MAX_HEADER_BYTES + 1
    raw = (b"\x93NUMPY\x01\x00" + header_length.to_bytes(2, "little")
           + b" " * header_length + b"\x00" * 16)
    archive_io = io.BytesIO()
    with zipfile.ZipFile(archive_io, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("origin.npy", raw)
        archive.writestr("future.npy", _npy_bytes(np.array([[3., 4.]])))
    root, plan, _ = _synthetic_plan(tmp_path, archive_bytes=archive_io.getvalue())
    instance = InputConsistencyReader(root, plan, tmp_path / "audit-large-header")
    _begin(instance, tmp_path)
    with pytest.raises(PermissionError, match="bounded header/payload size"):
        instance.read_context("origin-context")
    assert instance.failed and instance.header_bytes == 0 and instance.array_reads == 0
    instance.close()


def test_plan_is_deep_copied_and_recursively_immutable(tmp_path: Path) -> None:
    root, plan, _ = _synthetic_plan(tmp_path)
    instance = InputConsistencyReader(root, plan, tmp_path / "audit-frozen-plan")
    original_path = instance.plan["contexts"][0]["path"]
    original_shape = instance.plan["contexts"][0]["members"][0]["shape"]
    plan["contexts"][0]["path"] = "results/other/evil.npz"
    plan["contexts"][0]["members"][0]["shape"][0] = 999
    assert instance.plan["contexts"][0]["path"] == original_path
    assert instance.plan["contexts"][0]["members"][0]["shape"] == original_shape == (1, 2)
    assert isinstance(instance.plan, MappingProxyType)
    assert isinstance(instance.contexts, MappingProxyType)
    assert isinstance(instance.files, MappingProxyType)
    with pytest.raises(TypeError):
        instance.plan["contexts"][0]["path"] = "changed"
    with pytest.raises(TypeError):
        instance.contexts["other"] = {}
    instance.close()


def test_directory_outside_root_is_rejected(tmp_path: Path) -> None:
    root, plan, _ = _synthetic_plan(tmp_path)
    with pytest.raises(PermissionError, match="inside the bounded root"):
        InputConsistencyReader(root, plan, tmp_path.parent / "outside-a17-reader")


def test_closed_reader_rejects_all_access_without_mutating_selectors(tmp_path: Path) -> None:
    root, plan, _ = _synthetic_plan(tmp_path)
    instance = InputConsistencyReader(root, plan, tmp_path / "audit-closed")
    instance.close()
    before = dict(instance.used)
    path, digest, size = _freeze(instance, 2)
    operations = [
        lambda: instance.read_metadata(),
        lambda: instance.integrity("before"),
        lambda: instance.advance(2, path, digest, size),
        lambda: instance.read_context("origin-context"),
        lambda: instance.finish(),
    ]
    for operation in operations:
        with pytest.raises(PermissionError, match="closed"):
            operation()
        assert instance.used == before


def test_returned_array_survives_terminal_log_failure_with_actual_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, plan, values = _synthetic_plan(tmp_path)
    instance = InputConsistencyReader(root, plan, tmp_path / "audit-return-terminal")
    _begin(instance, tmp_path)
    original_append = instance.journal.append
    failed_once = False

    def fail_return_transition(entry):
        nonlocal failed_once
        if (not failed_once and entry.get("operation") == "array"
                and entry.get("transition") == "returned"):
            failed_once = True
            raise OSError("synthetic terminal fsync fault")
        return original_append(entry)

    monkeypatch.setattr(instance.journal, "append", fail_return_transition)
    array_key = plan["contexts"][0]["path"] + "::origin::array"
    with pytest.raises(OSError, match="terminal fsync") as caught:
        instance.read_context("origin-context")
    assert instance.used[array_key] == "function_returned"
    assert instance.array_reads == 1 and instance.array_elements == 2
    assert instance.payload_bytes_read == 16
    assert instance.decoded_array_elements == 2 and instance.decoded_array_bytes == 16
    assert np.array_equal(instance.partial_record["contexts"]["origin-context"]["arrays"]["origin"],
                          values["origin"])
    payload_state = instance.partial_record["contexts"]["origin-context"]["payloads"]["origin"]
    assert payload_state["status"] == "validated" and payload_state["validated"] is True
    serialized_callbacks = instance.snapshot()["partial_record"]["callback_returns"]
    json.dumps(serialized_callbacks, allow_nan=False)
    assert serialized_callbacks[array_key.replace("::array", "::descriptor")] == [
        [1, 2], "<f8", False, instance.header_bytes
    ]
    assert np.array_equal(caught.value.input_consistency_partial_record["partial_record"]["contexts"]
                          ["origin-context"]["arrays"]["origin"], values["origin"])
    assert instance.failed
    instance.close()


def test_first_member_is_preserved_when_later_member_fails(tmp_path: Path) -> None:
    root, plan, values = _synthetic_plan(
        tmp_path, two_origin_members=True, bad_second_array_sha=True,
    )
    instance = InputConsistencyReader(root, plan, tmp_path / "audit-member-prefix")
    _begin(instance, tmp_path)
    with pytest.raises(PermissionError, match="identity/nonfiniteness"):
        instance.read_context("origin-context")
    context = instance.partial_record["contexts"]["origin-context"]
    assert list(context["arrays"]) == ["origin"]
    assert np.array_equal(context["arrays"]["origin"], values["origin"])
    assert instance.array_reads == 1 and instance.array_elements == 2
    instance.close()


def test_header_bytes_and_parsed_header_survive_descriptor_validation_failure(tmp_path: Path) -> None:
    root, plan, _ = _synthetic_plan(tmp_path, member_shape=[1, 3])
    instance = InputConsistencyReader(root, plan, tmp_path / "audit-header-prefix")
    _begin(instance, tmp_path)
    with pytest.raises(PermissionError, match="unregistered NPY header"):
        instance.read_context("origin-context")
    context = instance.partial_record["contexts"]["origin-context"]
    header = context["headers"]["origin"]
    assert header["status"] == "rejected_descriptor"
    assert header["bytes_read"] > 0 and instance.header_bytes == header["bytes_read"]
    assert header["parsed"] == {"shape": [1, 2], "dtype": "<f8",
                                 "fortran_order": False, "header_bytes": header["bytes_read"]}
    assert instance.descriptors == 0 and instance.array_reads == 0
    snapshot_callbacks = instance.snapshot()["partial_record"]["callback_returns"]
    json.dumps(snapshot_callbacks, allow_nan=False)
    descriptor_key = plan["contexts"][0]["path"] + "::origin::descriptor"
    assert descriptor_key not in snapshot_callbacks
    instance.close()


def test_array_sha_mismatch_keeps_decoded_unvalidated_values_and_counts(tmp_path: Path) -> None:
    root, plan, values = _synthetic_plan(tmp_path, expected_array_sha="0" * 64)
    instance = InputConsistencyReader(root, plan, tmp_path / "audit-array-sha")
    _begin(instance, tmp_path)
    with pytest.raises(PermissionError, match="array identity"):
        instance.read_context("origin-context")
    context = instance.partial_record["contexts"]["origin-context"]
    assert np.array_equal(context["actual_decoded_arrays"]["origin"], values["origin"])
    assert context["arrays"] == {}
    state = context["payloads"]["origin"]
    assert state["status"] == "array_sha_mismatch"
    assert state["decoded"] is True and state["validated"] is False
    assert state["actual_bytes"] == 16 and state["actual_sha256_c_order_bytes"] != "0" * 64
    assert instance.payload_bytes_read == instance.decoded_array_bytes == 16
    assert instance.decoded_array_elements == 2
    assert instance.array_reads == instance.array_elements == 0
    instance.close()


def test_nonfinite_decoded_float_array_is_kept_as_unvalidated_failure(tmp_path: Path) -> None:
    values = np.array([[np.nan, 0.0]], dtype=np.float64)
    root, plan, _ = _synthetic_plan(tmp_path, origin_array=values)
    instance = InputConsistencyReader(root, plan, tmp_path / "audit-nonfinite-array")
    _begin(instance, tmp_path)
    with pytest.raises(PermissionError, match="nonfinite values"):
        instance.read_context("origin-context")
    context = instance.partial_record["contexts"]["origin-context"]
    assert np.isnan(context["actual_decoded_arrays"]["origin"][0, 0])
    assert context["payloads"]["origin"]["status"] == "nonfinite_values"
    assert context["payloads"]["origin"]["decoded"] is True
    assert context["payloads"]["origin"]["validated"] is False
    assert instance.decoded_array_elements == 2 and instance.array_reads == 0
    instance.close()


def test_metadata_prefix_and_returned_document_survive_terminal_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, plan, _ = _synthetic_plan(tmp_path, metadata_count=2)
    instance = InputConsistencyReader(root, plan, tmp_path / "audit-metadata-prefix")
    original_append = instance.journal.append
    failed_once = False

    def fail_first_return(entry):
        nonlocal failed_once
        if (not failed_once and entry.get("operation") == "metadata"
                and entry.get("transition") == "returned"):
            failed_once = True
            raise OSError("synthetic metadata terminal fault")
        return original_append(entry)

    monkeypatch.setattr(instance.journal, "append", fail_first_return)
    with pytest.raises(OSError, match="metadata terminal"):
        instance.read_metadata()
    first = plan["metadata_files"][0]["path"]
    assert instance.used["metadata::" + first] == "function_returned"
    assert instance.partial_record["metadata_documents"][first] == {"synthetic": True, "index": 0}
    assert instance.failed
    instance.close()


def test_metadata_prefix_is_kept_when_a_later_document_fails(tmp_path: Path) -> None:
    root, plan, _ = _synthetic_plan(tmp_path, metadata_count=2)
    second = plan["metadata_files"][1]
    (root / second["path"]).write_bytes(b'{"changed":true}\n')
    instance = InputConsistencyReader(root, plan, tmp_path / "audit-metadata-second-failure")
    with pytest.raises(PermissionError, match="source metadata byte length differs"):
        instance.read_metadata()
    first = plan["metadata_files"][0]["path"]
    assert instance.partial_record["metadata_documents"][first] == {"synthetic": True, "index": 0}
    assert instance.used["metadata::" + first] == "returned"
    assert instance.used["metadata::" + second["path"]] == "failed"
    assert instance.failed
    instance.close()


def test_original_exception_and_partial_survive_failed_failure_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, plan, _ = _synthetic_plan(tmp_path)
    instance = InputConsistencyReader(root, plan, tmp_path / "audit-original-exception")
    original_append = instance.journal.append
    original_partial = {"completed_prefix": [1, 2]}
    original_error = FloatingPointError("synthetic source arithmetic failure")
    original_error.partial_record = original_partial

    def fail_failure_transition(entry):
        if entry.get("transition") == "failed":
            raise OSError("synthetic failure-log error")
        return original_append(entry)

    monkeypatch.setattr(instance.journal, "append", fail_failure_transition)

    def fail_callback():
        raise original_error

    with pytest.raises(FloatingPointError, match="source arithmetic") as caught:
        instance._call("synthetic::callback", "synthetic", fail_callback)
    assert caught.value is original_error
    assert caught.value.partial_record is original_partial
    assert caught.value.input_consistency_log_error["type"] == "OSError"
    assert caught.value.input_consistency_partial_record["selectors"]["synthetic::callback"] == "failed"
    assert instance.failed
    instance.close()

