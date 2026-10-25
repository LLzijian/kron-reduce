"""Synthetic fail-closed tests for the staged A17 saved-array reader."""
from __future__ import annotations

import hashlib
import io
import json
import os
import tokenize
import zipfile
from pathlib import Path

import numpy as np
import pytest

from kron_reduce.data.input_consistency_reader import (
    AUTH,
    RUNS,
    InputConsistencyReader,
    safe_path,
    validate_saved_plan,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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
                    expected_array_sha: str | None = None) -> tuple[Path, dict, dict[str, np.ndarray]]:
    root = tmp_path / "RUNS"
    a14 = RUNS["A14"]
    a15 = RUNS["A15"]
    archive_path = f"results/{a14}/synthetic.npz"
    values = {
        "origin": np.array([[1.0, 2.0]], dtype=np.float64),
        "future": np.array([[3.0, 4.0]], dtype=np.float64),
    }
    data = archive_bytes if archive_bytes is not None else _archive_bytes(values)
    physical = root / archive_path
    physical.parent.mkdir(parents=True, exist_ok=True)
    physical.write_bytes(data)
    metadata_path = f"results/{a14}/metadata.json"
    metadata = b'{"synthetic":true}\n'
    (root / metadata_path).write_bytes(metadata)
    freeze1_path = f"results/{a14}/origin-freeze.json"
    freeze2_path = f"results/{a15}/future-freeze.json"
    (root / freeze1_path).parent.mkdir(parents=True, exist_ok=True)
    (root / freeze1_path).write_bytes(b'{"frozen":"origin"}\n')
    (root / freeze2_path).parent.mkdir(parents=True, exist_ok=True)
    (root / freeze2_path).write_bytes(b'{"frozen":"future"}\n')

    def descriptor(name: str, array: np.ndarray) -> dict:
        return {
            "name": name,
            "dtype": member_dtype,
            "shape": member_shape if member_shape is not None else list(array.shape),
            "sha256_c_order_bytes": expected_array_sha or _sha(array.tobytes(order="C")),
        }

    plan = {
        "authorization": AUTH,
        "metadata_files": [{
            "path": metadata_path,
            "size_bytes": len(metadata),
            "sha256": _sha(metadata),
        }],
        "contexts": [
            {
                "context_id": "origin-context",
                "source": "A14",
                "phase": 1,
                "path": archive_path,
                "size_bytes": len(data),
                "sha256": _sha(data),
                "members": [descriptor("origin", values["origin"])],
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
        "freezes": {"2": freeze1_path, "3": freeze2_path},
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


def _freeze(root: Path, phase: int) -> tuple[str, str]:
    run = RUNS["A14"] if phase == 2 else RUNS["A15"]
    name = "origin-freeze.json" if phase == 2 else "future-freeze.json"
    relative = f"results/{run}/{name}"
    return relative, _sha((root / relative).read_bytes())


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
    assert documents[plan["metadata_files"][0]["path"]] == {"synthetic": True}
    with pytest.raises(PermissionError, match="cannot advance"):
        reader.advance(2, *_freeze(root, 2))
    assert reader.failed
    reader.close()

    reader = InputConsistencyReader(root, plan, tmp_path / "audit-success")
    reader.read_metadata()
    reader.integrity("before")
    first = reader.read_context("origin-context")
    assert np.array_equal(first["origin"], values["origin"])
    reader.advance(2, *_freeze(root, 2))
    second = reader.read_context("future-context")
    assert np.array_equal(second["future"], values["future"])
    reader.advance(3, *_freeze(root, 3))
    reader.integrity("after")
    snapshot = reader.finish()
    assert snapshot["failed"] is False
    assert snapshot["array_reads"] == snapshot["descriptors"] == 2
    assert snapshot["array_elements"] == 4
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
            with pytest.raises(PermissionError, match="permanently poisoned"):
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
            instance.advance(2, *_freeze(root, 2))
            instance.read_context("future-context")
            instance.advance(3, *_freeze(root, 3))
            fail_at = calls + 1
            with pytest.raises(OSError):
                instance.integrity("after")
    assert instance.failed
    with pytest.raises(PermissionError, match="permanently poisoned"):
        instance.read_context("origin-context")
    instance.close()

