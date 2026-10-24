"""Fail-closed, member-scoped NPZ reader prototype for A14 synthetic tests.

This module never guesses scientific authorization. Callers provide the registered
file paths; production inventory must exactly match the fixed 10-file/42-member
contract below. Synthetic tests may opt into a reduced private plan.
"""
from __future__ import annotations

import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from kron_reduce.data.audit_access_ledger import AccessLedger, AccessLedgerError

MAX_TEXT_CHARS = 1_048_576
EXPECTED_NUMERIC_ELEMENTS = 440_733
EXPECTED_NUMERIC_MEMBERS = 39
EXPECTED_TEXT_MEMBERS = 3
EXPECTED_MEMBER_COUNT = 42
EXPECTED_FILE_COUNT = 10


@dataclass(frozen=True)
class MemberSpec:
    key: str
    shape: tuple[int, ...]
    kind: str  # "f8" or "unicode_scalar"

    @property
    def expected_elements(self) -> int:
        if self.kind == "f8":
            return int(np.prod(self.shape, dtype=np.int64))
        return 1


@dataclass(frozen=True)
class FileSpec:
    key: str
    members: tuple[MemberSpec, ...]


def fixed_file_specs() -> tuple[FileSpec, ...]:
    """Return the immutable A14 source-layout contract from the protocol."""
    files: list[FileSpec] = []
    files.append(FileSpec("A10/real_train_arrays.npz", (
        MemberSpec("history", (258, 11, 20), "f8"),
        MemberSpec("history_time", (258, 11), "f8"),
        MemberSpec("constant_power", (258, 10), "f8"),
        MemberSpec("equilibrium", (258, 10), "f8"),
        MemberSpec("mass", (10,), "f8"),
        MemberSpec("target_scale", (20,), "f8"),
    )))
    for seed in (324, 325, 326):
        files.append(FileSpec(f"A09/seed-{seed}-h005/paths.npz", (
            MemberSpec("history", (258, 11, 22), "f8"),
        )))
    for seed in (324, 325, 326):
        files.append(FileSpec(f"A10/seed-{seed}-h0025/paths.npz", (
            MemberSpec("history", (258, 11, 22), "f8"),
        )))
    model_shapes = (
        ("mass", (10,)), ("edge_stiffness", (45,)), ("damping", (10, 10)),
        ("coupling", (10, 2)), ("latent_skew", (2, 2)),
        ("latent_dissipation", (2, 2)), ("encoder", (2, 80)),
        ("feature_scale", (80,)), ("target_scale", (20,)),
    )
    for seed in (324, 325, 326):
        members = tuple(MemberSpec(name, shape, "f8") for name, shape in model_shapes)
        members += (MemberSpec("metadata", (), "unicode_scalar"),)
        files.append(FileSpec(f"A06/checkpoint-240-{seed}/model.npz", members))
    return tuple(files)


def _validate_file_specs(files: Sequence[FileSpec], *, allow_reduced: bool) -> tuple[int, int]:
    if not isinstance(files, (tuple, list)) or not files:
        raise ValueError("file specs must be a nonempty fixed sequence")
    file_keys = [f.key for f in files]
    if len(set(file_keys)) != len(file_keys):
        raise ValueError("duplicate NPZ file selector")
    member_keys: set[tuple[str, str]] = set()
    numeric_count = text_count = elements = 0
    for file_spec in files:
        if not isinstance(file_spec, FileSpec) or not file_spec.key or not file_spec.members:
            raise ValueError("invalid NPZ file spec")
        names: set[str] = set()
        for member in file_spec.members:
            if not isinstance(member, MemberSpec) or not member.key or member.key in names:
                raise ValueError("invalid or duplicate member spec")
            names.add(member.key)
            pair = (file_spec.key, member.key)
            if pair in member_keys:
                raise ValueError("duplicate member selector")
            member_keys.add(pair)
            if member.kind == "f8":
                if not member.shape or any(type(dim) is not int or dim <= 0 for dim in member.shape):
                    raise ValueError("numeric member shape must have positive integer dimensions")
                numeric_count += 1
                elements += member.expected_elements
            elif member.kind == "unicode_scalar":
                if member.shape != ():
                    raise ValueError("metadata text must be a zero-dimensional Unicode scalar")
                text_count += 1
            else:
                raise ValueError("unsupported member kind")
    if not allow_reduced and (len(files) != EXPECTED_FILE_COUNT
                              or len(member_keys) != EXPECTED_MEMBER_COUNT
                              or numeric_count != EXPECTED_NUMERIC_MEMBERS
                              or text_count != EXPECTED_TEXT_MEMBERS
                              or elements != EXPECTED_NUMERIC_ELEMENTS):
        raise ValueError("inventory does not match the fixed A14 10-file/42-member contract")
    return numeric_count, elements


def _ledger_plan(files: Sequence[FileSpec]) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    plan: list[dict[str, Any]] = []
    metadata: dict[str, dict[str, str]] = {}
    for file_spec in files:
        for member in file_spec.members:
            base = f"{file_spec.key}::{member.key}"
            descriptor_key = base + "::descriptor"
            payload_key = base + ("::array" if member.kind == "f8" else "::text")
            plan.append({"key": descriptor_key, "phase": 1, "kind": "countdescriptor", "expected_elements": 1})
            plan.append({"key": payload_key, "phase": 1,
                         "kind": "array" if member.kind == "f8" else "attr",
                         "expected_elements": member.expected_elements})
            for key in (descriptor_key, payload_key):
                metadata[key] = {
                    "source": file_spec.key.split("/", 1)[0],
                    "event_id": "registered_cache",
                    "role": "train",
                    "dataset_or_attr": member.key,
                    "selection": "whole_registered_member",
                }
    return plan, metadata


def _read_npy_header(stream) -> tuple[tuple[int, ...], np.dtype, bool, int]:
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
    elif version == (3, 0):
        raise ValueError("NPY version 3 is outside the registered A14 format contract")
    else:
        raise ValueError(f"unsupported NPY version {version}")
    if not isinstance(shape, tuple) or any(type(x) is not int or x < 0 for x in shape):
        raise ValueError("invalid NPY shape")
    if not isinstance(dtype, np.dtype):
        raise TypeError("invalid NPY dtype")
    header_bytes = stream.tell()
    return shape, dtype, bool(fortran_order), header_bytes


class NpzWhitelistReader:
    """One-open-per-file, one-payload-read-per-member reader.

    `source_paths` keys must exactly equal registered FileSpec keys. The reader
    rejects any access after its first failure. Unregistered ZIP members are
    enumerated by name only and never opened.
    """

    def __init__(self, source_paths: Mapping[str, str | Path], *,
                 files: Sequence[FileSpec] | None = None,
                 ledger_factory: Callable[..., Any] | None = None,
                 _allow_reduced_plan_for_tests: bool = False,
                 zip_factory: Callable[..., Any] = zipfile.ZipFile):
        self.files = tuple(fixed_file_specs() if files is None else files)
        if not _allow_reduced_plan_for_tests and self.files != fixed_file_specs():
            raise ValueError("production file specs must exactly equal the fixed A14 inventory")
        _validate_file_specs(self.files, allow_reduced=_allow_reduced_plan_for_tests)
        expected_keys = {item.key for item in self.files}
        if not isinstance(source_paths, Mapping) or set(source_paths) != expected_keys:
            raise ValueError("source path keys must exactly match the registered file inventory")
        self.source_paths = {key: Path(value) for key, value in source_paths.items()}
        plan, self.metadata_by_key = _ledger_plan(self.files)
        _, logical_elements = _validate_file_specs(self.files, allow_reduced=_allow_reduced_plan_for_tests)
        if ledger_factory is None:
            self.ledger = AccessLedger(plan, logical_element_limit=logical_elements)
        else:
            self.ledger = ledger_factory(plan, logical_element_limit=logical_elements,
                                         metadata_by_key=self.metadata_by_key)
        self._zip_factory = zip_factory
        self._opened: set[str] = set()
        self._failed = False
        self._arrays: dict[tuple[str, str], np.ndarray | str] = {}
        self._stats: dict[tuple[str, str], dict[str, Any]] = {}

    def _require_live(self) -> None:
        if self._failed:
            raise AccessLedgerError("reader is fail-closed after an earlier access failure")

    def read_file(self, file_key: str) -> dict[str, np.ndarray | str]:
        self._require_live()
        spec = next((item for item in self.files if item.key == file_key), None)
        if spec is None:
            self._failed = True
            raise AccessLedgerError("unknown NPZ file selector")
        if file_key in self._opened:
            self._failed = True
            raise AccessLedgerError("NPZ file context already spent")
        self._opened.add(file_key)  # opening failure still consumes the one allowed attempt
        try:
            result: dict[str, np.ndarray | str] = {}
            with self._zip_factory(self.source_paths[file_key], "r") as archive:
                infos = archive.infolist()
                names = [info.filename for info in infos]
                if len(names) != len(set(names)):
                    raise ValueError("duplicate ZIP member names")
                info_by_name = {info.filename: info for info in infos}
                for member in spec.members:
                    zip_name = member.key + ".npy"
                    info = info_by_name.get(zip_name)
                    descriptor_key = f"{file_key}::{member.key}::descriptor"
                    payload_key = f"{file_key}::{member.key}::{'array' if member.kind == 'f8' else 'text'}"
                    stream_holder = {}
                    holder = stream_holder

                    def inspect_registered_member(info=info, member=member, zip_name=zip_name, holder=holder):
                        if info is None:
                            raise ValueError(f"registered member missing: {zip_name}")
                        if type(info.file_size) is not int or info.file_size <= 0:
                            raise ValueError("invalid ZIP member size")
                        # Opening and validating the NPY descriptor is part of the
                        # durable descriptor attempt. Keep this same stream open for
                        # the one separately accounted payload read.
                        stream = archive.open(info, "r")
                        holder["stream"] = stream
                        try:
                            shape, dtype, fortran_order, header_bytes = _read_npy_header(stream)
                            self._validate_header(member, shape, dtype)
                            elements = int(np.prod(shape, dtype=np.int64)) if shape else 1
                            payload_bytes = elements * dtype.itemsize
                            if header_bytes + payload_bytes != info.file_size:
                                raise ValueError("NPY header and payload size do not match ZIP member size")
                            # Return only JSON-safe metadata; never pass a stream or
                            # archive handle through the ledger or durable event sink.
                            return {
                                "filename": info.filename,
                                "file_size": info.file_size,
                                "compress_type": info.compress_type,
                                "shape": shape,
                                "dtype": dtype,
                                "fortran_order": fortran_order,
                                "header_bytes": header_bytes,
                                "payload_bytes": payload_bytes,
                            }
                        except BaseException:
                            stream.close()
                            holder.clear()
                            raise

                    try:
                        descriptor = self.ledger.call(
                            descriptor_key,
                            inspect_registered_member,
                            element_count=lambda value: 1,
                        )
                    except BaseException:
                        opened_stream = stream_holder.pop("stream", None)
                        if opened_stream is not None:
                            opened_stream.close()
                        raise
                    stream = stream_holder["stream"]
                    shape = descriptor["shape"]
                    dtype = descriptor["dtype"]
                    fortran_order = descriptor["fortran_order"]
                    payload_bytes = descriptor["payload_bytes"]
                    try:
                        # Exactly one bounded payload read; no trailing EOF probe.
                        raw = self.ledger.call(
                            payload_key,
                            lambda stream=stream, payload_bytes=payload_bytes: self._read_exact_payload(stream, payload_bytes),
                            element_count=(lambda value, member=member, payload_bytes=payload_bytes:
                                           member.expected_elements if len(value) == payload_bytes else None),
                        )
                    finally:
                        stream.close()
                    value, text_stats = self._decode_payload(member, dtype, shape, fortran_order, raw, info.file_size)
                    result[member.key] = value
                    self._arrays[(file_key, member.key)] = value
                    self._stats[(file_key, member.key)] = {
                        "shape": list(shape), "dtype": dtype.str, "npy_storage_bytes": info.file_size,
                        **text_stats,
                    }
            return result
        except BaseException:
            self._failed = True
            raise

    @staticmethod
    def _read_exact_payload(stream, count: int) -> bytes:
        data = stream.read(count)
        if not isinstance(data, bytes) or len(data) != count:
            raise EOFError("NPY payload shorter than registered header size")
        return data

    @staticmethod
    def _validate_header(member: MemberSpec, shape: tuple[int, ...], dtype: np.dtype) -> None:
        if shape != member.shape:
            raise ValueError(f"shape mismatch for {member.key}: {shape} != {member.shape}")
        if dtype.hasobject or dtype.fields is not None or dtype.subdtype is not None:
            raise ValueError("object or structured NPY dtype is forbidden")
        if member.kind == "f8":
            if dtype.kind != "f" or dtype.itemsize != 8 or dtype.byteorder == ">":
                raise ValueError(f"numeric member {member.key} must use little/native float64")
        elif member.kind == "unicode_scalar":
            if dtype.kind != "U" or dtype.itemsize % 4:
                raise ValueError("metadata member must be fixed-width Unicode")
            if dtype.itemsize // 4 > MAX_TEXT_CHARS:
                raise ValueError("metadata text exceeds the fixed character limit")
        else:
            raise ValueError("unsupported member kind")

    @staticmethod
    def _decode_payload(member: MemberSpec, dtype: np.dtype, shape: tuple[int, ...],
                        fortran_order: bool, raw: bytes, storage_bytes: int):
        if member.kind == "f8":
            array = np.frombuffer(raw, dtype=dtype, count=member.expected_elements)
            array = array.reshape(shape, order="F" if fortran_order else "C")
            if array.flags.writeable:
                raise AssertionError("NPZ arrays must remain read-only")
            return array, {}
        text = np.frombuffer(raw, dtype=dtype, count=1)[0]
        if not isinstance(text, str):
            raise TypeError("metadata did not decode to a Unicode string")
        utf8_bytes = text.encode("utf-8", errors="strict")
        if len(text) > MAX_TEXT_CHARS:
            raise ValueError("metadata text exceeds the fixed character limit")
        return text, {"characters": len(text), "utf8_bytes": len(utf8_bytes),
                      "storage_bytes": storage_bytes}

    def read_all(self) -> dict[str, dict[str, np.ndarray | str]]:
        result: dict[str, dict[str, np.ndarray | str]] = {}
        for spec in self.files:
            result[spec.key] = self.read_file(spec.key)
        return result

    def snapshot(self) -> dict[str, Any]:
        return {"reader_failed": self._failed, "opened_file_keys": sorted(self._opened),
                "read_member_count": len(self._arrays), "member_stats": {
                    f"{file}::{member}": value for (file, member), value in self._stats.items()},
                "ledger": self.ledger.snapshot()}
