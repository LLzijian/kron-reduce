"""Explicit-capacity, fail-closed reader for the A15 train future transfer.

The reader opens only registered NPZ members. Each NPZ context and each member
header/payload is spent once; a failed file is never resumed or reopened.
It does not verify source file hashes, interpret labels, or run model code.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SEEDS = (324, 325, 326)
STEPS = ("0.005", "0.0025")
EXPECTED_FILES = 1559
EXPECTED_MEMBERS = 3136
EXPECTED_ELEMENTS = 11_213_485
EXPECTED_I8_ELEMENTS = 1813


class FutureTransferAccessError(RuntimeError):
    """An A15 selector, capacity, phase, or durable read failed."""


@dataclass(frozen=True)
class MemberSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str = "f8"

    @property
    def elements(self):
        return int(np.prod(self.shape, dtype=np.int64)) if self.shape else 1


@dataclass(frozen=True)
class FileSpec:
    key: str
    phase: str
    members: tuple[MemberSpec, ...]


def fixed_file_specs() -> tuple[FileSpec, ...]:
    """Construct the immutable A15 1,559-file/3,136-member plan."""
    files = [
        FileSpec(
            "A14/permitted_inputs.npz",
            "prerequisite",
            (
                MemberSpec("history", (258, 11, 20)),
                MemberSpec("history_time", (258, 11)),
                MemberSpec("constant_power", (258, 10)),
                MemberSpec("equilibrium", (258, 10)),
                MemberSpec("mass", (10,)),
                MemberSpec("target_scale", (20,)),
            ),
        )
    ]
    static = (
        MemberSpec("parameters__mass", (10,)),
        MemberSpec("parameters__edge_stiffness", (45,)),
        MemberSpec("parameters__damping", (10, 10)),
        MemberSpec("parameters__coupling", (10, 2)),
        MemberSpec("parameters__latent_skew", (2, 2)),
        MemberSpec("parameters__latent_dissipation", (2, 2)),
    )
    for seed in SEEDS:
        files.append(FileSpec(f"A14/static-{seed}.npz", "prerequisite", static))
    for seed in SEEDS:
        for index in range(258):
            for step in STEPS:
                files.append(
                    FileSpec(
                        f"A14/seed-{seed}-event-{index:03d}/history-{step}.npz",
                        "prerequisite",
                        (MemberSpec("reset", (22,)), MemberSpec("baseline_reset", (22,))),
                    )
                )
    for seed in SEEDS:
        files.append(
            FileSpec(
                f"A09/seed-{seed}-h005/paths.npz",
                "prerequisite",
                (MemberSpec("future", (72405, 22)), MemberSpec("offsets", (259,), "i8")),
            )
        )
    for seed in SEEDS:
        files.append(
            FileSpec(
                f"A10/seed-{seed}-h0025/paths.npz",
                "prerequisite",
                (MemberSpec("future", (72405, 22)), MemberSpec("offsets", (259,), "i8")),
            )
        )
    files.append(
        FileSpec(
            "A10/real_train_arrays.npz",
            "truth",
            (
                MemberSpec("future", (72405, 20)),
                MemberSpec("forecast_time", (72405,)),
                MemberSpec("offsets", (259,), "i8"),
                MemberSpec("omega_s", (258,)),
            ),
        )
    )
    return tuple(files)


def _validate_specs(specs, *, allow_reduced=False):
    specs = tuple(specs)
    if not specs or len({f.key for f in specs}) != len(specs):
        raise ValueError("file plan must be nonempty and unique")
    seen = set()
    members = elements = i8 = 0
    for f in specs:
        if not isinstance(f, FileSpec) or f.phase not in {"prerequisite", "truth"} or not f.members:
            raise ValueError("invalid A15 file specification")
        for m in f.members:
            key = (f.key, m.name)
            if key in seen or not m.name or m.dtype not in {"f8", "i8"}:
                raise ValueError("invalid or duplicate A15 member")
            seen.add(key)
            members += 1
            elements += m.elements
            if m.dtype == "i8":
                i8 += m.elements
            if any(type(d) is not int or d <= 0 for d in m.shape):
                raise ValueError("member dimensions must be positive native integers")
    if not allow_reduced and (
        specs != fixed_file_specs()
        or len(specs) != EXPECTED_FILES
        or members != EXPECTED_MEMBERS
        or elements != EXPECTED_ELEMENTS
        or i8 != EXPECTED_I8_ELEMENTS
    ):
        raise ValueError("production A15 plan differs from its exact file/member/element capacity")
    return members, elements, i8


class _FileJournal:
    """Per-file durable reservation and transition journal with range receipt."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("x", encoding="utf-8", newline="\n")
        self.path = path
        self.active = None
        self.start = None
        self.seen = set()
        self.pending = set()
        self.counters = {}
        self.spec = None
        self.expected = None
        self.actual = {
            "zip_contexts": 0,
            "headers_returned": 0,
            "payloads_returned": 0,
            "elements_returned": 0,
            "i8_elements_returned": 0,
        }
        self.closed_files = []

    def _write(self, value):
        self.stream.write(
            json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
        )
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def reserve(self, spec):
        if self.active is not None:
            raise FutureTransferAccessError("nested file reservation")
        self.start = self.stream.buffer.tell()
        self.active = spec.key
        self.seen = set()
        self.pending = set()
        self.counters = {}
        self.spec = spec
        self.expected = {
            "zip_contexts": 1,
            "members": len(spec.members),
            "elements": sum(m.elements for m in spec.members),
            "i8_elements": sum(m.elements for m in spec.members if m.dtype == "i8"),
        }
        self.actual = {
            "zip_contexts": 0,
            "headers_returned": 0,
            "payloads_returned": 0,
            "elements_returned": 0,
            "i8_elements_returned": 0,
        }
        self._write(
            {
                "kind": "file_reserved",
                "file": spec.key,
                "phase": spec.phase,
                "members": [
                    {
                        "selector": f"{spec.key}::{m.name}",
                        "name": m.name,
                        "order": i,
                        "shape": list(m.shape),
                        "dtype": m.dtype,
                        "elements": m.elements,
                        "payload_bytes": m.elements * 8,
                    }
                    for i, m in enumerate(spec.members)
                ],
                "member_count": len(spec.members),
                "element_count": sum(m.elements for m in spec.members),
                "spent": True,
                "resume": False,
            }
        )

    def transition(self, selector, operation, state, metadata=None):
        if self.active is None:
            raise FutureTransferAccessError("source call outside spent file")
        if state not in {"attempted", "returned", "failed"}:
            raise FutureTransferAccessError("unknown transition state")
        if operation == "zip_context":
            if selector != self.active:
                raise FutureTransferAccessError("unknown file selector")
            if state == "attempted" and (selector, operation) in self.seen:
                raise FutureTransferAccessError("ZIP context hard limit exceeded")
            if any(k[1] in {"header", "payload"} for k in self.seen):
                raise FutureTransferAccessError("ZIP context must precede member access")
        elif operation in {"header", "payload"}:
            if (self.active, "zip_context") not in self.seen:
                raise FutureTransferAccessError("member access attempted before ZIP context")
            if self.counters.get(("zip_context", "returned"), 0) != 1:
                raise FutureTransferAccessError("member access requires returned ZIP context")
            specs = {f"{self.active}::{m.name}": m for m in self.spec.members}
            member = specs.get(selector)
            if member is None:
                raise FutureTransferAccessError("unknown member selector")
            if operation not in {"header", "payload"}:
                raise FutureTransferAccessError("unknown operation")
            if state == "attempted" and (selector, operation) in self.seen:
                raise FutureTransferAccessError("member operation hard limit exceeded")
            members = list(specs)
            index = members.index(selector)
            if operation == "header" and state == "attempted":
                if index and (members[index - 1], "payload") not in self.seen:
                    raise FutureTransferAccessError("member order differs from reservation")
                if (selector, "payload") in self.seen:
                    raise FutureTransferAccessError("header follows payload")
            if (
                operation == "payload"
                and state == "attempted"
                and (selector, "header") not in self.seen
            ):
                raise FutureTransferAccessError("payload attempted before header")
            if (
                operation == "payload"
                and state == "attempted"
                and (selector, "header") in self.pending
            ):
                raise FutureTransferAccessError("payload attempted before terminal header")
            if state == "returned" and operation == "header":
                if (
                    not isinstance(metadata, dict)
                    or tuple(metadata.get("shape", ())) != member.shape
                ):
                    raise FutureTransferAccessError(
                        "returned header shape differs from reservation"
                    )
                if metadata.get("payload_bytes") != member.elements * 8:
                    raise FutureTransferAccessError(
                        "returned header byte count differs from reservation"
                    )
                try:
                    dtype = np.dtype(metadata.get("dtype"))
                except (TypeError, ValueError) as error:
                    raise FutureTransferAccessError("returned header dtype is invalid") from error
                expected_kind = "f" if member.dtype == "f8" else "i"
                if dtype.kind != expected_kind or dtype.itemsize != 8 or dtype.byteorder == ">":
                    raise FutureTransferAccessError(
                        "returned header dtype differs from reservation"
                    )
            if state == "returned" and operation == "payload":
                if not isinstance(metadata, dict) or metadata.get("elements") != member.elements:
                    raise FutureTransferAccessError(
                        "returned payload element count differs from reservation"
                    )
                if metadata.get("dtype") != member.dtype:
                    raise FutureTransferAccessError(
                        "returned payload dtype differs from reservation"
                    )
        else:
            raise FutureTransferAccessError("unknown operation")
        identity = (selector, operation)
        if state == "attempted":
            if identity in self.seen:
                raise FutureTransferAccessError("member operation already spent")
        elif identity not in self.pending:
            raise FutureTransferAccessError("terminal transition lacks matching attempt")
        self._write(
            {
                "kind": "member_transition",
                "file": self.active,
                "selector": selector,
                "operation": operation,
                "transition": state,
                "metadata": metadata,
            }
        )
        if state == "attempted":
            self.seen.add(identity)
            self.pending.add(identity)
        else:
            self.pending.remove(identity)
        self.counters[(operation, state)] = self.counters.get((operation, state), 0) + 1
        if operation == "zip_context" and state == "returned":
            self.actual["zip_contexts"] += 1
        if operation == "header" and state == "returned":
            self.actual["headers_returned"] += 1
        if operation == "payload" and state == "returned":
            self.actual["payloads_returned"] += 1
            self.actual["elements_returned"] += metadata["elements"]
            if metadata.get("dtype") == "i8":
                self.actual["i8_elements_returned"] += metadata["elements"]

    def close_file(self, status, reason=None):
        if self.active is None:
            raise FutureTransferAccessError("no reserved file to close")
        if self.pending:
            raise FutureTransferAccessError("cannot close file with unfinished member calls")
        if status not in {"complete", "failed"}:
            raise FutureTransferAccessError("invalid file status")
        if status == "complete" and (
            self.actual["zip_contexts"] != 1
            or self.actual["headers_returned"] != self.expected["members"]
            or self.actual["payloads_returned"] != self.expected["members"]
            or self.actual["elements_returned"] != self.expected["elements"]
            or self.actual["i8_elements_returned"] != self.expected["i8_elements"]
        ):
            raise FutureTransferAccessError(
                "complete file counters differ from its reserved selectors"
            )
        end = self.stream.buffer.tell()
        self.stream.flush()
        os.fsync(self.stream.fileno())
        with self.path.open("rb") as source:
            source.seek(self.start)
            payload = source.read(end - self.start)
        if len(payload) != end - self.start:
            raise FutureTransferAccessError("journal range truncated")
        replay = self._replay_range(payload, status)
        if replay["actual"] != self.actual or replay["counters"] != {
            f"{k[0]}:{k[1]}": v for k, v in self.counters.items()
        }:
            raise FutureTransferAccessError("durable range differs from in-memory file accounting")
        receipt = {
            "kind": "file_closed",
            "file": self.active,
            "start": self.start,
            "end": end,
            "range_sha256": hashlib.sha256(payload).hexdigest(),
            "status": status,
            "reason": reason,
            "expected": self.expected,
            "actual": self.actual,
            "counters": {f"{k[0]}:{k[1]}": v for k, v in self.counters.items()},
        }
        self._write(receipt)
        self.closed_files.append(receipt)
        self.active = None
        self.start = None

    def _replay_range(self, payload, status):
        """Rebuild member order, paired calls, and counters from durable bytes."""
        try:
            rows = [json.loads(line) for line in payload.decode("utf-8").splitlines()]
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise FutureTransferAccessError("durable file range is malformed") from error
        if not rows or rows[0].get("kind") != "file_reserved" or rows[0].get("file") != self.active:
            raise FutureTransferAccessError("durable file reservation is missing or changed")
        wanted = [
            {
                "selector": f"{self.active}::{m.name}",
                "name": m.name,
                "order": i,
                "shape": list(m.shape),
                "dtype": m.dtype,
                "elements": m.elements,
                "payload_bytes": m.elements * 8,
            }
            for i, m in enumerate(self.spec.members)
        ]
        reserved = rows[0]
        if (
            reserved.get("phase") != self.spec.phase
            or reserved.get("members") != wanted
            or reserved.get("member_count") != len(wanted)
            or reserved.get("element_count") != sum(x["elements"] for x in wanted)
            or reserved.get("spent") is not True
            or reserved.get("resume") is not False
        ):
            raise FutureTransferAccessError("durable reservation differs from fixed file plan")
        pending = set()
        seen = set()
        counters = {}
        actual = {
            "zip_contexts": 0,
            "headers_returned": 0,
            "payloads_returned": 0,
            "elements_returned": 0,
            "i8_elements_returned": 0,
        }
        zip_returned = False
        order = {item["selector"]: i for i, item in enumerate(wanted)}
        next_header = 0
        for row in rows[1:]:
            if row.get("kind") != "member_transition" or row.get("file") != self.active:
                raise FutureTransferAccessError("unexpected durable record inside file range")
            selector = row.get("selector")
            operation = row.get("operation")
            state = row.get("transition")
            identity = (selector, operation)
            if operation == "zip_context":
                if selector != self.active or (state == "attempted" and identity in seen):
                    raise FutureTransferAccessError("durable ZIP cap/order violation")
                if any(op in {"header", "payload"} for _, op in seen):
                    raise FutureTransferAccessError("durable ZIP order violation")
            elif operation in {"header", "payload"}:
                if not zip_returned:
                    raise FutureTransferAccessError(
                        "durable member access before returned ZIP context"
                    )
                if selector not in order:
                    raise FutureTransferAccessError("durable unknown member selector")
                idx = order[selector]
                if state == "attempted" and identity in seen:
                    raise FutureTransferAccessError("durable member operation repeated")
                if operation == "header" and state == "attempted" and idx != next_header:
                    raise FutureTransferAccessError("durable member order violation")
                if (
                    operation == "payload"
                    and state == "attempted"
                    and (selector, "header") not in seen
                ):
                    raise FutureTransferAccessError("durable payload lacks returned header")
                if (
                    operation == "payload"
                    and state == "attempted"
                    and (selector, "header") in pending
                ):
                    raise FutureTransferAccessError("durable payload precedes terminal header")
            else:
                raise FutureTransferAccessError("durable unknown operation")
            if state == "attempted":
                if identity in seen:
                    raise FutureTransferAccessError("durable attempt repeated")
                seen.add(identity)
                pending.add(identity)
            elif state in {"returned", "failed"}:
                if identity not in pending:
                    raise FutureTransferAccessError("durable terminal lacks attempt")
                pending.remove(identity)
                meta = row.get("metadata") or {}
                if state == "returned" and operation == "zip_context":
                    actual["zip_contexts"] += 1
                if state == "returned" and operation == "zip_context":
                    zip_returned = True
                if state == "returned" and operation == "header":
                    spec = self.spec.members[order[selector]]
                    if (
                        tuple(meta.get("shape", ())) != spec.shape
                        or meta.get("payload_bytes") != spec.elements * 8
                    ):
                        raise FutureTransferAccessError("durable header metadata differs from plan")
                    try:
                        dtype = np.dtype(meta.get("dtype"))
                    except (TypeError, ValueError) as error:
                        raise FutureTransferAccessError("bad durable dtype") from error
                    if (
                        dtype.kind != ("f" if spec.dtype == "f8" else "i")
                        or dtype.itemsize != 8
                        or dtype.byteorder == ">"
                    ):
                        raise FutureTransferAccessError("durable dtype differs from plan")
                    actual["headers_returned"] += 1
                    next_header += 1
                if state == "returned" and operation == "payload":
                    spec = self.spec.members[order[selector]]
                    if meta.get("elements") != spec.elements or meta.get("dtype") != spec.dtype:
                        raise FutureTransferAccessError(
                            "durable payload metadata differs from plan"
                        )
                    actual["payloads_returned"] += 1
                    actual["elements_returned"] += spec.elements
                    if spec.dtype == "i8":
                        actual["i8_elements_returned"] += spec.elements
            else:
                raise FutureTransferAccessError("durable unknown transition")
            counters[f"{operation}:{state}"] = counters.get(f"{operation}:{state}", 0) + 1
        if pending:
            raise FutureTransferAccessError("durable range has unfinished calls")
        if status == "complete" and (
            next_header != len(wanted)
            or actual["zip_contexts"] != 1
            or actual["payloads_returned"] != len(wanted)
        ):
            raise FutureTransferAccessError("complete durable range is missing registered members")
        return {"actual": actual, "counters": counters}

    def close(self):
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()


def _read_header(stream):
    magic = stream.read(6)
    if magic != b"\x93NUMPY":
        raise ValueError("invalid NPY magic")
    version = stream.read(2)
    if len(version) != 2:
        raise EOFError("short NPY version")
    if version == b"\x01\x00":
        length_bytes = 2
    elif version == b"\x02\x00":
        length_bytes = 4
    else:
        raise ValueError("unsupported NPY version")
    encoded = stream.read(length_bytes)
    if len(encoded) != length_bytes:
        raise EOFError("short NPY header length")
    header_length = int.from_bytes(encoded, "little")
    if header_length > 10_000:
        raise ValueError("NPY header exceeds registered safety limit")
    header = stream.read(header_length)
    if len(header) != header_length:
        raise EOFError("short NPY header")
    try:
        value = ast.literal_eval(header.decode("latin1").strip())
    except (SyntaxError, ValueError) as error:
        raise ValueError("invalid NPY header dictionary") from error
    if not isinstance(value, dict) or set(value) != {"descr", "fortran_order", "shape"}:
        raise ValueError("unexpected NPY header schema")
    shape = value["shape"]
    if not isinstance(shape, tuple) or any(type(d) is not int or d < 0 for d in shape):
        raise ValueError("invalid NPY shape")
    if type(value["fortran_order"]) is not bool:
        raise ValueError("invalid NPY memory order")
    return shape, np.dtype(value["descr"]), value["fortran_order"], 8 + length_bytes + header_length


class FutureTransferReader:
    """Read registered A15 arrays in prerequisite-then-truth phases exactly once."""

    def __init__(
        self, source_paths, *, journal_path, zip_factory=zipfile.ZipFile, before_open=None
    ):
        self._configure(
            source_paths,
            journal_path,
            fixed_file_specs(),
            zip_factory,
            allow_reduced=False,
            before_open=before_open,
        )

    def _configure(
        self, source_paths, journal_path, specs, zip_factory, *, allow_reduced, before_open=None
    ):
        self.specs = tuple(specs)
        self.member_count, self.element_count, self.i8_count = _validate_specs(
            self.specs, allow_reduced=allow_reduced
        )
        if not isinstance(source_paths, Mapping) or set(source_paths) != {
            f.key for f in self.specs
        }:
            raise ValueError("source path selectors must exactly match registered A15 files")
        self.paths = {k: Path(v) for k, v in source_paths.items()}
        self.zip_factory = zip_factory
        if before_open is not None and not callable(before_open):
            raise TypeError("before_open must be callable or None")
        self.before_open = before_open
        self.before_open_calls = []
        self.journal = _FileJournal(Path(journal_path))
        self.arrays = {}
        self.opened = set()
        self.failed = False
        self.phase1_done = False
        self.freeze_sha256 = None
        self.truth_done = False
        self.closed = False
        self._spec_by_key = {f.key: f for f in self.specs}

    @classmethod
    def _for_tests(
        cls, source_paths, *, journal_path, specs, zip_factory=zipfile.ZipFile, before_open=None
    ):
        """Private reduced-plan constructor used only by synthetic unit tests."""
        reader = cls.__new__(cls)
        reader._configure(
            source_paths,
            journal_path,
            specs,
            zip_factory,
            allow_reduced=True,
            before_open=before_open,
        )
        return reader

    def _require_live(self):
        if self.closed:
            raise FutureTransferAccessError("reader is closed")
        if self.failed:
            raise FutureTransferAccessError("reader is poisoned after an earlier failure")

    @staticmethod
    def _check_header(spec, shape, dtype):
        if shape != spec.shape:
            raise ValueError(f"shape mismatch for {spec.name}")
        if dtype.hasobject or dtype.fields is not None or dtype.subdtype is not None:
            raise ValueError("object/structured/subarray dtypes are forbidden")
        expected = "f" if spec.dtype == "f8" else "i"
        if dtype.kind != expected or dtype.itemsize != 8 or dtype.byteorder == ">":
            raise ValueError(f"{spec.name} must preserve registered little-endian {spec.dtype}")

    def _read_file(self, spec):
        if spec.key in self.opened:
            raise FutureTransferAccessError("file context already spent")
        self.opened.add(spec.key)  # even ZIP open failure consumes this file
        self.journal.reserve(spec)
        try:
            if self.before_open is not None:
                if spec.key in self.before_open_calls:
                    raise FutureTransferAccessError("before_open callback already spent for file")
                self.before_open_calls.append(spec.key)
                self.before_open(spec.key, self.paths[spec.key])
            self.journal.transition(spec.key, "zip_context", "attempted")
            try:
                archive_context = self.zip_factory(self.paths[spec.key], "r")
                self.journal.transition(spec.key, "zip_context", "returned")
            except BaseException as error:
                self.journal.transition(
                    spec.key, "zip_context", "failed", {"error_type": type(error).__name__}
                )
                raise
            with archive_context as archive:
                infos = archive.infolist()
                names = [x.filename for x in infos]
                if len(names) != len(set(names)):
                    raise ValueError("duplicate ZIP names")
                byname = {x.filename: x for x in infos}
                result = {}
                for member in spec.members:
                    selector = f"{spec.key}::{member.name}"
                    info = byname.get(member.name + ".npy")
                    stream_holder = {}

                    def inspect_header(info=info, member=member, holder=stream_holder):
                        if info is None:
                            raise ValueError(f"missing registered member {member.name}")
                        stream = archive.open(info, "r")
                        holder["stream"] = stream
                        shape, dtype, fortran, header_bytes = _read_header(stream)
                        self._check_header(member, shape, dtype)
                        count = member.elements
                        payload_bytes = count * dtype.itemsize
                        if info.file_size != header_bytes + payload_bytes:
                            raise ValueError("NPY header/payload length differs from ZIP metadata")
                        return {
                            "shape": shape,
                            "dtype": dtype.str,
                            "fortran_order": fortran,
                            "header_bytes": header_bytes,
                            "payload_bytes": payload_bytes,
                        }

                    self.journal.transition(selector, "header", "attempted")
                    try:
                        desc = inspect_header()
                        self.journal.transition(selector, "header", "returned", desc)
                    except BaseException as error:
                        self.journal.transition(
                            selector, "header", "failed", {"error_type": type(error).__name__}
                        )
                        if "stream" in stream_holder:
                            stream_holder["stream"].close()
                        raise
                    stream = stream_holder["stream"]
                    self.journal.transition(
                        selector, "payload", "attempted", {"bytes": desc["payload_bytes"]}
                    )
                    try:
                        raw = stream.read(desc["payload_bytes"])
                        if not isinstance(raw, bytes) or len(raw) != desc["payload_bytes"]:
                            raise EOFError("short NPY payload")
                        # ZipExtFile CRC is checked at EOF by the ZIP reader. The
                        # NPY size is exact; consume no unregistered member bytes.
                        value = np.frombuffer(
                            raw, dtype=np.dtype(desc["dtype"]), count=member.elements
                        )
                        value = value.reshape(
                            member.shape, order="F" if desc["fortran_order"] else "C"
                        )
                        value.setflags(write=False)
                        self.journal.transition(
                            selector,
                            "payload",
                            "returned",
                            {"elements": member.elements, "dtype": member.dtype},
                        )
                    except BaseException as error:
                        self.journal.transition(
                            selector, "payload", "failed", {"error_type": type(error).__name__}
                        )
                        raise
                    finally:
                        stream.close()
                    self.arrays[(spec.key, member.name)] = value
                    result[member.name] = value
            expected_selectors = {(spec.key, m.name) for m in spec.members}
            if not expected_selectors.issubset(self.arrays):
                raise FutureTransferAccessError("file closed without all registered members")
            self.journal.close_file("complete")
            return result
        except BaseException as error:
            try:
                if self.journal.active is not None:
                    try:
                        self.journal.close_file("failed", type(error).__name__)
                    except FutureTransferAccessError:
                        # A failed durable write may leave an unpaired in-memory
                        # attempt. Preserve the original error and leave the
                        # reservation open/incomplete; never resume it.
                        pass
            finally:
                self.failed = True
            raise

    def read_prerequisite(self):
        self._require_live()
        if self.phase1_done:
            raise FutureTransferAccessError("prerequisite phase already spent")
        if any(f.phase == "truth" for f in self.specs) and any(
            f.phase != "truth" for f in self.specs
        ):
            # The only phase-1 call may consume all non-truth sources.
            selected = [f for f in self.specs if f.phase == "prerequisite"]
        else:
            selected = [f for f in self.specs if f.phase == "prerequisite"]
        try:
            result = {f.key: self._read_file(f) for f in selected}
            self.phase1_done = True
            return result
        except BaseException:
            self.failed = True
            raise

    def seal_resets(self, freeze_sha256):
        self._require_live()
        if self.freeze_sha256 is not None:
            raise FutureTransferAccessError("reset freeze may be sealed only once")
        if (
            not self.phase1_done
            or not isinstance(freeze_sha256, str)
            or not re.fullmatch(r"[0-9a-fA-F]{64}", freeze_sha256)
        ):
            raise FutureTransferAccessError(
                "all prerequisites and a valid reset-freeze SHA are required"
            )
        self.freeze_sha256 = freeze_sha256.lower()

    def read_truth(self):
        self._require_live()
        if self.freeze_sha256 is None:
            raise FutureTransferAccessError("truth phase is locked until reset freeze")
        if self.truth_done:
            raise FutureTransferAccessError("truth phase already spent")
        try:
            specs = [f for f in self.specs if f.phase == "truth"]
            if len(specs) != 1:
                raise FutureTransferAccessError("A15 requires exactly one truth NPZ")
            result = {f.key: self._read_file(f) for f in specs}
            self.truth_done = True
            return result
        except BaseException:
            self.failed = True
            raise

    def snapshot(self):
        transitions = {}
        for receipt in self.journal.closed_files:
            for key, value in receipt.get("counters", {}).items():
                transitions[key] = transitions.get(key, 0) + value
        actual = {
            "file_contexts_returned": sum(
                x["actual"]["zip_contexts"] for x in self.journal.closed_files
            ),
            "members_header_returned": sum(
                x["actual"]["headers_returned"] for x in self.journal.closed_files
            ),
            "members_payload_returned": sum(
                x["actual"]["payloads_returned"] for x in self.journal.closed_files
            ),
            "logical_elements_returned": sum(
                x["actual"]["elements_returned"] for x in self.journal.closed_files
            ),
            "i8_elements_returned": sum(
                x["actual"]["i8_elements_returned"] for x in self.journal.closed_files
            ),
        }
        return {
            "failed": self.failed,
            "opened_files": sorted(self.opened),
            "file_count": len(self.opened),
            "expected": {
                "files": len(self.specs),
                "members": self.member_count,
                "header_calls": self.member_count,
                "payload_calls": self.member_count,
                "logical_elements": self.element_count,
                "i8_elements": self.i8_count,
            },
            "actual": actual,
            "transition_counts": transitions,
            "i8_elements_planned": self.i8_count,
            "freeze_sha256": self.freeze_sha256,
            "truth_read": self.truth_done,
            "arrays_read_only": all(not x.flags.writeable for x in self.arrays.values()),
            "file_receipts": list(self.journal.closed_files),
        }

    def close(self):
        if self.closed:
            return
        self.journal.close()
        self.closed = True
