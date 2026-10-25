"""Finite, staged A17 saved-array reads with durable single-use selectors.

This module cannot discover datasets or read a general provider. The dedicated
workflow must verify its exact reviewed plan and spend its budget first.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np

from kron_reduce.data.durable_audit_ledger import AppendOnlyAuditJournal
from kron_reduce.data.limited_npz_audit import _read_npy_header

AUTH = "kiki_a17_development_20261025"
RUNS = {
    "A14": "paper_kiki_a14_integral_history-history_mechanism-20261024T121439628465Z-34de0389-4cf0648f",
    "A15": "paper_kiki_a15_future_transfer-future_transfer-20261024T145313076879Z-f23e9b23-c2066d4e",
    "A16": "paper_kiki_a16_pm_impulse-input_impulse-20261025T023255684964Z-45a2d989-3cb2e832",
}


def safe_path(root, relative):
    """Reject alternate streams, traversal, links and junctions before reads."""
    root = Path(root).resolve(strict=True)
    if (not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative
            or relative.startswith("/") or "//" in relative
            or any(part in {"", ".", ".."} for part in relative.split("/"))):
        raise PermissionError("unsafe bounded source path")
    target = Path(root).joinpath(*PurePosixPath(relative).parts)
    for component in (target, *target.parents):
        if component == root:
            break
        if component.is_symlink() or (hasattr(component, "is_junction") and component.is_junction()):
            raise PermissionError("linked bounded source path")
    target.resolve(strict=True).relative_to(Path(root).resolve(strict=True))
    if not target.is_file():
        raise PermissionError("bounded source is not a file")
    return target


def validate_saved_plan(plan):
    """Validate metadata without opening any source file."""
    if plan.get("authorization") != AUTH:
        raise PermissionError("wrong input-consistency authorization")
    contexts, paths, selectors = set(), {}, set()
    for row in plan["contexts"]:
        identity, source, phase, path = row["context_id"], row["source"], row["phase"], row["path"]
        if (identity in contexts or source not in RUNS or type(phase) is not int
                or phase not in {1, 2, 3} or not path.startswith(f"results/{RUNS[source]}/")
                or not path.endswith(".npz") or not row["members"]):
            raise PermissionError("invalid frozen context identity")
        contexts.add(identity)
        file_identity = (row["size_bytes"], row["sha256"])
        if (type(row["size_bytes"]) is not int or row["size_bytes"] <= 0
                or re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is None
                or (path in paths and paths[path] != file_identity)):
            raise PermissionError("inconsistent bounded archive identity")
        paths[path] = file_identity
        for member in row["members"]:
            pair = (path, member["name"])
            if (pair in selectors or not member["name"] or "/" in member["name"]
                    or "\\" in member["name"] or member["dtype"] not in {"<f8", "<i8"}
                    or not isinstance(member["shape"], list)
                    or any(type(n) is not int or n <= 0 for n in member["shape"])
                    or re.fullmatch(r"[0-9a-f]{64}", member["sha256_c_order_bytes"]) is None):
                raise PermissionError("invalid or repeated member selector")
            selectors.add(pair)
    metadata_paths = set()
    for row in plan["metadata_files"]:
        if (row["path"] in metadata_paths or not row["path"].endswith(".json")
                or type(row["size_bytes"]) is not int or row["size_bytes"] <= 0
                or re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is None):
            raise PermissionError("invalid metadata inventory")
        metadata_paths.add(row["path"])
    return plan


class InputConsistencyReader:
    """Read every planned member once, at its frozen phase, without retries."""

    def __init__(self, root, plan, directory):
        self.root = Path(root).resolve(strict=True)
        self.plan = validate_saved_plan(plan)
        self.phase = 1
        self.failed = False
        self.journal = AppendOnlyAuditJournal(Path(directory) / "source_access.jsonl")
        self.contexts = {row["context_id"]: row for row in plan["contexts"]}
        self.files = {row["path"]: row for row in plan["contexts"]}
        self.used = {}
        self.hash_passes = set()
        self.metadata_done = False
        self.before_done = False
        self.array_elements = 0
        self.descriptors = 0
        self.array_reads = 0

    def _check(self):
        if self.failed:
            raise PermissionError("input-consistency reader is permanently poisoned")

    def _call(self, key, operation, function, *, details=None):
        self._check()
        if key in self.used:
            self.failed = True
            raise PermissionError("bounded source selector already spent")
        self.used[key] = "attempted"
        try:
            self.journal.append({"key": key, "operation": operation, "phase": self.phase,
                                 "transition": "attempted", "details": details})
            result = function()
            self.used[key] = "returned"
            self.journal.append({"key": key, "operation": operation, "phase": self.phase,
                                 "transition": "returned", "details": details})
            return result
        except BaseException as error:
            self.failed = True
            self.used[key] = "failed"
            self.journal.append({"key": key, "operation": operation, "phase": self.phase,
                                 "transition": "failed", "error_type": type(error).__name__,
                                 "message": str(error), "details": details})
            raise

    def read_metadata(self):
        self._check()
        if self.phase != 1 or self.metadata_done:
            self.failed = True
            raise PermissionError("source metadata phase is already spent")
        documents = {}
        for row in self.plan["metadata_files"]:
            def read(row=row):
                path = safe_path(self.root, row["path"])
                raw = path.read_bytes()
                if len(raw) != row["size_bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
                    raise PermissionError("source metadata identity differs")
                return json.loads(raw)
            documents[row["path"]] = self._call("metadata::" + row["path"], "metadata", read,
                                                details={"size_bytes": row["size_bytes"]})
        self.metadata_done = True
        return documents

    def integrity(self, position):
        self._check()
        if position not in {"before", "after"} or position in self.hash_passes:
            self.failed = True
            raise PermissionError("invalid or repeated opaque integrity pass")
        if not self.metadata_done or (position == "before" and self.phase != 1):
            self.failed = True
            raise PermissionError("integrity pass precedes its source metadata")
        for relative, row in self.files.items():
            def verify(relative=relative, row=row):
                path = safe_path(self.root, relative)
                if path.stat().st_size != row["size_bytes"]:
                    raise PermissionError("archive byte length differs")
                digest, returned = hashlib.sha256(), 0
                with path.open("rb") as stream:
                    while returned < row["size_bytes"]:
                        block = stream.read(min(1048576, row["size_bytes"] - returned))
                        if not block:
                            break
                        returned += len(block)
                        digest.update(block)
                if (returned != row["size_bytes"] or path.stat().st_size != row["size_bytes"]
                        or digest.hexdigest() != row["sha256"]):
                    raise PermissionError("opaque archive digest differs")
            self._call(f"hash::{position}::{relative}", "opaque_hash", verify,
                       details={"size_bytes": row["size_bytes"]})
        self.hash_passes.add(position)
        if position == "before":
            self.before_done = True

    def advance(self, phase, freeze_path, freeze_sha256):
        """A saved, hash-bound population freeze is mandatory before each phase."""
        self._check()
        if type(phase) is not int or phase != self.phase + 1 or phase not in {2, 3}:
            self.failed = True
            raise PermissionError("invalid source phase transition")
        pending = [key for key, row in self.contexts.items()
                   if row["phase"] == self.phase and self.used.get("context::" + key) != "returned"]
        if pending:
            self.failed = True
            raise PermissionError("cannot advance past an unread registered context")

        def freeze():
            path = safe_path(self.root, freeze_path)
            if hashlib.sha256(path.read_bytes()).hexdigest() != freeze_sha256:
                raise PermissionError("candidate freeze digest differs")
        self._call(f"phase::{phase}", "candidate_freeze", freeze,
                   details={"path": freeze_path, "sha256": freeze_sha256})
        self.phase = phase

    def read_context(self, identity):
        self._check()
        row = self.contexts.get(identity)
        if row is None or row["phase"] != self.phase or not self.before_done:
            self.failed = True
            raise PermissionError("unknown context, wrong phase, or missing before-integrity pass")

        def read():
            target = safe_path(self.root, row["path"])
            if target.stat().st_size != row["size_bytes"]:
                raise PermissionError("bounded archive changed size")
            arrays = {}
            with zipfile.ZipFile(target) as archive:
                names = archive.namelist()
                if len(names) != len(set(names)):
                    raise PermissionError("duplicate ZIP members are forbidden")
                for member in row["members"]:
                    key = row["path"] + "::" + member["name"]
                    zipped = member["name"] + ".npy"
                    with archive.open(zipped) as stream:
                        def header(member=member, zipped=zipped, stream=stream, archive=archive):
                            shape, dtype, fortran, length = _read_npy_header(stream)
                            if (list(shape) != member["shape"] or dtype.str != member["dtype"]
                                    or dtype.hasobject or archive.getinfo(zipped).file_size
                                    != length + math.prod(shape) * dtype.itemsize):
                                raise PermissionError("unregistered NPY header or byte count")
                            return shape, dtype, fortran
                        shape, dtype, fortran = self._call(key + "::descriptor", "descriptor", header)
                        self.descriptors += 1

                        def payload(shape=shape, dtype=dtype, fortran=fortran,
                                    stream=stream, member=member):
                            count = math.prod(shape)
                            raw = stream.read(count * dtype.itemsize)
                            if len(raw) != count * dtype.itemsize:
                                raise PermissionError("truncated registered array")
                            value = np.frombuffer(raw, dtype=dtype).reshape(shape, order="F" if fortran else "C")
                            if (not np.isfinite(value).all() or hashlib.sha256(value.tobytes(order="C")).hexdigest()
                                    != member["sha256_c_order_bytes"]):
                                raise PermissionError("registered array identity/nonfiniteness differs")
                            value.flags.writeable = False
                            return value
                        arrays[member["name"]] = self._call(key + "::array", "array", payload,
                                                           details={"elements": math.prod(shape)})
                        self.array_elements += math.prod(shape)
                        self.array_reads += 1
            return arrays
        return self._call("context::" + identity, "npz_context", read,
                          details={"path": row["path"], "members": len(row["members"])})

    def snapshot(self):
        return {"phase": self.phase, "failed": self.failed, "selectors": dict(self.used),
                "array_reads": self.array_reads, "descriptors": self.descriptors,
                "array_elements": self.array_elements, "opaque_passes": sorted(self.hash_passes)}

    def finish(self):
        self._check()
        if (self.phase != 3 or any(self.used.get("context::" + key) != "returned" for key in self.contexts)
                or self.hash_passes != {"before", "after"}):
            self.failed = True
            raise PermissionError("bounded source inventory not fully consumed")
        self.close()
        return self.snapshot()

    def close(self):
        try:
            self.journal.close()
        except BaseException:
            self.failed = True
            raise
