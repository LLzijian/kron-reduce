"""Finite, staged A17 saved-array reads with durable single-use selectors."""
from __future__ import annotations

import hashlib
import json
import math
import re
import zipfile
from pathlib import Path, PurePosixPath
from types import MappingProxyType

import numpy as np

from kron_reduce.data.durable_audit_ledger import AppendOnlyAuditJournal
from kron_reduce.data.limited_npz_audit import _read_npy_header

AUTH = "kiki_a17_development_20261025"
MAX_HEADER_BYTES = 10_000
MAX_FREEZE_BYTES = 1_048_576
RUNS = {
    "A14": "paper_kiki_a14_integral_history-history_mechanism-20261024T121439628465Z-34de0389-4cf0648f",
    "A15": "paper_kiki_a15_future_transfer-future_transfer-20261024T145313076879Z-f23e9b23-c2066d4e",
    "A16": "paper_kiki_a16_pm_impulse-input_impulse-20261025T023255684964Z-45a2d989-3cb2e832",
}


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


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
    target.resolve(strict=True).relative_to(root)
    if not target.is_file():
        raise PermissionError("bounded source is not a file")
    return target


class _BoundedHeaderReader:
    """Do not let NumPy request more bytes than the frozen header cap."""

    def __init__(self, stream, limit=MAX_HEADER_BYTES):
        self.stream = stream
        self.limit = limit
        self.bytes_read = 0
        self.read_failure_unknown = False

    def read(self, size=-1):
        if type(size) is not int or size < 0 or size > self.limit - self.bytes_read:
            raise PermissionError("NPY header read exceeds the fixed byte cap")
        try:
            value = self.stream.read(size)
        except BaseException:
            self.read_failure_unknown = True
            raise
        self.bytes_read += len(value)
        return value

    def tell(self):
        return self.bytes_read


def _plain_record(value):
    """Convert callback descriptors to serializer-safe JSON primitives."""
    if isinstance(value, np.dtype):
        return value.str
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": value.dtype.str,
                "nbytes": int(value.nbytes)}
    if isinstance(value, np.generic):
        return _plain_record(value.item())
    if isinstance(value, dict):
        return {str(key): _plain_record(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_record(item) for item in value]
    return value


def validate_saved_plan(plan):
    """Validate JSON-compatible metadata without opening any source file."""
    if not isinstance(plan, dict) or plan.get("authorization") != AUTH:
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
    contract = plan["freeze_contract"]
    if (not isinstance(contract, dict) or type(contract.get("max_bytes")) is not int
            or not 0 < contract["max_bytes"] <= MAX_FREEZE_BYTES
            or re.fullmatch(r"[0-9a-f]{64}", contract.get("population_sha256", "")) is None
            or not isinstance(contract.get("slot_ids"), list) or not contract["slot_ids"]
            or any(not isinstance(slot, str) or not slot for slot in contract["slot_ids"])
            or len(set(contract["slot_ids"])) != len(contract["slot_ids"])):
        raise PermissionError("invalid population freeze contract")
    return plan


class InputConsistencyReader:
    """Read every planned member once, at its frozen phase, without retries."""

    def __init__(self, root, plan, directory):
        self.root = Path(root).resolve(strict=True)
        frozen_plan = json.loads(json.dumps(plan, ensure_ascii=False, allow_nan=False))
        validate_saved_plan(frozen_plan)
        self.plan = _freeze(frozen_plan)
        self.directory = Path(directory).resolve(strict=False)
        try:
            self.directory.relative_to(self.root)
        except ValueError as error:
            raise PermissionError("audit directory must remain inside the bounded root") from error
        self.phase = 1
        self.failed = False
        self.closed = False
        self.journal = AppendOnlyAuditJournal(self.directory / "source_access.jsonl")
        contexts = {row["context_id"]: row for row in self.plan["contexts"]}
        files = {row["path"]: row for row in self.plan["contexts"]}
        self.contexts = MappingProxyType(contexts)
        self.files = MappingProxyType(files)
        self.used = {}
        self.hash_passes = set()
        self.metadata_done = False
        self.before_done = False
        self.array_elements = 0
        self.descriptors = 0
        self.array_reads = 0
        self.header_bytes = 0
        self.payload_bytes_read = 0
        self.decoded_array_elements = 0
        self.decoded_array_bytes = 0
        self.callback_return_values = {}
        self.partial_record = {"metadata_documents": {}, "metadata_byte_returns": {},
                               "contexts": {}, "callback_returns": {}}
        self._metadata_envelopes = None

    def _check(self):
        if self.closed:
            raise PermissionError("input-consistency reader is closed")
        if self.failed:
            raise PermissionError("input-consistency reader is permanently poisoned")

    @staticmethod
    def _attach_partial(error, partial):
        error.input_consistency_partial_record = partial
        if not hasattr(error, "partial_record"):
            error.partial_record = partial

    def _call(self, key, operation, function, *, details=None, on_return=None):
        self._check()
        if key in self.used:
            self.failed = True
            raise PermissionError("bounded source selector already spent")
        self.used[key] = "attempted"
        called = False
        try:
            self.journal.append({"key": key, "operation": operation, "phase": self.phase,
                                 "transition": "attempted", "details": details})
            result = function()
            called = True
            self.used[key] = "function_returned"
            self.callback_return_values[key] = result
            self.partial_record["callback_returns"][key] = _plain_record(result)
            if on_return is not None:
                on_return(result)
            try:
                self.journal.append({"key": key, "operation": operation, "phase": self.phase,
                                     "transition": "returned", "details": details})
            except BaseException as terminal_error:
                self.failed = True
                self._attach_partial(terminal_error, self.snapshot())
                raise
            self.used[key] = "returned"
            return result
        except BaseException as error:
            self.failed = True
            if not called:
                self.used[key] = "failed"
                try:
                    self.journal.append({"key": key, "operation": operation, "phase": self.phase,
                                         "transition": "failed", "error_type": type(error).__name__,
                                         "message": str(error), "details": details})
                except BaseException as log_error:  # noqa: BLE001
                    error.input_consistency_log_error = {
                        "type": type(log_error).__name__, "message": str(log_error)
                    }
            self._attach_partial(error, self.snapshot())
            raise

    def read_metadata(self):
        self._check()
        if self.phase != 1 or self.metadata_done:
            self.failed = True
            raise PermissionError("source metadata phase is already spent")
        documents = self.partial_record["metadata_documents"]
        for row in self.plan["metadata_files"]:
            def read(row=row):
                path = safe_path(self.root, row["path"])
                if path.stat().st_size != row["size_bytes"]:
                    raise PermissionError("source metadata byte length differs before read")
                with path.open("rb") as stream:
                    raw = stream.read(row["size_bytes"])
                actual_sha256 = hashlib.sha256(raw).hexdigest()
                self.partial_record["metadata_byte_returns"][row["path"]] = {
                    "size_bytes": len(raw), "sha256": actual_sha256,
                    "validated": False,
                }
                if (len(raw) != row["size_bytes"] or path.stat().st_size != row["size_bytes"]
                        or actual_sha256 != row["sha256"]):
                    raise PermissionError("source metadata identity differs")
                self.partial_record["metadata_byte_returns"][row["path"]]["validated"] = True
                return json.loads(raw)

            def keep(document, row=row):
                documents[row["path"]] = document

            documents[row["path"]] = self._call(
                "metadata::" + row["path"], "metadata", read,
                details={"size_bytes": row["size_bytes"]}, on_return=keep,
            )
        try:
            self._metadata_envelopes = MappingProxyType({
                relative: _freeze({"document": document,
                                   "size_bytes": self.partial_record["metadata_byte_returns"][relative]["size_bytes"],
                                   "sha256": self.partial_record["metadata_byte_returns"][relative]["sha256"]})
                for relative, document in documents.items()
            })
        except BaseException as error:
            self.failed = True
            self._attach_partial(error, self.snapshot())
            raise
        self.metadata_done = True
        return documents

    def metadata_envelopes(self):
        """Expose parsed metadata bound to actual first-read bytes, with no reread."""
        self._check()
        if not self.metadata_done:
            self.failed = True
            raise PermissionError("metadata byte evidence is not complete")
        return self._metadata_envelopes

    def integrity(self, position):
        self._check()
        if position not in {"before", "after"} or position in self.hash_passes:
            self.failed = True
            raise PermissionError("invalid or repeated opaque integrity pass")
        if not self.metadata_done:
            self.failed = True
            raise PermissionError("integrity pass precedes its source metadata")
        if position == "before":
            if (self.phase != 1 or self.before_done
                    or any(key.startswith("context::") for key in self.used)):
                self.failed = True
                raise PermissionError("before-integrity pass must precede all payload contexts")
        else:
            if (self.phase != 3 or not self.before_done
                    or any(self.used.get("context::" + key) != "returned" for key in self.contexts)):
                self.failed = True
                raise PermissionError("after-integrity requires phase 3 and all contexts returned")
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

    def _freeze_payload(self, phase, freeze_path, freeze_sha256, size_bytes):
        expected_name = "origins_freeze.json" if phase == 2 else "futures_freeze.json"
        expected_path = self.directory / expected_name
        if Path(freeze_path).resolve(strict=False) != expected_path:
            raise PermissionError("population freeze path is not the phase-specific run marker")
        if (type(size_bytes) is not int or size_bytes <= 0
                or size_bytes > self.plan["freeze_contract"]["max_bytes"]
                or re.fullmatch(r"[0-9a-f]{64}", freeze_sha256) is None):
            raise PermissionError("invalid population freeze byte contract")
        relative = expected_path.relative_to(self.root).as_posix()
        path = safe_path(self.root, relative)
        if path.stat().st_size != size_bytes:
            raise PermissionError("population freeze byte length differs before read")
        with path.open("rb") as stream:
            raw = stream.read(size_bytes)
        if (len(raw) != size_bytes or path.stat().st_size != size_bytes
                or hashlib.sha256(raw).hexdigest() != freeze_sha256):
            raise PermissionError("population freeze identity differs")
        marker = json.loads(raw)
        contract = self.plan["freeze_contract"]
        kind = "a17_origins_freeze" if phase == 2 else "a17_futures_freeze"
        if (not isinstance(marker, dict) or marker.get("authorization") != AUTH
                or marker.get("kind") != kind or marker.get("run_id") != self.directory.name
                or marker.get("population_sha256") != contract["population_sha256"]
                or not isinstance(marker.get("records"), list)):
            raise PermissionError("population freeze semantic identity differs")
        records = marker["records"]
        if len(records) != len(contract["slot_ids"]):
            raise PermissionError("population freeze does not cover every slot exactly once")
        statuses = {"complete", "numerical_failure", "prerequisite_failed"}
        for record, slot_id in zip(records, contract["slot_ids"], strict=True):
            if (not isinstance(record, dict) or record.get("slot_id") != slot_id
                    or record.get("status") not in statuses
                    or not isinstance(record.get("record_path"), str)
                    or Path(record["record_path"]).name != record["record_path"]
                    or not record["record_path"].endswith(".json")
                    or re.fullmatch(r"[0-9a-f]{64}", record.get("record_sha256", "")) is None):
                raise PermissionError("population freeze record identity/status differs")
            record_path = self.directory / record["record_path"]
            safe_record = safe_path(self.root, record_path.relative_to(self.root).as_posix())
            if safe_record.is_symlink() or (hasattr(safe_record, "is_junction") and safe_record.is_junction()):
                raise PermissionError("population freeze record must be a regular saved file")

    def advance(self, phase, freeze_path, freeze_sha256, size_bytes):
        """Advance only after a bounded, phase-specific population receipt."""
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
            self._freeze_payload(phase, freeze_path, freeze_sha256, size_bytes)

        self._call(f"phase::{phase}", "candidate_freeze", freeze,
                   details={"path": str(freeze_path), "sha256": freeze_sha256,
                            "size_bytes": size_bytes})
        self.phase = phase

    def read_context(self, identity):
        self._check()
        row = self.contexts.get(identity)
        if row is None or row["phase"] != self.phase or not self.before_done:
            self.failed = True
            raise PermissionError("unknown context, wrong phase, or missing before-integrity pass")
        context_partial = self.partial_record["contexts"].setdefault(identity, {"arrays": {}})

        def read():
            target = safe_path(self.root, row["path"])
            if target.stat().st_size != row["size_bytes"]:
                raise PermissionError("bounded archive changed size")
            arrays = context_partial["arrays"]
            with zipfile.ZipFile(target) as archive:
                names = archive.namelist()
                if len(names) != len(set(names)):
                    raise PermissionError("duplicate ZIP members are forbidden")
                for member in row["members"]:
                    key = row["path"] + "::" + member["name"]
                    zipped = member["name"] + ".npy"
                    payload_size = math.prod(member["shape"]) * np.dtype(member["dtype"]).itemsize
                    info = archive.getinfo(zipped)
                    header_record = {"status": "attempted_unknown", "bytes_read": 0,
                                     "read_failure_unknown": False, "parsed": None}
                    context_partial.setdefault("headers", {})[member["name"]] = header_record
                    if not payload_size <= info.file_size <= payload_size + MAX_HEADER_BYTES:
                        header_record["status"] = "rejected_member_size"
                        raise PermissionError("NPY ZIP member exceeds its bounded header/payload size")
                    with archive.open(zipped) as stream:
                        def header(member=member, zipped=zipped, stream=stream,
                                   archive=archive, info=info, header_record=header_record):
                            bounded = _BoundedHeaderReader(stream)
                            try:
                                shape, dtype, fortran, length = _read_npy_header(bounded)
                                header_record["parsed"] = {"shape": list(shape), "dtype": dtype.str,
                                                            "fortran_order": bool(fortran),
                                                            "header_bytes": int(length)}
                                if (list(shape) != list(member["shape"]) or dtype.str != member["dtype"]
                                        or dtype.hasobject or length != bounded.bytes_read
                                        or length > MAX_HEADER_BYTES
                                        or info.file_size != length + math.prod(shape) * dtype.itemsize):
                                    header_record["status"] = "rejected_descriptor"
                                    raise PermissionError("unregistered NPY header or byte count")
                                header_record["status"] = "validated"
                                return shape, dtype, fortran, length
                            finally:
                                header_record["bytes_read"] = int(bounded.bytes_read)
                                header_record["read_failure_unknown"] = bounded.read_failure_unknown
                                if bounded.read_failure_unknown:
                                    header_record["status"] = "attempted_unknown"
                                elif header_record["status"] == "attempted_unknown":
                                    header_record["status"] = "header_parse_failed"
                                self.header_bytes += bounded.bytes_read

                        def keep_header(value):
                            self.descriptors += 1

                        shape, dtype, fortran, _ = self._call(
                            key + "::descriptor", "descriptor", header,
                            on_return=keep_header,
                        )

                        def payload(shape=shape, dtype=dtype, fortran=fortran,
                                    stream=stream, member=member, name=member["name"]):
                            count = math.prod(shape)
                            payload_record = {"status": "attempted_unknown", "expected_bytes": count * dtype.itemsize,
                                              "actual_bytes": None, "decoded": False, "validated": False,
                                              "decoded_elements": None}
                            context_partial.setdefault("payloads", {})[name] = payload_record
                            raw = stream.read(count * dtype.itemsize)
                            payload_record["actual_bytes"] = len(raw)
                            payload_record["status"] = "short_read" if len(raw) != count * dtype.itemsize else "bytes_returned"
                            self.payload_bytes_read += len(raw)
                            if len(raw) != count * dtype.itemsize:
                                raise PermissionError("truncated registered array")
                            value = np.frombuffer(raw, dtype=dtype).reshape(shape, order="F" if fortran else "C")
                            value.flags.writeable = False
                            payload_record.update(status="decoded_unvalidated", decoded=True,
                                                  decoded_elements=int(value.size), decoded_bytes=int(value.nbytes))
                            self.decoded_array_elements += int(value.size)
                            self.decoded_array_bytes += int(value.nbytes)
                            context_partial.setdefault("actual_decoded_arrays", {})[name] = value
                            finite = bool(np.isfinite(value).all())
                            payload_record["finite"] = finite
                            if not finite:
                                payload_record["status"] = "nonfinite_values"
                                raise PermissionError("registered array contains nonfinite values")
                            actual_sha256 = hashlib.sha256(value.tobytes(order="C")).hexdigest()
                            payload_record["actual_sha256_c_order_bytes"] = actual_sha256
                            if actual_sha256 != member["sha256_c_order_bytes"]:
                                payload_record["status"] = "array_sha_mismatch"
                                raise PermissionError("registered array identity/nonfiniteness differs")
                            payload_record.update(status="validated", validated=True)
                            return value

                        def keep_array(value, name=member["name"]):
                            self.array_elements += int(value.size)
                            self.array_reads += 1
                            arrays[name] = value

                        arrays[member["name"]] = self._call(
                            key + "::array", "array", payload,
                            details={"elements": math.prod(shape)}, on_return=keep_array,
                        )
            return arrays

        def keep_context(value):
            context_partial["returned_arrays"] = value

        return self._call("context::" + identity, "npz_context", read,
                          details={"path": row["path"], "members": len(row["members"])},
                          on_return=keep_context)

    def snapshot(self):
        return {"phase": self.phase, "failed": self.failed, "closed": self.closed,
                "selectors": dict(self.used), "array_reads": self.array_reads,
                "descriptors": self.descriptors, "array_elements": self.array_elements,
                "header_bytes": self.header_bytes, "payload_bytes_read": self.payload_bytes_read,
                "decoded_array_elements": self.decoded_array_elements,
                "decoded_array_bytes": self.decoded_array_bytes,
                "opaque_passes": sorted(self.hash_passes),
                "partial_record": self.partial_record}

    def finish(self):
        self._check()
        if (self.phase != 3 or any(self.used.get("context::" + key) != "returned" for key in self.contexts)
                or self.hash_passes != {"before", "after"}):
            self.failed = True
            raise PermissionError("bounded source inventory not fully consumed")
        self.close()
        return self.snapshot()

    def close(self):
        if self.closed:
            return
        try:
            self.journal.close()
        except BaseException:
            self.failed = True
            self.closed = True
            raise
        self.closed = True
