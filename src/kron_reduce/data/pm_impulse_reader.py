"""Finite A16 history-power reader; no model, labels or general data provider.

The production inventory is exactly 778 A14 archives and two source HDFs.
Every descriptor is inspected before its selected payload. Paths, hashes and
selectors come from a frozen plan, whose population and member families are
validated independently here. Failed or repeated access poisons the reader.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import zipfile
from pathlib import Path

import h5py
import numpy as np

from kron_reduce.data.durable_audit_ledger import AppendOnlyAuditJournal
from kron_reduce.data.limited_hdf_audit import ROOT_ATTRS
from kron_reduce.data.limited_npz_audit import MemberSpec, NpzWhitelistReader, _read_npy_header
from project_harness.report_storage import locate_report

SEEDS = (324, 325, 326)
SOURCES = ("ieee39_expansion", "ieee39_postfault")
RUN = "paper_kiki_a14_integral_history-history_mechanism-20261024T121439628465Z-34de0389-4cf0648f"
FAMILIES = {
    "inputs": {"history_time": (258, 11), "constant_power": (258, 10), "mass": (10,)},
    "static": {"P": (10, 10), "PC": (10, 2), "force_residual_projector": (10, 10)},
    "integral": {"Ltilde": (100, 2), "c": (10,), "S": (10,), "h": (10,),
                 "time": (11,), "g4": (100,), "btilde4": (100,),
                 "dynamic_residual_scaled4": (100,), "force_residual4": (10, 10)},
}


class FiniteAccessLedger:
    """A16-specific finite selectors; preserves the older A12/A14 hard limits."""

    def __init__(self, slots, metadata, path):
        self.slots = {row["key"]: {**row, "status": "not_attempted"} for row in slots}
        if (len(self.slots) != len(slots) or len(slots) != 15014
                or sum(s["kind"] == "array" for s in slots) != 7498
                or sum(s["kind"] == "countdescriptor" for s in slots) != 7498
                or sum(s["kind"] == "attr" for s in slots) != 18
                or sum(s["expected_elements"] for s in slots if s["kind"] == "array") != 533480):
            raise ValueError("A16 selector totals differ")
        self.metadata = metadata
        self.journal = AppendOnlyAuditJournal(path)
        self.failed = False

    def call(self, key, function, *, element_count):
        if self.failed:
            raise PermissionError("A16 access ledger is poisoned after failure")
        try:
            return self._call(key, function, element_count=element_count)
        except BaseException:
            self.failed = True
            raise

    def _call(self, key, function, *, element_count):
        if self.failed or key not in self.slots or self.slots[key]["status"] != "not_attempted":
            self.failed = True
            raise PermissionError("A16 selector is unknown, repeated or poisoned")
        row = self.slots[key]
        row["status"] = "attempted"
        self.journal.append({**row, "metadata": self.metadata[key]})
        try:
            value = function()
            count = element_count(value)
            if type(count) is not int or count != row["expected_elements"]:
                raise ValueError("A16 source returned a different logical count")
            row["status"] = "returned"
            self.journal.append({**row, "returned_elements": count, "metadata": self.metadata[key]})
            return value
        except BaseException as error:
            self.failed = True
            row["status"] = "failed"
            self.journal.append({**row, "error_type": type(error).__name__, "message": str(error), "metadata": self.metadata[key]})
            raise

    def snapshot(self):
        return {"failed": self.failed, "selectors": list(self.slots.values()),
                "counts": {kind: {status: sum(r["kind"] == kind and r["status"] == status for r in self.slots.values())
                                  for status in ("not_attempted", "attempted", "returned", "failed")}
                           for kind in ("array", "countdescriptor", "attr")},
                "returned_array_elements": sum(r["expected_elements"] for r in self.slots.values()
                                               if r["kind"] == "array" and r["status"] == "returned")}

    def close(self):
        self.journal.close()


def validate_plan(plan):
    """Pure metadata check; cannot open data or relax production dimensions."""
    files, events, sources = plan["npz_files"], plan["events"], plan["source_inventory"]
    if len(files) != 778 or len(events) != 258 or set(sources) != set(SOURCES):
        raise ValueError("A16 fixed inventory count mismatch")
    paths = {f"results/{RUN}/permitted_inputs.npz": "inputs"}
    for seed in SEEDS:
        paths[f"results/{RUN}/static-{seed}.npz"] = "static"
        for index in range(258):
            paths[f"results/{RUN}/seed-{seed}-event-{index:03d}/integral.npz"] = "integral"
    if len({r["path"] for r in files}) != 778 or {r["path"] for r in files} != set(paths):
        raise ValueError("A16 NPZ path whitelist differs")
    for row in files:
        expected = FAMILIES[paths[row["path"]]]
        members = row["members"]
        if len(members) != len(expected) or {m["name"] for m in members} != set(expected):
            raise ValueError("A16 independent member whitelist differs")
        for member in members:
            if tuple(member["shape"]) != expected[member["name"]] or member["dtype"] != "<f8":
                raise ValueError("A16 member descriptor differs")
        if (type(row["size_bytes"]) is not int or row["size_bytes"] <= 0
                or re.fullmatch("[0-9a-f]{64}", row["sha256"]) is None):
            raise ValueError("A16 NPZ identity is invalid")
    pairs = set()
    for index, event in enumerate(events):
        source, raw = event["source_dataset"], event["raw_id"]
        if (source not in SOURCES or event["role"] != "train" or event["source_role"] != "train"
                or event["cache_index"] != index or type(event["declared_length"]) is not int
                or event["declared_length"] < 12 or not re.fullmatch(r"[A-Za-z0-9_.-]+", raw)
                or raw in {".", ".."} or event["event_id"] != f"{source}/{raw}"
                or (source, raw) in pairs):
            raise ValueError("A16 original train event whitelist differs")
        pairs.add((source, raw))
    for source, row in sources.items():
        if (row["path"] != f"{source}/dataset.h5" or type(row["size_bytes"]) is not int
                or row["size_bytes"] <= 0 or re.fullmatch("[0-9a-f]{64}", row["sha256"]) is None):
            raise ValueError("A16 source identity differs")
    metadata = plan["metadata_files"]
    if len({r["path"] for r in metadata}) != len(metadata):
        raise ValueError("duplicate metadata selector")
    for row in metadata:
        if (Path(row["path"]).is_absolute() or ".." in Path(row["path"]).parts
                or "\\" in row["path"] or not row["path"].endswith(".json")
                or type(row["size_bytes"]) is not int or row["size_bytes"] <= 0
                or re.fullmatch("[0-9a-f]{64}", row["sha256"]) is None):
            raise ValueError("unsafe A16 metadata selector")
    return plan


class PmImpulseReader:
    """One-pass reads with durable accounting, and two opaque integrity passes."""

    def __init__(self, root, source_root, plan, directory):
        self.root = Path(root).resolve(strict=True)
        self.source_root = Path(source_root).resolve(strict=True)
        self.plan = validate_plan(plan)
        self.directory = Path(directory)
        self.io = AppendOnlyAuditJournal(self.directory / "io_access.jsonl")
        self.failed = False
        self.opened = set()
        self.hash_passes = set()
        self.metadata_read = set()
        slots, metadata = [], {}

        def slot(key, kind, count, source, event, name, selection):
            slots.append({"key": key, "phase": 1, "kind": kind, "expected_elements": count})
            metadata[key] = {"source": source, "event_id": event, "role": "train" if event else "static_prior",
                             "dataset_or_attr": name, "selection": selection}

        for row in plan["npz_files"]:
            for member in row["members"]:
                key = row["path"] + "::" + member["name"]
                slot(key + "::descriptor", "countdescriptor", 1, "A14", "registered_train_cache", member["name"], "NPY header")
                slot(key + "::array", "array", math.prod(member["shape"]), "A14", "registered_train_cache", member["name"], "whole registered member")
        for source in SOURCES:
            for attr in ROOT_ATTRS:
                slot(f"{source}::attr::{attr}", "attr", 1, source, None, attr, "root attribute")
            for name in ("inertia", "equilibrium/Pm"):
                key = f"{source}::{name}"
                slot(key + "::descriptor", "countdescriptor", 1, source, None, name, "HDF descriptor")
                slot(key + "::array", "array", 10, source, None, name, "[:]")
        for event in plan["events"]:
            for name, count in (("Pm", 110), ("t", 11)):
                key = f"{event['event_id']}::{name}"
                path = f"events/{event['raw_id']}/{name}"
                slot(key + "::descriptor", "countdescriptor", 1, event["source_dataset"], event["event_id"], path, "HDF descriptor")
                slot(key + "::array", "array", count, event["source_dataset"], event["event_id"], path, "[0:11,:]" if name == "Pm" else "[0:11]")
        self.ledger = FiniteAccessLedger(slots, metadata, self.directory / "array_access.jsonl")

    def _check(self):
        if self.failed:
            raise RuntimeError("A16 reader is poisoned after failure")

    def _once(self, key):
        self._check()
        if key in self.opened:
            self.failed = True
            raise RuntimeError("A16 source context already spent")
        self.opened.add(key)

    def integrity(self, phase):
        """Opaque bytes only; before/after are each finite and never repeatable."""
        self._check()
        if phase not in {"before", "after"} or phase in self.hash_passes or (phase == "after" and "before" not in self.hash_passes):
            self.failed = True
            raise RuntimeError("invalid integrity phase")
        self.hash_passes.add(phase)
        try:
            files = [(self.root / r["path"], r) for r in self.plan["npz_files"]]
            files += [(self.source_root / r["path"], r) for r in self.plan["source_inventory"].values()]
            for path, row in files:
                self.io.append({"category": "opaque_integrity", "phase": phase, "path": str(path), "transition": "attempted", "bytes": row["size_bytes"]})
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    if path.stat().st_size != row["size_bytes"]:
                        raise ValueError("A16 source byte length changed")
                    remaining = row["size_bytes"]
                    while remaining:
                        block = stream.read(min(1024 * 1024, remaining))
                        if not block:
                            raise EOFError("A16 source integrity stream is short")
                        digest.update(block)
                        remaining -= len(block)
                if digest.hexdigest() != row["sha256"]:
                    raise ValueError("A16 source SHA differs")
                self.io.append({"category": "opaque_integrity", "phase": phase, "path": str(path), "transition": "returned", "bytes": row["size_bytes"], "sha256": digest.hexdigest()})
        except BaseException:
            self.failed = True
            raise

    def read_metadata(self):
        self._once("metadata")
        values = {}
        try:
            for row in self.plan["metadata_files"]:
                self.io.append({"category": "source_metadata", "path": row["path"], "transition": "attempted", "bytes": row["size_bytes"]})
                path = self.root / locate_report(self.root, row["path"], row["sha256"])
                with path.open("rb") as stream:
                    if path.stat().st_size != row["size_bytes"]:
                        raise ValueError("A16 metadata size differs")
                    raw = stream.read(row["size_bytes"])
                if len(raw) != row["size_bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
                    raise ValueError("A16 metadata identity differs")
                values[row["path"]] = json.loads(raw.decode("utf-8"))
                self.metadata_read.add(row["path"])
                self.io.append({"category": "source_metadata", "path": row["path"], "transition": "returned", "bytes": len(raw)})
            return values
        except BaseException:
            self.failed = True
            raise

    def read_npz(self, row):
        if self.hash_passes != {"before"}:
            self.failed = True
            raise PermissionError("A16 payload requires the initial integrity freeze")
        if row not in self.plan["npz_files"]:
            self.failed = True
            raise PermissionError("unregistered A16 NPZ selector")
        self._once(row["path"])
        try:
            result = {}
            self.io.append({"category": "npz_context", "path": row["path"], "transition": "attempted"})
            with zipfile.ZipFile(self.root / row["path"], "r") as archive:
                infos = archive.infolist()
                if len({i.filename for i in infos}) != len(infos):
                    raise ValueError("duplicate ZIP member name")
                for member in row["members"]:
                    spec = MemberSpec(member["name"], tuple(member["shape"]), "f8")
                    key = row["path"] + "::" + member["name"]
                    holder = {}
                    try:
                        def descriptor(spec=spec, holder=holder):
                            stream = archive.open(spec.key + ".npy", "r")
                            holder["stream"] = stream
                            shape, dtype, fortran, header = _read_npy_header(stream)
                            NpzWhitelistReader._validate_header(spec, shape, dtype)
                            count = math.prod(shape) * 8
                            if header + count != archive.getinfo(spec.key + ".npy").file_size:
                                raise ValueError("A16 NPY storage length differs")
                            return shape, dtype, fortran, count
                        shape, dtype, fortran, count = self.ledger.call(key + "::descriptor", descriptor, element_count=lambda _: 1)
                        stream = holder["stream"]
                        elements = math.prod(shape)
                        raw = self.ledger.call(key + "::array", lambda stream=stream, count=count: NpzWhitelistReader._read_exact_payload(stream, count), element_count=lambda _, n=elements: n)
                        array = np.frombuffer(raw, dtype=dtype).reshape(shape, order="F" if fortran else "C")
                        result[member["name"]] = array
                    finally:
                        if "stream" in holder:
                            holder["stream"].close()
            self.io.append({"category": "npz_context", "path": row["path"], "transition": "returned"})
            return result
        except BaseException:
            self.failed = True
            raise

    def read_source(self, source, root_validator):
        if self.hash_passes != {"before"}:
            self.failed = True
            raise PermissionError("A16 payload requires the initial integrity freeze")
        if source not in SOURCES:
            self.failed = True
            raise PermissionError("unregistered HDF source")
        self._once(source)
        try:
            row = self.plan["source_inventory"][source]
            self.io.append({"category": "hdf_context", "source": source, "transition": "attempted", "mode": "r"})
            with h5py.File(self.source_root / row["path"], "r") as handle:
                attrs = {name: self.ledger.call(f"{source}::attr::{name}", lambda name=name: handle.attrs[name], element_count=lambda _: 1) for name in ROOT_ATTRS}
                root_validator(source, attrs)
                static = {}
                events = {}

                def read(name, key, shape, selection):
                    def descriptor():
                        dataset = handle[name]
                        if tuple(dataset.shape) != shape or dataset.dtype.str != "<f8":
                            raise ValueError("A16 HDF descriptor differs")
                        return {"shape": list(dataset.shape), "dtype": dataset.dtype.str}
                    self.ledger.call(key + "::descriptor", descriptor, element_count=lambda _: 1)
                    selected = (11, 10) if name.endswith("/Pm") and name.startswith("events/") else (11,) if name.endswith("/t") else (10,)
                    def payload():
                        value = np.asarray(handle[name][selection])
                        if value.shape != selected or value.dtype.str != "<f8":
                            raise ValueError("A16 selected HDF payload differs")
                        value.setflags(write=False)
                        return value
                    return self.ledger.call(key + "::array", payload, element_count=lambda value: int(value.size))

                for name in ("inertia", "equilibrium/Pm"):
                    static[name] = read(name, f"{source}::{name}", (10,), slice(None))
                for event in self.plan["events"]:
                    if event["source_dataset"] != source:
                        continue
                    raw, length, event_id = event["raw_id"], event["declared_length"], event["event_id"]
                    events[event_id] = {
                        "Pm": read(f"events/{raw}/Pm", f"{event_id}::Pm", (length, 10), (slice(0, 11), slice(None))),
                        "t": read(f"events/{raw}/t", f"{event_id}::t", (length,), slice(0, 11)),
                    }
            self.io.append({"category": "hdf_context", "source": source, "transition": "returned", "mode": "r"})
            return attrs, static, events
        except BaseException:
            self.failed = True
            raise

    def finish(self):
        self._check()
        snapshot = self.ledger.snapshot()
        if (self.hash_passes != {"before", "after"} or len(self.opened) != 781
                or len(self.metadata_read) != len(self.plan["metadata_files"])
                or snapshot["returned_array_elements"] != 533480):
            raise RuntimeError("A16 reader inventory is incomplete")
        self.ledger.close()
        self.io.close()
        return snapshot
