from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kron_reduce.data.audit_access_ledger import AccessLedger, AccessLedgerError
from kron_reduce.data.limited_hdf_audit import (
    EVENT_ATTRS,
    ROOT_ATTRS,
    STATIC_ARRAYS,
    LimitedHdfReader,
    ReaderError,
)

HERE = Path(__file__).resolve().parent
PLAN_PATH = HERE.parent / "configs/proposals/kiki_a12_record20_reads.json"


class SpyAttrs:
    def __init__(self, values, log, label):
        self._values, self._log, self._label = values, log, label

    def __getitem__(self, name):
        self._log.append(("attr", self._label, name))
        return self._values[name]

    def keys(self):
        raise AssertionError("attribute enumeration is forbidden")

    def __iter__(self):
        raise AssertionError("attribute enumeration is forbidden")


class SpyDataset:
    def __init__(self, value, log, label, dtype=None):
        self.value = np.asarray(value, dtype=dtype)
        self.log, self.label = log, label

    @property
    def shape(self):
        self.log.append(("shape", self.label))
        return self.value.shape

    @property
    def dtype(self):
        self.log.append(("dtype", self.label))
        return self.value.dtype

    def __getitem__(self, selection):
        self.log.append(("payload", self.label, selection))
        return self.value[selection]


class SpyGroup:
    def __init__(self, entries, log, label="group", attrs=None):
        self.entries, self.log, self.label = entries, log, label
        self.attrs = SpyAttrs(attrs or {}, log, label)

    def __getitem__(self, name):
        self.log.append(("lookup", self.label, name))
        if "/" in name:
            head, tail = name.split("/", 1)
            return self.entries[head][tail]
        return self.entries[name]

    def keys(self):
        raise AssertionError("HDF group enumeration is forbidden")

    def __iter__(self):
        raise AssertionError("HDF group enumeration is forbidden")


class SpyFile(SpyGroup):
    def __init__(self, entries, log, label):
        super().__init__(entries, log, label, attrs={name: f"{label}:{name}" for name in ROOT_ATTRS})
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.closed = True
        return False


class SpyOpener:
    def __init__(self, files):
        self.files, self.calls = files, []

    def __call__(self, path, mode):
        self.calls.append((path, mode))
        if mode != "r":
            raise AssertionError("HDF must be opened read-only")
        return self.files[path]


def _synthetic_plan():
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    plan["fixed_population"]["declared_length_sum_S"] = 268 * 3
    plan["phase1"]["planned_scalar_count"] = 21 * 268 * 3 + 2740
    plan["phase1"]["planned_float64_logical_bytes"] = plan["phase1"]["planned_scalar_count"] * 8
    for row in plan["slots"]:
        row["declared_length"] = 3
        row["phase1_array_selections"] = [
            {"dataset": "t", "selection": "[0:3]", "expected_shape": "(3,)"},
            {"dataset": "delta", "selection": "[0:3, 0:10]", "expected_shape": "(3,10)"},
            {"dataset": "omega", "selection": "[0:3, 0:10]", "expected_shape": "(3,10)"},
            {"dataset": "Pm", "selection": "[2, 0:10]", "expected_shape": "(10,)", "attempt_only_if_index_exists": True},
        ]
    return plan


def _fake_files(plan):
    log, files = [], {}
    for source in sorted(plan["source_inventory"]):
        static = {
            "inertia": SpyDataset(np.ones(10), log, f"{source}:inertia"),
            "equilibrium": SpyGroup({
                "delta": SpyDataset(np.ones(10), log, f"{source}:eqdelta"),
                "Pm": SpyDataset(np.ones(10), log, f"{source}:eqpm"),
            }, log, f"{source}:equilibrium"),
        }
        events = {}
        for row in plan["slots"]:
            if row["source_dataset"] != source:
                continue
            rid = row["raw_id"]
            arrays = {
                "t": SpyDataset(np.arange(3.0), log, f"{source}:{rid}:t"),
                "delta": SpyDataset(np.ones((3, 10)), log, f"{source}:{rid}:delta"),
                "omega": SpyDataset(np.ones((3, 10)), log, f"{source}:{rid}:omega"),
                "Pm": SpyDataset(np.ones((3, 10)), log, f"{source}:{rid}:Pm"),
            }
            events[rid] = SpyGroup(arrays, log, f"{source}:{rid}", attrs={name: f"{rid}:{name}" for name in EVENT_ATTRS})
        entries = {
            **static,
            "events": SpyGroup(events, log, f"{source}:events"),
        }
        files[f"synthetic/{source}/dataset.h5"] = SpyFile(entries, log, source)
    return log, files


def _reader(plan=None):
    plan = _synthetic_plan() if plan is None else plan
    log, files = _fake_files(plan)
    paths = {source: f"synthetic/{source}/dataset.h5" for source in plan["source_inventory"]}
    opener = SpyOpener(files)
    return LimitedHdfReader(plan, paths, file_opener=opener), log, opener


def test_two_phase_reads_are_fixed_readonly_and_descriptor_first():
    made_ledgers = []

    def ledger_factory(rows, *, logical_element_limit):
        ledger = AccessLedger(rows, logical_element_limit=logical_element_limit)
        made_ledgers.append(ledger)
        return ledger

    plan = _synthetic_plan()
    log, files = _fake_files(plan)
    paths = {source: f"synthetic/{source}/dataset.h5" for source in plan["source_inventory"]}
    opener = SpyOpener(files)
    reader = LimitedHdfReader(plan, paths, file_opener=opener, ledger_factory=ledger_factory)
    assert reader.ledger is made_ledgers[0]
    phase1 = {source: reader.read_phase1_source(source) for source in sorted(reader.source_paths)}
    assert len(opener.calls) == 2
    assert all(mode == "r" for _, mode in opener.calls)
    assert len(phase1) == 2
    payloads = [entry for entry in log if entry[0] == "payload"]
    assert len(payloads) == 1078
    first_payload = payloads[0]
    desc_positions = [i for i, entry in enumerate(log) if entry[0] in {"shape", "dtype"} and first_payload[1] in entry]
    payload_position = log.index(first_payload)
    assert desc_positions and max(desc_positions) < payload_position
    descriptor_reads = [(entry[0], entry[1]) for entry in log if entry[0] in {"shape", "dtype"}]
    assert len(descriptor_reads) == len(set(descriptor_reads))
    assert not any(entry[0] == "enumerate" for entry in log)

    reader.seal_prelabel_freeze("a" * 64)
    phase2 = {source: reader.read_phase2_source(source) for source in sorted(reader.source_paths)}
    assert len(opener.calls) == 4
    assert all(mode == "r" for _, mode in opener.calls)
    assert len(phase2) == 2
    assert sum(entry[0] == "attr" for entry in log) == 18 + 268 * 11
    snapshot = reader.audit_log()
    assert snapshot["slot_status_counts"]["not_attempted"] == 0
    assert snapshot["slot_status_counts"]["failed"] == 0
    assert snapshot["slot_status_counts"]["skipped"] == 0
    assert all(not isinstance(value, SpyFile) for result in phase1.values() for value in result.values())
    assert all(file.closed for file in files.values())


def test_unsafe_plan_is_rejected_before_any_source_call():
    plan = _synthetic_plan()
    plan["slots"][0]["phase1_array_selections"][3]["selection"] = "[10, 0:10]"
    opener = SpyOpener({})
    with pytest.raises(ReaderError, match="selector"):
        LimitedHdfReader(plan, {s: f"synthetic/{s}/dataset.h5" for s in plan["source_inventory"]}, file_opener=opener)
    assert opener.calls == []


def test_unknown_source_and_phase2_before_freeze_do_not_open_files():
    reader, _, opener = _reader()
    with pytest.raises(ReaderError, match="unregistered source"):
        reader.read_phase1_source("unknown")
    with pytest.raises(AccessLedgerError, match="belongs to phase 2"):
        reader.read_phase2_source("ieee39_postfault")
    assert opener.calls == []


def test_bad_shape_skips_payload_and_never_retries():
    plan = _synthetic_plan()
    log, files = _fake_files(plan)
    source = plan["slots"][0]["source_dataset"]
    raw_id = plan["slots"][0]["raw_id"]
    files[f"synthetic/{source}/dataset.h5"].entries["events"].entries[raw_id].entries["delta"] = SpyDataset(
        np.ones((3, 9)), log, f"{source}:{raw_id}:delta-bad"
    )
    opener = SpyOpener(files)
    reader = LimitedHdfReader(plan, {s: f"synthetic/{s}/dataset.h5" for s in plan["source_inventory"]}, file_opener=opener)
    reader.read_phase1_source(source)
    delta_payload = [x for x in log if x[0] == "payload" and x[1] == f"{source}:{raw_id}:delta-bad"]
    assert delta_payload == []
    key = f"event-array:{source}:{source}/{raw_id}:delta"
    snap = reader.audit_log()
    slot = next(x for x in snap["selectors"] if x["key"] == key)
    assert slot["status"] == "not_attempted" and slot["skip_reason"] == "descriptor_invalid"
    with pytest.raises(ReaderError, match="already been opened"):
        reader.read_phase1_source(source)
    assert opener.calls == [(f"synthetic/{source}/dataset.h5", "r")]


def test_unsafe_dtype_skips_array_payload():
    plan = _synthetic_plan()
    log, files = _fake_files(plan)
    row = plan["slots"][0]
    source, raw_id = row["source_dataset"], row["raw_id"]
    files[f"synthetic/{source}/dataset.h5"].entries["events"].entries[raw_id].entries["omega"] = SpyDataset(
        np.ones((3, 10), dtype=np.int64), log, f"{source}:{raw_id}:omega-int"
    )
    opener = SpyOpener(files)
    reader = LimitedHdfReader(plan, {s: f"synthetic/{s}/dataset.h5" for s in plan["source_inventory"]}, file_opener=opener)
    reader.read_phase1_source(source)
    assert not [x for x in log if x[0] == "payload" and x[1] == f"{source}:{raw_id}:omega-int"]
    key = f"event-array:{source}:{source}/{raw_id}:omega"
    slot = next(x for x in reader.audit_log()["selectors"] if x["key"] == key)
    assert slot["skip_reason"] == "descriptor_invalid"


def test_missing_origin_power_index_is_terminal_skip():
    plan = _synthetic_plan()
    row = plan["slots"][0]
    row["declared_length"] = 2
    row["phase1_array_selections"] = [
        {"dataset": "t", "selection": "[0:2]", "expected_shape": "(2,)"},
        {"dataset": "delta", "selection": "[0:2, 0:10]", "expected_shape": "(2,10)"},
        {"dataset": "omega", "selection": "[0:2, 0:10]", "expected_shape": "(2,10)"},
        {"dataset": "Pm", "selection": "[2, 0:10]", "expected_shape": "(10,)", "attempt_only_if_index_exists": True},
    ]
    plan["fixed_population"]["declared_length_sum_S"] -= 1
    plan["phase1"]["planned_scalar_count"] -= 21
    plan["phase1"]["planned_float64_logical_bytes"] -= 21 * 8
    log, files = _fake_files(plan)
    source, raw_id = row["source_dataset"], row["raw_id"]
    event = files[f"synthetic/{source}/dataset.h5"].entries["events"].entries[raw_id]
    event.entries["t"] = SpyDataset(np.arange(2.0), log, f"{source}:{raw_id}:t2")
    event.entries["delta"] = SpyDataset(np.ones((2, 10)), log, f"{source}:{raw_id}:delta2")
    event.entries["omega"] = SpyDataset(np.ones((2, 10)), log, f"{source}:{raw_id}:omega2")
    event.entries["Pm"] = SpyDataset(np.ones((2, 10)), log, f"{source}:{raw_id}:Pm2")
    reader = LimitedHdfReader(plan, {s: f"synthetic/{s}/dataset.h5" for s in plan["source_inventory"]}, file_opener=SpyOpener(files))
    reader.read_phase1_source(source)
    assert not [x for x in log if x[0] == "payload" and x[1] == f"{source}:{raw_id}:Pm2"]
    key = f"event-array:{source}:{source}/{raw_id}:Pm"
    slot = next(x for x in reader.audit_log()["selectors"] if x["key"] == key)
    assert slot["skip_reason"] == "descriptor_invalid"
    assert slot["validation_reason"] is None


def test_source_open_failure_is_recorded_and_cannot_be_retried():
    plan = _synthetic_plan()

    class BrokenOpener:
        def __init__(self):
            self.calls = []

        def __call__(self, path, mode):
            self.calls.append((path, mode))
            raise OSError("synthetic open failure")

    opener = BrokenOpener()
    reader = LimitedHdfReader(plan, {s: f"synthetic/{s}/dataset.h5" for s in plan["source_inventory"]}, file_opener=opener)
    assert reader.read_phase1_source("ieee39_expansion") == {}
    snapshot = reader.audit_log()
    assert snapshot["slot_status_counts"]["failed"] == 1
    assert snapshot["slot_status_counts"]["skipped"] > 0
    with pytest.raises(ReaderError, match="already been opened"):
        reader.read_phase1_source("ieee39_expansion")
    assert opener.calls == [("synthetic/ieee39_expansion/dataset.h5", "r")]


def test_root_validator_stops_before_static_or_event_reads_and_closes_file():
    reader, log, opener = _reader()
    checked = []

    def reject_root(source, values):
        checked.append((source, tuple(values)))
        assert len(values) == 9
        raise RuntimeError("synthetic root identity mismatch")

    reader.root_validator = reject_root
    source = "ieee39_expansion"
    with pytest.raises(RuntimeError, match="identity mismatch"):
        reader.read_phase1_source(source)
    file = opener.files[f"synthetic/{source}/dataset.h5"]
    assert file.closed
    assert checked and checked[0][0] == source
    assert not [entry for entry in log if entry[0] in {"shape", "dtype", "payload"}]
    assert len([entry for entry in log if entry[0] == "attr"]) == len(ROOT_ATTRS)


def test_static_validator_stops_before_event_descriptors_and_closes_file():
    reader, log, opener = _reader()
    checked = []

    def reject_static(source, values):
        checked.append((source, values))
        assert len(values) == len(ROOT_ATTRS) + len(STATIC_ARRAYS)
        assert all(not values[f"{name}[:]"].flags.writeable for name in STATIC_ARRAYS)
        with pytest.raises(TypeError):
            values["injected"] = 1
        raise RuntimeError("synthetic static contract mismatch")

    reader.static_validator = reject_static
    source = "ieee39_expansion"
    with pytest.raises(RuntimeError, match="static contract"):
        reader.read_phase1_source(source)
    file = opener.files[f"synthetic/{source}/dataset.h5"]
    assert file.closed
    assert checked and checked[0][0] == source
    assert len([entry for entry in log if entry[0] == "shape"]) == len(STATIC_ARRAYS)
    assert len([entry for entry in log if entry[0] == "payload"]) == len(STATIC_ARRAYS)
