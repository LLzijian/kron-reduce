"""Plan-bound, read-only HDF accessor candidate for the A12 audit.

This module performs only registered metadata and array reads. It does not
construct samples, interpret attributes, or decide whether an audit passes.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from kron_reduce.data.audit_access_ledger import AccessLedger, AccessLedgerError

ALLOWED_SOURCES = {"ieee39_postfault", "ieee39_expansion"}
ROOT_ATTRS = (
    "andes_version", "case_sha256", "config", "git_commit", "mva_base",
    "omega_s", "postfault_signature", "raw_units", "state_mapping",
)
STATIC_ARRAYS = ("inertia", "equilibrium/delta", "equilibrium/Pm")
EVENT_ARRAYS = ("t", "delta", "omega", "Pm")
EVENT_ATTRS = (
    "label", "event_time_s", "termination_reason", "trajectory_end_s",
    "valid_length", "clearing_time_s", "record_start_delay_s",
    "solver_success", "failed_steps", "postfault_signature", "fault_line",
)
_RAW_ID = re.compile(r"[A-Za-z0-9_.-]+\Z")


class ReaderError(ValueError):
    """A plan or API request is outside the registered read contract."""


def _selector_key(category: str, source: str, name: str, event_id: str | None = None) -> str:
    if event_id is None:
        return f"{category}:{source}:{name}"
    return f"{category}:{source}:{event_id}:{name}"


class LimitedHdfReader:
    """Read one registered phase at a time, opening each source at most once.

    ``file_opener`` is injectable for synthetic tests. Production callers
    should pass ``h5py.File``; this class always requests mode ``"r"``.
    """

    def __init__(self, plan: Mapping[str, Any], source_paths: Mapping[str, str | Path],
                 *, file_opener: Callable[..., Any], ledger_factory: Callable[..., Any] = AccessLedger,
                 root_validator: Callable[[str, Mapping[str, Any]], Any] | None = None,
                 static_validator: Callable[[str, Mapping[str, Any]], Any] | None = None):
        if (not isinstance(plan, Mapping) or not callable(file_opener)
                or not callable(ledger_factory)
                or (root_validator is not None and not callable(root_validator))
                or (static_validator is not None and not callable(static_validator))):
            raise ReaderError("plan, file opener, ledger factory, and optional validators must be callable")
        self.plan = plan
        self.file_opener = file_opener
        self.root_validator = root_validator
        self.static_validator = static_validator
        inventory = plan.get("source_inventory")
        rows = plan.get("slots")
        if not isinstance(inventory, Mapping) or set(inventory) != ALLOWED_SOURCES:
            raise ReaderError("source inventory must name the two registered sources")
        if not isinstance(source_paths, Mapping) or set(source_paths) != ALLOWED_SOURCES:
            raise ReaderError("source paths must exactly match the registered sources")
        if not isinstance(rows, list) or len(rows) != 268:
            raise ReaderError("the fixed train whitelist must contain 268 rows")
        fixed = plan.get("fixed_population", {})
        if (fixed.get("filter") != {"source_role": "train", "role": "train"}
                or fixed.get("ordering") != ["source_dataset", "raw_id"]
                or fixed.get("slots") != 268
                or fixed.get("length_bounds_inclusive") != [1, 1501]):
            raise ReaderError("fixed population contract differs from the registered train whitelist")
        phase1 = plan.get("phase1", {})
        expected_event_selectors = [
            {"dataset": "t", "selection": "[0:N_i]", "expected_shape": "(N_i,)"},
            {"dataset": "delta", "selection": "[0:N_i, 0:10]", "expected_shape": "(N_i,10)"},
            {"dataset": "omega", "selection": "[0:N_i, 0:10]", "expected_shape": "(N_i,10)"},
            {"dataset": "Pm", "selection": "[2, 0:10]", "expected_shape": "(10,)"},
        ]
        if (phase1.get("event_array_selections_each") != expected_event_selectors
                or phase1.get("static_arrays_per_source") != [f"{name}[:]" for name in STATIC_ARRAYS]
                or phase1.get("static_array_selection_count") != 6
                or phase1.get("root_attributes_per_source") != list(ROOT_ATTRS)
                or phase1.get("root_attribute_read_count") != 18
                or phase1.get("event_attributes_read") != 0
                or phase1.get("scalar_formula") != "21*S + 2740"):
            raise ReaderError("phase-1 selectors or read counts differ from the registered contract")
        phase2 = plan.get("phase2", {})
        if (phase2.get("event_attribute_read_count") != 2948
                or phase2.get("only_after_successful_prelabel_freeze") is not True):
            raise ReaderError("phase-2 contract differs from the registered post-freeze scope")
        self.source_paths = dict(source_paths)
        for source, metadata in inventory.items():
            relative = metadata.get("dataset_h5_relative_path")
            if relative != f"{source}/dataset.h5":
                raise ReaderError("registered source HDF path is invalid")
            actual_parts = Path(str(self.source_paths[source])).parts
            relative_parts = Path(relative).parts
            if len(actual_parts) < len(relative_parts) or tuple(actual_parts[-len(relative_parts):]) != tuple(relative_parts):
                raise ReaderError("source path does not match its registered HDF path")
            digest = metadata.get("dataset_h5_registered_sha256")
            if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ReaderError("registered source HDF SHA is invalid")
        self.rows_by_source: dict[str, list[dict[str, Any]]] = {s: [] for s in ALLOWED_SOURCES}
        seen_ids: set[str] = set()
        seen_raw: set[tuple[str, str]] = set()
        for row in rows:
            if not isinstance(row, Mapping):
                raise ReaderError("whitelist row must be an object")
            source, raw_id = row.get("source_dataset"), row.get("raw_id")
            event_id, length = row.get("event_id"), row.get("declared_length")
            if source not in ALLOWED_SOURCES or row.get("role") != "train" or row.get("source_role") != "train":
                raise ReaderError("only registered source train rows are permitted")
            if not isinstance(raw_id, str) or _RAW_ID.fullmatch(raw_id) is None or raw_id in {".", ".."}:
                raise ReaderError("unsafe raw event identifier")
            if event_id != f"{source}/{raw_id}" or event_id in seen_ids or (source, raw_id) in seen_raw:
                raise ReaderError("event identity is invalid or duplicated")
            if type(length) is not int or not 1 <= length <= 1501:
                raise ReaderError("declared event length is outside the registered bounds")
            expected_selections = [
                {"dataset": "t", "selection": f"[0:{length}]", "expected_shape": f"({length},)"},
                {"dataset": "delta", "selection": f"[0:{length}, 0:10]", "expected_shape": f"({length},10)"},
                {"dataset": "omega", "selection": f"[0:{length}, 0:10]", "expected_shape": f"({length},10)"},
                {"dataset": "Pm", "selection": "[2, 0:10]", "expected_shape": "(10,)", "attempt_only_if_index_exists": True},
            ]
            if row.get("phase1_array_selections") != expected_selections:
                raise ReaderError("event array selector differs from the registered plan")
            if row.get("phase2_event_attributes") != list(EVENT_ATTRS):
                raise ReaderError("event attribute selector differs from the registered plan")
            seen_ids.add(event_id)
            seen_raw.add((source, raw_id))
            self.rows_by_source[source].append(dict(row))
        if [row["slot"] for row in rows] != list(range(268)):
            raise ReaderError("whitelist slot numbers must be complete and ordered")
        if [(r["source_dataset"], r["raw_id"]) for r in rows] != sorted(
                (r["source_dataset"], r["raw_id"]) for r in rows):
            raise ReaderError("whitelist rows must follow registered source/raw_id ordering")
        declared_sum = sum(row["declared_length"] for row in rows)
        if fixed.get("declared_length_sum_S") != declared_sum:
            raise ReaderError("declared-length sum differs from the fixed whitelist")
        expected_scalars = 21 * declared_sum + 2740
        if phase1.get("planned_scalar_count") != expected_scalars or phase1.get("planned_float64_logical_bytes") != expected_scalars * 8:
            raise ReaderError("registered logical array budget differs from the fixed whitelist")
        for source, metadata in inventory.items():
            if metadata.get("root_attribute_names_planned") != list(ROOT_ATTRS):
                raise ReaderError("root attribute selector differs from the registered plan")
            if metadata.get("root_attribute_read_count_planned") != len(ROOT_ATTRS):
                raise ReaderError("registered source root-attribute count is invalid")
            if metadata.get("static_array_selections_planned") != [f"{name}[:]" for name in STATIC_ARRAYS]:
                raise ReaderError("static array selector differs from the registered plan")
            if metadata.get("static_array_selection_count_planned") != len(STATIC_ARRAYS):
                raise ReaderError("registered static-array count is invalid")
        if plan.get("phase2", {}).get("event_attributes_per_slot") != list(EVENT_ATTRS):
            raise ReaderError("phase-2 attribute selector differs from the registered plan")

        ledger_plan: list[dict[str, Any]] = []
        self._meta: dict[str, dict[str, Any]] = {}
        self._phase_keys: dict[tuple[str, int], list[str]] = {(s, p): [] for s in ALLOWED_SOURCES for p in (1, 2)}

        def add(key: str, phase: int, kind: str, elements: int, meta: dict[str, Any]) -> None:
            ledger_plan.append({"key": key, "phase": phase, "kind": kind, "expected_elements": elements})
            self._meta[key] = meta
            self._phase_keys[(meta["source"], phase)].append(key)

        for source in sorted(ALLOWED_SOURCES):
            for attr in ROOT_ATTRS:
                key = _selector_key("root-attr", source, attr)
                add(key, 1, "attr", 1, {"source": source, "selector": attr, "kind": "root_attr"})
            for name in STATIC_ARRAYS:
                desc = _selector_key("descriptor", source, name)
                arr = _selector_key("static-array", source, name)
                add(desc, 1, "countdescriptor", 1, {"source": source, "selector": name, "kind": "descriptor"})
                add(arr, 1, "array", 10, {"source": source, "selector": name, "kind": "static_array", "descriptor": desc})
            for row in self.rows_by_source[source]:
                event_id, n = row["event_id"], row["declared_length"]
                for name in EVENT_ARRAYS:
                    desc = _selector_key("descriptor", source, name, event_id)
                    arr = _selector_key("event-array", source, name, event_id)
                    elements = n if name == "t" else (n * 10 if name in {"delta", "omega"} else 10)
                    add(desc, 1, "countdescriptor", 1, {"source": source, "event_id": event_id, "selector": name, "kind": "descriptor"})
                    add(arr, 1, "array", elements, {"source": source, "event_id": event_id, "selector": name, "kind": "event_array", "descriptor": desc})
                for attr in EVENT_ATTRS:
                    key = _selector_key("event-attr", source, attr, event_id)
                    add(key, 2, "attr", 1, {"source": source, "event_id": event_id, "selector": attr, "kind": "event_attr"})
        scalar_limit = plan.get("phase1", {}).get("planned_scalar_count")
        if type(scalar_limit) is not int or scalar_limit <= 0:
            raise ReaderError("registered logical scalar budget is invalid")
        self.ledger = ledger_factory(ledger_plan, logical_element_limit=scalar_limit)
        self._opened: set[tuple[str, int]] = set()
        self._errors: dict[str, dict[str, str]] = {}
        self._values: dict[str, Any] = {}

    def _capture(self, key: str, callback: Callable[[], Any], *, element_count=None) -> Any:
        try:
            value = self.ledger.call(key, callback, element_count=element_count)
        except AccessLedgerError:
            raise
        except Exception as exc:  # noqa: BLE001 - source callback failures must be preserved in the ledger.
            self._errors[key] = {"type": type(exc).__name__, "message": str(exc)}
            return None
        self._values[key] = value
        return value

    @staticmethod
    def _dataset(handle: Any, meta: Mapping[str, Any]) -> Any:
        if meta["kind"] == "static_array":
            return handle[meta["selector"]]
        if meta["kind"] == "descriptor" or meta["kind"] == "event_array":
            if "event_id" in meta:
                raw_id = meta["event_id"].split("/", 1)[1]
                return handle["events"][raw_id][meta["selector"]]
            return handle[meta["selector"]]
        raise ReaderError("selector is not a dataset")

    @staticmethod
    def _selection(name: str, n: int):
        if name == "t":
            return slice(0, n), (n,)
        if name in {"delta", "omega"}:
            return (slice(0, n), slice(0, 10)), (n, 10)
        if name == "Pm":
            return (2, slice(0, 10)), (10,)
        if name == "inertia" or name in {"equilibrium/delta", "equilibrium/Pm"}:
            return slice(None), (10,)
        raise ReaderError("unregistered dataset selection")

    def _read_array(self, handle: Any, array_key: str, descriptor_key: str, *, n: int | None) -> Any:
        meta = self._meta[array_key]
        descriptor = self._capture(
            descriptor_key,
            lambda: self._describe(self._dataset(handle, self._meta[descriptor_key])),
            element_count=lambda _: 1,
        )
        if descriptor is None:
            self.ledger.skip(array_key, "descriptor_failed", prerequisite_key=descriptor_key)
            return None
        field = meta["selector"]
        selection, expected_shape = self._selection(field, n if n is not None else 0)
        expected_descriptor_shape = (n, 10) if field == "Pm" and n is not None else expected_shape
        shape, dtype_text = descriptor
        try:
            dtype = np.dtype(dtype_text)
        except (TypeError, ValueError):
            dtype = None
        if tuple(shape) != tuple(expected_descriptor_shape):
            self.ledger.declare_invalid_descriptor(descriptor_key, "descriptor_shape_mismatch")
            self.ledger.skip(array_key, "descriptor_invalid", prerequisite_key=descriptor_key)
            return None
        if dtype is None or dtype.kind != "f" or not np.can_cast(dtype, np.dtype("float64"), casting="safe"):
            self.ledger.declare_invalid_descriptor(descriptor_key, "descriptor_dtype_unsafe")
            self.ledger.skip(array_key, "descriptor_invalid", prerequisite_key=descriptor_key)
            return None

        def read_payload():
            dataset = self._dataset(handle, meta)
            data = np.asarray(dataset[selection])
            if tuple(data.shape) != tuple(expected_shape):
                raise ValueError("returned payload shape differs from registered selection")
            return data.astype(np.float64, copy=False)

        return self._capture(array_key, read_payload, element_count=lambda value: int(value.size))

    @staticmethod
    def _describe(dataset: Any) -> tuple[tuple[int, ...], str]:
        return tuple(int(x) for x in dataset.shape), np.dtype(dataset.dtype).str

    def _on_open_failure(self, source: str, phase: int, error: Exception) -> None:
        keys = self._phase_keys[(source, phase)]
        if not keys:
            return
        first = keys[0]

        def fail():
            raise error

        self._capture(first, fail, element_count=lambda _: 1)
        for key in keys[1:]:
            self.ledger.skip(key, "source_open_failed", prerequisite_key=first)

    def read_phase1_source(self, source: str) -> dict[str, Any]:
        if source not in ALLOWED_SOURCES:
            raise ReaderError("unregistered source")
        phase_key = (source, 1)
        if phase_key in self._opened:
            raise ReaderError("source phase has already been opened")
        self._opened.add(phase_key)
        if self.ledger.phase != 1:
            self.ledger.call(self._phase_keys[(source, 1)][0], lambda: None)
        try:
            context = self.file_opener(self.source_paths[source], "r")
        except Exception as exc:  # noqa: BLE001 - record any HDF open failure and prohibit retry.
            self._on_open_failure(source, 1, exc)
            return {}
        try:
            handle = context.__enter__()
        except Exception as exc:  # noqa: BLE001 - record any HDF enter failure and prohibit retry.
            self._on_open_failure(source, 1, exc)
            return {}
        try:
            result: dict[str, Any] = {}
            for attr in ROOT_ATTRS:
                key = _selector_key("root-attr", source, attr)
                value = self._capture(key, lambda a=attr: handle.attrs[a], element_count=lambda _: 1)
                if key in self._values:
                    result[key] = value
            if self.root_validator is not None:
                self.root_validator(source, self._readonly_named_values(source, result, "root_attr"))
            for name in STATIC_ARRAYS:
                desc = _selector_key("descriptor", source, name)
                arr = _selector_key("static-array", source, name)
                value = self._read_array(handle, arr, desc, n=None)
                if arr in self._values:
                    result[arr] = value
            if self.static_validator is not None:
                root_and_static = self._readonly_named_values(source, result, "root_attr", "static_array")
                self.static_validator(source, root_and_static)
            for row in self.rows_by_source[source]:
                event_id, n = row["event_id"], row["declared_length"]
                for name in EVENT_ARRAYS:
                    desc = _selector_key("descriptor", source, name, event_id)
                    arr = _selector_key("event-array", source, name, event_id)
                    if name == "Pm" and n < 3:
                        # Descriptor metadata still records shape/dtype; the row index 2 is unavailable.
                        descriptor = self._capture(desc, lambda m=self._meta[desc]: self._describe(self._dataset(handle, m)), element_count=lambda _: 1)
                        if descriptor is None:
                            self.ledger.skip(arr, "descriptor_failed", prerequisite_key=desc)
                        else:
                            self.ledger.declare_invalid_descriptor(desc, "origin_power_row_unavailable")
                            self.ledger.skip(arr, "descriptor_invalid", prerequisite_key=desc)
                        continue
                    value = self._read_array(handle, arr, desc, n=n)
                    if arr in self._values:
                        result[arr] = value
        except BaseException as exc:
            context.__exit__(type(exc), exc, exc.__traceback__)
            raise
        context.__exit__(None, None, None)
        return result

    def _readonly_named_values(self, source: str, result: Mapping[str, Any], *kinds: str) -> Mapping[str, Any]:
        values: dict[str, Any] = {}
        allowed = set(kinds)
        for key, value in result.items():
            meta = self._meta[key]
            if meta["source"] != source or meta["kind"] not in allowed:
                continue
            named_key = f"{meta['selector']}" if meta["kind"] == "root_attr" else f"{meta['selector']}[:]"
            if isinstance(value, np.ndarray):
                readonly_copy = value.copy()
                readonly_copy.setflags(write=False)
                value = readonly_copy
            values[named_key] = value
        return MappingProxyType(values)

    def seal_prelabel_freeze(self, sha256: str) -> None:
        self.ledger.seal_prelabel_freeze(sha256)

    def read_phase2_source(self, source: str) -> dict[str, Any]:
        if source not in ALLOWED_SOURCES:
            raise ReaderError("unregistered source")
        phase_key = (source, 2)
        if phase_key in self._opened:
            raise ReaderError("source phase has already been opened")
        self._opened.add(phase_key)
        if self.ledger.phase != 2:
            first = self._phase_keys[(source, 2)][0]
            self.ledger.call(first, lambda: None)
        try:
            context = self.file_opener(self.source_paths[source], "r")
        except Exception as exc:  # noqa: BLE001 - record any HDF open failure and prohibit retry.
            self._on_open_failure(source, 2, exc)
            return {}
        try:
            handle = context.__enter__()
        except Exception as exc:  # noqa: BLE001 - record any HDF enter failure and prohibit retry.
            self._on_open_failure(source, 2, exc)
            return {}
        try:
            result: dict[str, Any] = {}
            for row in self.rows_by_source[source]:
                event_id, raw_id = row["event_id"], row["raw_id"]
                for attr in EVENT_ATTRS:
                    key = _selector_key("event-attr", source, attr, event_id)
                    value = self._capture(
                        key,
                        lambda a=attr, rid=raw_id: handle["events"][rid].attrs[a],
                        element_count=lambda _: 1,
                    )
                    if key in self._values:
                        result[key] = value
        except BaseException as exc:
            context.__exit__(type(exc), exc, exc.__traceback__)
            raise
        context.__exit__(None, None, None)
        return result

    def audit_log(self) -> dict[str, Any]:
        """Return selector accounting without file handles or source contents."""
        ledger = self.ledger.snapshot()
        for selector in ledger.get("selectors", []):
            meta = self._meta.get(selector["key"], {})
            selector["source"] = meta.get("source")
            selector["event_id"] = meta.get("event_id")
            selector["selector"] = meta.get("selector")
            if selector["key"] in self._errors:
                selector["read_error"] = dict(self._errors[selector["key"]])
        return ledger
