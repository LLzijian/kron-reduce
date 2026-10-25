"""Independent A17 saved-evidence audit, confined to one newly sealed run.

No source provider, model, initializer, solver, RHS or production metric is
imported. Metric functions below replay only an already saved comparison.
The public run audit requires externally supplied manifest/control hashes.
It never writes into the run and never follows artifact paths outside it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import zipfile
from collections import Counter
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

import numpy as np

AUTH = "kiki_a17_development_20261025"
SEEDS = (324, 325, 326)
STEPS = (("h005", .005), ("h0025", .0025))
FIELDS = ("primary", "q", "v", "horizon1", "horizon3")
ATOL = 1e-12
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
CONDITIONS = frozenset({"primary_strictly_lower", "horizon1_not_higher", "horizon3_not_higher",
                       "q_not_higher", "v_not_higher", "worst10_cvar_not_higher",
                       "complete_3s_horizon3_conditional_mse_not_higher"})


def _finite(value):
    if isinstance(value, Mapping):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    return not isinstance(value, (float, np.floating)) or math.isfinite(float(value))


def _finite_result(value):
    if not _finite(value):
        raise FloatingPointError("nonfinite saved-metric replay result")
    return value


def _array(value, shape=None):
    result = np.asarray(value)
    if (result.dtype.kind not in "fiu" or (shape is not None and result.shape != shape)
            or not np.isfinite(result).all()):
        raise ValueError("saved replay array violates finite real shape contract")
    return np.asarray(result, dtype=np.float64)


def replay_event_scores(prediction, truth, clock, target_scale):
    """Replay the frozen visible MSE from saved arrays; origin is excluded."""
    time = _array(clock)
    pred, target, scale = _array(prediction), _array(truth), _array(target_scale, (20,))
    if (time.ndim != 1 or not 2 <= len(time) <= 301 or time[0] != 0
            or np.any(np.diff(time) <= 0) or time[-1] > 3 + ATOL
            or pred.shape != (len(time), 22) or target.shape != (len(time), 20)
            or np.any(scale <= 0) or not np.array_equal(pred[0, :20], target[0])):
        raise ValueError("saved replay origin, clock or scale differs")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        normalized = (pred[1:, :20] - target[1:]) / scale
        horizons = {}
        for key, end in (("horizon1", 1.), ("horizon3", 3.)):
            keep = time[1:] <= end + ATOL
            if not np.any(keep):
                raise ValueError("saved horizon contains no scored point")
            q_mse = float(np.mean(normalized[keep, :10] ** 2))
            v_mse = float(np.mean(normalized[keep, 10:] ** 2))
            horizons[key] = {"q": q_mse, "v": v_mse, "all": .5 * q_mse + .5 * v_mse,
                             "scored_points": int(np.count_nonzero(keep))}
    first, third = horizons["horizon1"], horizons["horizon3"]
    return _finite_result({"primary": .5 * (first["all"] + third["all"]),
                           "q": .5 * (first["q"] + third["q"]), "v": .5 * (first["v"] + third["v"]),
                           "horizon1": first["all"], "horizon3": third["all"],
                           "future_points": len(time) - 1, "horizons": horizons})


def replay_population_scores(records, weights, counts):
    """Original weights and failures retained; only named full3s mean conditions."""
    n = len(records)
    weight = _array(weights, (n,))
    count = np.asarray(counts)
    if (n == 0 or count.shape != (n,) or count.dtype.kind not in "iu"
            or np.any(count < 1) or np.any(count > 300) or np.any(weight <= 0)
            or abs(float(np.sum(weight)) - 1.) > ATOL):
        raise ValueError("saved population/weight contract differs")
    available = [index for index, row in enumerate(records) if row is not None]
    for index in available:
        row = records[index]
        if (row.get("future_points") != count[index]
                or any(not isinstance(row.get(key), (int, float, np.number))
                       or not math.isfinite(float(row[key])) or row[key] < 0 for key in FIELDS)):
            raise ValueError("saved event score is invalid or belongs to another population")
    def contribution(indices, key):
        # NumPy scalar left-to-right accumulation is the frozen production order.
        # Python 3.12's sum(float generator) uses a different compensated path.
        total = np.float64(0.)
        for index in indices:
            total = total + weight[index] * np.float64(records[index][key])
        return float(total)

    partial = {key: contribution(available, key) for key in FIELDS}
    present = np.array([row is not None for row in records], dtype=bool)
    complete = len(available) == n
    groups = {}
    for key, mask in (("under_1s", count < 100), ("from_1s_under_3s", (count >= 100) & (count < 300)),
                      ("full_3s", count == 300)):
        groups[key] = {"events": int(np.count_nonzero(mask)),
                       "original_weight": float(np.sum(weight[mask])),
                       "missing_weight": float(np.sum(weight[mask & ~present])),
                       "partial_primary_contribution_same_denominator": contribution(
                           (i for i in available if mask[i]), "primary")}
    full = count == 300
    mass = float(np.sum(weight[full]))
    conditional = (contribution((i for i in available if full[i]), "horizon3") / mass
                   if mass > 0 and bool(np.all(present[full])) else None)
    tail = None
    if complete:
        remaining, total = .1, 0.
        # Python's stable sort preserves original event order for exact ties.
        for index in sorted(range(n), key=lambda i: -float(records[i]["primary"])):
            take = min(float(weight[index]), remaining)
            total += take * float(records[index]["primary"])
            remaining -= take
            if remaining <= 8 * np.finfo(np.float64).eps * .1:
                break
        if remaining > 8 * np.finfo(np.float64).eps * .1:
            raise ValueError("saved weights do not cover the frozen CVaR tail")
        tail = total / .1
    return _finite_result({"complete": complete, "allocated_eligible_events": n,
                           "completed_events": len(available), "missing_weight": float(np.sum(weight[~present])),
                           "partial_contributions_same_denominator": partial,
                           "metrics": partial if complete else None, "worst10_cvar": tail,
                           "complete_3s_horizon3_conditional_mse": conditional, "duration_groups": groups})


def replay_paired_group(candidate, reference):
    """Replay seven fixed conditions from saved population metrics, no selection."""
    if type(candidate.get("complete")) is not bool or type(reference.get("complete")) is not bool:
        raise ValueError("saved paired completeness is not a boolean")
    _finite_result(candidate)
    _finite_result(reference)
    if not candidate["complete"] or not reference["complete"]:
        return {"complete": False, "pass": False, "reason": "original_population_incomplete",
                "conditions": None, "differences": None}
    c, r = candidate["metrics"], reference["metrics"]
    conditions = {"primary_strictly_lower": c["primary"] < r["primary"] - ATOL}
    differences = {key: c[key] - r[key] for key in FIELDS}
    for key in ("horizon1", "horizon3", "q", "v"):
        conditions[key + "_not_higher"] = c[key] <= r[key] + ATOL
    for key in ("worst10_cvar", "complete_3s_horizon3_conditional_mse"):
        cv, rv = candidate[key], reference[key]
        conditions[key + "_not_higher"] = cv is not None and rv is not None and cv <= rv + ATOL
        differences[key] = None if cv is None or rv is None else cv - rv
    return _finite_result({"complete": True, "pass": bool(all(conditions.values())),
                           "conditions": conditions, "differences": differences, "comparison_atol": ATOL})


def replay_campaign_verdict(V_pass, groups):
    expected = {(ref, seed, step) for ref in ("I", "E") for seed in SEEDS for _code, step in STEPS}
    if type(V_pass) is not bool or set(groups) != expected:
        raise ValueError("saved verdict lacks twelve exact paired groups or strict V boolean")
    for row in groups.values():
        if (not isinstance(row, dict) or type(row.get("complete")) is not bool
                or type(row.get("pass")) is not bool):
            raise ValueError("saved paired verdict requires strict booleans")
        _finite_result(row)
        if row["complete"]:
            gates = row.get("conditions")
            if (not isinstance(gates, dict) or set(gates) != CONDITIONS
                    or any(type(value) is not bool for value in gates.values())
                    or row["pass"] != all(gates.values())):
                raise ValueError("saved paired verdict contradicts its conditions")
        elif row["pass"]:
            raise ValueError("saved incomplete group cannot pass")
    gi = V_pass and all(groups[("I", seed, step)]["pass"] for seed in SEEDS for _c, step in STEPS)
    ge = V_pass and all(groups[("E", seed, step)]["pass"] for seed in SEEDS for _c, step in STEPS)
    return {"V_pass": V_pass, "G_I": gi, "G_E": ge,
            "scientific_verdict": "ACCEPT" if V_pass and gi and ge else "REJECT",
            "promotion_decision": "KEEP_BASELINE", "new_missed_instability_result_established": False}


class EvidenceError(ValueError):
    """Saved evidence is incomplete, inconsistent or outside the run boundary."""


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def _pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, "duplicate JSON object key")
        result[key] = value
    return result


def strict_json(raw):
    def invalid_constant(value):
        raise EvidenceError(f"nonfinite literal in saved JSON: {value}")
    return json.loads(raw, object_pairs_hook=_pairs, parse_constant=invalid_constant)


def canonical_closed(record):
    return (json.dumps(record, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def same_array(left, right, message):
    require(isinstance(left, np.ndarray) and isinstance(right, np.ndarray), message + ": not arrays")
    require(left.shape == right.shape and left.dtype.str == right.dtype.str
            and left.tobytes(order="C") == right.tobytes(order="C"), message)


def same_record(saved, expected, where="record"):
    """Exact scalar/boolean replay; no relaxed acceptance threshold is applied."""
    if isinstance(expected, dict):
        require(isinstance(saved, dict) and set(saved) == set(expected), where + ": fields differ")
        for key in expected:
            same_record(saved[key], expected[key], where + "/" + key)
    elif isinstance(expected, (list, tuple)):
        require(isinstance(saved, (list, tuple)) and len(saved) == len(expected), where + ": length differs")
        for index, value in enumerate(expected):
            same_record(saved[index], value, f"{where}/{index}")
    elif isinstance(expected, np.ndarray):
        same_array(saved, expected, where)
    elif isinstance(expected, bool):
        require(type(saved) is bool and saved is expected, where + ": boolean differs")
    elif isinstance(expected, (int, float, np.number)):
        require(not isinstance(saved, (bool, np.bool_)) and isinstance(saved, (int, float, np.number))
                and math.isfinite(float(saved)) and math.isfinite(float(expected))
                and saved == expected, where + ": numeric value differs")
    else:
        require(saved == expected, where + ": value differs")


class SealedRun:
    """Read only manifest-bound regular files below one explicit run directory."""

    def __init__(self, directory, expected_manifest_sha256):
        supplied = Path(directory).absolute()
        require(not supplied.is_symlink() and not supplied.is_junction(), "linked run root")
        self.root = supplied.resolve(strict=True)
        require(self.root.is_dir(), "run root is not a directory")
        require(isinstance(expected_manifest_sha256, str)
                and HEX64.fullmatch(expected_manifest_sha256) is not None, "external manifest SHA required")
        self.bytes_read = Counter()
        self.opens = Counter()
        self.decoded_members = 0
        self.manifest = {}
        raw = self.raw("artifact_hashes.json", sealed=False)
        self.manifest_size = len(raw)
        require(hashlib.sha256(raw).hexdigest() == expected_manifest_sha256, "external manifest SHA mismatch")
        self.manifest = strict_json(raw)
        require(isinstance(self.manifest, dict) and self.manifest, "empty or invalid manifest")
        require("artifact_hashes.json" not in self.manifest, "manifest cannot hash itself")
        for relative, digest in self.manifest.items():
            self.path(relative)
            require(isinstance(digest, str) and HEX64.fullmatch(digest) is not None, "invalid manifest digest")
        found = set()
        for base, directories, files in os.walk(self.root, followlinks=False):
            for name in directories:
                child = Path(base) / name
                require(not child.is_symlink() and not child.is_junction(), "linked artifact directory")
            for name in files:
                relative = (Path(base) / name).relative_to(self.root).as_posix()
                self.path(relative)
                found.add(relative)
        require(found == set(self.manifest) | {"artifact_hashes.json"}, "unsealed or absent artifact file")
        self.sizes = {}
        for relative, digest in self.manifest.items():
            measured, size = self.hash_file(relative)
            require(measured == digest, f"sealed file SHA differs: {relative}")
            self.sizes[relative] = size

    def path(self, relative):
        require(isinstance(relative, str) and relative and "\\" not in relative and ":" not in relative
                and not relative.startswith("/") and "//" not in relative
                and all(part not in {"", ".", ".."} for part in relative.split("/")), "unsafe artifact path")
        target = self.root.joinpath(*PurePosixPath(relative).parts)
        for component in (target, *target.parents):
            if component == self.root:
                break
            require(not component.is_symlink() and not component.is_junction(), "linked artifact path")
        try:
            target.resolve(strict=True).relative_to(self.root)
        except (ValueError, OSError) as error:
            raise EvidenceError("artifact escaped or is absent from run") from error
        require(target.is_file(), "artifact is not a regular file")
        return target

    def hash_file(self, relative):
        digest, count = hashlib.sha256(), 0
        with self.path(relative).open("rb") as stream:
            self.opens[relative] += 1
            while block := stream.read(1_048_576):
                digest.update(block)
                count += len(block)
        self.bytes_read[relative] += count
        return digest.hexdigest(), count

    def raw(self, relative, *, sealed=True):
        if sealed:
            require(relative in self.manifest, "artifact is not in seal")
        raw = self.path(relative).read_bytes()
        self.opens[relative] += 1
        self.bytes_read[relative] += len(raw)
        if sealed:
            require(hashlib.sha256(raw).hexdigest() == self.manifest[relative], "artifact changed while auditing")
        return raw

    def json(self, relative):
        return strict_json(self.raw(relative))

    def record(self, name, *, decode=True):
        require(isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9_-]+", name) is not None,
                "invalid flat saved-record name")
        metadata = self.json(name + ".json")
        require(name + ".npz" in self.manifest, "saved record lacks its sealed NPZ")
        if not decode:
            return metadata
        descriptors = {}

        def find(value, prefix):
            if isinstance(value, dict):
                # Old-source metadata may itself contain array descriptors. Only
                # the exact pack traversal prefix identifies a new-run array.
                if value.get("array_member") == prefix:
                    require(set(value) == {"array_member", "shape", "dtype", "sha256_c_order_bytes"},
                            "malformed new-run array descriptor")
                    require(prefix not in descriptors, "duplicate new-run array descriptor")
                    descriptors[prefix] = value
                else:
                    for key, item in value.items():
                        find(item, prefix + "__" + key)
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    find(item, prefix + "__" + str(index))

        find(metadata, "root")
        filename = name + ".npz"
        target = self.path(filename)
        measured, _size = self.hash_file(filename)
        require(measured == self.manifest[filename], "saved archive changed while auditing")
        arrays = {}
        with zipfile.ZipFile(target) as archive:
            self.opens[filename] += 1
            names = archive.namelist()
            require(len(names) == len(set(names)) and set(names) == {key + ".npy" for key in descriptors},
                    "NPZ inventory differs from saved-record descriptors")
        with np.load(target, allow_pickle=False, max_header_size=10_000) as archive:
            self.opens[filename] += 1
            for member, descriptor in descriptors.items():
                shape = descriptor["shape"]
                require(isinstance(shape, list) and all(type(x) is int and x >= 0 for x in shape),
                        "invalid saved-array shape")
                value = archive[member]
                require(not value.dtype.hasobject and list(value.shape) == shape
                        and value.dtype.str == descriptor["dtype"], "saved-array dtype/shape differs")
                require(hashlib.sha256(value.tobytes(order="C")).hexdigest()
                        == descriptor["sha256_c_order_bytes"], "saved-array logical SHA differs")
                arrays[member] = value
                self.decoded_members += 1

        def unpack(value, prefix):
            if prefix in arrays:
                return arrays[prefix]
            if isinstance(value, dict):
                return {key: unpack(item, prefix + "__" + key) for key, item in value.items()}
            if isinstance(value, list):
                return [unpack(item, prefix + "__" + str(index)) for index, item in enumerate(value)]
            return value

        return unpack(metadata, "root")


def audit_scope_transitions(rows, operators, status, scope_id):
    """Rebuild real call counts from one saved byte range; never run a callback."""
    require(status in {"complete", "numerical_failure", "prerequisite_failed"}, "invalid scope status")
    require(all(type(limit) is int and limit >= 0 for limit in operators.values()), "invalid operator caps")
    counts = {key: Counter() for key in ("attempted", "returned", "failed")}
    stack, seen, failed_seen, line_count = [], set(), False, 0
    for row in rows:
        line_count += 1
        category, index, transition = row.get("category"), row.get("index"), row.get("transition")
        require(row.get("scope_id") == scope_id and category in operators
                and type(index) is int and index >= 0, "operation identity outside frozen scope")
        key = (category, index)
        if transition == "attempted":
            require(not failed_seen, "scientific callback attempted after a scope failure")
            require(key not in seen and index == counts["attempted"][category], "repeated/skipped operation index")
            require(counts["attempted"][category] < operators[category], "operation exceeds cap")
            stack.append(key)
            seen.add(key)
        else:
            require(transition in {"returned", "failed"} and stack and stack[-1] == key,
                    "unmatched or out-of-order operation terminal")
            stack.pop()
            failed_seen |= transition == "failed"
        counts[transition][category] += 1
    require(not stack, "unclosed operation in a closed scope")
    if status == "complete":
        require(all(counts["attempted"][key] == limit == counts["returned"][key]
                    and counts["failed"][key] == 0 for key, limit in operators.items()),
                "complete scope lacks exact attempted/returned caps")
    elif status == "prerequisite_failed":
        require(line_count == 0, "failed prerequisite performed scientific calls")
    # Numerical gates may fail outside a nested callback after earlier returns.
    # Their actual prefix is retained; this does not manufacture a failed call.
    return {key: dict(value) for key, value in counts.items()}, line_count


def audit_accounting(specification, record, closed):
    """Check derived facts and diagnostic array operations, not scalar primitives."""
    status = record["status"]
    derived, diagnostics = record.get("derived_actual"), record.get("diagnostic_actual")
    require(isinstance(derived, dict) and isinstance(diagnostics, dict), "missing scope accounting")
    require(derived == closed.get("derived_actual") and diagnostics == closed.get("diagnostic_actual"),
            "scope body and closure accounting differ")
    expected = specification["derived_limits"]
    diagnostic_limits = specification["diagnostic_operation_limits"]
    require(set(derived) <= set(expected) and set(diagnostics) <= set(diagnostic_limits),
            "unregistered derived/diagnostic category")
    for key, value in derived.items():
        require(type(value) is int and 0 <= value <= expected[key], "derived count exceeds cap")
    for category, actual in diagnostics.items():
        limits = {f"{key}_{state}": limit for key, limit in diagnostic_limits[category].items()
                  for state in ("attempted", "returned")}
        require(set(actual) == set(limits), "diagnostic counter fields differ")
        require(all(type(value) is int and 0 <= value <= limits[key] for key, value in actual.items()),
                "diagnostic count exceeds cap")
        require(all(actual[key + "_returned"] <= actual[key + "_attempted"]
                    for key in diagnostic_limits[category]), "diagnostic returned exceeds attempted")
    if status == "complete":
        require(derived == expected and set(diagnostics) == set(diagnostic_limits), "complete accounting incomplete")
        for category, limits in diagnostic_limits.items():
            require(diagnostics[category] == {f"{key}_{state}": value for key, value in limits.items()
                                             for state in ("attempted", "returned")}, "complete diagnostics incomplete")
    elif status == "prerequisite_failed":
        require(not any(derived.values()) and not any(any(v.values()) for v in diagnostics.values()),
                "prerequisite failure has actual derived/diagnostic work")
    value = record["value"]
    evidence = value.get("input_consistency_campaign_partial", value)
    reconstructed = Counter(evidence.get("derived_actual", {}))
    reconstructed_diagnostics = {}
    for helper in evidence.get("helpers", {}).values():
        reconstructed.update(helper.get("derived_actual", {}))
        local = dict(helper.get("diagnostic_operation_counts", {}))
        if "operation_counts" in helper:
            require(helper["category"] not in local, "double-counted refinement operations")
            local[helper["category"]] = helper["operation_counts"]
        require(not set(local) & set(reconstructed_diagnostics), "double-counted diagnostic helper")
        reconstructed_diagnostics.update(local)
    require(dict(reconstructed) == derived and reconstructed_diagnostics == diagnostics,
            "scope accounting differs from actual helper evidence")


def _reader_rows(plan, freezes):
    def pair(key, operation, phase, details=None):
        for transition in ("attempted", "returned"):
            yield {"key": key, "operation": operation, "phase": phase,
                   "transition": transition, "details": details}
    for row in plan["metadata_files"]:
        yield from pair("metadata::" + row["path"], "metadata", 1, {"size_bytes": row["size_bytes"]})
    files = {row["path"]: row for row in plan["contexts"]}
    for path, row in files.items():
        yield from pair("hash::before::" + path, "opaque_hash", 1, {"size_bytes": row["size_bytes"]})
    for phase in (1, 2, 3):
        for row in plan["contexts"]:
            if row["phase"] != phase:
                continue
            opening = {"key": "context::" + row["context_id"], "operation": "npz_context", "phase": phase,
                       "details": {"path": row["path"], "members": len(row["members"])}}
            yield {**opening, "transition": "attempted"}
            for member in row["members"]:
                key = row["path"] + "::" + member["name"]
                yield from pair(key + "::descriptor", "descriptor", phase)
                yield from pair(key + "::array", "array", phase, {"elements": math.prod(member["shape"])})
            yield {**opening, "transition": "returned"}
        if phase < 3:
            yield from pair(f"phase::{phase + 1}", "candidate_freeze", phase, freezes[phase + 1])
    for path, row in files.items():
        yield from pair("hash::after::" + path, "opaque_hash", 3, {"size_bytes": row["size_bytes"]})


def audit_reader_journal(actual_rows, plan, freezes):
    """Compare the exact finite reader schedule, including both freeze terminals."""
    iterator, count, returned, selectors = iter(actual_rows), 0, Counter(), set()
    for expected in _reader_rows(plan, freezes):
        try:
            actual = next(iterator)
        except StopIteration as error:
            raise EvidenceError("reader journal is incomplete") from error
        require(set(actual) == set(expected), "reader transition fields differ")
        if expected["operation"] == "candidate_freeze":
            # The recorded absolute path is data, never a file-open instruction.
            details = dict(actual["details"])
            path = details.pop("path", None)
            wanted = dict(expected["details"])
            filename = wanted.pop("path")
            require(isinstance(path, str) and path.replace("\\", "/").split("/")[-1] == filename,
                    "reader freeze path name differs")
            require(details == wanted, "reader freeze SHA or bytes differ")
            actual = {**actual, "details": expected["details"]}
        require(actual == expected, "reader selector/order/phase/transition differs")
        count += 1
        if actual["transition"] == "returned":
            returned[actual["operation"]] += 1
            selectors.add(actual["key"])
    require(next(iterator, None) is None, "reader journal has unregistered trailing reads")
    return {"transition_records": count, "returned_operations": dict(returned), "selector_keys": selectors}


def _stream_jsonl(store, relative):
    require(relative in store.manifest, "unsealed journal")
    digest = hashlib.sha256()
    with store.path(relative).open("rb") as stream:
        store.opens[relative] += 1
        for raw in stream:
            store.bytes_read[relative] += len(raw)
            digest.update(raw)
            require(raw.endswith(b"\n"), "unterminated saved journal line")
            yield strict_json(raw)
    require(digest.hexdigest() == store.manifest[relative], "journal changed during audit")


def audit_scope_store(store, compute):
    """Stream scientific ranges once and reconstruct the entire fixed scope DAG."""
    order, specifications = compute["scope_order"], compute["scopes"]
    require(len(order) == len(set(order)) and set(order) == set(specifications), "scope schedule differs")
    require([specifications[name]["phase"] for name in order]
            == sorted(specifications[name]["phase"] for name in order), "scientific phases are interleaved")
    saved_index = store.json("scientific_scopes.json")
    require(isinstance(saved_index, list) and len(saved_index) == len(order), "scope index is incomplete")
    statuses, closures, prefixes = {}, [], {}
    chain, stream_digest, previous_end, total_lines = hashlib.sha256(), hashlib.sha256(), 0, 0
    totals = {key: Counter() for key in ("attempted", "returned", "failed")}
    with store.path("operations/calls.jsonl").open("rb") as journal:
        store.opens["operations/calls.jsonl"] += 1
        for ordinal, identity in enumerate(order):
            specification = specifications[identity]
            reserved = store.json(f"operations/{identity}-reserved.json")
            closed = store.json(f"operations/{identity}-closed.json")
            record = store.record(identity, decode=False)
            require(record.get("scope_id") == identity == closed.get("scope_id") == reserved.get("scope_id"),
                    "scope artifact identity differs")
            status = record["status"]
            require(status == closed["status"] and closed.get("durably_closed") is True,
                    "scope record was not durably closed with its saved status")
            if status == "numerical_failure":
                record = store.record(identity)
                require(record["value"].get("error_type") in {"FloatingPointError", "OverflowError", "LinAlgError"}
                        and record["value"].get("retry_allowed") is False,
                        "local numerical failure lacks its preserved error record")
            require(reserved.get("reservation") == specification["operators"]
                    == reserved.get("reserved_operation_limits") and reserved.get("spent") is True
                    and reserved.get("retry_allowed") is False, "scope reservation differs")
            start, end = closed["start_byte_offset"], closed["end_byte_offset"]
            require(type(start) is int and type(end) is int and previous_end == start <= end
                    and reserved["start_byte_offset"] == start, "scope byte ranges have a gap or overlap")
            require(end <= store.sizes["operations/calls.jsonl"], "scope range exceeds journal")
            dependencies = specification["prerequisite_scope_ids"]
            require(all(key in statuses for key in dependencies), "missing or forward scope dependency")
            failed = [key for key in dependencies if statuses[key] != "complete"]
            require(record.get("dependencies") == dependencies and record.get("failed_dependencies") == failed
                    and closed.get("failed_dependencies") == failed, "saved dependency evidence differs")
            required_failure = bool(failed) and specification["dependency_mode"] == "all_success"
            require((status == "prerequisite_failed") is required_failure, "failed dependency was retried or ignored")
            if status == "prerequisite_failed":
                require(record["value"] == {"failed_dependencies": failed, "retry_allowed": False},
                        "failed prerequisite body differs")
            require(record.get("retry_allowed") is False, "retry flag differs")
            hashes = {identity + suffix: store.manifest[identity + suffix] for suffix in (".json", ".npz")}
            require(closed.get("artifact_hashes") == hashes, "closure did not bind saved JSON and NPZ")
            digest = hashlib.sha256()

            def lines(end=end, identity=identity, digest=digest):
                while journal.tell() < end:
                    raw = journal.readline(end - journal.tell() + 1)
                    require(raw and raw.endswith(b"\n") and journal.tell() <= end,
                            f"invalid byte-range line in {identity}")
                    store.bytes_read["operations/calls.jsonl"] += len(raw)
                    digest.update(raw)
                    stream_digest.update(raw)
                    yield strict_json(raw)

            counters, line_count = audit_scope_transitions(lines(), specification["operators"], status, identity)
            require(digest.hexdigest() == closed["range_sha256"], "scope durable range SHA differs")
            require(counters == closed["actual_counters"], "scope counters differ from saved transition bytes")
            audit_accounting(specification, record, closed)
            require(closed == saved_index[ordinal], "scope closure index differs from actual closure file")
            for key, values in counters.items():
                totals[key].update(values)
            statuses[identity] = status
            closures.append(closed)
            total_lines += line_count
            previous_end = end
            chain.update(canonical_closed(closed))
            phase = specification["phase"]
            if ordinal + 1 == len(order) or specifications[order[ordinal + 1]]["phase"] != phase:
                prefixes[phase] = {"scope_count": ordinal + 1, "end_byte_offset": end,
                                   "closed_scope_chain_sha256": chain.hexdigest()}
        require(not journal.read(1), "unregistered trailing scientific operations")
    require(stream_digest.hexdigest() == store.manifest["operations/calls.jsonl"],
            "scientific journal changed during audit")
    return {"statuses": statuses, "closures": closures, "phase_prefixes": prefixes,
            "transition_records": total_lines, "actual_counters": {key: dict(value) for key, value in totals.items()}}


def audit_freezes(store, plan, compute, receipt, scope_audit):
    records = plan["freeze_contract"]["slot_ids"]
    result = {}
    for phase, filename, prefix, kind in (
        (2, "origins_freeze.json", "hist", "a17_origins_freeze"),
        (3, "futures_freeze.json", "future", "a17_futures_freeze"),
    ):
        marker = store.json(filename)
        require(marker.get("authorization") == AUTH and marker.get("run_id") == receipt["run_id"]
                and marker.get("source_commit") == receipt["source_commit"] and marker.get("kind") == kind
                and marker.get("population_sha256") == plan["freeze_contract"]["population_sha256"],
                "phase freeze identity differs")
        require(marker.get("scientific_prefix") == scope_audit["phase_prefixes"][phase - 1],
                "phase freeze does not bind the complete preceding scientific prefix")
        expected = []
        for slot in records:
            scope = prefix + "-" + slot
            require(scope in compute["scopes"], "freeze names an unregistered path")
            expected.append({"slot_id": slot, "status": scope_audit["statuses"][scope],
                             "record_path": scope + ".json", "record_sha256": store.manifest[scope + ".json"],
                             "record_npz_sha256": store.manifest[scope + ".npz"]})
        require(marker.get("records") == expected, "phase freeze population/artifact/status differs")
        require(store.sizes[filename] <= plan["freeze_contract"]["max_bytes"], "freeze exceeds bound")
        result[phase] = {"path": filename, "sha256": store.manifest[filename], "size_bytes": store.sizes[filename]}
    return result


def audit_saved_inputs(store, plan):
    """Compare only newly saved copies to registered logical fingerprints."""
    copies = {}
    for phase, name in ((1, "causal_inputs"), (2, "forecast_clocks"), (3, "comparison_inputs")):
        values = store.record(name)
        rows = [row for row in plan["contexts"] if row["phase"] == phase]
        require(set(values) == {row["context_id"] for row in rows}, "saved phase input context inventory differs")
        for row in rows:
            arrays = values[row["context_id"]]
            require(set(arrays) == {member["name"] for member in row["members"]}, "saved input member inventory differs")
            for member in row["members"]:
                value = arrays[member["name"]]
                require(isinstance(value, np.ndarray) and list(value.shape) == member["shape"]
                        and value.dtype.str == member["dtype"] and np.isfinite(value).all(), "saved input shape/dtype/finite differs")
                require(hashlib.sha256(value.tobytes(order="C")).hexdigest() == member["sha256_c_order_bytes"],
                        "saved input is not its originally registered member")
        copies[phase] = values
    return copies


def audit_access_snapshot(store, plan, reader_audit):
    # Duplicate input arrays in this diagnostic snapshot need no second decode;
    # their archive is sealed and the three canonical phase copies were verified.
    access = store.record("access_ledger", decode=False)
    limits = plan["limits"]
    require(access.get("closed") is True and access.get("failed") is False and access.get("phase") == 3,
            "source reader did not finish normally")
    for field, expected in {"array_reads": limits["arrays"], "descriptors": limits["descriptors"],
                            "array_elements": limits["array_elements"],
                            "decoded_array_elements": limits["array_elements"],
                            "decoded_array_bytes": limits["array_logical_hash_bytes"],
                            "payload_bytes_read": limits["array_logical_hash_bytes"]}.items():
        require(type(access.get(field)) is int and access[field] == expected, "reader scalar/byte count differs: " + field)
    require(access.get("opaque_passes") == ["after", "before"], "reader lacks both integrity passes")
    selectors = access.get("selectors", {})
    require(set(selectors) == reader_audit["selector_keys"]
            and len(selectors) * 2 == reader_audit["transition_records"]
            and all(value == "returned" for value in selectors.values()), "reader selector closure differs")
    partial = access["partial_record"]
    metadata = partial["metadata_byte_returns"]
    require(set(metadata) == {row["path"] for row in plan["metadata_files"]}, "metadata read population differs")
    for row in plan["metadata_files"]:
        require(metadata[row["path"]] == {"size_bytes": row["size_bytes"], "sha256": row["sha256"], "validated": True},
                "actual metadata returned bytes/SHA differ")
    contexts = partial["contexts"]
    require(set(contexts) == {row["context_id"] for row in plan["contexts"]}, "reader context snapshot differs")
    header_bytes = 0
    for row in plan["contexts"]:
        saved = contexts[row["context_id"]]
        for member in row["members"]:
            header, payload = saved["headers"][member["name"]], saved["payloads"][member["name"]]
            amount = header["bytes_read"]
            require(type(amount) is int and 0 < amount <= limits["npy_header_per_member_cap_including_magic_and_length"]
                    and header.get("read_failure_unknown") is False and header.get("status") == "validated",
                    "header read exceeded bound or is unknown")
            parsed = header["parsed"]
            require(parsed["shape"] == member["shape"] and parsed["dtype"] == member["dtype"]
                    and parsed["header_bytes"] == amount and type(parsed["fortran_order"]) is bool,
                    "actual parsed header differs")
            header_bytes += amount
            require(payload.get("status") == "validated" and payload.get("decoded") is True
                    and payload.get("validated") is True and payload.get("finite") is True,
                    "source payload has no validated actual return")
            elements = math.prod(member["shape"])
            require(payload.get("actual_bytes") == payload.get("expected_bytes") == elements * 8
                    and payload.get("decoded_elements") == elements and payload.get("decoded_bytes") == elements * 8
                    and payload.get("actual_sha256_c_order_bytes") == member["sha256_c_order_bytes"],
                    "source payload actual count/SHA differs")
    require(header_bytes == access["header_bytes"] <= limits["npy_header_bytes_max"], "header total differs")
    return {"returned_array_elements": limits["array_elements"], "actual_header_bytes": header_bytes,
            "actual_payload_bytes": access["payload_bytes_read"]}


def audit_saved_provenance(store, plan):
    """Connect compact source proofs to actual saved metadata, never old files."""
    documents = store.record("access_ledger", decode=False)["partial_record"]["metadata_documents"]
    proof = store.record("sources", decode=False)["value"]
    rows = {row["path"]: row for row in plan["metadata_files"]}
    require(set(documents) == set(rows), "saved metadata document whitelist differs")
    require(proof["source_runs"] == plan["source_runs"]
            and proof["source_manifest_sha256"] == plan["source_manifest_sha256"], "source proof run identities differ")
    roots = {key: "results/" + value + "/" for key, value in plan["source_runs"].items()}
    manifests = {key: documents[prefix + "artifact_hashes.json"] for key, prefix in roots.items()}
    for key, prefix in roots.items():
        require(rows[prefix + "artifact_hashes.json"]["sha256"] == plan["source_manifest_sha256"][key],
                "saved source manifest binding differs")
        for path, row in rows.items():
            if path.startswith(prefix) and path != prefix + "artifact_hashes.json":
                require(manifests[key].get(path[len(prefix):]) == row["sha256"], "metadata source manifest chain differs")

    def descriptor(value):
        require(isinstance(value, Mapping), "source array descriptor must be an object")
        require(all(key in value for key in ("shape", "dtype", "sha256_c_order_bytes")),
                "source array descriptor lacks its fingerprint")
        return {key: value[key] for key in ("array_member", "shape", "dtype", "sha256_c_order_bytes", "columns") if key in value}

    def fingerprint(value):
        require(isinstance(value, Mapping), "source array fingerprint must be an object")
        require(all(key in value for key in ("shape", "dtype", "sha256_c_order_bytes")),
                "source array fingerprint lacks required fields")
        return {key: value[key] for key in ("shape", "dtype", "sha256_c_order_bytes")}

    parameter_names = ("mass", "edge_stiffness", "damping", "coupling", "latent_skew", "latent_dissipation")
    contexts, context_docs = {}, {}
    for row in plan["contexts"]:
        identity, path = row["context_id"], row["path"]
        contexts[identity] = row
        require(path.startswith(roots[row["source"]])
                and manifests[row["source"]].get(path[len(roots[row["source"]]):]) == row["sha256"],
                "payload descriptor/source manifest binding differs")
        doc = documents[path.removesuffix(".npz") + ".json"]
        context_docs[identity] = doc
        if identity.startswith("static-"):
            members = row["members"]
            require(isinstance(members, list) and len(members) == len(parameter_names)
                    and all(isinstance(member, Mapping) and isinstance(member.get("name"), str)
                            for member in members)
                    and {member.get("name") for member in members}
                    == {"parameters__" + name for name in parameter_names},
                    identity + ": frozen static member inventory differs")
            parameters = doc.get("parameters")
            require(isinstance(parameters, Mapping)
                    and set(parameters) == {*parameter_names, "backbone"},
                    identity + ": saved static parameters schema differs")
            require(parameters["backbone"] == "sine",
                    identity + ": saved static backbone differs from the fixed sine family")
            for name in parameter_names:
                value = parameters[name]
                require(isinstance(value, Mapping)
                        and value.get("array_member") == "parameters__" + name,
                        identity + ": required parameter descriptor differs: " + name)
        for member in row["members"]:
            name = member["name"]
            if identity.startswith("weighted-"):
                actual = doc["values"]["weighted_impulse"]
            elif identity.startswith("impulse-"):
                actual = doc["u11"][int(name.removeprefix("root__u11__"))]
            elif identity.startswith("static-"):
                actual = doc["parameters"][name.removeprefix("parameters__")]
            else:
                actual = doc[name]
            require(fingerprint(actual) == fingerprint(member), "saved source descriptor differs from planned read")
            if "array_member" in actual:
                require(actual["array_member"] == name, "saved source member name differs from actual selector")
    a14, a15, a16 = (roots[key] for key in ("A14", "A15", "A16"))
    population_fields = ("allocated_slots", "cache_rows", "weights", "seed_event_slots")
    for source_population in (documents[a14 + "allocated_population.json"], documents[a15 + "allocated_population.json"],
                              documents[a16 + "allocated_slots.json"]["original_population"]):
        require(all(source_population[key] == plan["population"][key] for key in population_fields),
                "saved ancestor population/weights differ")
    control = documents[plan["ancestry_control"]]
    require(documents[a16 + "frozen_plan.json"]["read_plan"] == control
            and control["events"] == plan["population"]["cache_rows"]
            and control["source_run_id"] == plan["source_runs"]["A14"]
            and control["source_manifest_sha256"] == plan["source_manifest_sha256"]["A14"], "A16 executed source chain differs")
    a16_inventory = {row["path"]: row for row in control["npz_files"]}
    allocated = documents[a16 + "allocated_slots.json"]["diagnostic_slots"]
    expected_slots = plan["population"]["seed_event_slots"]
    require(len(allocated) == len(expected_slots) == len(proof["slot_bindings"]), "source slot population differs")
    for original, allocation in zip(expected_slots, allocated, strict=True):
        seed, index = original["seed"], original["cache_index"]
        key = f"{seed}-{index:03d}"
        saved = proof["slot_bindings"][key]
        require(all(saved[name] == allocation[name] == original[name] for name in ("seed", "cache_index", "event_id"))
                and saved["complete"] is allocation["complete"] is True, "source seed/event/slot identity differs")
        integral = context_docs["integral-" + key]
        weighted = context_docs["weighted-" + key]
        impulse = context_docs[f"impulse-{index:03d}"]
        integral_context = contexts["integral-" + key]
        inherited = a16_inventory[integral_context["path"]]
        require(inherited["sha256"] == integral_context["sha256"], "A16 Gram used a different original integral archive")
        require(any(row["name"] == "Ltilde" for row in inherited["members"]), "A16 did not read original same-slot B")
        expected_parameters = {"rank": integral["rank"], "threshold": integral["svd"]["threshold"],
                               "tol": integral["tol"], "natural_norm": integral["natural_norm"]}
        same_record(saved["integral_parameters"], expected_parameters, "same-slot original decomposition metadata")
        require(integral["rank"] == integral["svd"]["rank"] and integral["status"] == "complete"
                and weighted["complete"] is True and weighted["status"] == allocation["status"] == saved["status"],
                "source completion/rank/status differs")
        for name in ("trace", "determinant"):
            same_record(saved[name], weighted["values"][name], "same-slot A16 " + name)
            interval = saved[name]
            require(0 < interval["lower"] <= interval["point"] <= interval["upper"], "source Gram interval is not positive ordered")
        same_record(saved["integral_descriptors"], {row["name"]: descriptor(integral[row["name"]]) for row in integral_context["members"]}, "integral source descriptors")
        same_record(saved["weighted_impulse"], descriptor(weighted["values"]["weighted_impulse"]), "weighted impulse source")
        same_record(saved["gram"], [descriptor(value) for value in weighted["values"]["gram"]], "Gram source descriptors")
        same_record(saved["impulse"], {name: [descriptor(value) for value in impulse[name]] for name in ("u11", "d")}, "impulse source descriptors")
        same_record(saved["source_paths"], {"integral": integral_context["path"].removesuffix(".npz") + ".json",
                    "slot": contexts["weighted-" + key]["path"].removesuffix(".npz") + ".json",
                    "impulse": contexts[f"impulse-{index:03d}"]["path"].removesuffix(".npz") + ".json"}, "source path identities")
        for code, _step in STEPS:
            doc = context_docs[f"reference-{seed}-{index:03d}-{code}"]
            same_record(saved["future_records"][code], {name: descriptor(doc[name]) for name in ("candidate", "baseline", "reset", "truth") if name in doc}, "same-slot future reference descriptors")
    ancestor_members = documents[a15 + "source_member_fingerprints.json"]
    lineage = proof["ancestry_fingerprints"]
    a14_members = documents[a14 + "source_members.json"]
    input_descriptors = context_docs["inputs"]
    input_ancestor = a14_members["A10/real_train_arrays.npz"]
    expected_input_lineage = {}
    for name, value in input_descriptors.items():
        require(fingerprint(value) == fingerprint(input_ancestor[name]), "A14 causal input ancestor differs")
        expected_input_lineage[name] = {"a14_saved_member": descriptor(value), "ancestor_member": descriptor(input_ancestor[name])}
    same_record(lineage["A14_inputs"], expected_input_lineage, "A14 causal input proof")
    for seed in SEEDS:
        source_key = f"A06/checkpoint-240-{seed}/model.npz"
        static_descriptors = context_docs[f"static-{seed}"]["parameters"]
        source_descriptors = a14_members[source_key]
        # The A14 dataclass record also contains the scalar backbone identity.
        # Only the six registered members are arrays; none may be skipped.
        for name in parameter_names:
            require(fingerprint(static_descriptors[name]) == fingerprint(source_descriptors[name]), "fixed A06 parameter ancestor differs")
        same_record(lineage["A14_static"][str(seed)], {"source_key": source_key,
                    "descriptors": {name: descriptor(static_descriptors[name]) for name in parameter_names},
                    "source_member_descriptors": {name: descriptor(source_descriptors[name]) for name in parameter_names}},
                    "fixed A06 parameter proof")
        for code, _step in STEPS:
            ancestor = "A09" if code == "h005" else "A10"
            relative = f"seed-{seed}-{code}/paths.npz"
            key = ancestor + "/" + relative
            expected = {"source": key, "file_sha256": manifests[ancestor][relative],
                        "future": fingerprint(ancestor_members[key]["future"]), "offsets": fingerprint(ancestor_members[key]["offsets"])}
            same_record(lineage["E"][f"{seed}-{code}"], expected, "E ancestor source proof")
    truth_key = "A10/real_train_arrays.npz"
    same_record(lineage["truth"][truth_key], {"file_sha256": manifests["A10"]["real_train_arrays.npz"],
                "members": {name: fingerprint(ancestor_members[truth_key][name]) for name in ("future", "forecast_time", "offsets", "omega_s")}}, "truth ancestor source proof")
    return {"source_metadata_documents": len(documents), "source_contexts": len(contexts), "same_slot_bindings": len(expected_slots)}


def _qualified(record, name):
    expected_gates = {
        "decomposition_check": {"left_orthogonality", "right_orthogonality", "reconstruction"},
        "ls_diagnostics": {"normal_residual", "conditioning_proxy"},
        "quadrature_proxy": {"inherited_initial_quadrature", "inherited_endpoint_quadrature"},
        "path_refinement": {"visible_refinement", "hidden_refinement"},
        "physical_judgement": {"complete_finite_physical_records", "energy_identity", "q_mass_coi", "v_mass_coi"},
    }
    require(isinstance(record, dict) and record.get("qualified") is True
            and not record.get("reasons") and isinstance(record.get("gates"), dict)
            and set(record["gates"]) == expected_gates[name]
            and all(type(flag) is bool and flag for flag in record["gates"].values())
            and _finite(record.get("values", {})) and not record.get("nonfinite_values"),
            "saved numerical gate did not pass: " + name)
    values = record["values"]
    require(all(not isinstance(value, bool) and isinstance(value, (int, float, np.number))
                and math.isfinite(float(value)) and value >= 0 for value in values.values()),
            "saved diagnostic norm, error, bound or scale is negative/nonfinite")
    tau = 64 * np.finfo(np.float64).eps * 100
    if name == "decomposition_check":
        require(values["tau_num"] == tau and values["left_orthogonality_frobenius"] <= tau
                and values["right_orthogonality_frobenius"] <= tau
                and values["reconstruction_bound"] == tau * max(1., values["B_frobenius"])
                and values["reconstruction_frobenius"] <= values["reconstruction_bound"], "decomposition scalar gates differ")
    elif name == "ls_diagnostics":
        require(values["tau_num"] == tau and values["lambda_lower"] > 0
                and values["determinant_lower"] > 0 and values["trace_upper"] > 0
                and values["gram_ratio_before_outward_rounding"] == values["determinant_lower"] / values["trace_upper"]
                and values["lambda_lower"] == np.nextafter(values["gram_ratio_before_outward_rounding"], -np.inf)
                and values["normal_scale"] == max(1., values["B_frobenius"] * (
                    values["rhs_norm"] + values["B_frobenius"] * values["z0_norm"]))
                and values["normal_residual_bound"] == tau * values["normal_scale"]
                and values["normal_residual_norm"] <= values["normal_residual_bound"]
                and values["conditioning_proxy"] == (values["normal_residual_norm"] + values["normal_residual_bound"]) / values["lambda_lower"]
                and values["conditioning_proxy_bound"] == 1e-6 * (1 + values["z0_norm"])
                and values["conditioning_proxy"] <= values["conditioning_proxy_bound"], "LS saved scalar gates differ")
    elif name == "quadrature_proxy":
        require(values["initial_bound"] == 1e-6 * (1 + values["new_initial_norm"])
                and values["endpoint_bound"] == 1e-6 * (1 + values["new_observed_driven_endpoint_norm"])
                and values["inherited_initial_difference_norm"] <= values["initial_bound"]
                and values["inherited_endpoint_difference_norm"] <= values["endpoint_bound"], "quadrature saved scalar gates differ")
    elif name == "path_refinement":
        require(values["visible_bound"] == 1e-5 and values["visible_max_normalized"] <= 1e-5
                and values["hidden_bound"] == 1e-5 * (1 + values["fine_hidden_max_norm"])
                and values["hidden_max_norm"] <= values["hidden_bound"], "refinement saved scalar gates differ")


def replay_refinement_record(record, coarse, fine, scale):
    """Replay an existing two-step comparison from its saved paths and scale."""
    low, high, scale = _array(coarse), _array(fine), _array(scale, (20,))
    require(low.shape == high.shape and low.ndim == 2 and low.shape[1] == 22
            and 1 <= len(low) <= 301 and np.all(scale > 0), "saved refinement inputs differ")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        visible = (low[:, :20] - high[:, :20]) / scale

        def row_norm(array):
            maximum = np.max(np.abs(array), axis=1)
            divisor = np.where(maximum == 0, 1., maximum)
            return maximum * np.sqrt(np.sum((array / divisor[:, None]) ** 2, axis=1))

        hidden, fine_hidden = row_norm(low[:, 20:] - high[:, 20:]), row_norm(high[:, 20:])
    require(np.isfinite(visible).all() and np.isfinite(hidden).all() and np.isfinite(fine_hidden).all(),
            "saved complete refinement produces a nonfinite replay")
    same_record(record["arrays"], {"visible_normalized_difference": visible, "hidden_difference_norms": hidden,
                                   "fine_hidden_norms": fine_hidden}, "refinement path-to-diagnostic connection")
    same_record(record["values"], {"visible_max_normalized": float(np.max(np.abs(visible))),
                                   "hidden_max_norm": float(np.max(hidden)), "fine_hidden_max_norm": float(np.max(fine_hidden)),
                                   "hidden_bound": 1e-5 * (1. + float(np.max(fine_hidden))), "visible_bound": 1e-5},
                "refinement saved maxima")
    require(record["point_count"] == len(low), "refinement point count differs")
    _qualified(record, "path_refinement")


def audit_input_schedule(schedule, clock, power, *, linear, step):
    """Replay saved PL stage witnesses, never evaluate a model or RHS."""
    require(schedule["index_fields"] == ["interval_index", "substep_index", "substep_count", "numerator", "stage_code"]
            and schedule["stage_codes"] == {"k1": 1, "k2": 2, "k3": 3, "k4": 4}, "PL witness schema differs")
    indices, returned, used = (schedule[key] for key in ("indices", "returned_power", "used_power"))
    require(all(isinstance(value, np.ndarray) and value.dtype.str == "<f8" and np.isfinite(value).all()
                for value in (indices, returned, used)), "PL witness is not finite float64")
    require(indices.ndim == 2 and indices.shape[1] == 5
            and returned.shape == used.shape == (len(indices), 10), "PL witness shapes differ")
    same_array(returned, used, "actual PL callback return differs from input used by RHS")
    if not linear:
        require(len(indices) == 0, "constant future evaluated a PL input")
        return 0
    ordinal = 0
    for interval_index, width in enumerate(np.diff(clock)):
        count = max(1, int(np.ceil(width / step - 1e-12)))
        for substep in range(count):
            for code, numerator in ((1, substep), (2, substep + .5), (3, substep + .5), (4, substep + 1)):
                require(ordinal < len(indices), "PL witness has missing stages")
                same_array(indices[ordinal], np.array([interval_index, substep, count, numerator, code], dtype=np.float64),
                           "PL stage order/index/fraction differs")
                left, right = power[interval_index], power[interval_index + 1]
                expected = left if numerator == 0 else right if numerator == count else left + (numerator / count) * (right - left)
                same_array(returned[ordinal], expected, "PL stage input differs from its two observed endpoints")
                ordinal += 1
    require(ordinal == len(indices), "PL witness has extra stage evaluations")
    return ordinal


def _check_saved_path(helper, clock, power, equilibrium, scale, *, linear, step, mass):
    require(helper["status"] == "complete", "successful scope has incomplete path helper")
    path = helper["path"]
    require(isinstance(path, np.ndarray) and path.shape == (len(clock), 22) and np.isfinite(path).all(),
            "saved complete path is not finite with its registered shape")
    same_array(helper["rollout"]["path"], path, "path duplicate differs")
    audit_input_schedule(helper["rollout"]["input_schedule"], clock, power, linear=linear, step=step)
    inputs = helper["inputs"]
    require(inputs["linear"] is linear and inputs["max_step"] == step, "path input mode/step differs")
    for field, expected in (("clock", clock), ("power", power), ("equilibrium", equilibrium), ("target_scale", scale)):
        same_array(inputs[field], expected, "saved path input differs: " + field)
    same_array(path[0], np.concatenate([inputs["q"], inputs["v"], inputs["z"]]), "path initial state differs")
    points = helper["physical_points"]
    require(len(points) == len(path), "physical check point count differs")
    for index, point in enumerate(points):
        require(point["index"] == index, "physical point order differs")
        raw = point["energy_identity_return"]
        require(set(raw) == {"energy_rate", "supply_minus_dissipation", "identity_error"}, "physical scalar fields differ")
        for field in raw:
            require(point[field] == raw[field] and math.isfinite(float(raw[field])), "physical actual return differs")
        require(raw["identity_error"] == abs(raw["energy_rate"] - raw["supply_minus_dissipation"])
                and 0 <= raw["identity_error"] <= 1e-8, "saved energy identity gate failed")
        require(all(math.isfinite(float(point[key])) and 0 <= point[key] <= 1e-10
                    for key in ("q_mass_coi", "v_mass_coi")), "saved COI gate failed")
        require(point["q_mass_coi"] == abs(float(mass @ path[index, :10]))
                and point["v_mass_coi"] == abs(float(mass @ path[index, 10:20])),
                "saved COI values do not belong to the saved path and fixed mass")
    _qualified(helper["checks"]["physical_judgement"], "physical_judgement")
    judgement = helper["checks"]["physical_judgement"]
    require(judgement["point_count"] == judgement["valid_point_count"] == len(path)
            and judgement["failed_points"] == [] and judgement["values"]["energy_identity_bound"] == 1e-8
            and judgement["values"]["coi_bound"] == 1e-10, "physical judgment scope/bounds differ")
    for field in ("identity_error", "q_mass_coi", "v_mass_coi"):
        require(judgement["values"]["valid_records_max_" + field] == max(row[field] for row in points),
                "physical judgment maxima differ from saved returns")
    require(helper["rollout"]["rk4_updates"] == helper["derived_actual"]["successful_rk4_updates"],
            "saved path update counters differ")
    return path


def audit_saved_scientific_values(store, plan, statuses, copies):
    """Verify saved state connections and replay fixed C/I/E scores only."""
    pop, causal, clocks, comparisons = plan["population"], copies[1], copies[2], copies[3]
    inputs, n = causal["inputs"], len(pop["cache_rows"])
    weights, counts, scale = pop["weights"], pop["future_point_counts"], inputs["target_scale"]
    source = store.record("sources")["value"]
    origins, initial_values, before_parameters = {}, {}, {}
    event_records = {(group, seed, code): [None] * n for group in ("C", "I", "E")
                     for seed in SEEDS for code, _step in STEPS}
    result = Counter()
    parameter_names = ("mass", "edge_stiffness", "damping", "coupling", "latent_skew", "latent_dissipation")
    for seed in SEEDS:
        static_name = f"static-{seed}"
        if statuses[static_name] == "complete":
            static = store.record(static_name)["value"]
            before_parameters[str(seed)] = static["identity"]["parameters"]
            for name in parameter_names:
                saved = static["model"][name]
                original = causal[static_name]["parameters__" + name]
                same_array(saved, original, "constructed parameters differ from saved permitted inputs")
                fingerprint = before_parameters[str(seed)][name]
                require(fingerprint == {"sha256": hashlib.sha256(original.tobytes(order="C")).hexdigest(),
                                        "readonly": True, "shape": list(original.shape)}, "initial parameter identity differs")
            require(len(static["eigvalsh"]) == 3, "structural eigvalsh return population differs")
            require(static["seed"] == seed and static["model"]["backbone"] == "sine", "fixed closure family differs")
            for saved, expected in zip(static["eigvalsh"], (np.zeros((10, 10)), static["model"]["damping"],
                                                            static["model"]["latent_dissipation"]), strict=True):
                same_array(saved["matrix"], expected, "structural eigenvalue call matrix differs")
                require(saved["returned"].shape == (len(expected),) and np.isfinite(saved["returned"]).all()
                        and np.min(saved["returned"]) >= -1e-12, "saved structural PSD gate failed")
        for index in range(n):
            init_name = f"init-{seed}-{index:03d}"
            if statuses[init_name] != "complete":
                continue
            initialization = store.record(init_name)["value"]
            helper = initialization["helpers"]["initialization"]
            require(helper["status"] == "complete", "initializer helper did not complete")
            for category in ("decomposition_check", "ls_diagnostics", "quadrature_proxy"):
                _qualified(helper["checks"][category], category)
            z0 = helper["z0"]
            require(z0.shape == (2,) and np.isfinite(z0).all(), "new initial state is invalid")
            same_array(z0, helper["ls"]["z0"], "initializer differs from its single saved LS return")
            initial_values[(seed, index)] = z0
            integral = causal[f"integral-{seed}-{index:03d}"]
            slot = source["slot_bindings"][f"{seed}-{index:03d}"]
            parameters = slot["integral_parameters"]
            for key in ("rank", "threshold", "natural_norm"):
                same_record(helper["inputs"][key], parameters[key], "original initializer " + key)
            require(type(parameters["rank"]) is int and parameters["rank"] == 2
                    and parameters["threshold"] == parameters["tol"]
                    == helper["expected_original_threshold"]
                    == 64 * np.finfo(np.float64).eps * 100 * max(float(integral["s"][0]), parameters["natural_norm"])
                    and np.all(integral["s"] > parameters["threshold"]), "original rank/tolerance differs")
            diagnostic = helper["checks"]["ls_diagnostics"]["values"]
            require(helper["inputs"]["determinant_lower"] == diagnostic["determinant_lower"] == slot["determinant"]["lower"]
                    and helper["inputs"]["trace_upper"] == diagnostic["trace_upper"] == slot["trace"]["upper"],
                    "same-slot Gram bounds differ")
            for name, value in integral.items():
                same_array(helper["validated_integral"][name], value, "initializer input is not its registered slot")
            same_array(helper["inputs"]["weighted_impulse_point"],
                       causal[f"weighted-{seed}-{index:03d}"]["root__values__weighted_impulse"][:, 0],
                       "initializer impulse point is not its registered saved column")
            same_array(helper["ls"]["rhs"], integral["btilde4"] + helper["inputs"]["weighted_impulse_point"],
                       "single saved LS right-hand side is not b plus the registered impulse")
            history_clock = inputs["history_time"][index] - inputs["history_time"][index, 0]
            same_array(initialization["history_clock"]["clock"], history_clock, "original history subtraction order differs")
            same_array(history_clock, integral["time"], "history clock differs from A14 saved clock")
            same_array(np.diff(history_clock), integral["h"], "history interval subtraction order differs")
            u11 = np.stack([causal[f"impulse-{index:03d}"][f"root__u11__{j}"][:, 0] for j in range(11)])
            delta = inputs["constant_power"][index] - u11[-1]
            same_array(initialization["input_anchor"]["delta_u"], delta, "saved input anchor jump differs")
            require(np.isfinite(delta).all() and np.max(np.abs(delta)) <= 1e-12, "input anchor exceeds frozen tolerance")
            require(initialization["input_anchor"]["absolute_tolerance"] == 1e-12
                    and initialization["input_anchor"]["bitwise_equal"] is bool(np.array_equal(inputs["constant_power"][index], u11[-1])),
                    "input anchor gate record differs")
            for code, step in STEPS:
                history_name = f"hist-{seed}-{index:03d}-{code}"
                if statuses[history_name] != "complete":
                    continue
                history = store.record(history_name)["value"]["helpers"]
                path = _check_saved_path(history["path_evaluation"], history_clock, u11,
                                         inputs["equilibrium"][index], scale, linear=True, step=step, mass=inputs["mass"])
                same_array(path[0, :20], inputs["history"][index, 0], "history visible initial state differs")
                same_array(path[0, 20:], z0, "history hidden initial state differs from unique LS")
                reset = history["visible_reset"]["reset"]
                same_array(reset["before"], path[-1], "reset before-state differs from same-step history endpoint")
                same_array(reset["origin"][:20], inputs["history"][index, -1], "reset visible observations differ")
                same_array(reset["origin"][20:], path[-1, 20:], "reset replaced own hidden endpoint")
                require(all(math.isfinite(float(reset[name])) for name in ("energy_before", "energy_after", "reset_energy_change"))
                        and reset["reset_energy_change"] == reset["energy_after"] - reset["energy_before"],
                        "reset energy/change is invalid")
                origins[(seed, index, code)] = reset["origin"]
                result["history_paths_verified"] += 1
    for identity, status in statuses.items():
        if identity.startswith(("hrefine-", "frefine-", "refine-")) and status == "complete":
            check = store.record(identity)["value"]["helpers"]["refinement"]
            parts = identity.split("-")
            if parts[0] == "refine":
                group, seed, index = parts[1:]
                pair = [comparisons[f"reference-{seed}-{index}-{code}"]["candidate" if group == "I" else "baseline"]
                        for code, _step in STEPS]
            else:
                seed, index = parts[1:]
                prefix = "hist" if parts[0] == "hrefine" else "future"
                pair = [store.record(f"{prefix}-{seed}-{index}-{code}")["value"]["helpers"]["path_evaluation"]["path"]
                        for code, _step in STEPS]
            replay_refinement_record(check, pair[0], pair[1], scale)
            result["refinement_records_verified"] += 1
    for seed in SEEDS:
        for index in range(n):
            clock = clocks[f"clock-{index:03d}"]["clock"]
            truth = comparisons[f"reference-324-{index:03d}-h005"]["truth"]
            require(clock.shape == (counts[index] + 1,) and truth.shape == (counts[index] + 1, 20),
                    "saved common future population differs")
            require(np.isfinite(clock).all() and clock[0] == 0 and np.all(np.diff(clock) > 0)
                    and clock[-1] <= 3 + ATOL, "future clock gate failed")
            if statuses[f"clock-{index:03d}"] == "complete":
                same_array(store.record(f"clock-{index:03d}")["value"]["clock"], clock, "saved checked future clock differs")
            for code, step in STEPS:
                refs = comparisons[f"reference-{seed}-{index:03d}-{code}"]
                same_array(refs["candidate"][0], refs["reset"], "I reference origin differs from saved A15 reset")
                for path in (refs["candidate"], refs["baseline"], truth):
                    same_array(path[0, :20], inputs["history"][index, -1], "saved reference visible origin differs")
                identity = f"identity-{seed}-{index:03d}-{code}"
                if statuses[identity] == "complete":
                    same_record(store.record(identity)["value"], {
                        "common_observed_origin_bitwise": True, "I_reset_origin_bitwise": True,
                        "C_frozen_origin_bitwise": True, "clock_source_descriptor_bound": True}, identity)
                future_name, score_name = f"future-{seed}-{index:03d}-{code}", f"score-{seed}-{index:03d}-{code}"
                candidate = None
                if statuses[future_name] == "complete":
                    helper = store.record(future_name)["value"]["helpers"]["path_evaluation"]
                    candidate = _check_saved_path(helper, clock, inputs["constant_power"][index],
                                                  inputs["equilibrium"][index], scale, linear=False, step=step, mass=inputs["mass"])
                    same_array(candidate[0], origins[(seed, index, code)], "future did not start at own frozen reset")
                    result["future_paths_verified"] += 1
                if statuses[score_name] == "complete":
                    require(candidate is not None, "scoring ran without a candidate path")
                    saved_scores = store.record(score_name)["value"]
                    require(set(saved_scores) == {"C", "I", "E"}, "score group inventory differs")
                    for group, path in (("C", candidate), ("I", refs["candidate"]), ("E", refs["baseline"])):
                        expected = replay_event_scores(path, truth, clock, scale)
                        same_record(saved_scores[group], expected, score_name + "/" + group)
                        event_records[(group, seed, code)][index] = saved_scores[group]
                        result["event_scores_replayed"] += 1
    if statuses["references"] == "complete":
        hashes = store.record("references")["value"]["hashes"]
        lineage = source["ancestry_fingerprints"]
        common = lineage["truth"]["A10/real_train_arrays.npz"]["members"]
        expected_hashes = {}

        def verify_concat(name, chunks, fingerprints):
            # Byte concatenation/hash of saved evidence, without a new numerical operator.
            digest = hashlib.sha256()
            shape = [0, *chunks[0].shape[1:]]
            amount, dtype = 0, chunks[0].dtype.str
            for array in chunks:
                require(array.dtype.str == dtype and list(array.shape[1:]) == shape[1:], "reference concatenation layout differs")
                digest.update(array.tobytes(order="C"))
                shape[0] += len(array)
                amount += array.nbytes
            item = {"shape": shape, "dtype": dtype, "logical_bytes": amount, "sha256_c_order_bytes": digest.hexdigest()}
            same_record(hashes[name], item, "reference ancestry " + name)
            for original in fingerprints:
                require(all(item[key] == original[key] for key in ("shape", "dtype", "sha256_c_order_bytes")),
                        "reference ancestry differs from frozen source proof")
            expected_hashes[name] = item

        for seed in SEEDS:
            for code, _step in STEPS:
                verify_concat(f"E-{seed}-{code}", [comparisons[f"reference-{seed}-{i:03d}-{code}"]["baseline"] for i in range(n)],
                              [lineage["E"][f"{seed}-{code}"]["future"]])
        verify_concat("truth", [comparisons[f"reference-324-{i:03d}-h005"]["truth"] for i in range(n)], [common["future"]])
        verify_concat("clock", [clocks[f"clock-{i:03d}"]["clock"] for i in range(n)], [common["forecast_time"]])
        offsets = np.asarray([0, *np.cumsum(np.asarray(counts, dtype=np.int64) + 1)], dtype=np.int64)
        verify_concat("offsets", [offsets], [common["offsets"], *(row["offsets"] for row in lineage["E"].values())])
        require(set(hashes) == set(expected_hashes), "reference ancestry hash inventory differs")
        result["reference_ancestry_hashes_verified"] = len(expected_hashes)
    if statuses["parameters-final"] == "complete":
        after = store.record("parameters-final")["value"]["seeds"]
        require(set(after) == {str(seed) for seed in SEEDS}, "final parameter identity lacks seeds")
        for seed in SEEDS:
            require(after[str(seed)]["parameters"] == before_parameters[str(seed)], "parameters changed between first and final identity")
    groups, pairs = {}, {}
    for group in ("C", "I", "E"):
        for seed in SEEDS:
            for code, _step in STEPS:
                identity = f"group-{group}-{seed}-{code}"
                if statuses[identity] == "complete":
                    saved = store.record(identity)["value"]
                    expected = replay_population_scores(event_records[(group, seed, code)], weights, counts)
                    same_record(saved, expected, identity)
                    groups[(group, seed, code)] = saved
                    result["population_scores_replayed"] += 1
    for reference in ("I", "E"):
        for seed in SEEDS:
            for code, step in STEPS:
                identity = f"pair-{reference}-{seed}-{code}"
                if statuses[identity] == "complete":
                    saved = store.record(identity)["value"]
                    c, r = groups.get(("C", seed, code)), groups.get((reference, seed, code))
                    expected = (replay_paired_group(c, r) if c is not None and r is not None else
                                {"complete": False, "pass": False, "conditions": None, "differences": None,
                                 "reason": "population_summary_numerical_failure"})
                    same_record(saved, expected, identity)
                    pairs[(reference, seed, step)] = saved
                    result["paired_groups_replayed"] += 1
                else:
                    pairs[(reference, seed, step)] = {"complete": False, "pass": False}
    summaries = store.record("weighted_summaries")
    same_record(summaries["groups"], {f"{g}-{s}-{c}": v for (g, s, c), v in groups.items()}, "weighted groups")
    same_record(summaries["pairs"], {f"{r}-{s}-{h}": v for (r, s, h), v in pairs.items()}, "weighted pairs")
    verdict = replay_campaign_verdict(all(value == "complete" for value in statuses.values()), pairs)
    return dict(result), verdict


def audit_preceding_seal(store, preceding, preseal, captured_records, expected_read_bytes):
    """Verify the saved N+1 tail actions; the final four remain external."""
    order = [(name, "hash_file") for name in sorted(preseal, key=Path)] + [("source_access.jsonl", "count_lines")]
    require(preceding.get("poisoned") is False and preceding.get("first_error") is None
            and preceding.get("unknown_operations") == 0 and preceding.get("actual_write_bytes") == 0
            and preceding.get("action_count") == len(order)
            and preceding.get("transition_records") == 2 * len(order)
            and preceding.get("actual_read_bytes") == expected_read_bytes
            and preceding.get("max_actions") == (270_000 - captured_records) // 2,
            "saved sealing prefix totals or state differ")
    actions = preceding["actions"]
    require(len(actions) == len(order), "saved sealing prefix inventory differs")
    for index, (action, (name, operation)) in enumerate(zip(actions, order, strict=True)):
        amount = store.sizes[name]
        calls = math.ceil(amount / 1_048_576) + 1
        require(action.get("action_id") == index and action.get("name") == name
                and action.get("operation") == operation and action.get("status") == "returned"
                and action.get("stage") == "complete" and action.get("cleanup_errors") == []
                and action.get("durability_unknown") is False and action.get("unknown_operations") == 0
                and action.get("short_writes") == 0 and action.get("write_bytes_returned") == 0
                and action.get("write_operations_attempted") == action.get("write_operations_returned") == 0
                and action.get("read_operations_attempted") == action.get("read_operations_returned") == calls
                and action.get("read_bytes_returned") == action.get("size_bytes_before") == action.get("size_bytes_after") == amount
                and action.get("hash_prefix_sha256") == store.manifest[name], "saved actual sealing action differs")
        if operation == "count_lines":
            require(action.get("ends_with_newline") is True, "sealed source journal has an unfinished transition")


def audit_output_io(store, source_transition_count, *, compute=None):
    summary = store.json("output_io_summary.json")
    captured, sealing = summary["captured_run_io"], summary["sealing"]
    require(captured.get("poisoned") is False and captured.get("journal_closed") is True
            and captured.get("journal_durably_synced") is True and not captured.get("active_open_ids")
            and not captured.get("unknown_operations") and captured.get("first_error") is None
            and captured.get("context_error") is None, "output journal did not close durably and normally")
    pending, totals, record_count, writes, write_lifecycles = {}, Counter(), 0, Counter(), {}
    measured_fields = ("read_calls_returned", "read_bytes_returned", "read_bytes_unknown_operations",
                       "write_calls_returned", "write_bytes_returned", "write_bytes_unknown_operations",
                       "seek_calls_returned", "fsync_calls_returned")
    for row in _stream_jsonl(store, "output_io.jsonl"):
        record_count += 1
        identity, relative = row.get("open_id"), row.get("path")
        require(type(identity) is int and relative in store.manifest and relative != "output_io.jsonl",
                "output I/O names an unsealed or recursive path")
        if row.get("transition") == "attempted":
            require(identity == totals["open_attempted"] and identity not in pending,
                    "output open identifier repeated or skipped")
            pending[identity] = (relative, row["mode"], record_count)
            totals["open_attempted"] += 1
        else:
            opening = pending.pop(identity, None)
            require(row.get("transition") == "returned" and row.get("failed") is False
                    and row.get("opened") is True and opening is not None and opening[:2] == (relative, row["mode"]),
                    "output open failed or has no matching attempted record")
            totals["open_returned"] += 1
            totals["open_closed_returned"] += 1
            for operation in ("read", "write", "seek", "fsync"):
                attempted, returned = row[operation + "_calls_attempted"], row[operation + "_calls_returned"]
                require(type(attempted) is int and type(returned) is int and attempted == returned >= 0,
                        "output operation count is unknown or incomplete")
            for field in ("read_bytes_unknown_operations", "write_bytes_unknown_operations",
                          "write_durability_unknown_operations", "fsync_calls_failed", "short_writes"):
                require(row.get(field) == 0, "output contains an unknown or short failed operation")
            for field in measured_fields:
                require(type(row[field]) is int and row[field] >= 0, "invalid output byte/operation count")
                totals[field] += row[field]
            writes[relative] += row["write_bytes_returned"]
            require(row["write_bytes_returned"] == row["write_bytes_requested"], "successful writes were short")
            if row["write_calls_returned"]:
                require(row["fsync_calls_returned"] > 0 and relative not in write_lifecycles,
                        "output artifact was rewritten or lacked durable synchronization")
                write_lifecycles[relative] = (opening[2], record_count)
    require(not pending and record_count == captured["records"] == captured["transition_records"],
            "output journal closure/count differs")
    for field in ("open_attempted", "open_returned", "open_closed_returned", *measured_fields):
        require(captured[field] == totals[field], "output snapshot aggregate differs: " + field)
    require(captured["open_failed"] == 0 and captured["max_records"] == 270_000, "output failure or cap differs")
    # ZIP seek/rewrites legitimately make returned write bytes exceed final size.
    for relative, amount in writes.items():
        if amount:
            require(amount >= store.sizes[relative], "recorded writes cannot account for final file size")
    preseal = set(store.manifest) - {"output_io_summary.json"}
    require(set(writes) == preseal - {"output_io.jsonl"}, "output journal omits an artifact lifecycle")
    require(record_count <= 270_000, "captured output journal exceeds cap")
    require(set(write_lifecycles) == preseal - {"output_io.jsonl"}, "artifact has no single completed durable write lifecycle")
    if compute is not None:
        for phase, marker_name, next_input in ((1, "origins_freeze.json", "forecast_clocks.json"),
                                               (2, "futures_freeze.json", "comparison_inputs.json")):
            marker_start, marker_end = write_lifecycles[marker_name]
            require(marker_end < write_lifecycles[next_input][0], "next-phase inputs were saved before the candidate freeze closed")
            for identity in compute["scope_order"]:
                scope_phase = compute["scopes"][identity]["phase"]
                for filename in (identity + ".json", identity + ".npz", f"operations/{identity}-reserved.json",
                                 f"operations/{identity}-closed.json"):
                    start, end = write_lifecycles[filename]
                    if scope_phase <= phase:
                        require(end < marker_start, "freeze preceded closure of an earlier scientific scope")
                    else:
                        require(start > marker_end, "later scientific scope preceded its candidate freeze")
    actions = len(preseal) + 5
    preceding_bytes = sum(store.sizes[name] for name in preseal) + store.sizes["source_access.jsonl"]
    audit_preceding_seal(store, sealing["preceding_tail"], preseal, record_count, preceding_bytes)
    output_records = record_count + 2 * actions
    require(summary["source_transition_lines"] == source_transition_count
            and sealing["preceding_hash_files"] == len(preseal)
            and sealing["preceding_hash_read_bytes"] == preceding_bytes
            and sealing["actions_including_last_writes"] == actions
            and sealing["output_transitions_including_seal"] == output_records <= 270_000
            and sealing["combined_transitions_including_seal"] == output_records + source_transition_count <= 380_000,
            "final sealing counts or byte identities differ")
    return {"captured_transition_records": record_count, "output_transitions_including_seal": output_records,
            "combined_transitions_including_seal": output_records + source_transition_count,
            "reconstructed_seal_read_bytes": preceding_bytes + store.sizes["output_io_summary.json"] + store.manifest_size,
            "reconstructed_last_two_write_bytes": store.sizes["output_io_summary.json"] + store.manifest_size,
            "external_common_budget_completion_receipt_read": False,
            "freeze_output_lifecycle_order_verified": compute is not None,
            "boundary": "Manifest does not hash itself. External completed receipt and its final write success are not verified by this run-confined audit."}


def audit_result_run(directory, *, expected_manifest_sha256, expected_read_plan_sha256,
                     expected_compute_plan_sha256):
    """Audit one sealed production A17 result, using only files inside that run.

    Returns PASS/FAIL evidence, never an execution authorization. Failures stop
    this read-only audit without altering any run artifact or scientific result.
    """
    report = {"status": "FAIL", "audit_kind": "A17_independent_saved_evidence_replay_v2",
              "execution_authorized": False, "new_scientific_comparison": False,
              "model_ls_rhs_fit_calls": 0, "old_source_payload_reads": 0,
              "run_directory": str(Path(directory).absolute())}
    store = None
    stage = "run_directory_identity"
    try:
        require(Path(directory).name.startswith("paper_kiki_a17_input_consistency-"), "not a new A17 result directory")
        stage = "manifest_and_sealed_files"
        store = SealedRun(directory, expected_manifest_sha256)
        stage = "saved_controls_and_reservation"
        require(store.manifest.get("read_plan.json") == expected_read_plan_sha256
                and store.manifest.get("compute_plan.json") == expected_compute_plan_sha256,
                "saved controls differ from externally approved hashes")
        plan, compute, receipt = store.json("read_plan.json"), store.json("compute_plan.json"), store.json("run_receipt.json")
        require(receipt.get("authorization") == AUTH and receipt.get("run_id") == store.root.name
                and isinstance(receipt.get("source_commit"), str)
                and re.fullmatch(r"[0-9a-f]{40}", receipt["source_commit"]) is not None
                and receipt.get("spent") is True and receipt.get("retry_allowed") is False
                and receipt.get("read_plan_sha256") == expected_read_plan_sha256
                and receipt.get("compute_plan_sha256") == expected_compute_plan_sha256,
                "run reservation identity differs")
        require(plan.get("authorization") == compute.get("authorization") == AUTH
                and compute["bindings"]["read_plan"]["sha256"] == expected_read_plan_sha256,
                "control authorization or cross-binding differs")
        stage = "original_population"
        pop = plan["population"]
        same_record(store.json("allocated_slots.json"), pop, "original allocation")
        population_sha = hashlib.sha256(json.dumps(pop, sort_keys=True, separators=(",", ":"),
                                                   ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()
        require(population_sha == plan["freeze_contract"]["population_sha256"]
                == compute["population"]["population_sha256"], "population SHA differs")
        require(len(pop["allocated_slots"]) == 268 and len(pop["cache_rows"]) == len(pop["weights"]) == 258
                and len(pop["seed_event_slots"]) == 774 and sum(pop["future_point_counts"]) == 72147
                and len(compute["scope_order"]) == compute["scope_count"] == 10356,
                "production population or scope allocation differs")
        require([row for row in pop["allocated_slots"] if row["input_status"] == "eligible"] == pop["cache_rows"],
                "original eligible allocation was filtered or reordered")
        stage = "scientific_scope_store"
        scopes = audit_scope_store(store, compute)
        stage = "scientific_phase_population"
        require(Counter(compute["scopes"][name]["phase"] for name in compute["scope_order"])
                == {1: 3100, 2: 2580, 3: 4676}, "registered phase scope counts differ")
        stage = "phase_freezes"
        freezes = audit_freezes(store, plan, compute, receipt, scopes)
        stage = "source_reader_schedule"
        reader = audit_reader_journal(_stream_jsonl(store, "source_access.jsonl"), plan, freezes)
        stage = "saved_access_snapshot"
        access = audit_access_snapshot(store, plan, reader)
        stage = "saved_source_provenance"
        provenance = audit_saved_provenance(store, plan)
        stage = "saved_phase_inputs"
        copies = audit_saved_inputs(store, plan)
        stage = "saved_scientific_values"
        values, verdict = audit_saved_scientific_values(store, plan, scopes["statuses"], copies)
        stage = "final_scientific_summary"
        summary = store.json("summary.json")
        same_record({key: summary[key] for key in verdict}, verdict, "final scientific verdict")
        require(summary.get("diagnostic_status") == "COMPLETE" and summary.get("allocated_train") == 268
                and summary.get("eligible_train") == 258 and summary.get("seed_event_slots") == 774
                and summary.get("scope_count") == 10356 and summary.get("source_commit") == receipt["source_commit"]
                and summary.get("authorization") == AUTH
                and summary.get("scope_statuses") == dict(Counter(scopes["statuses"].values())),
                "final diagnostic population/status summary differs")
        stage = "harness_stage_evidence"
        stage_evidence = store.json("stage_evidence.json")
        for field, expected in {"status": "PASS", "process_status": "PASS", "source_commit": receipt["source_commit"],
                                "diagnostic_status": "COMPLETE", "scientific_verdict": verdict["scientific_verdict"],
                                "promotion_decision": "KEEP_BASELINE", "development_authorization_id": AUTH,
                                "certified_envelope_established": False, "deployment_predictions_allowed": False,
                                "model_promoted": False, "level_6_claimed": False, "hidden_training_targets_read": False,
                                "labels_read": 0, "internal_val_ids_read": [], "source_val_ids_read": [], "test_ids_read": [],
                                "new_missed_instability_result_established": False, "result_dir": store.root.name}.items():
            same_record(stage_evidence[field], expected, "stage evidence " + field)
        stage = "output_io_and_preceding_seal"
        io = audit_output_io(store, reader["transition_records"], compute=compute)
        report.update(status="PASS", manifest_sha256=expected_manifest_sha256,
                      population_sha256=population_sha, scientific_verdict=verdict,
                      scope_count=len(scopes["statuses"]), scope_statuses=dict(Counter(scopes["statuses"].values())),
                      scientific_transition_records=scopes["transition_records"],
                      scientific_actual_counters=scopes["actual_counters"],
                      reader={key: value for key, value in reader.items() if key != "selector_keys"},
                      access=access, provenance=provenance, saved_value_checks=values, output_io=io,
                      boundary="Saved-evidence consistency only. No new fit, solve, rollout, labels, FSR or promotion; KEEP_BASELINE.")
    except (OSError, ValueError, TypeError, KeyError, IndexError, ArithmeticError, zipfile.BadZipFile) as error:
        trace, locations = error.__traceback__, []
        while trace is not None:
            code = trace.tb_frame.f_code
            locations.append({"file": code.co_filename, "function": code.co_name,
                              "line": trace.tb_lineno})
            trace = trace.tb_next
        report["failure"] = {"type": type(error).__name__, "message": str(error),
                             "stage": stage, "last_code_location": locations[-1] if locations else None,
                             "code_locations": locations}
    if store is not None:
        report["audit_reads"] = {"run_files_opened": len(store.opens), "file_open_count": sum(store.opens.values()),
                                 "hash_json_and_journal_bytes_measured": sum(store.bytes_read.values()),
                                 "npz_members_decoded": store.decoded_members,
                                 "npz_decode_stream_bytes": "not separately metered; archive hashes and logical member hashes verified"}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--read-plan-sha256", required=True)
    parser.add_argument("--compute-plan-sha256", required=True)
    args = parser.parse_args()
    report = audit_result_run(args.run_directory, expected_manifest_sha256=args.manifest_sha256,
                              expected_read_plan_sha256=args.read_plan_sha256,
                              expected_compute_plan_sha256=args.compute_plan_sha256)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
