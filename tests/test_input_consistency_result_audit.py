"""Synthetic saved-record tests: no model, LS, RHS, source archive or real run."""
from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter

import numpy as np
import pytest

from kron_reduce.evaluation.input_consistency_result_audit import (
    AUTH,
    EvidenceError,
    SealedRun,
    _check_saved_path,
    _qualified,
    _reader_rows,
    audit_access_snapshot,
    audit_accounting,
    audit_freezes,
    audit_input_schedule,
    audit_output_io,
    audit_reader_journal,
    audit_result_run,
    audit_saved_inputs,
    audit_saved_provenance,
    audit_scope_store,
    audit_scope_transitions,
    replay_population_scores,
    replay_refinement_record,
    same_record,
    strict_json,
)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def json_file(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False, sort_keys=True) + "\n", encoding="utf-8")


def seal(directory):
    manifest = {p.relative_to(directory).as_posix(): digest(p.read_bytes())
                for p in directory.rglob("*") if p.is_file() and p.name != "artifact_hashes.json"}
    json_file(directory / "artifact_hashes.json", manifest)
    return digest((directory / "artifact_hashes.json").read_bytes())


def save_record(directory, name, value):
    arrays = {}

    def pack(item, prefix="root"):
        if isinstance(item, np.ndarray):
            arrays[prefix] = item
            return {"array_member": prefix, "shape": list(item.shape), "dtype": item.dtype.str,
                    "sha256_c_order_bytes": digest(item.tobytes())}
        if isinstance(item, dict):
            return {key: pack(child, prefix + "__" + key) for key, child in item.items()}
        if isinstance(item, list):
            return [pack(child, prefix + "__" + str(index)) for index, child in enumerate(item)]
        return item

    json_file(directory / (name + ".json"), pack(value))
    np.savez(directory / (name + ".npz"), **arrays)


def test_sealed_record_roundtrip_and_old_descriptor_is_metadata(tmp_path):
    original = {"actual": np.array([1., -0.]), "old": {
        "array_member": "old_source_member", "shape": [999], "dtype": "<f8",
        "sha256_c_order_bytes": "0" * 64, "path": "../never_open_this.npz"}}
    save_record(tmp_path, "item", original)
    store = SealedRun(tmp_path, seal(tmp_path))
    same_record(store.record("item"), original)
    assert store.decoded_members == 1
    assert set(store.opens) == {"artifact_hashes.json", "item.json", "item.npz"}


@pytest.mark.parametrize("attack", ["wrong_external", "changed_file", "extra_file", "missing_file",
                                    "parent_path", "absolute_path", "backslash", "double_key"])
def test_manifest_fail_closed(tmp_path, attack):
    json_file(tmp_path / "x.json", {"ok": True})
    expected = seal(tmp_path)
    if attack == "wrong_external":
        expected = "0" * 64
    elif attack == "changed_file":
        json_file(tmp_path / "x.json", {"ok": False})
    elif attack == "extra_file":
        (tmp_path / "unsealed").write_bytes(b"new")
    elif attack == "missing_file":
        (tmp_path / "x.json").unlink()
    else:
        names = {"parent_path": "../x.json", "absolute_path": "/x.json", "backslash": "bad\\x.json"}
        if attack == "double_key":
            (tmp_path / "artifact_hashes.json").write_text('{"x.json":"' + "0" * 64 + '","x.json":"' + "0" * 64 + '"}')
        else:
            json_file(tmp_path / "artifact_hashes.json", {names[attack]: "0" * 64})
        expected = digest((tmp_path / "artifact_hashes.json").read_bytes())
    with pytest.raises((EvidenceError, FileNotFoundError)):
        SealedRun(tmp_path, expected)


@pytest.mark.parametrize("attack", ["logical_hash", "shape", "dtype", "extra_member", "missing_member", "object"])
def test_saved_array_inventory_and_identity(tmp_path, attack):
    save_record(tmp_path, "item", {"a": np.ones(2)})
    metadata = json.loads((tmp_path / "item.json").read_text())
    if attack in {"logical_hash", "shape", "dtype"}:
        key, value = {"logical_hash": ("sha256_c_order_bytes", "0" * 64),
                      "shape": ("shape", [3]), "dtype": ("dtype", "<f4")}[attack]
        metadata["a"][key] = value
        json_file(tmp_path / "item.json", metadata)
    elif attack == "extra_member":
        np.savez(tmp_path / "item.npz", root__a=np.ones(2), extra=np.zeros(1))
    elif attack == "missing_member":
        np.savez(tmp_path / "item.npz")
    else:
        np.savez(tmp_path / "item.npz", root__a=np.array([object(), object()], dtype=object))
    store = SealedRun(tmp_path, seal(tmp_path))
    with pytest.raises(ValueError):
        store.record("item")


@pytest.mark.parametrize("raw", ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', '{"a":-Infinity}'])
def test_strict_json_rejects_ambiguous_or_nonfinite(raw):
    with pytest.raises(ValueError):
        strict_json(raw)


def transition(category, index, state, scope="s"):
    return {"scope_id": scope, "category": category, "index": index, "transition": state, "details": None}


def test_nested_actual_return_count_and_numeric_failure_prefix():
    rows = [transition("outer", 0, "attempted"), transition("inner", 0, "attempted"),
            transition("inner", 0, "returned"), transition("outer", 0, "returned")]
    counters, count = audit_scope_transitions(rows, {"outer": 1, "inner": 1}, "complete", "s")
    assert count == 4
    assert counters == {"attempted": {"outer": 1, "inner": 1}, "returned": {"inner": 1, "outer": 1}, "failed": {}}
    rows[2]["transition"] = rows[3]["transition"] = "failed"
    counters, _ = audit_scope_transitions(rows, {"outer": 1, "inner": 1}, "numerical_failure", "s")
    assert counters["failed"] == {"inner": 1, "outer": 1}
    assert audit_scope_transitions([], {"outer": 1}, "numerical_failure", "s")[1] == 0
    assert audit_scope_transitions([], {"outer": 1}, "prerequisite_failed", "s")[1] == 0


@pytest.mark.parametrize("attack", ["retry", "skip", "wrong_scope", "wrong_category", "unclosed",
                                    "terminal_order", "extra_return", "complete_missing", "prerequisite_work"])
def test_scientific_transition_attacks(attack):
    rows = [transition("a", 0, "attempted"), transition("a", 0, "returned")]
    limits, status = {"a": 1, "b": 1}, "numerical_failure"
    if attack == "retry":
        rows[1]["transition"] = "failed"
        rows += [transition("b", 0, "attempted"), transition("b", 0, "returned")]
    elif attack == "skip":
        rows[0]["index"] = 1
    elif attack == "wrong_scope":
        rows[0]["scope_id"] = "other"
    elif attack == "wrong_category":
        rows[0]["category"] = "new"
    elif attack == "unclosed":
        rows.pop()
    elif attack == "terminal_order":
        rows.insert(1, transition("b", 0, "attempted"))
    elif attack == "extra_return":
        rows.append(transition("a", 0, "returned"))
    elif attack == "complete_missing":
        status = "complete"
    else:
        status = "prerequisite_failed"
    with pytest.raises(EvidenceError):
        audit_scope_transitions(rows, limits, status, "s")


def accounting_fixture():
    spec = {"derived_limits": {"successful_rk4_updates": 2},
            "diagnostic_operation_limits": {"path_refinement": {"matmul": 0, "norm_calls": 2}}}
    diagnostic = {"matmul_attempted": 0, "matmul_returned": 0, "norm_calls_attempted": 2, "norm_calls_returned": 2}
    actual = {"successful_rk4_updates": 2}
    record = {"status": "complete", "derived_actual": actual,
              "diagnostic_actual": {"path_refinement": diagnostic}, "value": {"helpers": {
                  "path": {"derived_actual": actual},
                  "refinement": {"category": "path_refinement", "operation_counts": diagnostic}}}}
    return spec, record, {key: record[key] for key in ("derived_actual", "diagnostic_actual")}


def test_derived_accounting_actual_return_reconstruction():
    audit_accounting(*accounting_fixture())


@pytest.mark.parametrize("attack", ["overflow", "missing_success", "fake_actual", "duplicate", "unknown", "failed_prerequisite"])
def test_derived_accounting_rejects(attack):
    spec, record, closed = accounting_fixture()
    if attack == "overflow":
        record["derived_actual"]["successful_rk4_updates"] = 3
    elif attack == "missing_success":
        record["derived_actual"]["successful_rk4_updates"] = 1
    elif attack == "fake_actual":
        record["value"]["helpers"]["path"]["derived_actual"] = {}
    elif attack == "duplicate":
        record["value"]["helpers"]["second"] = copy.deepcopy(record["value"]["helpers"]["refinement"])
    elif attack == "unknown":
        record["derived_actual"]["invented"] = 0
    else:
        record["status"] = "prerequisite_failed"
    with pytest.raises(EvidenceError):
        audit_accounting(spec, record, closed)


def reader_fixture():
    plan = {"metadata_files": [{"path": "old/meta.json", "size_bytes": 17}], "contexts": [
        {"path": "old/a.npz", "size_bytes": 30, "phase": phase, "context_id": f"c{phase}",
         "members": [{"name": f"a{phase}", "shape": [2, 3]}]} for phase in (1, 2, 3)]}
    freezes = {2: {"path": "origins_freeze.json", "sha256": "a" * 64, "size_bytes": 20},
               3: {"path": "futures_freeze.json", "sha256": "b" * 64, "size_bytes": 21}}
    return plan, freezes, list(_reader_rows(plan, freezes))


def test_finite_reader_schedule_exact_and_no_source_paths_followed():
    plan, freezes, rows = reader_fixture()
    for row in rows:
        if row["operation"] == "candidate_freeze":
            row["details"] = {**row["details"], "path": "Z:\\never_follow\\" + row["details"]["path"]}
    report = audit_reader_journal(rows, plan, freezes)
    assert report["returned_operations"] == {"metadata": 1, "opaque_hash": 2, "npz_context": 3,
                                              "descriptor": 3, "array": 3, "candidate_freeze": 2}


@pytest.mark.parametrize("attack", ["phase", "duplicate", "missing", "extra", "unfrozen", "freeze_sha", "array_count"])
def test_reader_schedule_attacks(attack):
    plan, freezes, rows = reader_fixture()
    rows = copy.deepcopy(rows)
    if attack == "phase":
        rows[4]["phase"] = 3
    elif attack == "duplicate":
        rows.insert(1, copy.deepcopy(rows[0]))
    elif attack == "missing":
        rows.pop()
    elif attack == "extra":
        rows.append(copy.deepcopy(rows[-1]))
    elif attack == "unfrozen":
        rows = [row for row in rows if row["operation"] != "candidate_freeze"]
    elif attack == "freeze_sha":
        next(row for row in rows if row["operation"] == "candidate_freeze")["details"]["sha256"] = "c" * 64
    else:
        next(row for row in rows if row["operation"] == "array")["details"]["elements"] += 1
    with pytest.raises(EvidenceError):
        audit_reader_journal(rows, plan, freezes)


def schedule_fixture():
    clock = np.array([0., .01, .02])
    power = np.arange(30., dtype=float).reshape(3, 10) / 10
    indices, values = [], []
    for j in range(2):
        for k in range(2):
            for code, number in ((1, k), (2, k + .5), (3, k + .5), (4, k + 1)):
                indices.append([j, k, 2, number, code])
                values.append(power[j].copy() if number == 0 else power[j + 1].copy() if number == 2
                              else power[j] + number / 2 * (power[j + 1] - power[j]))
    return {"index_fields": ["interval_index", "substep_index", "substep_count", "numerator", "stage_code"],
            "stage_codes": {"k1": 1, "k2": 2, "k3": 3, "k4": 4}, "indices": np.array(indices, dtype=float),
            "returned_power": np.array(values), "used_power": np.array(values)}, clock, power


def test_piecewise_linear_stage_witnesses_and_constant_empty():
    schedule, clock, power = schedule_fixture()
    assert audit_input_schedule(schedule, clock, power, linear=True, step=.005) == 16
    for key in ("indices", "returned_power", "used_power"):
        schedule[key] = schedule[key][:0]
    assert audit_input_schedule(schedule, clock, power[0], linear=False, step=.005) == 0


@pytest.mark.parametrize("attack", ["order", "midpoint", "used", "endpoint", "missing", "extra", "constant"])
def test_input_witness_rejects(attack):
    schedule, clock, power = schedule_fixture()
    if attack == "order":
        schedule["indices"][[1, 2]] = schedule["indices"][[2, 1]]
    elif attack == "midpoint":
        schedule["indices"][1, 3] += .5
    elif attack == "used":
        schedule["used_power"][2, 0] += 1
    elif attack == "endpoint":
        schedule["used_power"][0, 0] = schedule["returned_power"][0, 0] = np.nextafter(power[0, 0], 1.)
    elif attack in {"missing", "extra"}:
        for key in ("indices", "returned_power", "used_power"):
            schedule[key] = schedule[key][:-1] if attack == "missing" else np.concatenate([schedule[key], schedule[key][-1:]])
    with pytest.raises(EvidenceError):
        audit_input_schedule(schedule, clock, power, linear=attack != "constant", step=.005)


def scope_fixture(directory):
    specifications, closures, chunks = {}, [], []
    statuses = ("numerical_failure", "prerequisite_failed", "complete")
    for ordinal, status in enumerate(statuses):
        name = f"s{ordinal}"
        failed = [] if ordinal == 0 else ["s0"]
        spec = {"phase": ordinal + 1, "operators": {"op": 1}, "derived_limits": {}, "diagnostic_operation_limits": {},
                "prerequisite_scope_ids": failed, "dependency_mode": "records_present_including_failures" if ordinal == 2 else "all_success"}
        specifications[name] = spec
        rows = [] if ordinal == 1 else [transition("op", 0, "attempted", name), transition("op", 0, "failed" if ordinal == 0 else "returned", name)]
        raw = b"".join((json.dumps(row) + "\n").encode() for row in rows)
        start = sum(map(len, chunks))
        chunks.append(raw)
        counters = {"attempted": {} if ordinal == 1 else {"op": 1},
                    "returned": {"op": 1} if ordinal == 2 else {}, "failed": {"op": 1} if ordinal == 0 else {}}
        value = ({"error_type": "FloatingPointError", "message": "synthetic", "retry_allowed": False} if ordinal == 0 else
                 {"failed_dependencies": failed, "retry_allowed": False} if ordinal == 1 else {})
        body = {"scope_id": name, "status": status, "value": value, "derived_actual": {}, "diagnostic_actual": {},
                "dependencies": failed, "failed_dependencies": failed, "retry_allowed": False}
        save_record(directory, name, body)
        hashes = {name + suffix: digest((directory / (name + suffix)).read_bytes()) for suffix in (".json", ".npz")}
        reserved = {"scope_id": name, "reservation": {"op": 1}, "reserved_operation_limits": {"op": 1},
                    "start_byte_offset": start, "spent": True, "retry_allowed": False, "interrupted_status": "unknown_spent"}
        closed = {"scope_id": name, "status": status, "start_byte_offset": start, "end_byte_offset": start + len(raw),
                  "range_sha256": digest(raw), "actual_counters": counters, "durably_closed": True,
                  "artifact_hashes": hashes, "failed_dependencies": failed, "derived_actual": {}, "diagnostic_actual": {}}
        json_file(directory / "operations" / (name + "-reserved.json"), reserved)
        json_file(directory / "operations" / (name + "-closed.json"), closed)
        closures.append(closed)
    (directory / "operations" / "calls.jsonl").write_bytes(b"".join(chunks))
    json_file(directory / "scientific_scopes.json", closures)
    return {"scope_order": list(specifications), "scopes": specifications}


def test_actual_scope_ranges_failed_dependency_and_aggregate_continue(tmp_path):
    compute = scope_fixture(tmp_path)
    report = audit_scope_store(SealedRun(tmp_path, seal(tmp_path)), compute)
    assert report["statuses"] == {"s0": "numerical_failure", "s1": "prerequisite_failed", "s2": "complete"}
    assert report["transition_records"] == 4
    assert [v["scope_count"] for v in report["phase_prefixes"].values()] == [1, 2, 3]


@pytest.mark.parametrize("attack", ["range_gap", "range_sha", "counter", "unclosed", "missing_dependency", "failed_retried", "index", "body_hash"])
def test_scope_store_rejects_corruption(tmp_path, attack):
    compute = scope_fixture(tmp_path)
    path = tmp_path / "operations" / "s1-closed.json"
    row = json.loads(path.read_text())
    if attack == "range_gap":
        row["start_byte_offset"] += 1
    elif attack == "range_sha":
        row["range_sha256"] = "0" * 64
    elif attack == "counter":
        row["actual_counters"]["returned"]["op"] = 1
    elif attack == "unclosed":
        row["durably_closed"] = False
    elif attack == "missing_dependency":
        compute["scopes"]["s1"]["prerequisite_scope_ids"] = ["absent"]
    elif attack == "failed_retried":
        compute["scopes"]["s1"]["dependency_mode"] = "records_present_including_failures"
    elif attack == "index":
        json_file(tmp_path / "scientific_scopes.json", [])
    else:
        json_file(tmp_path / "s1.json", {})
    json_file(path, row)
    with pytest.raises((EvidenceError, KeyError)):
        audit_scope_store(SealedRun(tmp_path, seal(tmp_path)), compute)


def io_fixture(directory):
    (directory / "source_access.jsonl").write_bytes(b"{}\n{}\n")
    totals, rows = Counter(), []
    fields = ("read_calls_returned", "read_bytes_returned", "read_bytes_unknown_operations", "write_calls_returned",
              "write_bytes_returned", "write_bytes_unknown_operations", "seek_calls_returned", "fsync_calls_returned")
    terminal = {"open_id": 0, "path": "source_access.jsonl", "mode": "wb", "transition": "returned", "opened": True,
                "failed": False, "read_calls_attempted": 0, "read_calls_returned": 0, "read_bytes_returned": 0,
                "read_bytes_unknown_operations": 0, "write_calls_attempted": 1, "write_calls_returned": 1,
                "write_bytes_returned": 6, "write_bytes_requested": 6, "write_bytes_unknown_operations": 0,
                "seek_calls_attempted": 0, "seek_calls_returned": 0, "fsync_calls_attempted": 1,
                "fsync_calls_returned": 1, "fsync_calls_failed": 0, "write_durability_unknown_operations": 0, "short_writes": 0}
    rows += [{"open_id": 0, "path": "source_access.jsonl", "mode": "wb", "transition": "attempted"}, terminal]
    for field in fields:
        totals[field] += terminal[field]
    raw = b"".join((json.dumps(row) + "\n").encode() for row in rows)
    (directory / "output_io.jsonl").write_bytes(raw)
    captured = {**dict(totals), "poisoned": False, "journal_closed": True, "journal_durably_synced": True,
                "active_open_ids": [], "unknown_operations": {}, "first_error": None, "context_error": None,
                "records": 2, "transition_records": 2, "open_attempted": 1, "open_returned": 1,
                "open_closed_returned": 1, "open_failed": 0, "max_records": 270000}
    actions = []
    for index, (name, operation) in enumerate((("output_io.jsonl", "hash_file"), ("source_access.jsonl", "hash_file"),
                                              ("source_access.jsonl", "count_lines"))):
        payload = (directory / name).read_bytes()
        action = {"action_id": index, "name": name, "operation": operation, "status": "returned", "stage": "complete",
                  "read_bytes_returned": len(payload), "write_bytes_returned": 0, "read_operations_attempted": 2,
                  "read_operations_returned": 2, "write_operations_attempted": 0, "write_operations_returned": 0,
                  "unknown_operations": 0, "short_writes": 0, "hash_prefix_sha256": digest(payload),
                  "durability_unknown": False, "cleanup_errors": [], "size_bytes_before": len(payload),
                  "size_bytes_after": len(payload)}
        if operation == "count_lines":
            action["ends_with_newline"] = True
        actions.append(action)
    summary = {"captured_run_io": captured, "source_transition_lines": 2, "sealing": {
        "preceding_hash_files": 2, "preceding_hash_read_bytes": len(raw) + 12,
        "actions_including_last_writes": 7, "output_transitions_including_seal": 16,
        "combined_transitions_including_seal": 18, "preceding_tail": {
            "poisoned": False, "first_error": None, "unknown_operations": 0, "actual_write_bytes": 0,
            "actual_read_bytes": len(raw) + 12, "action_count": 3, "transition_records": 6,
            "max_actions": 134999, "actions": actions}}}
    json_file(directory / "output_io_summary.json", summary)
    return summary


def test_output_io_actual_bytes_durable_close_and_final_tail(tmp_path):
    io_fixture(tmp_path)
    result = audit_output_io(SealedRun(tmp_path, seal(tmp_path)), 2)
    assert result["output_transitions_including_seal"] == 16
    assert result["external_common_budget_completion_receipt_read"] is False


@pytest.mark.parametrize("attack", ["no_fsync", "active", "unknown", "fake_bytes", "tail_missing_manifest_hash", "source_count", "unlogged_file"])
def test_output_io_fail_closed(tmp_path, attack):
    summary = io_fixture(tmp_path)
    if attack == "no_fsync":
        summary["captured_run_io"]["journal_durably_synced"] = False
    elif attack == "active":
        summary["captured_run_io"]["active_open_ids"] = [1]
    elif attack == "unknown":
        summary["captured_run_io"]["unknown_operations"] = {"write": 1}
    elif attack == "fake_bytes":
        summary["captured_run_io"]["write_bytes_returned"] += 1
    elif attack == "tail_missing_manifest_hash":
        summary["sealing"]["actions_including_last_writes"] -= 1
    elif attack == "source_count":
        summary["source_transition_lines"] += 1
    else:
        (tmp_path / "unlogged").write_bytes(b"x")
    json_file(tmp_path / "output_io_summary.json", summary)
    with pytest.raises(EvidenceError):
        audit_output_io(SealedRun(tmp_path, seal(tmp_path)), 2)


@pytest.mark.parametrize("attack", ["poisoned", "missing_action", "reordered", "short_read", "different_hash", "unknown_read", "write_added", "no_newline"])
def test_preceding_seal_action_prefix_rejects(tmp_path, attack):
    summary = io_fixture(tmp_path)
    prefix = summary["sealing"]["preceding_tail"]
    if attack == "poisoned":
        prefix["poisoned"] = True
    elif attack == "missing_action":
        prefix["actions"].pop()
    elif attack == "reordered":
        prefix["actions"][:2] = prefix["actions"][1::-1]
    elif attack == "short_read":
        prefix["actions"][0]["read_bytes_returned"] -= 1
    elif attack == "different_hash":
        prefix["actions"][0]["hash_prefix_sha256"] = "0" * 64
    elif attack == "unknown_read":
        prefix["actions"][0]["unknown_operations"] = 1
    elif attack == "write_added":
        prefix["actions"][0]["write_operations_attempted"] = 1
    else:
        prefix["actions"][-1]["ends_with_newline"] = False
    json_file(tmp_path / "output_io_summary.json", summary)
    with pytest.raises(EvidenceError):
        audit_output_io(SealedRun(tmp_path, seal(tmp_path)), 2)


def test_public_audit_refuses_non_a17_directory_without_open(tmp_path):
    result = audit_result_run(tmp_path, expected_manifest_sha256="0" * 64,
                              expected_read_plan_sha256="0" * 64, expected_compute_plan_sha256="0" * 64)
    assert result["status"] == "FAIL"
    assert result["old_source_payload_reads"] == result["model_ls_rhs_fit_calls"] == 0
    assert "audit_reads" not in result
    assert result["audit_kind"] == "A17_independent_saved_evidence_replay_v2"
    assert result["failure"]["stage"] == "run_directory_identity"
    assert result["failure"]["last_code_location"]["function"] == "require"
    assert all(type(location["line"]) is int and location["line"] > 0
               for location in result["failure"]["code_locations"])


def test_public_audit_preserves_typeerror_stage_and_code_location(tmp_path, monkeypatch):
    from kron_reduce.evaluation import input_consistency_result_audit as audit

    def malformed_manifest(*_args):
        raise TypeError("synthetic malformed manifest")

    monkeypatch.setattr(audit, "SealedRun", malformed_manifest)
    result = audit.audit_result_run(tmp_path / "paper_kiki_a17_input_consistency-synthetic",
                                    expected_manifest_sha256="0" * 64,
                                    expected_read_plan_sha256="0" * 64,
                                    expected_compute_plan_sha256="0" * 64)
    assert result["status"] == "FAIL"
    assert result["failure"]["stage"] == "manifest_and_sealed_files"
    assert result["failure"]["type"] == "TypeError"
    assert result["failure"]["message"] == "synthetic malformed manifest"
    assert result["failure"]["last_code_location"] == {
        "file": malformed_manifest.__code__.co_filename,
        "function": "malformed_manifest",
        "line": malformed_manifest.__code__.co_firstlineno + 1,
    }
    assert "audit_reads" not in result


class SavedMemory:
    """Only test-owned, already returned records; no filesystem behavior."""
    def __init__(self, records):
        self.records = records
        self.manifest = {}
        self.sizes = {}

    def record(self, name, *, decode=True):
        return self.records[name]

    def json(self, name):
        return self.records[name]


def access_fixture():
    member = {"name": "a", "shape": [2], "dtype": "<f8", "sha256_c_order_bytes": digest(np.ones(2).tobytes())}
    plan = {"metadata_files": [{"path": "old/meta", "size_bytes": 17, "sha256": "d" * 64}],
            "contexts": [{"path": "old/a.npz", "context_id": "one", "phase": 1, "members": [member]}],
            "limits": {"arrays": 1, "descriptors": 1, "array_elements": 2, "array_logical_hash_bytes": 16,
                       "npy_header_per_member_cap_including_magic_and_length": 10000, "npy_header_bytes_max": 10000}}
    selectors = {"metadata::old/meta", "context::one", "old/a.npz::a::descriptor", "old/a.npz::a::array"}
    snapshot = {"closed": True, "failed": False, "phase": 3, "array_reads": 1, "descriptors": 1,
                "array_elements": 2, "decoded_array_elements": 2, "decoded_array_bytes": 16, "payload_bytes_read": 16,
                "header_bytes": 128, "opaque_passes": ["after", "before"],
                "selectors": dict.fromkeys(selectors, "returned"), "partial_record": {
                    "metadata_byte_returns": {"old/meta": {"size_bytes": 17, "sha256": "d" * 64, "validated": True}},
                    "contexts": {"one": {"headers": {"a": {"bytes_read": 128, "read_failure_unknown": False,
                         "status": "validated", "parsed": {"shape": [2], "dtype": "<f8", "header_bytes": 128, "fortran_order": False}}},
                         "payloads": {"a": {"status": "validated", "decoded": True, "validated": True, "finite": True,
                            "actual_bytes": 16, "expected_bytes": 16, "decoded_elements": 2, "decoded_bytes": 16,
                            "actual_sha256_c_order_bytes": member["sha256_c_order_bytes"]}}}}}}
    return SavedMemory({"access_ledger": snapshot}), plan, {"selector_keys": selectors, "transition_records": 8}


def test_actual_read_snapshot_header_payload_bytes_and_identity():
    assert audit_access_snapshot(*access_fixture()) == {
        "returned_array_elements": 2, "actual_header_bytes": 128, "actual_payload_bytes": 16}


@pytest.mark.parametrize("attack", ["not_closed", "missing_selector", "decoded_count", "metadata_hash", "header_overflow",
                                    "header_unknown", "header_dtype", "short_payload", "payload_hash"])
def test_actual_read_snapshot_rejects(attack):
    store, plan, journal = access_fixture()
    access = store.records["access_ledger"]
    context = access["partial_record"]["contexts"]["one"]
    if attack == "not_closed":
        access["closed"] = False
    elif attack == "missing_selector":
        access["selectors"].pop("context::one")
    elif attack == "decoded_count":
        access["decoded_array_elements"] += 1
    elif attack == "metadata_hash":
        access["partial_record"]["metadata_byte_returns"]["old/meta"]["sha256"] = "0" * 64
    elif attack == "header_overflow":
        context["headers"]["a"]["bytes_read"] = 10001
    elif attack == "header_unknown":
        context["headers"]["a"]["read_failure_unknown"] = True
    elif attack == "header_dtype":
        context["headers"]["a"]["parsed"]["dtype"] = "<f4"
    elif attack == "short_payload":
        context["payloads"]["a"]["actual_bytes"] = 8
    else:
        context["payloads"]["a"]["actual_sha256_c_order_bytes"] = "0" * 64
    with pytest.raises(EvidenceError):
        audit_access_snapshot(store, plan, journal)


def freeze_fixture():
    slot = "324-000-h005"
    plan = {"freeze_contract": {"slot_ids": [slot], "population_sha256": "a" * 64, "max_bytes": 5000}}
    receipt = {"run_id": "new_synthetic", "source_commit": "b" * 40}
    compute = {"scopes": {"hist-" + slot: {}, "future-" + slot: {}}}
    scopes = {"statuses": {"hist-" + slot: "numerical_failure", "future-" + slot: "prerequisite_failed"},
              "phase_prefixes": {1: {"scope_count": 1, "end_byte_offset": 0, "closed_scope_chain_sha256": "c" * 64},
                                 2: {"scope_count": 2, "end_byte_offset": 0, "closed_scope_chain_sha256": "d" * 64}}}
    store = SavedMemory({})
    for phase, name, prefix, kind in ((2, "origins_freeze.json", "hist", "a17_origins_freeze"),
                                      (3, "futures_freeze.json", "future", "a17_futures_freeze")):
        identity = prefix + "-" + slot
        for suffix in (".json", ".npz"):
            store.manifest[identity + suffix] = "e" * 64
        store.manifest[name], store.sizes[name] = "f" * 64, 900
        store.records[name] = {"authorization": AUTH, **receipt, "kind": kind, "population_sha256": "a" * 64,
                               "scientific_prefix": copy.deepcopy(scopes["phase_prefixes"][phase - 1]),
                               "records": [{"slot_id": slot, "status": scopes["statuses"][identity],
                                            "record_path": identity + ".json", "record_sha256": "e" * 64,
                                            "record_npz_sha256": "e" * 64}]}
    return store, plan, compute, receipt, scopes


def test_freezes_bind_failed_population_both_artifacts_and_scientific_prefix():
    result = audit_freezes(*freeze_fixture())
    assert set(result) == {2, 3}
    assert result[2]["path"] == "origins_freeze.json"


@pytest.mark.parametrize("attack", ["prefix", "status", "npz", "population", "slot_missing", "slot_repeat", "oversize"])
def test_freeze_rejects(attack):
    store, plan, compute, receipt, scopes = freeze_fixture()
    marker = store.records["origins_freeze.json"]
    if attack == "prefix":
        marker["scientific_prefix"]["scope_count"] += 1
    elif attack == "status":
        marker["records"][0]["status"] = "complete"
    elif attack == "npz":
        marker["records"][0]["record_npz_sha256"] = "0" * 64
    elif attack == "population":
        marker["population_sha256"] = "0" * 64
    elif attack == "slot_missing":
        marker["records"] = []
    elif attack == "slot_repeat":
        marker["records"] *= 2
    else:
        store.sizes["origins_freeze.json"] = 5001
    with pytest.raises(EvidenceError):
        audit_freezes(store, plan, compute, receipt, scopes)


def test_saved_phase_copy_fingerprint_only_no_oldsource_open():
    array = np.array([1., 2.])
    plan = {"contexts": [{"context_id": "c", "path": "Z:/closed/source.npz", "phase": 1,
                           "members": [{"name": "a", "shape": [2], "dtype": "<f8",
                                        "sha256_c_order_bytes": digest(array.tobytes())}]}]}
    store = SavedMemory({"causal_inputs": {"c": {"a": array}}, "forecast_clocks": {}, "comparison_inputs": {}})
    assert audit_saved_inputs(store, plan)[1]["c"]["a"] is array
    array[0] = 3.
    with pytest.raises(EvidenceError):
        audit_saved_inputs(store, plan)


def test_replay_preserves_numpy_scalar_accumulation_on_python312():
    records = [{**dict.fromkeys(("primary", "q", "v", "horizon1", "horizon3"), value), "future_points": 300}
               for value in (3e16, 3., 3.)]
    report = replay_population_scores(records, [1 / 3] * 3, [300] * 3)
    assert report["metrics"]["primary"] == 1e16
    assert report["duration_groups"]["full_3s"]["partial_primary_contribution_same_denominator"] == 1e16
    assert report["complete_3s_horizon3_conditional_mse"] == 1e16


def ls_diagnostic_fixture():
    tau = 64 * np.finfo(float).eps * 100
    lower = float(np.nextafter(1., -np.inf))
    return {"qualified": True, "reasons": [], "nonfinite_values": {},
            "gates": {"normal_residual": True, "conditioning_proxy": True}, "values": {
                "tau_num": tau, "lambda_lower": lower, "determinant_lower": 2., "trace_upper": 2.,
                "gram_ratio_before_outward_rounding": 1., "B_frobenius": 1., "rhs_norm": 1., "z0_norm": 1.,
                "normal_scale": 2., "normal_residual_bound": 2 * tau, "normal_residual_norm": 0.,
                "conditioning_proxy": 2 * tau / lower, "conditioning_proxy_bound": 2e-6}}


def test_positive_saved_scalar_diagnostic_formulas():
    _qualified(ls_diagnostic_fixture(), "ls_diagnostics")


@pytest.mark.parametrize("field,value", [("normal_residual_norm", -1.), ("conditioning_proxy", -1.),
                                       ("lambda_lower", 100.), ("normal_scale", 3.),
                                       ("gram_ratio_before_outward_rounding", 2.), ("conditioning_proxy", 0.),
                                       ("rhs_norm", -1.)])
def test_diagnostic_cannot_pass_with_negative_or_inconsistent_saved_values(field, value):
    record = ls_diagnostic_fixture()
    record["values"][field] = value
    with pytest.raises(EvidenceError):
        _qualified(record, "ls_diagnostics")


def refinement_fixture():
    coarse, fine = np.zeros((2, 22)), np.zeros((2, 22))
    coarse[0, 20:] = [3 / 2**20, 4 / 2**20]
    record = {"qualified": True, "reasons": [], "nonfinite_values": {}, "point_count": 2,
              "gates": {"visible_refinement": True, "hidden_refinement": True}, "arrays": {
                  "visible_normalized_difference": np.zeros((2, 20)), "hidden_difference_norms": np.array([5 / 2**20, 0.]),
                  "fine_hidden_norms": np.zeros(2)}, "values": {
                      "visible_max_normalized": 0., "hidden_max_norm": 5 / 2**20, "fine_hidden_max_norm": 0.,
                      "hidden_bound": 1e-5, "visible_bound": 1e-5}}
    return record, coarse, fine, np.ones(20)


def test_refinement_connects_actual_saved_path_pair_without_check_helper():
    replay_refinement_record(*refinement_fixture())


@pytest.mark.parametrize("attack", ["wrong_pair", "visible_array", "hidden_array", "maxima", "negative_norm", "point_count", "scale"])
def test_refinement_saved_record_swaps_or_inconsistent_returns_rejected(attack):
    record, coarse, fine, scale = refinement_fixture()
    if attack == "wrong_pair":
        coarse[0, 0] = 1.
    elif attack == "visible_array":
        record["arrays"]["visible_normalized_difference"][0, 0] = 1e-10
    elif attack == "hidden_array":
        record["arrays"]["hidden_difference_norms"][0] = 0.
    elif attack == "maxima":
        record["values"]["hidden_max_norm"] = 0.
    elif attack == "negative_norm":
        record["values"]["hidden_max_norm"] = -1.
    elif attack == "point_count":
        record["point_count"] = 3
    else:
        scale[0] = 0.
    with pytest.raises(EvidenceError):
        replay_refinement_record(record, coarse, fine, scale)


def provenance_fixture():
    runs = {key: "synthetic_" + key for key in ("A09", "A10", "A14", "A15", "A16")}
    roots = {key: "results/" + value + "/" for key, value in runs.items()}
    documents, contexts, slots, ancestors, allocations = {}, [], {}, {}, []
    fp = {"shape": [1], "dtype": "<f8", "sha256_c_order_bytes": "1" * 64}
    population = {"allocated_slots": [{"event_id": "synthetic/event"}], "cache_rows": [{"event_id": "synthetic/event", "cache_index": 0}],
                  "weights": [1.], "seed_event_slots": [{"seed": seed, "cache_index": 0, "event_id": "synthetic/event"} for seed in (324, 325, 326)]}

    def desc(name):
        return {"array_member": name, **copy.deepcopy(fp)}

    def context(identity, source, name, document, members):
        path = roots[source] + name + ".npz"
        documents[roots[source] + name + ".json"] = document
        row = {"context_id": identity, "source": source, "path": path, "sha256": "2" * 64,
               "members": [{"name": member, **copy.deepcopy(fp)} for member in members]}
        contexts.append(row)
        return row

    impulse = {"u11": [desc("root__u11__" + str(i)) for i in range(11)], "d": [desc("d" + str(i)) for i in range(10)]}
    context("impulse-000", "A16", "impulse-000", impulse, ["root__u11__" + str(i) for i in range(11)])
    input_doc = {name: desc(name) for name in ("history", "history_time", "constant_power", "equilibrium", "mass", "target_scale")}
    context("inputs", "A14", "permitted_inputs", input_doc, list(input_doc))
    a14_members = {"A10/real_train_arrays.npz": copy.deepcopy(input_doc)}
    static_lineage = {}
    inherited = []
    for seed in (324, 325, 326):
        key = f"{seed}-000"
        parameters = {name: desc("parameters__" + name) for name in ("mass", "edge_stiffness", "damping", "coupling", "latent_skew", "latent_dissipation")}
        # Actual A14 dataclass schema: six array descriptors plus scalar
        # backbone metadata. The compact proof contains only the six arrays.
        context(f"static-{seed}", "A14", f"static-{seed}",
                {"parameters": {**parameters, "backbone": "sine"}},
                ["parameters__" + name for name in parameters])
        source_key = f"A06/checkpoint-240-{seed}/model.npz"
        a14_members[source_key] = {name: desc(name) for name in parameters}
        static_lineage[str(seed)] = {"source_key": source_key, "descriptors": copy.deepcopy(parameters),
                                     "source_member_descriptors": copy.deepcopy(a14_members[source_key])}
        integral = {"Ltilde": desc("Ltilde"), "rank": 2, "svd": {"rank": 2, "threshold": 1e-12},
                    "tol": 1e-12, "natural_norm": 1., "status": "complete"}
        integral_context = context("integral-" + key, "A14", "integral-" + key, integral, ["Ltilde"])
        inherited.append(copy.deepcopy(integral_context))
        values = {"weighted_impulse": desc("root__values__weighted_impulse"), "gram": [desc("g0"), desc("g1")],
                  "trace": {"lower": 1., "point": 2., "upper": 3.}, "determinant": {"lower": 1., "point": 2., "upper": 3.}}
        weighted = {"complete": True, "status": "resolved_nonzero", "values": values}
        context("weighted-" + key, "A16", "slot-" + key, weighted, ["root__values__weighted_impulse"])
        records = {}
        for code in ("h005", "h0025"):
            ref = {name: desc(name) for name in ("candidate", "baseline", "reset")}
            context(f"reference-{seed}-000-{code}", "A15", f"future-{seed}-000-{code}", ref, list(ref))
            records[code] = copy.deepcopy(ref)
            ancestor = "A09" if code == "h005" else "A10"
            ancestors[f"{ancestor}/seed-{seed}-{code}/paths.npz"] = {"future": copy.deepcopy(fp), "offsets": copy.deepcopy(fp)}
        slots[key] = {"seed": seed, "cache_index": 0, "event_id": "synthetic/event", "complete": True, "status": "resolved_nonzero",
                      "integral_parameters": {"rank": 2, "threshold": 1e-12, "tol": 1e-12, "natural_norm": 1.},
                      "integral_descriptors": {"Ltilde": desc("Ltilde")}, "weighted_impulse": values["weighted_impulse"],
                      "gram": values["gram"], "trace": values["trace"], "determinant": values["determinant"],
                      "impulse": copy.deepcopy(impulse), "source_paths": {"integral": roots["A14"] + "integral-" + key + ".json",
                          "slot": roots["A16"] + "slot-" + key + ".json", "impulse": roots["A16"] + "impulse-000.json"}, "future_records": records}
        allocations.append({key: slots[f"{seed}-000"][key] for key in ("seed", "cache_index", "event_id", "complete", "status")})
    ancestors["A10/real_train_arrays.npz"] = {name: copy.deepcopy(fp) for name in ("future", "forecast_time", "offsets", "omega_s")}
    documents[roots["A15"] + "source_member_fingerprints.json"] = ancestors
    documents[roots["A14"] + "source_members.json"] = a14_members
    documents[roots["A14"] + "allocated_population.json"] = copy.deepcopy(population)
    documents[roots["A15"] + "allocated_population.json"] = copy.deepcopy(population)
    documents[roots["A16"] + "allocated_slots.json"] = {"original_population": copy.deepcopy(population), "diagnostic_slots": allocations}

    def hashed(value):
        return digest(json.dumps(value, sort_keys=True).encode())

    def manifest(source):
        prefix = roots[source]
        result = {path[len(prefix):]: hashed(value) for path, value in documents.items() if path.startswith(prefix)}
        result.update({row["path"][len(prefix):]: row["sha256"] for row in contexts if row["source"] == source})
        for path in ancestors:
            if path.startswith(source + "/"):
                result[path[len(source) + 1:]] = "3" * 64
        documents[prefix + "artifact_hashes.json"] = result
        return hashed(result)

    manifests = {source: manifest(source) for source in ("A09", "A10", "A14", "A15")}
    control = {"events": copy.deepcopy(population["cache_rows"]), "source_run_id": runs["A14"],
               "source_manifest_sha256": manifests["A14"], "npz_files": inherited}
    documents["configs/control.json"] = control
    documents[roots["A16"] + "frozen_plan.json"] = {"read_plan": copy.deepcopy(control)}
    manifests["A16"] = manifest("A16")
    lineage = {"E": {}, "A14_static": static_lineage,
               "A14_inputs": {name: {"a14_saved_member": copy.deepcopy(value), "ancestor_member": copy.deepcopy(value)} for name, value in input_doc.items()},
               "truth": {"A10/real_train_arrays.npz": {"file_sha256": "3" * 64,
                                                                 "members": copy.deepcopy(ancestors["A10/real_train_arrays.npz"])}}}
    for seed in (324, 325, 326):
        for code in ("h005", "h0025"):
            key = ("A09" if code == "h005" else "A10") + f"/seed-{seed}-{code}/paths.npz"
            lineage["E"][f"{seed}-{code}"] = {"source": key, "file_sha256": "3" * 64, **copy.deepcopy(ancestors[key])}
    proof = {"source_runs": runs, "source_manifest_sha256": manifests, "slot_bindings": slots, "ancestry_fingerprints": lineage}
    plan = {"source_runs": runs, "source_manifest_sha256": manifests, "population": population, "contexts": contexts,
            "ancestry_control": "configs/control.json", "metadata_files": [{"path": path, "sha256": hashed(value)} for path, value in documents.items()]}
    store = SavedMemory({"sources": {"value": copy.deepcopy(proof)}, "access_ledger": {"partial_record": {"metadata_documents": documents}}})
    return store, plan


def test_compact_source_proof_connects_saved_metadata_slots_and_manifest_chain():
    store, plan = provenance_fixture()
    parameters = store.records["access_ledger"]["partial_record"]["metadata_documents"][
        "results/synthetic_A14/static-324.json"]["parameters"]
    assert parameters["backbone"] == "sine"
    assert len(parameters) == 7
    assert len(store.records["sources"]["value"]["ancestry_fingerprints"]["A14_static"]["324"]["descriptors"]) == 6
    result = audit_saved_provenance(store, plan)
    assert result["same_slot_bindings"] == 3


@pytest.mark.parametrize("attack", [
    "parameter_string", "parameter_missing", "parameter_member", "parameter_fingerprint_missing",
    "backbone_missing", "backbone_wrong", "backbone_descriptor", "unexpected_parameter",
    "context_missing", "context_duplicate", "context_prefix", "context_extra",
    "ancestor_hash", "ancestor_shape", "ancestor_dtype", "ancestor_string",
    "proof_missing", "proof_extra", "proof_swapped",
])
def test_production_static_schema_keeps_all_six_parameter_identity_checks(attack):
    store, plan = provenance_fixture()
    documents = store.records["access_ledger"]["partial_record"]["metadata_documents"]
    parameters = documents["results/synthetic_A14/static-324.json"]["parameters"]
    context = next(row for row in plan["contexts"] if row["context_id"] == "static-324")
    ancestor = documents["results/synthetic_A14/source_members.json"]["A06/checkpoint-240-324/model.npz"]
    proof = store.records["sources"]["value"]["ancestry_fingerprints"]["A14_static"]["324"]["descriptors"]
    if attack == "parameter_string":
        parameters["mass"] = "must not silently skip this required array"
    elif attack == "parameter_missing":
        parameters.pop("mass")
    elif attack == "parameter_member":
        parameters["mass"]["array_member"] = "parameters__damping"
    elif attack == "parameter_fingerprint_missing":
        parameters["mass"].pop("shape")
    elif attack == "backbone_missing":
        parameters.pop("backbone")
    elif attack == "backbone_wrong":
        parameters["backbone"] = "linear"
    elif attack == "backbone_descriptor":
        parameters["backbone"] = copy.deepcopy(parameters["mass"])
    elif attack == "unexpected_parameter":
        parameters["extra_parameter"] = copy.deepcopy(parameters["mass"])
    elif attack == "context_missing":
        context["members"].pop()
    elif attack == "context_duplicate":
        context["members"][1] = copy.deepcopy(context["members"][0])
    elif attack == "context_prefix":
        context["members"][0]["name"] = "mass"
    elif attack == "context_extra":
        context["members"].append(copy.deepcopy(context["members"][0]))
    elif attack == "ancestor_hash":
        ancestor["mass"]["sha256_c_order_bytes"] = "f" * 64
    elif attack == "ancestor_shape":
        ancestor["mass"]["shape"] = [2]
    elif attack == "ancestor_dtype":
        ancestor["mass"]["dtype"] = "<f4"
    elif attack == "ancestor_string":
        ancestor["mass"] = "not a fingerprint"
    elif attack == "proof_missing":
        proof.pop("mass")
    elif attack == "proof_extra":
        proof["backbone"] = "sine"
    else:
        proof["mass"], proof["damping"] = proof["damping"], proof["mass"]
    with pytest.raises(EvidenceError):
        audit_saved_provenance(store, plan)


@pytest.mark.parametrize("attack", ["determinant", "threshold", "event", "source_B", "future_descriptor", "ancestor_hash", "manifest", "dropped_slot"])
def test_self_consistent_compact_source_proof_cannot_replace_actual_metadata(attack):
    store, plan = provenance_fixture()
    proof = store.records["sources"]["value"]
    if attack == "determinant":
        proof["slot_bindings"]["324-000"]["determinant"]["lower"] = .5
    elif attack == "threshold":
        proof["slot_bindings"]["324-000"]["integral_parameters"]["threshold"] = 1e-10
    elif attack == "event":
        proof["slot_bindings"]["324-000"]["event_id"] = "wrong/event"
    elif attack == "source_B":
        store.records["access_ledger"]["partial_record"]["metadata_documents"]["configs/control.json"]["npz_files"][0]["sha256"] = "f" * 64
    elif attack == "future_descriptor":
        proof["slot_bindings"]["324-000"]["future_records"]["h005"]["baseline"]["sha256_c_order_bytes"] = "f" * 64
    elif attack == "ancestor_hash":
        proof["ancestry_fingerprints"]["E"]["324-h005"]["future"]["sha256_c_order_bytes"] = "f" * 64
    elif attack == "manifest":
        plan["metadata_files"][0]["sha256"] = "f" * 64
    else:
        proof["slot_bindings"].pop("324-000")
    with pytest.raises(EvidenceError):
        audit_saved_provenance(store, plan)


def physical_path_fixture():
    path = np.zeros((2, 22))
    clock, power, equilibrium, scale, mass = np.array([0., .01]), np.zeros(10), np.zeros(10), np.ones(20), np.ones(10)
    points = [{"index": i, "energy_identity_return": {"energy_rate": 0., "supply_minus_dissipation": 0., "identity_error": 0.},
               "energy_rate": 0., "supply_minus_dissipation": 0., "identity_error": 0., "q_mass_coi": 0., "v_mass_coi": 0.} for i in range(2)]
    judgement = {"qualified": True, "reasons": [], "nonfinite_values": {}, "gates": dict.fromkeys(
        ("complete_finite_physical_records", "energy_identity", "q_mass_coi", "v_mass_coi"), True),
        "point_count": 2, "valid_point_count": 2, "failed_points": [], "values": {
            "valid_records_max_identity_error": 0., "valid_records_max_q_mass_coi": 0., "valid_records_max_v_mass_coi": 0.,
            "energy_identity_bound": 1e-8, "coi_bound": 1e-10}}
    schedule = {"index_fields": ["interval_index", "substep_index", "substep_count", "numerator", "stage_code"],
                "stage_codes": {"k1": 1, "k2": 2, "k3": 3, "k4": 4}, "indices": np.empty((0, 5)),
                "returned_power": np.empty((0, 10)), "used_power": np.empty((0, 10))}
    helper = {"status": "complete", "path": path, "rollout": {"path": path, "rk4_updates": 2, "input_schedule": schedule},
              "derived_actual": {"successful_rk4_updates": 2}, "inputs": {"clock": clock, "power": power,
              "equilibrium": equilibrium, "target_scale": scale, "linear": False, "max_step": .005,
              "q": path[0, :10], "v": path[0, 10:20], "z": path[0, 20:]}, "physical_points": points,
              "checks": {"physical_judgement": judgement}}
    return helper, clock, power, equilibrium, scale, mass


def test_physical_saved_returns_connect_to_mass_and_path():
    helper, clock, power, equilibrium, scale, mass = physical_path_fixture()
    _check_saved_path(helper, clock, power, equilibrium, scale, linear=False, step=.005, mass=mass)


@pytest.mark.parametrize("attack", ["q_path", "v_path", "wrong_mass", "fake_maximum", "point_count", "energy_return"])
def test_physical_gate_cannot_hide_changed_state_or_return(attack):
    helper, clock, power, equilibrium, scale, mass = physical_path_fixture()
    if attack in {"q_path", "v_path"}:
        helper["path"][1, 0 if attack == "q_path" else 10] = 1.
    elif attack == "wrong_mass":
        helper["path"][1, :2] = [1., -1.]
        mass[0] = 2.
    elif attack == "fake_maximum":
        helper["checks"]["physical_judgement"]["values"]["valid_records_max_q_mass_coi"] = 1e-12
    elif attack == "point_count":
        helper["checks"]["physical_judgement"]["point_count"] = 1
    else:
        helper["physical_points"][1]["energy_identity_return"]["energy_rate"] = 1.
    with pytest.raises(EvidenceError):
        _check_saved_path(helper, clock, power, equilibrium, scale, linear=False, step=.005, mass=mass)
