import hashlib
import json

import numpy as np
import pytest

from scripts import audit_state_port_saved as audit_saved


def test_journal_maps_transition_names_to_budget_names(tmp_path):
    directory = tmp_path / "journal"
    directory.mkdir()
    raw = b""
    index = 0
    for scope, expected in audit_saved.EXPECTED_SCOPES.items():
        start = len(raw)
        rows = []
        for budget, count in expected.items():
            for _ in range(count):
                index += 1
                for transition in ("attempted", "returned"):
                    rows.append({"scope_id": scope, "index": index,
                                 "category": "rollout" if budget == "rollouts" else budget,
                                 "transition": transition})
        payload = b"".join((json.dumps(row) + "\n").encode() for row in rows)
        raw += payload
        (directory / f"{scope}-reserved.json").write_text(json.dumps({
            "spent": True, "retry_allowed": False, "reserved_operation_limits": expected,
        }), encoding="utf-8")
        (directory / f"{scope}-closed.json").write_text(json.dumps({
            "status": "completed_within_budget", "durably_closed": True,
            "actual_counters": {"attempted": expected, "returned": expected, "failed": {}},
            "start_byte_offset": start, "end_byte_offset": len(raw),
            "range_sha256": hashlib.sha256(payload).hexdigest(),
        }), encoding="utf-8")
    (directory / "calls.jsonl").write_bytes(raw)
    audit = audit_saved.Audit()
    audit_saved.audit_journal(tmp_path, audit)
    assert not audit.mismatches


def test_audit_and_numpy_scalars_are_strict_json_safe():
    audit = audit_saved.Audit()
    audit.check("numpy_bool", np.bool_(True), value=np.float64(1.25))
    encoded = json.dumps({"checks": audit.checks, "scalar": audit_saved.json_safe(np.float64(2.5))}, allow_nan=False)
    decoded = json.loads(encoded)
    assert decoded["checks"] == [{"name": "numpy_bool", "passed": True, "value": 1.25}]
    assert decoded["scalar"] == 2.5


def test_decode_record_resolves_array_refs_and_loads_once(tmp_path, monkeypatch):
    values = np.array([1.25, -3.5], dtype=np.float64)
    np.savez(tmp_path / "sample.npz", values=values)
    (tmp_path / "sample.json").write_text(
        json.dumps({"payload": {"array": "values", "shape": [2], "dtype": values.dtype.str}}),
        encoding="utf-8",
    )
    calls = []
    original_load = audit_saved.np.load

    def counted_load(*args, **kwargs):
        calls.append((args, kwargs))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(audit_saved.np, "load", counted_load)
    record, arrays = audit_saved.decode_record(tmp_path, "sample")
    assert len(calls) == 1
    assert calls[0][1]["allow_pickle"] is False
    np.testing.assert_array_equal(record["payload"], values)
    np.testing.assert_array_equal(arrays["values"], values)


def test_decode_record_rejects_object_array_without_pickle(tmp_path):
    np.savez(tmp_path / "object.npz", values=np.array(["x", "y"], dtype=object))
    (tmp_path / "object.json").write_text(
        json.dumps({"payload": {"array": "values", "shape": [2], "dtype": "|O"}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Object arrays cannot be loaded"):
        audit_saved.decode_record(tmp_path, "object")


def test_run_audit_refuses_before_decoding_when_authorization_fails(tmp_path, monkeypatch):
    def refuse(_root):
        raise RuntimeError("not authorized")

    monkeypatch.setattr(audit_saved, "authorize_and_reserve", refuse)
    monkeypatch.setattr(audit_saved, "decode_record", lambda *_args: pytest.fail("decoded before authorization"))
    with pytest.raises(RuntimeError, match="not authorized"):
        audit_saved.run_audit(tmp_path)
