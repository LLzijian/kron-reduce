"""A17 failure records remain reproducible without replaying source operations."""
import json
from types import MappingProxyType

import numpy as np
import pytest

from kron_reduce.evaluation.input_consistency_evidence import (
    error_record,
    save_manifest,
    save_record,
)


def test_failure_record_keeps_arrays_dtype_nonfinite_and_all_upstream_namespaces(tmp_path):
    original = FloatingPointError("original numerical error")
    error = OSError("terminal persistence error")
    error.__cause__ = original
    error.partial_record = MappingProxyType({"a": np.array([1., np.nan, np.inf]),
                                           "descriptor": (np.dtype("<f8"), (3,)),
                                           "scalar": float("inf"), "exact_count": 123456789123456789})
    error.input_consistency_paths_partial_record = {"completed_updates": 3}
    binding = save_record(tmp_path, "fatal-prefix", error_record(error))
    doc = json.loads((tmp_path / "fatal-prefix.json").read_text())
    assert doc["partial_record"]["exact_count"] == 123456789123456789
    assert doc["partial_record"]["scalar"]["representation"] == "inf"
    assert doc["partial_record"]["descriptor"][0] == {"numpy_dtype": "<f8"}
    assert doc["input_consistency_paths_partial_record"]["completed_updates"] == 3
    assert doc["exception_chain"][0]["type"] == "FloatingPointError"
    with np.load(tmp_path / "fatal-prefix.npz", allow_pickle=False) as archive:
        actual = archive[doc["partial_record"]["a"]["array_member"]]
    assert actual[0] == 1 and np.isnan(actual[1]) and np.isposinf(actual[2])
    assert set(binding) == {"fatal-prefix.json", "fatal-prefix.npz"}
    assert save_manifest(tmp_path) == binding


def test_result_records_cannot_be_overwritten_or_escape_directory(tmp_path):
    save_record(tmp_path, "unique", {"value": np.ones(2)})
    before = (tmp_path / "unique.npz").read_bytes()
    with pytest.raises(FileExistsError):
        save_record(tmp_path, "unique", {"value": np.zeros(2)})
    assert (tmp_path / "unique.npz").read_bytes() == before
    for name in ("../escape", "a/b", "a:b", "", "a.b"):
        with pytest.raises(ValueError):
            save_record(tmp_path, name, {})


def test_fsync_failure_leaves_no_false_json_or_manifest(tmp_path, monkeypatch):
    def fail(_):
        raise OSError("injected fsync failure")

    monkeypatch.setattr("kron_reduce.evaluation.input_consistency_evidence.os.fsync", fail)
    with pytest.raises(OSError):
        save_record(tmp_path, "failed", {"value": np.ones(2)})
    assert (tmp_path / "failed.npz").exists()
    assert not (tmp_path / "failed.json").exists()
    assert not (tmp_path / "artifact_hashes.json").exists()
