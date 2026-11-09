from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from kron_reduce.data import energy_saved_inputs as saved_inputs
from kron_reduce.data.energy_saved_inputs import SavedEnergyInputs, SavedEnergyInputsError

_PLAN_PATH = Path(__file__).parents[1] / "configs/proposals/constant_c_energy_saved_sources_v1.json"
_CASE_IDS = (
    "L100_clean", "L100_alt", "L100_weakdir", "S100_clean", "S100_alt",
    "W100_clean", "W100_weakdir", "K100_mismatch", "U_hidden_null",
)


class _FakeOps:
    def __init__(self, *, cap=214):
        self.scope_id = "source-inputs"
        self.caps = {"source-inputs": {"probe": cap}}
        self.counts = Counter()
        self.tags = []

    def call(self, category, function, *, tag=""):
        assert category == "probe"
        self.counts["attempted"] += 1
        self.tags.append(tag)
        try:
            value = function()
        except BaseException:
            self.counts["failed"] += 1
            raise
        self.counts["returned"] += 1
        return value


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _expanded_members(plan):
    whitelist = plan["npz_member_whitelist"]
    result = {
        "linear-fixtures": whitelist["linear-fixtures.npz"],
        "sine-truth": whitelist["sine-truth.npz"],
        "U_hidden_null": whitelist["U_hidden_null.npz"],
    }
    for key, base, expected_cases in (
        ("linear_100ms_cases", "L100_clean", _CASE_IDS[:3] + _CASE_IDS[5:8]),
        ("sine_cases", "S100_clean", _CASE_IDS[3:5]),
    ):
        template = whitelist[key]["member_template"]
        assert whitelist[key]["case_ids"] == list(expected_cases)
        prefix = base + "__"
        for case in expected_cases:
            result[case] = [
                {**member, "member": case + "__" + member["member"][len(prefix):]}
                for member in template
            ]
    return result


def _write_npz(path: Path, members, *, stem: str, fault=None, add_unregistered=False):
    arrays = {}
    for spec in members:
        name = spec["member"]
        shape = tuple(spec["shape"])
        dtype = np.dtype(spec["dtype"])
        if fault and fault[0] == stem and fault[1] == name:
            if fault[2] == "shape":
                shape = shape + (1,)
            elif fault[2] == "dtype":
                dtype = np.dtype("<f4")
            elif fault[2] == "nonfinite":
                array = np.full(shape, np.nan, dtype=dtype)
                arrays[name] = array
                continue
        arrays[name] = np.zeros(shape, dtype=dtype)
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    if add_unregistered:
        with zipfile.ZipFile(stream, "a") as archive:
            archive.writestr("UNREGISTERED.npy", b"not an NPY file; never decode")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(stream.getvalue())


def _fixture(tmp_path: Path, monkeypatch, *, fault=None, add_unregistered=False):
    plan = json.loads(_PLAN_PATH.read_text(encoding="utf-8"))
    plan = deepcopy(plan)
    root = tmp_path / "repo"
    run_dir = root / "results" / plan["phase1_source"]["run_id"]
    run_dir.mkdir(parents=True)
    expanded = _expanded_members(plan)

    for stem, members in expanded.items():
        _write_npz(run_dir / f"{stem}.npz", members, stem=stem, fault=fault,
                   add_unregistered=add_unregistered and stem == "L100_clean")
        descriptors = [
            {"array": member["member"], "shape": member["shape"], "dtype": member["dtype"]}
            for member in members
        ]
        if stem == "L100_clean":
            descriptors.append({
                "array": "NOT_REGISTERED__array", "shape": [1], "dtype": "<f8",
            })
        (run_dir / f"{stem}.json").write_text(json.dumps({
            "note": "synthetic fixture metadata",
            "scalar_check": 7,
            "nested": {"array_descriptors": descriptors},
        }), encoding="utf-8")

    for stem in ("execution_freeze", "summary"):
        (run_dir / f"{stem}.json").write_text(
            json.dumps({"metadata_only": stem}), encoding="utf-8"
        )

    files = {}
    for path in sorted(run_dir.iterdir()):
        files[f"results/{run_dir.name}/{path.name}"] = _sha(path.read_bytes())
    artifact_contents = {
        rel.removeprefix(f"results/{run_dir.name}/"): digest
        for rel, digest in files.items()
    }
    for index in range(80 - len(artifact_contents)):
        artifact_contents[f"unread-entry-{index:02d}.json"] = "0" * 64
    artifact_path = run_dir / "artifact_hashes.json"
    artifact_path.write_text(json.dumps(artifact_contents), encoding="utf-8")
    artifact_digest = _sha(artifact_path.read_bytes())
    files[f"results/{run_dir.name}/artifact_hashes.json"] = artifact_digest
    plan["files"] = files
    plan["phase1_source"]["artifact_manifest_sha256"] = artifact_digest
    monkeypatch.setattr(saved_inputs, "ARTIFACT_MANIFEST_SHA256", artifact_digest)
    return root, plan


def test_verify_and_load_only_registered_members_once(tmp_path, monkeypatch):
    root, plan = _fixture(tmp_path, monkeypatch, add_unregistered=True)
    ops = _FakeOps()
    reader = SavedEnergyInputs(root, plan, ops)

    verified = reader.verify()
    assert verified["source_files_verified"] == 25
    assert verified["json_file_count"] == 14
    assert verified["npz_file_count"] == 11
    assert ops.counts["returned"] == 25

    loaded = reader.load("L100_clean")
    assert loaded["note"] == "synthetic fixture metadata"
    assert loaded["scalar_check"] == 7
    arrays = loaded["nested"]["array_descriptors"]
    assert isinstance(arrays[0], np.ndarray)
    assert arrays[0].shape == (11, 20)
    assert arrays[0].dtype == np.dtype("<f8")
    assert arrays[0].flags.writeable is False
    assert arrays[-1]["array"] == "NOT_REGISTERED__array"
    before = ops.counts.copy()
    assert reader.load("L100_clean") is loaded
    assert ops.counts == before

    for stem in reader._members:
        reader.load(stem)
    report = reader.report()
    assert report["source_files_verified"] == 25
    assert report["npz_files_loaded"] == 11
    assert report["array_members_registered"] == 189
    assert report["array_members_decoded"] == 189
    assert report["duplicate_payload_reads"] == 0
    assert report["probe_calls"] == 214
    assert ops.counts["returned"] == 214


def test_scope_must_pre_reserve_exact_finite_read_budget(tmp_path, monkeypatch):
    root, plan = _fixture(tmp_path, monkeypatch)
    ops = _FakeOps(cap=213)
    reader = SavedEnergyInputs(root, plan, ops)
    with pytest.raises(SavedEnergyInputsError, match="exactly 214"):
        reader.verify()
    assert ops.counts["attempted"] == 0


def test_path_escape_is_rejected_before_any_source_read(tmp_path, monkeypatch):
    root, plan = _fixture(tmp_path, monkeypatch)
    first = next(iter(plan["files"]))
    digest = plan["files"].pop(first)
    plan["files"][f"results/{plan['phase1_source']['run_id']}/../outside.json"] = digest
    ops = _FakeOps()
    with pytest.raises(ValueError, match="file allowlist mismatch"):
        SavedEnergyInputs(root, plan, ops)
    assert ops.counts["attempted"] == 0


def test_source_hash_mismatch_spends_one_probe_and_closes_reader(tmp_path, monkeypatch):
    root, plan = _fixture(tmp_path, monkeypatch)
    relative = next(iter(plan["files"]))
    (root / relative).write_bytes(b"tampered source")
    ops = _FakeOps()
    reader = SavedEnergyInputs(root, plan, ops)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        reader.verify()
    with pytest.raises(SavedEnergyInputsError, match="closed"):
        reader.load("L100_clean")
    assert ops.counts["attempted"] == 1
    assert ops.counts["failed"] == 1


@pytest.mark.parametrize("fault_kind", ["shape", "dtype", "nonfinite"])
def test_member_shape_dtype_and_finiteness_are_checked(tmp_path, monkeypatch, fault_kind):
    plan0 = json.loads(_PLAN_PATH.read_text(encoding="utf-8"))
    first_member = _expanded_members(plan0)["L100_clean"][0]["member"]
    root, plan = _fixture(
        tmp_path, monkeypatch, fault=("L100_clean", first_member, fault_kind)
    )
    ops = _FakeOps()
    reader = SavedEnergyInputs(root, plan, ops)
    reader.verify()
    with pytest.raises(ValueError, match={
        "shape": "shape mismatch", "dtype": "dtype mismatch", "nonfinite": "non-finite",
    }[fault_kind]):
        reader.load("L100_clean")
    report = reader.report()
    assert report["array_members_attempted"] == 1
    assert report["array_members_failed"] == 1
    assert report["array_members_decoded"] == 0
