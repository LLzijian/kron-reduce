from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import h5py
import numpy as np
import pytest

from kron_reduce.data.p1_causal import P1_TIME, P1EventRole, P1RoleManifest
from kron_reduce.data.p1_pm_targets import PmReadPurpose, read_pm_batch
from kron_reduce.physics import coi_project_power, complete_incidence


def _manifest() -> P1RoleManifest:
    rows = (
        P1EventRole("fit_a", "source_a", "line_a", "fit", 1001),
        P1EventRole("fit_b", "source_b", "line_b", "fit", 1001),
        P1EventRole("selection", "source_c", "line_c", "selection", 1001),
        P1EventRole("audit", "source_d", "line_d", "internal_audit", 1001),
    )
    return P1RoleManifest(rows, "F1", hashlib.sha256(b"synthetic pm target manifest").hexdigest())


def _sample(row: P1EventRole, manifest: P1RoleManifest, pm_clear: np.ndarray) -> dict:
    mass = np.array([1.0, 2.0, 3.0])
    incidence, _ = complete_incidence(3)
    zero = np.zeros((1001, 3))
    return {
        "event_id": row.event_id,
        "model_inputs": {"static": {
            "delta0": zero[[0]].copy(), "omega0": zero[[0]].copy(),
            "Pm_clear": pm_clear[None, :].copy(), "Pm0": pm_clear[None, :].copy(),
            "equilibrium_delta": np.zeros(3), "Pm_equilibrium": np.zeros(3),
            "mass": mass.copy(), "incidence": incidence,
        }, "exogenous": {"t": P1_TIME.copy(), "Pm": np.repeat(pm_clear[None, :], 1001, axis=0)}},
        "training_context": {
            "targets": {"t": P1_TIME.copy(), "delta": zero.copy(), "omega": zero.copy()},
            "metadata": {"event_id": row.event_id, "source_dataset": row.source_dataset, "fault_line": row.fault_line},
            "role": row.role, "role_manifest_sha256": manifest.source_sha256,
            "claim_type": "causal_zoh_rollout", "representation": "generator_10",
        },
    }


def _pm(event_index: int) -> np.ndarray:
    raw = np.column_stack((np.sin(P1_TIME) + event_index, np.cos(P1_TIME), np.full(1001, -0.25)))
    return coi_project_power(raw, np.array([1.0, 2.0, 3.0]))


def _fixture(tmp_path: Path) -> tuple[P1RoleManifest, list[dict], Path]:
    manifest = _manifest()
    pms = {row.event_id: _pm(index) for index, row in enumerate(manifest.records)}
    samples = [_sample(row, manifest, pms[row.event_id][0]) for row in manifest.records]
    path = tmp_path / "synthetic_pm.h5"
    with h5py.File(path, "w") as handle:
        for row in manifest.records:
            group = handle.create_group(f"events/{row.event_id}")
            group.attrs["outcome_label"] = "must not read"
            group.create_dataset("postclear/t", data=P1_TIME)
            group.create_dataset("postclear/Pm", data=pms[row.event_id])
            group.create_dataset("postclear/Pe", data=np.full((1001, 3), np.nan))
    return manifest, samples, path


def _for_role(samples: list[dict], role: str) -> list[dict]:
    return [sample for sample in samples if sample["training_context"]["role"] == role]


def _arrays_by_path(value, path="") -> dict[str, np.ndarray]:
    if isinstance(value, np.ndarray):
        return {path: value.copy()}
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            result.update(_arrays_by_path(child, f"{path}/{key}"))
        return result
    if isinstance(value, (list, tuple)):
        result = {}
        for index, child in enumerate(value):
            result.update(_arrays_by_path(child, f"{path}/{index}"))
        return result
    return {}


def test_fit_reads_only_pm_and_time_and_returns_readonly_copies(tmp_path, monkeypatch) -> None:
    manifest, samples, path = _fixture(tmp_path)
    reads = []
    original = h5py.Dataset.__getitem__

    def traced(dataset, selection):
        reads.append((dataset.name, selection))
        return original(dataset, selection)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", traced)
    monkeypatch.setattr(h5py.AttributeManager, "__getitem__", lambda *_: (_ for _ in ()).throw(AssertionError("attrs forbidden")))
    before = _arrays_by_path(copy.deepcopy(_for_role(samples, "fit")))
    assert before
    batch = read_pm_batch(path, manifest, _for_role(samples, "fit"), role="fit", purpose=PmReadPurpose.FIT_CLOSURE_TARGET)
    assert batch.event_ids == ("fit_a", "fit_b")
    assert batch.core_checkpoint_sha256 is None and batch.main_model_input_allowed is False
    assert {name.rsplit("/", 1)[-1] for name, _ in reads} == {"t", "Pm"}
    assert {name.split("/")[2] for name, _ in reads} == {"fit_a", "fit_b"}
    assert all(selection == slice(0, 1001) for _, selection in reads)
    assert all(item["dataset"] in {"postclear/t", "postclear/Pm"} for item in batch.accesses)
    assert all(not value.flags.writeable for value in batch.pm_by_event.values())
    with pytest.raises(ValueError):
        batch.pm_by_event["fit_a"][0, 0] = 1
    # Projection works on a copy: the evidence array itself remains untouched.
    with h5py.File(path, "r") as handle:
        assert np.array_equal(handle["events/fit_a/postclear/Pm"][:], _pm(0))
    after = _arrays_by_path(_for_role(samples, "fit"))
    assert before.keys() == after.keys()
    assert all(np.array_equal(before[key], after[key]) for key in before)


def test_conditional_reads_selection_after_verified_checkpoint(tmp_path) -> None:
    manifest, samples, path = _fixture(tmp_path)
    checkpoint = tmp_path / "core.bin"
    checkpoint.write_bytes(b"frozen core")
    expected = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    batch = read_pm_batch(path, manifest, _for_role(samples, "selection"), role="selection", purpose=PmReadPurpose.CONDITIONAL_DIAGNOSTIC, core_checkpoint=checkpoint, expected_core_sha256=expected)
    assert batch.event_ids == ("selection",)
    assert batch.core_checkpoint_sha256 == expected


@pytest.mark.parametrize("role", ["selection", "internal_audit"])
def test_fit_purpose_rejects_nonfit_before_file_open(tmp_path, monkeypatch, role) -> None:
    manifest, samples, _ = _fixture(tmp_path)
    monkeypatch.setattr(h5py, "File", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not open")))
    with pytest.raises(ValueError, match="fit closure"):
        read_pm_batch(tmp_path / "must_not_open.h5", manifest, _for_role(samples, role), role=role, purpose=PmReadPurpose.FIT_CLOSURE_TARGET)


@pytest.mark.parametrize("role,samples", [("test", []), ("unknown", []), ("fit", [])])
def test_locked_unknown_or_missing_ids_reject_before_file_open(tmp_path, monkeypatch, role, samples) -> None:
    manifest, all_samples, _ = _fixture(tmp_path)
    monkeypatch.setattr(h5py, "File", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not open")))
    requested = samples if role == "test" else _for_role(all_samples, "fit")[:-1]
    with pytest.raises(ValueError):
        read_pm_batch(tmp_path / "must_not_open.h5", manifest, requested, role=role, purpose=PmReadPurpose.FIT_CLOSURE_TARGET)


@pytest.mark.parametrize("case", ["missing", "tampered", "symlink"])
def test_conditional_checkpoint_gates_precede_file_open(tmp_path, monkeypatch, case) -> None:
    manifest, samples, _ = _fixture(tmp_path)
    checkpoint = tmp_path / "core.bin"
    checkpoint.write_bytes(b"core")
    expected = hashlib.sha256(b"core").hexdigest()
    if case == "missing":
        checkpoint_arg, expected_arg = None, None
    elif case == "tampered":
        checkpoint.write_bytes(b"altered")
        checkpoint_arg, expected_arg = checkpoint, expected
    else:
        link = tmp_path / "core_link.bin"
        try:
            link.symlink_to(checkpoint)
        except OSError:
            pytest.skip("symlinks unavailable")
        checkpoint_arg, expected_arg = link, expected
    monkeypatch.setattr(h5py, "File", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not open")))
    with pytest.raises(ValueError):
        read_pm_batch(tmp_path / "must_not_open.h5", manifest, _for_role(samples, "selection"), role="selection", purpose=PmReadPurpose.CONDITIONAL_DIAGNOSTIC, core_checkpoint=checkpoint_arg, expected_core_sha256=expected_arg)


@pytest.mark.parametrize("attack", ["nan", "initial", "projection_overflow"])
def test_invalid_pm_rejected(tmp_path, attack) -> None:
    manifest, samples, path = _fixture(tmp_path)
    with h5py.File(path, "r+") as handle:
        dataset = handle["events/fit_a/postclear/Pm"]
        changed = dataset[:]
        if attack == "nan":
            changed[10, 1] = np.nan
        elif attack == "initial":
            changed[0, 0] += 0.1
        else:
            changed[:] = np.finfo(float).max
        dataset[:] = changed
    with pytest.raises(ValueError, match="Pm"):
        read_pm_batch(path, manifest, _for_role(samples, "fit"), role="fit", purpose=PmReadPurpose.FIT_CLOSURE_TARGET)
