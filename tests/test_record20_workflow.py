from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import kron_reduce.workflows.kiki_record20_input_audit as workflow


class _Journal:
    def __init__(self):
        self.rows = []

    def append(self, row):
        self.rows.append(dict(row))


class _ReadSpy:
    def __init__(self, stream):
        self._stream = stream
        self.read_sizes = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return self._stream.__exit__(*args)

    def read(self, size=-1):
        self.read_sizes.append(size)
        return self._stream.read(size)

    def __getattr__(self, name):
        return getattr(self._stream, name)


def test_bounded_integrity_hash_reads_exact_registered_size_without_eof_probe(tmp_path, monkeypatch):
    path = tmp_path / "payload.bin"
    payload = b"small synthetic registered bytes"
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    journal = _Journal()
    original_open = Path.open
    spies = []

    def spy_open(target, *args, **kwargs):
        stream = original_open(target, *args, **kwargs)
        if target == path:
            spy = _ReadSpy(stream)
            spies.append(spy)
            return spy
        return stream

    monkeypatch.setattr(Path, "open", spy_open)
    receipt = workflow.bounded_integrity_hash(path, len(payload), digest, journal, "before")
    assert receipt["actual_bytes"] == len(payload)
    assert receipt["sha256"] == digest
    assert spies[0].read_sizes == [len(payload)]
    assert [row["status"] for row in journal.rows] == ["attempted", "returned"]


@pytest.mark.parametrize("size,sha", [(4, hashlib.sha256(b"else").hexdigest()),
                                        (3, hashlib.sha256(b"dat").hexdigest())])
def test_bounded_integrity_hash_rejects_changed_size_or_content(tmp_path, size, sha):
    path = tmp_path / "payload.bin"
    path.write_bytes(b"data")
    journal = _Journal()
    with pytest.raises(ValueError, match="source_binding_mismatch"):
        workflow.bounded_integrity_hash(path, size, sha, journal, "before")
    assert [row["status"] for row in journal.rows] == ["attempted", "failed"]


def _approval():
    return {
        "status": "user_approved", "authority": "explicit_user_reply",
        "scientific_design_approver": "Astra Ultra", "experiment": workflow.EXPERIMENT,
        "workflow": workflow.WORKFLOW, "stage": workflow.STAGE, "module": workflow.MODULE,
        "dataset": "kiki_ieee39_v1", "allocated_train": 268, "max_hdf_opens": 4,
        "max_array_selections": 1078, "max_descriptors": 1078, "max_attribute_reads": 2966,
        "max_logical_array_elements": 7501168, "max_integrity_bytes": 6870747346,
        "max_fit_attempts": 0, "max_optimizer_updates": 0, "max_prediction_paths": 0,
        "max_total_rhs": 0, "labels_only_after_prelabel_freeze": True,
        "internal_val_payload_allowed": False, "source_val_payload_allowed": False,
        "source_test_payload_allowed": False, "model_promotion_allowed": False,
        "deployment_predictions_allowed": False,
        "scientific_review_path": "review.json", "scientific_review_sha256": "a" * 64,
        "implementation_bindings": {workflow.MODULE.replace(".", "/") + ".py": "b" * 64},
    }


def _authorization_environment(monkeypatch, approval):
    config = {
        "input_contract": workflow.CONTRACT,
        "read_plan_path": "read-plan.json", "read_plan_sha256": "c" * 64,
        "source_evidence_path": "source-evidence.json", "source_evidence_sha256": "d" * 64,
    }
    approval.update(config)
    experiment = {"execution_mode": "bounded_input_audit_cli_only", "config": "audit.toml"}
    monkeypatch.setattr(workflow, "validate_run_policy", lambda *_: None)
    monkeypatch.setattr(workflow, "_verify_development_text_bindings", lambda *_: None)
    monkeypatch.setattr(workflow, "load_policy", lambda *_: {"development_evaluations": {workflow.AUTH: approval}})
    monkeypatch.setattr(workflow, "load_manifest", lambda *_: {"experiments": {workflow.EXPERIMENT: experiment}})
    verified = []
    monkeypatch.setattr(workflow, "verify_bound_file", lambda *args: verified.append(args[1:]))
    monkeypatch.setattr(workflow, "assert_clean", lambda *_: None)
    return config, experiment, verified


@pytest.mark.parametrize("status", ["consumed", "revoked"])
def test_inactive_authorization_stops_before_any_bound_source_check(monkeypatch, tmp_path, status):
    approval = _approval()
    approval["status"] = status
    config, experiment, verified = _authorization_environment(monkeypatch, approval)
    source_calls = []
    with pytest.raises(PermissionError, match="active bounded"):
        workflow.authorization(tmp_path, experiment, config)
    assert verified == []
    assert source_calls == []


def test_changed_registered_config_binding_is_rejected_before_source_access(monkeypatch, tmp_path):
    approval = _approval()
    config, experiment, verified = _authorization_environment(monkeypatch, approval)
    config["read_plan_sha256"] = "e" * 64
    with pytest.raises(PermissionError, match="config differs"):
        workflow.authorization(tmp_path, experiment, config)
    assert verified == []


def test_changed_bound_file_is_rejected_before_any_source_reader_is_created(monkeypatch, tmp_path):
    approval = _approval()
    config, experiment, _ = _authorization_environment(monkeypatch, approval)
    verified = []

    def reject_changed_binding(root, relative, digest):
        verified.append(relative)
        if relative == config["read_plan_path"]:
            raise PermissionError("bound file digest changed")

    monkeypatch.setattr(workflow, "verify_bound_file", reject_changed_binding)
    reader_constructions = []
    monkeypatch.setattr(workflow, "LimitedHdfReader", lambda *a, **k: reader_constructions.append(True))
    with pytest.raises(PermissionError, match="digest changed"):
        workflow.authorization(tmp_path, experiment, config)
    assert verified == [config["read_plan_path"]]
    assert reader_constructions == []


def test_outer_authorization_failure_never_reaches_inner_runner_or_data_verification(
    monkeypatch, tmp_path,
):
    (tmp_path / "configs/experiments").mkdir(parents=True)
    (tmp_path / "configs/experiments/audit.toml").write_text(
        'input_contract = "synthetic"\n', encoding="utf-8"
    )
    experiment = {"execution_mode": "bounded_input_audit_cli_only", "config": "audit.toml"}
    monkeypatch.setattr(workflow, "load_manifest", lambda *_: {"experiments": {workflow.EXPERIMENT: experiment}})
    monkeypatch.setattr(workflow, "authorization", lambda *_: (_ for _ in ()).throw(PermissionError("inactive")))
    metadata_calls, runner_calls = [], []
    monkeypatch.setattr(workflow, "bound_metadata", lambda *_: metadata_calls.append(True))
    monkeypatch.setattr(workflow, "_run_experiment", lambda *_: runner_calls.append(True))
    with pytest.raises(PermissionError, match="inactive"):
        workflow.execute_registered_audit(tmp_path)
    assert metadata_calls == []
    assert runner_calls == []


def _synthetic_plan():
    sources = sorted(workflow.ALLOWED_SOURCES)
    return {
        "source_inventory": {
            source: {
                "metadata_stat_only": {"size_bytes": 1},
                "dataset_h5_registered_sha256": "0" * 64,
            }
            for source in sources
        },
        "slots": [
            {
                "slot": index,
                "event_id": f"synthetic-{index:03d}",
                "source_dataset": sources[index % len(sources)],
                "raw_id": f"raw-{index:03d}",
                "fault_line": f"L{index % 4}",
                "declared_length": 12,
            }
            for index in range(268)
        ],
    }


class _SyntheticLedger:
    def __init__(self):
        self.events = []

    def close_journal(self):
        self.events.append("close")

    def close_phase1_journal(self):
        self.events.append("close_phase1")

    def rotate_to_phase2_journal(self, path):
        self.events.append(("rotate", Path(path).name))


class _SyntheticReader:
    def __init__(self, plan, paths, **kwargs):
        self.plan = plan
        self.ledger = _SyntheticLedger()
        self.events = []
        self.phase2_calls = 0

    def read_phase1_source(self, source):
        return {}

    def audit_log(self):
        return {"synthetic": True, "phase": "phase1"}

    def seal_prelabel_freeze(self, sha):
        self.events.append(("seal", sha))

    def read_phase2_source(self, source):
        self.phase2_calls += 1
        self.events.append(("phase2", source))
        raise RuntimeError("synthetic stop at first metadata read")


def _execute_inputs(tmp_path):
    result_dir = tmp_path / "run"
    result_dir.mkdir()
    config = {"source_root": str(tmp_path), "claim_boundary": "synthetic-only"}
    plan = _synthetic_plan()
    registry = {}
    receipt = {"run_id": "synthetic", "source_commit": "deadbeef"}
    return result_dir, config, plan, registry, receipt


def _mock_integrity(monkeypatch):
    calls = []

    def fake_hash(path, expected_size, expected_sha, journal, phase):
        calls.append(phase)
        journal.append({"kind": "synthetic_integrity", "phase": phase, "status": "returned"})
        return {"phase": phase, "actual_bytes": expected_size, "sha256": expected_sha}

    monkeypatch.setattr(workflow, "bounded_integrity_hash", fake_hash)
    return calls


def test_prelabel_freeze_write_failure_keeps_all_allocated_slots_and_reads_no_labels(
    monkeypatch, tmp_path,
):
    integrity_calls = _mock_integrity(monkeypatch)
    reader = _SyntheticReader(None, None)
    monkeypatch.setattr(workflow, "LimitedHdfReader", lambda *args, **kwargs: reader)
    result_dir, config, plan, registry, receipt = _execute_inputs(tmp_path)
    original_write_json = workflow.write_json

    def fail_freeze(path, value):
        if Path(path).name == "prelabel_freeze.json":
            raise OSError("synthetic freeze persistence failure")
        original_write_json(path, value)

    monkeypatch.setattr(workflow, "write_json", fail_freeze)
    with pytest.raises(OSError, match="freeze persistence"):
        workflow._execute(tmp_path, result_dir, config, plan, registry, receipt)
    slots = json.loads((result_dir / "prelabel_slots.json").read_text(encoding="utf-8"))
    assert len(slots) == 268
    assert reader.phase2_calls == 0
    assert integrity_calls == ["before"] * len(workflow.ALLOWED_SOURCES)
    failure = json.loads((result_dir / "input_audit_failure.json").read_text(encoding="utf-8"))
    assert failure["allocated"] == 268
    assert failure["compute"] == workflow.ZERO_COMPUTE


def test_phase2_label_access_starts_only_after_freeze_and_preserves_full_population(
    monkeypatch, tmp_path,
):
    _mock_integrity(monkeypatch)
    reader = _SyntheticReader(None, None)
    monkeypatch.setattr(workflow, "LimitedHdfReader", lambda *args, **kwargs: reader)
    result_dir, config, plan, registry, receipt = _execute_inputs(tmp_path)
    with pytest.raises(RuntimeError, match="first metadata read"):
        workflow._execute(tmp_path, result_dir, config, plan, registry, receipt)
    freeze_path = result_dir / "prelabel_freeze.json"
    assert freeze_path.exists()
    assert reader.ledger.events[0] == "close_phase1"
    assert reader.ledger.events[1] == ("rotate", "phase2_access.jsonl")
    assert reader.events[0][0] == "seal"
    assert reader.events[0][1] == hashlib.sha256(freeze_path.read_bytes()).hexdigest()
    assert reader.events[1] == ("phase2", min(workflow.ALLOWED_SOURCES))
    slots = json.loads((result_dir / "prelabel_slots.json").read_text(encoding="utf-8"))
    assert len(slots) == 268
    assert all(slot["rule"]["status"] == "invalid" for slot in slots)


class _MemoryDataset:
    def __init__(self, value, name, read_log):
        self.value = np.asarray(value)
        self.name = name
        self.read_log = read_log

    @property
    def shape(self):
        return self.value.shape

    @property
    def dtype(self):
        return self.value.dtype

    def __getitem__(self, selection):
        self.read_log.append((self.name, selection))
        return self.value[selection]


class _FailOnceMemoryDataset(_MemoryDataset):
    def __init__(self, value, name, read_log):
        super().__init__(value, name, read_log)
        self.payload_attempts = 0

    def __getitem__(self, selection):
        self.payload_attempts += 1
        self.read_log.append((self.name, selection))
        if self.payload_attempts == 1:
            raise OSError("synthetic one-time Pm payload read failure")
        return self.value[selection]


class _MemoryAttrs:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, name):
        return self.values[name]


class _MemoryGroup:
    def __init__(self, entries=None, attrs=None):
        self.entries = entries or {}
        self.attrs = _MemoryAttrs(attrs or {})

    def __getitem__(self, name):
        if "/" not in name:
            return self.entries[name]
        head, tail = name.split("/", 1)
        return self.entries[head][tail]


class _MemoryFile(_MemoryGroup):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def _complete_synthetic_plan():
    proposal_path = Path(__file__).resolve().parents[1] / "configs/proposals/kiki_a12_record20_reads.json"
    plan = json.loads(proposal_path.read_text(encoding="utf-8"))
    sources = sorted(workflow.ALLOWED_SOURCES)
    lengths = [1501] * 100 + [5] * 10 + [1310] * 96 + [1309] * 62
    assert len(lengths) == 268 and sum(lengths) == 357068
    rows = []
    for source_index, source in enumerate(sources):
        for local_index in range(134):
            index = source_index * 134 + local_index
            raw_id = f"synthetic-{local_index:03d}"
            length = lengths[index]
            rows.append({
                "slot": index, "event_id": f"{source}/{raw_id}", "raw_id": raw_id,
                "source_dataset": source, "role": "train", "source_role": "train",
                "fault_line": f"Line_{index % 39 + 1}", "declared_length": length,
                "phase1_array_selections": [
                    {"dataset": "t", "selection": f"[0:{length}]", "expected_shape": f"({length},)"},
                    {"dataset": "delta", "selection": f"[0:{length}, 0:10]", "expected_shape": f"({length},10)"},
                    {"dataset": "omega", "selection": f"[0:{length}, 0:10]", "expected_shape": f"({length},10)"},
                    {"dataset": "Pm", "selection": "[2, 0:10]", "expected_shape": "(10,)",
                     "attempt_only_if_index_exists": True},
                ],
                "phase2_event_attributes": list(workflow.EVENT_ATTRS),
            })
    rows.sort(key=lambda row: (row["source_dataset"], row["raw_id"]))
    for slot, row in enumerate(rows):
        row["slot"] = slot
    plan["slots"] = rows
    plan["fixed_population"]["declared_length_sum_S"] = sum(row["declared_length"] for row in rows)
    plan["fixed_population"]["old_N_ge_12_common_count"] = sum(row["declared_length"] >= 12 for row in rows)
    plan["fixed_population"]["old_N_lt_12_count"] = sum(row["declared_length"] < 12 for row in rows)
    scalar_count = 21 * sum(row["declared_length"] for row in rows) + 2740
    plan["phase1"]["planned_scalar_count"] = scalar_count
    plan["phase1"]["planned_float64_logical_bytes"] = scalar_count * 8
    plan["phase2"]["event_attribute_read_count"] = 268 * len(workflow.EVENT_ATTRS)
    sizes = {sources[0]: 1_500_000_000, sources[1]: 1_935_373_673}
    for source in sources:
        plan["source_inventory"][source].update({
            "dataset_h5_relative_path": f"{source}/dataset.h5",
            "dataset_h5_registered_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "metadata_stat_only": {"size_bytes": sizes[source]},
        })
    return plan


def _synthetic_registry_and_files(plan, result_dir):
    read_log, files, registry = [], {}, {"shards": {}}
    signature_by_source = {source: f"synthetic-signature-{i}" for i, source in enumerate(sorted(workflow.ALLOWED_SOURCES))}
    state_mapping = [[i, "delta" if i < 10 else "omega", "GENROU", f"GENROU_{i % 10 + 1}"]
                     for i in range(20)]
    for source in sorted(workflow.ALLOWED_SOURCES):
        attrs = {
            "andes_version": "synthetic", "case_sha256": "case-synthetic",
            "config": json.dumps({"synthetic": True}), "git_commit": "synthetic",
            "mva_base": 100.0, "omega_s": 120 * np.pi,
            "postfault_signature": signature_by_source[source], "raw_units": "synthetic",
            "state_mapping": json.dumps(state_mapping),
        }
        registry["shards"][source] = {"attributes": attrs}
        events = {}
        source_rows = [row for row in plan["slots"] if row["source_dataset"] == source]
        for row in source_rows:
            index = row["slot"]
            n = row["declared_length"]
            t = np.arange(n, dtype=np.float64) * 0.01
            delta = np.zeros((n, 10), dtype=np.float64)
            omega = np.ones((n, 10), dtype=np.float64)
            if 100 <= index < 268:
                delta[-5:, 0] = 4.0
                omega[-5:, 0] = 1.03
            pm = np.zeros((n, 10), dtype=np.float64)
            label = "stable" if index < 100 else "unstable"
            event_attrs = {
                "label": label,
                "event_time_s": "" if label == "stable" else float(t[-1]),
                "termination_reason": "stable_horizon_reached" if label == "stable"
                                      else "physical_loss_of_synchronism",
                "trajectory_end_s": float(t[-1]), "valid_length": int(n),
                "clearing_time_s": 0.08, "record_start_delay_s": 0.005,
                "solver_success": True, "failed_steps": 0,
                "postfault_signature": signature_by_source[source],
                "fault_line": row["fault_line"],
            }
            arrays = {
                name: _MemoryDataset(value, (source, row["raw_id"], name), read_log)
                for name, value in (("t", t), ("delta", delta), ("omega", omega), ("Pm", pm))
            }
            events[row["raw_id"]] = _MemoryGroup(arrays, event_attrs)
        static = {
            "inertia": _MemoryDataset(np.ones(10), (source, "inertia"), read_log),
            "equilibrium": _MemoryGroup({
                "delta": _MemoryDataset(np.zeros(10), (source, "equilibrium/delta"), read_log),
                "Pm": _MemoryDataset(np.zeros(10), (source, "equilibrium/Pm"), read_log),
            }),
            "events": _MemoryGroup(events),
        }
        files[source] = _MemoryFile(static, attrs)

    opened = []

    def open_synthetic(path, mode):
        source = Path(path).parent.name
        opened.append((source, mode))
        if len(opened) > 2:
            assert (result_dir / "prelabel_freeze.json").exists()
        return files[source]

    return registry, files, read_log, opened, open_synthetic


def test_execute_complete_synthetic_run_accepts_with_freeze_boundaries_and_no_model_work(
    monkeypatch, tmp_path,
):
    result_dir = tmp_path / "results/successful-run"
    result_dir.mkdir(parents=True)
    plan = _complete_synthetic_plan()
    config = {"source_root": str(tmp_path), "claim_boundary": "synthetic record-20 audit only"}
    receipt = {"run_id": "synthetic-complete", "source_commit": "synthetic-commit"}
    registry, _files, read_log, opened, open_synthetic = _synthetic_registry_and_files(plan, result_dir)

    def fake_integrity(path, expected_size, expected_sha, journal, phase):
        journal.append({"kind": "integrity_hash", "phase": phase, "status": "returned",
                        "actual_bytes": expected_size, "sha256": expected_sha})
        return {"phase": phase, "actual_bytes": expected_size, "sha256": expected_sha}

    monkeypatch.setattr(workflow, "bounded_integrity_hash", fake_integrity)
    monkeypatch.setattr(workflow.h5py, "File", open_synthetic)
    summary = workflow._execute(tmp_path, result_dir, config, plan, registry, receipt)

    assert summary["execution_completed"] is True
    assert summary["scientific_verdict"] == "ACCEPT"
    assert summary["record20_train_contract_supported"] is True
    assert summary["budget_checks_pass"] is True
    assert summary["allocated"] == 268
    assert summary["prelabel_groups"]["all_268"]["allocated"] == 268
    assert summary["prelabel_groups"]["old100_common_258"]["allocated"] == 258
    assert summary["prelabel_groups"]["old100_unavailable_10"]["allocated"] == 10
    assert summary["metadata_consistent_count"] == 268
    assert summary["compute"] == workflow.ZERO_COMPUTE
    assert summary["new_training_authorized"] is False
    sorted_sources = sorted(workflow.ALLOWED_SOURCES)
    assert opened == [(source, "r") for source in sorted_sources] * 2
    pm_reads = [
        (dataset, selection) for dataset, selection in read_log
        if len(dataset) == 3 and dataset[2] == "Pm"
    ]
    assert len(pm_reads) == 268
    assert all(selection == (2, slice(0, 10)) for _dataset, selection in pm_reads)

    prelabel_slots = json.loads((result_dir / "prelabel_slots.json").read_text(encoding="utf-8"))
    postlabel_slots = json.loads((result_dir / "postlabel_slots.json").read_text(encoding="utf-8"))
    assert len(prelabel_slots) == len(postlabel_slots) == 268
    assert sum(row["input"]["record20_input_supported"] for row in prelabel_slots) == 268
    assert sum(row["rule"]["status"] == "valid" for row in prelabel_slots) == 268
    assert all(row["metadata"]["consistent"] for row in postlabel_slots)
    assert sum(row["metadata"]["label"] == "stable" for row in postlabel_slots) == 100
    assert sum(row["metadata"]["label"] == "unstable" for row in postlabel_slots) == 168
    assert len(list((result_dir / "inputs").glob("slot-*.npz"))) == 268

    freeze_path = result_dir / "prelabel_freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    for relative, digest in freeze["files"].items():
        assert hashlib.sha256((result_dir / relative).read_bytes()).hexdigest() == digest
    with np.load(result_dir / "inputs/slot-0000.npz", allow_pickle=False) as input_arrays:
        assert set(input_arrays.files) == {
            "time", "q", "v", "power", "mass", "equilibrium_delta", "equilibrium_pm", "omega_s",
        }
        assert all(input_arrays[name].dtype.kind in "f" for name in input_arrays.files)
    input_text = (result_dir / "inputs/slot-0000.npz").read_bytes()
    assert b"stable" not in input_text and b"unstable" not in input_text
    assert b"fault_line" not in input_text and b"event_id" not in input_text


def test_execute_keeps_visible_rule_diagnostic_when_one_origin_power_payload_fails(
    monkeypatch, tmp_path,
):
    result_dir = tmp_path / "results/partial-input-run"
    result_dir.parent.mkdir()
    result_dir.mkdir()
    plan = _complete_synthetic_plan()
    config = {"source_root": str(tmp_path), "claim_boundary": "synthetic partial input diagnostic"}
    receipt = {"run_id": "synthetic-pm-failure", "source_commit": "synthetic-commit"}
    registry, files, read_log, opened, open_synthetic = _synthetic_registry_and_files(plan, result_dir)
    row0 = plan["slots"][0]
    pm_dataset = _FailOnceMemoryDataset(
        np.zeros((row0["declared_length"], 10), dtype=np.float64),
        (row0["source_dataset"], row0["raw_id"], "Pm"), read_log,
    )
    files[row0["source_dataset"]].entries["events"].entries[row0["raw_id"]].entries["Pm"] = pm_dataset

    def fake_integrity(path, expected_size, expected_sha, journal, phase):
        journal.append({"kind": "integrity_hash", "phase": phase, "status": "returned",
                        "actual_bytes": expected_size, "sha256": expected_sha})
        return {"phase": phase, "actual_bytes": expected_size, "sha256": expected_sha}

    monkeypatch.setattr(workflow, "bounded_integrity_hash", fake_integrity)
    monkeypatch.setattr(workflow.h5py, "File", open_synthetic)
    summary = workflow._execute(tmp_path, result_dir, config, plan, registry, receipt)

    assert summary["scientific_verdict"] == "REJECT"
    assert summary["budget_checks_pass"] is False
    assert summary["allocated"] == 268
    assert summary["compute"] == workflow.ZERO_COMPUTE
    assert summary["metadata_consistent_count"] == 268
    assert opened == [(source, "r") for source in sorted(workflow.ALLOWED_SOURCES)] * 2
    assert pm_dataset.payload_attempts == 1
    assert sum(1 for name, _selection in read_log if name == pm_dataset.name) == 1

    slots = json.loads((result_dir / "prelabel_slots.json").read_text(encoding="utf-8"))
    slot0 = slots[0]
    assert len(slots) == 268
    assert slot0["input"]["history_available"] is False
    assert slot0["input"]["at_least_two_real_future_points"] is True
    assert slot0["input"]["full_visible_record_valid"] is True
    assert slot0["input"]["record20_input_supported"] is False
    assert slot0["rule"]["status"] == "valid"
    assert not (result_dir / "inputs/slot-0000.npz").exists()
    with np.load(result_dir / "allowed_arrays.npz", allow_pickle=False) as saved:
        assert "s0000_Pm" not in saved.files
        assert "s0000_transformed_power" not in saved.files
        assert "s0000_transformed_q" in saved.files
        assert "s0000_transformed_v" in saved.files

    postlabel = json.loads((result_dir / "postlabel_slots.json").read_text(encoding="utf-8"))
    assert len(postlabel) == 268
    assert all(row["metadata"]["consistent"] for row in postlabel)
    ledger = json.loads((result_dir / "access_ledger.json").read_text(encoding="utf-8"))
    failed = [selector for selector in ledger["selectors"] if selector["status"] == "failed"]
    assert len(failed) == 1
    assert failed[0]["key"] == (
        f"event-array:{row0['source_dataset']}:{row0['event_id']}:Pm"
    )
    # Exercise the unchanged inner runner's strict artifact and v2 stage
    # evidence checks before spending the single real-data execution slot.
    from project_harness.evidence import (
        EvidenceContext,
        sha256_file,
        validate_stage_evidence,
        verify_artifact_directory,
    )
    verify_artifact_directory(result_dir, strict=True)
    evidence = json.loads((result_dir / "stage_evidence.json").read_text(encoding="utf-8"))
    evidence["result_dir"] = result_dir.name
    evidence["artifact_manifest"] = {
        "run_id": result_dir.name, "path": "artifact_hashes.json",
        "sha256": sha256_file(result_dir / "artifact_hashes.json"),
    }
    frozen_workflow = json.loads((Path(__file__).resolve().parents[1]
        / "configs/workflows/kiki_a12_record20_input_v1.json").read_text(encoding="utf-8"))
    validate_stage_evidence(EvidenceContext(root=tmp_path, workflow=frozen_workflow,
                            target="input_audit", predecessor="scaffold"), evidence)


def test_reserve_budget_is_one_use_and_never_touches_a_real_git_budget_directory(
    monkeypatch, tmp_path,
):
    private_budget = tmp_path / "synthetic-budget"
    monkeypatch.setattr(workflow, "budget_directory", lambda _root: private_budget)
    monkeypatch.setattr(workflow, "git_head", lambda _root: "synthetic-head")
    config = {"read_plan_sha256": "a" * 64}
    directory, first_receipt = workflow.reserve_budget(tmp_path, "run-one", config)
    reservation_path = directory / "reservation.json"
    original_bytes = reservation_path.read_bytes()
    original_receipt = json.loads(original_bytes)
    assert first_receipt["spent"] is True
    assert original_receipt["run_id"] == "run-one"
    assert original_receipt["source_commit"] == "synthetic-head"
    with pytest.raises(FileExistsError):
        workflow.reserve_budget(tmp_path, "run-two", config)
    assert reservation_path.read_bytes() == original_bytes


def test_run_order_and_failed_execution_spends_budget_without_retry(
    monkeypatch, tmp_path,
):
    config_dir = tmp_path / "configs/experiments"
    config_dir.mkdir(parents=True)
    (config_dir / "synthetic.toml").write_text(
        'read_plan_sha256 = "' + "b" * 64 + '"\n', encoding="utf-8"
    )
    experiment = {"config": "synthetic.toml"}
    private_budget = tmp_path / "one-use-budget"
    monkeypatch.setattr(workflow, "budget_directory", lambda _root: private_budget)
    monkeypatch.setattr(workflow, "git_head", lambda _root: "synthetic-head")
    events = []

    def authorization(*_args):
        events.append("authorization")
        return {"scientific_review_sha256": "c" * 64}

    def bound_metadata(*_args):
        events.append("bound_metadata")
        return {"synthetic": "plan"}, {"synthetic": "registry"}

    real_reserve = workflow.reserve_budget

    def reserve(*args):
        events.append("reserve_budget")
        return real_reserve(*args)

    def fail_execute(*_args):
        events.append("execute")
        raise OSError("synthetic source audit failure")

    monkeypatch.setattr(workflow, "authorization", authorization)
    monkeypatch.setattr(workflow, "bound_metadata", bound_metadata)
    monkeypatch.setattr(workflow, "reserve_budget", reserve)
    monkeypatch.setattr(workflow, "_execute", fail_execute)

    with pytest.raises(OSError, match="synthetic source audit failure"):
        workflow.run(
            root=tmp_path, experiment=experiment, stage=workflow.STAGE,
            result_dir=tmp_path / "first-run", run_id="one-use-run",
        )
    assert events == ["authorization", "bound_metadata", "reserve_budget", "execute"]
    budget_dir = private_budget
    reservation_path = budget_dir / "reservation.json"
    reservation_bytes = reservation_path.read_bytes()
    failed = json.loads((budget_dir / "failed.json").read_text(encoding="utf-8"))
    assert failed["run_id"] == "one-use-run"
    assert failed["authorization_spent"] is True
    assert failed["automatic_retry"] is False

    events.clear()
    with pytest.raises(FileExistsError):
        workflow.run(
            root=tmp_path, experiment=experiment, stage=workflow.STAGE,
            result_dir=tmp_path / "second-run", run_id="one-use-run-retry",
        )
    assert events == ["authorization", "bound_metadata", "reserve_budget"]
    assert reservation_path.read_bytes() == reservation_bytes
