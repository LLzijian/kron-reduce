from __future__ import annotations

import json
import math
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from kron_reduce.evaluation import energy_cap_transfer as transfer


class FakeJournal:
    def __init__(self):
        self.entries = []

    def append(self, entry):
        self.entries.append(dict(entry))

    def close(self):
        return None


class FakeOperations:
    def __init__(self, *, integration_limit=16, physical_limit=3):
        self.scope_id = "test-scope"
        self.limits = {
            "rollouts": 1,
            "integration_rhs": integration_limit,
            "physical_rhs": physical_limit,
        }
        self.counts = {key: 0 for key in self.limits}
        self.returns = {key: 0 for key in self.limits}
        self.index = 0
        self.journal = FakeJournal()
        self.last_rhs = {}
        self.rhs_context = {}
        self.partial = {}

    def call(self, category, function, *, tag=""):
        del tag
        if self.counts[category] >= self.limits[category]:
            raise RuntimeError(f"budget exhausted: {category}")
        self.counts[category] += 1
        value = function()
        self.returns[category] += 1
        return value


class FakeLinearModel:
    def __init__(self, *, fail_physics=False):
        self.mass = np.ones(10)
        self.fail_physics = fail_physics
        self.rhs_calls = 0
        self.physical_calls = 0

    def rhs(self, state, _power, _equilibrium):
        self.rhs_calls += 1
        return np.zeros_like(state)

    def rollout(self, q, v, z, clock, power, equilibrium, *, max_step):
        state = np.r_[q, v, z].astype(float)
        output = [state.copy()]
        for interval in np.diff(clock):
            updates = max(1, math.ceil(interval / max_step - 1e-12))
            h = interval / updates
            for _ in range(updates):
                k1 = self.rhs(state, power, equilibrium)
                k2 = self.rhs(state + h*k1/2, power, equilibrium)
                k3 = self.rhs(state + h*k2/2, power, equilibrium)
                k4 = self.rhs(state + h*k3, power, equilibrium)
                state = state + h*(k1 + 2*k2 + 2*k3 + k4)/6
            output.append(state.copy())
        return np.asarray(output)

    def energy_identity(self, _state, _power, _equilibrium):
        self.physical_calls += 1
        return {"identity_error": 1.0 if self.fail_physics else 0.0}


def _run_path(monkeypatch, *, model=None, ops=None, clock=None, events=None):
    model = FakeLinearModel() if model is None else model
    ops = FakeOperations() if ops is None else ops
    clock = np.array([0.0, 0.01, 0.02]) if clock is None else clock
    events = [] if events is None else events
    monkeypatch.setattr(
        transfer, "save_record",
        lambda _directory, name, record: events.append(("save", name, record["path"].shape)),
    )
    monkeypatch.setattr(
        transfer, "write_json",
        lambda _path, value: events.append(("physics_json", value["qualified"])),
    )
    before_rhs = model.rhs.__func__
    path = transfer.audited_path(
        model=model,
        initial=np.zeros(22),
        clock=clock,
        power=np.zeros(10),
        equilibrium=np.zeros(10),
        step=0.005,
        ops=ops,
        identity={"seed": 324, "event_id": "fake-event", "step": 0.005,
                  "phase": "history_integration"},
        directory=Path("unused-artifact-directory"),
        name="fake-history",
    )
    assert model.rhs.__func__ is before_rhs
    assert ops.rhs_context == {}
    return path, model, ops, events


def test_path_difference_reports_visible_normalized_and_hidden_norms():
    fine = np.zeros((3, 22))
    fine[:, 20] = 0.2
    coarse = fine.copy()
    coarse[1, 0] += 5e-7
    coarse[2, 20] += 2e-6

    result = transfer.path_difference(coarse, fine, np.full(20, 0.1))

    assert result["visible_max_normalized"] == pytest.approx(5e-6)
    assert result["hidden_max_norm"] == pytest.approx(2e-6)
    assert result["hidden_bound"] == pytest.approx(1e-5 * 1.2)
    assert result["qualified"] is True


@pytest.mark.parametrize(
    ("coarse", "fine"),
    [
        (np.zeros((2, 22)), np.zeros((3, 22))),
        (np.full((2, 22), np.nan), np.zeros((2, 22))),
    ],
)
def test_path_difference_rejects_unaligned_or_nonfinite_paths(coarse, fine):
    with pytest.raises(ValueError, match="finite and aligned"):
        transfer.path_difference(coarse, fine, np.ones(20))


def test_audited_path_counts_each_rk4_rhs_and_each_physical_check(monkeypatch):
    events = []
    path, model, ops, _events = _run_path(monkeypatch, events=events)

    assert path.shape == (3, 22)
    assert ops.counts == {"rollouts": 1, "integration_rhs": 16, "physical_rhs": 3}
    assert ops.returns == ops.counts
    assert model.rhs_calls == 16
    assert model.physical_calls == 3
    assert len(ops.partial) == 3
    assert all(value["phase"] == "physical_checks" for value in ops.partial.values())
    assert events[0][:2] == ("save", "fake-history")
    assert events[-1] == ("physics_json", True)
    transitions = ops.journal.entries
    integration = [row for row in transitions if row.get("phase") == "history_integration"]
    physical = [row for row in transitions if row.get("phase") == "physical_checks"]
    assert len(integration) == 32
    assert len(physical) == 6
    assert all(row["category"] == "rhs" for row in (*integration, *physical))
    assert [row["transition"] for row in integration[:2]] == ["attempted", "returned"]


def test_audited_path_saves_output_before_reporting_physical_failure(monkeypatch):
    events = []
    model = FakeLinearModel(fail_physics=True)

    with pytest.raises(FloatingPointError, match="candidate physical checks failed"):
        _run_path(monkeypatch, model=model, events=events)

    assert events[0][:2] == ("save", "fake-history")
    assert events[-1] == ("physics_json", False)
    assert model.rhs_calls == 16
    assert model.physical_calls == 3


@pytest.mark.parametrize(
    ("integration_limit", "physical_limit", "expected_rhs", "expected_physical"),
    [(15, 3, 15, 0), (16, 0, 16, 0)],
)
def test_rhs_budget_exhaustion_refuses_callback_and_restores_rhs(
        monkeypatch, integration_limit, physical_limit, expected_rhs, expected_physical):
    events = []
    model = FakeLinearModel()
    ops = FakeOperations(integration_limit=integration_limit, physical_limit=physical_limit)
    original_rhs = model.rhs.__func__
    monkeypatch.setattr(
        transfer, "save_record",
        lambda _directory, name, record: events.append(("save", name, record["path"].shape)),
    )
    monkeypatch.setattr(
        transfer, "write_json",
        lambda _path, value: events.append(("physics_json", value["qualified"])),
    )

    with pytest.raises(RuntimeError, match="unreserved or exhausted RHS call"):
        transfer.audited_path(
            model, np.zeros(22), np.array([0.0, 0.01, 0.02]), np.zeros(10), np.zeros(10),
            0.005, ops,
            {"seed": 324, "event_id": "fake-event", "step": 0.005,
             "phase": "history_integration"},
            Path("unused-artifact-directory"), "budget-limited",
        )

    assert model.rhs.__func__ is original_rhs
    assert model.rhs_calls == expected_rhs
    assert model.physical_calls == expected_physical
    assert ops.counts["integration_rhs"] == expected_rhs
    assert ops.counts["physical_rhs"] == expected_physical
    assert ops.rhs_context == {}
    if integration_limit < 16:
        assert events == []
        assert len([row for row in ops.journal.entries
                    if row.get("phase") == "history_integration"]) == 2 * expected_rhs
    else:
        assert events[0][0] == "save"
        assert all(row.get("phase") != "physical_checks" for row in ops.journal.entries)


def test_invalid_clock_is_rejected_before_monkeypatch_or_callbacks(monkeypatch):
    model = FakeLinearModel()
    ops = FakeOperations()
    original_rhs = model.rhs.__func__
    saves = []
    monkeypatch.setattr(transfer, "save_record", lambda *_args, **_kwargs: saves.append(True))

    with pytest.raises(ValueError, match="registered substep counts"):
        transfer.audited_path(
            model, np.zeros(22), np.array([0.0, 0.009, 0.02]), np.zeros(10), np.zeros(10),
            0.005, ops,
            {"seed": 324, "event_id": "fake-event", "step": 0.005,
             "phase": "history_integration"},
            "unused-artifact-directory", "invalid-clock",
        )

    assert model.rhs.__func__ is original_rhs
    assert model.rhs_calls == model.physical_calls == 0
    assert ops.counts == {"rollouts": 0, "integration_rhs": 0, "physical_rhs": 0}
    assert ops.journal.entries == []
    assert saves == []


class ExecuteOperations:
    def __init__(self, probe_limit):
        self.scope_id = None
        self.limits = {"probe": probe_limit, "rollouts": 0,
                       "integration_rhs": 0, "physical_rhs": 0}
        self.counts = dict.fromkeys(self.limits, 0)
        self.returns = dict.fromkeys(self.limits, 0)
        self.partial = {}
        self.last_rhs = {}
        self.rhs_context = {}
        self.events = []
        self.tags = []
        self.journal = FakeJournal()

    @contextmanager
    def scope(self, name):
        previous = self.scope_id
        self.scope_id = name
        self.events.append(("scope", name))
        try:
            yield self
        finally:
            self.scope_id = previous

    def call(self, category, function, *, tag=""):
        if self.counts[category] >= self.limits[category]:
            raise RuntimeError(f"budget exhausted: {category}")
        self.counts[category] += 1
        self.tags.append(tag)
        value = function()
        self.returns[category] += 1
        return value

    def close(self):
        self.events.append(("journal-close",))


class ExecuteReader:
    def __init__(self, root, source_plan, operations, *, rows, future_counts):
        self.operations = operations
        self.rows = rows
        self.future_counts = future_counts
        self.files = {}
        self.attempted = set()
        self.decoded = set()
        self.bytes_read = 0
        self.npy_header_bytes = 0
        self.logical_bytes = 0
        self.events = operations.events
        self.arrays_by_path = {}
        self._inputs = {
            "history": np.zeros((rows, 11, 20)),
            "history_time": np.tile(np.arange(11) * 0.01, (rows, 1)),
            "constant_power": np.zeros((rows, 10)),
            "equilibrium": np.zeros((rows, 10)),
            "mass": np.ones(10),
            "target_scale": np.ones(20),
        }
        self._static = {"parameters__" + name: np.ones(10) for name in transfer.PARAMETERS}
        self._integral_arrays = {
            "Ltilde": np.eye(2), "btilde4": np.zeros(2), "zh4": np.zeros(2),
            "svd__U": np.eye(2), "svd__s": np.ones(2), "svd__Vh": np.eye(2),
        }
        self._history_clock = np.arange(11) * 0.01
        self._add_file("results/run-a14/artifact_hashes.json", "json")
        self._add_file("results/run-a15/artifact_hashes.json", "json")
        for run in ("run-a14", "run-a15"):
            self._add_file(f"results/{run}/allocated_population.json", "json")
        self._add_pair("results/run-a14/permitted_inputs", {
            "history": 1, "history_time": 1, "constant_power": 1,
            "equilibrium": 1, "mass": 1, "target_scale": 1,
        })
        self._add_pair("results/run-a14/static-324", {
            name: 1 for name in self._static
        })
        for index in range(rows):
            key = f"seed-324-event-{index:03d}"
            self._add_pair(f"results/run-a14/{key}/integral", {
                name: 1 for name in self._integral_arrays
            })
            self._add_file(f"results/run-a14/{key}/slot.json", "json")
            for step in transfer.STEPS:
                self._add_pair(f"results/run-a14/{key}/history-{step}", {
                    name: 1 for name in
                    ("observed", "clock", "baseline", "baseline_reset", "candidate", "reset")
                })
        self._add_pair("results/run-a15/permitted_truth", {
            "offsets": 1, "forecast_time": 1, "omega_s": 1,
        })
        for index in range(rows):
            key = f"seed-324-event-{index:03d}"
            for step in transfer.STEPS:
                self._add_pair(f"results/run-a15/{key}/future-{step}", {
                    name: 1 for name in ("clock", "reset", "truth", "baseline", "candidate")
                })
        self.source_hashes = {}
        for run in ("run-a14", "run-a15"):
            prefix = f"results/{run}/"
            self.source_hashes[run] = {
                path.removeprefix(prefix): spec["sha256"]
                for path, spec in self.files.items()
                if path.startswith(prefix) and not path.endswith("/artifact_hashes.json")
            }
        self.population = {
            "cache_rows": [{"event_id": f"event-{i:03d}"} for i in range(rows)],
            "allocated_slots": list(range(268)),
            "weights": [1.0 / rows] * rows,
            "future_point_counts": list(future_counts),
        }
        del root, source_plan

    def _add_file(self, relative, kind, members=None):
        digest = __import__("hashlib").sha256(relative.encode()).hexdigest()
        self.files[relative] = {
            "kind": kind, "size_bytes": 1, "sha256": digest,
            "members": members or {},
        }

    def _add_pair(self, stem, members):
        self._add_file(stem + ".json", "json")
        self._add_file(stem + ".npz", "npz", {
            name: {"shape": [1], "dtype": "<f8"} for name in members
        })

    def _consume_file(self, relative):
        if relative in self.attempted:
            raise AssertionError(f"duplicate synthetic source read: {relative}")
        self.attempted.add(relative)
        self.bytes_read += self.files[relative]["size_bytes"]
        self.operations.call("probe", lambda: None, tag="source_bytes:" + relative)

    def read_json(self, relative):
        self._consume_file(relative)
        if relative.endswith("/artifact_hashes.json"):
            run = "run-a14" if "/run-a14/" in relative else "run-a15"
            return self.source_hashes[run]
        if relative.endswith("/allocated_population.json"):
            return self.population
        if relative.endswith("/permitted_inputs.json"):
            return {}
        if relative.endswith("/static-324.json"):
            return {"parameters": {"backbone": "sine"}}
        if relative.endswith("/integral.json"):
            tol = 64 * np.finfo(float).eps * 100
            return {"svd": {"rank": 2, "tol": tol, "natural_norm": 1.0}}
        if relative.endswith("/slot.json"):
            event_key = relative.rsplit("/", 1)[0]
            index = int(event_key.rsplit("event-", 1)[1])
            return {"seed": 324, "cache_index": index, "event_id": f"event-{index:03d}",
                    "hidden_energy_encoder": 0.5}
        if relative.endswith("/permitted_truth.json"):
            return {}
        return {}

    def read_arrays(self, relative):
        self._consume_file(relative)
        stem = relative.removesuffix(".npz")
        if stem.endswith("/permitted_inputs"):
            arrays = self._inputs
        elif stem.endswith("/static-324"):
            arrays = self._static
        elif stem.endswith("/integral"):
            arrays = self._integral_arrays
        elif stem.endswith("/permitted_truth"):
            offsets = np.r_[0, np.cumsum(np.asarray(self.future_counts) + 1)]
            clocks = [np.arange(int(n) + 1) * 0.01 for n in self.future_counts]
            arrays = {
                "offsets": offsets,
                "forecast_time": np.concatenate(clocks),
                "omega_s": np.zeros(self.rows),
            }
            self.events.append(("clock-read", relative))
        elif "/history-" in stem:
            key, step = stem.rsplit("/history-", 1)
            index = int(key.rsplit("event-", 1)[1])
            observed = self._inputs["history"][index]
            baseline = np.zeros((11, 22))
            baseline[:, 20] = 1.0
            candidate = np.zeros((11, 22))
            arrays = {
                "observed": observed, "clock": self._history_clock,
                "baseline": baseline,
                "baseline_reset": np.r_[observed[-1], [1.0, 0.0]],
                "candidate": candidate,
                "reset": np.r_[observed[-1], [0.0, 0.0]],
            }
            del step
        elif "/future-" in stem:
            key, _step = stem.rsplit("/future-", 1)
            index = int(key.rsplit("event-", 1)[1])
            count = self.future_counts[index]
            clock = np.arange(count + 1) * 0.01
            truth = np.zeros((count + 1, 20))
            baseline = np.zeros((count + 1, 22))
            baseline[:, 20] = 1.0
            candidate = np.zeros((count + 1, 22))
            arrays = {
                "clock": clock, "reset": np.zeros(22), "truth": truth,
                "baseline": baseline, "candidate": candidate,
            }
            self.events.append(("source-future-read", relative))
        else:
            raise AssertionError(f"unexpected synthetic array source: {relative}")
        expected = set(self.files[relative]["members"])
        assert set(arrays) == expected
        for name, value in arrays.items():
            self.operations.call("probe", lambda: None, tag=f"source_member:{relative}:{name}")
            self.decoded.add(relative + ":" + name)
            self.npy_header_bytes += 128
            self.logical_bytes += np.asarray(value).nbytes
        self.arrays_by_path[relative] = arrays
        return arrays

    def snapshot(self):
        return {"files_attempted": len(self.attempted), "members_attempted": len(self.decoded),
                "bytes_read": self.bytes_read, "npy_header_bytes": self.npy_header_bytes,
                "logical_bytes": self.logical_bytes}


class ExecuteModel:
    def __init__(self, **parameters):
        self.mass = np.ones(10)
        for name in transfer.PARAMETERS:
            setattr(self, name, parameters[name])
        self.backbone = "sine"

    def energy(self, state, _equilibrium):
        return 0.5 * float(state[20:] @ state[20:])


def test_execute_inactive_slots_freeze_before_reading_future_source_and_never_roll_out(
        tmp_path, monkeypatch):
    rows = 258
    future_counts = [300] * 237 + [50] * 20 + [47]
    assert sum(future_counts) == 72147
    source_plan = {"sources": {"a14": {"run_id": "run-a14"}, "a15": {"run_id": "run-a15"}}}
    source_plan_path = tmp_path / "sources.json"
    source_plan_path.write_text(json.dumps(source_plan), encoding="utf-8")
    reader_holder = {}
    event_log = []

    def make_reader(root, plan, operations):
        reader = ExecuteReader(root, plan, operations, rows=rows, future_counts=future_counts)
        reader_holder["reader"] = reader
        return reader

    monkeypatch.setattr(transfer, "SEEDS", (324,))
    monkeypatch.setattr(transfer, "EnergyCapSavedInputs", make_reader)
    def make_operations(_directory, plan):
        operations = ExecuteOperations(plan["operator_limits"]["probe"])
        operations.events = event_log
        ops_holder["ops"] = operations
        return operations

    monkeypatch.setattr(transfer, "Operations", make_operations)
    monkeypatch.setattr(transfer, "NonlinearPassiveClosure", lambda **kwargs: ExecuteModel(**kwargs))
    monkeypatch.setattr(transfer, "audited_path", lambda *_args, **_kwargs:
                        pytest.fail("inactive candidates must reuse saved paths"))
    monkeypatch.setattr(transfer, "save_record", lambda *_args, **_kwargs: None)

    def freeze(_directory, name, _records):
        event_log.append(("freeze", name))
        return "frozen-" + name

    monkeypatch.setattr(transfer, "_freeze", freeze)
    monkeypatch.setattr(transfer, "write_json", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(transfer, "git_head", lambda _root: "artificial-commit")
    monkeypatch.setattr(transfer, "score_history", lambda *_args, **_kwargs: {
        "all": 0.0, "q": 0.0, "v": 0.0,
    })
    score_calls = []
    ops_holder = {}

    def fake_score_pair(**kwargs):
        score_calls.append((ops_holder["ops"].tags[-1], kwargs))
        return {
            "refinement": {"qualified": True},
            "steps": {step: {"all": 0.0} for step in transfer.STEPS},
        }

    monkeypatch.setattr(transfer, "score_pair", fake_score_pair)
    monkeypatch.setattr(transfer, "aggregate_group", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(transfer, "verdict", lambda _groups, **_kwargs: {
        "scientific_verdict": "MOCK_PASS",
    })

    # The exact synthetic reader and direct-probe calls form this frozen small fixture cap.
    member_count = 6 + 6 + rows * 6 + rows * 2 * 6 + 3 + rows * 2 * 5
    file_count = 8 + rows * 7 + 2 + rows * 4
    direct_probe_count = len((324,)) + rows * 9 + 3
    probe_limit = member_count + file_count + direct_probe_count
    future_points = sum(count + 1 for count in future_counts)
    input_bytes = (rows * 11 * 20 + rows * 11 + rows * 10 * 2 + 10 + 20) * 8
    static_bytes = len(transfer.PARAMETERS) * 10 * 8
    integral_bytes = rows * 18 * 8
    history_bytes = rows * 2 * 759 * 8
    clock_bytes = (259 + future_points + rows) * 8
    future_bytes = sum(2 * (65 * (count + 1) + 22) * 8 for count in future_counts)
    logical_byte_cap = input_bytes + static_bytes + integral_bytes + history_bytes + clock_bytes + future_bytes
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    plan = {
        "sources_path": "sources.json",
        "future_counts": future_counts,
        "operator_limits": {"probe": probe_limit},
        "source_read_plan": {
            "decoded_array_byte_cap": logical_byte_cap,
            "npy_header_max_bytes_per_member": 4108,
        },
    }

    summary = transfer.execute(root=tmp_path, result_dir=result_dir, plan=plan, runtime={"fake": "test"})

    reader = reader_holder["reader"]
    ops = ops_holder["ops"]
    assert len(summary["slots"]) == rows
    assert all(slot["status"] == "complete" and slot["solver_status"] == "inactive"
               for slot in summary["slots"])
    assert ops.counts == {"probe": probe_limit, "rollouts": 0,
                          "integration_rhs": 0, "physical_rhs": 0}
    assert ops.returns == ops.counts
    assert reader.attempted == set(reader.files)
    assert len(reader.decoded) == sum(len(spec["members"]) for spec in reader.files.values())
    assert reader.bytes_read == sum(spec["size_bytes"] for spec in reader.files.values())
    assert reader.logical_bytes == logical_byte_cap
    assert reader.npy_header_bytes <= len(reader.decoded) * 4108
    assert len(score_calls) == rows * 3
    assert summary["conditional_budget"]["expected_calls"] == {
        "rollouts": 0, "integration_rhs": 0, "physical_rhs": 0,
    }

    for index in range(rows):
        candidate_tag, candidate_score = score_calls[index * 3 + 2]
        assert candidate_tag == "paired_scores:C"
        key = f"seed-324-event-{index:03d}"
        for step, path_name in zip(transfer.STEPS, ("coarse", "fine"), strict=True):
            relative = f"results/run-a15/{key}/future-{step}.npz"
            assert candidate_score[path_name] is reader.arrays_by_path[relative]["candidate"]

    freeze_init = event_log.index(("freeze", "initialization-freeze.json"))
    clock_read = next(i for i, event in enumerate(event_log) if event[0] == "clock-read")
    freeze_future = event_log.index(("freeze", "future-freeze.json"))
    first_future = next(i for i, event in enumerate(event_log) if event[0] == "source-future-read")
    assert freeze_init < clock_read < freeze_future < first_future


def test_registered_operations_reserve_energy_cap_scope_names(tmp_path):
    from kron_reduce.evaluation.constant_c_observability import Operations as RealOperations

    names = ["source-root", "init-324-000", "history-324-000-h005",
             "history-324-000-h0025", "history-review-324-000", "clock-inputs",
             "future-324-000-h005", "future-324-000-h0025", "score-324-000", "summary"]
    zero_budget = {key: 0 for key in (
        "expm", "svd", "force", "ls", "features", "encoder", "model_loads",
        "model_constructions", "rollouts", "integration_rhs", "physical_rhs", "probe",
    )}
    plan = {
        "operator_limits": dict(zero_budget),
        "scope_limits": {name: dict(zero_budget) for name in names},
    }
    operations = RealOperations(tmp_path / "operations", plan)

    for name in names:
        with operations.scope(name):
            assert operations.scope_id == name

    assert operations.scope_id is None
    assert len(operations.journal.closed_scopes) == len(names)
    operations.journal.close()
