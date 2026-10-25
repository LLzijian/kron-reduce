"""Actual three-phase orchestration using synthetic arrays and a staged provider.

The production control validator remains strict. Only this private Campaign
fixture substitutes a small source population/provider, never the public CLI.
"""

from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from kron_reduce.data.durable_audit_ledger import AppendOnlyAuditJournal
from kron_reduce.evaluation.input_consistency_paths import path_success_caps
from kron_reduce.workflows import kiki_input_consistency as workflow


def synthetic_campaign(tmp_path, monkeypatch, fail=False, *, make=True):
    production = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs/proposals/kiki_a17_input_consistency_compute.json"
        ).read_text(encoding="utf-8")
    )
    scopes = {}
    for name in production["scope_order"]:
        row = copy.deepcopy(production["scopes"][name])
        if "cache_index" in row["identity"] and row["identity"]["cache_index"] != 0:
            continue
        row["prerequisite_scope_ids"] = [
            key for key in row["prerequisite_scope_ids"] if key in scopes
        ]
        if name.startswith("future-"):
            multiplier = 2 if name.endswith("h005") else 4
            row["operators"] = path_success_caps(
                point_count=4, rk4_updates=3 * multiplier, linear=False
            )
            row["derived_limits"] = {"successful_rk4_updates": 3 * multiplier}
        if name.startswith(("frefine-", "refine-")):
            row["diagnostic_operation_limits"]["path_refinement"]["vector_norms"] = 8
        if name == "references":
            row["derived_limits"] = {
                "reference_ancestry_logical_bytes": 6 * 4 * 22 * 8 + 4 * 20 * 8 + 4 * 8 + 2 * 8
            }
        scopes[name] = row
    compute = {"scope_order": list(scopes), "scopes": scopes}
    clock = np.linspace(0, 0.1, 11)
    U = np.zeros((100, 2))
    U[0, 0] = U[1, 1] = 1
    singular = np.array([3.0, 2.0])
    integral = {
        "Ltilde": U * singular,
        "U": U,
        "s": singular,
        "Vh": np.eye(2),
        "btilde4": np.zeros(100),
        "c": np.ones(10),
        "S": np.ones(10),
        "h": np.diff(clock),
        "time": clock,
        "zh2": np.zeros(2),
        "zh4": np.zeros(2),
        "zTobs2": np.zeros(2),
        "zTobs4": np.zeros(2),
        "PhiT": np.eye(2),
        "wT": np.zeros(2),
    }
    inputs = {
        "history": np.zeros((1, 11, 20)),
        "history_time": clock[None, :],
        "constant_power": np.zeros((1, 10)),
        "equilibrium": np.zeros((1, 10)),
        "mass": np.linspace(1, 2, 10),
        "target_scale": np.ones(20),
    }
    parameters = {
        "mass": inputs["mass"],
        "edge_stiffness": np.ones(45) * 0.01,
        "damping": np.zeros((10, 10)),
        "coupling": np.zeros((10, 2)),
        "latent_skew": np.zeros((2, 2)),
        "latent_dissipation": np.eye(2) * 0.1,
    }
    arrays, contexts, slots = {"inputs": inputs}, [{"context_id": "inputs", "phase": 1}], {}
    future_clock = np.array([0.0, 0.01, 0.02, 0.03])
    for seed in workflow.SEEDS:
        arrays[f"static-{seed}"] = {"parameters__" + k: v.copy() for k, v in parameters.items()}
        arrays[f"integral-{seed}-000"] = {k: v.copy() for k, v in integral.items()}
        arrays[f"weighted-{seed}-000"] = {"root__values__weighted_impulse": np.zeros((100, 3))}
        for identity in (f"static-{seed}", f"integral-{seed}-000", f"weighted-{seed}-000"):
            contexts.append({"context_id": identity, "phase": 1})
        threshold = float(64 * np.finfo(float).eps * 100 * 3)
        slots[f"{seed}-000"] = {
            "integral_parameters": {
                "rank": 2,
                "threshold": threshold,
                "tol": threshold,
                "natural_norm": 1.0,
            },
            "determinant": {"lower": 36.0},
            "trace": {"upper": 13.0},
        }
        for code, _ in workflow.STEPS:
            identity = f"reference-{seed}-000-{code}"
            arrays[identity] = {
                "candidate": np.zeros((4, 22)),
                "baseline": np.zeros((4, 22)),
                "reset": np.zeros(22),
            }
            if seed == 324 and code == "h005":
                arrays[identity]["truth"] = np.zeros((4, 20))
            contexts.append({"context_id": identity, "phase": 3})
    arrays["impulse-000"] = {f"root__u11__{i}": np.zeros((10, 3)) for i in range(11)}
    arrays["clock-000"] = {"clock": future_clock}
    contexts.extend(
        [{"context_id": "impulse-000", "phase": 1}, {"context_id": "clock-000", "phase": 2}]
    )

    def fingerprint(value):
        return {
            "shape": list(value.shape),
            "dtype": value.dtype.str,
            "sha256_c_order_bytes": workflow.array_sha(value),
        }

    offsets = fingerprint(np.array([0, 4], dtype=np.int64))
    source = {
        "slot_bindings": slots,
        "ancestry_fingerprints": {
            "E": {
                f"{seed}-{code}": {"future": fingerprint(np.zeros((4, 22))), "offsets": offsets}
                for seed in workflow.SEEDS
                for code, _ in workflow.STEPS
            },
            "truth": {
                "A10/real_train_arrays.npz": {
                    "members": {
                        "future": fingerprint(np.zeros((4, 20))),
                        "forecast_time": fingerprint(future_clock),
                        "offsets": offsets,
                    }
                }
            },
        },
    }
    pop = {
        "cache_rows": [{"cache_index": 0, "event_id": "synthetic/event"}],
        "weights": [1.0],
        "future_point_counts": [3],
        "allocated_slots": ["synthetic/event", "synthetic/uncovered"],
    }
    plan = {
        "contexts": contexts,
        "population": pop,
        "freeze_contract": {"population_sha256": "a" * 64},
    }

    class Provider:
        def __init__(self, root, plan, directory):
            self.phase, self.used, self.closed, self.failed = 1, [], False, False
            self.directory = Path(directory)
            self.journal = AppendOnlyAuditJournal(self.directory / "source_access.jsonl")

        def _observed(self, operation, identity, callback):
            details = {"operation": operation, "identity": identity, "phase": self.phase}
            self.journal.append({**details, "transition": "attempted"})
            try:
                value = callback()
            except BaseException as error:
                self.journal.append(
                    {**details, "transition": "failed", "error_type": type(error).__name__}
                )
                self.failed = True
                raise
            self.journal.append({**details, "transition": "returned"})
            return value

        def read_metadata(self):
            return self._observed("read_metadata", "synthetic-metadata", dict)

        def metadata_envelopes(self):
            return self._observed("metadata_envelopes", "synthetic-metadata", dict)

        def integrity(self, position):
            def check():
                assert position == "before" or self.phase == 3

            return self._observed("integrity", position, check)

        def read_context(self, identity):
            def read():
                assert identity not in self.used
                assert (
                    next(row["phase"] for row in contexts if row["context_id"] == identity)
                    == self.phase
                )
                self.used.append(identity)
                return arrays[identity]

            return self._observed("read_context", identity, read)

        def advance(self, phase, path, digest, size):
            def check():
                marker = json.loads(path.read_text(encoding="utf-8"))
                assert phase == self.phase + 1
                assert digest == workflow.sha256_file(path) and size == path.stat().st_size
                assert len(marker["records"]) == 6
                expected = sum(row["phase"] < phase for row in scopes.values())
                assert marker["scientific_prefix"]["scope_count"] == expected
                self.phase = phase

            return self._observed("advance", str(path.name), check)

        def snapshot(self):
            return {
                "phase": self.phase,
                "partial_record": {"arrays": np.zeros(2)},
                "used": self.used,
            }

        def finish(self):
            def finalize():
                assert len(self.used) == len(contexts)
                self.closed = True
                return self.snapshot()

            try:
                result = self._observed("finish", "reader", finalize)
                self.journal.close()
                return result
            except BaseException:
                self.close()
                raise

        def close(self):
            if not self.journal._closed:
                self.journal.close()
            self.closed = True

    monkeypatch.setattr(workflow, "InputConsistencyReader", Provider)
    monkeypatch.setattr(workflow, "verify_source_documents", lambda plan, documents: source)
    receipt = {
        "source_commit": "synthetic",
        "run_id": tmp_path.name,
        "authorization": workflow.AUTH,
    }
    if not make:
        return plan, compute, receipt
    campaign = workflow.Campaign(tmp_path, tmp_path, plan, compute, receipt)
    if fail:
        actual = workflow.estimate_initial
        used = False

        def fail_once(*args, **kwargs):
            nonlocal used
            if not used:
                used = True
                raise FloatingPointError("synthetic first initializer failure")
            return actual(*args, **kwargs)

        monkeypatch.setattr(workflow, "estimate_initial", fail_once)
    return campaign


@pytest.mark.parametrize("fail", [False, True])
def test_three_phase_actual_controller_retains_all_scopes_and_failures(tmp_path, monkeypatch, fail):
    campaign = synthetic_campaign(tmp_path, monkeypatch, fail)
    summary = campaign.run()
    assert summary["V_pass"] is (not fail)
    assert summary["scientific_verdict"] == "REJECT"
    assert summary["promotion_decision"] == "KEEP_BASELINE"
    assert not summary["new_missed_instability_result_established"]
    assert campaign.reader.closed and campaign.reader.phase == 3
    assert list(campaign.store.records) == campaign.compute["scope_order"]
    assert len(campaign.pairs) == 12 and len(campaign.groups) == 18
    if fail:
        assert campaign.store.records["init-324-000"]["status"] == "numerical_failure"
        assert campaign.store.records["future-324-000-h005"]["status"] == "prerequisite_failed"
        assert campaign.store.records["future-324-000-h005"]["actual_counters"]["attempted"] == {}
        assert campaign.groups[("C", 324, "h005")]["missing_weight"] == 1.0
        assert not summary["G_I"] and not summary["G_E"]
    else:
        for name, row in campaign.store.records.items():
            counts = Counter(row["actual_counters"]["returned"])
            assert counts == Counter(campaign.compute["scopes"][name]["operators"])
    assert (tmp_path / "access_ledger.npz").is_file()


def test_scope_store_dependency_failure_is_never_recomputed(tmp_path):
    scopes = {
        "a": {
            "operators": {},
            "prerequisite_scope_ids": [],
            "derived_limits": {},
            "diagnostic_operation_limits": {},
        },
        "b": {
            "operators": {},
            "prerequisite_scope_ids": ["a"],
            "derived_limits": {},
            "diagnostic_operation_limits": {},
        },
    }
    store = workflow.ScopeStore(tmp_path, {"scope_order": ["a", "b"], "scopes": scopes})
    store.execute("a", lambda: (_ for _ in ()).throw(FloatingPointError("known failure")))
    store.execute("b", lambda: pytest.fail("a failed prerequisite must prevent this callback"))
    assert store.records["b"]["status"] == "prerequisite_failed"
    with pytest.raises(RuntimeError):
        store.execute("b", dict)
    assert store.journal.poisoned
    store.journal.close()
