"""Dedicated A17 campaign. Inactive until exact implementation approval.

Real source access is confined to its finite saved-array reader. The public
CLI never offers a smaller population, retry, resume, or alternate initializer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tomllib
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import numpy as np
from filelock import FileLock

from kron_reduce.data.input_consistency_reader import InputConsistencyReader
from kron_reduce.evaluation.input_consistency_checks import check_refinement
from kron_reduce.evaluation.input_consistency_evidence import (
    error_record,
    save_record,
    sha256_file,
    write_json,
)
from kron_reduce.evaluation.input_consistency_journal import (
    LOCAL_NUMERICAL_ERRORS,
    InputConsistencyJournal,
)
from kron_reduce.evaluation.input_consistency_metrics import (
    campaign_verdict,
    future_event_scores,
    paired_group,
    population_scores,
)
from kron_reduce.evaluation.input_consistency_paths import (
    estimate_initial,
    evaluate_fixed_path,
    reset_path_origin,
)
from kron_reduce.evaluation.input_consistency_plan import validate_production_plan
from kron_reduce.evaluation.input_consistency_seal import SealLedger
from kron_reduce.evaluation.input_consistency_sources import verify_source_documents
from kron_reduce.models.nonlinear_latent import NonlinearPassiveClosure
from project_harness.git_state import assert_clean, git_head, verify_bound_file
from project_harness.manifest import load_manifest
from project_harness.policy import load_policy, validate_run_policy
from project_harness.runner import _run_experiment, _verify_development_text_bindings

AUTH = "kiki_a17_development_20261025"
EXPERIMENT = "paper_kiki_a17_input_consistency"
WORKFLOW = "kiki_a17_input_consistency_v1"
STAGE = "input_consistency"
MODULE = "kron_reduce.workflows.kiki_input_consistency"
MODE = "bounded_history_cache_audit_cli_only"
CONTRACT = "measured_history_pm11_linear_100ms_zoh_v1"
SEEDS = (324, 325, 326)
STEPS = (("h005", .005), ("h0025", .0025))
PARAMETERS = ("mass", "edge_stiffness", "damping", "coupling", "latent_skew", "latent_dissipation")
BOUNDARY = ("Uncertified original-train input-consistency comparison. No labels, instability "
            "claim, deployment, certification, model promotion, new training or automatic continuation.")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def closed_chain_sha(rows):
    digest = hashlib.sha256()
    for row in rows:
        digest.update((json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                                  allow_nan=False) + "\n").encode("utf-8"))
    return digest.hexdigest()


def authorization(root, experiment, config):
    validate_run_policy(root, EXPERIMENT, experiment)
    _verify_development_text_bindings(root, experiment)
    approval = load_policy(root).get("development_evaluations", {}).get(AUTH, {})
    expected = {"status": "user_approved", "authority": "explicit_user_reply",
                "scientific_design_approver": "Astra Ultra", "experiment": EXPERIMENT,
                "workflow": WORKFLOW, "stage": STAGE, "module": MODULE,
                "dataset": "kiki_ieee39_v1", "max_campaign_executions": 1,
                "retry_allowed": False, "max_fit_attempts": 0, "max_optimizer_updates": 0,
                "max_rhs": 5831802, "model_promotion_allowed": False,
                "deployment_predictions_allowed": False, "labels_allowed": False,
                "internal_val_payload_allowed": False, "source_val_payload_allowed": False,
                "source_test_payload_allowed": False}
    if any(type(approval.get(key)) is not type(value) or approval.get(key) != value
           for key, value in expected.items()):
        raise PermissionError("A17 independent one-use authorization is inactive")
    if any(key not in approval or approval[key] != value for key, value in config.items()):
        raise PermissionError("A17 configuration differs from its reviewed independent scope")
    if (experiment != load_manifest(root)["experiments"].get(EXPERIMENT)
            or experiment.get("execution_mode") != MODE or config.get("input_contract") != CONTRACT):
        raise PermissionError("A17 registered execution identity differs")
    for path_key, sha_key in (("read_plan_path", "read_plan_sha256"),
                              ("compute_plan_path", "compute_plan_sha256")):
        verify_bound_file(root, config[path_key], config[sha_key])
    verify_bound_file(root, approval["scientific_review_path"], approval["scientific_review_sha256"])
    bindings = approval.get("implementation_bindings", {})
    if "src/" + MODULE.replace(".", "/") + ".py" not in bindings:
        raise PermissionError("A17 lacks exact reviewed implementation bindings")
    for relative, digest in bindings.items():
        verify_bound_file(root, relative, digest)
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("A17 requires its explicitly selected constrained Python 3.12 runtime")
    assert_clean(root)
    return approval


def bound_controls(root, config):
    plan = read_json(root / config["read_plan_path"])
    validation = validate_production_plan(plan)
    compute = read_json(root / config["compute_plan_path"])
    if (compute["authorization"] != AUTH or compute["bindings"]["read_plan"]["sha256"]
            != config["read_plan_sha256"] or len(compute["scopes"]) != 10356
            or compute["population"]["population_sha256"] != plan["freeze_contract"]["population_sha256"]):
        raise PermissionError("A17 read/compute/population controls differ")
    return plan, compute, validation


def reserve(root, run_id, config):
    common = Path(subprocess.check_output(["git", "-C", str(root), "rev-parse",
        "--path-format=absolute", "--git-common-dir"], text=True).strip()).resolve(strict=True)
    directory = common / "a17_input_consistency_budgets" / AUTH
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir(exist_ok=False)
    receipt = {"authorization": AUTH, "run_id": run_id, "source_commit": git_head(root),
               "spent": True, "retry_allowed": False, "read_plan_sha256": config["read_plan_sha256"],
               "compute_plan_sha256": config["compute_plan_sha256"]}
    write_json(directory / "reservation.json", receipt)
    return directory, receipt


class ScopeStore:
    """Visit the immutable scope graph once, including every failed dependency."""

    def __init__(self, directory, compute):
        self.directory = Path(directory)
        self.compute = compute
        self.journal = InputConsistencyJournal(self.directory / "operations")
        self.records = {}
        self.ordinal = 0

    def execute(self, identity, function):
        try:
            self.journal._check()
            return self._execute(identity, function)
        except BaseException:
            self.journal.poisoned = True
            raise

    def _execute(self, identity, function):
        if (self.ordinal >= len(self.compute["scope_order"])
                or self.compute["scope_order"][self.ordinal] != identity):
            self.journal.poisoned = True
            raise RuntimeError("A17 attempted an unregistered, repeated or reordered scope")
        row = self.compute["scopes"][identity]
        self.ordinal += 1
        self.journal.begin(identity, row["operators"])
        dependencies = row["prerequisite_scope_ids"]
        missing = [key for key in dependencies if key not in self.records]
        failed = [key for key in dependencies if self.records.get(key, {}).get("status") != "complete"]
        if missing:
            self.journal.poisoned = True
            raise RuntimeError("A17 scope dependency record is absent")
        if failed and row.get("dependency_mode", "all_success") == "all_success":
            status, value = "prerequisite_failed", {"failed_dependencies": failed, "retry_allowed": False}
        else:
            try:
                value = function()
                status = "complete"
            except LOCAL_NUMERICAL_ERRORS as error:
                if self.journal.poisoned:
                    raise
                self.journal.scope_failed = True
                value, status = error_record(error), "numerical_failure"
                self.journal.failure_partial = value
        accounting = scope_accounting(value)
        if status == "complete":
            validate_accounting(row, accounting)
        record = {"scope_id": identity, "status": status, "value": value,
                  **accounting,
                  "dependencies": dependencies, "failed_dependencies": failed,
                  "retry_allowed": False}
        artifact_hashes = {}

        def persist():
            artifact_hashes.update(save_record(self.directory, identity, record))
            return artifact_hashes

        closed = self.journal.persist_and_finish(persist, status=status,
            evidence={"failure_reason": None if status == "complete" else value.get("message", status),
                      "failed_dependencies": failed, **accounting})
        record.update(record_path=identity + ".json", record_sha256=artifact_hashes[identity + ".json"],
                      record_npz_sha256=artifact_hashes[identity + ".npz"],
                      actual_counters=closed["actual_counters"])
        self.records[identity] = record
        return value if status == "complete" else None

    def value(self, identity):
        row = self.records.get(identity)
        return row["value"] if row is not None and row["status"] == "complete" else None

    def complete(self, identity):
        return self.records.get(identity, {}).get("status") == "complete"


def array_sha(value):
    return hashlib.sha256(np.asarray(value).tobytes(order="C")).hexdigest()


def scope_accounting(value):
    """Only actual helper counters; never fill unused work from reservations."""
    if "input_consistency_campaign_partial" in value:
        value = value["input_consistency_campaign_partial"]
    derived = Counter(value.get("derived_actual", {}))
    diagnostics = {}
    for item in value.get("helpers", {}).values():
        derived.update(item.get("derived_actual", {}))
        for kind, counts in item.get("diagnostic_operation_counts", {}).items():
            if kind in diagnostics:
                raise RuntimeError("duplicate A17 diagnostic accounting")
            diagnostics[kind] = dict(counts)
        if "operation_counts" in item:
            kind = item["category"]
            if kind in diagnostics:
                raise RuntimeError("duplicate A17 refinement accounting")
            diagnostics[kind] = dict(item["operation_counts"])
    return {"derived_actual": dict(derived), "diagnostic_actual": diagnostics}


def validate_accounting(row, actual):
    expected = row["derived_limits"]
    diagnostics = row["diagnostic_operation_limits"]
    if actual["derived_actual"] != expected or set(actual["diagnostic_actual"]) != set(diagnostics):
        raise RuntimeError("A17 successful scope derived/diagnostic inventory differs")
    for kind, limits in diagnostics.items():
        expected_counts = {f"{name}_{state}": count for name, count in limits.items()
                           for state in ("attempted", "returned")}
        if actual["diagnostic_actual"][kind] != expected_counts:
            raise RuntimeError("A17 successful scope diagnostic operations differ")


def clock_checks(clock, count=None, *, history=False):
    clock = np.asarray(clock, dtype=np.float64)
    if (clock.ndim != 1 or len(clock) != (11 if history else count + 1)
            or not np.isfinite(clock).all() or clock[0] != 0 or np.any(np.diff(clock) <= 0)
            or (not history and clock[-1] > 3 + 1e-12)):
        raise PermissionError("A17 original clock identity/availability differs")
    steps = {code: int(sum(max(1, int(np.ceil(h / step - 1e-12))) for h in np.diff(clock)))
             for code, step in STEPS}
    expected = {"h005": 20 if history else 2 * count, "h0025": 40 if history else 4 * count}
    if steps != expected:
        raise PermissionError("A17 original clock cannot satisfy the frozen exact integration counts")
    return {"clock": clock, "updates": steps, "history": history}


class Campaign:
    """The registered three-phase execution; caller has already spent authority."""

    def __init__(self, root, directory, plan, compute, receipt):
        self.root, self.directory = Path(root), Path(directory)
        self.plan, self.compute, self.receipt = plan, compute, receipt
        self.store = ScopeStore(directory, compute)
        self.reader = InputConsistencyReader(root, plan, directory)
        self.cache, self.models, self.origins, self.futures, self.clocks = {}, {}, {}, {}, {}
        self.rows = plan["population"]["cache_rows"]
        self.n = len(self.rows)
        self.weights = np.array(plan["population"]["weights"], dtype=np.float64)
        self.counts = plan["population"]["future_point_counts"]
        self.scores = {(arm, seed, code): [None] * self.n
                       for arm in ("C", "I", "E") for seed in SEEDS for code, _ in STEPS}
        self.groups, self.pairs = {}, {}

    def load_phase(self, phase):
        completed = 0
        for row in self.plan["contexts"]:
            if row["phase"] == phase:
                self.cache[row["context_id"]] = self.reader.read_context(row["context_id"])
                completed += 1
                if completed % 256 == 0:
                    self.progress("source_contexts", phase=phase, returned_contexts=completed)

    def progress(self, action, **details):
        print(json.dumps({"authorization": AUTH, "action": action,
                          "closed_scopes": len(self.store.records), **details}), flush=True)

    def parameter_identity(self, seed):
        model = self.models[seed]
        source = self.cache[f"static-{seed}"]
        if not np.array_equal(model.mass, self.cache["inputs"]["mass"]):
            raise PermissionError("A17 fixed model and causal input mass differ")
        result = {}
        for name in PARAMETERS:
            value = getattr(model, name)
            expected = source["parameters__" + name]
            if value.flags.writeable or not np.array_equal(value, expected):
                raise PermissionError("A17 fixed physical parameters changed")
            result[name] = {"sha256": array_sha(value), "readonly": True, "shape": list(value.shape)}
        return {"parameters": result, "derived_actual": {
            "parameter_identity_values": sum(getattr(model, name).size for name in PARAMETERS),
            "parameter_identity_logical_bytes": sum(getattr(model, name).nbytes for name in PARAMETERS)}}

    def construct(self, seed):
        record = {"seed": seed, "eigvalsh": []}
        original = np.linalg.eigvalsh

        def eigvalsh(matrix):
            item = {"matrix": np.array(matrix, copy=True)}
            record["eigvalsh"].append(item)

            def calculate():
                value = original(matrix)
                item["returned"] = value
                return value
            return self.store.journal.call("structural_eigvalsh", calculate)

        def calculate():
            parameters = {name: self.cache[f"static-{seed}"]["parameters__" + name] for name in PARAMETERS}
            model = NonlinearPassiveClosure(**parameters, backbone="sine")
            record["model"] = model
            self.models[seed] = model
            return model

        np.linalg.eigvalsh = eigvalsh
        try:
            self.store.journal.call("closure_constructions", calculate)
        except BaseException as error:
            error.input_consistency_construction_partial = record
            raise
        finally:
            np.linalg.eigvalsh = original
        record["identity"] = self.store.journal.call("parameter_identity", lambda: self.parameter_identity(seed))
        record["derived_actual"] = record["identity"]["derived_actual"]
        return record

    def u11(self, index):
        values = self.cache[f"impulse-{index:03d}"]
        return np.stack([values[f"root__u11__{j}"][:, 0] for j in range(11)])

    def freeze(self, phase):
        kind = "a17_origins_freeze" if phase == 2 else "a17_futures_freeze"
        prefix = "hist" if phase == 2 else "future"
        records = []
        for seed in SEEDS:
            for index in range(self.n):
                for code, _ in STEPS:
                    row = self.store.records[f"{prefix}-{seed}-{index:03d}-{code}"]
                    records.append({"slot_id": f"{seed}-{index:03d}-{code}", "status": row["status"],
                                    "record_path": row["record_path"], "record_sha256": row["record_sha256"],
                                    "record_npz_sha256": row["record_npz_sha256"]})
        marker = {"authorization": AUTH, "kind": kind, "run_id": self.directory.name,
                  "source_commit": self.receipt["source_commit"],
                  "population_sha256": self.plan["freeze_contract"]["population_sha256"], "records": records,
                  "scientific_prefix": {
                      "scope_count": len(self.store.journal.closed_scopes),
                      "end_byte_offset": self.store.journal.closed_scopes[-1]["end_byte_offset"],
                      "closed_scope_chain_sha256": closed_chain_sha(self.store.journal.closed_scopes)}}
        path = self.directory / ("origins_freeze.json" if phase == 2 else "futures_freeze.json")
        write_json(path, marker)
        self.reader.advance(phase, path, sha256_file(path), path.stat().st_size)

    def capture(self, record, name, function):
        """Retain a returned helper, or its exact partial, before propagating failure."""
        try:
            value = function()
        except BaseException as error:
            partial = getattr(error, "input_consistency_paths_partial_record",
                              getattr(error, "partial_record", None))
            if partial is not None:
                record.setdefault("helpers", {})[name] = partial
            error.input_consistency_campaign_partial = record
            raise
        record.setdefault("helpers", {})[name] = value
        return value

    def initialize(self, seed, index):
        record = {"helpers": {}}
        inputs = self.cache["inputs"]
        integral = self.cache[f"integral-{seed}-{index:03d}"]
        slot = self.source["slot_bindings"][f"{seed}-{index:03d}"]
        original = inputs["history_time"][index]
        clock = original - original[0]

        def history_clock():
            result = clock_checks(clock, history=True)
            if (not np.array_equal(clock, integral["time"])
                    or not np.array_equal(np.diff(clock), integral["h"])
                    or abs(float(clock[-1]) - .1) > 1e-12):
                raise PermissionError("A17 saved history clocks differ")
            return result

        record["history_clock"] = self.store.journal.call("history_clock_checks", history_clock)
        power = self.u11(index)

        def anchor():
            delta = inputs["constant_power"][index] - power[-1]
            if not np.isfinite(delta).all() or np.max(np.abs(delta)) > 1e-12:
                raise PermissionError("A17 observed last input differs from its declared future anchor")
            return {"delta_u": delta, "bitwise_equal": bool(np.array_equal(
                inputs["constant_power"][index], power[-1])), "absolute_tolerance": 1e-12}

        record["input_anchor"] = self.store.journal.call("input_anchor_checks", anchor)
        parameters = slot["integral_parameters"]
        if parameters["threshold"] != parameters["tol"]:
            raise PermissionError("A17 original threshold/tolerance binding differs")
        self.capture(record, "initialization", lambda: estimate_initial(
            integral, self.cache[f"weighted-{seed}-{index:03d}"]["root__values__weighted_impulse"][:, 0],
            rank=parameters["rank"], threshold=parameters["threshold"], natural_norm=parameters["natural_norm"],
            determinant_lower=slot["determinant"]["lower"], trace_upper=slot["trace"]["upper"],
            dispatch=self.store.journal.call))
        return record

    def history(self, seed, index, code, step):
        record = {"helpers": {}}
        inputs = self.cache["inputs"]
        origin = inputs["history"][index, 0]
        initial = self.store.value(f"init-{seed}-{index:03d}")
        path = self.capture(record, "path_evaluation", lambda: evaluate_fixed_path(
            self.models[seed], origin[:10], origin[10:], initial["helpers"]["initialization"]["z0"],
            initial["history_clock"]["clock"], self.u11(index), inputs["equilibrium"][index],
            max_step=step, linear=True, target_scale=inputs["target_scale"], dispatch=self.store.journal.call))
        reset = self.capture(record, "visible_reset", lambda: reset_path_origin(
            self.models[seed], path["path"], inputs["history"][index, -1], inputs["equilibrium"][index],
            dispatch=self.store.journal.call))
        self.origins[(seed, index, code)] = reset["reset"]["origin"]
        return record

    def refine(self, coarse, fine):
        record = {"helpers": {}}

        def inspect():
            result = check_refinement(coarse, fine, self.cache["inputs"]["target_scale"])
            # A failed numerical gate still retains the complete returned diagnosis.
            record["helpers"]["refinement"] = result
            if result["qualified"] is not True:
                error = FloatingPointError("A17 two-step refinement gate failed")
                error.partial_record = result
                error.input_consistency_campaign_partial = record
                raise error
            return result

        self.capture(record, "refinement", lambda: self.store.journal.call("path_refinement", inspect))
        return record

    def history_refinement(self, seed, index):
        return self.refine(*(self.store.value(f"hist-{seed}-{index:03d}-{code}")["helpers"]
                             ["path_evaluation"]["path"] for code, _ in STEPS))

    def future(self, seed, index, code, step):
        record = {"helpers": {}}
        origin = self.origins[(seed, index, code)]
        inputs = self.cache["inputs"]
        result = self.capture(record, "path_evaluation", lambda: evaluate_fixed_path(
            self.models[seed], origin[:10], origin[10:20], origin[20:], self.clocks[index],
            inputs["constant_power"][index], inputs["equilibrium"][index], max_step=step, linear=False,
            target_scale=inputs["target_scale"], dispatch=self.store.journal.call))
        if not np.array_equal(result["path"][0], origin):
            raise PermissionError("A17 future changed its separately frozen initial state")
        self.futures[(seed, index, code)] = result["path"]
        return record

    def reference(self, seed, index, code, group):
        row = self.cache[f"reference-{seed}-{index:03d}-{code}"]
        return row["candidate" if group == "I" else "baseline"]

    def truth(self, index):
        return self.cache[f"reference-324-{index:03d}-h005"]["truth"]

    def references(self):
        """Nine hashes of the decoded saved copies; no old archive is reopened."""
        record = {"hashes": {}, "derived_actual": {"reference_ancestry_logical_bytes": 0}}
        lineage = self.source["ancestry_fingerprints"]

        def verify(name, array, fingerprints):
            item = {"shape": list(array.shape), "dtype": array.dtype.str,
                    "logical_bytes": int(array.nbytes)}
            record["hashes"][name] = item

            def calculate():
                item["sha256_c_order_bytes"] = array_sha(array)
                record["derived_actual"]["reference_ancestry_logical_bytes"] += int(array.nbytes)
                if any(any(item[key] != fingerprint[key] for key in ("shape", "dtype", "sha256_c_order_bytes"))
                       for fingerprint in fingerprints):
                    raise PermissionError("A17 saved-copy ancestry hash differs")
                return item

            self.store.journal.call("reference_ancestry_hashes", calculate)

        for seed in SEEDS:
            for code, _ in STEPS:
                verify(f"E-{seed}-{code}", np.concatenate(
                    [self.reference(seed, i, code, "E") for i in range(self.n)]),
                    [lineage["E"][f"{seed}-{code}"]["future"]])
        common = lineage["truth"]["A10/real_train_arrays.npz"]["members"]
        verify("truth", np.concatenate([self.truth(i) for i in range(self.n)]), [common["future"]])
        verify("clock", np.concatenate([self.clocks[i] for i in range(self.n)]), [common["forecast_time"]])
        offsets = np.r_[0, np.cumsum(np.asarray(self.counts, dtype=np.int64) + 1)]
        verify("offsets", offsets, [common["offsets"], *(
            row["offsets"] for row in lineage["E"].values())])
        return record

    def reference_identity(self, seed, index, code):
        def inspect():
            row = self.cache[f"reference-{seed}-{index:03d}-{code}"]
            truth = self.truth(index)
            observed = self.cache["inputs"]["history"][index, -1]
            candidate = self.futures[(seed, index, code)]
            if (not np.array_equal(row["candidate"][0], row["reset"])
                    or not np.array_equal(candidate[0], self.origins[(seed, index, code)])
                    or any(not np.array_equal(path[0, :20], observed)
                           for path in (candidate, row["candidate"], row["baseline"], truth))):
                raise PermissionError("A17 saved reference/common observed origin differs")
            return {"common_observed_origin_bitwise": True, "I_reset_origin_bitwise": True,
                    "C_frozen_origin_bitwise": True, "clock_source_descriptor_bound": True}
        return self.store.journal.call("reference_identity_checks", inspect)

    def score(self, seed, index, code):
        result = {}
        try:
            for group in ("C", "I", "E"):
                path = self.futures[(seed, index, code)] if group == "C" else self.reference(seed, index, code, group)
                result[group] = self.store.journal.call("event_scores", lambda path=path: future_event_scores(
                    path, self.truth(index), self.clocks[index], self.cache["inputs"]["target_scale"]))
        except BaseException as error:
            error.input_consistency_campaign_partial = {"returned_event_scores": result}
            raise
        # An interrupted three-method score never creates a completed event.
        for group, value in result.items():
            self.scores[(group, seed, code)][index] = value
        return result

    def aggregate(self, group, seed, code):
        result = self.store.journal.call("population_scores", lambda: population_scores(
            self.scores[(group, seed, code)], self.weights, np.asarray(self.counts, dtype=np.int64)))
        self.groups[(group, seed, code)] = result
        return result

    def pair(self, reference, seed, code):
        def compare():
            candidate = self.groups.get(("C", seed, code))
            baseline = self.groups.get((reference, seed, code))
            if candidate is None or baseline is None:
                return {"complete": False, "pass": False, "conditions": None, "differences": None,
                        "reason": "population_summary_numerical_failure"}
            return paired_group(candidate, baseline)
        result = self.store.journal.call("paired_groups", compare)
        self.pairs[(reference, seed, dict(STEPS)[code])] = result
        return result

    def final_parameters(self):
        result = {str(seed): self.store.journal.call("parameter_identity", lambda seed=seed:
                    self.parameter_identity(seed)) for seed in SEEDS}
        return {"seeds": result, "derived_actual": dict(sum((Counter(row["derived_actual"])
                                                              for row in result.values()), Counter()))}

    def run(self):
        self.progress("source_metadata")
        self.reader.read_metadata()
        self.source = self.store.execute("sources", lambda: self.store.journal.call(
            "source_identity_checks", lambda: verify_source_documents(self.plan, self.reader.metadata_envelopes())))
        self.reader.integrity("before")
        self.load_phase(1)
        save_record(self.directory, "causal_inputs", self.cache)
        for seed in SEEDS:
            self.store.execute(f"static-{seed}", lambda seed=seed: self.construct(seed))
            for index in range(self.n):
                self.store.execute(f"init-{seed}-{index:03d}", lambda seed=seed, index=index:
                                   self.initialize(seed, index))
                for code, step in STEPS:
                    self.store.execute(f"hist-{seed}-{index:03d}-{code}", lambda seed=seed, index=index,
                                       code=code, step=step: self.history(seed, index, code, step))
                self.store.execute(f"hrefine-{seed}-{index:03d}", lambda seed=seed, index=index:
                                   self.history_refinement(seed, index))
                if (index + 1) % 32 == 0:
                    self.progress("history", seed=seed, retained_events=index + 1)
        self.freeze(2)
        self.progress("origins_frozen")
        self.load_phase(2)
        save_record(self.directory, "forecast_clocks", {key: value for key, value in self.cache.items()
                                                       if key.startswith("clock-")})
        for index, count in enumerate(self.counts):
            value = self.store.execute(f"clock-{index:03d}", lambda index=index, count=count:
                self.store.journal.call("future_clock_checks", lambda: clock_checks(
                    self.cache[f"clock-{index:03d}"]["clock"], count)))
            self.clocks[index] = value["clock"]
        for seed in SEEDS:
            for index in range(self.n):
                for code, step in STEPS:
                    self.store.execute(f"future-{seed}-{index:03d}-{code}", lambda seed=seed, index=index,
                                       code=code, step=step: self.future(seed, index, code, step))
                self.store.execute(f"frefine-{seed}-{index:03d}", lambda seed=seed, index=index:
                    self.refine(*(self.futures[(seed, index, code)] for code, _ in STEPS)))
                if (index + 1) % 32 == 0:
                    self.progress("candidate_futures", seed=seed, retained_events=index + 1)
        self.freeze(3)
        self.progress("candidate_futures_frozen")
        self.load_phase(3)
        save_record(self.directory, "comparison_inputs", {key: value for key, value in self.cache.items()
                                                        if key.startswith("reference-")})
        self.store.execute("references", self.references)
        for seed in SEEDS:
            for index in range(self.n):
                for code, _ in STEPS:
                    self.store.execute(f"identity-{seed}-{index:03d}-{code}", lambda seed=seed, index=index,
                                       code=code: self.reference_identity(seed, index, code))
                for reference in ("I", "E"):
                    self.store.execute(f"refine-{reference}-{seed}-{index:03d}", lambda seed=seed, index=index,
                        reference=reference: self.refine(*(self.reference(seed, index, code, reference)
                                                          for code, _ in STEPS)))
                for code, _ in STEPS:
                    self.store.execute(f"score-{seed}-{index:03d}-{code}", lambda seed=seed, index=index,
                                       code=code: self.score(seed, index, code))
                if (index + 1) % 32 == 0:
                    self.progress("saved_reference_scoring", seed=seed, retained_events=index + 1)
        for group in ("C", "I", "E"):
            for seed in SEEDS:
                for code, _ in STEPS:
                    self.store.execute(f"group-{group}-{seed}-{code}", lambda group=group, seed=seed,
                                       code=code: self.aggregate(group, seed, code))
        for reference in ("I", "E"):
            for seed in SEEDS:
                for code, _ in STEPS:
                    self.store.execute(f"pair-{reference}-{seed}-{code}", lambda reference=reference, seed=seed,
                                       code=code: self.pair(reference, seed, code))
        self.store.execute("parameters-final", self.final_parameters)
        self.reader.integrity("after")
        access = self.reader.finish()
        self.store.journal.close()
        if (self.store.ordinal != len(self.compute["scope_order"])
                or list(self.store.records) != self.compute["scope_order"]):
            raise RuntimeError("A17 final scope graph is incomplete")
        V = all(row["status"] == "complete" for row in self.store.records.values())
        pairs = {(reference, seed, step): self.pairs.get((reference, seed, step),
                 {"complete": False, "pass": False})
                 for reference in ("I", "E") for seed in SEEDS for _, step in STEPS}
        summary = {**campaign_verdict(V, pairs), "diagnostic_status": "COMPLETE", "allocated_train": 268,
                   "eligible_train": self.n, "seed_event_slots": len(SEEDS) * self.n, "scope_count": self.store.ordinal,
                   "source_commit": self.receipt["source_commit"], "authorization": AUTH,
                   "scope_statuses": dict(Counter(row["status"] for row in self.store.records.values())),
                   "claim_boundary": BOUNDARY}
        save_record(self.directory, "access_ledger", access)
        write_json(self.directory / "scientific_scopes.json", self.store.journal.closed_scopes)
        save_record(self.directory, "weighted_summaries", {
            "groups": {f"{group}-{seed}-{code}": value for (group, seed, code), value in self.groups.items()},
            "pairs": {f"{reference}-{seed}-{step}": value for (reference, seed, step), value in pairs.items()}})
        return summary


def seal_result(directory, output_snapshot):
    """Measure the finite seal tail after the output journal becomes immutable.

    The final four actions cannot be self-hashed inside the run. Their complete
    actual prefix is returned to the independent shared-budget receipt; failures
    carry the same prefix to the separate fatal writer, without retry.
    """
    directory = Path(directory)
    if (output_snapshot.get("poisoned") is not False
            or output_snapshot.get("journal_durably_synced") is not True
            or output_snapshot.get("journal_closed") is not True
            or output_snapshot.get("active_open_ids") or output_snapshot.get("unknown_operations")):
        raise RuntimeError("A17 output I/O journal is not completely closed and durable")
    if (directory / "artifact_hashes.json").exists():
        raise FileExistsError("A17 evidence is already sealed and cannot be resealed")
    available = (270000 - output_snapshot["records"]) // 2
    ledger = SealLedger(directory, max_actions=max(1, available))
    phase = "inventory"
    try:
        targets = []
        for path in sorted(directory.rglob("*")):
            if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
                raise PermissionError("A17 finalizer cannot follow linked evidence")
            if path.is_file():
                targets.append(path.relative_to(directory).as_posix())
        tail_actions = len(targets) + 5
        if tail_actions > available:
            raise RuntimeError("A17 final output I/O transition cap exceeded")
        phase = "preceding_hashes"
        manifest = {name: ledger.hash_file(name) for name in targets}
        phase = "source_transition_count"
        source_lines = ledger.count_lines("source_access.jsonl")
        if not ledger.actions[-1].get("ends_with_newline", True):
            raise OSError("A17 source journal ends with an incomplete transition")
        output_transitions = output_snapshot["records"] + 2 * tail_actions
        if output_transitions + source_lines > 380000:
            raise RuntimeError("A17 combined I/O transition cap exceeded")
        preceding = ledger.snapshot()
        io_summary = {"captured_run_io": output_snapshot, "source_transition_lines": source_lines,
            "sealing": {"actions_including_last_writes": tail_actions,
                "output_transitions_including_seal": output_transitions,
                "combined_transitions_including_seal": output_transitions + source_lines,
                "preceding_hash_read_bytes": preceding["actual_read_bytes"],
                "preceding_hash_files": len(manifest), "preceding_tail": preceding,
                "boundary": "Manifest excludes itself. Final four summary/manifest writes and hashes, with actual byte returns, are in the external shared-budget receipt; interruption is unknown/spent. The harness independently verifies the sealed files."}}
        phase = "io_summary_write"
        ledger.write_json("output_io_summary.json", io_summary)
        phase = "io_summary_hash"
        manifest["output_io_summary.json"] = ledger.hash_file("output_io_summary.json")
        phase = "manifest_write"
        ledger.write_json("artifact_hashes.json", manifest)
        phase = "manifest_hash"
        manifest_sha = ledger.hash_file("artifact_hashes.json")
        actual = ledger.snapshot()
        if actual["action_count"] != tail_actions or actual["transition_records"] != 2 * tail_actions:
            raise RuntimeError("A17 successful seal action count differs")
        return {"output_transitions_including_seal": output_transitions,
                "combined_transitions_including_seal": output_transitions + source_lines,
                "seal_actual_read_bytes": actual["actual_read_bytes"],
                "seal_actual_write_bytes": actual["actual_write_bytes"],
                "artifact_manifest_sha256": manifest_sha, "tail_ledger": actual,
                "manifest_final_hash_is_external_completion_action": True}
    except BaseException as error:
        ledger.poisoned = True
        if not hasattr(error, "input_consistency_seal_partial_record"):
            error.input_consistency_seal_partial_record = ledger.snapshot()
        error.input_consistency_seal_phase = phase
        raise


def _execute(root, result_dir, config, plan, compute, receipt):
    from kron_reduce.evaluation.input_consistency_output_io import OutputIO

    campaign = None
    io = OutputIO(result_dir, max_records=270000)
    try:
        with io:
            for name, key in (("read_plan.json", "read_plan_path"), ("compute_plan.json", "compute_plan_path")):
                raw = (root / config[key]).read_bytes()
                with (result_dir / name).open("xb") as stream:
                    if stream.write(raw) != len(raw):
                        raise OSError("A17 short control-copy write")
                    stream.flush()
                    os.fsync(stream.fileno())
            write_json(result_dir / "run_receipt.json", receipt)
            write_json(result_dir / "environment.json", {"python": sys.version, "executable": sys.executable,
                "versions": {name: version(name) for name in ("numpy", "scipy", "h5py", "torch", "filelock")},
                "constraints_sha256": sha256_file(root / "requirements/constraints-py312.txt")})
            write_json(result_dir / "allocated_slots.json", plan["population"])
            campaign = Campaign(root, result_dir, plan, compute, receipt)
            summary = campaign.run()
            write_json(result_dir / "summary.json", summary)
            write_json(result_dir / "stage_evidence.json", {"status": "PASS", "process_status": "PASS",
                "source_commit": receipt["source_commit"], "diagnostic_status": "COMPLETE",
                "scientific_verdict": summary["scientific_verdict"], "promotion_decision": "KEEP_BASELINE",
                "data_policy": "kiki_original_train_only", "development_authorization_id": AUTH,
                "evidence_kind": "uncertified_train_input_consistency", "claim_boundary": BOUNDARY,
                "certified_envelope_established": False, "deployment_predictions_allowed": False,
                "model_promoted": False, "level_6_claimed": False, "hidden_training_targets_read": False,
                "labels_read": 0, "internal_val_ids_read": [], "source_val_ids_read": [], "test_ids_read": [],
                "result_dir": result_dir.name, "new_missed_instability_result_established": False})
        tail = seal_result(result_dir, io.snapshot())
        return summary, tail
    except BaseException as error:
        if campaign is not None:
            campaign.store.journal.poisoned = True
            campaign.reader.failed = True
        # The independent fatal writer runs only after the poisoned I/O guard
        # restored Path.open. It performs no source/scientific operation.
        save_record(result_dir, "fatal_scientific_partial", error_record(error))
        cleanup_errors = []
        if campaign is not None:
            for name, close in (("reader", campaign.reader.close), ("scientific_journal", campaign.store.journal.close)):
                try:
                    close()
                except BaseException as cleanup_error:  # noqa: BLE001 - retain the primary failure during cleanup
                    cleanup_errors.append({"resource": name, **error_record(cleanup_error)})
        save_record(result_dir, "fatal_prefix", {"reservation": receipt, "output_io": io.snapshot(),
            "cleanup_errors": cleanup_errors,
            "retry_allowed": False, "read_prefix": campaign.reader.snapshot() if campaign else None,
            "closed_scopes_prefix": campaign.store.journal.closed_scopes if campaign else [],
            "unclosed_scope_unknown_spent": campaign.store.journal.active if campaign else None,
            "attempted_scope_without_closed_record": (
                compute["scope_order"][campaign.store.ordinal - 1]
                if campaign is not None and campaign.store.ordinal > len(campaign.store.records) else None),
            "unvisited_scopes": compute["scope_order"][campaign.store.ordinal if campaign else 0:]})
        raise


def run(*, root, experiment, stage, result_dir, run_id):
    if stage != STAGE:
        raise PermissionError("A17 stage differs")
    config = tomllib.loads((root / "configs/experiments" / experiment["config"]).read_text(encoding="utf-8"))
    approval = authorization(root, experiment, config)
    plan, compute, _ = bound_controls(root, config)
    budget, receipt = reserve(root, run_id, config)
    try:
        summary, tail = _execute(root, result_dir, config, plan, compute, receipt)
        authorization(root, experiment, config)
        if git_head(root) != receipt["source_commit"]:
            raise RuntimeError("A17 source HEAD changed")
        write_json(budget / "completed.json", {"run_id": run_id, "authorization_spent": True,
            "diagnostic_status": summary["diagnostic_status"], "V_pass": summary["V_pass"],
            "scientific_verdict": summary["scientific_verdict"],
            "scientific_review_sha256": approval["scientific_review_sha256"], "final_seal": tail})
    except BaseException as error:
        write_json(budget / "failed.json", {"run_id": run_id, "authorization_spent": True,
            "error_type": type(error).__name__, "message": str(error), "retry_allowed": False})
        raise


def execute_registered_audit(root):
    root = Path(root).resolve(strict=True)
    experiment = load_manifest(root)["experiments"][EXPERIMENT]
    config = tomllib.loads((root / "configs/experiments" / experiment["config"]).read_text(encoding="utf-8"))
    authorization(root, experiment, config)
    bound_controls(root, config)
    lock_dir = root / ".project_state/locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with FileLock(lock_dir / f"{WORKFLOW}.lock", timeout=0):
        return _run_experiment(root, EXPERIMENT, STAGE)


def main():
    parser = argparse.ArgumentParser(description="One-use A17 fixed-model train input consistency comparison")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    print(execute_registered_audit(parser.parse_args().root))


if __name__ == "__main__":
    main()
