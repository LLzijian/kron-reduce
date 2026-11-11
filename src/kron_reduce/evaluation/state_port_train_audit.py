"""One frozen paired fit from manifest-bound saved train arrays."""
from __future__ import annotations

import json
import sys
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from importlib.metadata import version

import numpy as np
import torch

from kron_reduce.data.energy_cap_saved_inputs import EnergyCapSavedInputs
from kron_reduce.evaluation.constant_c_observability import Operations, save_record, write_json
from kron_reduce.evaluation.energy_cap_scores import aggregate_group, score_pair
from kron_reduce.evaluation.state_port_train_scores import descriptive, paired_line_summary, verdict
from kron_reduce.evaluation.state_port_training import loss, rollout
from kron_reduce.models.state_port import StatePortClosure
from project_harness.evidence import sha256_file
from project_harness.git_state import git_head

SEEDS = (324, 325, 326)
METHODS = ("baseline", "constant", "state")
STEPS = ("0.005", "0.0025")
PARAMETERS = ("mass", "edge_stiffness", "damping", "coupling", "latent_skew", "latent_dissipation")


class Calls:
    def __init__(self, limits):
        self.limits, self.counts, self.returned = limits, Counter(), Counter()
        self.last_rhs = None
        self.numpy_kind = "numpy_integration_rhs"
        self.context, self.fit_state = {}, None
        self.scope_limits, self.scope_start = None, None

    def add(self, key, count=1):
        if (self.scope_limits is not None
                and self.counts[key] - self.scope_start[key] + count > self.scope_limits.get(key, 0)):
            raise RuntimeError(f"state-port scope budget exhausted: {key}")
        if self.counts[key] + count > self.limits[key]:
            raise RuntimeError(f"state-port train budget exhausted: {key}")
        self.counts[key] += count

    def call(self, key, function):
        self.add(key)
        value = function()
        self.returned[key] += 1
        return value

    def __call__(self, transition, *, batch_size, state, error=None, result=None, **location):
        if transition == "attempted":
            self.add("torch_batch_rhs")
            self.add("torch_event_rhs", batch_size)
            self.last_rhs = {"backend": "torch", "state": np.array(state, copy=True),
                             "batch_size": batch_size, "rhs_call": self.counts["torch_batch_rhs"],
                             "context": self.context.copy(), **location}
        elif transition == "returned":
            self.returned["torch_batch_rhs"] += 1
            self.returned["torch_event_rhs"] += batch_size
            self.last_rhs["returned"] = result
        elif transition != "failed":
            raise ValueError("unknown RHS transition")


@contextmanager
def _scope(ops, calls, name):
    with ops.scope(name):
        calls.scope_limits = ops.caps[name]["logical_limits"]
        calls.scope_start = calls.counts.copy()
        try:
            yield
            if dict(calls.counts - calls.scope_start) != calls.scope_limits:
                raise RuntimeError("scope calls differ from frozen allocation: " + name)
        finally:
            calls.scope_limits, calls.scope_start = None, None


@dataclass(frozen=True, kw_only=True)
class CountedPort(StatePortClosure):
    calls: Calls

    def rhs(self, state, power, equilibrium):
        location = self.calls.context.copy()
        sequence = self.calls.counts[self.calls.numpy_kind] - location.get("rhs_start", 0)
        if self.calls.numpy_kind == "numpy_integration_rhs":
            location.update(output_index=sequence // 16 + 1, substep=(sequence // 4) % 4, rk_stage=sequence % 4 + 1)
        self.calls.last_rhs = {"backend": "numpy", "state": np.array(state, copy=True),
                               "context": location, "rhs_sequence": self.calls.counts[self.calls.numpy_kind]}
        result = self.calls.call(self.calls.numpy_kind, lambda: super(CountedPort, self).rhs(state, power, equilibrium))
        self.calls.last_rhs["returned"] = result.copy()
        return result


def _model(parameters, kind, delta, calls):
    return calls.call("model_constructions", lambda: CountedPort(
        **parameters, backbone="sine", port_kind=kind, delta=delta, calls=calls))


def _torch_operation(ops, category, function, tag):
    result = {}

    def run():
        result["value"] = function()
        return {"shape": list(result["value"].shape), "dtype": str(result["value"].dtype)}

    ops.call(category, run, tag=tag)
    return result["value"]


def _load(root, plan, ops, calls, directory):
    sources = json.loads((root / plan["sources_path"]).read_text(encoding="utf-8"))
    write_json(directory / "source-inventory.json", sources)
    reader = EnergyCapSavedInputs(root, sources, ops)
    roots = {key: "results/" + value["run_id"] for key, value in sources["sources"].items()}
    with _scope(ops, calls, "inputs"):
        for prefix_root in roots.values():
            manifest = reader.read_json(prefix_root + "/artifact_hashes.json")
            prefix = prefix_root + "/"
            for relative, spec in reader.files.items():
                if (relative.startswith(prefix) and not relative.endswith("/artifact_hashes.json")
                        and manifest.get(relative.removeprefix(prefix)) != spec["sha256"]):
                    raise ValueError("source manifest and whitelist differ")
        population = reader.read_json(roots["a14"] + "/allocated_population.json")
        other = reader.read_json(roots["a15"] + "/allocated_population.json")
        if any(other.get(key) != value for key, value in population.items()):
            raise ValueError("source populations differ")
        counts = np.array(plan["future_counts"], dtype=np.int64)
        if other["future_point_counts"] != counts.tolist():
            raise ValueError("future counts differ from frozen metadata")
        rows, weights = population["cache_rows"], np.asarray(population["weights"], dtype=float)
        if (len(rows) != 258 or len(population["allocated_slots"]) != 268
                or weights.shape != (258,) or np.any(weights <= 0) or abs(weights.sum() - 1) > 1e-12):
            raise ValueError("population or weights differ")
        inputs = reader.read_arrays(roots["a14"] + "/permitted_inputs.npz")
        parameters, initial, saved_baseline = {}, {}, {}
        for seed in SEEDS:
            meta = reader.read_json(roots["a14"] + f"/static-{seed}.json")
            static = reader.read_arrays(roots["a14"] + f"/static-{seed}.npz")
            if meta["parameters"]["backbone"] != "sine":
                raise ValueError("baseline potential differs")
            parameters[seed] = {name: static["parameters__" + name] for name in PARAMETERS}
            if not np.array_equal(parameters[seed]["mass"], inputs["mass"]):
                raise ValueError("model mass differs")
            initial[seed] = np.zeros((258, 22))
            for index in range(258):
                key = f"seed-{seed}-event-{index:03d}"
                x0 = reader.read_arrays(roots["a14"] + f"/{key}/history-0.0025.npz")["baseline_reset"]
                if x0.shape != (22,) or not np.array_equal(x0[:20], inputs["history"][index, -1]):
                    raise ValueError("A06 forecast state does not match its visible history endpoint")
                initial[seed][index] = x0
        causal_access = reader.snapshot()
        for key, limit in plan["source_phase_limits"]["through_common_initial_state_freeze"].items():
            if causal_access[key] != limit:
                raise ValueError("causal source phase differs from its frozen budget")
        write_json(directory / "population.json", population)
        save_record(directory, "common-initial-state", {"inputs": inputs, "parameters": parameters, "initial": initial})
        write_json(directory / "common-initial-state-freeze.json", {
            "artifacts": {f"common-initial-state.{ext}": sha256_file(directory / f"common-initial-state.{ext}")
                          for ext in ("json", "npz")}, "population_sha256": sha256_file(directory / "population.json"),
            "source_access_at_freeze": causal_access})
        omega_s = reader.read_arrays(roots["a15"] + "/permitted_truth.npz")["omega_s"]
        if omega_s.shape != (258,) or not np.isfinite(omega_s).all() or np.any(omega_s <= 0):
            raise ValueError("event angular-frequency scales are invalid")
        truth = np.zeros((258, 301, 20))
        for seed in SEEDS:
            saved_baseline[seed] = []
            for index, count in enumerate(counts):
                key = f"seed-{seed}-event-{index:03d}"
                x0 = initial[seed][index]
                future = reader.read_arrays(roots["a15"] + f"/{key}/future-0.0025.npz")
                if (x0.shape != (22,) or future["baseline"].shape != (count + 1, 22)
                        or not np.array_equal(x0, future["baseline"][0])
                        or not np.array_equal(x0[:20], inputs["history"][index, -1])):
                    raise ValueError("shared complete A06 forecast state differs")
                saved_baseline[seed].append(future["baseline"])
                if seed == SEEDS[0]:
                    if (future["truth"].shape != (count + 1, 20)
                            or not np.array_equal(future["clock"], np.arange(count + 1) * .01)
                            or not np.array_equal(future["truth"][0], x0[:20])):
                        raise ValueError("truth origin or clock differs")
                    truth[index, :count + 1] = future["truth"]
        access = reader.snapshot()
        expected = plan["source_limits"]
        for key in ("files_attempted", "members_attempted", "bytes_read", "logical_bytes"):
            if access[key] != expected[key]:
                raise ValueError("source access count differs: " + key)
        if access["npy_header_bytes"] > expected["npy_header_bytes"]:
            raise ValueError("source headers exceed budget")
        write_json(directory / "source-access.json", access)
        save_record(directory, "fixed-targets", {"truth": truth, "counts": counts, "weights": weights, "omega_s": omega_s})
    return inputs, parameters, initial, saved_baseline, truth, counts, weights, omega_s, rows


def _fit(model, initial, inputs, counts, target, weights, calls, ops, directory):
    with _scope(ops, calls, f"fit-start-{directory.name}"):
        calls.add("fit_attempts")
        theta = torch.zeros(model.delta.shape, dtype=torch.float64, requires_grad=True)
        scale = float(np.linalg.norm(model.coupling) / np.sqrt(model.delta.size))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("frozen coupling gives no positive finite parameter scale")
        optimizer = torch.optim.Adam([theta], lr=.01, betas=(.9, .999), eps=1e-8,
                                     weight_decay=0, amsgrad=False, foreach=False, fused=False)
        calls.returned["fit_attempts"] += 1
        ops.call("probe", lambda: {"theta0": theta.detach().numpy().copy(), "scale": scale,
                                   "optimizer": "fresh_adam", "parameter_count": theta.numel()}, tag="fit_start")
    calls.fit_state = (theta, optimizer)
    for update in range(1, 41):
        with _scope(ops, calls, f"fit-{directory.name}-{update:02d}"):
            optimizer.zero_grad(set_to_none=True)
            before = theta.detach().numpy().copy()
            event_losses = {}
            calls.context = {"group": directory.name, "update": update,
                             "rhs_start": calls.counts["torch_batch_rhs"]}

            def forward(event_losses=event_losses):
                path = rollout(model, initial, inputs["constant_power"], inputs["equilibrium"],
                               counts, max_step=.01, delta=scale * torch.tanh(theta), calls=calls)
                scalar, events = loss(path, target, weights, inputs["target_scale"], counts, return_events=True)
                event_losses["value"] = events.detach().numpy().copy()
                return scalar

            objective = _torch_operation(ops, "probe", lambda: calls.call("training_forward", forward), "training_forward")
            if not torch.isfinite(objective):
                raise FloatingPointError("nonfinite training objective")
            ops.call("probe", partial(calls.call, "backward", objective.backward), tag="backward")
            gradient = theta.grad.detach().numpy().copy()
            norm = calls.call("gradient_clip", lambda: torch.nn.utils.clip_grad_norm_(
                [theta], 10., error_if_nonfinite=True))
            clipped = theta.grad.detach().numpy().copy()
            ops.call("probe", lambda: calls.call("optimizer_update", optimizer.step), tag="optimizer_update")
            if not torch.isfinite(theta).all():
                raise FloatingPointError("nonfinite updated parameter")
            state = optimizer.state[theta]
            if not all(torch.isfinite(value).all() for value in state.values()):
                raise FloatingPointError("nonfinite optimizer state")
            save_record(directory, f"update-{update:02d}", {
                "update": update, "objective_parameter_step": update - 1, "loss": float(objective.detach()),
                "theta_before": before, "gradient_unclipped": gradient, "gradient_norm_unclipped": float(norm),
                "event_loss": event_losses["value"], "gradient_clipped": clipped,
                "theta_after": theta.detach().numpy().copy(),
                "adam": {key: value.detach().numpy().copy() for key, value in state.items()},
                "logical_attempted": dict(calls.counts), "logical_returned": dict(calls.returned),
            })
        del objective
        if update % 5 == 0:
            print(f"state port {directory.name}: {update}/40", flush=True)
    delta = scale * np.tanh(theta.detach().numpy())
    save_record(directory, "final", {"theta": theta.detach().numpy(), "delta": delta, "scale": scale})
    calls.fit_state = None
    return delta


def _physics(model, paths, counts, inputs, calls):
    calls.numpy_kind = "physical_rhs"
    samples, coi = [], 0.
    try:
        for index, count in enumerate(counts):
            path = paths[index, :count + 1]
            coi = max(coi, float(np.max(np.abs(path[:, :10] @ model.mass))),
                      float(np.max(np.abs(path[:, 10:20] @ model.mass))))
            for point in (0, int(count) // 2, int(count)):
                calls.context.update(phase="physical_check", event_index=index, point=point)
                state, power = path[point], inputs["constant_power"][index]
                check = model.energy_identity(state, power, inputs["equilibrium"][index])
                v, z = state[10:20], state[20:]
                denominator = max(1., abs(float(v @ power)) + abs(float(v @ model.damping @ v))
                                  + abs(float(z @ model.latent_dissipation @ z)))
                samples.append({"index": index, "point": point, **check,
                                "denominator": denominator, "hidden_energy": .5 * float(z @ z),
                                "total_energy": float(model.energy(state, inputs["equilibrium"][index])),
                                "normalized_error": float(check["identity_error"] / denominator)})
    finally:
        calls.numpy_kind = "numpy_integration_rhs"
    return {"points": samples, "max_mass_coi": coi,
            "qualified": bool(coi <= 1e-10 and all(row["identity_error"] <= 1e-8 for row in samples))}


def execute(*, root, result_dir, plan):
    ops, calls = Operations(result_dir / "operations", plan), Calls(plan["logical_operation_limits"])
    old_threads, old_deterministic = torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    groups, checks = {}, []
    try:
        write_json(result_dir / "execution-plan.json", plan)
        inputs, parameters, initial, old_baseline, target, counts, weights, omega_s, rows = _load(
            root, plan, ops, calls, result_dir)
        ids = [row["event_id"] for row in rows]
        valid = True
        all_models = {}
        for seed in SEEDS:
            models = {}
            with _scope(ops, calls, f"models-{seed}"):
                for method in METHODS:
                    models[method] = ops.call("probe", lambda method=method, seed=seed: _model(
                        parameters[seed], "constant" if method == "baseline" else method,
                        np.zeros((9, 2)), calls), tag="model_construction")
            for method in METHODS[1:]:
                directory = result_dir / f"{seed}-{method}"
                directory.mkdir()
                delta = _fit(models[method], initial[seed], inputs, counts, target, weights,
                             calls, ops, directory)
                with _scope(ops, calls, f"final-model-{seed}-{method}"):
                    models[method] = ops.call("probe", lambda method=method, delta=delta, seed=seed: _model(
                        parameters[seed], method, delta, calls), tag="final_model")
            all_models[seed] = models
        final_files = sorted(result_dir.glob("*-*/final.*"))
        if len(final_files) != 12:
            raise ValueError("all six final parameter records must be frozen before evaluation")
        final_files += sorted(result_dir.glob("*-*/update-40.*"))
        if len(final_files) != 24:
            raise ValueError("all six final optimizer records must be frozen before evaluation")
        write_json(result_dir / "final-model-freeze.json", {
            "artifacts": {path.relative_to(result_dir).as_posix(): sha256_file(path) for path in final_files},
            "optimizer_updates": calls.counts["optimizer_update"]})
        all_paths = {}
        for seed in SEEDS:
            models = all_models[seed]
            all_paths[seed] = {}
            groups[str(seed)] = {step: {} for step in STEPS}
            for method in METHODS:
                paths = {}
                for step in STEPS:
                    with _scope(ops, calls, f"evaluate-{seed}-{method}-{step.replace('.', '')}"):
                        calls.context = {"seed": seed, "method": method, "max_step": step,
                                         "rhs_start": calls.counts["torch_batch_rhs"]}
                        with torch.no_grad():
                            run_path = partial(rollout,
                                models[method], initial[seed], inputs["constant_power"], inputs["equilibrium"],
                                counts, max_step=float(step), calls=calls)
                            path = _torch_operation(ops, "rollouts", partial(calls.call, "evaluation_forward", run_path),
                                                    "evaluation_forward")
                        paths[step] = path.numpy()
                        physics = ops.call("probe", partial(_physics, models[method], paths[step], counts, inputs, calls),
                                           tag="three_point_physics_and_full_coi")
                        valid = valid and physics["qualified"]
                        save_record(result_dir, f"paths-{seed}-{method}-{step}", {
                            "path": paths[step], "physics": physics, "counts": counts})
                        if not physics["qualified"]:
                            raise FloatingPointError("finite physical-point or full-path COI check failed")
                        if not np.array_equal(paths[step][:, 0], initial[seed]):
                            raise ValueError("evaluation origin differs from the common state")
                all_paths[seed][method] = paths
            print(f"state port seed {seed}: endpoint paths complete", flush=True)
        path_files = sorted(result_dir.glob("paths-*.*"))
        write_json(result_dir / "future-paths-freeze.json", {
            "artifacts": {path.name: sha256_file(path) for path in path_files}})
        for seed in SEEDS:
            models = all_models[seed]
            for method in METHODS:
                paths = all_paths[seed][method]
                with _scope(ops, calls, f"score-{seed}-{method}"):
                    records = {step: [] for step in STEPS}
                    refinements = []
                    for index, count in enumerate(counts):
                        pair = ops.call("probe", partial(score_pair,
                            history=inputs["history"][index], truth=target[index, :count + 1],
                            coarse=paths[STEPS[0]][index, :count + 1], fine=paths[STEPS[1]][index, :count + 1],
                            equilibrium=inputs["equilibrium"][index], mass=inputs["mass"], omega_s=float(omega_s[index]),
                            scale=inputs["target_scale"], clock=np.arange(count + 1) * .01), tag=f"event:{index}")
                        valid = valid and pair["refinement"]["qualified"]
                        refinements.append(pair["refinement"])
                        for step in STEPS:
                            records[step].append(pair["steps"][step])
                    for step in STEPS:
                        groups[str(seed)][step][method] = ops.call("probe", partial(aggregate_group,
                            records[step], weights, counts, event_ids=ids), tag="aggregate")
                    save_record(result_dir, f"scores-{seed}-{method}", {"steps": records, "refinements": refinements})
                    if not valid:
                        raise FloatingPointError("trajectory or rule refinement check failed")
                with _scope(ops, calls, f"backend-{seed}-{method}"):
                    if method == "baseline":
                        reference = old_baseline[seed]
                        indices = range(258)
                    else:
                        index = int(np.flatnonzero(counts == 300)[0])
                        x0 = initial[seed][index]
                        calls.context = {"seed": seed, "method": method, "event_index": index, "max_step": .0025,
                                         "rhs_start": calls.counts["numpy_integration_rhs"], "phase": "numpy_backend"}
                        run_numpy = partial(models[method].rollout, x0[:10], x0[10:20], x0[20:], np.arange(301) * .01,
                                            inputs["constant_power"][index], inputs["equilibrium"][index], max_step=.0025)
                        reference = {index: ops.call("rollouts", partial(calls.call, "numpy_rollout", run_numpy),
                            tag="numpy_backend")}
                        save_record(result_dir, f"numpy-{seed}-{method}", {"index": index, "path": reference[index]})
                        indices = [index]
                    comparison = []
                    for index in indices:
                        ref = reference[index]
                        actual = paths["0.0025"][index, :counts[index] + 1]
                        visible = float(np.max(np.abs((actual[:, :20] - ref[:, :20]) / inputs["target_scale"])))
                        hidden = float(np.max(np.abs(actual[:, 20:] - ref[:, 20:])))
                        bound = 1e-10 * (1 + float(np.max(np.abs(ref[:, 20:]))))
                        comparison.append({"index": index, "visible": visible, "hidden": hidden,
                                           "hidden_bound": bound, "qualified": visible <= 1e-10 and hidden <= bound})
                    valid = valid and all(row["qualified"] for row in comparison)
                    checks.append({"seed": seed, "method": method, "comparisons": comparison})
                    write_json(result_dir / f"backend-{seed}-{method}.json", comparison)
                    if not valid:
                        raise FloatingPointError("NumPy or saved A06 backend check failed")
            print(f"state port seed {seed}: fit and evaluation complete", flush=True)
        with _scope(ops, calls, "summary"):
            descriptive_paths = {str(seed): {step: {method: all_paths[seed][method][step] for method in METHODS}
                                             for step in STEPS} for seed in SEEDS}
            appendix = ops.call("probe", partial(descriptive, descriptive_paths, target, counts,
                                                inputs["target_scale"], rows), tag="descriptive_scores")
            line_summary = ops.call("probe", partial(paired_line_summary, groups, rows), tag="line_summary")
            write_json(result_dir / "descriptive-scores.json", appendix)
            write_json(result_dir / "paired-line-summary.json", line_summary)
            decision = ops.call("probe", lambda: verdict(groups, valid=bool(valid)), tag="scientific_verdict")
            if (dict(calls.counts) != plan["logical_operation_limits"]
                    or calls.counts != calls.returned or dict(ops.counts) != plan["operator_limits"]
                    or ops.counts != ops.returns):
                raise RuntimeError("completed calls do not equal the frozen budget")
            summary = {**decision, "groups": groups, "backend_checks": checks, "process_status": "PASS",
                "source_commit": git_head(root), "data_policy": "frozen_saved_train_only",
                "logical_operations": dict(calls.counts), "logical_returned": dict(calls.returned),
                "operator_counts": dict(ops.counts), "operator_returned": dict(ops.returns),
                "runtime": {"python": sys.version, "executable": sys.executable,
                            "torch_threads": torch.get_num_threads(),
                            "torch_deterministic": torch.are_deterministic_algorithms_enabled(),
                            **{name: version(name) for name in ("numpy", "scipy", "torch")}},
                "claim_boundary": "Fixed-state, fixed-backbone train mechanism comparison at 40 updates; no val/test or deployment claim."}
            write_json(result_dir / "summary.json", summary)
            write_json(result_dir / "stage_evidence.json", {key: summary[key] for key in (
                "source_commit", "process_status", "data_policy", "scientific_verdict", "promotion_decision", "claim_boundary")}
                | {"status": "PASS"})
    except BaseException as error:
        fit = None
        if calls.fit_state is not None:
            theta, optimizer = calls.fit_state
            fit = {"theta": theta.detach().numpy(),
                   "gradient": None if theta.grad is None else theta.grad.detach().numpy(),
                   "adam": {key: value.detach().numpy() for key, value in optimizer.state.get(theta, {}).items()}}
        save_record(result_dir, "failed-operation", {"last_rhs": calls.last_rhs,
            "fit": fit, "context": calls.context,
            "attempted": dict(calls.counts), "returned": dict(calls.returned)}, failed=True)
        write_json(result_dir / "failure.json", {"error_type": type(error).__name__, "message": str(error), "retry_allowed": False})
        raise
    finally:
        ops.journal.close()
        torch.set_num_threads(old_threads)
        torch.use_deterministic_algorithms(old_deterministic)
    write_json(result_dir / "artifact_hashes.json", {path.relative_to(result_dir).as_posix(): sha256_file(path)
        for path in sorted(result_dir.rglob("*")) if path.is_file()})
    return summary
