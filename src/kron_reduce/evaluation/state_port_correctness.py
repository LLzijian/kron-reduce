"""Finite analytic correctness study for reciprocal, state-dependent ports."""
from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass, replace
from importlib.metadata import version

import numpy as np
import torch
from scipy.linalg import helmert

from kron_reduce.evaluation.constant_c_observability import Operations, save_record, write_json
from kron_reduce.models.nonlinear_latent import NonlinearPassiveClosure
from kron_reduce.models.state_port import StatePortClosure
from project_harness.evidence import sha256_file
from project_harness.git_state import git_head


class Calls:
    def __init__(self, limits):
        self.limits, self.counts, self.returned = limits, Counter(), Counter()
        self.last_rhs = None

    def add(self, key):
        if self.counts[key] >= self.limits[key]:
            raise RuntimeError(f"state-port operation budget exhausted: {key}")
        self.counts[key] += 1

    def finish(self, key, value):
        self.returned[key] += 1
        if key in {"numpy_rhs", "torch_rhs"}:
            self.last_rhs["returned"] = value.detach().numpy().copy() if torch.is_tensor(value) else value
        return value


class _Counted:
    def __post_init__(self):
        self.calls.add("model_constructions")
        super().__post_init__()
        self.calls.finish("model_constructions", None)

    def rhs(self, *args):
        self.calls.add("numpy_rhs")
        self.calls.last_rhs = {"backend": "numpy", "state": np.array(args[0], copy=True)}
        return self.calls.finish("numpy_rhs", super().rhs(*args))


@dataclass(frozen=True, kw_only=True)
class _Baseline(_Counted, NonlinearPassiveClosure):
    calls: Calls


@dataclass(frozen=True, kw_only=True)
class _Port(_Counted, StatePortClosure):
    calls: Calls


def fixture(calls):
    i = np.arange(1, 11, dtype=float)
    mass, basis = 1 + .1 * np.arange(10), helmert(10, full=False).T
    q = np.eye(10) - np.outer(np.ones(10), mass) / mass.sum()
    p = q.T
    parameters = {
        "mass": mass, "edge_stiffness": .6 + .01 * np.arange(45),
        "damping": .15 * basis @ basis.T,
        "coupling": basis[:, :2] @ np.diag([.35, .25]),
        "latent_skew": np.array([[0., -.7], [.7, 0.]]),
        "latent_dissipation": np.diag([.12, .18]), "calls": calls,
    }
    scale = np.linalg.norm(parameters["coupling"]) / np.sqrt(18)
    j = np.arange(1, 10, dtype=float)
    theta = .25 * np.column_stack([np.cos(j), np.sin(2*j)])
    states = {
        "x1": np.r_[.6*q @ np.sin(i*np.pi/5), .08*q @ np.cos(i*np.pi/5), [.4, -.3]],
        "x2": np.r_[.9*q @ np.cos(i*np.pi/5), -.05*q @ np.sin(i*np.pi/5), [-.25, .45]],
        "zero": np.zeros(22),
    }
    models = {"baseline": _Baseline(**parameters)}
    for kind in ("constant", "state"):
        models[kind + "0"] = _Port(**parameters, port_kind=kind, delta=np.zeros((9, 2)))
        models[kind] = _Port(**parameters, port_kind=kind, delta=scale*np.tanh(theta))
    return models, states, .03*p @ np.sin(i), .1*q @ np.cos(i), theta, scale


def _maximum(value):
    return float(np.max(np.abs(value)))


def _gradient_comparison(actual, expected):
    error = _maximum(actual - expected)
    tolerance = 1e-8 + 2e-6*_maximum(expected)
    return {"error": error, "tolerance": tolerance, "pass": error <= tolerance}


def _numpy_path(model, state, time, power, equilibrium, step):
    model.calls.add("rollout_validations")
    value = model.rollout(state[:10], state[10:20], state[20:], time,
                          power, equilibrium, max_step=step)
    return model.calls.finish("rollout_validations", value)


def _torch_field(model, calls, delta):
    raw = model.torch_vector_field(delta)

    def field(state, power, equilibrium):
        calls.add("torch_rhs")
        calls.last_rhs = {"backend": "torch", "state": state.detach().numpy().copy()}
        return calls.finish("torch_rhs", raw(state, power, equilibrium))

    return field


def _torch_path(model, state, time, power, equilibrium, step, theta, scale, calls):
    field = _torch_field(model, calls, scale*torch.tanh(theta))
    state = torch.tensor(state, dtype=torch.float64)
    mass = torch.tensor(model.mass, dtype=torch.float64)
    output = [state]
    for interval in np.diff(time):
        count = max(1, int(np.ceil(interval/step - 1e-12)))
        h = interval/count
        for _ in range(count):
            a = field(state, power, equilibrium)
            b = field(state + h*a/2, power, equilibrium)
            c = field(state + h*b/2, power, equilibrium)
            d = field(state + h*c, power, equilibrium)
            state = state + h*(a + 2*b + 2*c + d)/6
            # Match the public NumPy RK4's COI projection at every update.
            q = state[:10] - (mass @ state[:10])/mass.sum()
            v = state[10:20] - (mass @ state[10:20])/mass.sum()
            state = torch.cat([q, v, state[20:]])
            if not torch.isfinite(state).all() or torch.max(torch.abs(state)) > 1e12:
                raise FloatingPointError("divergent state-port Torch rollout")
        output.append(state)
    return torch.stack(output)


def _algebra(models, states, power, eq, theta, scale):
    checks, energy_samples = [], []
    for name, model in models.items():
        for state_name, state in states.items():
            p = np.zeros(10) if state_name == "zero" else power
            identity = model.energy_identity(state, p, eq)
            v, z = state[10:20], state[20:]
            denominator = max(1., abs(v @ p) + abs(v @ model.damping @ v)
                              + abs(z @ model.latent_dissipation @ z))
            row = {"model": name, "state": state_name,
                   "energy_residual": identity["identity_error"]/denominator}
            energy_samples.append({"model": name, "state": state_name, **identity,
                                   "denominator": denominator})
            if name != "baseline":
                increment = model._increment(state[:10])
                pmat, qmat = model._projections
                left, right = -v @ pmat @ increment @ z, z @ increment.T @ qmat @ v
                row.update(power_residual=abs(left + right)/max(1., abs(left)+abs(right)),
                           balanced_port=_maximum(increment.sum(axis=0)),
                           common_angle_error=_maximum(model._increment(state[:10]+.7)-increment))
                if name in {"constant", "state"} and state_name != "zero":
                    field = _torch_field(model, model.calls, torch.tensor(model.delta))
                    torch_value = field(torch.tensor(state), power, eq).detach().numpy()
                    row["backend_rhs_error"] = _maximum(torch_value-model.rhs(state, power, eq))
            checks.append(row)
    model = models["state"]
    theta_neg = np.zeros((9, 2))
    theta_neg[0, 0] = .5
    bad_model = replace(model, delta=scale*np.tanh(theta_neg))
    pmat, qmat = model._projections
    q = qmat[:, 0]
    v, z = q.copy(), np.array([1., 0.])
    increment = bad_model._increment(q)
    # Removing the hidden-side reciprocal term leaves this uncancelled power.
    bad_power = abs(v @ pmat @ increment @ z)
    normalized_bad = bad_power/max(1., 2*bad_power)
    return {"checks": checks, "energy_samples": energy_samples,
            "negative_one_sided_residual": normalized_bad,
            "pass": all(all(row[key] <= 1e-11 for key in row
                            if key not in {"model", "state"}) for row in checks)
                    and bool(normalized_bad > 1e-10)}


def _jacobians(models, state, power, eq, theta, scale, calls):
    records = {}
    for kind in ("constant", "state"):
        model = models[kind]
        for zero in (False, True):
            parameter = np.zeros_like(theta) if zero else theta
            packed = parameter.ravel() if zero else np.r_[state, parameter.ravel()]

            def torch_map(vector, zero=zero, model=model):
                x = torch.tensor(state) if zero else vector[:22]
                th = vector if zero else vector[22:]
                field = _torch_field(model, calls, scale*torch.tanh(th.reshape(9, 2)))
                return field(x, power, eq)

            calls.add("jacobian")
            automatic = torch.autograd.functional.jacobian(
                torch_map, torch.tensor(packed, requires_grad=True), vectorize=False,
            ).detach().numpy()
            calls.finish("jacobian", None)
            finite, stencils = {}, {}
            for eps in (1e-5, 1e-6):
                columns, samples = [], []
                for index in range(len(packed)):
                    values = []
                    for sign in (1, -1):
                        moved = packed.copy()
                        moved[index] += sign*eps
                        x = state if zero else moved[:22]
                        th = moved if zero else moved[22:]
                        moved_model = replace(model, delta=scale*np.tanh(th.reshape(9, 2)))
                        values.append(moved_model.rhs(x, power, eq))
                    columns.append((values[0]-values[1])/(2*eps))
                    samples.append(values)
                finite[str(eps)] = np.column_stack(columns)
                stencils[str(eps)] = np.asarray(samples)
            records[kind + ("0" if zero else "")] = {
                "automatic": automatic, "finite_difference": finite,
                "finite_difference_samples": stencils,
                "comparisons": {eps: _gradient_comparison(automatic, value)
                                for eps, value in finite.items()},
            }
    model = models["state"]
    pmat, qmat = (torch.tensor(value) for value in model._projections)
    basis, mass = torch.tensor(model._basis), torch.tensor(model.mass)
    delta = torch.tensor(scale*np.tanh(theta))
    field = _torch_field(model, calls, delta)

    def detached_port(vector):
        q, v, z = vector[:10], vector[10:20], vector[20:]
        difference = pmat @ ((torch.sin(qmat @ q.detach())-torch.sin(qmat @ q))[:, None]
                             * (basis @ delta))
        correction = torch.cat([torch.zeros(10), -(pmat @ difference @ z)/mass,
                                difference.T @ qmat @ v])
        return field(vector, power, eq) + correction

    calls.add("jacobian")
    bad = torch.autograd.functional.jacobian(detached_port, torch.tensor(state, requires_grad=True))
    calls.finish("jacobian", None)
    expected = records["state"]["finite_difference"]["1e-06"][:, :22]
    negative = _gradient_comparison(bad.detach().numpy(), expected)
    return {"records": records, "negative_detached_port": negative,
            "negative_detached_jacobian": bad.detach().numpy(),
            "pass": all(row["pass"] for record in records.values()
                        for row in record["comparisons"].values())
                    and negative["error"] > 10*negative["tolerance"]}


def _paths(models, states, power, eq, theta, scale, calls, ops):
    paths, checks, trajectory_gradients = {}, {}, {}
    time = np.arange(301)*.01
    for name, model in models.items():
        roster = [("x1", .0025)] if name.endswith("0") else [
            (state, step) for state in ("x1", "x2") for step in (.005, .0025)]
        for state_name, step in roster:
            tag = f"{name}-{state_name}-{step}"
            path = ops.call("rollouts", lambda model=model, state_name=state_name, step=step: _numpy_path(
                model, states[state_name], time, power, eq, step), tag=tag)
            paths[tag] = path
            residual, energy_samples = [], []
            for point in path:
                row = model.energy_identity(point, power, eq)
                v, z = point[10:20], point[20:]
                scale_energy = max(1., abs(v@power)+abs(v@model.damping@v)
                                   + abs(z@model.latent_dissipation@z))
                residual.append(row["identity_error"]/scale_energy)
                energy_samples.append([row["energy_rate"], row["supply_minus_dissipation"], scale_energy])
            checks[tag] = {"energy_residual": max(residual),
                           "energy_samples": np.asarray(energy_samples),
                           "coi_error": max(_maximum(path[:, :10] @ model.mass),
                                            _maximum(path[:, 10:20] @ model.mass))}
    comparisons = {}
    for kind in ("constant", "state"):
        model = models[kind]
        torch_path = ops.call("rollouts", lambda model=model: _torch_path(
            model, states["x1"], time, power, eq, .0025, torch.tensor(theta), scale, calls).detach().numpy(),
            tag=kind+"-torch")
        paths[kind+"-torch"] = torch_path
        checks[kind+"-torch"] = {
            "coi_error": max(_maximum(torch_path[:, :10] @ model.mass),
                             _maximum(torch_path[:, 10:20] @ model.mass)),
        }
        comparisons[kind+"_torch_numpy"] = _maximum(torch_path-paths[kind+"-x1-0.0025"])
        comparisons[kind+"_zero_baseline"] = _maximum(
            paths[kind+"0-x1-0.0025"]-paths["baseline-x1-0.0025"])
    for name in ("baseline", "constant", "state"):
        for state in ("x1", "x2"):
            comparisons[name+"_"+state+"_step_refinement"] = _maximum(
                paths[f"{name}-{state}-0.005"]-paths[f"{name}-{state}-0.0025"])
    short_time = np.arange(11)*.01
    for kind in ("constant", "state"):
        model = models[kind]
        def path_gradient(model=model):
            th = torch.tensor(theta, requires_grad=True)
            path = _torch_path(model, states["x1"], short_time, power, eq, .005, th, scale, calls)
            calls.add("scalar_backward")
            loss = torch.mean(path[:, :20]**2)
            gradient = torch.autograd.grad(loss, th)[0].detach().numpy()
            calls.finish("scalar_backward", None)
            return {"gradient": gradient, "path": path.detach().numpy(), "loss": float(loss.detach())}

        nominal = ops.call("rollouts", path_gradient, tag=kind+"-gradient")
        automatic = nominal["gradient"]
        finite, stencils = {}, {}
        for eps in (1e-5, 1e-6):
            gradient = np.empty((9, 2))
            samples = []
            for index in np.ndindex(theta.shape):
                losses, short_paths = [], []
                for sign in (1, -1):
                    moved = theta.copy()
                    moved[index] += sign*eps
                    moved_model = replace(model, delta=scale*np.tanh(moved))
                    short = ops.call("rollouts", lambda moved_model=moved_model: _numpy_path(
                        moved_model, states["x1"], short_time, power, eq, .005), tag=kind+"-fd")
                    losses.append(float(np.mean(short[:, :20]**2)))
                    short_paths.append(short)
                gradient[index] = (losses[0]-losses[1])/(2*eps)
                samples.append({"index": list(index), "losses": losses, "paths": np.asarray(short_paths)})
            finite[str(eps)] = gradient
            stencils[str(eps)] = samples
        trajectory_gradients[kind] = {"automatic": automatic, "finite_difference": finite,
            "nominal": nominal, "finite_difference_samples": stencils,
            "comparisons": {eps: _gradient_comparison(automatic, value)
                            for eps, value in finite.items()}}
    passed = all(row.get("energy_residual", 0.) <= 1e-11 and row["coi_error"] <= 1e-10
                 for row in checks.values())
    for name, error in comparisons.items():
        tolerance = 1e-12 if name.endswith("baseline") else (1e-10 if name.endswith("numpy") else 1e-6)
        passed &= error <= tolerance
    passed &= all(row["pass"] for record in trajectory_gradients.values()
                  for row in record["comparisons"].values())
    passed &= all(np.isfinite(value).all() for value in paths.values())
    return {"paths": paths, "checks": checks, "comparisons": comparisons,
            "trajectory_gradients": trajectory_gradients, "pass": bool(passed)}


def execute(*, plan, root, result_dir):
    calls = Calls(plan["logical_operation_limits"])
    ops = Operations(result_dir / "journal", plan)
    records = {}
    try:
        with ops.scope("setup"):
            models, states, power, eq, theta, scale = fixture(calls)
            model_fields = ("mass", "edge_stiffness", "damping", "coupling", "latent_skew", "latent_dissipation")
            setup = ops.call("probe", lambda: {
                "states": states, "power": power, "equilibrium": eq, "theta": theta,
                "delta_scale": scale, "N": models["state"]._basis,
                "P": models["state"]._projections[0], "Q": models["state"]._projections[1],
                "models": {name: {**{key: getattr(model, key) for key in model_fields},
                                    **({"delta": model.delta, "port_kind": model.port_kind}
                                       if isinstance(model, StatePortClosure) else {})}
                           for name, model in models.items()},
            })
            save_record(result_dir, "inputs", setup)
        with ops.scope("algebra"):
            records["algebra"] = ops.call("probe", lambda: _algebra(models, states, power, eq, theta, scale))
            save_record(result_dir, "algebra", records["algebra"])
            if not records["algebra"]["pass"]:
                raise FloatingPointError("state-port algebra or negative-control gate failed")
        with ops.scope("jacobians"):
            records["jacobians"] = ops.call("probe", lambda: _jacobians(
                models, states["x1"], power, eq, theta, scale, calls))
            save_record(result_dir, "jacobians", records["jacobians"])
            if not records["jacobians"]["pass"]:
                raise FloatingPointError("state-port gradient or detached-port gate failed")
        with ops.scope("paths"):
            records["paths"] = _paths(models, states, power, eq, theta, scale, calls, ops)
            save_record(result_dir, "paths", records["paths"])
            if not records["paths"]["pass"]:
                raise FloatingPointError("state-port path or propagation-gradient gate failed")
        expected = plan["expected_logical_operations"]
        if (dict(calls.counts) != expected or dict(calls.returned) != expected
                or dict(ops.counts) != plan["operator_limits"] or ops.counts != ops.returns):
            raise RuntimeError(f"state-port fixed call tree differs: {dict(calls.counts)}, {dict(ops.counts)}")
        summary = {"process_status": "PASS", "correctness_pass": all(r["pass"] for r in records.values()),
            "gates": {name: record["pass"] for name, record in records.items()},
            "algebra": records["algebra"], "path_comparisons": records["paths"]["comparisons"],
            "logical_operations": dict(calls.counts), "operator_counts": dict(ops.counts),
            "logical_returned": dict(calls.returned),
            "real_data_reads": 0, "fit_attempts": 0, "optimizer_updates": 0,
            "promotion_decision": "KEEP_BASELINE", "source_commit": git_head(root),
            "runtime": {"python": sys.version, "executable": sys.executable,
                        **{name: version(name) for name in ("numpy", "scipy", "torch")}},
            "claim_boundary": "Finite synthetic correctness of paired ports and gradients; no learned or real-data performance claim."}
        write_json(result_dir / "summary.json", summary)
        write_json(result_dir / "stage_evidence.json", {"status": "PASS", "process_status": "PASS",
            "source_commit": git_head(root), "data_policy": "analytic_synthetic_only",
            "promotion_decision": "KEEP_BASELINE", "claim_boundary": summary["claim_boundary"]})
    except BaseException as error:
        save_record(result_dir, "failed-operation", {"last_rhs": calls.last_rhs,
            "attempted": dict(calls.counts), "returned": dict(calls.returned)}, failed=True)
        write_json(result_dir / "failure.json", {"error_type": type(error).__name__, "message": str(error),
            "completed_scopes": list(records), "logical_operations": dict(calls.counts), "retry_allowed": False})
        raise
    finally:
        ops.journal.close()
    write_json(result_dir / "artifact_hashes.json", {
        p.relative_to(result_dir).as_posix(): sha256_file(p)
        for p in sorted(result_dir.rglob("*")) if p.is_file()})
    return summary
