"""Finite known-truth observation study; invoked only by its registered workflow."""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from contextlib import contextmanager
from importlib.metadata import version

import numpy as np
from scipy.linalg import expm

from kron_reduce.evaluation.scoped_audit_journal import ScopedAuditJournal
from kron_reduce.models.nonlinear_latent import NonlinearPassiveClosure, complete_incidence
from kron_reduce.physics.coordinates import coi_project
from project_harness.evidence import sha256_file
from project_harness.git_state import git_head


def write_json(path, value):
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def save_record(directory, name, record, *, failed=False):
    arrays = {}

    def pack(value, key):
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject or (not failed and not np.isfinite(value).all()):
                raise ValueError("evidence arrays must be finite and numeric")
            arrays[key] = value
            return {"array": key, "shape": list(value.shape), "dtype": value.dtype.str}
        if isinstance(value, dict):
            return {str(k): pack(v, key + "__" + str(k)) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [pack(v, key + "__" + str(i)) for i, v in enumerate(value)]
        value = value.item() if isinstance(value, np.generic) else value
        if failed and isinstance(value, float) and not np.isfinite(value):
            return {"nonfinite_scalar": str(value)}
        return value

    metadata = pack(record, name)
    with (directory / f"{name}.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    write_json(directory / f"{name}.json", metadata)


class Operations:
    """Shared global limits plus durable, independently capped case scopes."""

    def __init__(self, directory, plan):
        self.journal = ScopedAuditJournal(directory)
        self.limits = plan["operator_limits"]
        self.caps = plan["scope_limits"]
        self.counts = Counter()
        self.returns = Counter()
        self.index = 0
        self.scope_id = None
        self.partial = {}
        self.last_rhs = None
        self.clock_checks = []
        self.rhs_context = {}

    @contextmanager
    def scope(self, name):
        self.scope_id = name
        self.partial, self.last_rhs = {}, None
        self.journal.begin(name.replace("_", "-"), self.caps[name])
        try:
            yield
        except BaseException as error:
            # Preserve returned arrays and the last RHS state without new science.
            try:
                save_record(self.journal.directory.parent, f"partial-{name}", {
                    "returned_operations": self.partial, "last_rhs": self.last_rhs,
                    "error_type": type(error).__name__, "message": str(error),
                }, failed=True)
            except (OSError, ValueError, TypeError) as save_error:
                error.add_note(f"Partial evidence could not be saved: {save_error}")
            # An interrupted or poisoned scope remains reserved, never reusable.
            if not self.journal.poisoned and not self.journal.pending:
                self.journal.finish({"status": "failed"})
            raise
        else:
            self.journal.finish({"status": "completed_within_budget"})
        finally:
            self.scope_id = None

    def call(self, category, function, *, tag=""):
        if self.scope_id is None or self.counts[category] >= self.limits.get(category, 0):
            raise RuntimeError(f"unreserved or exhausted operation: {category}")
        self.index += 1
        row = {"category": category, "index": self.index, "tag": tag}
        if category in ("integration_rhs", "physical_rhs"):
            row.update(category="rhs", seed=0, event_id=self.scope_id, step=0.0,
                       phase=("physical_checks" if category == "physical_rhs"
                              else "history_integration"), sequence=self.index)
            if category == "integration_rhs":
                row.update(self.rhs_context)
        elif category == "rollouts":
            row.update(category="rollout", seed=0, event_id=str(self.index), step=0.0)
            if self.rhs_context:
                row.update(self.rhs_context)
        self.journal.append({**row, "transition": "attempted"})
        self.counts[category] += 1
        try:
            value = function()
        except BaseException as error:
            self.journal.append({**row, "transition": "failed",
                                 "error_type": type(error).__name__})
            raise
        if category == "integration_rhs":
            self.last_rhs["returned"] = value
        else:
            self.partial[str(row["index"])] = {
                "category": category, "tag": tag,
                "value": model_parameters(value) if isinstance(value, NonlinearPassiveClosure)
                else value,
            }
        self.journal.append({**row, "transition": "returned"})
        self.returns[category] += 1
        return value


def helmert():
    basis = np.zeros((10, 9))
    for k in range(1, 10):
        basis[:k, k - 1] = 1 / np.sqrt(k * (k + 1))
        basis[k, k - 1] = -k / np.sqrt(k * (k + 1))
    return basis


def model_parameters(model):
    return {name: getattr(model, name) for name in (
        "mass", "edge_stiffness", "damping", "coupling", "latent_skew",
        "latent_dissipation", "backbone")}


def make_model(base, ops, *, weak=False, mismatch=False, sine=False):
    n = helmert()
    mass = np.asarray(base["mass"], dtype=float)
    b = complete_incidence(10)
    k = np.full(45, base["edge_stiffness"])
    if mismatch:
        k[0] += base["stiffness_mismatch"]
    c = n[:, :2] * np.asarray(base["port_gains"])
    if weak:
        c *= base["weak_port_factor"]
    rate = base["latent_rate"]
    model = ops.call("probe", lambda: NonlinearPassiveClosure(
        mass, k, base["damping_gain"] * (b @ b.T), c,
        np.array([[0., -rate], [rate, 0.]]), np.diag(base["latent_decay"]),
        "sine" if sine else "linear",
    ), tag="model_construction_with_parameter_validation")
    power = np.asarray(base["power"], dtype=float)
    power -= mass * np.sum(power) / np.sum(mass)
    initial = np.r_[coi_project(base["q_initial"], mass),
                    coi_project(base["v_initial"], mass), base["z_initial"]]
    return model, power, initial


def generator(model, power):
    b = complete_incidence(10)
    p = np.eye(10) - np.outer(model.mass, np.ones(10))/np.sum(model.mass)
    g = np.zeros((23, 23))
    g[:10, 10:20] = np.eye(10)
    g[10:20, :10] = -(p @ (b * model.edge_stiffness) @ b.T) / model.mass[:, None]
    g[10:20, 10:20] = -(p @ model.damping) / model.mass[:, None]
    g[10:20, 20:22] = -(p @ model.coupling) / model.mass[:, None]
    g[10:20, 22] = (p @ power) / model.mass
    g[20:22, 10:20] = model.coupling.T
    g[20:22, 20:22] = model.latent_skew - model.latent_dissipation
    return g


def exact_path(transition, initial, intervals, ops):
    def propagate():
        state = np.r_[initial, 1.]
        values = [state[:22].copy()]
        for _ in range(intervals):
            state = transition @ state
            values.append(state[:22].copy())
        return np.stack(values)
    return ops.call("rollouts", propagate, tag="affine_exponential_recurrence")


def check_path(model, power, path, ops, tolerances):
    if not np.isfinite(path).all():
        raise FloatingPointError("nonfinite trajectory")
    coi = max(float(np.max(np.abs(path[:, :10] @ model.mass))),
              float(np.max(np.abs(path[:, 10:20] @ model.mass))))
    physics = []
    for index in sorted({0, (len(path)-1) // 2, len(path) - 1}):
        physics.append(ops.call("physical_rhs", lambda index=index:
            model.energy_identity(path[index], power, np.zeros(10)),
            tag="stored_state_energy_identity"))
    error = max(row["identity_error"] / max(
        1., abs(row["energy_rate"]), abs(row["supply_minus_dissipation"])) for row in physics)
    coi_scaled = max(
        float(np.max(np.abs(path[:, start:start+10] @ model.mass) /
                     np.maximum(1., np.linalg.norm(model.mass) *
                                np.linalg.norm(path[:, start:start+10], axis=1))))
        for start in (0, 10))
    if coi_scaled > tolerances["coi"] or error > tolerances["energy_identity"]:
        raise FloatingPointError("COI or sampled energy identity gate failed")
    return {"coi_max": coi, "coi_scaled_max": coi_scaled,
            "energy_identity_scaled_max": error, "checks": physics}


def sampled_operator(transition, intervals, ops):
    return ops.call("probe", lambda: _sampled_operator(transition, intervals),
                    tag="complete_sampled_map")


def _sampled_operator(transition, intervals):
    p = np.eye(23)
    rows = []
    for _ in range(intervals):
        p = transition @ p
        rows.append(p[:20, 20:22].copy())
    return np.vstack(rows) / np.sqrt(20 * intervals)


def decompose(matrix, ops, natural=0.):
    u, s, vh = ops.call("svd", lambda: np.linalg.svd(matrix, full_matrices=False))
    tol = 64 * np.finfo(float).eps * max(matrix.shape) * max(float(s[0]), natural)
    rank = int(np.sum(s > tol))
    return {"u": u, "s": s, "vh": vh, "rank": rank, "threshold": tol,
            "condition": float(s[0] / s[-1]) if rank == matrix.shape[1] else None,
            "inverse_sigma_min": float(1/s[-1]) if s[-1] > 0 else None}


def solve(system, right, ops):
    if system["rank"] < len(system["s"]):
        return None
    return ops.call("ls", lambda:
        system["vh"].T @ ((system["u"].T @ right) / system["s"]))


def integral_system(model, q, v, power, ops):
    """A new linear/sine synthetic estimator; frozen A14 code stays untouched."""
    intervals = len(q) - 1
    h, duration = .01, .01 * intervals
    mass = model.mass
    p = np.eye(10) - np.outer(mass, np.ones(10)) / np.sum(mass)
    a = model.latent_skew - model.latent_dissipation
    pc = p @ model.coupling
    aug = np.zeros((8, 8))
    aug[:2, :2] = a
    aug[:2, 2:4] = aug[2:4, 4:6] = aug[4:6, 6:8] = np.eye(2)
    block = ops.call("expm", lambda: expm(h * aug))
    phi, f1, f2, f3 = block[:2, :2], block[:2, 2:4], block[:2, 4:6], block[:2, 6:8]
    prefix, w = np.eye(2), np.zeros(2)
    lb, gs2, gs4, corrections, wn = [], [], [], [], [w.copy()]
    b = complete_incidence(10)
    root = np.sqrt(6 / 5)
    nodes4 = np.array([-np.sqrt((3+2*root)/7), -np.sqrt((3-2*root)/7),
                       np.sqrt((3-2*root)/7), np.sqrt((3+2*root)/7)])
    weights4 = np.array([18-np.sqrt(30), 18+np.sqrt(30),
                         18+np.sqrt(30), 18-np.sqrt(30)]) / 36
    rules = [(np.array([-1., 1.]) / np.sqrt(3), np.ones(2), gs2),
             (nodes4, weights4, gs4)]
    for j in range(intervals):
        lb.append(pc @ prefix @ f1)
        prefix = phi @ prefix
        u = model.coupling.T @ v[j]
        slope = model.coupling.T @ (v[j+1] - v[j]) / h
        corrections.append(pc @ (f1 @ w + f2 @ u + f3 @ slope))
        w = phi @ w + f1 @ u + f2 @ slope
        wn.append(w.copy())
        for nodes, weights, target in rules:
            force = np.zeros(10)
            for node, weight in zip(nodes, weights, strict=True):
                point = q[j] + (node+1)/2 * (q[j+1]-q[j])
                def evaluate(point=point):
                    edge = b.T @ point
                    shape = np.sin(edge) if model.backbone == "sine" else edge
                    return b @ (model.edge_stiffness * shape)
                force += h/2 * weight * ops.call("force", evaluate)
            target.append(p @ (h*power - force - h/2*model.damping @ (v[j]+v[j+1]))
                          - mass*(v[j+1]-v[j]))
    lb, correction = np.stack(lb), np.stack(corrections)
    scale = 1 / mass / np.sqrt(duration*h)
    matrix = (scale[None, :, None] * lb).reshape(-1, 2)
    decomposition = decompose(matrix, ops, float(np.linalg.norm(pc / mass[:, None])))
    pc_svd = decompose(pc, ops)
    active = pc_svd["u"][:, :pc_svd["rank"]]
    force_projection = active @ active.T
    right2 = ((np.stack(gs2)-correction) * scale).reshape(-1)
    right4 = ((np.stack(gs4)-correction) * scale).reshape(-1)
    z2, z4 = solve(decomposition, right2, ops), solve(decomposition, right4, ops)
    if model.backbone == "linear":
        if np.max(np.abs(right2-right4)) > 1e-10 * max(1., np.linalg.norm(right4)):
            raise FloatingPointError("linear Gauss quadratures disagree")
        if z4 is not None and np.max(np.abs(z2-z4)) > 1e-10 * max(1., np.linalg.norm(z4)):
            raise FloatingPointError("linear Gauss initial states disagree")
    return {"matrix": matrix, "raw_matrix": lb.reshape(-1, 2),
            "right2": right2, "right4": right4, "z2": z2, "z4": z4,
            "g4": np.stack(gs4), "pc": pc, "scale": scale, "w_nodes": np.stack(wn),
            "convolution_integral_correction": correction,
            "g2": np.stack(gs2), "hidden_convolution_blocks": block,
            "phi": prefix, "svd": decomposition, "pc_svd": pc_svd,
            "force_projection": force_projection,
            "force_residual": np.stack(gs4) @ (np.eye(10)-force_projection).T,
            "dynamic_residual_scaled": None if z4 is None else right4-matrix@z4,
            "dynamic_residual_physical": None if z4 is None else
                np.stack(gs4)-correction-lb@z4,
            "zT_observed": None if z4 is None else prefix@z4+w,
            "kinematic_residual": np.diff(q, axis=0) - h/2*(v[:-1]+v[1:])}


def add_noise(observed, model, mode, decomposition, noise):
    values = observed.copy()
    if mode == "alternating":
        for column, amplitude, node in ((0, noise["q"], 2), (10, noise["v"], 3)):
            direction = coi_project(np.eye(10)[node], model.mass)
            direction /= np.linalg.norm(direction)
            values[:, column:column+10] += (
                (-1.)**np.arange(len(values))[:, None] * amplitude * direction)
    elif mode == "weak_direction":
        u = decomposition["u"][:, -1].copy()
        vector = decomposition["vh"][-1]
        if vector[np.argmax(np.abs(vector))] < 0:
            u *= -1
        values[1:] += (noise["weak_rms"] * np.sqrt(values[1:].size) * u).reshape(-1, 20)
    elif mode != "clean":
        raise ValueError("unregistered noise")
    return values


def errors(estimate, truth, ops):
    return ops.call("probe", lambda: _errors(estimate, truth), tag="future_metric_evaluation")


def _errors(estimate, truth):
    output = {}
    for horizon in (3, 8):
        delta = estimate[1:100*horizon+1, :20] - truth[1:100*horizon+1, :20]
        output[str(horizon)] = {
            "visible_max_abs": float(np.max(np.abs(delta))),
            "visible_mse": float(np.mean(delta**2)),
            "visible_rms": float(np.sqrt(np.mean(delta**2))),
            "q_max_abs": float(np.max(np.abs(delta[:, :10]))),
            "v_max_abs": float(np.max(np.abs(delta[:, 10:]))),
            "q_rms": float(np.sqrt(np.mean(delta[:, :10]**2))),
            "v_rms": float(np.sqrt(np.mean(delta[:, 10:]**2))),
        }
    return output


def rk_path(model, power, initial, intervals, step, ops, *, role, method):
    clock = np.arange(intervals+1)*.01
    dispatch_step = step*(1+1e-10)
    expected = round(.01/step)
    counts = np.ceil(np.diff(clock)/dispatch_step-1e-12).astype(int)
    if not np.all(counts == expected):
        raise RuntimeError("fixed RK4 substep count does not match public rollout clock")
    ops.clock_checks.append({
        "scope": ops.scope_id, "intervals": intervals, "nominal_step": step,
        "path_role": role, "method": method,
        "dispatch_max_step": dispatch_step, "substeps_per_interval": expected,
        "actual_step_min": float(np.min(np.diff(clock)/counts)),
        "actual_step_max": float(np.max(np.diff(clock)/counts)),
        "rk4_updates": int(np.sum(counts)),
    })
    original = model.rhs
    calls = 0
    ops.rhs_context = {"path_role": role, "method": method, "step": step,
                       "dispatch_max_step": dispatch_step}
    def counted(*args):
        nonlocal calls
        ops.last_rhs = {"state": np.array(args[0], copy=True)}
        ops.rhs_context["phase"] = (
            "future_integration" if role == "future" or (role == "truth" and calls >= 20*expected*4)
            else "history_integration")
        calls += 1
        return ops.call("integration_rhs", lambda: original(*args))
    object.__setattr__(model, "rhs", counted)
    try:
        return ops.call("rollouts", lambda: model.rollout(
            initial[:10], initial[10:20], initial[20:], clock,
            power, np.zeros(10), max_step=dispatch_step), tag="shared_COI_RK4")
    finally:
        object.__delattr__(model, "rhs")
        ops.rhs_context = {}


def linear_bundle(model, power, ops, tolerance):
    g = ops.call("probe", lambda: generator(model, power), tag="affine_generator")
    transition = ops.call("expm", lambda: expm(.01*g))
    endpoints = {count: ops.call("expm", lambda count=count: expm(.01*count*g))
                 for count in (2, 5, 10, 20, 800, 820)}
    running = np.eye(23)
    checks = {}
    special = np.r_[coi_project(np.eye(10)[2], model.mass),
                    coi_project(np.eye(10)[3], model.mass), 1., -1.]
    default = np.r_[coi_project([.3, -.2, .1, -.05, 0, 0, 0, 0, 0, 0], model.mass),
                    coi_project([.12, -.06, .04, 0, 0, 0, 0, 0, 0, 0], model.mass),
                    .6, -.4]
    for index, state in enumerate((np.zeros(22), default, special)):
        rhs = ops.call("physical_rhs", lambda state=state:
                       model.rhs(state, power, np.zeros(10)), tag="generator_RHS_check")
        error = float(np.max(np.abs((g@np.r_[state, 1.])[:22]-rhs)))
        if error > tolerance * max(1., np.linalg.norm(rhs)):
            raise FloatingPointError("affine generator and common RHS disagree")
        checks[f"rhs-{index}"] = error
    for count in range(1, 821):
        running = transition @ running
        if count in (20, 820):
            difference = float(np.max(np.abs(running - endpoints[count])))
            bound = tolerance * max(1., float(np.linalg.norm(endpoints[count])))
            if difference > bound:
                raise FloatingPointError("affine exponential recurrence verification failed")
            checks[str(count)] = difference
    return {"generator": g, "transition": transition, "endpoints": endpoints,
            "recurrence_errors": checks}


def checked_exact(bundle, initial, count, model, power, ops, tolerances):
    path = exact_path(bundle["transition"], initial, count, ops)
    direct = (bundle["endpoints"][count] @ np.r_[initial, 1.])[:22]
    error = float(np.max(np.abs(direct-path[-1])))
    if error > tolerances["exponential"] * max(1., float(np.linalg.norm(direct))):
        raise FloatingPointError("direct endpoint differs from affine recurrence")
    physical = check_path(model, power, path, ops, tolerances)
    return path, {"direct_endpoint_error": error, **physical}


def continuous_gram(model, integral, intervals, ops, tolerance):
    a = model.latent_skew - model.latent_dissipation
    f = integral["pc"] / model.mass[:, None]
    block = np.block([[-a.T, f.T @ f], [np.zeros((2, 2)), a]])
    exponential = ops.call("expm", lambda: expm(.01*intervals*block))
    gram = integral["phi"].T @ exponential[:2, 2:] / (.01*intervals)
    if np.max(np.abs(gram-gram.T)) > tolerance * max(1., np.linalg.norm(gram)):
        raise FloatingPointError("continuous Gram symmetry failed")
    discrete = integral["matrix"].T @ integral["matrix"]
    difference = gram - discrete
    eigenvalues = ops.call("probe", lambda: np.linalg.eigvalsh((difference+difference.T)/2),
                           tag="Gram_difference_symmetric_eigenvalues")
    if eigenvalues[0] < -tolerance * max(1., np.linalg.norm(gram), np.linalg.norm(discrete)):
        raise FloatingPointError("continuous/discrete Gram inequality failed")
    return {"continuous_time_average": gram, "discrete": discrete,
            "difference_eigenvalues": eigenvalues}


def linear_case(model, power, bundle, truth, intervals, noise_mode, ops, plan):
    tol = plan["tolerances"]
    observed_truth = truth[20-intervals:21, :20]
    operator = sampled_operator(bundle["transition"], intervals, ops)
    decomposition = decompose(operator, ops, natural=1.)
    observed = add_noise(observed_truth, model, noise_mode, decomposition, plan["noise"])
    zero_path, zero_checks = checked_exact(bundle, np.r_[observed[0], 0., 0.],
                                           intervals, model, power, ops, tol)
    right = (observed[1:] - zero_path[1:, :20]).reshape(-1) / np.sqrt(20*intervals)
    exact_z = solve(decomposition, right, ops)
    integral = integral_system(model, observed[:, :10], observed[:, 10:], power, ops)
    gram = continuous_gram(model, integral, intervals, ops, tol["algebra"])
    true_z = truth[20-intervals, 20:]
    sampled_perturbation = right - operator @ true_z
    checks = {"right_perturbation_norm": float(np.linalg.norm(sampled_perturbation))}
    matched = not np.isclose(model.edge_stiffness[0], plan["base"]["edge_stiffness"] +
                              plan["base"]["stiffness_mismatch"])
    if (exact_z is not None and matched and noise_mode == "clean"
            and np.linalg.norm(exact_z-true_z) > tol["exact_recovery"]):
        raise FloatingPointError("clean exact sampled observer failed")
    if exact_z is not None and noise_mode == "weak_direction":
        direction = decomposition["vh"][-1].copy()
        if direction[np.argmax(np.abs(direction))] < 0:
            direction *= -1
        expected = plan["noise"]["weak_rms"] * direction / decomposition["s"][-1]
        expected_right = operator @ expected
        right_error = float(np.linalg.norm(sampled_perturbation-expected_right))
        if right_error > tol["algebra"]:
            raise FloatingPointError("weak-direction actual sampled right perturbation failed")
        response_error = float(np.linalg.norm(exact_z-true_z-expected))
        if response_error > tol["noise_response"] * max(1., np.linalg.norm(expected)):
            raise FloatingPointError("weak direction exact observer response failed")
        checks.update(expected_noise_response=expected, noise_response_error=response_error,
                      expected_right_noise=expected_right, right_noise_error=right_error)
    paths, metrics = {}, {}
    for method, z in (("sampled_ls", exact_z), ("integral_ls", integral["z4"]),
                      ("true_z_reference", true_z)):
        if z is None:
            metrics[method] = {"status": "unresolved"}
            continue
        history, history_check = checked_exact(bundle, np.r_[observed[0], z], intervals,
                                               model, power, ops, tol)
        reset = np.r_[observed[-1], history[-1, 20:]]
        future, future_check = checked_exact(bundle, reset, 800, model, power, ops, tol)
        z_error = float(np.linalg.norm(z-true_z))
        future_errors = errors(future, truth[20:], ops)
        threshold = tol["state_clean"] if noise_mode == "clean" else tol["state_noisy"]
        future_threshold = tol["future_clean"] if noise_mode == "clean" else tol["future_noisy"]
        metrics[method] = {
            "status": "resolved", "hidden_initial_error": z_error,
            "hidden_origin_error": float(np.linalg.norm(reset[20:]-truth[20, 20:])),
            "hidden_initial_energy": float(z@z/2),
            "hidden_initial_energy_error": float((z@z-true_z@true_z)/2),
            "hidden_origin_energy": float(reset[20:]@reset[20:]/2),
            "hidden_origin_energy_error": float((reset[20:]@reset[20:] -
                                                  truth[20, 20:]@truth[20, 20:])/2),
            "history_visible_mse": float(np.mean((history[1:, :20]-observed_truth[1:])**2)),
            "state_recovered": z_error <= threshold,
            "future_3s": future_errors["3"]["visible_max_abs"] <= future_threshold,
            "future_8s": future_errors["8"]["visible_max_abs"] <= future_threshold,
            "future_within_tolerance": all(
                row["visible_max_abs"] <= future_threshold for row in future_errors.values()),
            "future_errors": future_errors, "history_checks": history_check,
            "future_checks": future_check,
        }
        paths[method] = {"z": z, "history": history, "reset": reset, "future": future}
        energies = [ops.call("probe", lambda state=state:
                            model.energy(state, np.zeros(10)), tag="reset_energy")
                    for state in (history[-1], reset)]
        paths[method]["reset_energy"] = energies
        paths[method]["reset_energy_change"] = energies[1]-energies[0]
    for method in paths:
        metrics[method]["future_relative_to_true_z_reference"] = errors(
            paths[method]["future"], paths["true_z_reference"]["future"], ops)
    return {"observed": observed, "truth_history": truth[20-intervals:21],
            "truth_future": truth[20:], "operator": operator, "operator_svd": decomposition,
            "sampled_right": right, "sampled_right_perturbation": sampled_perturbation,
            "matched_model": matched,
            "observation_noise": observed-observed_truth,
            "zero_path": zero_path, "zero_checks": zero_checks,
            "integral": integral, "gram": gram, "paths": paths, "metrics": metrics,
            "exact_observer_checks": checks,
            "observation_identifiability": "known_linear_unique" if decomposition["rank"] == 2
            else "unresolved"}


def sine_case(model, power, truth, mode, ops, plan):
    observed_truth = truth[10:21, :20]
    observed = add_noise(observed_truth, model, mode, None, plan["noise"])
    integral = integral_system(model, observed[:, :10], observed[:, 10:], power, ops)
    gram = continuous_gram(model, integral, 10, ops, plan["tolerances"]["algebra"])
    true_z = truth[10, 20:]
    tol = plan["tolerances"]
    paths, metrics = {}, {}
    for method, z in (("integral_ls", integral["z4"]), ("true_z_reference", true_z)):
        if z is None:
            metrics[method] = {"status": "unresolved"}
            continue
        candidates = []
        for step in plan["sine_steps"]:
            history = rk_path(model, power, np.r_[observed[0], z], 10, step, ops,
                              role="history", method=method)
            history_check = check_path(model, power, history, ops, tol)
            reset = np.r_[observed[-1], history[-1, 20:]]
            future = rk_path(model, power, reset, 800, step, ops, role="future", method=method)
            future_check = check_path(model, power, future, ops, tol)
            energies = [ops.call("probe", lambda state=state:
                                model.energy(state, np.zeros(10)), tag="reset_energy")
                        for state in (history[-1], reset)]
            candidates.append({"step": step, "history": history, "reset": reset,
                               "future": future, "history_checks": history_check,
                               "future_checks": future_check, "reset_energy": energies,
                               "reset_energy_change": energies[1]-energies[0]})
        refinement = max(float(np.max(np.abs(candidates[0][key]-candidates[1][key])))
                         for key in ("history", "future"))
        if refinement > tol["refinement"]:
            raise FloatingPointError("nonlinear candidate refinement gate failed")
        fine = candidates[-1]
        future_errors = errors(fine["future"], truth[20:], ops)
        z_error = float(np.linalg.norm(z-true_z))
        threshold = tol["state_clean"] if mode == "clean" else tol["state_noisy"]
        future_threshold = tol["future_clean"] if mode == "clean" else tol["future_noisy"]
        metrics[method] = {
            "status": "resolved", "hidden_initial_error": z_error,
            "hidden_origin_error": float(np.linalg.norm(fine["reset"][20:]-truth[20, 20:])),
            "hidden_initial_energy": float(z@z/2),
            "hidden_initial_energy_error": float((z@z-true_z@true_z)/2),
            "hidden_origin_energy": float(fine["reset"][20:]@fine["reset"][20:]/2),
            "hidden_origin_energy_error": float((fine["reset"][20:]@fine["reset"][20:] -
                                                  truth[20, 20:]@truth[20, 20:])/2),
            "state_recovered": z_error <= threshold,
            "future_3s": future_errors["3"]["visible_max_abs"] <= future_threshold,
            "future_8s": future_errors["8"]["visible_max_abs"] <= future_threshold,
            "future_within_tolerance": all(
                row["visible_max_abs"] <= future_threshold for row in future_errors.values()),
            "future_errors": future_errors, "refinement_max": refinement,
        }
        paths[method] = candidates
    for method in paths:
        metrics[method]["future_relative_to_true_z_reference"] = errors(
            paths[method][-1]["future"], paths["true_z_reference"][-1]["future"], ops)
    return {"observed": observed, "truth_history": truth[10:21], "truth_future": truth[20:],
            "integral": integral, "gram": gram, "paths": paths, "metrics": metrics,
            "observation_noise": observed-observed_truth,
            "observation_identifiability": "nonlinear_not_certified"}


def alias_case(ops, plan):
    n, rate = helmert()[:, :2], 2*np.pi/.01
    model = ops.call("probe", lambda: NonlinearPassiveClosure(
        np.ones(10), np.zeros(45), np.zeros((10, 10)), rate*n,
        np.zeros((2, 2)), np.zeros((2, 2)), "linear"))
    z1, z2 = np.array([.6, -.4]), np.array([1.6, -.4])
    def analytic(times, z):
        hidden = np.cos(rate*times)[:, None] * z
        visible_v = -np.sin(rate*times)[:, None] * (n@z)
        visible_q = (np.cos(rate*times)-1)[:, None] * (n@z)/rate
        return np.c_[visible_q, visible_v, hidden]
    history_times, future_times = np.arange(11)*.01, np.arange(801)*.01+.1
    history1, history2 = analytic(history_times, z1), analytic(history_times, z2)
    future1, future2 = analytic(future_times, z1), analytic(future_times, z2)
    off_times = .1 + np.array([.0025, 2.9975, 7.9975])
    off1, off2 = analytic(off_times, z1), analytic(off_times, z2)
    # The protocol gives the estimator mathematical zeros at these alias nodes.
    observed = np.zeros((11, 20))
    integral = integral_system(model, observed[:, :10], observed[:, 10:],
                               np.zeros(10), ops)
    gram = continuous_gram(model, integral, 10, ops, plan["tolerances"]["algebra"])
    grid_gap = float(np.max(np.abs(np.r_[history1[:, :20]-history2[:, :20],
                                          future1[:, :20]-future2[:, :20]])))
    off_gap = float(np.max(np.abs(off1[:, :20]-off2[:, :20])))
    if grid_gap > plan["tolerances"]["alias"] or off_gap < .1:
        raise FloatingPointError("analytic alias witness failed")
    if integral["svd"]["rank"] != 2 or np.linalg.norm(integral["z4"]) > 1e-12:
        raise FloatingPointError("alias integral equation must be uniquely zero")
    speed_gaps = np.linalg.norm(off1[:, 10:20]-off2[:, 10:20], axis=1)
    if np.max(np.abs(speed_gaps-1)) > plan["tolerances"]["alias"]:
        raise FloatingPointError("off-grid alias velocity separation failed")
    rhs_checks = []
    for time, z in ((0., z1), (off_times[0], z1), (off_times[-1], z2)):
        state = analytic(np.array([time]), z)[0]
        derivative = np.r_[state[10:20], -rate*np.cos(rate*time)*(n@z),
                           -rate*np.sin(rate*time)*z]
        actual = ops.call("physical_rhs", lambda state=state:
                         model.rhs(state, np.zeros(10), np.zeros(10)))
        error = float(np.max(np.abs(actual-derivative)))
        if error > plan["tolerances"]["algebra"] * max(1., np.linalg.norm(derivative)):
            raise FloatingPointError("analytic alias derivatives differ from common RHS")
        rhs_checks.append(error)
    checks1 = check_path(model, np.zeros(10), history1, ops, plan["tolerances"])
    checks2 = check_path(model, np.zeros(10), history2, ops, plan["tolerances"])
    remaining_checks = [check_path(model, np.zeros(10), path, ops, plan["tolerances"])
                        for path in (future1, future2, off1, off2)]
    return {"z1": z1, "z2": z2, "history1": history1, "history2": history2,
            "parameters": model_parameters(model), "remaining_checks": remaining_checks, "gram": gram,
            "future1": future1, "future2": future2, "off_times": off_times,
            "off1": off1, "off2": off2, "observed": observed, "integral": integral,
            "operator": np.zeros((200, 2)), "candidate_history": np.zeros((11, 22)),
            "candidate_reset": np.zeros(22), "candidate_future": np.zeros((801, 22)),
            "candidate_off_grid": np.zeros((3, 22)), "rhs_checks": rhs_checks,
            "checks1": checks1, "checks2": checks2,
            "observation_identifiability": "future_relevant_alias_counterexample",
            "metrics": {"hidden_unique": False, "future_grid_equivalent": True,
                        "off_grid_future_equivalent": False, "grid_gap": grid_gap,
                        "off_grid_gap": off_gap, "integral_rank": 2,
                        "off_grid_speed_norms": speed_gaps.tolist()}}


def hidden_null_case(base, ops, plan):
    model, power, initial = make_model(base, ops)
    model = ops.call("probe", lambda: NonlinearPassiveClosure(
        model.mass, model.edge_stiffness, model.damping,
        np.c_[model.coupling[:, 0], np.zeros(10)], np.zeros((2, 2)),
        model.latent_dissipation, "linear"))
    bundle = linear_bundle(model, power, ops, plan["tolerances"]["exponential"])
    first, checks1 = checked_exact(bundle, initial, 820, model, power, ops, plan["tolerances"])
    other = initial.copy()
    other[-1] = .6
    second, checks2 = checked_exact(bundle, other, 820, model, power, ops, plan["tolerances"])
    operator = sampled_operator(bundle["transition"], 10, ops)
    decomposition = decompose(operator, ops, natural=1.)
    integral = integral_system(model, first[10:21, :10], first[10:21, 10:20], power, ops)
    gram = continuous_gram(model, integral, 10, ops, plan["tolerances"]["algebra"])
    gap = float(np.max(np.abs(first[:, :20]-second[:, :20])))
    if gap > plan["tolerances"]["algebra"] or decomposition["rank"] != 1:
        raise FloatingPointError("future-null hidden witness failed")
    if integral["z4"] is not None or integral["svd"]["rank"] != 1:
        raise FloatingPointError("rank-deficient estimator must report unresolved")
    return {"first": first, "second": second, "initial1": initial, "initial2": other,
            "operator": operator, "operator_svd": decomposition, "integral": integral,
            "parameters": model_parameters(model), "power": power, "gram": gram,
            "bundle": bundle, "checks1": checks1, "checks2": checks2,
            "observation_identifiability": "hidden_nonunique_visible_future_equivalent",
            "metrics": {"hidden_unique": False, "visible_future_equivalent": True,
                        "visible_gap": gap, "hidden_energy_unique": False}}


def algebra_case(kind, reference, ops, tolerance):
    integral = reference["integral"]
    matrix, system = integral["matrix"], integral["svd"]
    true_z = reference["truth_history"][0, 20:]
    right = matrix @ true_z
    details = {}
    if kind == "D_column_confounded":
        shift = np.array([.25, -.15])
        perturbation = matrix @ shift
        estimated = solve(system, right+perturbation, ops)
        residual = right+perturbation-matrix@estimated
        if max(np.max(np.abs(estimated-true_z-shift)), np.max(np.abs(residual))) > tolerance:
            raise FloatingPointError("column-space confounding identity failed")
        metrics = {"zero_residual_with_state_bias": True,
                   "state_bias_norm": float(np.linalg.norm(shift))}
    else:
        raw = (np.tile(.0002*helmert()[:, 2], (10, 1)) if kind == "D_force_orth"
               else .01*((-1.)**np.arange(10))[:, None] *
               (integral["pc"] @ np.array([1., 0.])))
        perturbation = (raw*integral["scale"]).reshape(-1)
        if kind == "D_port_temporal":
            perturbation -= system["u"] @ (system["u"].T @ perturbation)
            if np.linalg.norm(perturbation) <= 1e-12:
                raise FloatingPointError("predeclared temporal direction has no complement")
            perturbation *= .01 / np.linalg.norm(perturbation)
            raw = perturbation.reshape(10, 10) / integral["scale"]
        estimated = solve(system, right+perturbation, ops)
        residual = right+perturbation-matrix@estimated
        projection = integral["force_projection"]
        physical_orthogonal = raw @ (np.eye(10)-projection).T
        details.update(raw_physical_perturbation=raw, physical_orthogonal=physical_orthogonal)
        if kind == "D_force_orth":
            if np.max(np.abs(raw-physical_orthogonal)) > tolerance:
                raise FloatingPointError("physical force complement witness failed")
            weighted_orthogonal = perturbation-system["u"]@(system["u"].T@perturbation)
            unchanged = solve(system, right+weighted_orthogonal, ops)
            details.update(weighted_orthogonal=weighted_orthogonal,
                           weighted_orthogonal_estimate=unchanged)
            if np.max(np.abs(unchanged-true_z)) > tolerance:
                raise FloatingPointError("weighted orthogonal noise must not bias LS")
            metrics = {"physical_orthogonal": True,
                       "physical_error_can_bias_scaled_ls": float(np.linalg.norm(estimated-true_z)),
                       "weighted_orthogonal_state_error": float(np.linalg.norm(unchanged-true_z))}
        else:
            if (np.max(np.abs(physical_orthogonal)) > tolerance
                    or abs(np.linalg.norm(residual)-.01) > tolerance
                    or np.max(np.abs(estimated-true_z)) > tolerance):
                raise FloatingPointError("port-internal temporal inconsistency witness failed")
            metrics = {"physical_port_internal": True, "dynamic_inconsistency": True}
    return {"matrix": matrix, "true_z": true_z, "right": right,
            "perturbation": perturbation, "estimated": estimated, "residual": residual,
            "metrics": metrics, "details": details, "kind": "algebra_only_not_an_ODE_trajectory"}


def noise_pair(reference, model, power, bundle, ops, plan):
    decomposition = reference["operator_svd"]
    if decomposition["rank"] != 2:
        return {"status": "unresolved", "reason": "sampled_map_rank_deficient"}
    direction = decomposition["vh"][-1].copy()
    if direction[np.argmax(np.abs(direction))] < 0:
        direction *= -1
    epsilon = plan["noise"]["weak_rms"]
    amplitude = min(.5, epsilon/decomposition["s"][-1])
    histories, futures, checks = [], [], []
    for sign in (-1, 1):
        initial = reference["truth_history"][0].copy()
        initial[20:] += sign*amplitude*direction
        history, hc = checked_exact(bundle, initial, 10, model, power, ops, plan["tolerances"])
        future, fc = checked_exact(bundle, history[-1], 800, model, power, ops, plan["tolerances"])
        histories.append(history)
        futures.append(future)
        checks.append({"history": hc, "future": fc})
    history_gap = float(np.linalg.norm(histories[1][1:, :20]-histories[0][1:, :20])/np.sqrt(200))
    expected = 2*amplitude*decomposition["s"][-1]
    if abs(history_gap-expected) > plan["tolerances"]["algebra"] or history_gap > 2*epsilon+1e-10:
        raise FloatingPointError("finite noisy observation-pair identity failed")
    future_gap = errors(futures[1], futures[0], ops)
    return {"histories": histories, "futures": futures, "checks": checks,
            "direction": direction, "amplitude": amplitude, "history_gap": history_gap,
            "expected_history_gap": expected, "future_gap": future_gap,
            "future_relevant_ambiguity_witness": {
                h: row["visible_max_abs"] > 2*plan["tolerances"]["future_noisy"]
                for h, row in future_gap.items()}}


def execute(*, plan, result_dir):
    """The workflow has already bound, spent and durably reserved this plan."""
    base, tol = plan["base"], plan["tolerances"]
    ops = Operations(result_dir / "journal", plan)
    records, summaries, identifiability = {}, {}, {}
    resolved_sine_methods = 0
    root = result_dir.parent.parent
    write_json(result_dir / "execution_freeze.json", {
        "plan": plan, "source_commit": git_head(root), "python": sys.version,
        "executable": sys.executable,
        "packages": {name: version(name) for name in ("numpy", "scipy", "torch", "filelock", "h5py")},
        "real_data_reads": 0, "fit_attempts": 0, "optimizer_updates": 0,
    })
    try:
        with ops.scope("linear-fixtures"):
            models, powers, bundles, truths, checks = {}, {}, {}, {}, {}
            for key in ("main", "weak", "mismatch"):
                model, power, initial = make_model(
                    base, ops, weak=key == "weak", mismatch=key == "mismatch")
                models[key], powers[key] = model, power
                bundles[key] = linear_bundle(model, power, ops, tol["exponential"])
                if key != "mismatch":
                    truths[key], checks[key] = checked_exact(
                        bundles[key], initial, 820, model, power, ops, tol)
            save_record(result_dir, "linear-fixtures",
                        {"bundles": bundles, "truths": truths, "checks": checks,
                         "parameters": {k: model_parameters(m) for k, m in models.items()},
                         "powers": powers})
        for spec in plan["linear_cases"]:
            name, key = spec["id"], spec["model"]
            with ops.scope(name):
                result = linear_case(
                    models[key], powers[key], bundles[key],
                    truths["weak" if key == "weak" else "main"], spec["intervals"],
                    spec["noise"], ops, plan)
                if name in ("L100_clean", "W100_clean"):
                    result["finite_noise_pair"] = noise_pair(
                        result, models[key], powers[key], bundles[key], ops, plan)
                save_record(result_dir, name, result)
                summaries[name] = result["metrics"]
                identifiability[name] = result["observation_identifiability"]
                if name == "L100_clean":
                    records[name] = result
        with ops.scope("sine-truth"):
            model, power, initial = make_model(base, ops, sine=True)
            sine_truth, truth_checks = [], []
            for step in plan["sine_steps"]:
                path = rk_path(model, power, initial, 820, step, ops,
                               role="truth", method="known_truth")
                truth_checks.append(check_path(model, power, path, ops, tol))
                sine_truth.append(path)
            refinement = float(np.max(np.abs(sine_truth[0]-sine_truth[1])))
            if refinement > tol["refinement"]:
                raise FloatingPointError("nonlinear truth refinement gate failed")
            save_record(result_dir, "sine-truth", {
                "paths": sine_truth, "checks": truth_checks, "refinement_max": refinement,
                "parameters": model_parameters(model), "power": power, "initial": initial})
        for name, mode in (("S100_clean", "clean"), ("S100_alt", "alternating")):
            with ops.scope(name):
                result = sine_case(model, power, sine_truth[-1], mode, ops, plan)
                if set(result["metrics"]) != {"integral_ls", "true_z_reference"}:
                    raise RuntimeError("missing registered sine method")
                resolved_sine_methods += sum(
                    row["status"] == "resolved" for row in result["metrics"].values())
                save_record(result_dir, name, result)
                summaries[name] = result["metrics"]
                identifiability[name] = result["observation_identifiability"]
        for name in ("A_alias", "U_hidden_null", "D_force_orth",
                     "D_column_confounded", "D_port_temporal", "I_constant_input"):
            with ops.scope(name):
                if name == "A_alias":
                    result = alias_case(ops, plan)
                elif name == "U_hidden_null":
                    result = hidden_null_case(base, ops, plan)
                elif name.startswith("D_"):
                    result = algebra_case(name, records["L100_clean"], ops, tol["algebra"])
                else:
                    power = powers["main"]
                    held, linear = .01*power, .01*(power+power)/2
                    gap = float(np.max(np.abs(held-linear)))
                    if gap > tol["algebra"]:
                        raise FloatingPointError("constant-input integral identity failed")
                    integral = records["L100_clean"]["integral"]
                    other_right = integral["right4"] + np.tile(
                        (linear-held)*integral["scale"], 10)
                    other_z = solve(integral["svd"], other_right, ops)
                    other_origin = integral["phi"]@other_z+integral["w_nodes"][-1]
                    if max(np.max(np.abs(other_right-integral["right4"])),
                           np.max(np.abs(other_z-integral["z4"])),
                           np.max(np.abs(other_origin-integral["zT_observed"]))) > 1e-12:
                        raise FloatingPointError("constant-input estimator identity failed")
                    result = {"held": held, "linear_interpolated": linear,
                              "other_right": other_right, "other_z": other_z,
                              "other_zT_observed": other_origin,
                              "metrics": {"constant_input_integrals_equal": True,
                                          "max_gap": gap, "kind": "saved_input_algebra"}}
                save_record(result_dir, name, result)
                summaries[name] = result["metrics"]
                identifiability[name] = result.get("observation_identifiability", "algebra_only")
        if list(summaries) != plan["case_roster"] or ops.counts != ops.returns:
            raise RuntimeError("incomplete roster or unreturned operations")
        required_rhs = 78720 + 77760*resolved_sine_methods
        if (ops.counts["integration_rhs"] != required_rhs
                or required_rhs > plan["expected_integration_rhs"]
                or len(ops.clock_checks) != 2+4*resolved_sine_methods
                or 4*sum(row["rk4_updates"] for row in ops.clock_checks) != required_rhs):
            raise RuntimeError("registered nonlinear resolution branches were not fully executed")
    except BaseException as error:
        write_json(result_dir / "failure_summary.json", {
            "process_status": "FAIL", "error_type": type(error).__name__,
            "message": str(error), "completed_cases": list(summaries),
            "incomplete_cases": [name for name in plan["case_roster"] if name not in summaries],
            "operator_attempted": dict(ops.counts), "operator_returned": dict(ops.returns),
            "retry_allowed": False, "promotion_decision": "KEEP_BASELINE",
        })
        raise
    finally:
        ops.journal.close()
    summary = {
        "process_status": "PASS", "case_count": len(summaries), "cases": summaries,
        "observation_identifiability": identifiability,
        "operator_attempted": dict(ops.counts), "operator_returned": dict(ops.returns),
        "rk4_clock_checks": ops.clock_checks,
        "resolved_sine_method_count": resolved_sine_methods,
        "required_integration_rhs_for_resolved_methods": required_rhs,
        "scope_count": len(ops.journal.closed_scopes), "real_data_reads": 0,
        "fit_attempts": 0, "optimizer_updates": 0, "promotion_decision": "KEEP_BASELINE",
        "claim_boundary": "fixed synthetic cases only; no real-data sufficiency or causal root claim",
    }
    write_json(result_dir / "summary.json", summary)
    write_json(result_dir / "stage_evidence.json", {
        "status": "PASS", "process_status": "PASS", "source_commit": git_head(root),
        "data_policy": "analytic_synthetic_only", "case_count": len(summaries),
        "promotion_decision": "KEEP_BASELINE", "claim_boundary": summary["claim_boundary"],
    })
    write_json(result_dir / "artifact_hashes.json", {
        path.relative_to(result_dir).as_posix(): sha256_file(path)
        for path in sorted(result_dir.rglob("*")) if path.is_file()
    })
    return summary
