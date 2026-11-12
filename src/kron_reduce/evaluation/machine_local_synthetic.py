"""Frozen analytic checks and learning controls for diagonal hidden memory."""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from contextlib import contextmanager
from importlib.metadata import version

import numpy as np
import torch
from scipy.linalg import expm, helmert

from kron_reduce.data.history_contract import CausalStateHistory
from kron_reduce.evaluation.constant_c_observability import save_record, write_json
from kron_reduce.evaluation.state_port_training import loss
from kron_reduce.models.machine_local import DiagonalMemoryModel
from kron_reduce.models.nonlinear_latent import (
    NonlinearPassiveClosure,
    _features,
    complete_incidence,
)
from project_harness.evidence import sha256_file
from project_harness.git_state import git_head


class Budget:
    def __init__(self, path, limits, scope_limits=None):
        self.limits, self.counts = limits, Counter()
        self.returned, self.scopes = Counter(), []
        self.scope_limits, self.active, self.scope_start = scope_limits, None, Counter()
        self.stream = path.open("x", encoding="utf-8", newline="\n")

    def spend(self, key, amount=1):
        if key not in self.limits or type(amount) is not int or amount < 1 or self.counts[key]+amount > self.limits[key]:
            raise RuntimeError(f"unregistered or exhausted operation budget: {key}")
        if self.scope_limits is not None:
            limits = self.scope_limits.get(self.active, {})
            if self.counts[key]-self.scope_start[key]+amount > limits.get(key, 0):
                raise RuntimeError(f"unregistered or exhausted scope budget: {self.active}/{key}")
        self.counts[key] += amount
        self.stream.write(json.dumps({"operation": key, "attempted": amount, "total": self.counts[key]})+"\n")
        self.stream.flush()

    @contextmanager
    def scope(self, name):
        if self.active is not None or any(row["scope"] == name for row in self.scopes):
            raise RuntimeError("nested or repeated scientific scope")
        if self.scope_limits is not None and name not in self.scope_limits:
            raise RuntimeError("unregistered scientific scope")
        before = self.counts.copy()
        self.active, self.scope_start = name, before
        self.stream.write(json.dumps({"scope": name, "transition": "started"})+"\n")
        self.stream.flush()
        try:
            yield
        except BaseException as error:
            record = {"scope": name, "transition": "failed", "error_type": type(error).__name__,
                      "message": str(error), "attempted_allocations": dict(self.counts-before),
                      "returns_within_failed_scope": "unknown; all allocations remain spent"}
            raise
        else:
            completed = self.counts-before
            self.returned.update(completed)
            record = {"scope": name, "transition": "returned", "returned_allocations": dict(completed)}
        finally:
            self.active = None
            self.scopes.append(record)
            self.stream.write(json.dumps(record, allow_nan=False)+"\n")
            self.stream.flush()


def fixture(kind="local", *, backbone="linear"):
    i = np.arange(10, dtype=float)
    mass, b = 1+.1*i, complete_incidence(10)
    p = np.eye(10)-np.outer(mass, np.ones(10))/mass.sum()
    g = .35+.025*i
    tau = np.geomspace(.05, 2., 10)
    c = p*np.asarray(g)[None, :]
    if kind == "nonlocal":
        o = np.sqrt(.2)*np.cos(np.pi*(i[:, None]+.5)*i[None, :]/10)
        o[:, 0] = 1/np.sqrt(10)
        c = c @ o
    elif kind == "equal":
        tau = np.full(10, .3)
    elif kind == "near":
        tau = .3*(1+1e-6*i)
    elif kind == "weak":
        c[:, -1] *= 1e-5
    elif kind == "zero":
        c[:, -1] = 0
    elif kind != "local":
        raise ValueError("unknown analytic fixture")
    return NonlinearPassiveClosure(mass, np.full(45, .12), .08*b@b.T, c,
                                   np.zeros((10, 10)), np.diag(1/tau), backbone)


def augmented_generator(model):
    n, r = model.n_nodes, model.latent_dimension
    p = np.eye(n)-np.outer(model.mass, np.ones(n))/model.mass.sum()
    b = complete_incidence(n)
    a = np.zeros((3*n+r, 3*n+r))
    a[:n, n:2*n] = np.eye(n)
    a[n:2*n, :n] = -(p @ (b*model.edge_stiffness) @ b.T)/model.mass[:, None]
    a[n:2*n, n:2*n] = -(p @ model.damping)/model.mass[:, None]
    a[n:2*n, 2*n:2*n+r] = -(p @ model.coupling)/model.mass[:, None]
    a[n:2*n, 2*n+r:] = p/model.mass[:, None]
    a[2*n:2*n+r, n:2*n] = model.coupling.T
    a[2*n:2*n+r, 2*n:2*n+r] = model.latent_skew-model.latent_dissipation
    return a


def new_model(backbone, kind, feature_scale, *, seed=324):
    n = backbone.n_nodes
    p = np.eye(n)-np.outer(backbone.mass, np.ones(n))/backbone.mass.sum()
    c = p*np.full((1, n), .4)
    basis = helmert(n, full=False).T
    if kind == "G2":
        r, tau = 2, np.array([.1, 1.])
        port = np.linalg.norm(c)/np.sqrt(2)*np.eye(n-1, 2)
    elif kind in {"G10", "L10"}:
        r, tau = n, np.geomspace(.06, 1.6, n)
        port = np.full(n, .4) if kind == "L10" else basis.T @ c
    else:
        raise ValueError("unknown model kind")
    encoder = np.random.default_rng(seed).normal(size=(r, 8*n))
    encoder *= .1/np.linalg.norm(encoder)
    return DiagonalMemoryModel(backbone, port_kind="local" if kind == "L10" else "global",
                               port=port, tau=tau, encoder=encoder, feature_scale=feature_scale)


def _features_batch(q, v, power, equilibrium):
    return np.stack([_features(CausalStateHistory(np.arange(-10, 1)*.01, a, b, p), e, 3)
                     for a, b, p, e in zip(q, v, power, equilibrium, strict=True)])


def gradient_check(model, q, v, power, eq, features, fast, budget, rng, checks, arrays, prefix):
    future = model.causal_rollout(q, v, power, eq, features, 10,
                                  linear_fast=fast, spend=budget.spend)
    objective = future[:, 1:, :20].square().mean()+model.regularization()
    budget.spend("gradients")
    parameters = tuple(model.parameters())
    gradients = torch.autograd.grad(objective, parameters)
    arrays[prefix+"_gradient"] = torch.cat([g.flatten() for g in gradients]).detach().numpy()
    for index, (parameter, grad) in enumerate(zip(parameters, gradients, strict=True)):
        direction = torch.tensor(rng.normal(size=tuple(parameter.shape)), dtype=torch.float64)
        direction /= torch.linalg.vector_norm(direction)
        original = parameter.detach().clone()
        arrays[prefix+f"_direction{index}"] = direction.detach().numpy()
        for epsilon in (1e-5, 1e-6):
            values = []
            for sign in (1, -1):
                with torch.no_grad():
                    parameter.copy_(original+sign*epsilon*direction)
                    prediction = model.causal_rollout(q, v, power, eq, features, 10,
                        linear_fast=fast, spend=budget.spend)
                    values.append(float(prediction[:, 1:, :20].square().mean()+model.regularization()))
            with torch.no_grad():
                parameter.copy_(original)
            finite, exact = (values[0]-values[1])/(2*epsilon), float((grad*direction).sum())
            key = prefix+f"_direction{index}_{epsilon}"
            checks[key] = abs(finite-exact) <= 1e-9+1e-5*abs(exact)
            arrays[key] = np.array([finite, exact, epsilon, *values])


def correctness(directory, budget, plan):
    rng = np.random.default_rng(20261112)
    checks, arrays = {}, {}
    for backbone in ("linear", "sine"):
        base = fixture(backbone=backbone)
        p = np.eye(10)-np.outer(base.mass, np.ones(10))/base.mass.sum()
        state = np.c_[.2*rng.normal(size=(2, 10))@p, .1*rng.normal(size=(2, 10))@p,
                      .15*rng.normal(size=(2, 10))]
        power, eq = .03*rng.normal(size=(2, 10))@p.T, .1*rng.normal(size=(2, 10))@p
        models = {k: new_model(base, k, np.ones(80)) for k in ("G10", "L10")}
        models["G2"] = new_model(base, "G2", np.ones(80))
        np.savez_compressed(directory/f"{backbone}_s0_inputs.npz", state=state, power=power, equilibrium=eq,
                            mass=base.mass, stiffness=base.edge_stiffness, damping=base.damping,
                            **{kind+"_"+name: value.detach().numpy() for kind, model in models.items()
                               for name, value in model.state_dict().items()})
        for kind, model in models.items():
            r = model.latent_dimension
            x = np.c_[state[:, :20], state[:, 20:20+r]]
            physical = model.closure()
            budget.spend("numpy_rhs", 4)
            identities = [physical.energy_identity(a, b, c)
                          for a, b, c in zip(x, power, eq, strict=True)]
            normalizers = [max(1., abs(a[10:20]@b)+abs(a[10:20]@physical.damping@a[10:20])
                              +abs(a[20:]@physical.latent_dissipation@a[20:]))
                           for a, b in zip(x, power, strict=True)]
            numpy_rhs = np.stack([physical.rhs(a, b, c) for a, b, c in zip(x, power, eq, strict=True)])
            budget.spend("torch_rhs", 1)
            torch_rhs = model.rhs(x, power, eq).detach().numpy()
            prefix = f"{backbone}_{kind}"
            checks[prefix+"_energy"] = max(row["identity_error"]/normalizer for row, normalizer in zip(identities, normalizers, strict=True)) < 1e-10
            arrays[prefix+"_energy"] = np.array([[row["energy_rate"], row["supply_minus_dissipation"], row["identity_error"], normalizer]
                                               for row, normalizer in zip(identities, normalizers, strict=True)])
            arrays[prefix+"_numpy_rhs"], arrays[prefix+"_torch_rhs"] = numpy_rhs, torch_rhs
            projection = model.power_projection.detach().numpy()
            checks[prefix+"_projection"] = np.max(np.abs(projection@projection-projection)) < 1e-12
            checks[prefix+"_balanced_port"] = np.max(np.abs(physical.coupling.sum(axis=0))) < 1e-12
            checks[prefix+"_trainable_set"] = set(dict(model.named_parameters())) == {"port_parameter", "rate_parameter", "encoder"}
            checks[prefix+"_frozen_buffers"] = not any(value.requires_grad for value in model.buffers())
            checks[prefix+"_rhs"] = np.max(np.abs(numpy_rhs-torch_rhs)) < 1e-10
            checks[prefix+"_coi"] = max(np.max(np.abs(torch_rhs[:, :10]@base.mass)),
                                                np.max(np.abs(torch_rhs[:, 10:20]@base.mass))) < 1e-12
            with torch.no_grad():
                fast = model.rollout(x, power, eq, 30, max_step=.005, spend=budget.spend)
                slow = model.rollout(x, power, eq, 30, max_step=.005, linear_fast=False, spend=budget.spend)
                fine = model.rollout(x, power, eq, 30, max_step=.0025, spend=budget.spend)
                finer = model.rollout(x, power, eq, 30, max_step=.00125, spend=budget.spend)
            checks[prefix+"_implementations"] = float(torch.max(torch.abs(fast-slow))) < 1e-10
            checks[prefix+"_refinement"] = float(torch.max(torch.abs(fine-finer))) < 1e-5
            checks[prefix+"_path_coi"] = max(float(torch.max(torch.abs(fine[:, :, :10]@model.mass))),
                                             float(torch.max(torch.abs(fine[:, :, 10:20]@model.mass)))) < 1e-10
            arrays[prefix+"_paths"] = np.stack([v.numpy() for v in (fast, slow, fine, finer)])
        checks[backbone+"_common_start"] = np.max(np.abs(
            models["G10"].coupling.detach().numpy()-models["L10"].coupling.detach().numpy())) < 1e-14
        model = models["L10"]
        if backbone == "linear":
            padded = torch.cat((model.encoder, torch.zeros((1, 80), dtype=torch.float64)), dim=0)
            penalty = 1e-4*padded.square().sum()/160
            budget.spend("gradients")
            penalty_grad = torch.autograd.grad(penalty, model.encoder)[0]
            checks["fixed_regularizer_padding"] = abs(float(penalty.detach()-model.regularization().detach())) < 1e-15
            checks["fixed_regularizer_gradient"] = float(torch.max(torch.abs(penalty_grad-2e-4*model.encoder/160)).detach()) < 1e-15
        history = arrays[f"{backbone}_L10_paths"][0, :, :11]
        q, v = history[:, :, :10], history[:, :, 10:20]
        features = _features_batch(q, v, power, eq)
        for fast in (True, False) if backbone == "linear" else (False,):
            gradient_check(model, q, v, power, eq, features, fast, budget, rng, checks, arrays,
                           f"{backbone}_fast{fast}")
        if backbone == "linear":
            checks["linear_fast_gradient_match"] = np.max(np.abs(
                arrays["linear_fastTrue_gradient"]-arrays["linear_fastFalse_gradient"])) < 1e-10
            for kind in ("G2", "G10"):
                gradient_check(models[kind], q, v, power, eq, features, True, budget, rng, checks, arrays,
                               "linear_"+kind)
        with torch.no_grad():
            local_h, local_f = model.causal_rollout(q, v, power, eq, features, 300,
                spend=budget.spend, return_history=True)
            global_h, global_f = models["G10"].causal_rollout(q, v, power, eq, features, 300,
                spend=budget.spend, return_history=True)
            checks[backbone+"_common_history"] = float(torch.max(torch.abs(local_h-global_h))) < 1e-10
            checks[backbone+"_common_future"] = float(torch.max(torch.abs(local_f-global_f))) < 1e-10
            arrays[backbone+"_common_history"] = np.stack([local_h.numpy(), global_h.numpy()])
            arrays[backbone+"_common_future"] = np.stack([local_f.numpy(), global_f.numpy()])
            checks[backbone+"_visible_reset"] = np.array_equal(local_f[:, 0, :20].numpy(), np.concatenate((q[:, -1], v[:, -1]), axis=1))
            checks[backbone+"_hidden_continuity"] = torch.equal(local_f[:, 0, 20:], local_h[:, -1, 20:])
            if backbone == "linear":
                slow_f = model.causal_rollout(q, v, power, eq, features, 300,
                    linear_fast=False, spend=budget.spend)
                checks["linear_complete3s_fast_match"] = float(torch.max(torch.abs(local_f-slow_f))) < 1e-10
                arrays["linear_complete3s_slow"] = slow_f.numpy()
        checkpoint = directory/f"{backbone}_checkpoint.npz"
        np.savez_compressed(checkpoint, **{name: value.detach().numpy() for name, value in model.state_dict().items()})
        restored = new_model(base, "L10", np.ones(80))
        budget.spend("checkpoint_loads")
        with np.load(checkpoint, allow_pickle=False) as saved:
            restored.load_state_dict({name: torch.tensor(saved[name]) for name in saved.files}, strict=True)
        with torch.no_grad():
            restored_future = restored.causal_rollout(q, v, power, eq, features, 300, spend=budget.spend)
        checks[backbone+"_checkpoint"] = torch.equal(local_f, restored_future)
        arrays[backbone+"_restored_future"] = restored_future.numpy()
        warm, full = model.causal_rollout(q, v, power, eq, features, 10,
                                          spend=budget.spend, return_history=True)
        budget.spend("gradients")
        full_gradient = torch.autograd.grad(full[:, 1:, :20].square().mean(), model.encoder)[0]
        detached_origin = torch.cat((torch.tensor(q[:, -1]), torch.tensor(v[:, -1]), warm[:, -1, 20:].detach()), dim=1)
        detached = model.rollout(detached_origin, power, eq, 10, spend=budget.spend)
        budget.spend("gradients")
        missing_gradient = torch.autograd.grad(detached[:, 1:, :20].square().mean(), model.encoder, allow_unused=True)[0]
        checks[backbone+"_detach_negative_control"] = missing_gradient is None and float(torch.linalg.vector_norm(full_gradient)) > 1e-8
        arrays[backbone+"_full_encoder_gradient"] = full_gradient.detach().numpy()
        arrays[backbone+"_detached_encoder_unused"] = np.array([missing_gradient is None])
        h = CausalStateHistory(np.arange(-10, 1)*.01, q[0], v[0], power[0])
        try:
            model.predict(h, eq[0], 10, future_targets=np.zeros((11, 20)))
        except TypeError:
            checks[backbone+"_future_input_rejected"] = True
        else:
            checks[backbone+"_future_input_rejected"] = False
        physical = model.closure()
        missing_reciprocal = abs(state[0, 20:] @ physical.coupling.T @ state[0, 10:20])
        checks[backbone+"_reciprocity_negative_control"] = missing_reciprocal > 1e-8
        arrays[backbone+"_missing_reciprocal_power"] = np.array([missing_reciprocal])
    checks = {key: bool(value) for key, value in checks.items()}
    if set(checks) != set(plan["s0_check_ids"]):
        raise RuntimeError("S0 check roster differs from the frozen plan")
    np.savez_compressed(directory/"s0_arrays.npz", **arrays)
    write_json(directory/"s0_checks.json", checks)
    return {"scientific_pass": bool(all(checks.values())), "checks": checks,
            "claim_boundary": "Finite correctness of base local/global diagonal memory; no learned or real-data accuracy claim."}


def observability(directory, budget):
    rows, arrays = {}, {}
    for kind in ("local", "equal", "near", "weak", "zero", "nonlocal"):
        model = fixture(kind)
        a = augmented_generator(model)
        budget.spend("expm")
        transition = expm(.01*a)
        power, blocks = np.eye(40), []
        for j in range(11):
            blocks.append(power[:20, 20:30])
            if j != 10:
                budget.spend("matrix_products")
                power = power @ transition
        design = np.concatenate(blocks)
        budget.spend("svd")
        _, singular, vh = np.linalg.svd(design, full_matrices=False)
        threshold = 64*np.finfo(float).eps*max(design.shape)*singular[0]
        rank = int(np.count_nonzero(singular > threshold))
        row_scale = np.tile(np.r_[np.full(10, .2), np.full(10, .1)], 11)
        scaled_design = design/row_scale[:, None]
        budget.spend("svd")
        scaled_singular = np.linalg.svd(scaled_design, compute_uv=False)
        hidden = .15*np.cos(np.arange(1, 11))
        budget.spend("matrix_products")
        observed = design @ hidden
        budget.spend("least_squares")
        recovered = np.linalg.lstsq(design, observed, rcond=threshold/singular[0])[0]
        null_error = None
        if kind == "equal":
            direction = model.mass/(.35+.025*np.arange(10))
            direction /= np.linalg.norm(direction)
            null_error = float(np.linalg.norm(design@direction))
        elif kind == "zero":
            null_error = float(np.linalg.norm(design[:, -1]))
        rows[kind] = {"rank": rank, "singular_values": singular.tolist(), "rank_threshold": float(threshold),
                      "scale_normalized_singular_values": scaled_singular.tolist(),
                      "scale_normalization_is_not_noise_covariance_whitening": True,
                      "state_error": float(np.linalg.norm(recovered-hidden)), "null_error": null_error,
                      "worst_noise_amplification": float(1/singular[-1]) if rank == 10 else None}
        if kind == "nonlocal":
            local_ports = np.eye(10)-np.outer(model.mass, np.ones(10))/model.mass.sum()
            distances = np.stack([np.linalg.norm(model.coupling-local_ports[:, i, None]*
                (local_ports[:, i]@model.coupling)[None, :]/np.dot(local_ports[:, i], local_ports[:, i]), axis=0)
                for i in range(10)])
            rows[kind]["nearest_local_direction_residual"] = distances.min(axis=0).tolist()
        arrays[kind+"_design"], arrays[kind+"_directions"] = design, vh
        arrays[kind+"_scaled_design"] = scaled_design
        arrays[kind+"_state_true"], arrays[kind+"_state_recovered"] = hidden, recovered
        arrays[kind+"_observed"], arrays[kind+"_transition"] = observed, transition
        arrays[kind+"_generator"], arrays[kind+"_coupling"] = a, model.coupling
        arrays[kind+"_singular"], arrays[kind+"_scaled_singular"] = singular, scaled_singular
    n = helmert(10, full=False).T[:, :2]
    w = 2*np.pi/.01
    alias = NonlinearPassiveClosure(np.ones(10), np.zeros(45), np.zeros((10, 10)),
                                    w*n, np.zeros((2, 2)), np.zeros((2, 2)))
    a = augmented_generator(alias)
    z = np.r_[np.zeros(20), [1., -.5], np.zeros(10)]
    budget.spend("expm", 2)
    budget.spend("matrix_products")
    transition, quarter = expm(.01*a), expm(.0025*a)@z
    sampled, state = [], z.copy()
    for _ in range(11):
        sampled.append(state[:20].copy())
        budget.spend("matrix_products")
        state = transition@state
    alias_max, interior = float(np.max(np.abs(sampled))), float(np.max(np.abs(quarter[:20])))
    analytic_quarter = np.r_[-n@np.array([1., -.5])/w, -n@np.array([1., -.5])]
    analytic_error = float(np.max(np.abs(quarter[:20]-analytic_quarter)))
    rows["alias_boundary"] = {"sampled_visible_max": alias_max, "interior_visible_max": interior,
                               "quarter_speed_norm": float(np.linalg.norm(quarter[10:20])),
                               "analytic_error": analytic_error,
                               "outside_fitted_strictly_dissipative_family": True}
    arrays["alias_samples"], arrays["alias_quarter"] = np.asarray(sampled), quarter
    arrays["alias_analytic_quarter"], arrays["alias_initial"] = analytic_quarter, z
    passed = (rows["local"]["rank"] == 10 and rows["equal"]["rank"] == 9
              and rows["zero"]["rank"] == 9 and rows["equal"]["null_error"] < 1e-10
              and rows["near"]["singular_values"][-1] < rows["local"]["singular_values"][-1]
              and rows["weak"]["singular_values"][-1] < rows["local"]["singular_values"][-1]
              and rows["zero"]["null_error"] < 1e-10
              and max(rows["nonlocal"]["nearest_local_direction_residual"]) > 1e-6
              and alias_max < 1e-10 and analytic_error < 1e-10
              and abs(np.linalg.norm(quarter[10:20])-np.sqrt(1.25)) < 1e-10)
    write_json(directory/"observability.json", {"pass": bool(passed), "cases": rows})
    np.savez_compressed(directory/"observability.npz", **arrays)
    return bool(passed)


def synthetic_data(kind, seed, budget):
    truth = fixture(kind)
    mass = truth.mass
    p = np.eye(10)-np.outer(mass, np.ones(10))/mass.sum()
    rng = np.random.default_rng(seed)
    initial = np.c_[.2*rng.normal(size=(64, 10))@p, .1*rng.normal(size=(64, 10))@p,
                    .15*rng.normal(size=(64, 10)), .03*rng.normal(size=(64, 10))@p.T]
    a = augmented_generator(truth)
    budget.spend("expm", 3)
    transition = expm(.01*a)
    paths, current = [initial.copy()], initial.copy()
    for _ in range(810):
        budget.spend("matrix_products", 1)
        current = current @ transition.T
        paths.append(current.copy())
    paths = np.stack(paths, axis=1)
    budget.spend("matrix_products", 2)
    endpoints = [initial@expm(t*a).T for t in (.1, 8.1)]
    error = max(np.max(np.abs(paths[:, index]-exact)) for index, exact in zip((10, 810), endpoints, strict=True))
    if error > 1e-10:
        raise FloatingPointError("analytic truth endpoint consistency failed")
    # Scales use only clean train, with equal event weights and actual target clocks.
    target_scale = np.maximum(np.sqrt(np.mean(paths[:32, 11:311, :20]**2, axis=(0, 1))), 1e-5)
    history_scale = np.maximum(np.sqrt(np.mean(paths[:32, :11, :20]**2, axis=(0, 1))), 1e-5)
    return truth, paths, target_scale, history_scale


def fit_one(model, inputs, target, scale, budget, directory, key):
    np.savez_compressed(directory/f"{key}_initial.npz",
                        **{name: value.detach().numpy() for name, value in model.state_dict().items()})
    optimizer = torch.optim.Adam(model.parameters(), lr=.01, betas=(.9, .999), eps=1e-8)
    history = []
    with (directory/f"{key}_updates.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        try:
            for update in range(240):
                optimizer.zero_grad(set_to_none=True)
                prediction = model.causal_rollout(*inputs, 300, max_step=.005, spend=budget.spend)
                main = loss(prediction, target, np.full(32, 1/32), scale, np.full(32, 300))
                objective = main+model.regularization()
                budget.spend("gradients")
                objective.backward()
                try:
                    norm = torch.nn.utils.clip_grad_norm_(tuple(model.parameters()), 10, error_if_nonfinite=True)
                except RuntimeError as error:
                    if "non-finite" in str(error):
                        raise FloatingPointError("nonfinite fitted gradient") from error
                    raise
                budget.spend("optimizer_updates")
                optimizer.step()
                row = [update+1, float(main.detach()), float(objective.detach()), float(norm)]
                history.append(row)
                stream.write(json.dumps(row, allow_nan=False)+"\n")
                stream.flush()
        except BaseException as error:
            save_record(directory, key+"_failed_state", {
                "parameters": {name: value.detach().numpy() for name, value in model.state_dict().items()},
                "last_completed_update": len(history), "error_type": type(error).__name__,
                "message": str(error)}, failed=True)
            raise
    return np.array(history)


def score_one(model, history, key, kind, noisy, model_kind, truth, paths, scale, q, v, power, eq, features, directory, budget):
    arrays = {name: value.detach().numpy() for name, value in model.state_dict().items()}
    arrays["training_curve"] = history
    metrics = {}
    with torch.no_grad():
        h, forecast = model.causal_rollout(q, v, power, eq, features, 800,
            max_step=.0025, spend=budget.spend, return_history=True)
        fine = model.causal_rollout(q, v, power, eq, features, 800,
            max_step=.00125, spend=budget.spend)
        arrays["history"], arrays["forecast"], arrays["refined_forecast"] = h.numpy(), forecast.numpy(), fine.numpy()
        for role, indices in (("train", slice(0, 32)), ("val", slice(32, 48)), ("test", slice(48, 64))):
            predicted = forecast[indices].numpy()
            exact = paths[indices, 10:, :20]
            row = {}
            for seconds in (1, 3, 8):
                error = (predicted[:, 1:seconds*100+1, :20]-exact[:, 1:seconds*100+1])/scale
                for channel, selection in (("q", slice(0, 10)), ("v", slice(10, 20)), ("all", slice(0, 20))):
                    row[f"nrmse_{seconds}s_{channel}"] = float(np.sqrt(np.mean(error[:, :, selection]**2)))
                    row[f"max_abs_{seconds}s_{channel}"] = float(np.max(np.abs(
                        predicted[:, 1:seconds*100+1, :20][:, :, selection]-exact[:, 1:seconds*100+1, selection])))
                    row[f"rmse_{seconds}s_{channel}"] = float(np.sqrt(np.mean(np.square(
                        predicted[:, 1:seconds*100+1, :20][:, :, selection]-exact[:, 1:seconds*100+1, selection]))))
            row["main_nrmse"] = float(np.sqrt((row["nrmse_1s_all"]**2+row["nrmse_3s_all"]**2)/2))
            row["coi_error"] = float(max(np.max(np.abs(predicted[:, :, :10]@truth.mass)),
                                         np.max(np.abs(predicted[:, :, 10:20]@truth.mass))))
            row["step_difference"] = float(np.max(np.abs(
                (predicted[:, :, :20]-fine[indices, :, :20].numpy())/scale)))
            row["hidden_start_energy_mean"] = float(np.mean(np.sum(h[indices, 0, 20:].numpy()**2, axis=1)/2))
            row["hidden_origin_energy_mean"] = float(np.mean(np.sum(h[indices, -1, 20:].numpy()**2, axis=1)/2))
            before = h[indices, -1].numpy()
            b = complete_incidence(10)
            visible_energy = lambda x, b=b: np.sum((x[:, :10]@b)**2*truth.edge_stiffness, axis=1)/2+np.sum(x[:, 10:20]**2*truth.mass, axis=1)/2
            row["visible_reset_energy_mean"] = float(np.mean(visible_energy(predicted[:, 0])-visible_energy(before)))
            row["history_nrmse"] = float(np.sqrt(np.mean(((h[indices, :, :20].numpy()-paths[indices, :11, :20])/scale)**2)))
            if kind == "local" and model_kind == "L10":
                row["hidden_start_rmse"] = float(np.sqrt(np.mean((h[indices, 0, 20:].numpy()-paths[indices, 0, 20:30])**2)))
                row["hidden_origin_rmse"] = float(np.sqrt(np.mean((h[indices, -1, 20:].numpy()-paths[indices, 10, 20:30])**2)))
            metrics[role] = row
    physical = model.closure()
    budget.spend("numpy_rhs", 3)
    energy_rows = []
    for i in (0, 32, 48):
        state = arrays["forecast"][i, 0]
        identity = physical.energy_identity(state, power[i], eq[i])
        v0, z0 = state[10:20], state[20:]
        denominator = max(1., abs(v0@power[i])+abs(v0@physical.damping@v0)+abs(z0@physical.latent_dissipation@z0))
        energy_rows.append({**identity, "normalizer": float(denominator)})
    energy_residual = max(row["identity_error"]/row["normalizer"] for row in energy_rows)
    np.savez_compressed(directory/f"{key}.npz", **arrays)
    record = {"case": kind, "noise": .01 if noisy else 0., "model": model_kind,
              "success": True, "updates": 240, "energy_identity_error": float(energy_residual),
              "energy_identity_samples": energy_rows,
              "tau": model.tau.detach().numpy().tolist(), "metrics": metrics}
    if kind == "local" and model_kind == "L10":
        record["local_parameter_relative_error"] = {
            "g": float(np.linalg.norm(np.exp(arrays["port_parameter"])-(.35+.025*np.arange(10)))/np.linalg.norm(.35+.025*np.arange(10))),
            "tau": float(np.linalg.norm(model.tau.detach().numpy()-np.geomspace(.05, 2., 10))/np.linalg.norm(np.geomspace(.05, 2., 10)))}
    write_json(directory/f"{key}.json", record)
    print(f"completed {key}: synthetic test 3s NRMSE={metrics['test']['nrmse_3s_all']:.6g}", flush=True)
    return record


def learning(directory, budget, plan):
    with budget.scope("observability"):
        diagnostics_pass = observability(directory, budget)
    datasets = {}
    # Save and hash every role, truth and noise array before the first optimizer update.
    for kind, seed in (("local", 2026111201), ("nonlocal", 2026111202)):
        with budget.scope("truth_"+kind):
            truth, paths, target_scale, history_scale = synthetic_data(kind, seed, budget)
        p = np.eye(10)-np.outer(truth.mass, np.ones(10))/truth.mass.sum()
        noise = np.random.default_rng(seed+100).normal(size=(64, 11, 20))*history_scale[None, None, :]*.01
        noise[:, :, :10] = noise[:, :, :10]@p
        noise[:, :, 10:] = noise[:, :, 10:]@p
        np.savez_compressed(directory/f"{kind}_truth.npz", paths=paths, target_scale=target_scale,
                            history_scale=history_scale, noise=noise,
                            mass=truth.mass, stiffness=truth.edge_stiffness, damping=truth.damping,
                            coupling=truth.coupling, decay=truth.latent_dissipation,
                            roles=np.array(["train"]*32+["val"]*16+["test"]*16))
        datasets[kind] = (truth, paths, target_scale, noise)
    write_json(directory/"frozen_inputs.json", {p.name: sha256_file(p) for p in sorted(directory.glob("*_truth.npz"))})
    records, jobs = [], []
    for kind, (truth, paths, scale, noise) in datasets.items():
        for noisy in (False, True):
            visible_history = paths[:, :11, :20].copy()+(noise if noisy else 0)
            q, v = visible_history[:, :, :10], visible_history[:, :, 10:]
            power, eq = paths[:, 0, 30:], np.zeros((64, 10))
            features = _features_batch(q, v, power, eq)
            feature_scale = np.maximum(np.sqrt(np.mean(features[:32]**2, axis=0)), 1e-5)
            for model_kind in ("G2", "G10", "L10"):
                key = f"{kind}_{'noise01' if noisy else 'clean'}_{model_kind}"
                model = new_model(truth, model_kind, feature_scale)
                train_target = torch.tensor(paths[:32, 10:311, :20])
                try:
                    with budget.scope("fit_"+key):
                        budget.spend("fits")
                        history = fit_one(model, (q[:32], v[:32], power[:32], eq[:32], features[:32]),
                                          train_target, scale, budget, directory, key)
                except FloatingPointError as error:
                    record = {"case": kind, "noise": .01 if noisy else 0., "model": model_kind,
                              "success": False, "failure": str(error)}
                    write_json(directory/f"{key}.json", record)
                    records.append(record)
                    continue
                np.savez_compressed(directory/f"{key}_fitted.npz",
                                    **{name: value.detach().numpy() for name, value in model.state_dict().items()})
                jobs.append((model, history, key, kind, noisy, model_kind, truth, paths, scale, q, v, power, eq, features))
    # All twelve fits are frozen before any held-out future is scored.
    for job in jobs:
        try:
            with budget.scope("score_"+job[2]):
                records.append(score_one(*job, directory, budget))
        except FloatingPointError as error:
            record = {"case": job[3], "noise": .01 if job[4] else 0., "model": job[5],
                      "success": False, "failure": str(error)}
            write_json(directory/(job[2]+".json"), record)
            records.append(record)
    limits = plan["learning_gates"]
    matched = [r for r in records if (r["case"], r["model"]) in (("local", "L10"), ("local", "G10"), ("nonlocal", "G10"))]
    prediction_pass = len(matched) == 6 and len(records) == 12 and all(r["success"] and r["metrics"]["test"][metric] <= limits["noisy_nrmse" if r["noise"] else "clean_nrmse"]
                          for r in matched for metric in ("nrmse_3s_all", "main_nrmse"))
    refinement_pass = all(r["success"] and all(m["step_difference"] <= limits["step_difference"]
                                              for m in r["metrics"].values()) for r in records)
    physics_pass = all(r["success"] and r["energy_identity_error"] < 1e-10
                      and all(m["coi_error"] < 1e-10 for m in r["metrics"].values()) for r in records)
    return {"scientific_pass": bool(diagnostics_pass and prediction_pass and refinement_pass and physics_pass),
            "diagnostics_pass": diagnostics_pass, "prediction_pass": prediction_pass,
            "refinement_pass": refinement_pass, "physics_pass": physics_pass, "records": records,
            "claim_boundary": "Base-model synthetic study only; fixed optimization failures are not proofs of non-identifiability. No physical function discovery or real-data claim."}


def peak_memory():
    """Process peak working set, including native Torch allocations."""
    if os.name != "nt":
        import resource
        return {"bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                "source": "ru_maxrss (Linux KiB)"}
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)]+[
            (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize",
            "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
            "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
    get_process = ctypes.windll.kernel32.GetCurrentProcess
    get_process.restype = wintypes.HANDLE
    get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
    get_memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    if not get_memory(get_process(), ctypes.byref(counters), counters.cb):
        return {"bytes": None, "source": "GetProcessMemoryInfo failed"}
    return {"bytes": int(counters.PeakWorkingSetSize), "source": "PeakWorkingSetSize"}


def execute(*, root, result_dir, stage, plan):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    start = time.perf_counter()
    budget = Budget(result_dir/"operations.jsonl", plan["stages"][stage]["operation_limits"],
                    plan["stages"][stage]["scope_limits"])
    try:
        if stage == "s0":
            with budget.scope("correctness"):
                summary = correctness(result_dir, budget, plan)
        else:
            summary = learning(result_dir, budget, plan)
        summary.update(process_status="PASS", promotion_decision="KEEP_BASELINE", real_data_reads=0,
                       operations=dict(budget.counts), operations_returned=dict(budget.returned),
                       scopes=budget.scopes, source_commit=git_head(root),
                       wall_seconds=time.perf_counter()-start,
                       process_peak_memory=peak_memory(),
                       runtime={"python": sys.version, "executable": sys.executable,
                                **{name: version(name) for name in ("numpy", "scipy", "torch")}})
        write_json(result_dir/"summary.json", summary)
        write_json(result_dir/"stage_evidence.json", {key: summary[key] for key in (
            "process_status", "promotion_decision", "source_commit", "claim_boundary", "scientific_pass")}
            | {"status": "PASS", "data_policy": "analytic_synthetic_only"})
    except BaseException as error:
        write_json(result_dir/"failure.json", {"error_type": type(error).__name__, "message": str(error),
                   "operations_attempted": dict(budget.counts), "retry_allowed": False})
        raise
    finally:
        budget.stream.close()
        write_json(result_dir/"artifact_hashes.json", {p.relative_to(result_dir).as_posix(): sha256_file(p)
            for p in sorted(result_dir.rglob("*")) if p.is_file() and p.name != "artifact_hashes.json"})
    return summary
