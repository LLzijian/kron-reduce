"""One full-library fit followed by a frozen independent-truth gate."""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.linalg import expm

from kron_reduce.evaluation.joint_energy_sindy_truth import operators, port_and_edges, truth_rollout
from kron_reduce.evaluation.joint_energy_sindy_weak import initialize
from kron_reduce.evaluation.machine_local_clean_learning import NUMERICAL_ERRORS, snapshot
from kron_reduce.evaluation.machine_local_equation import ROLES, STEPS, align_classes
from kron_reduce.evaluation.machine_local_topology_transfer import transfer_cases
from kron_reduce.evaluation.topology_memory_rom_synthetic import _scores, _write
from kron_reduce.evaluation.trajectory_stability import classify
from kron_reduce.models.joint_energy_sindy import JointEnergySINDy
from kron_reduce.models.machine_local_topology import predict_from_topology_history


def _observe(model, history, power, equilibrium, scale, count, settings, step, budget, *, rank=False):
    return predict_from_topology_history(model, history[:, :, :10], history[:, :, 10:20], power, equilibrium, scale,
                                         count, max_step=step, spend=budget.spend, memory_radius=settings["memory_radius"],
                                         prior_weight=settings["prior_weight"], return_observation_jacobian=rank)


def _population(plan):
    cases = transfer_cases(plan)
    n = len(cases["roles"])
    rng = np.random.default_rng(plan["seed"]+1)
    mass = 1.+.1*np.arange(10)
    proj = operators(mass)[-1]
    phases = np.sin(np.arange(10))*plan["truth"]["reference_amplitude"]
    eq = np.broadcast_to(phases, (n, 10)).copy() @ proj
    sign = np.where((np.arange(n)//2) % 2, -1., 1.)[:, None]
    common = .08*sign
    spatial = .25*np.eye(9)[np.arange(n) % 9]*sign
    memory = rng.normal(0., .08, (n, 20))
    initial = np.concatenate((cases["q"], cases["v"], memory, common, spatial), axis=1)
    cases["groups"] = [f"excitation-{i//2:02d}" for i in range(n)]
    return cases, mass, initial, eq


def _reference(mass, state, power, equilibrium, coeffs, scale, budget, *, fifth=0., train_mask=None):
    coarse, fine = [truth_rollout(mass, state, power, equilibrium, coeffs, count=810, step=step,
                                 spend=budget.spend, fifth=fifth) for step in (.00125, .000625)]
    if scale is None:
        scale = np.sqrt(np.mean(fine[train_mask, 11:, :20]**2, axis=(0, 1))).clip(min=1e-4)
    gaps = np.max(np.abs((coarse[:, :, :20]-fine[:, :, :20])/scale), axis=(1, 2))
    labels = []
    for a, b, ref, gap in zip(coarse, fine, equilibrium, gaps, strict=True):
        left, right = [classify(x[:11, :10], x[:11, 10:20], x[10:, :20], ref, mass) for x in (a, b)]
        right["reference_qualified"] = bool(gap <= 1e-6 and align_classes(left, right))
        labels.append(right)
    return fine, labels, float(gaps.max())


def _fit(model, observed, roles, power, equilibrium, scale, hscale, settings, budget):
    train, val = (np.flatnonzero(roles == role) for role in ("train", "val"))
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"], foreach=False, fused=False)
    train_curve, val_curve, removed = [], [], []
    budget.spend("fits")
    best = None
    failure = None
    completed = 0

    def objective(rows, count, training):
        history, u, eq = model._tensor(observed[rows, :11]), power[rows], equilibrium[rows]
        with torch.no_grad():
            inferred = _observe(model, history, u, eq, hscale, 0 if training else count, settings, .005, budget)
        forecast = (model.rollout(inferred["forecast"][:, 0], u, eq, count, max_step=.005, spend=budget.spend)
                    if training else inferred["forecast"])
        residual = (forecast[:, 1:, :20]-model._tensor(observed[rows, 11:11+count]))/scale
        loss = residual.square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite joint trajectory loss")
        return loss

    for update in range(settings["updates"]+1):
        before = snapshot(model)
        if update:
            optimizer.zero_grad(set_to_none=True)
            try:
                value = objective(train, settings["train_frames"], True)
                budget.spend("gradients")
                (value+model.regularization()).backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=False)
                if not torch.isfinite(norm):
                    raise FloatingPointError("nonfinite joint gradient")
                budget.spend("optimizer_updates")
                optimizer.step()
                if not all(torch.isfinite(p).all() for p in model.parameters()):
                    raise FloatingPointError("nonfinite joint coefficient")
            except NUMERICAL_ERRORS as error:
                model.load_state_dict(before, strict=True)
                failure = str(error)
                break
            completed = update
            train_curve.append({"update": update, "nmse_before_update": float(value.detach()), "gradient_norm": float(norm)})
        if update == settings["prune_update"]:
            removed = [name for name, norm in model.group_norms().items()
                       if name not in ("common", "spatial") and float(norm.detach()) < settings["support_threshold"]]
            model.prune_groups(removed)
        if update % settings["val_interval"] == 0:
            budget.spend("val_checks")
            try:
                loss = float(objective(val, settings["val_frames"], False))
                if update >= settings["prune_update"] and (best is None or loss < best[0]):
                    best = (loss, update, snapshot(model))
                val_curve.append({"update": update, "nmse": loss})
                print(f"joint synthetic update {update}: val NMSE={loss:.6g}", flush=True)
            except NUMERICAL_ERRORS as error:
                val_curve.append({"update": update, "nmse": None, "failure": str(error)})
    if best is not None:
        model.load_state_dict(best[2], strict=True)
    model.requires_grad_(False).eval()
    return {"actual_updates": completed, "selected_update": None if best is None else best[1], "failure": failure,
            "train_curve": train_curve, "val_curve": val_curve, "removed_groups": removed,
            "effective_group_norms": {name: float(norm) for name, norm in model.group_norms().items()}}


@torch.no_grad()
def _forecast(model, observed, roles, power, equilibrium, hscale, settings, budget):
    outputs = np.full((2, len(roles), 801, 50), np.nan)
    errors = []
    sv = []
    for role in ROLES:
        ix = np.flatnonzero(roles == role)
        if not len(ix):
            continue
        for index, step in enumerate(STEPS):
            rank = index == 0 and role == "val"
            try:
                result = _observe(model, model._tensor(observed[ix]), power[ix], equilibrium[ix], hscale,
                                  800, settings, step, budget, rank=rank)
                outputs[index, ix] = result["forecast"].cpu().numpy()
                if rank:
                    design = result["observation_jacobian"].cpu().numpy()
                    budget.spend("svd", 3*len(ix))
                    singular = np.linalg.svd(design, compute_uv=False)
                    u, s, _ = np.linalg.svd(design[:, :, :20], full_matrices=False)
                    valid = s > 1e-8*s[:, :1]
                    left = u*valid[:, None, :]
                    topology = design[:, :, 20:]
                    remaining = topology-left@(left.transpose(0, 2, 1)@topology)
                    sv = {"unregularized_history": singular.tolist(),
                          "topology_after_memory_projection": np.linalg.svd(remaining, compute_uv=False).tolist()}
            except NUMERICAL_ERRORS as error:
                errors.append({"role": role, "step": step, "size": len(ix), "error": str(error)})
    return outputs, errors, sv


def _rmse(score, role="val", horizon="8"):
    group = score["roles"][role]["trajectory"][horizon]["full_cohort"]
    return None if group is None else group["nrmse"]["mean"]


@torch.no_grad()
def _mechanisms(model, forecast, truth, mass, equilibrium, coefficients, budget):
    if not np.isfinite(forecast).all():
        return {"spatial_truth_rms": None, "relative_spatial_error": None, "relative_memory_kernel_error": None}
    estimated = model._tensor(forecast[:, :101:10])
    target = truth[:, 10:111:10]
    _, expected_spatial, expected_ports = port_and_edges(target, mass, coefficients)
    components = model._components(estimated, equilibrium[:, None, :])
    ports = components[11].cpu().numpy()
    computed_spatial = (components[13]@model.spatial_basis.T).cpu().numpy()
    expected_spatial = expected_spatial[:, 0]
    estimate = computed_spatial[:, 0]
    projection = operators(mass)[-1]
    actual = model.effective_coefficients()

    def kernels(g, tau, vector):
        c = (projection[None, None, :, None, :]*g[:, :, None, :, :]).reshape(len(g), g.shape[1], 10, 20)
        damping = np.diag(1./np.asarray(tau).ravel())+np.outer(vector, vector)
        budget.spend("kernel_exponentials", 10)
        propagators = np.stack([expm(-.1*lag*damping) for lag in range(1, 11)])
        return c[:, 1:]@propagators[None]@c[:, :1].transpose(0, 1, 3, 2)

    reference = kernels(expected_ports, coefficients["tau"], coefficients["memory_decay_factor"])
    measured = kernels(ports, actual["tau"].cpu().numpy(), actual["memory_decay_factor"].cpu().numpy())
    return {"spatial_truth_rms": float(np.sqrt(np.mean(expected_spatial**2))),
            "relative_spatial_error": float(np.linalg.norm(estimate-expected_spatial)/np.linalg.norm(expected_spatial)),
            "relative_memory_kernel_error": float(np.linalg.norm(measured-reference)/np.linalg.norm(reference))}


def _energy(model, state, power, equilibrium, budget):
    x = model._tensor(state).detach().requires_grad_(True)
    budget.spend("gradients")
    gradient, = torch.autograd.grad(model.storage(x, equilibrium).sum(), x)
    budget.spend("torch_rhs")
    field = model.rhs(x, power, equilibrium)
    q, v, z1, z2 = (x[:, offset:offset+10] for offset in (0, 10, 20, 30))
    edge = q@model.incidence
    local = edge.square()@model.incident_weights.T
    coefficients = model.effective_coefficients()
    root = coefficients["damping_root"]
    edge_gain = coefficients["edge_damping"]
    speed_gain = coefficients["speed_damping"]
    angle_gain = coefficients["angle_damping"]
    damp = ((1.-angle_gain*local/(1.+local))*(v@root).square()).sum(1)
    damp += (edge_gain*edge.square()/(1.+edge.square())*(v@model.incidence).square()).sum(1)
    damp += (speed_gain*v.pow(4)/(1.+v.square())).sum(1)
    memory = (z1.square()/model.tau[0]+z2.square()/model.tau[1]).sum(1)
    memory += (x[:, 20:40]@coefficients["memory_decay_factor"]).square()
    mu = coefficients["mobility"]
    topology = mu[0]*gradient[:, 40].square()+mu[1]*gradient[:, 41:].square().sum(1)
    balance = (v*power).sum(1)-damp-memory-topology
    work = gradient*field
    return float(((work.sum(1)-balance).abs()/work.abs().sum(1).clamp_min(1.)).max().detach())


def execute(root, directory, plan, budget):
    cases, mass, initial, eq = _population(plan)
    roles = np.array(cases["roles"])
    if {role: int((roles == role).sum()) for role in ROLES} != plan["cohorts"]:
        raise ValueError("joint synthetic cohort differs")
    settings, gates = plan["settings"], plan["gates"]
    coeffs = plan["truth"]["coefficients"]
    power = cases["power"]
    train, val = roles == "train", roles == "val"
    truth, labels, step_gap = _reference(mass, initial, power, eq, coeffs, None, budget, train_mask=train)
    visible = truth[:, :, :20]
    scale = np.sqrt(np.mean(visible[train, 11:]**2, axis=(0, 1))).clip(min=1e-4)
    hscale = np.sqrt(np.mean(visible[train, :11]**2, axis=(0, 1))).clip(min=1e-4)
    np.savez_compressed(directory/"truth.npz", truth=truth, power=power, equilibrium=eq, scale=scale,
                        history_scale=hscale, ids=cases["ids"], roles=roles)
    model = JointEnergySINDy.from_mass(mass).to("cuda")
    seed = initialize(model, ({"x": path, "p0": u, "equilibrium": r}
                              for path, u, r in zip(visible[train], power[train], eq[train], strict=True)),
                      budget, frames=settings["train_frames"])
    p, e, s, hs = map(model._tensor, (power, eq, scale, hscale))
    fit = _fit(model, visible, roles, p, e, s, hs, settings, budget)
    fit["weak_seed"] = seed
    _write(directory/"fit.json", fit)
    np.savez_compressed(directory/"selected-equation.npz", **{k: v.cpu().numpy() for k, v in snapshot(model).items()})
    history = visible[:, :11]
    paths, failures, ranks = _forecast(model, history, roles, p, e, hs, settings, budget)
    score = _scores(cases, paths, truth, history, eq, mass, scale, labels, gates)
    _write(directory/"scores.json", {**score, "failures": failures, "val_history_singular_values": ranks})
    np.savez_compressed(directory/"forecasts.npz", forecasts=paths)
    mechanism = _mechanisms(model, paths[0, val], truth[val], mass, e[val], coeffs, budget)
    energy = _energy(model, truth[:, 10], p, e, budget)

    val_cases = {key: [row for row, keep in zip(value, val, strict=True) if keep]
                 for key, value in cases.items() if key in ("ids", "groups", "roles", "excitation")}
    val_roles = roles[val]
    rng = np.random.default_rng(plan["seed"]+2)
    noise = rng.normal(size=history[val].shape)*hscale*settings["noise_sigma"]
    for offset in (0, 10):
        piece = noise[:, :, offset:offset+10]
        piece -= (piece@mass/mass.sum())[:, :, None]
    noisy_history = history[val]+noise
    noisy_paths, noisy_failures, _ = _forecast(model, noisy_history, val_roles, p[val], e[val], hs, settings, budget)
    val_labels = [row for row, keep in zip(labels, val, strict=True) if keep]
    noisy = _scores(val_cases, noisy_paths, truth[val], noisy_history, eq[val], mass, scale, val_labels, gates)
    _write(directory/"val-noise-scores.json", {**noisy, "failures": noisy_failures})

    out_truth, out_labels, out_gap = _reference(mass, initial[val], power[val], eq[val], coeffs, scale, budget,
                                                 fifth=plan["truth"]["out_of_library_fifth"])
    out_history = out_truth[:, :11, :20]
    out_paths, out_failures, _ = _forecast(model, out_history, val_roles, p[val], e[val], hs, settings, budget)
    out = _scores(val_cases, out_paths, out_truth, out_history, eq[val], mass, scale, out_labels, gates)
    _write(directory/"val-fifth-harmonic-scores.json", {**out, "failures": out_failures})
    np.savez_compressed(directory/"val-controls.npz", noisy_history=noisy_history, noisy_forecasts=noisy_paths,
                        fifth_truth=out_truth, fifth_forecasts=out_paths)

    val_3s, val_8s, noise_8s, out_8s = _rmse(score, horizon="3"), _rmse(score), _rmse(noisy), _rmse(out)
    val_rows = [r for r in score["events"] if r["role"] == "val"]
    timing_ok = all(r["prospective_eligible"] and r["truth_label"] is not None and r["predicted_label"] == r["truth_label"]
                    and (r["truth_label"] != "unstable" or (r["trigger_time_error_s"] is not None
                         and r["trigger_time_error_s"] <= gates["timing_error_s"])) for r in val_rows)
    checks = {
        "complete_fit": fit["failure"] is None and fit["actual_updates"] == settings["updates"] and fit["selected_update"] is not None,
        "all_numerically_covered": all(row["qualified"] for record in (score, noisy, out) for row in record["events"]),
        "energy_balance": math.isfinite(energy) and energy <= gates["energy_balance_max"],
        "clean_val_trajectory": val_3s is not None and val_3s <= gates["nrmse_3s"] and val_8s is not None and val_8s <= gates["nrmse_8s"],
        "clean_val_stability_and_timing": timing_ok,
        "mixed_truth_excitations": all({row["label"] for row, split in zip(labels, roles, strict=True)
                                        if split == role and (row["trigger_index"] is None or row["trigger_index"] > 10)}
                                       == {"stable", "unstable"} for role in ("train", "val")),
        "noisy_val_trajectory": noise_8s is not None and noise_8s <= gates["noisy_nrmse_8s"],
        "effective_redistribution": mechanism["relative_spatial_error"] is not None and mechanism["spatial_truth_rms"] >= .02
                                   and mechanism["relative_spatial_error"] <= gates["relative_spatial_error"],
        "effective_memory_kernel": mechanism["relative_memory_kernel_error"] is not None
                                 and mechanism["relative_memory_kernel_error"] <= gates["kernel_relative_error"],
        "out_of_library_rejected": val_8s is not None and val_8s <= gates["nrmse_8s"] and out_8s is not None
                                and out_8s > max(gates["nrmse_8s"], 3.*val_8s),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    return {"engineering_pass": checks["all_numerically_covered"] and checks["energy_balance"],
            "scientific_pass": all(checks.values()), "checks": checks, "fit": fit, "roles": score["roles"],
            "noisy_val_8s_nrmse": noise_8s, "fifth_harmonic_val_8s_nrmse": out_8s, "mechanisms": mechanism,
            "max_energy_balance_residual": energy, "max_truth_step_gap": max(step_gap, out_gap),
            "claim_boundary": "One from-scratch joint fit, independent matched truth and one frozen fifth-harmonic challenge."}
