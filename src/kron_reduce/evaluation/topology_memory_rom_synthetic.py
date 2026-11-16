"""Bounded support recovery for the joint topology and reciprocal-memory ROM."""
from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import torch

from kron_reduce.evaluation.constant_c_observability import write_json
from kron_reduce.evaluation.machine_local_clean_learning import NUMERICAL_ERRORS, snapshot
from kron_reduce.evaluation.machine_local_equation import ROLES, STEPS, align_classes
from kron_reduce.evaluation.machine_local_hidden_oracle import trajectory_scores
from kron_reduce.evaluation.machine_local_topology_transfer import transfer_cases
from kron_reduce.evaluation.machine_local_topology_truth import parameters
from kron_reduce.evaluation.topology_memory_rom_truth import truth_rollout
from kron_reduce.evaluation.trajectory_stability import classify, summarize
from kron_reduce.models.machine_local_topology import predict_from_topology_history
from kron_reduce.models.nonlinear_history import history_with_tangent
from kron_reduce.models.nonlinear_latent import NonlinearPassiveClosure
from kron_reduce.models.topology_memory_rom import TopologyMemoryROM

SUPPORTS = {f"C{c}S{s}H{h}": (bool(c), bool(s), bool(h))
            for c, s, h in itertools.product((0, 1), repeat=3)}
POPULATIONS = {"S0": "C0S0H0", "S1": "C1S1H0", "S2": "C0S0H1",
               "S3": "C1S1H1", "S4": "C0S0H0"}


def _safe(value):
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _safe(value.tolist())
    if isinstance(value, np.generic):
        return _safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write(path, value):
    write_json(path, _safe(value))


def _validate(plan):
    settings, gates = plan["settings"], plan["gates"]
    if (plan["cohorts"] != {"train": 24, "val": 12, "test": 12}
            or plan["ids"] != [f"transfer-{i:02d}" for i in range(48)]
            or settings["updates"] != 12 or settings["val_interval"] != 4
            or settings["train_frames"] != 100 or settings["val_frames"] != 300
            or settings["microbatch"] != 24 or len(settings["start"]) != 3
            or plan["truth"]["topology_stiffness"] != [1., 1.]
            or any(not math.isfinite(float(x)) or float(x) <= 0 for x in
                   (settings["learning_rate"], settings["parameter_max"], settings["memory_radius"],
                    settings["topology_radius"], *settings["start"]))
            or not 0 <= settings["noise_sigma"] <= .01 or settings["prior_weight"] < 0
            or any(not math.isfinite(float(v)) or float(v) < 0 for v in gates.values())):
        raise ValueError("exact bounded ROM cohort, clocks, update cap and finite settings required")


def _cases(plan):
    cases = transfer_cases(plan)
    for key in ("q", "power", "a", "z"):
        cases[key][1::2] = cases[key][::2]
    cases["groups"] = [f"excitation-{i//2:02d}" for i in range(48)]
    groups = {}
    for group, role in zip(cases["groups"], cases["roles"], strict=True):
        groups.setdefault(group, set()).add(role)
    if any(len(value) != 1 for value in groups.values()):
        raise ValueError("an excitation group crosses train/val/test")
    rng = np.random.default_rng(plan["seed"]+1)
    c = rng.uniform(-1., 1., (24, 1))*plan["truth"]["common_initial"]
    xi = rng.normal(size=(24, 9))
    xi *= plan["truth"]["spatial_initial_norm"]/np.linalg.norm(xi, axis=1, keepdims=True)
    cases["c"], cases["xi"] = np.repeat(c, 2, axis=0), np.repeat(xi, 2, axis=0)
    return cases


def _model(params, support, plan, *, truth=False):
    mass, stiffness, damping, port, tau, _ = params
    empty = np.zeros((0, 0))
    backbone = NonlinearPassiveClosure(mass, stiffness, damping, np.zeros((10, 0)), empty, empty)
    common, spatial, saturation = SUPPORTS[support]
    model = TopologyMemoryROM(backbone, port=port, tau=tau, common=common, spatial=spatial,
                              saturation=saturation, train_scales=False).to(plan.get("device", "cuda"))
    model.set_angle_damping(2./3., spatial_mix=1.)
    model.requires_grad_(False)
    values = ([plan["truth"]["common_rate"], plan["truth"]["spatial_rate"],
               plan["truth"]["saturation_strength"]] if truth else plan["settings"]["start"])
    with torch.no_grad():
        model.topology_rates.copy_(model.mass.new_tensor([v for v, active in zip(values[:2], (common, spatial), strict=True) if active]))
        if saturation:
            model.saturation_strength.fill_(values[2])
    if not truth:
        model.topology_rates.requires_grad_(common or spatial)
        if saturation:
            model.saturation_strength.requires_grad_(True)
    return model


def _observer(model, history, power, equilibrium, hscale, count, step, settings, spend, *, jacobian=False):
    return predict_from_topology_history(model, history[..., :10], history[..., 10:20], power,
        equilibrium, hscale, count, max_step=step, spend=spend,
        memory_radius=settings["memory_radius"], dynamic_radius=settings["topology_radius"],
        prior_weight=settings["prior_weight"], fit_dynamic_coupling=bool(model.dynamic_dimension),
        return_observation_jacobian=jacobian)


def _objective(model, truth, indices, power, equilibrium, scale, hscale, count, settings, spend):
    x = model._tensor(truth[indices])
    output = _observer(model, x[:, :11, :20], model._tensor(power[indices]),
        model._tensor(equilibrium[indices]), model._tensor(hscale), count, .005, settings, spend)
    error = ((output["forecast"][:, 1:, :20]-x[:, 11:11+count, :20])/model._tensor(scale)).square()
    if not torch.isfinite(error).all():
        raise FloatingPointError("nonfinite complete-cohort trajectory objective")
    return error.mean()


def _readout(model, support):
    flags = SUPPORTS[support]
    return {"support": support, "common_rate": float(model.topology_rates[0].detach()) if flags[0] else 0.,
            "spatial_rate": float(model.topology_rates[-1].detach()) if flags[1] else 0.,
            "saturation_strength": float(model.saturation_strength.detach()) if flags[2] else 0.}


def _fit(model, support, truth, roles, power, equilibrium, scale, hscale, plan, budget, directory):
    budget.spend("fits")
    settings = plan["settings"]
    train, val = (np.flatnonzero(roles == role) for role in ("train", "val"))
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=settings["learning_rate"], foreach=False, fused=False) if parameters else None
    best = None
    train_curve, val_curve = [], []
    completed, failure = 0, None

    def validate(update):
        nonlocal best
        budget.spend("val_checks")
        np.savez_compressed(directory/f"checkpoint-{update:03d}.npz",
                            **{k: v.cpu().numpy() for k, v in snapshot(model).items()})
        try:
            with torch.no_grad():
                value = float(_objective(model, truth, val, power, equilibrium, scale, hscale,
                                         settings["val_frames"], settings, budget.spend))
            if best is None or value < best[0]:
                best = value, update, snapshot(model)
            val_curve.append({"update": update, "nrmse_squared": value, "equation": _readout(model, support)})
        except NUMERICAL_ERRORS as error:
            val_curve.append({"update": update, "nrmse_squared": None, "failure": str(error)})

    validate(0)
    for update in range(1, settings["updates"]+1 if optimizer is not None else 1):
        optimizer.zero_grad(set_to_none=True)
        before = snapshot(model)
        total = 0.
        try:
            for start in range(0, len(train), settings["microbatch"]):
                indices = train[start:start+settings["microbatch"]]
                value = _objective(model, truth, indices, power, equilibrium, scale, hscale,
                                   settings["train_frames"], settings, budget.spend)
                weighted = value*(len(indices)/len(train))
                budget.spend("gradients")
                weighted.backward()
                total += float(weighted.detach())
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
                raise FloatingPointError("nonfinite or missing shared gradient")
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
            budget.spend("optimizer_updates")
            optimizer.step()
            with torch.no_grad():
                for parameter in parameters:
                    parameter.clamp_(0., settings["parameter_max"])
            if any(not torch.isfinite(p).all() for p in parameters):
                raise FloatingPointError("nonfinite shared parameter")
        except NUMERICAL_ERRORS as error:
            model.load_state_dict(before)
            failure = str(error)
            break
        completed = update
        train_curve.append({"update": update, "nrmse_squared_before_update": total, "gradient_norm": float(norm)})
        if update % settings["val_interval"] == 0:
            validate(update)
    if best is not None:
        model.load_state_dict(best[2])
    model.requires_grad_(False).eval()
    record = {"completed_updates": completed, "selected_update": None if best is None else best[1],
              "selected_val_nmse": None if best is None else best[0], "failure": failure,
              "train_curve": train_curve, "val_curve": val_curve, "equation": _readout(model, support)}
    _write(directory/"fit.json", record)
    np.savez_compressed(directory/"selected-equation.npz", **{k: v.cpu().numpy() for k, v in snapshot(model).items()})
    return record


def _rank_rows(jacobian, settings, floor, spend):
    design = np.asarray(jacobian)
    radii = np.array([settings["memory_radius"]]*10+[settings["topology_radius"]]*(design.shape[-1]-10))
    output = []
    for matrix in design*radii:
        record = {}
        def spectrum(label, block, record=record):
            if block.shape[1] == 0:
                record[label] = {"dimension": 0, "rank": 0, "singular": [], "condition": None}
                return np.empty((len(block), 0))
            spend("svd")
            left, singular, _ = np.linalg.svd(block, full_matrices=False)
            reference = singular[0] if label == "joint" else record["joint"]["singular"][0]
            cutoff = max(floor, 64*np.finfo(float).eps*max(block.shape))*reference
            rank = int(np.count_nonzero(singular > cutoff))
            record[label] = {"dimension": block.shape[1], "rank": rank, "singular": singular.tolist(),
                             "condition": float(singular[0]/singular[-1]) if singular[-1] else None,
                             "absolute_rank_cutoff": float(cutoff), "joint_reference_smax": float(reference)}
            return left[:, :rank]
        spectrum("joint", matrix)
        memory_basis = spectrum("memory", matrix[:, :10])
        spectrum("topology", matrix[:, 10:])
        residual = matrix[:, 10:]-memory_basis @ (memory_basis.T @ matrix[:, 10:])
        spectrum("topology_given_memory", residual)
        record["linearization_iterate_zero_based"] = 4
        output.append(record)
    return output


def _pack_truth(truth, truth_support, support):
    source, target = SUPPORTS[truth_support], SUPPORTS[support]
    parts = [truth[..., :30]]
    if target[0]:
        parts.append(truth[..., 30:31] if source[0] else np.zeros((*truth.shape[:-1], 1)))
    if target[1]:
        offset = 30+int(source[0])
        parts.append(truth[..., offset:offset+9] if source[1] else np.zeros((*truth.shape[:-1], 9)))
    return np.concatenate(parts, axis=-1)


@torch.no_grad()
def _forecast_group(model, ids, observed_history, power, equilibrium, hscale, plan, budget, *, origin, ranks):
    size, dimension = len(ids), 30+model.dynamic_dimension
    forecasts = np.full((2, size, 801, dimension), np.nan)
    histories = np.full((2, size, 11, dimension), np.nan)
    errors, rank, jacobian = [], None, None
    for slot, step in enumerate(STEPS):
        try:
            if origin is None:
                result = _observer(model, model._tensor(observed_history), model._tensor(power),
                    model._tensor(equilibrium), model._tensor(hscale), 800, step, plan["settings"],
                    budget.spend, jacobian=ranks and slot == 0)
                forecasts[slot] = result["forecast"].cpu().numpy()
                histories[slot] = result["history"].cpu().numpy()
                if ranks and slot == 0:
                    before = _rank_rows(result["observation_jacobian"].cpu().numpy(), plan["settings"],
                                        plan["gates"]["rank_relative_floor"], budget.spend)
                    _, tangent = history_with_tangent(model, result["history"][:, 0], model._tensor(power),
                        model._tensor(equilibrium), max_step=step, spend=budget.spend)
                    jacobian = (tangent[:, 1:, :20]/model._tensor(hscale)[None, None, :, None]).flatten(1, 2).cpu().numpy()
                    rank = _rank_rows(jacobian, plan["settings"], plan["gates"]["rank_relative_floor"], budget.spend)
                    for after, previous in zip(rank, before, strict=True):
                        after["linearization_iterate_zero_based"] = None
                        after["linearization_point"] = "accepted_hidden_initial_after_five_updates"
                        after["before_final_update"] = previous
            else:
                forecasts[slot] = model.rollout(model._tensor(origin), model._tensor(power),
                    model._tensor(equilibrium), 800, max_step=step, spend=budget.spend).cpu().numpy()
        except NUMERICAL_ERRORS as error:
            errors.append({"step": step, "affected_ids": ids, "failure": str(error)})
    return forecasts, histories, errors, rank, jacobian


def _forecast(model, roles, observed_history, power, equilibrium, hscale, plan, budget, *, origin=None, ranks=False):
    size, dimension = len(roles), 30+model.dynamic_dimension
    forecasts = np.full((2, size, 801, dimension), np.nan)
    histories = np.full((2, size, 11, dimension), np.nan)
    errors, rank = [], [None]*size if ranks else None
    jacobian = np.full((size, 200, model.latent_dimension), np.nan) if ranks else None
    for role in ROLES:
        indices = np.flatnonzero(roles == role)
        ids = [plan["ids"][i] for i in indices]
        state = None if origin is None else origin[indices]
        prediction, history, failures, ranks_by_event, design = _forecast_group(
            model, ids, observed_history[indices], power[indices], equilibrium[indices], hscale,
            plan, budget, origin=state, ranks=ranks,
        )
        forecasts[:, indices], histories[:, indices] = prediction, history
        errors.extend(failures)
        if ranks_by_event is not None:
            jacobian[indices] = design
            for i, record in zip(indices, ranks_by_event, strict=True):
                rank[i] = record
    return forecasts, histories, errors, rank, jacobian


def _scores(cases, paths, truth, observed, equilibrium, mass, scale, truth_classes, gates, rank=None):
    events = []
    for i, case in enumerate(cases["ids"]):
        fine, refined = paths[:, i]
        finite = bool(np.isfinite(paths[:, i]).all())
        gap = float(np.max(np.abs((fine[:, :20]-refined[:, :20])/scale))) if finite else None
        coi = float(np.max(np.abs(paths[:, i, :, :20].reshape(2, 801, 2, 10) @ mass))) if finite else None
        classes = [classify(observed[i, :, :10], observed[i, :, 10:20], p[:, :20], equilibrium[i], mass)
                   for p in (fine, refined)] if finite else [None, None]
        target = truth_classes[i]
        numerical_issues = []
        if not finite:
            numerical_issues.append("nonfinite_or_failed_cohort_prediction")
        else:
            if gap > gates["prediction_step_max"]:
                numerical_issues.append("integration_step_disagreement")
            if coi > gates["coi_max"]:
                numerical_issues.append("coi_drift")
            if not align_classes(*classes):
                numerical_issues.append("stability_step_disagreement")
        qualified = target["reference_qualified"] and not numerical_issues
        hit = None if classes[0] is None else classes[0]["trigger_index"]
        timing = abs(hit-target["trigger_index"])*.01 if hit is not None and target["trigger_index"] is not None else None
        row = {"id": case, "role": cases["roles"][i], "group": cases["groups"][i],
               "excitation": cases["excitation"][i], "truth_label": target["label"],
               "predicted_label": None if classes[0] is None else classes[0]["label"],
               "qualified": qualified, "numerical_failure": bool(numerical_issues), "step_gap": gap, "coi_error": coi,
               "qualification_issues": numerical_issues+([] if target["reference_qualified"] else ["truth_reference"]),
               "prospective_eligible": target["reference_qualified"] and
                   (target["trigger_index"] is None or target["trigger_index"] > 10),
               "unsettled": qualified and classes[0]["label"] is None,
               "trajectory": trajectory_scores(fine, truth[i], scale) if finite else None,
               "truth_trigger_index": target["trigger_index"], "predicted_trigger_index": hit,
               "trigger_time_error_s": timing, "initial_identifiability": None if rank is None else rank[i]}
        events.append(row)
    roles = {}
    for role in ROLES:
        group = [row for row in events if row["role"] == role]
        qualified = [row for row in group if row["qualified"] and row["trajectory"] is not None]
        trajectories = {}
        for horizon in ("3", "8"):
            statistics = {}
            for quantity in ("nrmse", "q_rmse", "v_rmse"):
                values = np.array([row["trajectory"][horizon][quantity] for row in qualified])
                statistics[quantity] = ({"mean": float(values.mean()), "median": float(np.median(values)),
                                         "pooled": float(np.sqrt(np.mean(values**2)))} if len(values) else None)
            trajectories[horizon] = {"allocated": len(group), "qualified": len(qualified),
                "coverage": len(qualified)/len(group) if group else None,
                "full_cohort": statistics if len(qualified) == len(group) else None,
                "qualified_only": statistics, "weighting": "event_equal; pooled over equal-length forecasts"}
        roles[role] = {"stability": summarize(group), "nrmse_3s_max": max(
            (row["trajectory"]["3"]["nrmse"] for row in group), default=None) if all(row["trajectory"] for row in group) else None,
            "nrmse_8s_max": max((row["trajectory"]["8"]["nrmse"] for row in group), default=None)
            if all(row["trajectory"] for row in group) else None, "trajectory": trajectories}
    return {"events": events, "roles": roles}


def _adequate(score, gates, *, noisy=False):
    rows = [r for r in score["events"] if r["role"] == "val"]
    return all(r["qualified"] and r["prospective_eligible"] and r["trajectory"] is not None
        and r["trajectory"]["8"]["nrmse"] <= gates["noisy_nrmse_8s" if noisy else "nrmse_8s"]
        and (noisy or r["trajectory"]["3"]["nrmse"] <= gates["nrmse_3s"])
        and r["predicted_label"] == r["truth_label"] and r["truth_label"] is not None
        and (r["truth_label"] != "unstable" or (r["trigger_time_error_s"] is not None
             and r["trigger_time_error_s"] <= gates["timing_error_s"])) for r in rows)


def _kernel_error(model, support, truth, truth_gamma, scale, roles):
    q = truth[roles != "test", :, :10]
    incidence = model.incidence.detach().cpu().numpy()
    incident = np.abs(incidence); incident /= incident.sum(1, keepdims=True)
    projection = model.power_projection.detach().cpu().numpy()
    g = model.port_parameter.detach().exp().cpu().numpy()
    tau = model.tau.detach().cpu().numpy()
    gamma = float(model.saturation_strength.detach()) if SUPPORTS[support][2] else 0.
    factor = scale[10:][None, :]/(model.mass.cpu().numpy()[:, None]*scale[10:, None])
    def port(x, strength):
        s = ((x @ incidence)**2) @ incident.T
        return projection[None, :, :]*(g/(1.+strength*s))[:, None, :]
    numerator = denominator = 0.
    for start in (10, 110):
        for lag in (10, 50, 100):
            decay = np.exp(-.01*lag/tau)
            kernels = [np.einsum("bij,j,bkj->bik", port(q[:, start+lag], strength), decay,
                                 port(q[:, start], strength))*factor for strength in (gamma, truth_gamma)]
            numerator += float(np.square(kernels[0]-kernels[1]).sum())
            denominator += float(np.square(kernels[1]).sum())
    return math.sqrt(numerator/denominator) if denominator else None


def _physical(model, state, power, equilibrium, budget):
    x = model._tensor(state).detach().requires_grad_(True)
    budget.spend("gradients")
    gradient, = torch.autograd.grad(model.storage(x, model._tensor(equilibrium)).sum(), x)
    budget.spend("torch_rhs")
    derivative = model.rhs(x, model._tensor(power), model._tensor(equilibrium))
    v, z = x[:, 10:20], x[:, 20:30]
    root = model._damping_root()
    local = (x[:, :10, None]-x[:, None, :10]).square().sum(2)/9
    drag = v @ model.effective_damping.T-((v @ root)*(2./3.*local/(1.+local))) @ root
    rates = []
    if model.dynamic_dimension in (1, 10):
        rates.append(model.topology_rates[0:1])
    if model.dynamic_dimension in (9, 10):
        rates.append(model.topology_rates[-1:].expand(9))
    loss = (v*drag).sum(1)+(z.square()/model.tau).sum(1)
    if rates:
        loss = loss+(gradient[:, 30:].square()*torch.cat(rates)).sum(1)
    expected = (v*model._tensor(power)).sum(1)-loss
    return float(((gradient*derivative).sum(1)-expected).abs().div(
        (gradient*derivative).abs().sum(1).clamp_min(1.)).max().detach())


def execute_saturation(root, directory, plan, budget):
    """Recover one positive memory saturation coefficient on the paired S2 truth."""
    _validate(plan)
    cases = _cases(plan)
    params = list(parameters())
    params[4] = np.geomspace(*plan["tau_range"], 10)
    mass, _, _, _, _, reference = params
    roles = np.asarray(cases["roles"])
    train, val = (np.flatnonzero(roles == role) for role in ("train", "val"))
    power = cases["power"]
    equilibrium = np.broadcast_to(reference, power.shape).copy()
    settings, gates = plan["settings"], plan["gates"]
    gamma = plan["truth"]["saturation_strength"]
    coarse, truth = [truth_rollout(params, cases["q"], cases["v"], power, cases["z"], count=810,
                     max_step=step, spend=budget.spend, gamma_s=gamma) for step in (.00125, .000625)]
    observed = truth[:, :11, :20]
    scale = np.maximum(np.sqrt(np.mean(truth[train, 11:, :20]**2, axis=(0, 1))), 1e-4)
    hscale = np.maximum(np.sqrt(np.mean(observed[train]**2, axis=(0, 1))), 1e-4)
    labels = [[classify(path[:11, :10], path[:11, 10:20], path[10:, :20], reference, mass)
               for path in paths] for paths in (coarse, truth)]
    truth_gaps = np.max(np.abs((coarse[:, :, :20]-truth[:, :, :20])/scale), axis=(1, 2))
    for a, b, gap in zip(*labels, truth_gaps, strict=True):
        b["reference_qualified"] = bool(np.isfinite(gap) and gap <= gates["truth_step_max"] and align_classes(a, b))
    np.savez_compressed(directory/"truth.npz", fine=coarse, refined=truth, ids=cases["ids"], roles=roles,
                        power=power, equilibrium=equilibrium, scale=scale, history_scale=hscale)

    anchor = _model(params, "C0S0H0", plan)
    budget.spend("val_checks")
    with torch.no_grad():
        baseline = float(_objective(anchor, truth, val, power, equilibrium, scale, hscale,
                                    settings["val_frames"], budget.spend))
    model = _model(params, "C0S0H1", plan)
    fit = _fit(model, "C0S0H1", truth, roles, power, equilibrium, scale, hscale, plan, budget, directory)
    print(f"S2 fit {fit['completed_updates']} updates; selected {fit['equation']}", flush=True)
    paths, histories, failures, _, _ = _forecast(model, roles, observed, power, equilibrium, hscale, plan, budget)
    np.savez_compressed(directory/"forecasts.npz", forecasts=paths, histories=histories)
    score = _scores(cases, paths, truth, observed, equilibrium, mass, scale, labels[1], gates)
    _write(directory/"scores.json", {**score, "failures": failures})
    energy = _physical(model, truth[:, 10], power, equilibrium, budget)
    learned = fit["equation"]["saturation_strength"]
    kernel = _kernel_error(model, "C0S0H1", truth, gamma, scale, roles)
    recovery = abs(learned-gamma)/gamma
    checks = {
        "fit_completed": fit["failure"] is None and fit["completed_updates"] == settings["updates"],
        "energy_balance": np.isfinite(energy) and energy <= gates["energy_balance_max"],
        "integration_and_coverage": all(row["qualified"] for row in score["events"]),
        "all_stability_classifiable": all(row["stability"]["resolved"] == row["stability"]["allocated"]
                                          for row in score["roles"].values()),
        "val_trajectory_and_stability": _adequate(score, gates),
        "val_improvement": fit["selected_val_nmse"] is not None and baseline > 0
                           and fit["selected_val_nmse"] <= (1.-gates["support_relative_tolerance"])*baseline,
        "coefficient_recovery": recovery <= gates["parameter_relative_error"],
        "memory_kernel_recovery": kernel is not None and kernel <= gates["kernel_relative_error"],
    }
    return {"engineering_pass": bool(checks["energy_balance"] and checks["integration_and_coverage"]),
            "scientific_pass": bool(all(checks.values())), "checks": {k: bool(v) for k, v in checks.items()},
            "fit": fit, "baseline_val_nmse": baseline, "truth_gamma": gamma, "coefficient_relative_error": recovery,
            "kernel_relative_error": kernel, "max_energy_balance_residual": energy,
            "max_truth_step_gap": float(truth_gaps.max()), "roles": score["roles"],
            "claim_boundary": "One S2 fit with known fixed topology, damping and memory poles; clean causal histories."}


def execute(root, directory, plan, budget):
    """Fit eight exact supports per truth once, freeze, then score every role."""
    _validate(plan)
    directory = Path(directory)
    cases = _cases(plan)
    params = list(parameters())
    params[4] = np.geomspace(*plan["tau_range"], 10)
    mass, _, _, _, _, reference = params
    roles = np.array(cases["roles"])
    power = cases["power"]
    equilibrium = np.broadcast_to(reference, power.shape).copy()
    settings, limits, truth_settings = plan["settings"], plan["gates"], plan["truth"]
    records = {}
    for population, true_support in POPULATIONS.items():
        destination = directory/population
        destination.mkdir()
        flags = SUPPORTS[true_support]
        topology = "both" if flags[0] and flags[1] else "common" if flags[0] else "spatial" if flags[1] else "none"
        gamma = truth_settings["saturation_strength"] if flags[2] else 0.
        truth_options = {"topology": topology, "c": cases["c"][:, 0] if flags[0] else None,
            "xi": cases["xi"] if flags[1] else None, "gamma_s": gamma,
            "topology_dynamics": ((truth_settings["common_rate"], truth_settings["spatial_rate"]),
                                  tuple(truth_settings["topology_stiffness"])) if any(flags[:2]) else None,
            "fifth_harmonic": truth_settings["fifth_harmonic"] if population == "S4" else 0.}
        paths = [truth_rollout(params, cases["q"], cases["v"], power, cases["z"], count=810,
                  max_step=step, spend=budget.spend, **truth_options) for step in (.00125, .000625)]
        truth = paths[1]
        scale = np.maximum(np.sqrt(np.mean(truth[roles == "train", 11:, :20]**2, axis=(0, 1))), 1e-4)
        hscale = np.maximum(np.sqrt(np.mean(truth[roles == "train", :11, :20]**2, axis=(0, 1))), 1e-4)
        truth_classes = [[classify(p[:11, :10], p[:11, 10:20], p[10:, :20], reference, mass) for p in path] for path in paths]
        gaps = np.max(np.abs((paths[0][:, :, :20]-truth[:, :, :20])/scale), axis=(1, 2))
        for a, b, gap in zip(*truth_classes, gaps, strict=True):
            b["reference_qualified"] = bool(np.isfinite(gap) and gap <= limits["truth_step_max"] and align_classes(a, b))
        rng = np.random.default_rng(plan["seed"]+100+int(population[1]))
        noise = rng.normal(size=(48, 11, 20))*hscale*settings["noise_sigma"]
        for offset in (0, 10):
            noise[:, :, offset:offset+10] -= (noise[:, :, offset:offset+10] @ mass/mass.sum())[:, :, None]
        observed = {"clean": truth[:, :11, :20], "noise": truth[:, :11, :20]+noise}
        np.savez_compressed(destination/"truth.npz", fine=paths[0], refined=truth, ids=cases["ids"], roles=roles,
            groups=cases["groups"], power=power, equilibrium=equilibrium, scale=scale, history_scale=hscale, history_noise=noise)
        models, fits = {}, {}
        for support in SUPPORTS:
            folder = destination/support
            folder.mkdir()
            model = _model(params, support, plan)
            fits[support] = _fit(model, support, truth, roles, power, equilibrium, scale, hscale, plan, budget, folder)
            models[support] = model
            print(f"{population}/{support}: updates={fits[support]['completed_updates']} val={fits[support]['selected_val_nmse']}", flush=True)
        eligible = [key for key, fit in fits.items() if fit["selected_val_nmse"] is not None and fit["failure"] is None]
        selected = None
        if eligible:
            best = min(fits[key]["selected_val_nmse"] for key in eligible)
            near = [key for key in eligible if fits[key]["selected_val_nmse"] <=
                    best*(1+limits["support_relative_tolerance"])+limits["support_absolute_tolerance"]]
            selected = min(near, key=lambda key: (sum(SUPPORTS[key]), fits[key]["selected_val_nmse"], key))
        _write(destination/"support-selection.json", {"selected": selected, "uses": "val 3 s only",
            "expected": None if population == "S4" else true_support, "fits": fits})
        results = {}
        selected_initial, selected_estimated_origin = None, None
        for support, model in models.items():
            folder = destination/support
            scores = {}
            for mode, history in observed.items():
                forecast, estimated, failures, rank, jacobian = _forecast(model, roles, history, power, equilibrium,
                    hscale, plan, budget, ranks=mode == "clean")
                score = _scores(cases, forecast, truth, history, equilibrium, mass, scale, truth_classes[1], limits, rank)
                if support == selected and mode == "clean":
                    selected_initial = estimated[0, :, 0, 20:]
                    selected_estimated_origin = estimated[0, :, -1, 20:]
                score["failures"] = failures
                arrays = {"forecasts": forecast, "histories": estimated}
                if jacobian is not None:
                    arrays["observation_jacobian"] = jacobian
                np.savez_compressed(folder/f"{mode}-forecasts.npz", **arrays)
                _write(folder/f"{mode}-scores.json", score)
                scores[mode] = score
            scores["kernel_relative_error"] = _kernel_error(model, support, truth, gamma, scale, roles)
            results[support] = scores
        known = _model(params, true_support, plan, truth=True)
        origin = _pack_truth(truth[:, 10], true_support, true_support)
        oracle, _, errors, _, _ = _forecast(known, roles, observed["clean"], power, equilibrium, hscale,
                                            plan, budget, origin=origin)
        oracle_score = _scores(cases, oracle, truth, observed["clean"], equilibrium, mass, scale, truth_classes[1], limits)
        oracle_score["failures"] = errors
        np.savez_compressed(destination/"known-coefficient-true-origin.npz", forecasts=oracle)
        _write(destination/"known-coefficient-true-origin.json", oracle_score)
        energy = _physical(known, _pack_truth(truth[roles != "test", 10], true_support, true_support),
                           power[roles != "test"], equilibrium[roles != "test"], budget)
        chosen = None if selected is None else results[selected]
        selected_oracle_score, initial_error, origin_error = None, None, None
        if selected is not None:
            selected_model = models[selected]
            selected_origin = _pack_truth(truth[:, 10], true_support, selected)
            selected_oracle, _, oracle_errors, _, _ = _forecast(selected_model, roles, observed["clean"],
                power, equilibrium, hscale, plan, budget, origin=selected_origin)
            selected_oracle_score = _scores(cases, selected_oracle, truth, observed["clean"], equilibrium,
                                             mass, scale, truth_classes[1], limits)
            selected_oracle_score["failures"] = oracle_errors
            np.savez_compressed(destination/"selected-coefficient-true-origin.npz", forecasts=selected_oracle)
            _write(destination/"selected-coefficient-true-origin.json", selected_oracle_score)
            if selected == true_support:
                true_initial = _pack_truth(truth[:, 0], true_support, selected)[:, 20:]
                initial_error = np.linalg.norm(selected_initial-true_initial, axis=1)/np.maximum(
                    np.linalg.norm(true_initial, axis=1), .1)
                true_origin = _pack_truth(truth[:, 10], true_support, selected)[:, 20:]
                origin_error = np.linalg.norm(selected_estimated_origin-true_origin, axis=1)/np.maximum(
                    np.linalg.norm(true_origin, axis=1), .1)
        recovery = None
        if selected is not None and population != "S4":
            equation = fits[selected]["equation"]
            errors = [abs(equation[key]/truth_settings[key]-1.) for key, active in
                      zip(("common_rate", "spatial_rate", "saturation_strength"), flags, strict=True) if active]
            recovery = max(errors, default=0.)
        rank_ok = chosen is not None and all(r["initial_identifiability"] is not None
            and r["initial_identifiability"]["joint"]["rank"] == r["initial_identifiability"]["joint"]["dimension"]
            and r["initial_identifiability"]["topology_given_memory"]["rank"] ==
                r["initial_identifiability"]["topology_given_memory"]["dimension"]
            for r in chosen["clean"]["events"] if r["role"] == "val")
        val_errors = None if initial_error is None else np.array((initial_error, origin_error))[:, roles == "val"]
        initial_ok = (rank_ok and val_errors is not None and np.isfinite(val_errors).all()
                      and float(val_errors.max()) <= limits["initial_relative_error"])
        controls_ok = (all(fit["failure"] is None and fit["selected_val_nmse"] is not None for fit in fits.values())
            and all(all(r["qualified"] for r in result[mode]["events"] if r["role"] == "val")
                    for result in results.values() for mode in ("clean", "noise")))
        s4_numeric = all(all(r["qualified"] for r in result["clean"]["events"] if r["role"] == "val") for result in results.values())
        gates = {"truth": all(r["reference_qualified"] for r, role in zip(truth_classes[1], roles, strict=True)
                              if role != "test"),
            "mixed_truth": all({r["label"] for r, role in zip(truth_classes[1], roles, strict=True) if role == target}
                               == {"stable", "unstable"} for target in ("train", "val")),
            "energy": energy <= limits["energy_balance_max"],
            "complete_deletion_controls": controls_ok,
            "known_coefficient_true_origin": _adequate(oracle_score, limits) if population != "S4" else None,
            "support_recovery": selected == true_support if population != "S4" else None,
            "clean_prediction": chosen is not None and _adequate(chosen["clean"], limits) if population != "S4" else None,
            "noisy_prediction": chosen is not None and _adequate(chosen["noise"], limits, noisy=True) if population != "S4" else None,
            "parameter_recovery": recovery is not None and recovery <= limits["parameter_relative_error"] if population != "S4" else None,
            "kernel_recovery": chosen is not None and chosen["kernel_relative_error"] is not None
                               and chosen["kernel_relative_error"] <= limits["kernel_relative_error"] if population != "S4" else None,
            "history_identifiability": rank_ok if population != "S4" else None,
            "initial_state_recovery": initial_ok if population != "S4" else None,
            "selected_coefficient_true_origin": selected_oracle_score is not None and
                _adequate(selected_oracle_score, limits) if population != "S4" else None,
            "outside_library_rejection": s4_numeric and all(not _adequate(result["clean"], limits)
                for result in results.values()) if population == "S4" else None}
        record = {"selected_support": selected, "expected_support": None if population == "S4" else true_support,
            "outside_library_selected_extra_support": (selected != true_support
                if population == "S4" and selected is not None else None),
            "gates": {key: {"status": "NOT_APPLICABLE" if value is None else "PASS" if value else "FAIL"}
                      for key, value in gates.items()}, "energy_residual_max": energy,
            "parameter_relative_error": recovery, "truth_step_max": float(gaps.max()),
            "initial_relative_error": None if initial_error is None else initial_error.tolist(),
            "origin_relative_error": None if origin_error is None else origin_error.tolist(),
            "initial_state_claim": ("not_applicable_outside_library" if population == "S4" else
                "recovered_on_val" if initial_ok else "rank_sufficient_but_recovery_failed" if rank_ok else
                "unresolved_from_100ms_history"),
            "results": {key: {"clean": value["clean"]["roles"], "noise": value["noise"]["roles"],
                               "kernel_relative_error": value["kernel_relative_error"]} for key, value in results.items()},
            "engineering_pass": gates["truth"] and gates["energy"],
            "scientific_pass": all(value for value in gates.values() if value is not None)}
        _write(destination/"summary.json", record)
        records[population] = record
    return _safe({"populations": records, "engineering_pass": all(r["engineering_pass"] for r in records.values()),
        "scientific_pass": all(r["scientific_pass"] for r in records.values()),
        "claim_boundary": "Fixed synthetic truth, fixed fundamental backbone and three new shared coefficients; complete group-separated rosters. No real-data performance or independent real-data confirmation."})
