"""One causal closed-loop fit of shared curvature-matched recovery force terms."""
from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import lsq_linear

from kron_reduce.evaluation.machine_local_clean_learning import NUMERICAL_ERRORS, snapshot
from kron_reduce.evaluation.machine_local_equation import align_classes
from kron_reduce.evaluation.machine_local_topology_truth import parameters
from kron_reduce.evaluation.topology_memory_rom_synthetic import (
    _cases, _forecast, _objective, _observer, _physical, _scores, _write,
)
from kron_reduce.evaluation.topology_memory_rom_truth import truth_rollout
from kron_reduce.evaluation.topology_momentum_work import account_path
from kron_reduce.evaluation.trajectory_stability import classify
from kron_reduce.models.nonlinear_latent import NonlinearPassiveClosure
from kron_reduce.models.topology_memory_rom import TopologyMemoryROM


@torch.no_grad()
def _warm_start(model, train, power, equilibrium, hscale, settings, budget):
    tensor = model._tensor
    start = _observer(model, tensor(train[:, :11]), tensor(power), tensor(equilibrium),
                      tensor(hscale), 0, .005, settings, budget.spend)["forecast"][:, 0, 20:].cpu().numpy()
    mass, incidence, stiffness, damping, port, tau, root, projection = (
        item.cpu().numpy() for item in (model.mass, model.incidence, model.effective_stiffness,
            model.effective_damping, model.coupling, model.tau, model._damping_root(), model.power_projection)
    )
    blocks = (mass, incidence, stiffness, damping, port, tau)
    design, target = [], []
    for path, u, eq, z0 in zip(train[:, 10:111], power, equilibrium, start, strict=True):
        budget.spend("weak_events")
        residual = account_path(path, u, eq, blocks, .01, z0, root, 8)["momentum_residual"]
        ref = eq @ incidence
        angle = path[:, :10] @ incidence + ref
        base = np.sin(angle)-np.sin(ref)
        columns = []
        for order in (2, 3):
            restoring = np.cos(ref)*(np.sin(order*angle)-np.sin(order*ref))-order*np.cos(order*ref)*base
            force = -(restoring*stiffness) @ (projection @ incidence).T
            columns.append(.005*(force[:-1]+force[1:]))
        design.append(np.stack(columns, axis=-1).reshape(20, 5, 10, 2).sum(1).reshape(-1, 2))
        target.append(residual.reshape(20, 5, 10).sum(1).ravel())
    matrix, rhs = np.concatenate(design), np.concatenate(target)
    budget.spend("svd")
    singular = np.linalg.svd(matrix, compute_uv=False)
    budget.spend("weak_solves")
    bound = settings["coefficient_bound"]
    solution = lsq_linear(matrix, rhs, bounds=(-bound, bound), max_iter=20)
    if not solution.success or not np.isfinite(solution.x).all():
        raise FloatingPointError("bounded weak-form initialization failed")
    return solution.x, singular


def execute(root, directory, plan, budget):
    cases = _cases(plan)
    params = list(parameters())
    params[4] = np.geomspace(*plan["tau_range"], 10)
    mass, stiffness, damping, port, tau, reference = params
    roles = np.asarray(cases["roles"])
    train, val = (np.flatnonzero(roles == role) for role in ("train", "val"))
    settings, gates = plan["settings"], plan["gates"]
    power = cases["power"]
    equilibrium = np.broadcast_to(reference, power.shape).copy()
    true_coefficients = np.asarray(plan["truth"]["restoring_harmonics"])
    coarse, truth = [truth_rollout(params, cases["q"], cases["v"], power, cases["z"], count=810,
                     max_step=step, spend=budget.spend, restoring_harmonics=true_coefficients)
                     for step in (.00125, .000625)]
    observed = truth[:, :, :20]
    scale = np.maximum(np.sqrt(np.mean(observed[train, 11:]**2, axis=(0, 1))), 1e-4)
    hscale = np.maximum(np.sqrt(np.mean(observed[train, :11]**2, axis=(0, 1))), 1e-4)

    def labels(paths):
        return [classify(path[:11, :10], path[:11, 10:20], path[10:, :20], reference, mass)
                for path in paths]

    target_classes = labels(truth)
    truth_gaps = np.max(np.abs((coarse[:, :, :20]-observed)/scale), axis=(1, 2))
    for a, b, gap in zip(labels(coarse), target_classes, truth_gaps, strict=True):
        b["reference_qualified"] = bool(np.isfinite(gap) and gap <= gates["truth_step_max"] and align_classes(a, b))
    np.savez_compressed(directory/"truth.npz", fine=coarse, refined=truth, ids=cases["ids"],
                        roles=roles, power=power, equilibrium=equilibrium, scale=scale, history_scale=hscale)

    empty = np.zeros((0, 0))
    backbone = NonlinearPassiveClosure(mass, stiffness, damping, np.zeros((10, 0)), empty, empty)
    model = TopologyMemoryROM(backbone, port=port, tau=tau, learn_harmonics=True).to("cuda")
    model.set_angle_damping(2./3., spatial_mix=1.)
    model.requires_grad_(False)
    coefficients = model.restoring_harmonics

    def objective(indices, frames):
        return _objective(model, observed, indices, power, equilibrium, scale, hscale, frames, settings, budget.spend)

    @torch.no_grad()
    def validate():
        budget.spend("val_checks")
        return float(objective(val, settings["val_frames"]))

    budget.spend("fits")
    baseline, best_loss, chosen, failure = None, None, None, None
    initial, spectrum = np.zeros(2), np.zeros(2)
    best, actual_updates = snapshot(model), 0
    try:
        baseline = validate()
        initial, spectrum = _warm_start(model, observed[train], power[train], equilibrium[train], hscale, settings, budget)
        with torch.no_grad():
            coefficients.copy_(model._tensor(initial))
        best_loss, chosen, best = validate(), 0, snapshot(model)
    except NUMERICAL_ERRORS as error:
        failure = str(error)
    coefficients.requires_grad_(True)
    optimizer = torch.optim.Adam([coefficients], lr=settings["learning_rate"], foreach=False, fused=False)
    train_curve, val_curve = [], [{"update": 0, "nmse": best_loss, "coefficients": initial.tolist()}]
    print(f"ES-SINDy weak start {initial.tolist()}, val NMSE {best_loss}", flush=True)
    try:
        for update in range(1, settings["updates"]+1 if failure is None else 1):
            optimizer.zero_grad(set_to_none=True)
            loss = objective(train, settings["train_frames"])
            budget.spend("gradients")
            loss.backward()
            if coefficients.grad is None or not torch.isfinite(coefficients.grad).all():
                raise FloatingPointError("nonfinite causal harmonic gradient")
            budget.spend("optimizer_updates")
            optimizer.step()
            actual_updates = update
            with torch.no_grad():
                bound = settings["coefficient_bound"]
                coefficients.clamp_(-bound, bound)
            train_curve.append({"update": update, "nmse_before_update": float(loss.detach())})
            if update % settings["val_interval"] == 0:
                current = validate()
                val_curve.append({"update": update, "nmse": current, "coefficients": coefficients.detach().cpu().tolist()})
                print(f"ES-SINDy update {update}: val NMSE {current:.6g}", flush=True)
                if current < best_loss:
                    best_loss, chosen, best = current, update, snapshot(model)
    except NUMERICAL_ERRORS as error:
        failure = str(error)
    model.load_state_dict(best, strict=True)
    model.requires_grad_(False).eval()
    np.savez_compressed(directory/"selected-equation.npz", **{key: value.cpu().numpy() for key, value in best.items()})
    learned = coefficients.cpu().numpy()
    fit = {"actual_updates": actual_updates, "selected_update": chosen, "selected_val_nmse": best_loss, "failure": failure,
           "baseline_val_nmse": baseline, "train_curve": train_curve, "val_curve": val_curve,
           "truth": true_coefficients.tolist(), "weak_start": initial.tolist(),
           "weak_singular_values": spectrum.tolist(), "learned": learned.tolist()}
    _write(directory/"fit.json", fit)

    history = observed[:, :11]
    paths, histories, failures, _, _ = _forecast(model, roles, history, power, equilibrium, hscale, plan, budget)
    np.savez_compressed(directory/"forecasts.npz", forecasts=paths, histories=histories)
    score = _scores(cases, paths, truth, history, equilibrium, mass, scale, target_classes, gates)
    _write(directory/"scores.json", {**score, "failures": failures})
    energy = _physical(model, truth[:, 10], power, equilibrium, budget)
    val_metrics = score["roles"]["val"]["trajectory"]
    val_events = [row for row in score["events"] if row["role"] == "val"]
    relative = np.abs((learned-true_coefficients)/true_coefficients)
    checks = {
        "fit_completed": failure is None and actual_updates == settings["updates"],
        "physical": np.isfinite(energy) and energy <= gates["energy_balance_max"],
        "integration_and_coverage": all(row["qualified"] for row in score["events"]),
        "separable_dictionary": spectrum[-1] >= gates["dictionary_ratio_min"]*spectrum[0] > 0,
        "coefficient_recovery": bool(np.all(relative <= gates["parameter_relative_error"])),
        "trajectory": all(val_metrics[h]["full_cohort"] is not None
                          and val_metrics[h]["full_cohort"]["nrmse"]["mean"] <= gates[f"nrmse_{h}s"] for h in ("3", "8")),
        "val_improvement": baseline is not None and best_loss is not None
                           and baseline > gates["baseline_val_nmse_min"] and best_loss <= .5*baseline,
        "val_stability": all(row["prospective_eligible"] and row["predicted_label"] == row["truth_label"]
                             and row["truth_label"] is not None and (row["truth_label"] != "unstable"
                             or (row["trigger_time_error_s"] is not None
                                 and row["trigger_time_error_s"] <= gates["timing_error_s"])) for row in val_events),
    }
    return {"engineering_pass": bool(checks["physical"] and checks["integration_and_coverage"]),
            "scientific_pass": bool(all(checks.values())), "checks": {key: bool(value) for key, value in checks.items()},
            "fit": fit, "max_energy_balance_residual": energy, "max_truth_step_gap": float(truth_gaps.max()),
            "coefficient_relative_error": relative.tolist(), "roles": score["roles"],
            "claim_boundary": "clean analytic synthetic, fixed topology/damping/memory, one causal shared-coefficient fit"}
