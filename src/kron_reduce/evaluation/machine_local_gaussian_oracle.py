"""Exact Gaussian information and visible-reset controls on saved noisy cases."""
from __future__ import annotations

import hashlib
import io

import numpy as np
from scipy.linalg import block_diag, expm

from kron_reduce.evaluation.constant_c_observability import write_json
from kron_reduce.evaluation.gaussian_state_posterior import condition_initial_state
from kron_reduce.evaluation.machine_local_initializer_diagnosis import check_paths, metrics
from kron_reduce.evaluation.machine_local_observer import execute as execute_observer
from kron_reduce.evaluation.machine_local_synthetic import augmented_generator, fixture
from kron_reduce.models.joint_history_initializer import initial_state_basis
from project_harness.evidence import sha256_file


def expected_risks(variance, bias, scale):
    """Keep analytic conditional squared risk separate from realized errors."""
    rows = {}
    for role, ix in (("train", slice(0, 32)), ("val", slice(32, 48)), ("test", slice(48, 64))):
        row = {}
        for seconds in (1, 3, 8):
            count = 100*seconds
            var = variance[:count]/scale**2
            squared_bias = (bias[ix, 1:count+1, :20]/scale)**2
            for name, channels in (("all", slice(0, 20)), ("q", slice(0, 10)), ("v", slice(10, 20))):
                floor = float(np.mean(var[:, channels]))
                extra = float(np.mean(squared_bias[:, :, channels]))
                row[f"full_nmse_{seconds}s_{name}"] = floor
                row[f"reset_extra_nmse_{seconds}s_{name}"] = extra
                row[f"reset_nmse_{seconds}s_{name}"] = floor+extra
        for mode in ("full", "reset", "reset_extra"):
            row[mode+"_main_nmse"] = (row[f"{mode}_nmse_1s_all"]+row[f"{mode}_nmse_3s_all"])/2
        rows[role] = row
    return rows


def evaluate(kind, data, directory, budget):
    paths, hscale, scale = data["paths"], data["history_scale"], data["target_scale"]
    observed, power = paths[:, :11, :20]+data["noise"], paths[:, 0, 30:]
    if kind == "nonlocal":
        budget.spend("matrix_products")
    budget.spend("matrix_products")
    base = fixture(kind)
    if any(np.max(np.abs(data[key]-value)) > 1e-12 for key, value in
           (("mass", base.mass), ("coupling", base.coupling), ("decay", base.latent_dissipation))):
        raise RuntimeError("Gaussian oracle true model changed")
    basis = initial_state_basis(base.mass)
    v = basis[:10, :9]
    visible = block_diag(v.T, v.T)
    p = np.eye(10)-np.outer(base.mass, np.ones(10))/base.mass.sum()
    budget.spend("matrix_products", 4)
    generator = augmented_generator(base)
    budget.spend("expm")
    transition = expm(.01*generator)
    initial = np.zeros((64+28, 40))
    initial[:64, 30:], initial[64:, :30] = power, basis.T
    current, states = initial.copy(), [initial.copy()]
    budget.spend("rollouts")
    for _ in range(810):
        budget.spend("matrix_products")
        current = current@transition.T
        states.append(current.copy())
    states = np.stack(states, axis=1)
    baseline = states[:64, :, :30]
    response = states[64:, :, :30].transpose(1, 2, 0)
    budget.spend("matrix_products", 4)
    w = p@v
    prior_visible = w.T@w
    prior = block_diag(.04*prior_visible, .01*prior_visible, .0225*np.eye(10))
    noise = .0001*block_diag(w.T@(hscale[:10, None]**2*w), w.T@(hscale[10:, None]**2*w))
    budget.spend("matrix_products", 22)
    design = np.einsum("ij,tjr->tir", visible, response[:11, :20])
    residual = (observed-baseline[:, :11, :20])@visible.T
    posterior = condition_initial_state(design, residual, prior, noise, spend=budget.spend)
    budget.spend("matrix_products", 811)
    all_mean = baseline+np.einsum("tir,br->bti", response, posterior["mean"])
    full = all_mean[:, 10:]
    budget.spend("matrix_products")
    reset_coordinates = (observed[:, -1]-all_mean[:, 10, :20])@visible.T
    budget.spend("matrix_products", 801)
    bias = np.einsum("tir,br->bti", response[:801, :, :18], reset_coordinates)
    reset = full+bias
    budget.spend("matrix_products", 812)
    true_coordinates = paths[:, 0, :30]@basis
    true_reconstruction = baseline+np.einsum("tir,br->bti", response, true_coordinates)
    source_error = float(np.max(np.abs(true_reconstruction-paths[:, :, :30])))
    budget.spend("matrix_products", 1600)
    future_response = response[11:, :20]
    variance = np.einsum("tir,rs,tis->ti", future_response, posterior["covariance"], future_response)
    coi = check_paths((all_mean, reset, true_reconstruction), base.mass, budget)
    inverse = max(posterior[key] for key in ("prior_inverse_error", "posterior_inverse_error"))
    minimum_variance = float(np.min(variance))
    technical = source_error <= 1e-10 and inverse <= 1e-8 and coi <= 1e-10 and minimum_variance >= -1e-12
    risks = expected_risks(variance, bias, scale)
    full_metrics = metrics(full, all_mean[:, :11], paths, scale)
    reset_metrics = metrics(reset, all_mean[:, :11], paths, scale)
    row = {"case": kind, "technical_pass": bool(technical), "source_reconstruction_error": source_error,
           "coi_error": coi, "maximum_equation_error": inverse, "minimum_variance": minimum_variance,
           "factor_reconstruction_error": posterior["factor_reconstruction_error"],
           "solve_backward_error": posterior["solve_backward_error"],
           "covariance_symmetry_error": posterior["covariance_symmetry_error"],
           "expected_risk": risks, "full_mean_metrics": full_metrics, "reset_metrics": reset_metrics,
           "information_target_feasible": all(risks["val"][key] <= .0004 for key in
               ("full_main_nmse", "full_nmse_3s_all")),
           "reset_val_target_feasible": all(risks["val"][key] <= .0004 for key in
               ("reset_main_nmse", "reset_nmse_3s_all"))}
    arrays = {k: value for k, value in posterior.items() if isinstance(value, np.ndarray)}
    arrays.update(basis=basis, visible_projection=visible, power_projection=p, transition=transition,
        baseline=baseline, response=response, observed=observed, design=design, residual=residual,
        posterior_all_mean=all_mean, full_forecast=full, reset_forecast=reset, reset_bias=bias,
        reset_coordinates=reset_coordinates, visible_posterior_variance=variance,
        true_reconstruction=true_reconstruction, true_coordinates=true_coordinates)
    np.savez_compressed(directory/(kind+"_posterior.npz"), **arrays)
    write_json(directory/(kind+".json"), row)
    return row


def validate(root, directory, budget, plan):
    datasets = {}
    with budget.scope("sources"):
        for kind, entry in plan["saved_sources"].items():
            raw = (root/entry["path"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
                raise RuntimeError("Gaussian oracle saved source changed")
            budget.spend("npz_decodes")
            with np.load(io.BytesIO(raw), allow_pickle=False) as source:
                data = {key: source[key].copy() for key in source.files}
            if (data["paths"].shape != (64, 811, 40) or data["noise"].shape != (64, 11, 20)
                    or not np.array_equal(data["roles"], ["train"]*32+["val"]*16+["test"]*16)):
                raise RuntimeError("Gaussian oracle saved population changed")
            for key, frames in (("target_scale", slice(11, 311)), ("history_scale", slice(0, 11))):
                expected = np.maximum(np.sqrt(np.mean(data["paths"][:32, frames, :20]**2, axis=(0, 1))), 1e-5)
                if not np.array_equal(data[key], expected):
                    raise RuntimeError("Gaussian oracle scale is not frozen clean train RMS")
            datasets[kind] = data
            np.savez_compressed(directory/(kind+"_inputs.npz"), **data)
    write_json(directory/"frozen_inputs.json", {p.name: sha256_file(p) for p in sorted(directory.glob("*_inputs.npz"))})
    rows = {}
    for kind, data in datasets.items():
        with budget.scope(kind):
            rows[kind] = evaluate(kind, data, directory, budget)
    technical = all(r["technical_pass"] for r in rows.values())
    information = all(r["information_target_feasible"] for r in rows.values())
    reset = all(r["reset_val_target_feasible"] for r in rows.values())
    route = ("ORACLE_INVALID" if not technical else "INFORMATION_CONTRACT_REVIEW_REQUIRED" if not information else
             "RESET_INTERFACE_REPAIR_PRIORITY" if not reset else "REGISTER_TRAIN_ONLY_STATISTICAL_OBSERVER")
    return {"scientific_pass": bool(technical), "information_target_feasible": bool(information),
            "reset_val_target_feasible": bool(reset), "records": rows, "next_route": route,
            "learned_model_confirmed": False, "t0_allowed": False,
            "claim_boundary": "Known-generative-prior and known-noise Gaussian information oracle, not deployable. Conditional expected squared-risk lower bound applies to independent val/test, not train conditioned on its own scales; realized finite-sample errors are descriptive. Reset bias is for the same posterior hidden state, not every reset-constrained predictor."}


def execute(*, root, result_dir, stage, plan):
    return execute_observer(root=root, result_dir=result_dir, stage=stage, plan=plan, validator=validate)
