"""Frozen scoring views, causal noise injection, and split inference timing."""

from __future__ import annotations

import io
import json
import time
from pathlib import PurePosixPath

import numpy as np
import torch
from scipy.signal import savgol_coeffs

from kron_reduce.data.task3_measurement_noise import inject_raw_history
from kron_reduce.evaluation.kiki_topology_bridge import load_snapshot, make_model
from kron_reduce.models.first_swing_fit import (
    enable_first_swing_rho,
    load_first_swing_scoring_view,
    predict_first_swing_fit,
)
from kron_reduce.models.noise_robustness_mhe import predict_mhe
from kron_reduce.models.nonlinear_latent import LearnedNonlinearModel
from kron_reduce.models.unified_discovery_m3 import predict_m3
from kron_reduce.physics.coordinates import coi_project, coi_project_power
from kron_reduce.workflows.postfault_m4_m3b import _unified
from kron_reduce.workflows.postfault_m4_v2 import prior_receipt
from kron_reduce.workflows.unified_first_swing import _archive_asset
from project_harness.archive_storage import read_run_artifact
from project_harness.git_state import verify_bound_file

MODELS = ("v4_hard_r", "v4_soft_2r", "m3b_g2_orig", "m3b_g2_sg_gn",
          "m3b_g2_joint_l2_mhe", "m3b_g2_robust_mhe")
STEPS = (.0025, .00125)


def read_binding(root, binding):
    location = PurePosixPath(binding["path"])
    if location.parts[:1] != ("results",) or len(location.parts) < 3:
        raise ValueError("expected a frozen run artifact")
    return read_run_artifact(root, location.parts[1], PurePosixPath(*location.parts[2:]).as_posix(),
                             binding["sha256"])


def load_assets(root, plan, baseline):
    """Reopen just M5's bound assets, including members of relocated ZIPs."""
    verify_bound_file(root, baseline["source_contract"], baseline["bindings"][baseline["source_contract"]])
    source = json.loads((root / baseline["source_contract"]).read_bytes())
    v4 = prior_receipt(root, baseline, "v4")
    m3 = prior_receipt(root, baseline, "m3b")
    old = json.loads(read_binding(root, v4["frozen_model_manifest"]))
    new = json.loads(read_binding(root, m3["frozen_model_manifest"]))
    weights = v4["parameter_checkpoints"] | {"m3": m3["frozen_parameters"]}
    if (set(weights) != {"hard_r", "soft_2r", "m3"}
            or any(weights[k]["sha256"] != baseline["v4_parameters_sha256"] for k in ("hard_r", "soft_2r"))
            or weights["m3"]["sha256"] != baseline["m3b_parameters_sha256"]
            or weights["m3"] != {"path": new["checkpoint_path"], "sha256": new["checkpoint_sha256"]}
            or any(x["input_contract"] != plan["input_contract"] or not x["selection_closed"]
                   or not x["fitting_closed"] or x["settings"]["prior_weight"] != plan["original_prior_weight"]
                   for x in (old, new))
            or old["backbone"] != source["backbone"] or new["backbone_anchor"] != source["backbone"]
            or source["backbone"]["sha256"] != baseline["a06_sha256"]
            or not m3["scientific_pass"] or new["support"] != ["G2"] or new["decision"] != "SELECTED"):
        raise ValueError("M5 scoring identity changed")
    prep = [{"path": x["preprocessing_path"], "sha256": x["preprocessing_sha256"]} for x in (old, new)]
    if any(x["sha256"] != baseline["preprocessing_sha256"] for x in prep):
        raise ValueError("M5 preprocessing changed")
    preproc = [read_binding(root, x) for x in prep]
    if preproc[0] != preproc[1]:
        raise ValueError("the original scoring views require shared preprocessing")
    data = {key: read_binding(root, binding) for key, binding in weights.items()}
    backbone = LearnedNonlinearModel.load(
        io.BytesIO(_archive_asset(root, source["backbone"], baseline["backbone_archive"]))).closure
    models = {"m3": _unified(backbone, io.BytesIO(data["m3"]), new["support"])}
    for key in ("hard_r", "soft_2r"):
        fitted = load_snapshot(enable_first_swing_rho(make_model(backbone, "T0M1", train_scales=True)),
                               io.BytesIO(data[key]))
        models[key] = load_first_swing_scoring_view(fitted, make_model(backbone, "T0M1", train_scales=True))
    models = {key: model.to(plan["device"]).requires_grad_(False).eval() for key, model in models.items()}
    return source["sources"], models, json.loads(preproc[0]), {
        "receipts": {"v4": v4, "m3": m3}, "weights": weights, "weight_bytes": data,
        "preprocessing": prep[0], "preprocessing_bytes": preproc[0],
    }


def causal_sg(values, *, window, order):
    """Fit only the trailing observations, evaluating each fit at its right endpoint."""
    result = values.copy()
    for width in range(order + 2, window + 1):
        coeff = savgol_coeffs(width, order, pos=width - 1, use="dot")
        for stop in range(width, values.shape[1] + 1 if width == window else width + 1):
            result[:, stop - 1] = np.einsum("i,bij->bj", coeff, values[:, stop - width:stop])
    return result


def observation_inputs(events, bank, scales):
    histories, powers = [], []
    for event in events:
        draw = bank.draws[event["id"]]
        if draw.clean:
            histories.append(event["x"][:11])
            powers.append(event["p0"])
            continue
        mass = event["mass"]
        raw, power = inject_raw_history(event["raw_x"][:11], event["raw_p0"], mass, draw)
        histories.append(np.column_stack((coi_project(raw[:, :10], mass), coi_project(raw[:, 10:], mass))))
        powers.append(coi_project_power(power, mass))
    history = np.stack(histories)
    return (history[:, :, :10], history[:, :, 10:], np.stack(powers),
            np.stack([event["equilibrium"] for event in events]), np.asarray(scales["history"]))


def model_for(models, name):
    return models[name[3:] if name.startswith("v4_") else "m3"]


def initialize(model, name, inputs, step, plan, scales, candidate, spend):
    options = {"max_step": step, "spend": spend, "memory_radius": scales["memory_radius"]}
    prior = plan["original_prior_weight"]
    if name in MODELS[:2]:
        return predict_first_swing_fit(model, *inputs, 0, **options, prior_weight=prior,
                                       soft_radius=name == "v4_soft_2r")
    if name == "m3b_g2_sg_gn":
        mass = model.mass.cpu().numpy()
        sg = {"window": plan["causal_sg_window"], "order": plan["causal_sg_order"]}
        inputs = tuple(coi_project(causal_sg(x, **sg), mass) for x in inputs[:2]) + inputs[2:]
    if name in MODELS[2:4]:
        return predict_m3(model, *inputs, 0, **options, prior_weight=prior)
    return predict_mhe(model, *inputs, 0, **options,
                       method="l2" if name == "m3b_g2_joint_l2_mhe" else "huber",
                       **{key: candidate[key] for key in ("gate_rad", "huber_delta", "alpha")})


def _elapsed(start):
    torch.cuda.synchronize()
    return time.perf_counter() - start


def infer(model, name, inputs, step, plan, scales, candidate, spend):
    """Time both physical stages synchronously, retaining the actual issued state."""
    start = time.perf_counter()
    initial = initialize(model, name, inputs, step, plan, scales, candidate, spend)
    origin = initial["forecast"][:, 0]
    init_s = _elapsed(start)
    rollout_start = time.perf_counter()
    forecast = model.rollout(origin, inputs[2], inputs[3], plan["forecast_frames"],
                             max_step=step, spend=spend)
    rollout_s = _elapsed(rollout_start)
    memory = .5 * origin[:, 20:].square().sum(dim=1)
    total = model.storage(origin, model._tensor(inputs[3]))
    energy = torch.stack((total - memory, memory, total), dim=1)
    diagnostics = {k: initial[k].cpu().numpy() for k in ("joint_active", "kinematic_defect_rad",
                   "sigma_hat", "map_lambda") if k in initial}
    return (forecast[:, :, :20].cpu().numpy(), energy.cpu().numpy(), diagnostics,
            {"init_s": init_s, "rollout_s": rollout_s, "total_s": init_s + rollout_s})


def slice_inputs(inputs, start, end):
    return tuple(x[start:end] for x in inputs[:-1]) + (inputs[-1],)
