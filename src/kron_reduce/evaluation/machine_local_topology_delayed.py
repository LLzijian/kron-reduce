"""Known-truth delayed instability and train-only loss calibration."""
from __future__ import annotations

import math

import numpy as np
import torch

from kron_reduce.evaluation.constant_c_observability import write_json
from kron_reduce.evaluation.machine_local_clean_learning import NUMERICAL_ERRORS
from kron_reduce.evaluation.machine_local_equation import ROLES, align_classes
from kron_reduce.evaluation.machine_local_topology_margin import (
    evaluate_pair,
    margin,
    training_losses,
)
from kron_reduce.evaluation.machine_local_topology_truth import parameters, truth_rollout
from kron_reduce.evaluation.train_mechanism_audit import _row_margin
from kron_reduce.evaluation.trajectory_stability import classify


def opposed_direction(mass, fast):
    mass = np.asarray(mass, dtype=np.float64)
    if mass.shape != (10,) or not np.isfinite(mass).all() or np.any(mass <= 0):
        raise ValueError("ten finite positive masses are required")
    if (len(fast) != 2 or any(type(i) is not int or not 0 <= i < 10 for i in fast)
            or fast[0] == fast[1]):
        raise ValueError("two distinct machine indexes are required")
    direction = np.zeros(10)
    direction[list(fast)] = 1.
    return direction - (direction @ mass)/mass.sum()


def adam_direction(gradient):
    clipped = gradient * (1./(gradient.norm()+1e-6)).clamp_max(1.)
    return clipped/(clipped.abs()+1e-8)


def calibrate(model, visible, power, equilibrium, lengths, scale, hscale, settings, budget):
    targets = margin(visible, equilibrium, model.mass, lengths)
    params = dict(model.named_parameters())
    try:
        trajectory, margin_mse, predicted = training_losses(
            model, visible, power, equilibrium, lengths, scale, hscale, targets, budget.spend,
        )
        budget.spend("gradients")
        g_traj = torch.autograd.grad(trajectory, tuple(params.values()), retain_graph=True)
        budget.spend("gradients")
        g_margin = torch.autograd.grad(margin_mse, tuple(params.values()))
    except NUMERICAL_ERRORS as error:
        return {"weight": None, "reason": str(error)}
    gx, gu = (torch.cat([value.detach().reshape(-1) for value in block]) for block in (g_traj, g_margin))
    norms = [float(value.norm()) for value in (gx, gu)]
    missed = int(((targets > 0.) & (predicted <= 0.)).sum())
    positive = int((targets > 0.).sum())
    record = {"weight": None, "train_events": len(visible), "positive_witnesses": positive,
              "shared_start_misses": missed}
    if not all(math.isfinite(n) and n > 1e-12 for n in norms):
        return record | {"reason": "zero or nonfinite gradient norm"}
    weight = norms[0]/norms[1]
    base, adjusted = adam_direction(gx), adam_direction(gx+weight*gu)
    change = float((adjusted-base).norm()/base.norm())
    threshold = settings["minimum_adam_change"]
    record.update(trajectory_loss=float(trajectory.detach()), margin_loss=float(margin_mse.detach()),
                  parameter_shapes={name: list(value.shape) for name, value in params.items()},
                  trajectory_gradient=gx.tolist(), margin_gradient=gu.tolist(),
                  gradient_norms=norms, gradient_cosine=float((gx/norms[0]) @ (gu/norms[1])),
                  candidate_weight=weight, relative_adam_change=change)
    if missed < settings["minimum_train_misses"]:
        return record | {"reason": "insufficient missed train witnesses at the shared start"}
    if change < threshold:
        return record | {"reason": "shared start Adam direction change below the registered limit"}
    return record | {"weight": weight, "reason": "eligible"}


def generate(plan, params, budget):
    """Choose only from each paired slot's frozen trial excitations, using NumPy truth."""
    mass, _, _, _, _, ref = params
    roles = [role for role in ROLES for _ in range(plan["cohorts"][role]//2)]
    identifiers = [f"delayed-{p:02d}-{label}" for p, _ in enumerate(roles) for label in ("stable", "unstable")]
    if identifiers != plan["ids"] or len(roles) != len(plan["fast_pairs"]):
        raise ValueError("delayed excitation roster differs from the registered slots")
    groups, q, v = [], [], []
    for pair, (role, fast) in enumerate(zip(roles, plan["fast_pairs"], strict=True)):
        direction = opposed_direction(mass, fast)
        for label in ("stable", "unstable"):
            indexes = []
            for speed in plan["speed_gaps"][label]:
                indexes.append((len(q), speed))
                q.append(plan["initial_angle_gap"][label]*direction)
                v.append(speed*direction)
            groups.append({"id": f"delayed-{pair:02d}-{label}", "role": role, "expected": label,
                           "fast": fast, "alternatives": indexes})
    q, v = np.stack(q), np.stack(v)
    zeros = np.zeros_like(q)
    paths = [truth_rollout(params, q, v, zeros, zeros, zeros, count=810, max_step=h, spend=budget.spend)
             for h in (.00125, .000625)]
    diagnostics, accepted = [], []
    for group in groups:
        selected = None
        variants = []
        for index, gap in group["alternatives"]:
            a, b = (classify(x[index, :11, :10], x[index, :11, 10:20], x[index, 10:, :20], ref, mass)
                    for x in paths)
            hit = b["trigger_index"]
            matched = (float(_row_margin(paths[1][index, hit-4:hit+1, :20], ref, mass, 2*math.pi*60).min())
                       if hit is not None else None)
            same = align_classes(a, b) and b["label"] == group["expected"]
            if group["expected"] == "unstable":
                lower, upper = plan["trigger_frames"]
                margin_min, margin_max = plan["matched_margin_band"]
                eligible = (same and hit is not None and a["trigger_index"] == hit and lower <= hit-10 <= upper
                            and margin_min <= matched <= margin_max)
            else:
                eligible = same and hit is None
            if selected is None and eligible:
                selected = index
            variants.append({"index": index, "speed_gap": gap, "coarse_label": a["label"],
                             "refined_label": b["label"], "trigger_index": hit, "matched_margin": matched,
                             "qualified": bool(eligible)})
        diagnostics.append({key: group[key] for key in ("id", "role", "expected", "fast")}
                           | {"selected": selected, "trials": variants})
        accepted.append(selected)
    if any(index is None for index in accepted):
        return diagnostics, None, None
    cases = {"ids": identifiers, "roles": [group["role"] for group in groups],
             "power": zeros[accepted], "a": zeros[accepted]}
    return diagnostics, cases, tuple(path[accepted] for path in paths)


def execute(root, directory, plan, budget):
    params = parameters()
    diagnostic, cases, truth = generate(plan, params, budget)
    write_json(directory/"excitation-qualification.json", diagnostic)
    if cases is None:
        return {"engineering_pass": False, "scientific_pass": False,
                "claim_boundary": "Independent NumPy qualification of the frozen delayed synthetic roster.",
                "truth_failure": "one or more slots had no qualified truth excitation",
                "unqualified_slots": [row["id"] for row in diagnostic if row["selected"] is None]}
    return evaluate_pair(directory, plan, budget, cases, params, *truth, calibration=calibrate)
