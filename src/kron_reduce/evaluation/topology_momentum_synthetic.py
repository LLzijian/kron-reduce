"""Freeze truth and inject one wrong force block for weak-momentum controls."""

from __future__ import annotations

import json
import math

import numpy as np

from kron_reduce.evaluation.constant_c_observability import write_json
from kron_reduce.evaluation.machine_local_topology_ball_history_no_prior import _file
from kron_reduce.evaluation.machine_local_topology_delayed_origin import load_arrays
from kron_reduce.evaluation.machine_local_topology_truth import _incidence, parameters
from kron_reduce.evaluation.topology_momentum import direction_changes, inspect_momentum


def _records(truth, blocks, plan, budget, *, stride=1, inertia=1., radius=None):
    rows = []
    for i, (event, role) in enumerate(zip(plan["ids"], plan["roles"], strict=True)):
        path = truth["refined"][i, :311:stride, :20]
        result = inspect_momentum(path, truth["power"][i], truth["equilibrium"][i], blocks,
                                  .01*stride, budget.spend, radius=radius, memory_inertia=inertia)
        rows.append({"id": event, "role": role, **result})
    return rows


def groups_by_role(rows):
    return {role: [row for row in rows if row["role"] == role] for role in ("train", "val", "test")}


def role_rms(rows, name="orthogonal_sse", n=10):
    return {role: math.sqrt(sum(row[name] for row in group)/(n*len(group)))
            for role, group in groups_by_role(rows).items()}


def execute(root, directory, plan, budget):
    path = _file(root, plan["source"], "truth.npz")
    truth = load_arrays(path, ("fine", "refined", "power", "equilibrium", "ids", "roles"), budget)
    shapes = {"fine": (48, 811, 30), "refined": (48, 811, 30), "power": (48, 10), "equilibrium": (48, 10)}
    if (truth["ids"].tolist() != plan["ids"] or truth["roles"].tolist() != plan["roles"]
            or any(truth[k].shape != shape or truth[k].dtype != np.float64 or not np.isfinite(truth[k]).all()
                   for k, shape in shapes.items())):
        raise ValueError("fixed synthetic momentum truth differs")
    truth_classes = json.loads(_file(root, plan["source"], "truth-classes.json").read_bytes())
    if len(truth_classes) != 48 or any(not row["reference_qualified"] for row in truth_classes):
        raise ValueError("unqualified fixed synthetic momentum truth")
    mass, stiffness, damping, port, tau, equilibrium = parameters()
    if not np.array_equal(truth["equilibrium"], np.broadcast_to(equilibrium, (48, 10))):
        raise ValueError("synthetic equilibrium differs")
    coupling = (np.eye(10)-mass[:, None]/mass.sum()) @ np.diag(port)
    blocks = (mass, _incidence(), stiffness, damping, coupling, tau)
    baseline = _records(truth, blocks, plan, budget, radius=plan["radius"])
    coarse = _records(truth, blocks, plan, budget, stride=2)
    for i, row in enumerate(baseline):
        row["algebraic_state_error"] = float(np.linalg.norm(row["algebraic_state"]-truth["refined"][i, 0, 20:]))
    write_json(directory/"matched-10ms.json", baseline)
    write_json(directory/"matched-20ms.json", coarse)

    rms, rms_coarse = role_rms(baseline), role_rms(coarse)
    dev = ("train", "val")
    numerics = all(row["rank"] == 10 and row["ball"]["residual_sse"] is not None
                   for row in baseline if row["role"] in dev)
    refinement = all(rms[role] <= plan["gates"]["sampling_ratio"]*rms_coarse[role] for role in dev)
    tests = []
    for name, index in (("stiffness", 2), ("damping", 3)):
        other = "damping" if name == "stiffness" else "stiffness"
        for multiplier in plan["force_multipliers"]:
            changed = list(blocks)
            changed[index] = multiplier*changed[index]
            rows = _records(truth, changed, plan, budget)
            write_json(directory/f"{name}-{multiplier:.3f}.json", rows)
            changes = direction_changes(rows, groups_by_role(rows))
            target = changes[name]
            expected = 1./multiplier-1.
            passes = (abs(target["delta_multiplier"]-expected) <= plan["gates"]["scale_error"]
                      and all(target["by_group"][role]["ratio"] <= plan["gates"]["corrected_sse_ratio"]
                              and target["by_group"][role]["ratio"] < changes[other]["by_group"][role]["ratio"]
                              for role in dev))
            tests.append({"perturbation": name, "operator_multiplier": multiplier, "expected_correction": expected,
                          "direction_fit": changes, "passed": passes})
    profile = {"1.0": rms}
    for inertia in plan["inertias"]:
        rows = _records(truth, blocks, plan, budget, inertia=inertia)
        profile[str(inertia)] = role_rms(rows)
        write_json(directory/f"inertia-{inertia:.6f}.json", rows)
    best = min(profile, key=lambda key: profile[key]["train"])
    inertia_gate = best == "1.0" and all(rms[role] <= plan["gates"]["inertia_rms_ratio"]*row[role]
                                         for role in dev for key, row in profile.items() if key != "1.0")
    return {"engineering_pass": numerics, "scientific_pass": numerics and refinement and inertia_gate
            and all(test["passed"] for test in tests), "refinement_gate": refinement, "inertia_gate": inertia_gate,
            "matched_rms_10ms": rms, "matched_rms_20ms": rms_coarse, "wrong_force": tests,
            "inertia_profile": profile, "selected_inertia_train": float(best),
            "claim_boundary": "Observed-path momentum controls on fixed known-truth analytic trajectories."}
