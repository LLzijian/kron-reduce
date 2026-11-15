"""Frozen start-state and coefficient controls on the delayed synthetic roster."""
from __future__ import annotations

import json
from copy import deepcopy

import numpy as np
import torch

from kron_reduce.evaluation.constant_c_observability import write_json
from kron_reduce.evaluation.kiki_topology_bridge import load_snapshot
from kron_reduce.evaluation.machine_local_equation import ROLES, STEPS
from kron_reduce.evaluation.machine_local_topology import make_model, scoring, simulate
from kron_reduce.evaluation.machine_local_topology_diagnosis import rollout
from kron_reduce.evaluation.machine_local_topology_learning import readout
from kron_reduce.evaluation.machine_local_topology_margin import matched_margins
from kron_reduce.evaluation.machine_local_topology_truth import parameters
from project_harness.evidence import sha256_file

BLOCKS = {"stiffness": ("log_stiffness_factor",), "damping": ("log_damping_factor",),
          "memory": ("port_parameter", "rate_parameter")}
SOURCE_FILES = {"truth.npz", "trajectory-equation.npz", "trajectory-forecasts.npz",
                "preprocessing.json", "truth-classes.json", "trajectory-scores.json"}


@torch.no_grad()
def replace_block(model, truth_model, block):
    if block not in BLOCKS:
        raise ValueError("unknown coefficient block")
    candidate = deepcopy(model)
    for name in BLOCKS[block]:
        getattr(candidate, name).copy_(getattr(truth_model, name))
    return candidate.requires_grad_(False).eval()


def source_files(root, binding):
    directory = (root/"results"/binding["run_id"]).resolve(strict=True)
    manifest_file = directory/"artifact_hashes.json"
    if (directory.parent != (root/"results").resolve(strict=True)
            or directory.name != binding["run_id"] or sha256_file(manifest_file) != binding["manifest_sha256"]):
        raise ValueError("delayed source archive differs from the registered manifest")
    manifest = json.loads(manifest_file.read_bytes())
    if set(binding["files"]) != SOURCE_FILES:
        raise ValueError("unexpected delayed source files")
    paths = {name: (directory/name).resolve(strict=True) for name in SOURCE_FILES}
    for name, path in paths.items():
        if path.parent != directory or sha256_file(path) != binding["files"][name] or manifest[name] != binding["files"][name]:
            raise ValueError(f"delayed source binding differs for {name}")
    return paths


def load_arrays(path, keys, budget):
    budget.spend("npz_decodes")
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != set(keys):
            raise ValueError(f"unexpected fields in {path.name}")
        return {key: saved[key] for key in keys}


def checked_cohort(plan, truth, forecasts, labels, baseline):
    size = len(plan["ids"])
    if (truth["ids"].tolist() != plan["ids"] or truth["roles"].tolist() != plan["roles"]
            or {role: plan["roles"].count(role) for role in ROLES} != plan["cohorts"]
            or [(r["id"], r["role"]) for r in baseline["events"]] != list(zip(plan["ids"], plan["roles"], strict=True))
            or len(labels) != size or any(not row["reference_qualified"] for row in labels)):
        raise ValueError("saved delayed cohort differs from the registered qualified roster")
    arrays = [(truth[key], (size, 811, 30)) for key in ("fine", "refined")]
    arrays += [(truth[key], (size, 10)) for key in ("power", "equilibrium")]
    arrays += [(forecasts[key], (2, size, length, 30)) for key, length in (("forecasts", 801), ("histories", 11))]
    if any(x.shape != shape or x.dtype != np.float64 or not np.isfinite(x).all() for x, shape in arrays):
        raise ValueError("saved delayed trajectories have invalid arrays")
    stored_origin = forecasts["forecasts"][:, :, 0]
    visible = np.broadcast_to(truth["refined"][:, 10, :20], stored_origin[:, :, :20].shape)
    if (not np.array_equal(stored_origin[:, :, :20], visible)
            or not np.array_equal(stored_origin[:, :, 20:], forecasts["histories"][:, :, -1, 20:])):
        raise ValueError("saved visible reset or estimated origin memory differs")


def support(candidate, control):
    """Count paired newly triggered windows and regressions on train and val."""
    paired = {}
    for role in ("train", "val"):
        positive = [(before, after) for before, after in zip(control["events"], candidate["events"], strict=True)
                    if before["role"] == role and before["truth_label"] == "unstable"]
        valid = [(before["matched_margin"], after["matched_margin"]) for before, after in positive
                 if before["matched_margin"] is not None and after["matched_margin"] is not None]
        before = [a["predicted"][str(STEPS[0])] > 0 for a, _ in valid]
        after = [b["predicted"][str(STEPS[0])] > 0 for _, b in valid]
        paired[role] = {"allocated": len(positive), "qualified": len(valid),
                        "recovered": sum(b and not a for a, b in zip(before, after, strict=True)),
                        "lost": sum(a and not b for a, b in zip(before, after, strict=True))}
    return paired


def exact_anchor(record, tolerance):
    for role in ("train", "val"):
        row = record["roles"][role]
        margin = row["matched_margin"]["unstable"]
        events = [event for event in record["events"] if event["role"] == role]
        if (row["stability"]["accuracy"] != 1. or margin["predicted_triggers"] != margin["allocated"]
                or not all(event["truth_agreement"] and event["trajectory"]["8"]["nrmse"] <= tolerance
                           for event in events)):
            return False
    return True


def sufficient(candidate, control, counts, limits):
    for role in ("train", "val"):
        pair = counts[role]
        new, ref = candidate["roles"][role], control["roles"][role]
        if (pair["qualified"] != pair["allocated"] or not pair["recovered"] or pair["lost"]
                or new["stability"]["accuracy"] != 1.):
            return False
        a, b = (row["matched_margin"]["unstable"] for row in (ref, new))
        if (b["predicted_triggers"] != b["allocated"] or a["rmse"] is None or b["rmse"] is None
                or b["rmse"] > limits["unstable_margin_ratio"]*a["rmse"]):
            return False
        for x, y in ((new, ref), (new["matched_margin"]["stable"], ref["matched_margin"]["stable"])):
            for horizon in ("3", "8"):
                actual, previous = (row["pooled_nrmse"][horizon] for row in (x, y))
                if actual is None or previous is None or actual > limits["trajectory_ratio"]*previous:
                    return False
    return True


def execute(root, directory, plan, budget):
    source = source_files(root, plan["source"])
    truth = load_arrays(source["truth.npz"], ("fine", "refined", "ids", "roles", "power", "equilibrium"), budget)
    saved = load_arrays(source["trajectory-forecasts.npz"], ("forecasts", "histories"), budget)
    baseline = json.loads(source["trajectory-scores.json"].read_bytes())
    _, labels = json.loads(source["truth-classes.json"].read_bytes())
    scales = json.loads(source["preprocessing.json"].read_bytes())
    checked_cohort(plan, truth, saved, labels, baseline)
    params = parameters()
    mass, *_, ref = params
    path, power, eq = (truth[key] for key in ("refined", "power", "equilibrium"))
    if not np.array_equal(eq, np.broadcast_to(ref, eq.shape)) or np.any(power):
        raise ValueError("saved delayed truth inputs differ")
    scale, hscale = (np.asarray(scales[key], dtype=float) for key in ("target", "history"))
    if any(x.shape != (20,) or not np.isfinite(x).all() or np.any(x <= 0) for x in (scale, hscale)):
        raise ValueError("saved train scales are invalid")
    cases = {key: plan[key] for key in ("ids", "roles")}
    a = np.zeros_like(power)
    known = make_model(params, "T0M1", "cuda", train_scales=True).requires_grad_(False).eval()
    budget.spend("npz_decodes")
    endpoint = load_snapshot(deepcopy(known), source["trajectory-equation.npz"])
    models = {"all_truth_exact_z": known, "endpoint_exact_z": endpoint}
    models.update({block: replace_block(endpoint, known, block) for block in BLOCKS})
    models["all_truth_causal"] = known
    origin = known._tensor(path[:, 10])[None].expand(2, -1, -1)
    p, equilibrium = known._tensor(power), known._tensor(eq)
    scores = {"endpoint_causal": baseline}
    write_json(directory/"source.json", plan["source"])
    write_json(directory/"endpoint_causal-scores.json", baseline)
    for name, model in models.items():
        exact = name != "all_truth_causal"
        if exact:
            with torch.no_grad():
                predictions, failures = rollout(model, origin, p, equilibrium, budget)
            histories = saved["histories"]  # The scorer skips history diagnostics for oracle origins.
            np.savez_compressed(directory/f"{name}-forecasts.npz", forecasts=predictions)
        else:
            predictions, histories, failures = simulate(model, path, a, p, equilibrium, hscale, budget, oracle=False)
            np.savez_compressed(directory/f"{name}-forecasts.npz", forecasts=predictions, histories=histories)
        score = scoring(model, cases, predictions, histories, path, a, eq, mass, scale, hscale, labels, exact)
        matched_margins(model, predictions, path, labels, equilibrium, score)
        score.update(equation=readout(model), failures=failures, origin="truth" if exact else "causal")
        write_json(directory/f"{name}-scores.json", score)
        scores[name] = score
        print(f"Scored {name} on {len(path)} frozen synthetic events", flush=True)

    pairs = {"origin_at_endpoint": ("endpoint_exact_z", "endpoint_causal"),
             "origin_at_truth": ("all_truth_exact_z", "all_truth_causal")}
    pairs.update({block: (block, "endpoint_exact_z") for block in BLOCKS})
    comparisons = {key: support(scores[a], scores[b]) for key, (a, b) in pairs.items()}
    engineering = exact_anchor(scores["all_truth_exact_z"], plan["anchor_nrmse_limit"])
    supported = [key for key, by_role in comparisons.items()
                 if engineering and all(row["qualified"] == row["allocated"] and row["recovered"] > 0 and not row["lost"]
                                        for row in by_role.values())]
    sufficient_groups = [key for key, (a, b) in pairs.items() if engineering
                         and sufficient(scores[a], scores[b], comparisons[key], plan["gates"])]
    return {"engineering_pass": engineering, "scientific_pass": False,
            "claim_boundary": "Known-truth coefficient and forecast-origin interventions on one fixed synthetic cohort.",
            "all_truth_causal_anchor": exact_anchor(scores["all_truth_causal"], plan["anchor_nrmse_limit"]),
            "comparisons": comparisons, "supported_interventions": supported, "sufficient_interventions": sufficient_groups,
            "models": {key: value["roles"] for key, value in scores.items()}}
