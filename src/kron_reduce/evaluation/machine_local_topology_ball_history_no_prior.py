"""Complete the matched no-prior initialization comparison on frozen truth."""

from __future__ import annotations

import json
from functools import partial

import numpy as np

from kron_reduce.evaluation.constant_c_observability import write_json
from kron_reduce.evaluation.machine_local_topology import make_model, scoring, simulate
from kron_reduce.evaluation.machine_local_topology_delayed_origin import load_arrays
from kron_reduce.evaluation.machine_local_topology_margin import matched_margins
from kron_reduce.evaluation.machine_local_topology_truth import parameters
from kron_reduce.models.machine_local_topology import predict_from_topology_history
from project_harness.evidence import sha256_file


def _file(root, binding, name):
    directory = (root/"results"/binding["run_id"]).resolve(strict=True)
    if directory.parent != (root/"results").resolve() or directory.name != binding["run_id"]:
        raise ValueError("frozen observer source run differs")
    manifest = directory/"artifact_hashes.json"
    if sha256_file(manifest) != binding["manifest_sha256"]:
        raise ValueError("frozen observer source manifest differs")
    path = (directory/name).resolve(strict=True)
    digest = binding["files"][name]
    if path.parent != directory or sha256_file(path) != digest or json.loads(manifest.read_bytes())[name] != digest:
        raise ValueError(f"frozen observer source file differs: {name}")
    return path


def execute(root, directory, plan, budget):
    source = plan["source"]
    truth = load_arrays(_file(root, source, "truth.npz"), ("fine", "refined", "power", "equilibrium", "ids", "roles"), budget)
    load = lambda name: json.loads(_file(root, source, name).read_bytes())
    classes = load("truth-classes.json")
    exact = load("exact_no_prior-scores.json")
    scales = load("preprocessing.json")
    expected = list(zip(plan["ids"], plan["roles"], strict=True))
    shape = {"fine": (48, 811, 30), "refined": (48, 811, 30), "power": (48, 10), "equilibrium": (48, 10)}
    if (list(zip(truth["ids"].tolist(), truth["roles"].tolist(), strict=True)) != expected
            or [(r["id"], r["role"]) for r in exact["events"]] != expected
            or len(classes) != 48 or any(row["truth_label"] != label["label"]
                                         for row, label in zip(exact["events"], classes, strict=True))
            or any(truth[k].shape != size or truth[k].dtype != np.float64 or not np.isfinite(truth[k]).all()
                   for k, size in shape.items())):
        raise ValueError("frozen truth and no-prior reference cohorts differ")
    scale, hscale = (np.asarray(scales[key], dtype=float) for key in ("target", "history"))
    if any(x.shape != (20,) or not np.isfinite(x).all() or np.any(x <= 0) for x in (scale, hscale)):
        raise ValueError("frozen training scales differ")

    params = parameters()
    target, power, equilibrium = (truth[key] for key in ("refined", "power", "equilibrium"))
    cases = {key: plan[key] for key in ("ids", "roles")}
    model = make_model(params, "T0M1", "cuda").eval()
    p, eq = model._tensor(power), model._tensor(equilibrium)
    observer = partial(predict_from_topology_history, memory_radius=plan["radius"], prior_weight=0.)
    a = np.zeros_like(power)
    future, history, failures = simulate(model, target, a, p, eq, hscale, budget, oracle=False, observer=observer)
    radial = scoring(model, cases, future, history, target, a, equilibrium, params[0], scale, hscale, classes, False)
    matched_margins(model, future, target, classes, eq, radial)
    radial["failures"] = failures
    write_json(directory/"radial_no_prior-scores.json", radial)
    np.savez_compressed(directory/"radial_no_prior-predictions.npz", forecasts=future, histories=history)

    pairs = list(zip(radial["events"], exact["events"], strict=True))
    engineering = all(row["qualified"] for pair in pairs for row in pair if row["role"] in ("train", "val"))
    comparisons = []
    tolerance = plan["objective_tolerance"]
    for old, new in pairs:
        if not new["id"].startswith("exterior-"):
            continue
        both = old.get("history_nrmse") is not None and new.get("history_nrmse") is not None
        comparisons.append({"id": new["id"], "role": new["role"], "qualified": old["qualified"] and new["qualified"],
                            "objective_improvement": 200*(old["history_nrmse"]**2-new["history_nrmse"]**2) if both else None,
                            "at_radius": both and abs(2*new["memory_start_energy"]-plan["radius"]**2) <= 1e-9})
    gates = {}
    for role in ("train", "val"):
        exterior = [row for row in comparisons if row["role"] == role]
        improvement = [row["objective_improvement"] for row in exterior if row["qualified"] and row["at_radius"]]
        old, new = (record["roles"][role] for record in (radial, exact))
        gates[role] = {
            "constrained_geometry": len(improvement) == len(exterior) and bool(improvement)
            and min(improvement) >= -tolerance and max(improvement) >= tolerance,
            "exact_stability": new["stability"]["accuracy"] == 1.,
            "trajectory": all(new["pooled_nrmse"][h] is not None and old["pooled_nrmse"][h] is not None
                              and new["pooled_nrmse"][h] <= plan["trajectory_ratio"]*old["pooled_nrmse"][h]
                              for h in ("3", "8")),
        }
    write_json(directory/"paired-history-objective.json", comparisons)
    return {"engineering_pass": engineering, "scientific_pass": engineering and all(all(row.values()) for row in gates.values()),
            "gates": gates, "models": {"radial_no_prior": radial["roles"], "exact_no_prior": exact["roles"]},
            "source": source, "claim_boundary": "Matched no-prior initialization comparison on frozen analytic trajectories."}
