"""Cross four frozen topology and memory models on independent mixed excitations."""
from __future__ import annotations

import numpy as np

from kron_reduce.evaluation.constant_c_observability import write_json
from kron_reduce.evaluation.machine_local_equation import ROLES, align_classes
from kron_reduce.evaluation.machine_local_topology import (
    SUPPORTS,
    energy_residual,
    scoring,
    simulate,
)
from kron_reduce.evaluation.machine_local_topology_learning import selected_model
from kron_reduce.evaluation.machine_local_topology_truth import parameters, truth_rollout
from kron_reduce.evaluation.trajectory_stability import classify, summarize
from project_harness.evidence import sha256_file


def transfer_cases(plan):
    roles = [role for role in ROLES for _ in range(plan["cohorts"][role])]
    strata = [stratum for role in ROLES for _ in range(plan["cohorts"][role]//2) for stratum in ("low", "high")]
    size = len(roles)
    ids = [f"transfer-{i:02d}" for i in range(size)]
    if ids != plan["ids"] or len(strata) != size:
        raise ValueError("transfer roster differs from registered IDs or excitation counts")
    mass = parameters()[0]
    rng = np.random.default_rng(plan["seed"])
    projection = np.eye(len(mass))-np.outer(mass, np.ones(len(mass)))/mass.sum()
    dimensions = (size, len(mass))
    q = rng.uniform(-.14, .14, dimensions) @ projection
    v = rng.uniform(-.12, .12, dimensions)
    high = np.flatnonzero(np.array(strata) == "high")
    for i, speed in zip(high, np.linspace(*plan["speed_rad_s"], len(high)), strict=True):
        signs = np.tile([1., -1.], 5)
        v[i] += speed*rng.permutation(signs)
    v = v @ projection
    power = rng.normal(0., .035, dimensions) @ projection.T
    a = rng.normal(0., .095, dimensions)
    a -= a.mean(axis=1, keepdims=True)
    a *= np.minimum(1., .20/np.max(np.abs(a), axis=1, keepdims=True))
    z = rng.normal(0., .18, dimensions)
    z *= np.minimum(1., .45/np.linalg.norm(z, axis=1, keepdims=True))
    return {"ids": ids, "roles": roles, "excitation": strata, "q": q, "v": v, "power": power, "a": a, "z": z}


def execute(root, directory, plan, budget):
    cases, params = transfer_cases(plan), parameters()
    mass, _, _, _, _, ref = params
    q, v, power = (cases[key] for key in ("q", "v", "power"))
    equilibrium = np.broadcast_to(ref, power.shape).copy()
    roles = np.array(cases["roles"])
    train, development = roles == "train", roles != "test"
    models = {}
    for source in plan["sources"].values():
        folder = (root/"results"/source["run_id"]).resolve(strict=True)
        if folder.parent != root/"results" or sha256_file(folder/"artifact_hashes.json") != source["manifest_sha256"]:
            raise RuntimeError("frozen transfer source manifest differs")
        for support in source["equation_sha256"]:
            models[support] = selected_model(root, source, support, params, device="cuda").requires_grad_(False).eval()
    records = {}
    for support in SUPPORTS:
        output = directory/support
        output.mkdir()
        matched = models[support]
        a = cases["a"] if matched.topology_dimension else np.zeros_like(cases["a"])
        z = cases["z"][:, :matched.memory_dimension]
        fine, truth = (truth_rollout(params, q, v, power, a, z, count=810, max_step=step, spend=budget.spend)
                       for step in (.00125, .000625))
        scale = np.maximum(np.sqrt(np.mean(truth[train, 11:, :20]**2, axis=(0, 1))), 1e-4)
        hscale = np.maximum(np.sqrt(np.mean(truth[train, :11, :20]**2, axis=(0, 1))), 1e-4)
        truth_classes = [[classify(t[:11, :10], t[:11, 10:20], t[10:, :20], ref, mass) for t in path]
                         for path in (fine, truth)]
        truth_gap = np.max(np.abs((fine[:, :, :20]-truth[:, :, :20])/scale), axis=(1, 2))
        for coarse, refined, gap in zip(*truth_classes, truth_gap, strict=True):
            refined.update(reference_qualified=bool(gap <= 1e-6 and align_classes(coarse, refined)),
                           reference_step_gap=float(gap) if np.isfinite(gap) else None)
        truth_ok = all(row["reference_qualified"] for row, use in zip(truth_classes[1], development, strict=True) if use)
        np.savez_compressed(output/"truth.npz", fine=fine, refined=truth, q=q, v=v, a=a, z=z, power=power,
                            equilibrium=equilibrium, scale=scale, history_scale=hscale, ids=cases["ids"],
                            roles=roles, excitation=cases["excitation"])
        p, eq = matched._tensor(power), matched._tensor(equilibrium)
        scores, balance = {}, None
        for name, model in models.items():
            same_support = name == support
            paths, histories, failures = simulate(model, truth, a, p, eq, hscale, budget, oracle=False)
            score = scoring(model, cases, paths, histories, truth, a, equilibrium, mass, scale, hscale,
                            truth_classes[1], oracle=not same_support)
            for row, stratum, cls in zip(score["events"], cases["excitation"], truth_classes[1], strict=True):
                row.update(excitation=stratum, truth_trigger_index=cls["trigger_index"],
                           reference_qualified=cls["reference_qualified"], reference_step_gap=cls["reference_step_gap"])
            by_excitation = {}
            for role in ROLES:
                by_excitation[role] = {}
                for label in ("low", "high"):
                    rows = [row for row in score["events"] if row["role"] == role and row["excitation"] == label]
                    rmse = {key: float(np.sqrt(np.mean([row["trajectory"]["8"][key]**2 for row in rows])))
                            if all(row["qualified"] for row in rows) else None
                            for key in ("nrmse", "q_rmse", "v_rmse")}
                    by_excitation[role][label] = {"stability": summarize(rows), "pooled_8s": rmse}
            score["by_excitation"] = by_excitation
            score["failures"] = failures
            np.savez_compressed(output/f"{name}-forecasts.npz", forecasts=paths, histories=histories)
            write_json(output/f"{name}-scores.json", score)
            scores[name] = score
            if same_support and all(x["qualified"] for x in score["events"] if x["role"] != "test"):
                balance = energy_residual(model, model._tensor(paths[0, development, 0]), p[development],
                                          eq[development], budget.spend)

        known = [x for x in scores[support]["events"] if x["role"] != "test"]
        stable = scores[support]["roles"]["val"]["stability"]
        gates = {
            "truth": truth_ok,
            "physical": balance is not None and balance <= 1e-10,
            "mixed_truth": all({x["truth_label"] for x in known if x["role"] == role} == {"stable", "unstable"}
                               for role in ("train", "val")) and all(x["prospective_eligible"] for x in known),
            "stability": stable["accuracy"] is not None and stable["accuracy"] > .9,
            "trajectory": all(x["qualified"] and x["trajectory"]["3"]["nrmse"] <= .01
                              and x["trajectory"]["8"]["nrmse"] <= .03 for x in known),
            "edge": all(x["qualified"] and x["edge_relative_error"] <= .05 for x in known),
        }
        records[support] = {"gates": gates, "energy_residual_max": balance,
                            "scores": {name: {key: row[key] for key in ("roles", "by_excitation")}
                                       for name, row in scores.items()}}
        print(f"{support}: {gates}", flush=True)
    return {"populations": records, "engineering_pass": all(x["gates"]["truth"] and x["gates"]["physical"]
                                                            for x in records.values()),
            "scientific_pass": all(all(x["gates"].values()) for x in records.values()),
            "claim_boundary": "Four frozen equations on independent clean synthetic excitations; no support selection or fit."}
