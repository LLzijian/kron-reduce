"""Known-truth comparison of a conservative, tangent-preserving restoring term."""
from __future__ import annotations

from functools import partial

import numpy as np

from kron_reduce.evaluation.constant_c_observability import write_json
from kron_reduce.evaluation.machine_local_topology import energy_residual, make_model, scoring, simulate
from kron_reduce.evaluation.machine_local_topology_margin import matched_margins
from kron_reduce.evaluation.machine_local_topology_truth import _incidence, parameters, truth_rollout
from kron_reduce.evaluation.topology_angle_damping_synthetic import _source_population, _stable_supported, _truth_classes
from kron_reduce.evaluation.topology_restoring_momentum import fit_restoring_train, restoring_momentum, restoring_sse
from kron_reduce.evaluation.topology_spatial_momentum import spatial_damping_root
from kron_reduce.models.machine_local_topology import predict_from_topology_history


def restoring_gates(models, weak, truth, fit, limits):
    base, candidate = models["base"], models["shape"]
    supported = [row for row in candidate["events"] if row["role"] != "test"
                 and not row["id"].startswith("exterior-")]
    checks = {"coefficient": max(abs(fit["shape"][key]-truth) for key in ("raw", "value")) <= limits["fit_error"],
              "feasible_trace": all(row["truth_agreement"] and row["trajectory"]["8"]["nrmse"] <= limits["feasible_nrmse"]
                                    for row in supported)}
    if not truth:
        checks["scale_null"] = max(abs(fit["scale"][key]) for key in ("raw", "value")) <= limits["fit_error"]
    if truth:
        stable, control = (_stable_supported(model, "val") for model in (candidate, base))
        dev = [(candidate["roles"][role], base["roles"][role]) for role in ("train", "val")]
        checks.update({
            "weak_separation": all(weak[role][name] > limits["floor_ratio"]*weak[role]["truth"]
                                   and weak[role]["shape"] <= limits["weak_ratio"]*weak[role][name]
                                   for role in ("train", "val") for name in ("base", "scale")),
            "stable_improvement": stable is not None and control is not None and stable <= limits["stable_ratio"]*control,
            "matched_witness": all(a["matched_margin"]["unstable"]["predicted_triggers"]
                                   >= b["matched_margin"]["unstable"]["predicted_triggers"] for a, b in dev),
            "no_additional_false_alarm": all(a["stability"]["fp"] <= b["stability"]["fp"] for a, b in dev),
        })
    return checks


def execute(root, directory, plan, budget):
    saved, cases = _source_population(root, plan, budget)
    params = parameters()
    mass, stiffness, damping, port, tau, reference = params
    initial, power, eq = (saved[key] for key in ("refined", "power", "equilibrium"))
    if not np.array_equal(eq, np.broadcast_to(reference, power.shape)):
        raise ValueError("synthetic restoring equilibrium differs")
    q, v, z = initial[:, 0, :10], initial[:, 0, 10:20], initial[:, 0, 20:]
    a = np.zeros_like(power)
    blocks = (mass, _incidence(), stiffness, damping, (np.eye(10)-mass[:, None]/mass.sum())@np.diag(port), tau)
    budget.spend("numpy_damping_roots")
    damping_root = spatial_damping_root(blocks)
    indices = {role: [i for i, value in enumerate(cases["roles"]) if value == role] for role in ("train", "val", "test")}
    train = indices["train"]
    observer = partial(predict_from_topology_history, memory_radius=plan["radius"], prior_weight=plan["prior_weight"])
    model = make_model(params, "T0M1", "cuda").eval()
    model.set_angle_damping(2./3., spatial_mix=1.)
    p, equilibrium = model._tensor(power), model._tensor(eq)
    populations = {}

    def records(role, target):
        return [{"id": cases["ids"][i], "role": role, **restoring_momentum(target[i, :311, :20], power[i], eq[i],
                 blocks, .01, budget.spend, root=damping_root)} for i in indices[role]]

    for truth in plan["truth_alpha"]:
        tag = f"alpha-{truth:+g}"
        fine, refined = (truth_rollout(params, q, v, power, a, z, count=810, max_step=step, spend=budget.spend,
                                       angle_damping_rho=2./3., spatial_mix=1., restoring_alpha=truth)
                         for step in (.00125, .000625))
        scale = np.sqrt(np.mean(refined[train, 11:311, :20]**2, axis=(0, 1))).clip(min=1e-5)
        hscale = np.sqrt(np.mean(refined[train, :11, :20]**2, axis=(0, 1))).clip(min=1e-5)
        rows = {"train": records("train", refined)}
        budget.spend("fits", 2)
        fit = fit_restoring_train(rows["train"])
        if fit is None:
            raise RuntimeError("complete synthetic train did not identify both restoring-force directions")
        write_json(directory/f"{tag}-freeze.json", {"truth_alpha": truth, "fit": fit, "history_scale": hscale.tolist(),
                   "target_scale": scale.tolist(), "training_ids": [cases["ids"][i] for i in train]})
        rows.update({role: records(role, refined) for role in ("val", "test")})
        weak = {}
        for role, group in rows.items():
            write_json(directory/f"{tag}-{role}-weak.json", group)
            weak[role] = {"events": len(group), **restoring_sse(group, fit),
                          "truth": restoring_sse(group, {"shape": {"value": truth},
                                                         "scale": {"value": 0.}})["shape"]}
        labels = _truth_classes(fine, refined, eq, mass, scale)
        write_json(directory/f"{tag}-truth-labels.json", labels)
        np.savez_compressed(directory/f"{tag}-truth.npz", fine=fine, refined=refined, power=power, equilibrium=eq)
        model.set_restoring_shape(fit["shape"]["value"])
        energy = energy_residual(model, model._tensor(refined[train, 10]), p[train], equilibrium[train], budget.spend)
        models = {}
        for name, alpha in (("base", 0.), ("shape", fit["shape"]["value"])):
            model.set_restoring_shape(alpha)
            paths, histories, failures = simulate(model, refined, a, p, equilibrium, hscale, budget,
                                                  oracle=False, observer=observer)
            score = scoring(model, cases, paths, histories, refined, a, eq, mass, scale, hscale, labels, False)
            matched_margins(model, paths, refined, labels, equilibrium, score)
            score["failures"] = failures
            models[name] = score
            write_json(directory/f"{tag}-{name}-scores.json", score)
            np.savez_compressed(directory/f"{tag}-{name}-predictions.npz", forecasts=paths, histories=histories)
        numerics = (energy <= 1e-10 and all(labels[i]["label"] is not None for i in train+indices["val"])
                    and all(row["rank"] == 10 and row["quadrature_gap"] <= 1e-10
                            for role in ("train", "val") for row in rows[role])
                    and all(row["qualified"] for score in models.values()
                            for row in score["events"] if row["role"] != "test"))
        checks = restoring_gates(models, weak, truth, fit, plan["gates"])
        populations[tag] = {"fit": fit, "engineering_pass": numerics, "gates": checks, "weak": weak,
                            "energy_error": energy, "models": {key: model["roles"] for key, model in models.items()}}
        print(f"{tag}: complete 24/12/12, fit {fit['shape']['value']:.8g}, gates {checks}", flush=True)
    engineering = all(row["engineering_pass"] for row in populations.values())
    return {"engineering_pass": engineering,
            "scientific_pass": engineering and all(all(row["gates"].values()) for row in populations.values()),
            "populations": populations,
            "claim_boundary": "Paired known-truth restoring-force shape recovery and causal forecasts."}
