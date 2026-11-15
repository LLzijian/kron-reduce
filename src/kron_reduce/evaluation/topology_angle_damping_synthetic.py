"""Paired clean truth and causal scoring for the frozen angle-damping law."""

from __future__ import annotations

from functools import partial

import numpy as np

from kron_reduce.evaluation.constant_c_observability import write_json
from kron_reduce.evaluation.machine_local_equation import align_classes
from kron_reduce.evaluation.machine_local_topology import (
    energy_residual,
    make_model,
    scoring,
    simulate,
)
from kron_reduce.evaluation.machine_local_topology_ball_history_no_prior import _file
from kron_reduce.evaluation.machine_local_topology_delayed_origin import load_arrays
from kron_reduce.evaluation.machine_local_topology_margin import matched_margins
from kron_reduce.evaluation.machine_local_topology_truth import (
    _incidence,
    parameters,
    truth_rollout,
)
from kron_reduce.evaluation.topology_angle_momentum import (
    angle_momentum,
    damping_sse,
    fit_train_damping,
)
from kron_reduce.evaluation.trajectory_stability import classify
from kron_reduce.models.machine_local_topology import predict_from_topology_history


def _truth_classes(fine, refined, equilibrium, mass, scale):
    labels = []
    for low, high, eq in zip(fine, refined, equilibrium, strict=True):
        a, b = (classify(x[:11, :10], x[:11, 10:20], x[10:, :20], eq, mass) for x in (low, high))
        gap = float(np.max(np.abs((low[:, :20]-high[:, :20])/scale)))
        labels.append(b | {"reference_qualified": bool(align_classes(a, b)
                      and gap <= 1e-6 and (b["trigger_index"] is None or b["trigger_index"] > 10)),
                      "reference_step_gap": gap})
    return labels


def _stable_supported(score, role):
    rows = [row for row in score["events"] if row["role"] == role and row["truth_label"] == "stable"
            and not row["id"].startswith("exterior-")]
    return (float(np.sqrt(np.mean([row["trajectory"]["8"]["nmse"] for row in rows])))
            if rows and all(row["qualified"] for row in rows) else None)


def _gates(models, weak, rho, fit, gates):
    base, candidate = models["constant"], models["angle"]
    dev = [row for row in candidate["events"] if row["role"] in ("train", "val")]
    feasible = [row for row in dev if not row["id"].startswith("exterior-")]
    checks = {"coefficient": abs(fit["angle"]-rho) <= gates["rho_error"],
              "feasible_trace": all(row["truth_agreement"] and row["trajectory"]["8"]["nrmse"] <= gates["feasible_nrmse"]
                                    for row in feasible)}
    if not rho:
        checks["constant_null"] = abs(fit["constant"]) <= gates["rho_error"]
    if not rho:
        checks["constant_null"] = abs(fit["constant"]) <= gates["rho_error"]
    if rho:
        stable, stable_control = (_stable_supported(score, "val") for score in (candidate, base))
        pairs = [(candidate["roles"][role], base["roles"][role]) for role in ("train", "val")]
        checks.update({"weak_separation": all(weak[role]["constant"] > 100.*weak[role]["true_rho"]
                       and weak[role]["angle"] <= gates["weak_ratio"]*weak[role]["constant"]
                       for role in ("train", "val")),
                       "stable_improvement": stable is not None and stable_control is not None
                       and stable <= gates["stable_ratio"]*stable_control,
                       "matched_witness": all(x["matched_margin"]["unstable"]["predicted_triggers"]
                                              >= y["matched_margin"]["unstable"]["predicted_triggers"] for x,y in pairs),
                       "no_additional_false_alarm": all(x["stability"]["fp"] <= y["stability"]["fp"] for x,y in pairs)})
    return checks


def execute(root, directory, plan, budget):
    saved = load_arrays(_file(root, plan["source"], "truth.npz"),
                        ("fine", "refined", "power", "equilibrium", "ids", "roles"), budget)
    cases = {key: plan[key] for key in ("ids", "roles")}
    shapes = {"fine": (48, 811, 30), "refined": (48, 811, 30), "power": (48, 10), "equilibrium": (48, 10)}
    if (any(saved[key].tolist() != cases[key] for key in cases)
            or any(saved[key].shape != shape or saved[key].dtype != np.float64 or not np.isfinite(saved[key]).all()
                   for key, shape in shapes.items())):
        raise ValueError("frozen synthetic angle-damping roster or arrays differ")
    params = parameters()
    mass, stiffness, damping, port, tau, reference = params
    original, power, eq = (saved[key] for key in ("refined", "power", "equilibrium"))
    if not np.array_equal(eq, np.broadcast_to(reference, power.shape)):
        raise ValueError("synthetic equilibrium differs")
    q, v, a, z = original[:, 0, :10], original[:, 0, 10:20], np.zeros_like(power), original[:, 0, 20:]
    blocks = (mass, _incidence(), stiffness, damping, (np.eye(10)-mass[:, None]/mass.sum())@np.diag(port), tau)
    indices = {role: [i for i, value in enumerate(cases["roles"]) if value == role] for role in ("train", "val", "test")}
    train = indices["train"]
    observer = partial(predict_from_topology_history, memory_radius=plan["radius"], prior_weight=plan["prior_weight"])
    populations = {}
    for rho in (0., .5):
        tag = f"rho-{rho:g}"
        fine, refined = (saved["fine"], original) if rho == 0. else tuple(
            truth_rollout(params, q, v, power, a, z, count=810, max_step=step, spend=budget.spend, angle_damping_rho=rho)
            for step in (.00125, .000625))
        scale = np.sqrt(np.mean(refined[train, 11:311, :20]**2, axis=(0, 1))).clip(min=1e-5)
        hscale = np.sqrt(np.mean(refined[train, :11, :20]**2, axis=(0, 1))).clip(min=1e-5)

        def records(role, target):
            return [{"id": cases["ids"][i], "role": role, **angle_momentum(target[i, :311, :20], power[i], eq[i],
                     blocks, .01, budget.spend, check_quadrature=True)} for i in indices[role]]

        rows = {"train": records("train", refined)}
        budget.spend("fits", 2)
        fit = fit_train_damping(rows["train"])
        if fit is None:
            raise RuntimeError("complete synthetic train could not identify its weak damping coefficient")
        write_json(directory/f"{tag}-freeze.json", {"truth_rho": rho, "fit": fit, "training_ids": [cases["ids"][i] for i in train],
                                                     "history_scale": hscale.tolist(), "target_scale": scale.tolist()})
        rows.update({role: records(role, refined) for role in ("val", "test")})
        for role, result in rows.items():
            write_json(directory/f"{tag}-{role}-weak.json", result)
        weak = {role: damping_sse(group, fit) for role, group in rows.items()}
        for role, group in rows.items():
            weak[role]["true_rho"] = damping_sse(group, {"angle": rho})["angle"]
        labels = _truth_classes(fine, refined, eq, mass, scale)
        write_json(directory/f"{tag}-truth-labels.json", labels)
        np.savez_compressed(directory/f"{tag}-truth.npz", fine=fine, refined=refined, power=power, equilibrium=eq)
        base = make_model(params, "T0M1", "cuda").eval()
        base.set_angle_damping(rho)
        p, equilibrium = base._tensor(power), base._tensor(eq)
        energy_error = energy_residual(base, base._tensor(refined[train, 10]), p[train], equilibrium[train], budget.spend)
        models = {}
        for name in ("constant", "angle"):
            coefficients = (mass, stiffness, (1.+fit[name])*damping, port, tau, reference) if name == "constant" else params
            model = make_model(coefficients, "T0M1", "cuda").eval()
            if name == "angle":
                model.set_angle_damping(fit[name])
            paths, histories, failures = simulate(model, refined, a, p, equilibrium, hscale, budget,
                                                  oracle=False, observer=observer)
            score = scoring(model, cases, paths, histories, refined, a, eq, mass, scale, hscale, labels, False)
            matched_margins(model, paths, refined, labels, equilibrium, score)
            score["failures"] = failures
            models[name] = score
            write_json(directory/f"{tag}-{name}-scores.json", score)
            np.savez_compressed(directory/f"{tag}-{name}-predictions.npz", forecasts=paths, histories=histories)
        numerics = (energy_error <= 1e-10 and all(labels[i]["label"] is not None for i in train+indices["val"])
                    and all(row["rank"] == 10 and row["quadrature_gap"] <= 1e-10
                                                for role in ("train", "val") for row in rows[role])
                    and all(row["qualified"] for model in models.values() for row in model["events"] if row["role"] != "test"))
        checks = _gates(models, weak, rho, fit, plan["gates"])
        populations[tag] = {"fit": fit, "engineering_pass": numerics, "gates": checks, "weak": weak,
                            "energy_error": energy_error, "models": {key: model["roles"] for key, model in models.items()}}
        print(f"{tag}: complete 24/12/12, fit {fit}, gates {checks}", flush=True)
    engineering = all(row["engineering_pass"] for row in populations.values())
    return {"engineering_pass": engineering,
            "scientific_pass": engineering and all(all(row["gates"].values()) for row in populations.values()),
            "populations": populations, "claim_boundary": "Paired known-truth damping recovery and causal synthetic forecasts."}
