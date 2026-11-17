"""Known-coefficient comparison of overall and branch-dependent coupling."""
from __future__ import annotations

import numpy as np

from kron_reduce.evaluation.machine_local_clean_learning import NUMERICAL_ERRORS
from kron_reduce.evaluation.machine_local_equation import ROLES, STEPS, align_classes
from kron_reduce.evaluation.machine_local_topology_truth import (
    _helmert_basis,
    _incidence,
    parameters,
)
from kron_reduce.evaluation.topology_memory_rom_causal import _read_source
from kron_reduce.evaluation.topology_memory_rom_synthetic import (
    _adequate,
    _cases,
    _forecast,
    _model,
    _pack_truth,
    _physical,
    _safe,
    _scores,
    _write,
)
from kron_reduce.evaluation.topology_memory_rom_truth import truth_rollout
from kron_reduce.evaluation.trajectory_stability import classify
from project_harness.evidence import sha256_file

SUPPORT = {"fixed": "C0S0H0", "overall": "C1S0H0", "branch": "C0S1H0", "both": "C1S1H0"}
TRUTHS = (*SUPPORT, "fifth_harmonic")
FRAMES = 20


def _match_initial_coupling(cases, mass, target_rms=.08, angle_spans=(.45, .65)):
    """Set equal RMS branch log multipliers before any integration."""
    n = len(mass)
    log_limit = np.log(2.)/2.
    edge_basis = np.abs(_incidence()).T @ _helmert_basis(n)
    radial_radius = log_limit/np.linalg.norm(edge_basis, axis=1).max()
    direction = cases["xi"]/np.linalg.norm(cases["xi"], axis=1, keepdims=True)
    rms = np.sqrt(np.mean((direction @ edge_basis.T)**2, axis=1, keepdims=True))
    norm = target_rms/np.sqrt(rms**2-(target_rms/radial_radius)**2)
    cases["xi"] = norm*direction
    number = len(cases["q"])//2
    signs = np.where(np.arange(number)%2, -1., 1.)
    cases["c"] = np.repeat(signs*.5*log_limit*np.arctanh(target_rms/log_limit), 2)[:, None]

    q = .15*cases["q"][::2].copy()
    nodes = np.arange(number)%n
    for offset, value in ((0, 1.), (3, -.65), (7, -.35)):
        q[np.arange(number), (nodes+offset)%n] += value
    q -= (q @ mass/mass.sum())[:, None]
    span = np.linspace(*angle_spans, number)
    cases["q"] = np.repeat(q*(span/np.ptp(q, axis=1))[:, None], 2, axis=0)


def _momentum_increment(path, mass):
    velocity = path[:, :FRAMES+1, 10:20]
    return (velocity[:, 1:]-velocity[:, :1])*mass


def _momentum_error(forecast, target, mass, scale):
    residual = (_momentum_increment(forecast, mass)-_momentum_increment(target, mass))/scale
    rms = np.sqrt(np.mean(residual**2, axis=(1, 2)))
    return np.where(np.isfinite(rms), rms, np.nan)


def _hidden_errors(histories, truth, *, common, branch):
    observed = histories[:, :, (0, -1)]
    reference = truth[:, (0, 10)]
    groups = {"memory": (slice(20, 30), .1)}
    if common:
        groups["overall"] = (slice(30, 31), .01)
    if branch:
        groups["branch"] = (slice(30+int(common), None), .1)
    return {key: np.linalg.norm(observed[..., block]-reference[..., block], axis=-1)
            / np.linalg.norm(reference[..., block], axis=-1).clip(min=floor)
            for key, (block, floor) in groups.items()}


def execute(root, directory, plan, budget):
    cases = _cases(plan)
    params = list(parameters())
    params[4] = np.geomspace(*plan["tau_range"], 10)
    mass, _, _, _, _, reference = params
    geometry = plan["excitation"]
    _match_initial_coupling(cases, mass, geometry["initial_log_rms"], geometry["angle_spans"])
    roles = np.array(cases["roles"])
    train, val = roles == "train", roles == "val"
    power = cases["power"]
    equilibrium = np.broadcast_to(reference, power.shape).copy()
    rates = (plan["truth"]["common_rate"], plan["truth"]["spatial_rate"])
    limits = plan["gates"]
    source = plan.get("source")
    if source is not None:
        baseline = root/"results"/source["run_id"]
        if baseline.is_symlink() or baseline.is_junction() or baseline.resolve().parent != (root/"results").resolve():
            raise ValueError("frozen coupling comparison directory differs")
        manifest = _read_source(baseline/"artifact_hashes.json", source["manifest_sha256"])
    models = {key: _model(params, support, plan, truth=True).eval() for key, support in SUPPORT.items()}
    records = {}
    for population in TRUTHS:
        folder = directory/population
        folder.mkdir()
        generating = population if population in SUPPORT else "fixed"
        common = generating in ("overall", "both")
        branch = generating in ("branch", "both")
        topology = {"fixed": "none", "overall": "common", "branch": "spatial", "both": "both"}[generating]
        if source is None:
            fine, truth = (truth_rollout(params, cases["q"], cases["v"], power, cases["z"], count=810,
                          max_step=step, spend=budget.spend, topology=topology,
                          c=cases["c"][:, 0] if common else None, xi=cases["xi"] if branch else None,
                          topology_dynamics=(rates, (1., 1.)) if common or branch else None,
                          fifth_harmonic=plan["truth"]["fifth_harmonic"] if population == "fifth_harmonic" else 0.)
                           for step in (.00125, .000625))
        else:
            truth_name = f"{population}/truth.npz"
            path = baseline/truth_name
            if path.is_symlink() or path.is_junction() or sha256_file(path) != manifest[truth_name]:
                raise ValueError("frozen coupling truth archive differs")
            budget.spend("npz_decodes")
            with np.load(path, allow_pickle=False) as data:
                if (data["ids"].tolist() != cases["ids"] or not np.array_equal(data["roles"], roles)
                        or not np.array_equal(data["power"], power) or not np.array_equal(data["equilibrium"], equilibrium)):
                    raise ValueError("frozen coupling roster or inputs differ")
                fine, truth = data["fine"], data["refined"]
            old_summary = _read_source(baseline/population/"summary.json", manifest[f"{population}/summary.json"])
        scale = np.sqrt(np.mean(truth[train, 11:, :20]**2, axis=(0, 1))).clip(min=1e-4)
        hscale = np.sqrt(np.mean(truth[train, :11, :20]**2, axis=(0, 1))).clip(min=1e-4)
        momentum_scale = np.sqrt(np.mean(_momentum_increment(truth[train, 10:], mass)**2,
                                          axis=(0, 1))).clip(min=1e-4)
        truth_momentum_gap = float(_momentum_error(fine[:, 10:], truth[:, 10:], mass, momentum_scale)[train | val].max())
        classes_fine, classes = ([classify(row[:11, :10], row[:11, 10:20], row[10:, :20], reference, mass)
                                  for row in path] for path in (fine, truth))
        step_gap = np.max(np.abs((fine[..., :20]-truth[..., :20])/scale), axis=(1, 2))
        for coarse, refined, gap in zip(classes_fine, classes, step_gap, strict=True):
            refined["reference_qualified"] = bool(gap <= limits["truth_step_max"] and align_classes(coarse, refined))
        if source is None:
            np.savez_compressed(folder/"truth.npz", fine=fine, refined=truth, roles=roles, ids=cases["ids"],
                                power=power, equilibrium=equilibrium, scale=scale, history_scale=hscale,
                                momentum_scale=momentum_scale)
        else:
            _write(folder/"source.json", {"run_id": source["run_id"], "truth_sha256": manifest[truth_name],
                "reused_score_sha256": {key: manifest[f"{population}/{key}-scores.json"] for key in ("fixed", "overall")}})

        candidates = {}
        for candidate, model in models.items():
            if source is not None and candidate in ("fixed", "overall"):
                candidates[candidate] = old_summary["candidates"][candidate] | {"reused": True}
                continue
            forecast, histories, failures, _, _ = _forecast(model, roles, truth[:, :11, :20], power,
                                                             equilibrium, hscale, plan, budget)
            score = _scores(cases, forecast, truth, truth[:, :11, :20], equilibrium, mass, scale, classes, limits)
            momentum = np.stack([_momentum_error(path, truth[:, 10:], mass, momentum_scale) for path in forecast])
            for index, row in enumerate(score["events"]):
                row["momentum_nrmse"] = momentum[:, index].tolist()
            matched = population == candidate
            hidden_error = {}
            if matched:
                relative = _hidden_errors(histories, truth, common=common, branch=branch)
                for index, row in enumerate(score["events"]):
                    row["relative_hidden_error"] = {key: value[:, index].tolist() for key, value in relative.items()}
                hidden_error = {key: float(value[:, val].max()) for key, value in relative.items()}
            score["failures"] = failures
            _write(folder/f"{candidate}-scores.json", score)
            np.savez_compressed(folder/f"{candidate}-paths.npz", forecasts=forecast, histories=histories)

            origin = _pack_truth(truth[val, 10], SUPPORT[generating], SUPPORT[candidate])
            ideal_failure = None
            try:
                ideal_path = model.rollout(model._tensor(origin), model._tensor(power[val]),
                                           model._tensor(equilibrium[val]), FRAMES, max_step=STEPS[1], spend=budget.spend)
                ideal_error = _momentum_error(ideal_path.cpu().numpy(), truth[val, 10:], mass, momentum_scale)
            except NUMERICAL_ERRORS as error:
                ideal_error = np.full(int(val.sum()), np.nan)
                ideal_failure = str(error)
            errors_by_step = np.max(momentum[:, val], axis=1)
            observed_error = float(errors_by_step.max())
            trajectories_pass = _adequate(score, limits)
            candidates[candidate] = {"val_momentum_max": observed_error, "val_momentum_known_origin_max": float(ideal_error.max()),
                                     "val_momentum_max_by_step": errors_by_step.tolist(),
                                     "val_trajectory_pass": trajectories_pass,
                                     "val_hidden_error_max": hidden_error,
                                     "passes": bool(trajectories_pass and observed_error <= limits["momentum_nrmse_max"]),
                                     "val_complete": all(row["qualified"] for row in score["events"] if row["role"] == "val"),
                                     "roles": score["roles"]}
            _write(folder/f"{candidate}-known-origin.json", {"ids": np.array(cases["ids"])[val],
                                                              "nrmse": ideal_error, "failure": ideal_failure})

        expected = None if population == "fifth_harmonic" else generating
        adequate = [name for name, row in candidates.items() if row["passes"]]
        selected = min(adequate, key=lambda name: (name != "fixed", name == "both", candidates[name]["val_momentum_max"], name)) if adequate else None
        model = models[generating]
        energy = _physical(model, truth[val, 10], power[val], equilibrium[val], budget)
        physical = (all(row["reference_qualified"] for row, role in zip(classes, roles, strict=True) if role != "test")
                    and energy <= limits["energy_balance_max"]
                    and truth_momentum_gap <= limits["momentum_nrmse_max"]/10.)
        complete = all(row["val_complete"] for row in candidates.values())
        mixed = all({row["label"] for row, role in zip(classes, roles, strict=True) if role == split}
                    == {"stable", "unstable"} for split in ROLES if split != "test")
        identified = selected == expected and complete and mixed
        if expected is None:
            identified &= any(row["val_trajectory_pass"] for row in candidates.values())
            identified &= all(all(value > limits["momentum_nrmse_max"] for value in row["val_momentum_max_by_step"])
                              for row in candidates.values())
        else:
            matched_result = candidates[expected]
            identified &= (all(error <= limits["initial_relative_error"]
                               for error in matched_result["val_hidden_error_max"].values())
                           and matched_result["val_momentum_known_origin_max"] <= limits["momentum_nrmse_max"])
        record = {"expected": expected, "selected": selected, "candidates": candidates,
                  "val_numerical_coverage": complete, "mixed_train_val_truth": mixed,
                  "truth_momentum_step_gap_max": truth_momentum_gap,
                  "energy_residual_max": energy, "engineering_pass": physical,
                  "scientific_pass": physical and identified}
        _write(folder/"summary.json", record)
        records[population] = record
        print(f"{population}: selected={selected} scientific_pass={record['scientific_pass']}", flush=True)

    return _safe({"populations": records, "engineering_pass": all(row["engineering_pass"] for row in records.values()),
            "scientific_pass": all(row["scientific_pass"] for row in records.values()),
            "claim_boundary": "Clean synthetic support comparison at known coefficients. No fit or real data access."})
