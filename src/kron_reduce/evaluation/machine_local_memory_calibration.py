"""Test train-only integral coefficient calibration on the full synthetic cohorts."""
from __future__ import annotations

import numpy as np
import torch

from kron_reduce.evaluation.constant_c_observability import write_json
from kron_reduce.evaluation.kiki_machine_local_full import inputs, predict_checked
from kron_reduce.evaluation.machine_local_clean_learning import NUMERICAL_ERRORS
from kron_reduce.evaluation.machine_local_equation import POPULATIONS, ROLES, function_checks, score
from kron_reduce.evaluation.machine_local_equation_cases import make_population
from kron_reduce.evaluation.machine_local_equation_fit import freeze, support_values
from kron_reduce.evaluation.machine_local_memory_diagnosis import (
    load_selected,
    source_run,
    substitute,
)
from kron_reduce.evaluation.machine_local_memory_fit import momentum_fit
from project_harness.evidence import sha256_file


def monitor(model, events, hscale, tscale, budget):
    budget.spend("val_checks")
    model.to("cuda")
    try:
        with torch.no_grad():
            result = predict_checked(model, *inputs(events, hscale), 300, max_step=.005, spend=budget.spend)
            truth = model._tensor(np.stack([event["x"][11:311] for event in events]))
            errors = ((result["forecast"][:, 1:, :20]-truth)/model._tensor(tscale)).square()
            by_event = .5*(errors[:, :100].mean(dim=(1, 2))+errors.mean(dim=(1, 2)))
            return float(by_event.mean()), by_event.cpu().tolist()
    except NUMERICAL_ERRORS:
        return None, [None]*len(events)
    finally:
        model.to("cpu")


def recovery_gates(record, diagnostics, trained, strengths, expected):
    metrics = [record[role]["horizons"] for role in ("train", "val")]
    errors = diagnostics.get("function_relative_errors", {})
    return {"trajectory": all(m["3"]["pooled_nrmse"] is not None and m["3"]["pooled_nrmse"] <= .01
                              and m["8"]["pooled_nrmse"] <= .03 for m in metrics),
            "functions": all(errors.get(k, np.inf) <= tol for k, tol in
                             (("potential_force", .03), ("visible_damping_force", .05), ("memory_kernel", .05))),
            "support": all(got == 0 if true == 0 else abs(got/true-1) <= .1
                           for got, true in zip(strengths, expected, strict=True)),
            "valid_fit": trained["selected_val_nmse"] is not None and trained["failure"] is None}


def execute(root, directory, plan, budget):
    source = source_run(root, plan)
    summaries = {}
    for name in POPULATIONS:
        pop = make_population(name, budget)
        output = directory/name
        output.mkdir()
        starts = {label: load_selected(source, pop, spec, budget) for label, spec in plan["sources"][name].items()}
        models, fits = {"known": pop["truth_model"]}, {}
        for label, initial in starts.items():
            model = substitute(initial, initial, ())
            selection = lambda candidate, pop=pop: monitor(candidate, pop["roles"]["val"], pop["history_scale"],
                                                           pop["target_scale"], budget)
            fits[label] = momentum_fit(model, pop["roles"]["train"], pop["target_scale"], budget,
                                       max_calls=150, monitor=selection)
            models[f"{label}_before"] = initial
            models[f"{label}_calibrated"] = model
            print(f"{name}/{label}: {len(fits[label]['train_trace'])} train objectives; val={fits[label]['selected_val_nmse']}", flush=True)
        for label, model in models.items():
            path = output/label
            path.mkdir()
            freeze(model, path)
        write_json(output/"fit.json", fits)
        write_json(output/"freeze.json", {"models": {key: sha256_file(output/key/"frozen.npz") for key in models},
                   "selection_closed": True, "preprocessing": {key: pop[key].tolist() for key in ("history_scale", "target_scale")}})

        records, diagnostics = {}, {}
        for label, model in models.items():
            try:
                diagnostics[label] = function_checks(model, pop, budget)
            except (FloatingPointError, np.linalg.LinAlgError) as error:
                diagnostics[label] = {"failure": str(error)}
            model.to("cuda")
            records[label] = {role: score(model, pop, role, budget, output/label) for role in ROLES}
            model.to("cpu")
        gates = {label: recovery_gates(records[f"{label}_calibrated"], diagnostics[f"{label}_calibrated"], fits[label],
                                       support_values(models[f"{label}_calibrated"]), support_values(models["known"]))
                 for label in starts}
        known = all(records["known"][role]["horizons"]["3"]["pooled_nrmse"] is not None
                    and records["known"][role]["horizons"]["3"]["pooled_nrmse"] <= .001
                    and {e["truth_label"] for e in records["known"][role]["events"]} == {"stable", "unstable"}
                    and all(e["truth_numerical"] and e["truth_agreement"] and e["prospective_eligible"]
                            and e["history_max_error"] is not None and e["history_max_error"] <= 1e-7
                            and e["hidden_start_rmse"] <= 1e-4 and e["hidden_origin_rmse"] <= 1e-4
                            for e in records["known"][role]["events"]) for role in ("train", "val"))
        physical = all("failure" not in row and row["energy_balance_relative_max"] <= 1e-10
                       and row["minimum_dissipation"] >= -1e-12 and row["reference_min_coi_curvature"] > 1e-8
                       for row in diagnostics.values())
        passed = known and physical and all(all(gate.values()) for gate in gates.values())
        summaries[name] = {"fits": fits, "gates": gates, "known_control": known, "physical": physical,
            "scientific_pass": passed, "functions": diagnostics,
            "records": {label: {role: {k: r[k] for k in ("horizons", "stability", "failures")}
                                  for role, r in roles.items()} for label, roles in records.items()}}
        print(f"{name}: complete synthetic calibration evaluation, gate={passed}", flush=True)
    return {"populations": summaries, "scientific_pass": all(row["scientific_pass"] for row in summaries.values()),
            "claim_boundary": "Fixed-support coefficients fitted from synthetic train visible trajectories; causal GPU forecasts."}
