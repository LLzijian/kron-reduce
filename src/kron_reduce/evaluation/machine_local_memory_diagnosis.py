"""Frozen coefficient substitutions on the complete synthetic train cohorts."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from kron_reduce.evaluation.constant_c_observability import write_json
from kron_reduce.evaluation.kiki_machine_local_full import inputs, predict_checked
from kron_reduce.evaluation.machine_local_clean_learning import NUMERICAL_ERRORS
from kron_reduce.evaluation.machine_local_equation import POPULATIONS, STEPS, physical
from kron_reduce.evaluation.machine_local_equation_cases import make_population
from kron_reduce.evaluation.machine_local_equation_fit import make_model, support_values
from project_harness.evidence import sha256_file

MEMORY = ("port_parameter", "rate_parameter")
VISIBLE = ("edge_log_factor", "damping_log_factor")
SUBSTITUTIONS = {"selected": (), "memory_truth": MEMORY, "visible_truth": VISIBLE, "both_truth": MEMORY+VISIBLE}


def substitute(model, truth, names):
    candidate = deepcopy(model)
    with torch.no_grad():
        for name in names:
            getattr(candidate, name).copy_(getattr(truth, name))
    return candidate


def kernel(model):
    _, _, coupling, tau, _ = physical(model)
    time = np.geomspace(.01, 3., 32)
    return np.einsum("ij,tj,kj->tik", coupling, np.exp(-time[:, None]/tau), coupling)


def source_run(root, plan):
    run_id = plan["source_run_id"]
    if Path(run_id).name != run_id:
        raise ValueError("invalid source run ID")
    directory = root/"results"/run_id
    if sha256_file(directory/"artifact_hashes.json") != plan["source_manifest_sha256"]:
        raise ValueError("source artifact manifest differs")
    return directory


def load_selected(directory, population, source, budget):
    name = population["name"]
    support = source["support"]
    model = make_model(population, support)
    state_path = f"{name}/{support}/frozen.npz"
    for relative, digest in source["files"].items():
        path = (directory/relative).resolve(strict=True)
        if not path.is_relative_to(directory.resolve()) or sha256_file(path) != digest:
            raise ValueError(f"source file differs: {relative}")
    if state_path not in source["files"]:
        raise ValueError("selected model source is unbound")
    budget.spend("checkpoint_decodes")
    with np.load(directory/state_path, allow_pickle=False) as archive:
        expected = model.state_dict()
        if set(archive.files) != set(expected):
            raise ValueError("checkpoint fields differ from frozen support")
        loaded = {name: torch.from_numpy(archive[name]) for name in archive.files}
    for key, value in loaded.items():
        default = expected[key]
        if (value.shape != default.shape or value.dtype != default.dtype or not torch.isfinite(value).all()
                or key not in dict(model.named_parameters()) and not torch.equal(value, default)):
            raise ValueError(f"invalid checkpoint tensor: {key}")
    model.load_state_dict(loaded, strict=True)
    return model.requires_grad_(False).eval()


def forecast(model, population, budget):
    events = population["roles"]["train"]
    size = len(events)
    forecasts, histories, failures = [], [], []
    model = model.to("cuda")
    for step in STEPS:
        path = np.full((size, 301, 30), np.nan)
        history = np.full((size, 11, 30), np.nan)

        def forward(batch, indices, step=step, path=path, history=history):
            with torch.no_grad():
                result = predict_checked(model, *inputs(batch, population["history_scale"]),
                                         300, max_step=step, spend=budget.spend)
            path[indices] = result["forecast"].cpu().numpy()
            history[indices] = result["history"].cpu().numpy()

        try:
            forward(events, slice(None))
        except NUMERICAL_ERRORS as error:
            failures.append({"step": step, "batch": True, "error": str(error)})
            for i, event in enumerate(events):
                try:
                    forward([event], slice(i, i+1))
                except NUMERICAL_ERRORS as individual:
                    failures.append({"step": step, "id": event["id"], "error": str(individual)})
        forecasts.append(path)
        histories.append(history)
    return np.stack(forecasts), np.stack(histories), failures


def summarize_forecast(population, paths, histories):
    events = population["roles"]["train"]
    truth = population["role_paths"]["train"]["refined"]
    scale, hscale = population["target_scale"], population["history_scale"]
    mass = population["anchor"].mass
    rows = []
    for i, event in enumerate(events):
        finite = bool(np.isfinite(paths[:, i]).all() and np.isfinite(histories[:, i]).all())
        truth_gap = float(np.max(np.abs((truth[i, :, :20]-population["role_paths"]["train"]["fine"][i, :, :20])/scale)))
        row = {"id": event["id"], "qualified": False, "truth_step_gap": truth_gap}
        if finite:
            step_gap = float(np.max(np.abs((paths[0, i, :, :20]-paths[1, i, :, :20])/scale)))
            scores = ((paths[0, i, 1:, :20]-truth[i, 11:311, :20])/scale)**2
            coi = float(np.max(np.abs(paths[:, i, :, :20].reshape(2, 301, 2, 10)@mass)))
            row.update(qualified=step_gap <= 1e-5 and truth_gap <= 1e-6 and coi <= 1e-10, step_gap=step_gap,
                       coi_error=coi,
                       nmse={str(h): float(scores[:round(100*h)].mean()) for h in (.5, 1, 3)},
                       history_nrmse=float(np.sqrt(np.mean(((histories[0, i, 1:, :20]-truth[i, 1:11, :20])/hscale)**2))),
                       hidden_origin_rmse=float(np.sqrt(np.mean((histories[0, i, 10, 20:]-truth[i, 10, 20:])**2))),
                       hidden_origin_energy=float(.5*np.sum(histories[0, i, 10, 20:]**2)))
        rows.append(row)
    usable = [row for row in rows if row["qualified"]]
    aggregate = {str(h): float(np.sqrt(np.mean([row["nmse"][str(h)] for row in usable]))) if len(usable) == len(rows) else None
                 for h in (.5, 1, 3)}
    return {"allocated": len(rows), "qualified": len(usable), "pooled_nrmse": aggregate, "events": rows}


def execute(root, directory, plan, budget):
    source = source_run(root, plan)
    rows = {}
    for name in POPULATIONS:
        population = make_population(name, budget, train_only=True)
        selected = load_selected(source, population, plan["sources"][name], budget)
        truth = population["truth_model"]
        target_kernel = kernel(truth)
        results = {}
        for label, names in SUBSTITUTIONS.items():
            model = substitute(selected, truth, names)
            memory_error = float(np.linalg.norm(kernel(model)-target_kernel)/np.linalg.norm(target_kernel))
            if support_values(model) != support_values(selected):
                raise RuntimeError("diagnostic substitution changed frozen function terms")
            paths, histories, failures = forecast(model, population, budget)
            record = summarize_forecast(population, paths, histories)
            record.update(memory_kernel_relative_error=memory_error, failures=failures)
            np.savez_compressed(directory/f"{name}-{label}.npz", forecasts=paths, histories=histories)
            results[label] = record
        support = plan["sources"][name]["support"]
        saved = json.loads((source/name/support/"train.json").read_text(encoding="utf-8"))["events"]
        observed = results["selected"]["events"]
        if [row["id"] for row in saved] != [row["id"] for row in observed]:
            raise RuntimeError("regenerated train roster differs from frozen scores")
        gaps = [abs(a["trajectory"][h]["nmse"]-b["nmse"][h]) for a, b in zip(saved, observed, strict=True)
                for h in ("1", "3") if a["trajectory"] is not None and "nmse" in b]
        replay = {"pairs": len(gaps), "max_abs_nmse_difference": max(gaps, default=None),
                  "pass": len(gaps) == 2*len(observed) and max(gaps) <= 1e-10}
        rows[name] = {"support": support, "strengths": support_values(selected), "controls": results, "selected_replay": replay}
        write_json(directory/f"{name}.json", rows[name])
        print(f"{name}: 4 coefficient controls × 16 train events complete", flush=True)
    return {"populations": rows, "scientific_pass": all(population["selected_replay"]["pass"] and
            all(row["qualified"] == row["allocated"] for row in population["controls"].values()) for population in rows.values()),
            "claim_boundary": "Frozen function terms; complete train-only synthetic block substitutions."}
