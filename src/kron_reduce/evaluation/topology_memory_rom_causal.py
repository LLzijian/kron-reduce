"""Isolate the hidden-state prior with fixed coefficients and frozen synthetic truth."""
from __future__ import annotations

import json

import numpy as np

from kron_reduce.evaluation.machine_local_equation import ROLES, align_classes
from kron_reduce.evaluation.machine_local_topology_truth import parameters
from kron_reduce.evaluation.topology_memory_rom_synthetic import (
    POPULATIONS,
    _adequate,
    _cases,
    _forecast,
    _model,
    _scores,
    _write,
)
from kron_reduce.evaluation.trajectory_stability import classify
from project_harness.evidence import sha256_file


def _errors(estimated, truth):
    hidden = estimated[0][:, (0, -1), 20:]
    actual = truth[:, (0, 10), 20:]
    blocks = {"all": slice(None), "memory": slice(0, 10)}
    if actual.shape[-1] > 10:
        blocks["topology"] = slice(10, None)
    report = {}
    for name, block in blocks.items():
        error = np.linalg.norm(hidden[..., block]-actual[..., block], axis=-1)
        norm = np.linalg.norm(actual[..., block], axis=-1)
        report[name] = {"absolute": error, "relative": error/np.maximum(.1, norm),
                        "estimated_norm": np.linalg.norm(hidden[..., block], axis=-1), "truth_norm": norm}
    return report


def _read_source(path, expected):
    if path.is_symlink() or path.is_junction() or sha256_file(path) != expected:
        raise ValueError(f"frozen synthetic artifact differs: {path.name}")
    return json.loads(path.read_bytes())


def execute(root, directory, plan, budget):
    """Evaluate both observers on all four in-library truth systems and full rosters."""
    source = plan["source"]
    prior_plan = _read_source(root/source["plan_path"], source["plan_sha256"])
    base = root/"results"/source["run_id"]
    if base.is_symlink() or base.is_junction() or base.resolve().parent != (root/"results").resolve():
        raise ValueError("frozen synthetic result path differs")
    manifest = _read_source(base/"artifact_hashes.json", source["manifest_sha256"])
    cases = _cases(prior_plan)
    ids, roles = cases["ids"], np.array(cases["roles"])
    params = list(parameters())
    params[4] = np.geomspace(*prior_plan["tau_range"], 10)
    mass = params[0]
    limits = prior_plan["gates"]
    all_results = {}
    weights = plan["comparison_prior_weights"]
    if weights != {"frozen_prior": prior_plan["settings"]["prior_weight"], "zero_prior": 0.}:
        raise ValueError("the comparison may change only the frozen history prior")
    for tag in ("S0", "S1", "S2", "S3"):
        target = directory/tag
        target.mkdir()
        filename = f"{tag}/truth.npz"
        path = base/filename
        if path.is_symlink() or path.is_junction() or sha256_file(path) != manifest[filename]:
            raise ValueError(f"frozen synthetic trajectory archive differs: {tag}")
        budget.spend("npz_decodes")
        with np.load(path, allow_pickle=False) as data:
            if data["ids"].tolist() != ids or not np.array_equal(data["roles"], roles):
                raise ValueError(f"frozen roster differs: {tag}")
            fine, truth = data["fine"], data["refined"]
            power, eq, scale, hscale = (data[key] for key in ("power", "equilibrium", "scale", "history_scale"))
            noise = data["history_noise"]
        classes = [[classify(s[:11, :10], s[:11, 10:20], s[10:, :20], eq[i], mass)
                    for i, s in enumerate(path)] for path in (fine, truth)]
        gap = np.max(np.abs((fine[..., :20]-truth[..., :20])/scale), axis=(1, 2))
        for coarse, refined, error in zip(*classes, gap, strict=True):
            refined["reference_qualified"] = bool(error <= limits["truth_step_max"] and align_classes(coarse, refined))
        model = _model(params, POPULATIONS[tag], prior_plan, truth=True)
        results, paired_values = {}, {}
        for observer, weight in weights.items():
            evaluation = prior_plan | {"settings": prior_plan["settings"] | {"prior_weight": weight}}
            mode_results = {}
            for mode, history in (("clean", truth[:, :11, :20]), ("noise", truth[:, :11, :20]+noise)):
                predicted, estimated, failures, rank, jacobian = _forecast(
                    model, roles, history, power, eq, hscale, evaluation, budget, ranks=mode == "clean",
                )
                score = _scores(cases, predicted, truth, history, eq, mass, scale, classes[1], limits, rank)
                errors = _errors(estimated, truth)
                residual = (estimated[0, :, 1:, :20]-history[:, 1:])/hscale
                history_nrmse = np.sqrt(np.mean(residual**2, axis=(1, 2)))
                paired_values[observer, mode] = {"history_nrmse": history_nrmse,
                                                 "start_error": errors["all"]["relative"][:, 0],
                                                 "origin_error": errors["all"]["relative"][:, 1]}
                for i, event in enumerate(score["events"]):
                    event["state_recovery"] = {block: {metric: values[i] for metric, values in row.items()}
                                               for block, row in errors.items()}
                    event["history_nrmse"] = history_nrmse[i]
                score["failures"] = failures
                _write(target/f"{observer}-{mode}.json", score)
                arrays = {"predictions": predicted, "histories": estimated}
                if jacobian is not None:
                    arrays["observation_jacobian"] = jacobian
                np.savez_compressed(target/f"{observer}-{mode}.npz", **arrays)
                relative = errors["all"]["relative"]
                state = {}
                for role in ROLES:
                    block = relative[roles == role]
                    qualified = np.isfinite(block).all(axis=1)
                    state[role] = {"all_finite": bool(qualified.all()), "finite": int(qualified.sum()),
                                   "max_relative_start": float(block[:, 0].max()) if qualified.all() else None,
                                   "max_relative_origin": float(block[:, 1].max()) if qualified.all() else None}
                mode_results[mode] = {"roles": score["roles"], "state": state,
                                      "train_val_qualified": all(row["qualified"] for row in score["events"]
                                                                  if row["role"] != "test"),
                                      "val_prediction_gate": _adequate(score, limits, noisy=mode == "noise")}
            results[observer] = mode_results
        deltas = {}
        for mode in ("clean", "noise"):
            prior = paired_values["frozen_prior", mode]
            unregularized = paired_values["zero_prior", mode]
            changes = {metric: unregularized[metric]-value for metric, value in prior.items()}
            deltas[mode] = [{"id": event, "role": roles[i], **{key: value[i] for key, value in changes.items()}}
                            for i, event in enumerate(ids)]
        _write(target/"paired-zero-minus-prior.json", deltas)
        state = results["zero_prior"]["clean"]["state"]["val"]
        reference = results["frozen_prior"]["clean"]["state"]["val"]
        pass_clean = (state["all_finite"] and reference["all_finite"]
                      and results["zero_prior"]["clean"]["val_prediction_gate"]
                      and max(state["max_relative_start"], state["max_relative_origin"]) <= limits["initial_relative_error"]
                      and state["max_relative_start"] < reference["max_relative_start"])
        paired_qualified = all(row["train_val_qualified"] for modes in results.values() for row in modes.values())
        paired_prediction = all(row["val_prediction_gate"] for modes in results.values() for row in modes.values())
        oracle_file = f"{tag}/known-coefficient-true-origin.json"
        oracle = _read_source(base/oracle_file, manifest[oracle_file])
        all_results[tag] = {"prior_weights": weights, "results": results, "clean_recovery_pass": pass_clean,
                            "known_coefficient_true_origin": oracle["roles"],
                            "paired_train_val_coverage_pass": paired_qualified,
                            "paired_val_prediction_pass": paired_prediction,
                            "noisy_prediction_pass": results["zero_prior"]["noise"]["val_prediction_gate"]}
        _write(target/"summary.json", all_results[tag])
        print(f"{tag}: zero-prior val clean recovery {pass_clean}", flush=True)
    engineering = all(row["paired_train_val_coverage_pass"] for row in all_results.values())
    return {"populations": all_results, "engineering_pass": engineering,
            "scientific_pass": engineering and all(row["clean_recovery_pass"] and row["paired_val_prediction_pass"]
                                                    for row in all_results.values()),
            "claim_boundary": "Fixed coefficients isolate the history prior. This diagnostic does not authorize real fitting or certify structure recovery."}
