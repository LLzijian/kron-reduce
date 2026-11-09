"""Frozen encoder-energy-cap comparison on separately authorized saved train inputs."""
from __future__ import annotations

import json
import sys
from functools import partial

import numpy as np

from kron_reduce.data.energy_cap_saved_inputs import EnergyCapSavedInputs
from kron_reduce.evaluation.constant_c_observability import Operations, save_record, write_json
from kron_reduce.evaluation.encoder_energy_cap import encoder_energy_cap
from kron_reduce.evaluation.energy_cap_scores import (
    aggregate_group,
    score_history,
    score_pair,
    verdict,
)
from kron_reduce.evaluation.energy_constrained_initializer import EnergyBallNumericalError
from kron_reduce.evaluation.history_metrics import weighted_cvar
from kron_reduce.models.nonlinear_latent import NonlinearPassiveClosure
from project_harness.evidence import sha256_file
from project_harness.git_state import git_head

SEEDS = (324, 325, 326)
STEPS = ("0.005", "0.0025")
STEP_IDS = {"0.005": "h005", "0.0025": "h0025"}
PARAMETERS = ("mass", "edge_stiffness", "damping", "coupling", "latent_skew", "latent_dissipation")


def path_difference(coarse, fine, scale):
    if coarse.shape != fine.shape or not np.isfinite(coarse).all() or not np.isfinite(fine).all():
        raise ValueError("paired saved paths must be finite and aligned")
    visible = float(np.max(np.abs((coarse[:, :20] - fine[:, :20]) / scale)))
    hidden = float(np.max(np.linalg.norm(coarse[:, 20:] - fine[:, 20:], axis=1)))
    bound = 1e-5 * (1 + float(np.max(np.linalg.norm(fine[:, 20:], axis=1))))
    return {"visible_max_normalized": visible, "hidden_max_norm": hidden,
            "hidden_bound": bound, "qualified": visible <= 1e-5 and hidden <= bound}


def _rhs_call(ops, function, identity, *, physical=False):
    """Use real seed/event identities while sharing the existing durable scope journal."""
    category = "physical_rhs" if physical else "integration_rhs"
    if ops.scope_id is None or ops.counts[category] >= ops.limits.get(category, 0):
        raise RuntimeError("unreserved or exhausted RHS call")
    ops.index += 1
    row = {"category": "rhs", **identity, "sequence": ops.index}
    if physical:
        row["phase"] = "physical_checks"
    ops.journal.append({**row, "transition": "attempted"})
    ops.counts[category] += 1
    try:
        value = function()
    except BaseException as error:
        ops.journal.append({**row, "transition": "failed", "error_type": type(error).__name__})
        raise
    ops.last_rhs["returned"] = value
    if physical:
        ops.partial[f"physical-{ops.index}"] = dict(ops.last_rhs)
    ops.journal.append({**row, "transition": "returned"})
    ops.returns[category] += 1
    return value


def audited_path(model, initial, clock, power, equilibrium, step, ops, identity, directory, name):
    """Unchanged public scalar RK4; persist its path before checking each output point."""
    step = float(step)
    counts = np.ceil(np.diff(clock) / step - 1e-12).astype(int)
    if not np.all(counts == round(.01 / step)):
        raise ValueError("saved clock does not give the registered substep counts")
    original = model.rhs
    ops.rhs_context = identity

    def counted(*args):
        ops.last_rhs = {"state": np.array(args[0], copy=True)}
        return _rhs_call(ops, lambda: original(*args), identity)

    object.__setattr__(model, "rhs", counted)
    try:
        path = ops.call("rollouts", lambda: model.rollout(
            initial[:10], initial[10:20], initial[20:], clock, power, equilibrium, max_step=step),
            tag=name)
    finally:
        object.__delattr__(model, "rhs")
        ops.rhs_context = {}
    save_record(directory, name, {"path": path, "clock": clock, "initial": initial})
    if (path.shape != (len(clock), 22) or not np.isfinite(path).all()
            or not np.array_equal(path[0], initial)):
        raise FloatingPointError("candidate path shape, finiteness or complete origin differs")
    checks = []
    for state in path:
        ops.last_rhs = {"state": state.copy(), "phase": "physical_checks"}
        check = _rhs_call(ops, lambda state=state: model.energy_identity(state, power, equilibrium),
                          identity, physical=True)
        check.update(q_mass_coi=abs(float(model.mass @ state[:10])),
                     v_mass_coi=abs(float(model.mass @ state[10:20])))
        checks.append(check)
    qualified = all(np.isfinite(list(check.values())).all() and check["identity_error"] <= 1e-8
                    and check["q_mass_coi"] <= 1e-10 and check["v_mass_coi"] <= 1e-10 for check in checks)
    write_json(directory / f"{name}-physics.json", {"points": checks, "qualified": qualified,
                                                  "rk4_updates": int(counts.sum())})
    if not qualified:
        raise FloatingPointError("candidate physical checks failed")
    return path


def _pair(reader, stem):
    return reader.read_json(stem + ".json"), reader.read_arrays(stem + ".npz")


def _energy(model, history, reset, eq):
    if not np.array_equal(history[-1, 20:], reset[20:]):
        raise ValueError("visible reset changed the hidden state")
    start = .5 * float(history[0, 20:] @ history[0, 20:])
    end = .5 * float(reset[20:] @ reset[20:])
    before, after = model.energy(history[-1], eq), model.energy(reset, eq)
    if not np.isfinite([start, end, before, after]).all():
        raise FloatingPointError("nonfinite stored energy")
    return {"hidden_initial": start, "hidden_origin": end,
            "hidden_origin_over_initial": end / start if start else None,
            "before_reset": before, "after_reset": after, "reset_change": after - before}


def _energy_bundle(model, context, step, eq):
    result = {method: _energy(model, context["paths"][step, method], context["resets"][step, method], eq)
              for method in ("E", "I", "C")}
    result["candidate_relative_to"] = {reference: {point: {
        "difference": result["C"][point] - result[reference][point],
        "ratio": result["C"][point] / result[reference][point] if result[reference][point] else None,
    } for point in ("hidden_initial", "hidden_origin")} for reference in ("E", "I")}
    return result


def _freeze(directory, name, records):
    files = sorted(path for path in directory.rglob("*") if path.is_file()
                   and "operations" not in path.relative_to(directory).parts)
    payload = {"records": records, "artifact_hashes": {
        path.relative_to(directory).as_posix(): sha256_file(path) for path in files}}
    write_json(directory / name, payload)
    return sha256_file(directory / name)


def execute(*, root, result_dir, plan, runtime):
    source_plan = json.loads((root / plan["sources_path"]).read_text(encoding="utf-8"))
    ops = Operations(result_dir / "operations", plan)
    reader = EnergyCapSavedInputs(root, source_plan, ops)
    roots = {key: "results/" + value["run_id"] for key, value in source_plan["sources"].items()}
    slots, contexts, models = [], {}, {}
    try:
        write_json(result_dir / "execution-plan.json", plan)
        write_json(result_dir / "source-inventory.json", source_plan)
        with ops.scope("source-root"):
            for key in source_plan["sources"]:
                manifest = reader.read_json(roots[key] + "/artifact_hashes.json")
                prefix = roots[key] + "/"
                for relative, spec in reader.files.items():
                    if (relative.startswith(prefix) and not relative.endswith("/artifact_hashes.json")
                            and manifest.get(relative.removeprefix(prefix)) != spec["sha256"]):
                        raise ValueError("source manifest and new whitelist disagree")
            population = reader.read_json(roots["a14"] + "/allocated_population.json")
            other_population = reader.read_json(roots["a15"] + "/allocated_population.json")
            if any(other_population.get(key) != value for key, value in population.items()):
                raise ValueError("A14 and A15 populations differ")
            if other_population["future_point_counts"] != plan["future_counts"]:
                raise ValueError("frozen future count metadata differs")
            rows = population["cache_rows"]
            weights = np.asarray(population["weights"], dtype=float)
            if (len(rows) != 258 or len(population["allocated_slots"]) != 268 or len(weights) != 258
                    or np.any(weights <= 0) or abs(weights.sum() - 1) > 1e-12):
                raise ValueError("frozen population or weights differ")
            for seed in SEEDS:
                for index, row in enumerate(rows):
                    slots.append({"seed": seed, "index": index, "event_id": row["event_id"],
                                  "status": "not_attempted"})
            write_json(result_dir / "population.json", population)
            _, inputs = _pair(reader, roots["a14"] + "/permitted_inputs")
            scale = inputs["target_scale"]
            if scale.shape != (20,) or np.any(scale <= 0):
                raise ValueError("invalid frozen target scale")
            for seed in SEEDS:
                meta, arrays = _pair(reader, roots["a14"] + f"/static-{seed}")
                if meta["parameters"]["backbone"] != "sine":
                    raise ValueError("baseline potential changed")
                models[seed] = ops.call("probe", lambda arrays=arrays: NonlinearPassiveClosure(
                    **{name: arrays["parameters__" + name] for name in PARAMETERS}, backbone="sine"),
                    tag=f"fixed_model:{seed}")
                if not np.array_equal(models[seed].mass, inputs["mass"]):
                    raise ValueError("source model masses differ")
            save_record(result_dir, "fixed-models", {str(seed): {
                name: getattr(model, name) for name in (*PARAMETERS, "backbone")
            } for seed, model in models.items()})
            save_record(result_dir, "inputs", inputs)

        # Only history payloads are available while constructing every candidate.
        for slot in slots:
            seed, index = slot["seed"], slot["index"]
            key = f"seed-{seed}-event-{index:03d}"
            directory = result_dir / key
            directory.mkdir()
            slot["status"] = "initialization_attempted"
            observed, power, eq = (inputs[name][index] for name in ("history", "constant_power", "equilibrium"))
            context = {"paths": {}, "resets": {}, "energies": {}, "history_metrics": {}}
            with ops.scope(f"init-{seed}-{index:03d}"):
                meta, arrays = _pair(reader, roots["a14"] + f"/{key}/integral")
                old_slot = reader.read_json(roots["a14"] + f"/{key}/slot.json")
                if (old_slot["seed"] != seed or old_slot["cache_index"] != index
                        or old_slot["event_id"] != slot["event_id"]):
                    raise ValueError("source slot identity differs")
                integral = {name: arrays[name] for name in ("Ltilde", "btilde4", "zh4")}
                integral["svd"] = {**{name: arrays["svd__" + name] for name in ("U", "s", "Vh")},
                                   **{name: meta["svd"][name] for name in ("rank", "tol")}}
                expected_tol = 64 * np.finfo(float).eps * 100 * max(arrays["svd__s"][0], meta["svd"]["natural_norm"])
                if meta["svd"]["tol"] != expected_tol:
                    raise ValueError("A14 rank threshold changed")
                for step in STEPS:
                    _, history = _pair(reader, roots["a14"] + f"/{key}/history-{step}")
                    if (not np.array_equal(history["observed"], observed)
                            or not np.array_equal(history["clock"], inputs["history_time"][index])):
                        raise ValueError("saved history observation or clock differs")
                    for method, member, reset in (("E", "baseline", "baseline_reset"), ("I", "candidate", "reset")):
                        path = history[member]
                        if (path.shape != (11, 22) or not np.array_equal(path[0, :20], observed[0])
                                or not np.array_equal(history[reset][:20], observed[-1])
                                or not np.array_equal(path[-1, 20:], history[reset][20:])):
                            raise ValueError("saved history/reset identity differs")
                        context["paths"][step, method], context["resets"][step, method] = path, history[reset]
                    context["clock"] = history["clock"] - history["clock"][0]
                encoder = context["paths"][STEPS[0], "E"][0, 20:]
                if (not np.array_equal(encoder, context["paths"][STEPS[1], "E"][0, 20:])
                        or abs(.5 * float(encoder @ encoder) - old_slot["hidden_energy_encoder"]) > 1e-12):
                    raise ValueError("encoder radius source differs")
                for step in STEPS:
                    if not np.array_equal(context["paths"][step, "I"][0, 20:], integral["zh4"]):
                        raise ValueError("unconstrained history does not start at saved LS state")
                try:
                    solution = ops.call("probe", partial(encoder_energy_cap, integral, encoder), tag="energy_ball")
                except EnergyBallNumericalError as error:
                    ops.partial["solver_failure"] = error.diagnostics
                    raise
                context["solution"] = solution
                save_record(directory, "solver", {"integral": integral, "solution": solution})
                status = solution["candidate"]["status"]
                slot["solver_status"] = status
                if status == "inactive":
                    slot["candidate_source_references"] = {
                        role: {step: {
                            "path": roots[source] + f"/{key}/{role}-{step}.npz",
                            "sha256": reader.files[roots[source] + f"/{key}/{role}-{step}.npz"]["sha256"],
                            "member": "candidate",
                        } for step in STEPS} for role, source in (("history", "a14"), ("future", "a15"))}
                if status == "unresolved":
                    slot["status"] = "unresolved"
                    raise FloatingPointError("positive-radius rank unresolved; no candidate propagation")
                objective = solution["candidate_objective"]
                if objective > solution["encoder_objective"] + 1e-12 * max(1., solution["encoder_objective"]):
                    raise FloatingPointError("ball objective exceeds its feasible encoder comparator")
            for step in STEPS:
                with ops.scope(f"history-{seed}-{index:03d}-{STEP_IDS[step]}"):
                    if status == "inactive":
                        path = context["paths"][step, "I"]
                        reset = context["resets"][step, "I"]
                    else:
                        initial = np.r_[observed[0], solution["candidate"]["z"]]
                        identity = {"seed": seed, "event_id": slot["event_id"], "step": float(step),
                                    "phase": "history_integration"}
                        path = audited_path(models[seed], initial, context["clock"], power, eq, step, ops,
                                            identity, directory, "candidate-history-" + step)
                        reset = np.r_[observed[-1], path[-1, 20:]]
                    context["paths"][step, "C"], context["resets"][step, "C"] = path, reset
                    context["energies"][step] = ops.call("probe", partial(
                        _energy_bundle, models[seed], context, step, eq), tag="reset_energies")
                    context["history_metrics"][step] = ops.call("probe", lambda step=step, context=context, observed=observed: {
                        method: score_history(observed, context["paths"][step, method], scale)
                        for method in ("E", "I", "C")}, tag="history_scores")
            with ops.scope(f"history-review-{seed}-{index:03d}"):
                refinements = ops.call("probe", lambda context=context: {method: path_difference(
                    context["paths"][STEPS[0], method], context["paths"][STEPS[1], method], scale)
                    for method in ("E", "I", "C")}, tag="history_refinement")
                save_record(directory, "history-result", {
                    "energies": context["energies"], "metrics": context["history_metrics"], "refinement": refinements,
                    "resets": {step: {method: context["resets"][step, method] for method in ("E", "I", "C")} for step in STEPS}})
                if not all(value["qualified"] for value in refinements.values()):
                    raise FloatingPointError("history refinement failed")
            slot["status"] = "history_frozen"
            contexts[seed, index] = context
            if (index + 1) % 50 == 0 or index == 257:
                print(f"energy cap: seed={seed} histories={index + 1}/258", flush=True)
        initial_freeze = _freeze(result_dir, "initialization-freeze.json", slots)

        with ops.scope("clock-inputs"):
            _, clock_inputs = _pair(reader, roots["a15"] + "/permitted_truth")
            offsets = clock_inputs["offsets"]
            counts = np.diff(offsets) - 1
            if (offsets.shape != (259,) or offsets[0] != 0 or offsets[-1] != 72405
                    or not np.array_equal(counts, plan["future_counts"])):
                raise ValueError("frozen future lengths differ")
            clocks = [clock_inputs["forecast_time"][offsets[i]:offsets[i + 1]] for i in range(258)]
            for count, clock in zip(counts, clocks, strict=True):
                if not np.array_equal(clock, np.arange(int(count) + 1) * .01):
                    raise ValueError("saved future clock differs from the frozen clock construction")
            save_record(result_dir, "clock-inputs", clock_inputs)

        for slot in slots:
            seed, index = slot["seed"], slot["index"]
            context = contexts[seed, index]
            context["futures"] = {}
            slot["status"] = "future_attempted"
            if slot["solver_status"] != "inactive":
                for step in STEPS:
                    with ops.scope(f"future-{seed}-{index:03d}-{STEP_IDS[step]}"):
                        identity = {"seed": seed, "event_id": slot["event_id"], "step": float(step),
                                    "phase": "future_integration"}
                        context["futures"][step] = audited_path(
                            models[seed], context["resets"][step, "C"], clocks[index],
                            inputs["constant_power"][index], inputs["equilibrium"][index], step, ops,
                            identity, result_dir / f"seed-{seed}-event-{index:03d}", "candidate-future-" + step)
            slot["status"] = "future_frozen"
            if (index + 1) % 50 == 0 or index == 257:
                print(f"energy cap: seed={seed} futures={index + 1}/258", flush=True)
        future_freeze = _freeze(result_dir, "future-freeze.json", slots)

        grouped = {str(seed): {step: {method: [] for method in ("E", "I", "C")} for step in STEPS} for seed in SEEDS}
        valid = True
        for slot in slots:
            seed, index = slot["seed"], slot["index"]
            key, context = f"seed-{seed}-event-{index:03d}", contexts[seed, index]
            slot["status"] = "scoring_attempted"
            with ops.scope(f"score-{seed}-{index:03d}"):
                saved = {}
                for step in STEPS:
                    _, saved[step] = _pair(reader, roots["a15"] + f"/{key}/future-{step}")
                    record = saved[step]
                    if (not np.array_equal(record["clock"], clocks[index])
                            or not np.array_equal(record["reset"], context["resets"][step, "I"])
                            or not np.array_equal(record["truth"][0], inputs["history"][index, -1])):
                        raise ValueError("saved future identity differs")
                    for method, member in (("E", "baseline"), ("I", "candidate")):
                        if not np.array_equal(record[member][0], context["resets"][step, method]):
                            raise ValueError("saved future origin differs")
                if not np.array_equal(saved[STEPS[0]]["truth"], saved[STEPS[1]]["truth"]):
                    raise ValueError("paired future targets differ")
                scores = {}
                for method, member in (("E", "baseline"), ("I", "candidate"), ("C", "candidate")):
                    paths = [context["futures"][step] if method == "C" and slot["solver_status"] != "inactive"
                             else saved[step][member] for step in STEPS]
                    scores[method] = ops.call("probe", partial(score_pair,
                        history=inputs["history"][index], truth=saved[STEPS[0]]["truth"], coarse=paths[0], fine=paths[1],
                        equilibrium=inputs["equilibrium"][index], mass=models[seed].mass,
                        omega_s=float(clock_inputs["omega_s"][index]), scale=scale, clock=clocks[index]), tag="paired_scores:" + method)
                    valid = valid and scores[method]["refinement"]["qualified"]
                    for step in STEPS:
                        grouped[str(seed)][step][method].append(scores[method]["steps"][step])
                save_record(result_dir / key, "scores", scores)
                if not all(score["refinement"]["qualified"] for score in scores.values()):
                    raise FloatingPointError("future or rule refinement failed")
                slot["status"] = "complete"
            if (index + 1) % 50 == 0 or index == 257:
                print(f"energy cap: seed={seed} scored={index + 1}/258", flush=True)

        with ops.scope("summary"):
            groups = ops.call("probe", lambda: {seed: {step: {method: aggregate_group(records, weights, counts,
                      event_ids=[row["event_id"] for row in rows]) for method, records in methods.items()}
                      for step, methods in steps.items()} for seed, steps in grouped.items()}, tag="all_group_summaries")
            active = {str(seed): [slot["index"] for slot in slots if slot["seed"] == seed
                      and slot["solver_status"] != "inactive"] for seed in SEEDS}
            summary = ops.call("probe", lambda: verdict(groups, valid=valid, mechanism_exercised=any(active.values())), tag="verdict")
            summary.update(groups=groups, active_indices=active,
                           active_weight_mass={seed: float(weights[indices].sum()) for seed, indices in active.items()},
                           initialization_freeze_sha256=initial_freeze, future_freeze_sha256=future_freeze,
                           runtime={"python": sys.version, "executable": sys.executable, "packages": runtime},
                           source_commit=git_head(root), slots=slots)
            summary["history"] = ops.call("probe", lambda: {str(seed): {step: {method: {
                "weighted_mse": float(weights @ np.array([contexts[seed, i]["history_metrics"][step][method]["all"] for i in range(258)])),
                "q_mse": float(weights @ np.array([contexts[seed, i]["history_metrics"][step][method]["q"] for i in range(258)])),
                "v_mse": float(weights @ np.array([contexts[seed, i]["history_metrics"][step][method]["v"] for i in range(258)])),
                "cvar10": weighted_cvar([contexts[seed, i]["history_metrics"][step][method]["all"] for i in range(258)], weights),
            } for method in ("E", "I", "C")} for step in STEPS} for seed in SEEDS}, tag="all_history_summaries")
            if reader.attempted != set(reader.files):
                raise ValueError("source read plan was not exactly consumed")
        summary["operations"] = {"attempted": dict(ops.counts), "returned": dict(ops.returns)}
        active_count = sum(len(indices) for indices in active.values())
        active_future_points = sum(int(counts[i]) for indices in active.values() for i in indices)
        expected = {"rollouts": 4 * active_count, "integration_rhs": 240 * active_count + 24 * active_future_points,
                    "physical_rhs": 24 * active_count + 2 * active_future_points}
        if any(ops.counts[key] != value or ops.returns[key] != value for key, value in expected.items()):
            raise RuntimeError("actual candidate call tree differs from the frozen branch budget")
        summary["conditional_budget"] = {"active_slots": active_count, "active_future_points": active_future_points,
                                          "expected_calls": expected,
            "solver_updates": sum(context["solution"]["candidate"]["iterations"] for context in contexts.values()),
            "solver_evaluations": sum(context["solution"]["candidate"]["secular_evaluations"] for context in contexts.values())}
        positive_active = sum(slot["solver_status"] == "active" for slot in slots)
        summary["conditional_budget"]["positive_multiplier_slots"] = positive_active
        if (summary["conditional_budget"]["solver_updates"] > 64 * positive_active
                or summary["conditional_budget"]["solver_evaluations"] > 80 * positive_active
                or ops.counts["probe"] != plan["operator_limits"]["probe"]
                or ops.returns != ops.counts
                or len(reader.decoded) != sum(len(spec.get("members", {})) for spec in reader.files.values())
                or reader.bytes_read != sum(spec["size_bytes"] for spec in reader.files.values())
                or reader.logical_bytes != plan["source_read_plan"]["decoded_array_byte_cap"]
                or reader.npy_header_bytes > len(reader.decoded) * plan["source_read_plan"]["npy_header_max_bytes_per_member"]):
            raise RuntimeError("actual scalar, source or probe accounting differs from the frozen plan")
        write_json(result_dir / "source-access.json", reader.snapshot())
        write_json(result_dir / "summary.json", summary)
        write_json(result_dir / "stage_evidence.json", {
            "status": "PASS", "process_status": "PASS", "source_commit": git_head(root),
            "data_policy": "frozen_saved_train_only", "slot_count": 774,
            "scientific_verdict": summary["scientific_verdict"], "promotion_decision": "KEEP_BASELINE",
            "claim_boundary": "Fixed A06 encoder-referenced energy-ball train mechanism; no certification, promotion, val or test evidence.",
        })
        ops.journal.close()
        write_json(result_dir / "artifact_hashes.json", {path.relative_to(result_dir).as_posix(): sha256_file(path)
                   for path in sorted(result_dir.rglob("*")) if path.is_file()})
        return summary
    except BaseException as error:
        write_json(result_dir / "failure.json", {"error_type": type(error).__name__, "message": str(error),
            "slots": slots, "operations": dict(ops.counts), "source_access": reader.snapshot(), "retry_allowed": False})
        ops.journal.close()
        raise
