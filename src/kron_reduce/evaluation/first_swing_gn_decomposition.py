"""Paired GN initialization comparisons for one frozen TASK3 equation."""

from __future__ import annotations

import time
from collections import Counter

import numpy as np
import torch

from kron_reduce.evaluation.constant_c_observability import write_json
from kron_reduce.evaluation.first_swing_loss import observed_velocity_increment
from kron_reduce.evaluation.first_swing_probes import _history_stats, _label, speed_peak_gap
from kron_reduce.evaluation.kiki_first_swing import STEPS, _witnesses
from kron_reduce.evaluation.kiki_machine_local_full import evaluate, inputs
from kron_reduce.evaluation.kiki_topology_momentum import ORIGIN_FRAME
from kron_reduce.evaluation.machine_local_clean_learning import NUMERICAL_ERRORS
from kron_reduce.models.first_swing_gn_grid import predict_gn_grid

CONTROLS = ("fixed", "adaptive")


@torch.no_grad()
def _primary_residual(model, events, scale, budget, reference):
    q, v, power, equilibrium, scale = map(model._tensor, inputs(events, scale))
    initial = torch.cat((q[:, 0], v[:, 0], q.new_zeros((len(q), model.latent_dimension))), dim=1)
    history = model.rollout(initial, power, equilibrium, 10, max_step=STEPS[0], spend=budget.spend)
    truth = torch.cat((q[:, 1:], v[:, 1:]), dim=2)
    r0 = ((truth-history[:, 1:, :truth.shape[-1]])/scale).square().mean(dim=(1, 2)).sqrt().cpu().numpy()
    saved = [reference[event["id"]]["zero_history_nrmse"] for event in events]
    if not np.allclose(r0, saved, rtol=1e-10, atol=1e-12):
        raise RuntimeError("primary zero-memory residual differs from the archived causal history")
    return r0


def _forecast(model, events, directory, budget, scales, settings, modes, r0, batch_size):
    role, n, d = events[0]["role"], model.n_nodes, model.latent_dimension
    batch_shape = (len(modes), len(events))
    data, failures = {}, []
    for step in STEPS:
        paths = np.lib.format.open_memmap(directory/f"{role}-predictions-{step*1000:g}ms.npy", mode="w+",
                                          dtype=np.float64, shape=(*batch_shape, 801, 2*n))
        paths.fill(np.nan)
        data[step] = {"forecast": paths, "history": np.full((*batch_shape, 11, 2*n+d), np.nan),
                      "hidden_start": np.full((*batch_shape, d), np.nan),
                      "singular": np.full((*batch_shape, d), np.nan),
                      "prior_weight": np.full(batch_shape, np.nan), "radius_limit": np.full(batch_shape, np.nan),
                      "soft_active": np.zeros(batch_shape, dtype=bool), "zero_r0": np.full(batch_shape, np.nan)}

    def forward(step, indices, start, stop):
        result = predict_gn_grid(model, *inputs(events[start:stop], scales["history"]),
                                 max_step=step, spend=budget.spend, memory_radius=scales["memory_radius"],
                                 prior_weight=settings["prior_weight"], modes=modes[indices], primary_r0=r0[start:stop])
        residuals = result.pop("zero_history_nrmse").cpu().numpy()
        if step == STEPS[0] and not np.allclose(residuals, r0[start:stop], rtol=1e-10, atol=1e-12):
            raise RuntimeError("primary zero-memory residual changed within the paired prediction")
        for key, tensor in result.items():
            data[step][key][indices, start:stop] = tensor.cpu().numpy()
        data[step]["zero_r0"][indices, start:stop] = residuals

    for start in range(0, len(events), batch_size):
        stop = min(start+batch_size, len(events))
        for step in STEPS:
            try:
                forward(step, slice(None), start, stop)
            except NUMERICAL_ERRORS as error:
                failures.append({"step": step, "start": start, "stop": stop, "error": str(error)})
                for i in range(start, stop):
                    for j, mode in enumerate(modes):
                        try:
                            forward(step, slice(j, j+1), i, i+1)
                        except NUMERICAL_ERRORS as individual:
                            failures.append({"step": step, "id": events[i]["id"], "mode": mode["name"],
                                             "error": str(individual)})
        print(f"GN {role}: {stop}/{len(events)} for {len(modes)} modes at both steps", flush=True)
    for group in data.values():
        group["forecast"].flush()
    write_json(directory/f"{role}-forecast-failures.json", failures)
    return data


def _scores(model, events, data, r0, scales, budget, directory, modes):
    role = events[0]["role"]
    score = {}
    cases = {event["id"]: i for i, event in enumerate(events)}
    for mode_index, mode in enumerate(modes):
        def archived_predict(_batch, step, mode_index=mode_index):
            group = data[step]
            return {key: torch.from_numpy(group[key][mode_index]) for key in
                    ("forecast", "history", "hidden_start")} | {
                    "gn_singular": torch.from_numpy(group["singular"][mode_index:mode_index+1])}

        # Keep the frozen TASK3 scorer, including its two-step classification checks.
        result = evaluate(model, events, scales["history"], budget, directory, role,
                          {"eval_batch": len(events)}, predictor=archived_predict, save=False)
        for row in result["initialization"]:
            i = cases[row["id"]]
            step = row["step"]
            row["primary_r0"] = float(r0[i])
            group = data[step]
            for key in ("prior_weight", "radius_limit", "zero_r0", "soft_active"):
                row[key] = group[key][mode_index, i].item()
            for key, value in row.items():
                if isinstance(value, float) and not np.isfinite(value):
                    row[key] = None
            if row["radius_limit"] is None:
                row["soft_active"] = None
        result["initialization"] = [row for row in result["initialization"] if row["history_nrmse"] is not None]
        score[mode["name"]] = result
        write_json(directory/f"{role}-{mode['name']}.json", result)
    return score


def _distribution(values, expected):
    finite = np.asarray([x for x in values if x is not None and np.isfinite(x)])
    return {"qualified": len(finite), "allocated": expected} | {
        key: float(np.median(samples)) if len(finite) == expected else None for key, samples in
        (("signed_median", finite), ("absolute_median", np.abs(finite)), ("positive_median", finite.clip(min=0.)))}


@torch.no_grad()
def _witness_audit(model, events, data, budget, names, archived, missing_before):
    windows, missing = _witnesses(events)
    if missing != missing_before:
        raise RuntimeError("the two unavailable train witness cases changed")
    count = len(windows)
    indices = {e["id"]: i for i, e in enumerate(events)}
    rows = [indices[e["id"]] for e, _ in windows]
    lengths = np.array([w["witness_end"]-ORIGIN_FRAME for _, w in windows])
    clips = [e["x"][ORIGIN_FRAME:w["witness_end"]+1] for e, w in windows]
    path = np.stack([np.pad(clip, ((0, int(lengths.max())+1-len(clip)), (0, 0)), mode="edge") for clip in clips])
    _, _, power, equilibrium, _ = inputs([e for e, _ in windows], np.ones(20))
    output = {}
    for step in STEPS:
        hidden = data[step]["history"][:, rows, -1, 20:]
        differences = np.full((len(names), count), np.nan)

        def score(indices, hidden=hidden, step=step, differences=differences):
            mode_indices, case_indices = np.nonzero(np.isfinite(hidden[indices]).all(axis=-1))
            if not len(case_indices):
                return
            mode_indices = mode_indices+indices.start
            observed = model._tensor(path[case_indices])
            increment = observed_velocity_increment(model, observed, model._tensor(hidden[mode_indices, case_indices]),
                model._tensor(power[case_indices]), model._tensor(equilibrium[case_indices]), lengths[case_indices],
                step=step, spend=budget.spend)
            residual = (observed[..., 10:]-observed[:, :1, 10:]-increment).cpu().numpy()
            for i, (mode, case) in enumerate(zip(mode_indices, case_indices, strict=True)):
                high, low = windows[case][1]["machines"]
                differences[mode, case] = residual[i, lengths[case], high]-residual[i, lengths[case], low]

        failures = []
        try:
            score(slice(0, len(names)))
        except NUMERICAL_ERRORS as error:
            failures.append({"error": str(error)})
            for j, name in enumerate(names):
                try:
                    score(slice(j, j+1))
                except NUMERICAL_ERRORS as individual:
                    failures.append({"mode": name, "error": str(individual)})
        pairs = archived[str(step)]["rows"]
        for (event, window), old in zip(windows, pairs, strict=True):
            if event["id"] != old["id"] or any(old[key] != value for key, value in window.items()):
                raise RuntimeError("train witness identity, machines or window changed")
        cases = [old | {name: float(value) if np.isfinite(value) else None for name, value in
                        zip(names, differences[:, i], strict=True)} for i, old in enumerate(pairs)]
        output[str(step)] = {"rows": cases, "failures": failures, "summary": {
            name: _distribution([row[name] for row in cases], count) for name in (*CONTROLS, *names)}}
    return output | {"without_witness": missing}


def _peaks(events, data, names, archived):
    unstable = {event["id"]: (i, event) for i, event in enumerate(events) if event["label"] == "unstable"}
    if len(unstable) != 12 or set(unstable) != set(archived):
        raise RuntimeError("the frozen val speed-peak roster changed")
    cases = {key: {str(step): archived[key][str(step)] | {name: speed_peak_gap(event, data[step]["forecast"][j, i])
            for j, name in enumerate(names)} for step in STEPS} for key, (i, event) in unstable.items()}
    summary = {}
    for step in STEPS:
        summary[str(step)] = {}
        for name in (*CONTROLS, *names):
            rows = [row[str(step)][name] for row in cases.values()]
            summary[str(step)][name] = _distribution([r["gap_rad_s"] if r else None for r in rows], len(unstable))
    return {"rows": cases, "summary": summary}


def _case_table(events, scores, old_residuals):
    stability = {name: result["stability_events"] for name, result in scores.items()}
    events_by_name = {name: {row["id"]: row for row in rows} for name, rows in stability.items()}
    if any([(e["id"], e["label"]) for e in events] != [(r["id"], r["truth_label"]) for r in rows]
           for rows in stability.values()):
        raise RuntimeError("unpaired event ids, order or frozen truth labels")
    cases = {e["id"]: {"truth": e["label"]} for e in events}
    for name, scores_by_mode in scores.items():
        records = {row["event_id"]: row for row in scores_by_mode["records"]
                   if (row["horizon"], row["label"]) in {("8", "stable"), ("prefix", "unstable")}}
        history = {step: {row["id"]: row for row in scores_by_mode["initialization"] if row["step"] == step}
                   for step in STEPS}
        for event in events:
            key = event["id"]
            record = records.get(key, {})
            qualified = record.get("qualified", False)
            cases[key][name] = {"status": _label(events_by_name[name][key]),
                                "nrmse": record.get("rmse") if qualified else None,
                                "scored_seconds": record["n"]/1000. if record else None, "qualified": qualified,
                                "initialization": {str(step): history[step].get(key) for step in STEPS}}
            cases[key]["primary_r0"] = old_residuals[key]["zero_history_nrmse"]
    return cases


def _gate(scores, witness, peaks, settings, name):
    val = scores["val"][name]
    valid = all(result[name]["task3_six_term_score"] is not None
                and result[name]["stability"]["numerical_failures"] == 0
                and result[name]["stability"]["uncovered"] == 0
                and result[name]["stable_complete"]["8"]["qualified"] == stable_cases
                for result, stable_cases in zip((scores["train"], scores["val"]), (311, 95), strict=True))
    witness_ok, peak_ok = [], []
    for step in STEPS:
        w, p = witness[str(step)]["summary"], peaks["summary"][str(step)]
        w0, w1 = w["fixed"], w[name]
        p0, p1 = p["fixed"]["positive_median"], p[name]["positive_median"]
        witness_ok.append(w1["signed_median"] is not None and abs(w1["signed_median"])
                          <= settings["relative_limit"]*abs(w0["signed_median"])
                          and w1["absolute_median"] <= w0["absolute_median"])
        peak_ok.append(p1 is not None and p1 <= settings["relative_limit"]*p0)
        valid = valid and w1["qualified"] == 33 and p[name]["qualified"] == 12
    stability = val["stability"]
    nrmse = val["stable_complete"]["8"].get("mean")
    limits = {"qualification": bool(valid), "recall_or_peak": stability["tp"] == 12
              or (stability["tp"] == 11 and all(peak_ok)), "paired_witnesses": all(witness_ok),
              "stable_8s_mean": nrmse is not None and nrmse <= settings["stable_8s_mean_max"],
              "false_alarms": stability["fp"] <= settings["false_alarms_max"]}
    return {"pass": all(limits.values()), "criteria": limits}


def _frontier(scores, witness, peaks, gates):
    vectors = {}
    for name, gate in gates.items():
        if not gate["criteria"]["qualification"]:
            continue
        val = scores["val"][name]
        s = val["stability"]
        vectors[name] = (-s["tp"], s["fp"], s["unsettled_predictions"], val["stable_complete"]["8"]["mean"],
            max(peaks["summary"][str(step)][name]["positive_median"] for step in STEPS),
            max(abs(witness[str(step)]["summary"][name]["signed_median"]) for step in STEPS))
    return [name for name, vector in vectors.items() if not any(other != vector and
            all(a <= b for a, b in zip(other, vector, strict=True)) for other in vectors.values())]


def execute(model, reader, directory, budget, scales, plan, baseline):
    started = time.perf_counter()
    train = reader.load("train")
    if not np.allclose(reader.static["mass"], model.mass.cpu().numpy(), rtol=0., atol=1e-12):
        raise RuntimeError("source inertia differs from the frozen equation")
    old_residuals = baseline["residuals"][str(STEPS[0])]
    primary = {"train": _primary_residual(model, train, scales["history"], budget, old_residuals)}
    stable = [value for event, value in zip(train, primary["train"], strict=True) if event["label"] == "stable"]
    if len(stable) != 311:
        raise RuntimeError("the frozen stable train population changed")
    reference = baseline["reference"]
    if not np.isclose(np.median(primary["train"]), reference, rtol=1e-10, atol=1e-12):
        raise RuntimeError("archived v2 reference is inconsistent with the causal train residual")
    thresholds = {str(q): float(np.quantile(stable, q/100., method="linear")) for q in plan["quantiles"]}
    modes = [{**mode, "reference": thresholds[str(mode["quantile"])] if "quantile" in mode else reference}
             for mode in plan["mode"]]
    names = [mode["name"] for mode in modes]
    settings = baseline["freeze"]["settings"]
    freeze_path = directory/"freeze.json"
    write_json(freeze_path, baseline["freeze"] | {"zero_updates": True, "primary_step_s": STEPS[0],
               "thresholds_from_stable_train": thresholds, "quadratic_reference": reference,
               "forecast_mode_order": names, "modes": modes, "gates": plan["gates"],
               "upstream_runs": baseline["run_ids"]})
    val = reader.load("val", freeze_path=freeze_path)
    if Counter(e["label"] for e in val) != {"stable": 95, "unstable": 12}:
        raise RuntimeError("the frozen val class sizes changed")
    primary["val"] = _primary_residual(model, val, scales["history"], budget, old_residuals)
    scores, cases = {}, {}
    for role, events in (("train", train), ("val", val)):
        data = _forecast(model, events, directory, budget, scales, settings, modes, primary[role], plan["batch_size"])
        scores[role] = baseline["scores"][role] | _scores(model, events, data, primary[role], scales, budget, directory, modes)
        cases[role] = _case_table(events, scores[role], old_residuals)
        if role == "train":
            witness = _witness_audit(model, events, data, budget, names, baseline["witness"], baseline["without_witness"])
        else:
            peaks = _peaks(events, data, names, baseline["peaks"])
        del data
    targeted = {key: row for key, row in cases["val"].items() if row["fixed"]["status"] == "stable"
                and row["adaptive"]["status"] in ({"unstable", "undecided"} if row["truth"] == "stable" else {"unstable"})}
    if len(targeted) != 4:
        raise RuntimeError("the four archived val diagnostic cases changed")
    gates = {name: _gate(scores, witness, peaks, plan["gates"], name) for name in (*CONTROLS, *names)}
    write_json(directory/"classification-pairs.json", cases)
    write_json(directory/"train-witnesses.json", witness)
    write_json(directory/"val-speed-peaks.json", peaks)
    write_json(directory/"val-targeted.json", targeted)
    keys = ("stability", "stable_complete", "unstable_prefix", "task3_six_term_score")
    return {"modes": modes, "train_stable_quantiles": thresholds, "quadratic_reference": reference,
            "scores": {role: {name: {key: result[key] for key in keys} | {"history": _history_stats(result)}
                              for name, result in results.items()} for role, results in scores.items()},
            "witness": {str(step): witness[str(step)]["summary"] for step in STEPS}, "peaks": peaks["summary"],
            "gates": gates, "frontier": _frontier(scores, witness, peaks, gates),
            "passing_new_modes": [name for name in names if gates[name]["pass"]],
            "wall_seconds": time.perf_counter()-started}
