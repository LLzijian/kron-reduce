"""Decompose frozen score JSON without fitting, forecasting or raw-data access."""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from collections import Counter
from pathlib import Path

from project_harness.evidence import sha256_file, verify_artifact_directory

METRICS = ("normalized_mse", "delta_mse_rad2", "omega_mse_rad2_per_s2")
IDENTITY = ("source_dataset", "fault_line", "role")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def weights(rows):
    if not rows or len({r["event_id"] for r in rows}) != len(rows):
        raise ValueError("population must be nonempty with unique event IDs")
    if any(not isinstance(r.get(k), str) or not r[k] for r in rows for k in ("event_id", *IDENTITY)):
        raise ValueError("missing population identity")
    sources = {r["source_dataset"] for r in rows}
    lines = {s: {r["fault_line"] for r in rows if r["source_dataset"] == s} for s in sources}
    counts = Counter((r["source_dataset"], r["fault_line"]) for r in rows)
    return [1 / len(sources) / len(lines[r["source_dataset"]]) /
            counts[(r["source_dataset"], r["fault_line"])] for r in rows]


def number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def signature(rows):
    return {r["event_id"]: tuple(r[k] for k in (*IDENTITY, "actual_duration_s")) for r in rows}


def validate(rows, labels):
    w = weights(rows)
    if len({r["role"] for r in rows}) != 1 or rows[0]["role"] not in ("train", "val"):
        raise ValueError("audit accepts one internal train/val role")
    if set(labels) != {r["event_id"] for r in rows} or any(v not in ("stable", "unstable") for v in labels.values()):
        raise ValueError("unknown or mismatched coverage labels")
    if any(r.get("status") != "ok" or not all(number(r.get(k)) for k in (*METRICS, "actual_duration_s")) for r in rows):
        raise ValueError("failed, missing or nonfinite scores cannot be imputed")
    return w


def partitions(rows, labels):
    return {
        "label": [labels[r["event_id"]] for r in rows],
        "source_dataset": [r["source_dataset"] for r in rows],
        "fault_line": [r["fault_line"] for r in rows],
        "duration": ["less_than_1s" if r["actual_duration_s"] < 1-1e-10 else
                     "1_to_less_than_3s" if r["actual_duration_s"] < 3-1e-10 else
                     "complete_3s" for r in rows],
    }


def audit_rows(rows, labels, stored_rmse):
    """Group contributions retain the original population denominator."""
    w = validate(rows, labels)
    totals = {k: math.fsum(a*r[k] for a, r in zip(w, rows, strict=True)) for k in METRICS}
    rmse = math.sqrt(totals["normalized_mse"])
    if not number(stored_rmse) or abs(rmse-stored_rmse) > 1e-12:
        raise ValueError("reconstructed RMSE differs from stored report")
    groups = {}
    for field, values in partitions(rows, labels).items():
        groups[field] = {}
        for name in sorted(set(values)):
            ix = [i for i, v in enumerate(values) if v == name]
            mass = math.fsum(w[i] for i in ix)
            c = {k: math.fsum(w[i]*rows[i][k] for i in ix) for k in METRICS}
            groups[field][name] = {
                "event_count": len(ix), "weight_mass": mass,
                "same_denominator_mse_contribution": c["normalized_mse"],
                "squared_error_share": c["normalized_mse"]/totals["normalized_mse"] if totals["normalized_mse"] else None,
                "conditional_nrmse": math.sqrt(c["normalized_mse"]/mass),
                "physical": {k: {"same_denominator_mse_contribution": c[k],
                                 "conditional_rmse": math.sqrt(c[k]/mass)} for k in METRICS[1:]},
            }
        if abs(math.fsum(g["same_denominator_mse_contribution"] for g in groups[field].values())-totals["normalized_mse"]) > 1e-12:
            raise ValueError("group contributions do not reconstruct total MSE")
    return {"event_count": len(rows), "normalized_mse": totals["normalized_mse"],
            "normalized_rmse": rmse, "groups": groups,
            "physical": {k: {"weighted_mse": totals[k], "rmse": math.sqrt(totals[k])} for k in METRICS[1:]},
            "physical_metrics_are_not_normalized_channel_contributions": True}


def join_horizons(score, labels):
    section = score["available"]
    h1, h3 = [section["horizons"][h] for h in ("1.0", "3.0")]
    for horizon in (h1, h3):
        audit_rows(horizon["rows"], labels, horizon["normalized_rmse"])
    a, b = h1["rows"], h3["rows"]
    if {r["event_id"] for r in a} != {r["event_id"] for r in b}:
        raise ValueError("horizon populations differ")
    bmap = {r["event_id"]: r for r in b}
    joined = []
    for first in a:
        second = bmap[first["event_id"]]
        if any(first[k] != second[k] for k in IDENTITY) or abs(first["actual_duration_s"]-min(1., second["actual_duration_s"])) > 1e-10:
            raise ValueError("horizon identity or real duration differs")
        joined.append({**second, **{k: .5*(first[k]+second[k]) for k in METRICS}})
    result = audit_rows(joined, labels, section["multi_horizon_normalized_rmse"])
    result["duration_groups_use"] = "longest_real_available_future_capped_at_3s"
    result["physical_metrics_average"] = "equal_1s_and_3s_real_available_horizon_MSE"
    return joined, result


def paired_contributions(candidate, baseline, labels):
    if signature(candidate) != signature(baseline):
        raise ValueError("paired populations or clocks differ")
    w = validate(candidate, labels)
    validate(baseline, labels)
    reference = {r["event_id"]: r for r in baseline}
    delta = [r["normalized_mse"]-reference[r["event_id"]]["normalized_mse"] for r in candidate]
    groups = {field: {name: math.fsum(w[i]*delta[i] for i, v in enumerate(values) if v == name)
                      for name in sorted(set(values))} for field, values in partitions(candidate, labels).items()}
    return {"reference": "same_seed_A", "metric": "candidate_minus_A_main_normalized_MSE",
            "difference": math.fsum(a*d for a, d in zip(w, delta, strict=True)),
            "same_denominator_group_differences": groups}


def run(input_run, run_id, output_root, expected_manifest_sha256=None):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("unsafe immutable audit ID")
    input_run, out = Path(input_run).resolve(), (Path(output_root)/run_id).resolve()
    if out.is_relative_to(input_run) or input_run.is_relative_to(out):
        raise ValueError("audit output must be separate from source run")
    manifest = verify_artifact_directory(input_run, strict=True, expected_manifest_sha256=expected_manifest_sha256)
    manifest_hash = sha256_file(input_run/"artifact_hashes.json")
    used = {}

    def read(relative):
        if relative not in manifest or not relative.endswith(".json"):
            raise ValueError("only manifest-bound JSON may be opened")
        used[relative] = manifest[relative]
        return read_json(input_run/relative)

    coverage = read("coverage.json")["rows"]
    if len({r["event_id"] for r in coverage}) != len(coverage):
        raise ValueError("duplicate coverage identity")
    coverage = {r["event_id"]: r for r in coverage}
    plan = read("scientific_plan.json")
    caches, results, signatures = {}, [], {}
    for candidate in "ABCD":
        for seed in (324, 325, 326):
            model = f"{candidate}-{seed}"
            trace = read(f"candidate-{model}/fit_record.json")["fit_record"]["loss_trace"]
            for role in ("train", "val"):
                score = read(f"candidate-{model}/{role}_scores.json")
                raw = score["available"]["horizons"]["3.0"]["rows"]
                if any(r["event_id"] not in coverage or any(coverage[r["event_id"]][k] != r[k] for k in IDENTITY) for r in raw):
                    raise ValueError("score identity differs from frozen coverage")
                eligible = {k for k, r in coverage.items() if r["role"] == role and r["history_eligible"]}
                if {r["event_id"] for r in raw} != eligible:
                    raise ValueError("eligible coverage differs from scores")
                labels = {r["event_id"]: coverage[r["event_id"]]["label"] for r in raw}
                joined, summary = join_horizons(score, labels)
                sig = signature(joined)
                if role in signatures and signatures[role] != sig:
                    raise ValueError("models do not share one role population and clock")
                signatures[role] = sig
                caches[(candidate, seed, role)] = joined
                summary.update(candidate=candidate, seed=seed, role=role,
                    horizons={h: audit_rows(score["available"]["horizons"][h]["rows"], labels, score["available"]["horizons"][h]["normalized_rmse"]) for h in ("1.0", "3.0")},
                    paired_against_A=paired_contributions(joined, caches[("A", seed, role)], labels),
                    training_loss_trace=trace,
                    loss_trace_timing="before_each_logged_update; not final_checkpoint_score",
                    settings=plan["candidate_settings"][model])
                results.append(summary)
    if sha256_file(input_run/"artifact_hashes.json") != manifest_hash or any(sha256_file(input_run/p) != h for p, h in used.items()):
        raise ValueError("source records changed during audit")
    out.mkdir(parents=True, exist_ok=False)
    _write_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    write_json(out/"audit_summary.json", {
        "schema": "recorded_error_audit.v1", "source_run": input_run.name,
        "source_manifest_sha256": manifest_hash, "used_source_json_sha256": used,
        "source_commit": _write_commit, "audit_code_sha256": sha256_file(Path(__file__)),
        "results": results, "selection_unchanged": True, "predictions_computed": 0,
        "optimizer_updates": 0, "raw_trajectory_payloads_opened": 0,
        "source_val_test_payloads_opened": 0, "evidence_kind": "posthoc_audit_of_frozen_records",
        "interpretation": "Descriptive decomposition; labels are reporting strata, never new selection criteria."})
    write_json(out/"artifact_hashes.json", {"audit_summary.json": sha256_file(out/"audit_summary.json")})
    verify_artifact_directory(out, strict=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-run", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256")
    args = parser.parse_args()
    print(run(args.input_run, args.run_id, args.output_root, args.expected_manifest_sha256))


if __name__ == "__main__":
    main()
