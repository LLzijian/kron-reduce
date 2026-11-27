"""Join the bound TASK3 reporting tables by roster, condition and method."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from kron_reduce.evaluation.constant_c_observability import write_json
from project_harness.evidence import sha256_file
from project_harness.manifest import project_root
from scripts.export_noise_robustness_v4 import STATISTICS, metric_rows

ROSTER_COUNTS = {"val": 107, "test": 76, "blind_v2": 94}
PEER_LABELS = {
    "graph_gru_extension1": "Graph-GRU", "gru_seq2seq_extension1": "GRU seq2seq",
    "latent_neural_ode_extension1": "Latent neural ODE", "small_transformer": "Small Transformer",
    "koopman_shred_faithful": "Koopman-SHRED", "history_deeponet_p0": "History DeepONet P0",
    "lass_small_p0": "LASS P0", "stan_rotor_p0": "STAN Rotor P0", "actnet_history_p0": "ActNet History P0",
    "sindy_shred_authorized_retry": "SINDy-SHRED", "hankel_dmd": "Hankel DMD",
    "i_ikno_p0_inferencefix": "i-IKNO P0", "graph_gru_extension1_causal_sg": "Graph-GRU + SG",
    "gru_seq2seq_extension1_causal_sg": "GRU seq2seq + SG", "gru_classifier": "GRU classifier",
    "extra_trees": "Extra Trees", "pcdk_surrogate_diagnostic": "PCDK diagnostic",
    "gru_classifier_causal_sg": "GRU classifier + SG",
}
PROFILE_FIELDS = {
    "accuracy_when_all_decided": "accuracy", "classifiable_accuracy": "accuracy_on_resolved",
    "classifiable_coverage": "coverage", "stable_false_alarm_rate": "false_alarm_rate",
    "full_roster_accuracy": "accuracy_counting_unresolved_as_incorrect",
}
UNAVAILABLE_PEER_FIELDS = (
    "uncovered", "positive_peak_deficit_rad_s", "exact_cut_rate", "cut_jaccard_mean", "mechanical_energy_auroc",
    "memory_energy_auroc", "total_energy_auroc",
)


def _read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _number(token):
    if token in ("", "null"):
        return None
    value = float(token)
    if not math.isfinite(value):
        raise ValueError(f"unsupported nonfinite peer summary value: {token}")
    return value


def _stats(row, metric, repeats):
    available = int(row[f"{metric}_valid_reps"])
    if available < 0 or available > repeats:
        raise ValueError(f"invalid repetition coverage for {metric}")
    mean = _number(row[f"{metric}_mean"])
    std = _number(row[f"{metric}_std"])
    worst = _number(row[f"{metric}_worst"])
    worst_index = _number(row[f"{metric}_worst_rep"])
    complete = available == repeats
    return {"mean": mean if complete else None, "sample_std": std if complete else None,
            "replicates": repeats, "available_replicates": available, "coverage": available / repeats,
            "worst_rep": worst, "worst_rep_index": int(worst_index) if worst_index is not None else None,
            "worst_rep_complete": complete, "valid_subset_mean": mean, "valid_subset_std": std}


def _peer_model(row, kind):
    key = "/".join(("task3_baselines", *(row[field] for field in (
        "release", "model", "preprocessing", "training_status", "rule",
    ))))
    label = PEER_LABELS[row["model"]]
    if row["rule"] == "fixed":
        label += f" ({row['threshold']})"
    elif row["rule"] in {"dev_selected", "val_selected"}:
        label += " (val threshold)"
    if row["training_status"] == "valid_first_pass":
        label += " [campaign1]"
    fields = ("release", "model", "preprocessing", "training_status", "score_type", "rule", "threshold")
    meta = {"repository": "task3_baselines", "kind": kind} | {field: row[field] for field in fields}
    return key, label, meta


def _merge_peer_rows(metrics, delivery):
    labels = metrics["model_labels"]
    sources = metrics["model_sources"] = {
        model: {"repository": "kron-reduce", "kind": "trajectory", "model": model}
        for model in metrics["rosters"]["val"]["clean"]
    }
    count = {}
    for kind in ("trajectory", "classifier"):
        with (delivery / f"peer_{kind}_summary.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = csv.DictReader(stream)
            columns = [field.removesuffix("_valid_reps") for field in rows.fieldnames if field.endswith("_valid_reps")]
            count[kind] = 0
            for row in rows:
                roster, scenario = row["roster"], row["scenario"]
                repeats = int(row["repetitions"])
                if (row["scope"] != "all" or repeats != (1 if scenario == "clean" else 5)
                        or _number(row["n_events_mean"]) != ROSTER_COUNTS[roster]):
                    raise ValueError(f"peer case coverage differs from the registered TASK3 scope: {roster}/{scenario}")
                name, label, source = _peer_model(row, kind)
                labels[name] = label
                sources[name] = source
                models = metrics["rosters"][roster][scenario]
                if name in models:
                    raise ValueError(f"duplicate peer summary: {roster}/{scenario}/{name}")
                profile = {PROFILE_FIELDS.get(metric, metric): _stats(row, metric, repeats) for metric in columns}
                profile.update({metric: {"mean": None, "replicates": repeats, "available_replicates": 0,
                                         "coverage": 0.0, "worst_rep_complete": False}
                                for metric in UNAVAILABLE_PEER_FIELDS if metric not in profile})
                models[name] = {"summary": {"replicates": repeats, "by_step": {"native": {"all": profile}}},
                                "provenance": source}
                count[kind] += 1
    return count


def _merge_peer_timings(delivery, sources):
    local = _read(delivery / "latency.json")
    peer = _read(delivery / "peer_latency_benchmark.json")
    local["peer_device"] = peer["device"]
    local["peer_trace"] = peer["trace"]
    for row in peer["records"]:
        keys = [name for name, source in sources.items() if source["repository"] == "task3_baselines"
                and source["model"] == row["model"] and source["release"] == "campaign2"
                and source["rule"] in {"fixed", "los_rule"}]
        if len(keys) != 1 or len(row["ms"]) != row["timed_trials"]:
            raise ValueError(f"peer timing does not bind one reported model: {row['model']}")
        name = keys[0]
        kind = sources[name]["kind"]
        got = local["by_model"].setdefault(name, {"native": {}, "scope": f"Peer clean host-to-host {kind}"})
        batch = str(row["batch_size"])
        if batch in got["native"]:
            raise ValueError(f"duplicate peer timing: {name}/{batch}")
        got["native"][batch] = {"total_s": [value / 1000.0 for value in row["ms"]],
                                "reported_step_s": row.get("step_s"), "forecast_s": row.get("forecast_s")}
    return local


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("delivery", type=Path)
    args = parser.parse_args()
    root = project_root()
    delivery = (root / args.delivery).resolve()
    if not delivery.is_relative_to(root / "results"):
        raise ValueError("comparison inputs must be the verified local export")
    hashes = _read(delivery / "export_hashes.json")
    required = ("metrics.json", "latency.json", "peer_trajectory_summary.csv", "peer_classifier_summary.csv",
                "peer_latency_benchmark.json")
    if any(sha256_file(delivery / name) != hashes[name] for name in required):
        raise ValueError("comparison inputs differ from the export receipt")
    metrics = _read(delivery / "metrics.json")
    if set(metrics["rosters"]) != set(ROSTER_COUNTS):
        raise ValueError("the combined tables require all three rosters")
    row_counts = _merge_peer_rows(metrics, delivery)
    latency = _merge_peer_timings(delivery, metrics["model_sources"])
    write_json(delivery / "comparison_metrics.json", metrics)
    write_json(delivery / "comparison_latency.json", latency)
    with (delivery / "cross_repository_comparison.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        metadata = ("repository", "kind", "release", "preprocessing", "training_status", "rule", "threshold")
        writer.writerow(("roster", "scenario", "model", "step_s", "endpoint", "metric", *STATISTICS,
                         "model_label", *metadata))
        for row in metric_rows(metrics):
            model = row[2]
            writer.writerow((*row, metrics["model_labels"].get(model, model),
                             *(metrics["model_sources"][model].get(key) for key in metadata)))
    outputs = ("comparison_metrics.json", "comparison_latency.json", "cross_repository_comparison.csv")
    write_json(delivery / "comparison_hashes.json", {"sources": {name: hashes[name] for name in required},
                                                     "results": {name: sha256_file(delivery / name) for name in outputs}})
    print(json.dumps({"rosters": ROSTER_COUNTS, "peer_source_rows": row_counts, "model_rows": len(metrics["model_sources"])}))


if __name__ == "__main__":
    main()
