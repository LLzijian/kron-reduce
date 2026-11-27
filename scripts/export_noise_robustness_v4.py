"""Export bound V4 aggregates and candidate gates without reopening trajectories."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from kron_reduce.evaluation.constant_c_observability import write_json
from kron_reduce.evaluation.noise_stress_matrix_v4 import V4
from kron_reduce.evaluation.task3_noise_scores import summarize_reps
from kron_reduce.workflows.noise_robustness_esrom import _json
from kron_reduce.workflows.noise_robustness_esrom_v2 import v1_reference
from kron_reduce.workflows.noise_robustness_esrom_v4 import REPORT_DIR, WORKFLOW, _phase, _plans
from project_harness.evidence import sha256_file
from project_harness.loop import load_state
from project_harness.manifest import project_root


def _review(selection):
    return selection["evaluated"][selection["model_key"]]


def _val_choice(validation, stages):
    start = _json(validation / "selection.json")
    candidates = [(validation, start)]
    sections = {"zero_retrain": start}
    if "adapt" in stages:
        directory, evidence = stages["adapt"]
        final = _json(directory / "selection.json")
        training = _json(directory / "training_curves.json")
        sections["adapt"] = {"stage": evidence, "final": final, "training": training}
        for item in training["val"]:
            checkpoint = directory / f"val-update-{item['update']:02d}"
            selection_path = checkpoint / "selection.json"
            if sha256_file(selection_path) != item["selection_sha256"]:
                raise ValueError("conditional-fit checkpoint selection differs from its receipt")
            candidates.append((checkpoint, _json(selection_path)))

    def rank(item):
        comparison = _review(item[1])["mean_relative_improvement"]
        return item[1]["passed"], comparison if comparison is not None else float("-inf")

    return (*max(candidates, key=rank), sections)


def collect_local(root):
    plan, _, digest = _plans(root)
    history = load_state(root, WORKFLOW)["history"]
    stages = {row["to"]: _phase(root, row["to"], digest) for row in history
              if row.get("to") in {"validation", "adapt", "freeze", "blind"}}
    if "validation" not in stages:
        raise PermissionError("a complete bound val matrix is required before exporting results")
    validation, evidence = stages["validation"]
    directory, choice, selections = _val_choice(validation, stages)
    if "freeze" in stages:
        folder, status = stages["freeze"]
        freeze = _json(root / status["frozen_model_manifest"]["path"])
        if (sha256_file(folder / "freeze_receipt.json") != status["frozen_model_manifest"]["sha256"]
                or freeze["candidate"] != choice["candidate"] or not choice["passed"]):
            raise ValueError("the exported val candidate differs from the frozen choice")
    archive, _, _ = v1_reference(root, plan)
    measured = _json(directory / "metrics.json")["rosters"]["val"]
    val = {}
    for scenario, rows in measured.items():
        controls = {name: archive[scenario][name] for name in plan["models"][:-1]}
        # The archived repeats retain classification, energy and endpoint strata.
        val[scenario] = {name: result | {"summary": summarize_reps(result["repeats"])}
                         for name, result in controls.items()}
        val[scenario][V4] = rows[choice["model_key"]]
    rosters = {"val": val}
    if "blind" in stages:
        folder, _ = stages["blind"]
        heldout = _json(folder / "metrics.json")["rosters"]
        if set(heldout) != {"test", "blind_v2"}:
            raise ValueError("the joint holdout must provide both complete rosters")
        rosters.update(heldout)
    latency_root = (stages["adapt"][0] if choice["optimizer_updates"] else validation)
    latency_path = latency_root / "latency.json"
    latency = _json(latency_path) if latency_path.is_file() else None
    info = {"reported_candidate": choice, "validation": evidence, "selections": selections,
            "aggregate_source_path": directory.relative_to(root).as_posix(),
            "stages": {name: ev for name, (_, ev) in stages.items()}}
    models = {"rosters": rosters, "model_labels": {V4: "ES-ROM pooled MHE V4"}}
    return models, latency, info


STATISTICS = ("mean", "sample_std", "replicates", "available_replicates", "coverage",
              "worst_rep", "worst_rep_index", "worst_rep_complete", "valid_subset_mean", "valid_subset_std")


def metric_rows(metrics):
    for roster, conditions in metrics["rosters"].items():
        for scenario, variants in conditions.items():
            for model, result in variants.items():
                for step, endpoints in result["summary"]["by_step"].items():
                    for endpoint, values in endpoints.items():
                        for metric, stats in values.items():
                            yield (roster, scenario, model, step, endpoint, metric,
                                   *(stats.get(key) for key in STATISTICS))


def write_comparison_csv(output, metrics):
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(("roster", "scenario", "model", "step_s", "endpoint", "metric", *STATISTICS))
        writer.writerows(metric_rows(metrics))


def peer_aggregates(root, stages):
    heldout = stages.get("blind")
    if heldout is None:
        raise PermissionError("the peer aggregates are released only after the V4 joint holdout")
    receipt_path = root / REPORT_DIR / "_sealed" / "noise_cross_repository.json"
    receipt = _json(receipt_path)
    if (receipt["status"] != "PASS" or (receipt["matched"], receipt["allocated"]) != (138, 138)
            or receipt["holdout_run"] != heldout["artifact_manifest"]
            or receipt["local_sha256"] != heldout["noise_manifest_sha256"]):
        raise PermissionError("cross-repository summaries require a verified 138-group noise match")
    plan, _, _ = _plans(root)
    peer = Path(plan["counterpart"])
    return {name: (peer / name).read_bytes()
            for name in ("trajectory_summary.csv", "classifier_summary.csv", "latency_benchmark.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New export folder under results/")
    parser.add_argument("--include-peer", action="store_true",
                        help="Copy the authorized peer aggregates after verified holdout and exact hash match")
    args = parser.parse_args()
    root = project_root()
    output = (root / args.out).resolve()
    if output == root / "results" or not output.is_relative_to(root / "results"):
        raise ValueError("evaluation tables must stay under the workspace results directory")
    metrics, latency, info = collect_local(root)
    peer = peer_aggregates(root, info["stages"]) if args.include_peer else {}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "metrics.json", metrics)
    write_json(output / "selection_provenance.json", info)
    if latency is not None:
        write_json(output / "latency.json", latency)
    for name, content in peer.items():
        (output / f"peer_{name}").write_bytes(content)
    write_comparison_csv(output / "comparison_metrics.csv", metrics)
    write_json(output / "export_hashes.json", {p.name: sha256_file(p) for p in sorted(output.iterdir())})
    print(output)


if __name__ == "__main__":
    main()
