"""Frozen A03 LOS scoring and explicitly unmatched TASK3 reference tables."""

from __future__ import annotations

import csv
import hashlib
import json
import time
import tomllib
from pathlib import Path

import h5py
import numpy as np

from kron_reduce.data.history_contract import CausalStateHistory
from kron_reduce.models.learned_latent import history_features
from kron_reduce.models.nonlinear_latent import LearnedNonlinearModel
from kron_reduce.physics.coordinates import coi_project, coi_project_power
from kron_reduce.workflows.nonlinear_history_learning import _write
from project_harness.evidence import sha256_file, verify_artifact_directory
from project_harness.git_state import git_head, status_lines
from project_harness.report_storage import locate_report

SCALES = (0.47675314588683515, 1.0033667249602924)
ORIGIN = 0.2
CLOCK = np.arange(801) * 0.01


def _cohort_name(value):
    return {"DEV": "val", "TEST": "test", "DEV_TEST": "val_plus_test"}[value]


def los_classification(history, prediction, equilibrium, mass, omega_s):
    """Labels/outcome times are deliberately absent from this interface."""
    if prediction.shape != (801, 20):
        raise ValueError("classification requires 800 future points plus the origin")
    observed = np.c_[history.delta, history.omega]
    raw = np.concatenate((observed, prediction[1:]))
    finite = np.isfinite(raw).all(axis=1)
    bad = np.flatnonzero(~finite)
    valid = raw[: bad[0]] if len(bad) else raw
    angle = np.ptp(valid[:, :10] + equilibrium, axis=1)
    speed = np.max(np.abs(coi_project(valid[:, 10:], mass)), axis=1) / omega_s
    exceed = (angle > np.pi) & (speed > 0.02)
    hits = (
        np.flatnonzero(np.convolve(exceed.astype(int), np.ones(5, int), "valid") == 5)
        if len(valid) >= 5
        else []
    )
    hit = int(hits[0] + 4) if len(hits) else None
    settled = bool(
        hit is None
        and not len(bad)
        and len(valid) >= 100
        and np.all(angle[-100:] < np.pi)
        and np.all(speed[-100:] < 0.005)
    )
    return {
        "predicted_los": hit is not None,
        "trigger_index": hit,
        "predicted_los_time_record_s": None if hit is None else hit * 0.01,
        "predicted_los_time_forecast_s": None if hit is None else hit * 0.01 - ORIGIN,
        "window_settled": settled,
        "finite_full_path": not len(bad),
        "maximum_angle_spread_rad": float(np.max(angle)) if len(angle) else None,
        "maximum_coi_speed_pu": float(np.max(speed)) if len(speed) else None,
    }


def numerical_qualification(fine, coarse, fine_class, coarse_class):
    same = fine_class["predicted_los"] == coarse_class["predicted_los"]
    endpoint = 800
    time_ok = True
    if same and fine_class["predicted_los"]:
        endpoint = max(fine_class["trigger_index"], coarse_class["trigger_index"]) - 20
        endpoint = max(0, endpoint)
        time_ok = abs(fine_class["trigger_index"] - coarse_class["trigger_index"]) <= 2
    error = fine[: endpoint + 1] - coarse[: endpoint + 1]
    finite = bool(np.isfinite(fine).all() and np.isfinite(coarse).all())
    value = (
        float(
            np.sqrt(
                0.5
                * np.mean(
                    (error[:, :10] / SCALES[0]) ** 2
                    + (error[:, 10:] / SCALES[1]) ** 2
                )
            )
        )
        if finite
        else None
    )
    return {
        "qualified": bool(finite and same and time_ok and value <= 1e-3),
        "normalized_joint_state_difference": value,
        "comparison_future_points_including_origin": endpoint + 1,
        "same_los_decision": same,
        "los_time_difference_within_20ms": time_ok,
        "finite_full_paths": finite,
        "history_initialization_refined_too": True,
    }


def forecast(model, history, equilibrium, max_step):
    features = np.r_[history_features(history, model.settings.frames), equilibrium]
    seed = model.encoder @ (features / model.feature_scale)
    history_path = model.closure.rollout(
        history.delta[0], history.omega[0], seed, history.time - history.time[0],
        history.constant_power, equilibrium, max_step=max_step,
    )
    return model.closure.rollout(
        history.delta[-1], history.omega[-1], history_path[-1, 20:], CLOCK,
        history.constant_power, equilibrium, max_step=max_step,
    )


def confusion_metrics(tp, fn, fp, tn, unstable_unresolved=0, stable_unresolved=0):
    nu, ns = tp + fn + unstable_unresolved, fp + tn + stable_unresolved

    def divide(a, b):
        return None if not b else a / b

    complete = not (unstable_unresolved or stable_unresolved)
    return {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "eligible_unstable": nu, "eligible_stable": ns, "eligible": nu + ns,
        "unstable_unresolved": unstable_unresolved,
        "stable_unresolved": stable_unresolved,
        "numerical_unresolved": unstable_unresolved + stable_unresolved,
        "fsr": divide(fn, nu) if complete else None,
        "fur": divide(fp, ns) if complete else None,
        "fsr_lower": divide(fn, nu), "fsr_upper": divide(fn + unstable_unresolved, nu),
        "fur_lower": divide(fp, ns), "fur_upper": divide(fp + stable_unresolved, ns),
        "accuracy": divide(tp + tn, nu + ns) if complete else None,
        "balanced_accuracy": (divide(tp, nu) + divide(tn, ns)) / 2
        if complete and nu and ns else None,
        "recall": divide(tp, nu) if complete else None,
        "precision": divide(tp, tp + fp) if complete else None,
        "f1": divide(2 * tp, 2 * tp + fp + fn) if complete else None,
    }


def summarize(rows):
    active = [r for r in rows if r["prospective_eligible"]]
    u = [r for r in active if r["label"] == "unstable"]
    s = [r for r in active if r["label"] == "stable"]
    tp = sum(r["qualified"] and r["predicted_los"] for r in u)
    fn = sum(r["qualified"] and not r["predicted_los"] for r in u)
    fp = sum(r["qualified"] and r["predicted_los"] for r in s)
    tn = sum(r["qualified"] and not r["predicted_los"] for r in s)
    return {
        **confusion_metrics(tp, fn, fp, tn,
                            sum(not r["qualified"] for r in u),
                            sum(not r["qualified"] for r in s)),
        "history_eligible": len(rows),
        "already_los_within_observation": len(rows) - len(active),
        "false_settled": sum(r["qualified"] and r["window_settled"] for r in u),
        "stable_unsettled": sum(
            r["qualified"] and not r["predicted_los"] and not r["window_settled"] for r in s
        ),
        "weighting": "event_equal",
    }


def _csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, keys)
        writer.writeheader()
        writer.writerows(rows)


def load_frozen_val(source_root, frozen):
    allocation = json.loads((frozen / "role_allocation.json").read_text(encoding="utf-8"))
    score = json.loads((frozen / "candidate-sine-no-history-loss-val.json").read_text())
    eligible = {r["event_id"] for r in score["available"]["horizons"]["3.0"]["rows"]}
    events, coverage, accesses, bindings = [], [], [], {}
    for name, original_binding in allocation["source_bindings"].items():
        directory = source_root / name
        for filename in ("dataset_index.csv", "splits.json"):
            if sha256_file(directory / filename) != original_binding[filename]:
                raise ValueError("source metadata no longer matches frozen A03 run")
        index = {r["event_id"]: r for r in _csv_rows(directory / "dataset_index.csv")}
        split = json.loads((directory / "splits.json").read_text())
        stat = (directory / "dataset.h5").stat()
        bindings[name] = {
            "frozen_bindings": original_binding,
            "current_h5_stat": {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns},
            "current_run_full_h5_hash_read": False,
        }
        with h5py.File(directory / "dataset.h5", "r") as handle:
            ws = float(handle.attrs["omega_s"])
            mass = np.asarray(handle["inertia"][...]) / ws
            eq = coi_project(np.asarray(handle["equilibrium/delta"][...]), mass)
            pm_eq = np.asarray(handle["equilibrium/Pm"][...])
            for row in allocation["rows"]:
                if row["source_dataset"] != name or row["role"] != "val":
                    continue
                if row["source_role"] != "train" or row["raw_id"] not in split["train"]:
                    raise ValueError("source holdout read refused")
                meta = index[row["raw_id"]]
                if meta["label"] not in {"stable", "unstable"}:
                    raise ValueError("unknown truth label")
                entry = {**row, "label": meta["label"], "history_eligible": False}
                if row["event_id"] not in eligible:
                    if int(meta["valid_length"]) >= 22:
                        raise ValueError("frozen eligibility inconsistent with metadata")
                    coverage.append({**entry, "exclusion_reason": "frozen_history_ineligible"})
                    continue
                group = handle["events"][row["raw_id"]]
                if group["t"].shape[0] != int(meta["valid_length"]):
                    raise ValueError("source index/array length mismatch")
                clock = np.asarray(group["t"][:21])
                delta = coi_project(np.asarray(group["delta"][:21]), mass) - eq
                omega = coi_project((np.asarray(group["omega"][:21]) - 1) * ws, mass)
                power = coi_project_power(np.asarray(group["Pm"][0]) - pm_eq, mass)
                np.testing.assert_allclose(clock, np.arange(21) * 0.01, rtol=0, atol=1e-10)
                history = CausalStateHistory(clock - clock[-1], delta, omega, power)
                digest = hashlib.sha256()
                for value in (clock, delta, omega, power, eq, mass):
                    digest.update(np.ascontiguousarray(value).tobytes())
                truth_time = float(meta["event_time_s"]) if meta["event_time_s"] else None
                prospective = truth_time is None or truth_time > ORIGIN + 1e-10
                entry.update(history_eligible=True, prospective_eligible=prospective,
                             true_event_time_record_s=truth_time,
                             input_arrays_sha256=digest.hexdigest())
                coverage.append(entry)
                events.append({**entry, "history": history, "mass": mass,
                               "equilibrium": eq, "omega_s": ws})
                accesses.append({"event_id": row["event_id"], "source_role": "train",
                                 "slices": {"t": "0:21", "delta": "0:21", "omega": "0:21",
                                            "Pm": "0"}})
    if {e["event_id"] for e in events} != eligible or len(coverage) != 101:
        raise ValueError("frozen val cohort mismatch")
    return events, coverage, accesses, bindings


def task3_tables(task3_root):
    files = [
        "paper/latest_algorithm_others/comparison_all_views.csv",
        "paper/latest_algorithm_others/all_event_classifications.csv",
        "paper/latest_algorithm_others/original_probability_classifiers.csv",
        "paper/latest_algorithm_others/pcdk_score_all_views.csv",
        "paper/campaign2_final/classification_events_all_views.csv",
        *[f"paper/latest_algorithm_extension/{c}/event_evidence.json"
          for c in ("DEV", "TEST", "DEV_TEST")],
    ]
    hashes = {p: sha256_file(task3_root / p) for p in files}
    evidence = _csv_rows(task3_root / files[1]) + _csv_rows(task3_root / files[4])
    for cohort, filename in zip(("DEV", "TEST", "DEV_TEST"), files[5:], strict=True):
        extension = json.loads((task3_root / filename).read_text(encoding="utf-8-sig"))
        evidence.extend({**r, "cohort": cohort, "model": "history_deeponet_p0"}
                        for r in extension["classifications"])
    references = []
    for source in _csv_rows(task3_root / files[0]):
        cohort, model = source["cohort"], source["model"]
        events = [r for r in evidence if r["cohort"] == cohort and r["model"] == model]
        if len(events) != int(source["eligible"]):
            raise ValueError(f"TASK3 event denominator mismatch: {cohort}/{model}")
        u = [r for r in events if r["label"] == "unstable"]
        s = [r for r in events if r["label"] == "stable"]
        qualified = lambda r: str(r["qualified"]).lower() == "true"
        positive = lambda r: str(r["predicted_los"]).lower() == "true"
        values = (
            sum(qualified(r) and positive(r) for r in u),
            sum(qualified(r) and not positive(r) for r in u),
            sum(qualified(r) and positive(r) for r in s),
            sum(qualified(r) and not positive(r) for r in s),
            sum(not qualified(r) for r in u), sum(not qualified(r) for r in s),
        )
        metrics = confusion_metrics(*values)
        for key, source_key in (("tp", "detected"), ("fn", "missed"), ("fp", "false_los"),
                                ("unstable_unresolved", "unstable_unresolved"),
                                ("numerical_unresolved", "numerical_unresolved")):
            if metrics[key] != int(source[source_key]):
                raise ValueError(f"TASK3 event/count mismatch: {cohort}/{model}/{key}")
        references.append({**source, **metrics, "source_cohort": cohort,
                           "cohort": _cohort_name(cohort),
                           "online_observation_s": 0.1, "forecast_horizon_s": 8.0,
                           "matched_to_a03": False})
    classifiers = []
    for source in _csv_rows(task3_root / files[2]):
        metrics = confusion_metrics(*(int(source[k]) for k in ("tp", "fn", "fp", "tn")))
        if not np.isclose(metrics["balanced_accuracy"], float(source["balanced_accuracy"])):
            raise ValueError("TASK3 probability classifier arithmetic mismatch")
        classifiers.append({**source, **metrics, "source_cohort": source["cohort"],
                            "cohort": _cohort_name(source["cohort"]),
                            "online_observation_s": 0.1, "matched_to_a03": False})
    diagnostic = []
    for source in _csv_rows(task3_root / files[3]):
        metrics = confusion_metrics(*(int(source[k]) for k in ("tp", "fn", "fp", "tn")))
        diagnostic.append({**source, **metrics, "source_cohort": source["cohort"],
                           "cohort": _cohort_name(source["cohort"]),
                           "matched_to_a03": False})
    return references, classifiers, diagnostic, hashes


def run(*, root, experiment, stage, result_dir, run_id):
    if stage != "stability_evaluation" or experiment["data_policy"] != "kiki_original_train_only":
        raise ValueError("unregistered stability task")
    started = time.perf_counter()
    config_path = root / "configs/experiments" / experiment["config"]
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    frozen = root / config["frozen_run"]
    verify_artifact_directory(frozen, strict=True)
    freeze = json.loads((frozen / "selection_freeze.json").read_text())
    refs, classifiers, diagnostics, task3_hashes = task3_tables(Path(config["task3_root"]))
    from kron_reduce.data.kiki_history import resolve_source_root

    events, coverage, accesses, source_bindings = load_frozen_val(
        resolve_source_root(root, config["source_root"]), frozen,
    )
    _write(result_dir / "pre_evaluation_freeze.json", {
        "selected_model_unchanged": freeze["selected"],
        "candidate_checkpoints": freeze["candidate_table"],
        "frozen_run_artifacts_sha256": sha256_file(frozen / "artifact_hashes.json"),
        "protocol_sha256": sha256_file(root / locate_report(root, config["protocol_path"])),
        "task3_source_hashes": task3_hashes,
        "cohort": "a03_internal_val_from_source_train", "history_points": 21,
        "forecast_horizon_s": 8.0, "output_dt_s": 0.01, "integration_steps_s": [0.005, 0.01],
        "labels_used_for_fitting_selection_or_model_inputs": False,
        "fitting_executed": False, "stability_result_used_to_reselect": False,
    })
    all_summaries = []
    for candidate in freeze["candidate_table"]:
        name = candidate["name"]
        checkpoint = frozen / f"candidate-{name}.npz"
        if sha256_file(checkpoint) != candidate["checkpoint_sha256"]:
            raise ValueError("frozen model hash mismatch")
        model = LearnedNonlinearModel.load(checkpoint)
        fitted = set(model.fit_record["fitted_event_ids"])
        if fitted & {e["event_id"] for e in events}:
            raise ValueError("model fitted on scoring events")
        rows, arrays = [], {}
        for index, event in enumerate(events):
            np.testing.assert_allclose(model.closure.mass, event["mass"], rtol=0, atol=1e-12)
            row = {k: event[k] for k in ("event_id", "source_dataset", "fault_line", "label",
                                        "prospective_eligible", "true_event_time_record_s")}
            try:
                fine = forecast(model, event["history"], event["equilibrium"], 0.005)
                coarse = forecast(model, event["history"], event["equilibrium"], 0.01)
                arrays[f"fine_{index:03d}"] = fine
                arrays[f"coarse_{index:03d}"] = coarse
                args = (event["equilibrium"], event["mass"], event["omega_s"])
                c = los_classification(event["history"], fine[:, :20], *args)
                d = los_classification(event["history"], coarse[:, :20], *args)
                numerical = numerical_qualification(fine[:, :20], coarse[:, :20], c, d)
                row.update(c, **numerical)
                if not row["qualified"]:
                    row.update(unqualified_los=row["predicted_los"], predicted_los=False,
                               window_settled=False)
            except (FloatingPointError, OverflowError, ValueError) as error:
                row.update(qualified=False, predicted_los=False, window_settled=False,
                           error=f"{type(error).__name__}: {error}")
            rows.append(row)
            if (index + 1) % 20 == 0:
                print(f"{name}: {index + 1}/{len(events)} frozen forecasts scored", flush=True)
        summary = {"model": name, "cohort": "a03_internal_val", **summarize(rows),
                   "selected_before_stability_scoring": name == freeze["selected"]["name"],
                   "online_observation_s": ORIGIN, "forecast_horizon_s": 8.0,
                   "checkpoint_sha256": candidate["checkpoint_sha256"]}
        _write(result_dir / f"{name}-events.json", {"summary": summary, "rows": rows})
        np.savez_compressed(result_dir / f"{name}-predictions.npz", **arrays)
        all_summaries.append(summary)
        print(json.dumps(summary), flush=True)
    _write_csv(result_dir / "a03_stability.csv", all_summaries)
    _write_csv(result_dir / "task3_trajectory_references.csv", refs)
    _write_csv(result_dir / "task3_direct_classifiers.csv", classifiers)
    _write_csv(result_dir / "task3_surrogate_diagnostic.csv", diagnostics)
    _write(result_dir / "data_access_audit.json", {
        "coverage": coverage, "accesses": accesses, "source_bindings": source_bindings,
        "original_source_val_payload_ids_read": [], "test_payload_ids_read": [],
        "future_truth_read": False, "external_files_written": False,
    })
    for path, digest in task3_hashes.items():
        if sha256_file(Path(config["task3_root"]) / path) != digest:
            raise ValueError("TASK3 reference changed during scoring")
    verify_artifact_directory(frozen, strict=True)
    own_sources = [
        "src/kron_reduce/workflows/kiki_stability_comparison.py",
        "src/kron_reduce/models/nonlinear_latent.py",
        "src/kron_reduce/models/learned_latent.py",
        "src/kron_reduce/data/history_contract.py",
        "src/kron_reduce/physics/coordinates.py",
        config["protocol_path"],
    ]
    _write(result_dir / "lineage.json", {
        "run_id": run_id, "source_commit": git_head(root), "git_status": status_lines(root),
        "source_sha256": {p: sha256_file(root / p) for p in own_sources},
        "config_sha256": sha256_file(config_path), "task3_source_hashes": task3_hashes,
        "wall_seconds": time.perf_counter() - started, "model_updates": 0,
    })
    _write(result_dir / "stage_evidence.json", {
        "process_status": "PASS", "source_commit": git_head(root),
        "scientific_verdict": "RETRY",
        "evaluation_scope": "EXPLORATORY_UNMATCHED_COMPARISON",
        "promotion_decision": "KEEP_BASELINE", "data_policy": config["data_policy"],
        "source_val_ids_read": [], "test_ids_read": [], "fitting_executed": False,
        "level_6_claimed": False, "claim_boundary": config["claim_boundary"],
    })
    _write(result_dir / "artifact_hashes.json", {
        p.name: sha256_file(p) for p in sorted(result_dir.iterdir()) if p.is_file()
    })
    verify_artifact_directory(result_dir, strict=True)
