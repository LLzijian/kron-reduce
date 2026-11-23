#!/usr/bin/env python3
"""One-use arithmetic audit of saved full-roster TASK3 predictions only."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from project_harness.manifest import project_root
from project_harness.report_storage import locate_report

AUTHORITY = "kiki_machine_local_full_saved_audit_v1_20261113"
REFERENCE = Path("configs/proposals/kiki_task3_split_reference_v1.json")
FIT_PLAN = Path("configs/proposals/kiki_machine_local_full_fit_v1.json")
OMEGA_S = 2 * math.pi * 60
ROLES = {"train": "fit", "val": "fit", "test": "test"}
REQUIRED_ARRAYS = {"ids", "fine", "refined", "truth", "lengths", "mass", "equilibrium", "labels"}


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, np.generic):
        return safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def close(actual, expected, route, mismatches):
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            mismatches.append({"path": route, "reason": "type"})
            return
        if actual.keys() != expected.keys():
            mismatches.append({"path": route, "reason": "keys"})
        for key in actual.keys() & expected.keys():
            close(actual[key], expected[key], f"{route}.{key}", mismatches)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            mismatches.append({"path": route, "reason": "length_or_type"})
            return
        for index, (a, e) in enumerate(zip(actual, expected, strict=True)):
            close(a, e, f"{route}[{index}]", mismatches)
    elif isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if actual != expected or type(actual) is not type(expected):
            mismatches.append({"path": route, "reason": "value", "expected": expected, "actual": actual})
    elif isinstance(expected, (int, float)):
        if not isinstance(actual, (int, float)) or isinstance(actual, bool):
            mismatches.append({"path": route, "reason": "type"})
        elif not math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12):
            mismatches.append({"path": route, "reason": "value", "expected": expected, "actual": actual})
    elif actual != expected:
        mismatches.append({"path": route, "reason": "value"})


def expected_rosters(reference):
    result = {"train": [], "val": [], "test": []}
    for source in reference["official_kiki"]["sources"]:
        for role, source_role in (("train", "train"), ("val", "validation")):
            result[role].extend(source["roles"][source_role]["eligible_100ms"])
    result["test"] = [
        f"{'ieee39_expansion' if event_id.startswith('expansion_') else 'ieee39_holdout_supplement'}/{event_id}"
        for event_id in reference["task3_campaign2_final"]["eligible_ids"]
    ]
    for role, ids in result.items():
        result[role] = sorted(ids)
    if {role: len(ids) for role, ids in result.items()} != {"train": 346, "val": 107, "test": 76}:
        raise ValueError("frozen reference does not define the complete 346/107/76 rosters")
    return result


def reserve(root: Path, fit_arg: Path, test_arg: Path, output_arg: Path):
    policy = read_json(root / "configs/research_policy.json")
    entry = policy.get("saved_evidence_audits", {}).get(AUTHORITY)
    if not isinstance(entry, dict) or entry.get("status") != "active" or entry.get("max_executions") != 1 or entry.get("retry_allowed") is not False:
        raise RuntimeError("saved-evidence audit authorization is not active and one-use")
    plan_path = (root / entry["plan_path"]).resolve(strict=True)
    review_path = (root / locate_report(root, entry["review_path"], entry.get("review_sha256"))).resolve(strict=True)
    plan_hash, review_hash = sha256_file(plan_path), sha256_file(review_path)
    if entry.get("plan_sha256") != plan_hash or entry.get("review_sha256") != review_hash:
        raise RuntimeError("audit plan or review hash differs from active authorization")
    plan, review = read_json(plan_path), read_json(review_path)
    if (plan.get("authorization") != AUTHORITY or plan.get("schema") != "kiki_machine_local_full_saved_audit.v1"
            or review.get("approved") is not True or review.get("plan_sha256") != plan_hash
            or review.get("plan_path") != entry["plan_path"]
            or review.get("implementation_bindings") != plan.get("implementation_bindings")):
        raise RuntimeError("audit plan or approval does not match the authorized scope")
    if plan.get("fit_run_id") != fit_arg.name or plan.get("test_run_id") != test_arg.name:
        raise RuntimeError("supplied run directories differ from the registered run identities")
    for arg, key in ((fit_arg, "fit_run_dir"), (test_arg, "test_run_dir")):
        expected = (root / plan[key]).resolve(strict=True)
        if arg.resolve(strict=True) != expected or root not in expected.parents:
            raise RuntimeError(f"supplied {key} differs from the bound run directory")
    output = output_arg.resolve()
    if output.exists() or root not in output.parents or output in (fit_arg.resolve(), test_arg.resolve()):
        raise RuntimeError("audit output must be a new path inside this checkout")
    reference_path = root / REFERENCE
    if plan.get("reference_path") != REFERENCE.as_posix() or plan.get("reference_sha256") != sha256_file(reference_path):
        raise RuntimeError("frozen roster reference hash differs from audit plan")
    fit_plan_path = root / FIT_PLAN
    fit_plan_sha = sha256_file(fit_plan_path)
    if (plan.get("fit_plan_path") != FIT_PLAN.as_posix()
            or plan.get("fit_plan_sha256") != fit_plan_sha):
        raise RuntimeError("original fit plan hash differs from saved-audit plan")
    fit_plan = read_json(fit_plan_path)
    frozen_labels = {
        role: {item["id"]: item["label"] for item in fit_plan["sources"]["rosters"][role]}
        for role in ("train", "val", "test")
    }
    if plan.get("expected_labels") != frozen_labels:
        raise RuntimeError("audit expected-label mapping differs from original fit plan sources.rosters")
    for relative, digest in plan.get("implementation_bindings", {}).items():
        target = Path(relative)
        if target.is_absolute():
            actual = sha256_file(target)
        else:
            actual = sha256_file(root / locate_report(root, target.as_posix(), digest))
        if actual != digest:
            raise RuntimeError(f"audit implementation binding differs: {relative}")
    for absolute, digest in plan.get("external_code", {}).items():
        if sha256_file(Path(absolute)) != digest:
            raise RuntimeError(f"bound TASK3 code changed: {absolute}")
    if not plan.get("implementation_bindings") or not plan.get("external_code"):
        raise RuntimeError("audit plan must bind local and TASK3 scoring code")
    common = Path(subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=root, capture_output=True, text=True, check=True,
    ).stdout.strip()).resolve(strict=True)
    reservation_dir = common / "kiki_machine_local_full_saved_audit_budgets" / AUTHORITY
    reservation_dir.parent.mkdir(parents=True, exist_ok=True)
    reservation_dir.mkdir(exist_ok=False)
    marker = {"authorization": AUTHORITY, "plan_sha256": plan_hash, "review_sha256": review_hash,
              "fit_run_id": plan["fit_run_id"], "test_run_id": plan["test_run_id"],
              "reserved_at_utc": datetime.now(UTC).isoformat(), "spent": True, "retry_allowed": False}
    with (reservation_dir / "reservation.json").open("x", encoding="utf-8") as stream:
        json.dump(marker, stream, indent=2, allow_nan=False)
    return plan, output, frozen_labels


def load_saved_role(root: Path, run_dir: Path, role: str, plan: dict, roster: list[str]):
    record = plan["run_files"][role]
    npz_path, json_path = run_dir / record["npz_path"], run_dir / record["json_path"]
    for path in (npz_path, json_path):
        if path.resolve().parent != run_dir.resolve() or not path.is_file():
            raise ValueError(f"expected saved file missing or outside its bound run: {path}")
    npz_bytes = npz_path.read_bytes()
    json_bytes = json_path.read_bytes()
    if sha256_bytes(npz_bytes) != record["npz_sha256"] or sha256_bytes(json_bytes) != record["json_sha256"]:
        raise ValueError(f"saved {role} artifact hash differs from the audit plan")
    metadata = json.loads(json_bytes)
    with np.load(io.BytesIO(npz_bytes), allow_pickle=False) as archive:
        if set(archive.files) != REQUIRED_ARRAYS:
            raise ValueError(f"saved {role} array fields differ from the frozen contract")
        arrays = {key: archive[key].copy() for key in archive.files}
    ids = arrays["ids"].astype(str).tolist()
    n = len(roster)
    if ids != roster or arrays["labels"].shape != (n,) or arrays["lengths"].shape != (n,):
        raise ValueError(f"saved {role} IDs or one-dimensional arrays differ from the frozen roster")
    if arrays["fine"].shape != (n, 801, 20) or arrays["refined"].shape != (n, 801, 20):
        raise ValueError(f"saved {role} predictions do not contain 801 x 20 trajectories for all cases")
    if arrays["mass"].shape != (n, 10) or arrays["equilibrium"].shape != (n, 10):
        raise ValueError(f"saved {role} static arrays have invalid shapes")
    if not np.isfinite(arrays["mass"]).all() or not np.all(arrays["mass"] > 0) or not np.isfinite(arrays["equilibrium"]).all():
        raise ValueError(f"saved {role} static arrays contain invalid values")
    if arrays["lengths"].dtype.kind not in "iu":
        raise ValueError(f"saved {role} lengths must be integer typed")
    lengths = arrays["lengths"].astype(np.int64, copy=False)
    if (np.any(lengths < 12) or np.any(lengths > 811) or int(lengths.sum()) != len(arrays["truth"])
            or arrays["truth"].ndim != 2 or arrays["truth"].shape[1] != 20
            or not np.isfinite(arrays["truth"]).all()):
        raise ValueError(f"saved {role} truth lengths do not align")
    truth_hash = hashlib.sha256(np.ascontiguousarray(arrays["truth"]).tobytes()).hexdigest()
    label_values = np.array([plan["expected_labels"][role][event_id] for event_id in roster])
    if not np.array_equal(arrays["labels"], label_values):
        raise ValueError(f"saved {role} labels differ from the frozen fit-plan roster")
    if not np.array_equal(arrays["labels"], np.array([r["label"] for r in metadata["classifications"]])):
        raise ValueError(f"saved {role} labels differ from result metadata")
    classes = {row["event_id"]: row for row in metadata["classifications"]}
    groups = {row["event_id"]: row["group"] for row in metadata["records"]}
    if set(classes) != set(ids) or set(groups) != set(ids):
        raise ValueError(f"saved {role} event metadata does not cover the full roster")
    events, offset = [], 0
    for i, event_id in enumerate(ids):
        length = int(lengths[i])
        raw_x = arrays["truth"][offset:offset + length]
        offset += length
        event_class = classes[event_id]
        events.append({"id": event_id, "x": raw_x, "mass": arrays["mass"][i],
                       "equilibrium_delta": arrays["equilibrium"][i], "omega_s": OMEGA_S,
                       "label": str(arrays["labels"][i]), "group": groups[event_id],
                       "event_time_s": event_class.get("true_event_time_record_s")})
    return arrays, events, metadata, truth_hash


def audit_role(root: Path, run_dir: Path, role: str, plan: dict, roster: list[str], task3_eval, task3_classify, summarize):
    arrays, events, saved, truth_hash = load_saved_role(root, run_dir, role, plan, roster)
    recomputed = task3_eval(arrays["fine"], events, arrays["refined"])
    stability_events = []
    for index, (event, classification) in enumerate(zip(events, recomputed["classifications"], strict=True)):
        predicted = ("unstable" if classification["predicted_los"] else
                    "stable" if classification["window_settled"] else None)
        refined = task3_classify(arrays["refined"][index], event)
        refined_label = ("unstable" if refined["predicted_los"] else
                         "stable" if refined["window_settled"] else None)
        qualified = bool(classification["qualified"] and predicted == refined_label)
        stability_events.append({"id": event["id"], "truth_label": event["label"],
            "predicted_label": predicted, "qualified": qualified,
            "prospective_eligible": True, "numerical_failure": not qualified,
            "unsettled": predicted is None})
    stability = summarize(stability_events)
    for name, numerator, denominator in (("recall_on_resolved", stability["tp"], stability["tp"] + stability["fn"]),
            ("miss_rate_on_resolved", stability["fn"], stability["tp"] + stability["fn"]),
            ("false_alarm_rate_on_resolved", stability["fp"], stability["tn"] + stability["fp"])):
        stability[name] = numerator / denominator if denominator else None
    expected = {key: recomputed[key] for key in ("summary", "unstable_prefix", "stable_complete", "counts", "records", "classifications")}
    expected.update(stability=stability, stability_events=stability_events,
                    task3_six_term_score=recomputed["selection_score"])
    actual = {key: saved.get(key) for key in expected}
    mismatches = []
    close(safe(actual), safe(expected), role, mismatches)
    return {"role": role, "event_count": len(events), "truth_sha256": truth_hash,
            "saved_npz_sha256": plan["run_files"][role]["npz_sha256"],
            "recomputed_selection_score": safe(recomputed["selection_score"]),
            "train_diagnosis": train_diagnosis(saved) if role == "train" else None,
            "mismatch_count": len(mismatches), "mismatches": mismatches[:200]}


def train_diagnosis(saved: dict) -> dict:
    """Describe train-only history residual and future prefix-error association."""
    initialization = [row for row in saved.get("initialization", []) if row.get("step") == 0.0025]
    prefix = [row for row in saved.get("records", []) if row.get("horizon") == "prefix"]

    def unique_by_id(rows, name):
        mapping = {row["id"] if "id" in row else row["event_id"]: row for row in rows}
        if len(mapping) != len(rows):
            raise ValueError(f"duplicate {name} rows in saved train result")
        return mapping

    init_by_id, prefix_by_id = unique_by_id(initialization, "fine-step initialization"), unique_by_id(prefix, "prefix score")
    values = {name: [] for name in ("history_nrmse", "hidden_start_energy", "hidden_origin_energy", "prefix_task3_rmse")}
    invalid, missing_history, missing_prefix = 0, 0, 0
    matched_ids = []
    for event_id in sorted(init_by_id.keys() | prefix_by_id.keys()):
        if event_id not in init_by_id:
            missing_history += 1
            continue
        if event_id not in prefix_by_id:
            missing_prefix += 1
            continue
        init, score = init_by_id[event_id], prefix_by_id[event_id]
        row = (init.get("history_nrmse"), init.get("hidden_start_energy"),
               init.get("hidden_origin_energy"), score.get("rmse"))
        if any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0 for value in row):
            invalid += 1
            continue
        matched_ids.append(event_id)
        for name, value in zip(values, row, strict=True):
            values[name].append(float(value))

    summaries = {
        name: {"median": float(np.median(array)), "p90": float(np.percentile(array, 90)), "max": float(np.max(array))}
        for name, rows in values.items() if (array := np.asarray(rows, dtype=float)).size
    }
    history = np.asarray(values["history_nrmse"], dtype=float)
    future = np.asarray(values["prefix_task3_rmse"], dtype=float)
    correlation = None
    if len(history) >= 2 and np.std(np.log1p(history)) > 0 and np.std(np.log1p(future)) > 0:
        correlation = float(np.corrcoef(np.log1p(history), np.log1p(future))[0, 1])
    top_count = math.ceil(len(matched_ids) * 0.25)
    ranked = sorted(range(len(matched_ids)), key=lambda i: (history[i], matched_ids[i]))
    top = ranked[-top_count:] if top_count else []
    remaining = ranked[:-top_count] if top_count else []
    return {
        "method": "train-only descriptive arithmetic; no selection or tuning",
        "count": len(matched_ids), "missing_history_count": missing_history,
        "missing_prefix_count": missing_prefix, "invalid_join_count": invalid,
        "summaries": summaries,
        "log1p_pearson_history_nrmse_vs_prefix_task3_rmse": correlation,
        "history_residual_top_quartile": {
            "count": len(top), "prefix_task3_rmse_mean": float(np.mean(future[top])) if top else None,
        },
        "remaining_history_residuals": {
            "count": len(remaining), "prefix_task3_rmse_mean": float(np.mean(future[remaining])) if remaining else None,
        },
        "quartile_rule": "highest ceil(25% of matched count), ranked by history_nrmse then event ID",
    }


def run(root: Path, fit_arg: Path, test_arg: Path, output_arg: Path):
    fit_arg, test_arg = fit_arg.resolve(strict=True), test_arg.resolve(strict=True)
    plan, output, frozen_labels = reserve(root, fit_arg, test_arg, output_arg)
    reference_path = root / REFERENCE
    if plan.get("reference_sha256") != sha256_file(reference_path):
        raise RuntimeError("registered roster reference hash changed after authorization")
    reference = read_json(reference_path)
    rosters = expected_rosters(reference)
    if plan.get("rosters") != rosters:
        raise RuntimeError("audit plan roster order differs from the fixed TASK3 reference")
    if {role: sorted(labels) for role, labels in frozen_labels.items()} != {role: rosters[role] for role in rosters}:
        raise RuntimeError("original fit-plan source rosters differ from the frozen TASK3 reference")
    task3_root = Path(plan["task3_root"])
    sys.path.insert(0, str(task3_root / "src"))
    from task3.c2.evaluation import evaluate as task3_eval
    from task3.protocol import classify as task3_classify

    from kron_reduce.evaluation.trajectory_stability import summarize

    results = []
    for role, stage in ROLES.items():
        run_dir = fit_arg if stage == "fit" else test_arg
        results.append(audit_role(root, run_dir, role, plan, rosters[role], task3_eval, task3_classify, summarize))
    report = {"authorization": AUTHORITY, "status": "PASS" if all(not row["mismatch_count"] for row in results) else "MISMATCH",
              "fit_run_id": plan["fit_run_id"], "test_run_id": plan["test_run_id"],
              "roster_counts": {key: len(ids) for key, ids in rosters.items()}, "roles": results,
              "train_diagnosis": next(row["train_diagnosis"] for row in results if row["role"] == "train"),
              "audit_scope": "saved-array arithmetic only; no model, RHS, integration, fit, decomposition, or autodiff calls",
              "completed_at_utc": datetime.now(UTC).isoformat()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-run-dir", type=Path, required=True)
    parser.add_argument("--test-run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = project_root()
    report = run(root, args.fit_run_dir, args.test_run_dir, args.output)
    print(json.dumps({"status": report["status"], "mismatches": sum(r["mismatch_count"] for r in report["roles"])}, indent=2))


if __name__ == "__main__":
    main()
