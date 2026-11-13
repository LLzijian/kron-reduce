"""One-use arithmetic verification of saved R20/R21 TASK3 trajectories."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from project_harness.manifest import project_root

AUTHORITY = "kiki_machine_local_r20_r21_saved_audit_v1_20261114"
REFERENCE = "configs/proposals/kiki_task3_split_reference_v1.json"
FIT_PLAN = "configs/proposals/kiki_machine_local_r20_r21_fit_v1.json"
ROLES = ("train", "val", "test")
GROUPS = ("r20", "r21")
NPZ_DECODE_LIMIT = 6
KEYS = ("ids", "fine", "refined", "truth", "lengths", "mass", "equilibrium", "labels")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
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


def compare(actual, expected, path="", mismatches=None):
    mismatches = [] if mismatches is None else mismatches
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            mismatches.append(path + ":keys")
            return mismatches
        for key in expected:
            compare(actual[key], expected[key], f"{path}.{key}", mismatches)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            mismatches.append(path + ":length")
        else:
            for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
                compare(a, e, f"{path}[{i}]", mismatches)
    elif isinstance(expected, (float, int)) and not isinstance(expected, bool):
        if (not isinstance(actual, (float, int)) or isinstance(actual, bool)
                or not math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12)):
            mismatches.append(path + ":value")
    elif type(actual) is not type(expected) or actual != expected:
        mismatches.append(path + ":value")
    return mismatches


def expected_rosters(reference):
    result = {role: [] for role in ROLES}
    for source in reference["official_kiki"]["sources"]:
        for role, source_role in (("train", "train"), ("val", "validation")):
            result[role].extend(source["roles"][source_role]["eligible_100ms"])
    result["test"] = [
        f"{'ieee39_expansion' if event.startswith('expansion_') else 'ieee39_holdout_supplement'}/{event}"
        for event in reference["task3_campaign2_final"]["eligible_ids"]
    ]
    result = {role: sorted(ids) for role, ids in result.items()}
    if {k: len(v) for k, v in result.items()} != {"train": 346, "val": 107, "test": 76}:
        raise ValueError("frozen TASK3 reference does not define 346/107/76 rosters")
    return result


def reserve(root, plan_arg, output_path):
    policy = read_json(root / "configs/research_policy.json")
    auth = policy.get("saved_evidence_audits", {}).get(AUTHORITY)
    if (not isinstance(auth, dict) or auth.get("status") != "active"
            or auth.get("max_executions") != 1 or auth.get("retry_allowed") is not False
            or auth.get("max_npz_decodes") != 6 or auth.get("new_model_calls") != 0):
        raise RuntimeError("saved-evidence audit authority is not active and one-use")
    plan_rel, review_rel = auth["plan_path"], auth["review_path"]
    plan_path = (root / plan_rel).resolve(strict=True)
    if Path(plan_arg).resolve(strict=True) != plan_path:
        raise RuntimeError("supplied plan differs from the one-use registered plan")
    review_path = (root / review_rel).resolve(strict=True)
    plan_hash, review_hash = sha256_file(plan_path), sha256_file(review_path)
    if auth.get("plan_sha256") != plan_hash or auth.get("review_sha256") != review_hash:
        raise RuntimeError("audit plan or review hash differs from authorization")
    plan, review = read_json(plan_path), read_json(review_path)
    if (plan.get("authorization") != AUTHORITY or plan.get("schema") != "kiki_machine_local_r20_r21_saved_audit.v1"
            or review.get("approved") is not True or review.get("plan_sha256") != plan_hash
            or review.get("plan_path") != plan_rel
            or review.get("implementation_bindings") != plan.get("implementation_bindings")):
        raise RuntimeError("audit plan or scientific review differs from exact authorization")
    output = output_path.resolve()
    if (output.as_posix() != (root / plan["report_path"]).resolve().as_posix()
            or output.exists() or root not in output.parents):
        raise RuntimeError("output path must be the new report path bound by the plan")
    for rel, digest in plan["implementation_bindings"].items():
        target = Path(rel)
        actual = sha256_file(target if target.is_absolute() else root / target)
        if actual != digest:
            raise RuntimeError(f"local implementation binding differs: {rel}")
    for absolute, digest in plan["external_code"].items():
        if sha256_file(Path(absolute)) != digest:
            raise RuntimeError(f"external TASK3 code differs: {absolute}")
    if not plan.get("implementation_bindings") or not plan.get("external_code"):
        raise RuntimeError("plan must bind local source and official scoring code")
    reference_path = root / plan["reference_path"]
    if (plan["reference_path"] != REFERENCE or sha256_file(reference_path) != plan["reference_sha256"]):
        raise RuntimeError("frozen roster reference hash differs")
    fit_plan_path = root / plan["fit_plan_path"]
    if plan["fit_plan_path"] != FIT_PLAN or sha256_file(fit_plan_path) != plan["fit_plan_sha256"]:
        raise RuntimeError("paired fit plan hash differs")
    fit_plan = read_json(fit_plan_path)
    source_contract = fit_plan["source_contract"]
    source_path = root / source_contract["path"]
    if sha256_file(source_path) != source_contract["sha256"]:
        raise RuntimeError("paired fit plan source-contract hash differs")
    source_plan = read_json(source_path)
    source_rosters = source_plan["sources"]["rosters"]
    source_events = {role: {row["id"]: row for row in source_rosters[role]} for role in ROLES}
    labels = {role: {row["id"]: row["label"] for row in source_rosters[role]}
              for role in ROLES}
    if plan["expected_labels"] != labels:
        raise RuntimeError("expected labels differ from frozen paired fit roster")
    common = Path(subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=root,
        capture_output=True, text=True, check=True,
    ).stdout.strip()).resolve(strict=True)
    receipt_dir = common / "kiki_machine_local_r20_r21_saved_audit_budgets" / AUTHORITY
    receipt_dir.parent.mkdir(parents=True, exist_ok=True)
    receipt_dir.mkdir(exist_ok=False)
    marker = {"authorization": AUTHORITY, "plan_sha256": plan_hash, "review_sha256": review_hash,
              "reserved_at_utc": datetime.now(UTC).isoformat(), "spent": True, "retry_allowed": False}
    with (receipt_dir / "reservation.json").open("x", encoding="utf-8") as stream:
        json.dump(marker, stream, indent=2, allow_nan=False)
    return plan, labels, source_events


def safe_child(parent, rel):
    path = (parent / rel).resolve(strict=True)
    if not path.is_file() or not path.is_relative_to(parent.resolve(strict=True)):
        raise ValueError(f"bound artifact is missing or escapes its run directory: {rel}")
    return path


def load_saved(root, run_dir, key, plan, role, roster, labels, source_events):
    record = plan["run_files"][key]
    npz_path, json_path = safe_child(run_dir, record["npz_path"]), safe_child(run_dir, record["json_path"])
    manifest_path = run_dir / "artifact_hashes.json"
    manifest = read_json(manifest_path)
    npz_rel = npz_path.relative_to(run_dir).as_posix()
    json_rel = json_path.relative_to(run_dir).as_posix()
    npz_bytes = npz_path.read_bytes()
    json_bytes = json_path.read_bytes()
    if (hashlib.sha256(npz_bytes).hexdigest() != record["npz_sha256"]
            or hashlib.sha256(json_bytes).hexdigest() != record["json_sha256"]
            or manifest.get(npz_rel) != record["npz_sha256"]
            or manifest.get(json_rel) != record["json_sha256"]):
        raise ValueError(f"{key} bytes differ from the plan or run artifact manifest")
    metadata = json.loads(json_bytes)
    # One decode per forecast archive; arrays are copied before the archive closes.
    with np.load(io.BytesIO(npz_bytes), allow_pickle=False) as archive:
        if set(archive.files) != set(KEYS):
            raise ValueError(f"{key} archive fields differ from the frozen contract")
        arrays = {name: archive[name].copy() for name in archive.files}
    ids = arrays["ids"].astype(str).tolist()
    n = len(roster)
    if ids != roster or arrays["labels"].shape != (n,) or arrays["lengths"].shape != (n,):
        raise ValueError(f"{key} IDs/labels/lengths do not match the frozen roster")
    if (arrays["fine"].shape != (n, 801, 20) or arrays["refined"].shape != (n, 801, 20)
            or arrays["mass"].shape != (n, 10) or arrays["equilibrium"].shape != (n, 10)):
        raise ValueError(f"{key} array dimensions differ from the TASK3 score contract")
    if (arrays["lengths"].dtype.kind not in "iu" or np.any(arrays["lengths"] < 12)
            or np.any(arrays["lengths"] > 811) or arrays["truth"].ndim != 2
            or arrays["truth"].shape[1] != 20 or int(arrays["lengths"].sum()) != len(arrays["truth"])
            or not np.isfinite(arrays["truth"]).all() or not np.isfinite(arrays["mass"]).all()
            or np.any(arrays["mass"] <= 0) or not np.isfinite(arrays["equilibrium"]).all()):
        raise ValueError(f"{key} truth/static arrays are invalid")
    expected = np.array([labels[role][event] for event in roster])
    if not np.array_equal(arrays["labels"].astype(str), expected.astype(str)):
        raise ValueError(f"{key} labels differ from the frozen source-qualified roster")
    classes = {row["event_id"]: row for row in metadata["classifications"]}
    groups = {row["event_id"]: row["group"] for row in metadata["records"]}
    if set(classes) != set(ids) or set(groups) != set(ids):
        raise ValueError(f"{key} saved JSON metadata does not cover the full roster")
    if any(groups[event] != source_events[role][event]["group"]
           or classes[event].get("true_event_time_record_s") != source_events[role][event].get("event_time_s")
           or classes[event].get("label") != source_events[role][event]["label"]
           for event in ids):
        raise ValueError(f"{key} event group/time/label metadata differs from the frozen source contract")
    events, offset = [], 0
    for i, event_id in enumerate(ids):
        length = int(arrays["lengths"][i])
        raw = arrays["truth"][offset:offset + length]
        offset += length
        events.append({"id": event_id, "x": raw, "raw_x": raw, "mass": arrays["mass"][i],
                       "equilibrium_delta": arrays["equilibrium"][i], "omega_s": 2 * math.pi * 60,
                       "label": str(arrays["labels"][i]), "group": groups[event_id],
                       "event_time_s": classes[event_id].get("true_event_time_record_s")})
    return arrays, events, metadata


def score_saved(arrays, events, metadata, target_scale, task3_evaluate, task3_classify, summarize):
    fine, refined = arrays["fine"], arrays["refined"]
    scoring = [{**e, "equilibrium": e["equilibrium_delta"]} for e in events]
    result = task3_evaluate(fine, scoring, refined)
    stability_events = []
    for i, (event, classification) in enumerate(zip(events, result["classifications"], strict=True)):
        predicted = ("unstable" if classification["predicted_los"] else
                    "stable" if classification["window_settled"] else None)
        refined_class = task3_classify(refined[i], scoring[i])
        refined_label = ("unstable" if refined_class["predicted_los"] else
                         "stable" if refined_class["window_settled"] else None)
        qualified = bool(classification["qualified"] and predicted == refined_label)
        stability_events.append({"id": event["id"], "truth_label": event["label"],
            "predicted_label": predicted, "qualified": qualified, "prospective_eligible": True,
            "numerical_failure": not qualified, "unsettled": qualified and predicted is None})
    stability = summarize(stability_events)
    for key, numerator, denominator in (
        ("recall_on_resolved", stability["tp"], stability["tp"] + stability["fn"]),
        ("miss_rate_on_resolved", stability["fn"], stability["tp"] + stability["fn"]),
        ("false_alarm_rate_on_resolved", stability["fp"], stability["tn"] + stability["fp"]),
    ):
        stability[key] = numerator / denominator if denominator else None
    expected = {k: result[k] for k in ("summary", "unstable_prefix", "stable_complete", "counts", "records", "classifications")}
    expected.update(stability=stability, stability_events=stability_events,
                    task3_six_term_score=result["selection_score"])
    projected_truth = []
    offset = 0
    for i, length_value in enumerate(arrays["lengths"]):
        length = int(length_value)
        raw = arrays["truth"][offset:offset + length]
        offset += length
        mass = arrays["mass"][i]
        projected_truth.append(np.c_[raw[:, :10] - np.average(raw[:, :10], axis=1, weights=mass)[:, None],
                                     raw[:, 10:] - np.average(raw[:, 10:], axis=1, weights=mass)[:, None]])
    terminal = {}
    for step, paths in (("0.0025", arrays["fine"]), ("0.00125", arrays["refined"])):
        values = []
        for i, truth in enumerate(projected_truth):
            count = min(300, len(truth) - 11)
            if not stability_events[i]["qualified"]:
                values.append(None)
                continue
            delta = (paths[i, 1:count + 1] - truth[11:11 + count]) / target_scale
            value = .5 * (np.mean(delta[:100] ** 2) + np.mean(delta ** 2))
            values.append(float(value) if np.isfinite(value) else None)
        terminal[step] = {"event_values": values, "qualified": sum(v is not None for v in values),
                          "allocated": len(values),
                          "mean": float(np.mean(values)) if all(v is not None for v in values) else None}
    expected["terminal_train_scale_nmse_1s_3s"] = terminal
    mismatches = compare(safe({k: metadata.get(k) for k in expected}), safe(expected), "role")
    result["stability"] = stability
    result["stability_events"] = stability_events
    result["task3_six_term_score"] = result["selection_score"]
    result["terminal_train_scale_nmse_1s_3s"] = terminal
    return result, stability, stability_events, mismatches


def physical_rmse_summary(records, lengths, labels, roster):
    result = {}
    for horizon in ("1", "2", "5", "8"):
        eligible = [event for event in roster if lengths[event] >= int(horizon) * 100 + 11]
        rows = [r for r in records if r["horizon"] == horizon and r.get("qualified")]
        result[horizon + "s"] = {}
        for field in ("delta_rmse_rad", "omega_rmse_rad_s"):
            valid = [r for r in rows if isinstance(r.get(field), (int, float))
                     and math.isfinite(float(r[field])) and r.get("n", 0) > 0]
            total_n = sum(r["n"] // 10 for r in valid)
            result[horizon + "s"][field] = {
                "pooled": (float(np.sqrt(sum(float(r[field]) ** 2 * (r["n"] // 10) for r in valid) / total_n))
                           if eligible and len(valid) == len(eligible) else None),
                "valid_mean": float(np.mean([r[field] for r in valid])) if valid else None,
                "valid_count": len(valid), "eligible_by_truth_length": len(eligible),
                "coverage": len(valid) / len(eligible) if eligible else None,
            }
    unstable = [event for event in roster if labels[event] == "unstable" and lengths[event] >= 12]
    prefix = [r for r in records if r["horizon"] == "prefix" and r["label"] == "unstable" and r.get("qualified")]
    result["unstable_available_prefix"] = {}
    for field in ("delta_rmse_rad", "omega_rmse_rad_s"):
        valid = [r for r in prefix if isinstance(r.get(field), (int, float))
                 and math.isfinite(float(r[field])) and r.get("n", 0) > 0]
        total_n = sum(r["n"] // 10 for r in valid)
        result["unstable_available_prefix"][field] = {
            "pooled": (float(np.sqrt(sum(float(r[field]) ** 2 * (r["n"] // 10) for r in valid) / total_n))
                       if unstable and len(valid) == len(unstable) else None),
            "valid_mean": float(np.mean([r[field] for r in valid])) if valid else None,
            "valid_count": len(valid), "eligible_unstable_by_truth_length": len(unstable),
            "coverage": len(valid) / len(unstable) if unstable else None,
        }
    return result


def paired_value_differences(roster, values20, values21):
    differences = [{"id": event, "r20": values20[event], "r21": values21[event],
                    "r21_minus_r20": values21[event] - values20[event]}
                   for event in roster if event in values20 and event in values21
                   and math.isfinite(values20[event]) and math.isfinite(values21[event])]
    delta = [row["r21_minus_r20"] for row in differences]
    return {"full_mean": float(np.mean(delta)) if roster and len(delta) == len(roster) else None,
            "valid_mean": float(np.mean(delta)) if delta else None,
            "valid_count": len(delta), "eligible_count": len(roster), "per_event_differences": differences}


def main_run(root, plan_path, output_path):
    plan, labels, source_events = reserve(root, plan_path, output_path)
    ref = read_json(root / REFERENCE)
    rosters = expected_rosters(ref)
    if plan["rosters"] != rosters:
        raise RuntimeError("plan rosters differ from frozen TASK3 source-qualified IDs")
    if {role: sorted(labels[role]) for role in ROLES} != rosters:
        raise RuntimeError("fit-plan source rosters differ from frozen TASK3 IDs")
    run_dirs = {stage: (root / plan[f"{stage}_run_dir"]).resolve(strict=True) for stage in ("fit", "test")}
    for stage, run_id in (("fit", plan["fit_run_id"]), ("test", plan["test_run_id"])):
        if run_dirs[stage].name != run_id or not run_dirs[stage].is_relative_to(root.resolve(strict=True)):
            raise RuntimeError(f"{stage} run directory does not match bound run identity")
        manifest = read_json(run_dirs[stage] / "artifact_hashes.json")
        if sha256_file(run_dirs[stage] / "artifact_hashes.json") != plan[f"{stage}_artifact_manifest_sha256"]:
            raise RuntimeError(f"{stage} artifact manifest hash differs")
        if not isinstance(manifest, dict):
            raise TypeError(f"{stage} artifact manifest is invalid")
    preprocessing_path = (run_dirs["fit"] / plan["preprocessing_path"]).resolve(strict=True)
    if (not preprocessing_path.is_relative_to(run_dirs["fit"])
            or sha256_file(preprocessing_path) != plan["preprocessing_sha256"]):
        raise RuntimeError("fit preprocessing bytes differ from the saved-audit plan")
    preprocessing = read_json(preprocessing_path)
    target_scale = np.asarray(preprocessing["target"], dtype=float)
    if target_scale.shape != (20,) or not np.isfinite(target_scale).all() or np.any(target_scale <= 0):
        raise RuntimeError("saved train target scale is invalid")
    if set(plan["run_files"]) != {"r20_train", "r20_val", "r21_train", "r21_val", "r20_test", "r21_test"}:
        raise RuntimeError("plan must bind exactly the six paired forecast archives")
    task3_root = Path(plan["task3_root"])
    sys.path.insert(0, str(task3_root / "src"))
    from task3.c2.evaluation import evaluate as task3_evaluate
    from task3.protocol import classify as task3_classify

    from kron_reduce.evaluation.trajectory_stability import summarize

    rows, loaded = [], {}
    for role in ROLES:
        stage = "test" if role == "test" else "fit"
        pair = {}
        for group in GROUPS:
            key = f"{group}_{role}"
            arrays, events, metadata = load_saved(
                root, run_dirs[stage], key, plan, role, rosters[role], labels, source_events)
            other = pair.get("reference")
            if other is not None:
                for field in ("ids", "truth", "lengths", "mass", "equilibrium", "labels"):
                    if not np.array_equal(arrays[field], other[field]):
                        raise ValueError(f"paired {role} {field} differ between R20 and R21")
            else:
                pair["reference"] = arrays
            recomputed, stability, events, mismatches = score_saved(
                arrays, events, metadata, target_scale, task3_evaluate, task3_classify, summarize)
            saved_subset = {k: metadata.get(k) for k in ("summary", "unstable_prefix", "stability", "task3_six_term_score")}
            expected_subset = {k: (stability if k == "stability" else
                                   recomputed["selection_score"] if k == "task3_six_term_score" else
                                   recomputed.get(k)) for k in saved_subset}
            mismatches.extend(compare(safe(saved_subset), safe(expected_subset), f"{key}.summary"))
            loaded[key] = (recomputed, arrays, events, stability)
            lengths_by_id = dict(zip(rosters[role], arrays["lengths"].astype(int).tolist(), strict=True))
            rows.append({"key": key, "role": role, "model": group, "count": len(events),
                         "saved_npz_sha256": plan["run_files"][key]["npz_sha256"],
                         "trajectory_summary": safe(recomputed["summary"]),
                         "terminal_train_scale_nmse_1s_3s": safe(recomputed["terminal_train_scale_nmse_1s_3s"]),
                         "physical_rmse": physical_rmse_summary(
                             recomputed["records"], lengths_by_id, labels[role], rosters[role]),
                         "stability": safe(stability),
                         "mismatch_count": len(mismatches), "mismatches": mismatches[:100]})
    if len(loaded) != NPZ_DECODE_LIMIT:
        raise RuntimeError("saved audit did not decode exactly the six registered forecast archives")
    # Pair means use only shared, finite event/horizon scores; denominators come from true lengths.
    paired, classification_changes = {}, {}
    for role in ROLES:
        a = loaded[f"r20_{role}"][0]["records"]
        b = loaded[f"r21_{role}"][0]["records"]
        keyed = []
        for group_rows in (a, b):
            mapping = {}
            for row in group_rows:
                key = (row["event_id"], str(row["horizon"]))
                if key in mapping:
                    raise RuntimeError(f"duplicate {role} event/horizon score row: {key}")
                mapping[key] = row
            keyed.append(mapping)
        arrays = loaded[f"r20_{role}"][1]
        lengths = dict(zip(rosters[role], arrays["lengths"].astype(int).tolist(), strict=True))
        pairs = []
        populations = [(str(h), [event for event in rosters[role]
                                  if lengths[event] >= int(h) * 100 + 11]) for h in (1, 2, 5, 8)]
        populations.append(("prefix_unstable", [event for event in rosters[role]
                            if labels[role][event] == "unstable"]))
        metrics = ("rmse", "normalized_sse", "delta_sse", "omega_sse", "common_sse",
                   "delta_rmse_rad", "omega_rmse_rad_s")
        for horizon, eligible in populations:
            if horizon == "prefix_unstable":
                rows_a = {event: row for (event, h), row in keyed[0].items() if h == "prefix"}
                rows_b = {event: row for (event, h), row in keyed[1].items() if h == "prefix"}
            else:
                rows_a = {event: row for (event, h), row in keyed[0].items() if h == horizon}
                rows_b = {event: row for (event, h), row in keyed[1].items() if h == horizon}
            metric_summaries = {}
            for metric in metrics:
                differences = []
                for event in eligible:
                    ra, rb = rows_a.get(event), rows_b.get(event)
                    if ra is None or rb is None or not (ra.get("qualified") and rb.get("qualified")):
                        continue
                    va, vb = ra.get(metric), rb.get(metric)
                    if (isinstance(va, (int, float)) and not isinstance(va, bool)
                            and isinstance(vb, (int, float)) and not isinstance(vb, bool)
                            and math.isfinite(float(va)) and math.isfinite(float(vb))):
                        differences.append({"id": event, "r20": float(va), "r21": float(vb),
                                            "r21_minus_r20": float(vb) - float(va)})
                valid_diffs = [row["r21_minus_r20"] for row in differences]
                metric_summaries[metric] = {
                    "full_mean": float(np.mean(valid_diffs)) if len(valid_diffs) == len(eligible) and eligible else None,
                    "valid_mean": float(np.mean(valid_diffs)) if valid_diffs else None,
                    "valid_count": len(valid_diffs), "eligible_count": len(eligible),
                    "per_event_differences": differences,
                }
            pairs.append({"horizon": horizon, "eligible_by_truth_length": len(eligible),
                "r20_score_rows": sum(event in rows_a for event in eligible),
                "r21_score_rows": sum(event in rows_b for event in eligible),
                "missing_r20_ids": [event for event in eligible if event not in rows_a],
                "missing_r21_ids": [event for event in eligible if event not in rows_b],
                "unqualified_r20_ids": [event for event in eligible
                    if event in rows_a and not rows_a[event].get("qualified")],
                "unqualified_r21_ids": [event for event in eligible
                    if event in rows_b and not rows_b[event].get("qualified")],
                "metrics_r21_minus_r20": metric_summaries})
        if sorted({event for event, _ in keyed[0]}) != rosters[role] or sorted({event for event, _ in keyed[1]}) != rosters[role]:
            raise RuntimeError(f"paired {role} score rows do not cover the frozen roster")
        paired[role] = pairs
        cls20 = {row["id"]: row for row in loaded[f"r20_{role}"][2]}
        cls21 = {row["id"]: row for row in loaded[f"r21_{role}"][2]}
        if cls20.keys() != cls21.keys() or sorted(cls20) != rosters[role]:
            raise RuntimeError(f"paired {role} classifications do not cover the frozen roster")
        state20, state21 = (
            {event: (row["predicted_label"] or "unresolved") if row["qualified"] else "numerical_failure"
             for event, row in group.items()}
            for group in (cls20, cls21)
        )
        changes = [{"id": event, "truth": cls20[event]["truth_label"],
                    "r20": state20[event], "r21": state21[event]}
                   for event in rosters[role] if state20[event] != state21[event]]
        classification_changes[role] = {"changed_count": len(changes), "rows": changes,
            "transition_counts": dict(Counter(f"{row['r20']}_to_{row['r21']}" for row in changes))}
    paired_terminal = {}
    for role in ROLES:
        paired_terminal[role] = {}
        for step in ("0.0025", "0.00125"):
            v20 = loaded[f"r20_{role}"][0]["terminal_train_scale_nmse_1s_3s"][step]["event_values"]
            v21 = loaded[f"r21_{role}"][0]["terminal_train_scale_nmse_1s_3s"][step]["event_values"]
            values20 = {event: float(value) for event, value in zip(rosters[role], v20, strict=True)
                        if value is not None and math.isfinite(float(value))}
            values21 = {event: float(value) for event, value in zip(rosters[role], v21, strict=True)
                        if value is not None and math.isfinite(float(value))}
            paired_terminal[role][step] = paired_value_differences(rosters[role], values20, values21)
    summaries = {stage: read_json(run_dirs[stage] / "summary.json") for stage in ("fit", "test")}
    for stage in summaries:
        if sha256_file(run_dirs[stage] / "summary.json") != plan[f"{stage}_summary_sha256"]:
            raise RuntimeError(f"{stage} result summary hash differs from the plan")
    output = {"authorization": AUTHORITY,
        "status": "PASS" if all(row["mismatch_count"] == 0 for row in rows) else "MISMATCH",
        "fit_run_id": plan["fit_run_id"], "test_run_id": plan["test_run_id"],
        "roster_counts": {k: len(v) for k, v in rosters.items()}, "roles": rows,
        "npz_decodes": len(loaded), "max_npz_decodes": NPZ_DECODE_LIMIT,
        "paired_horizon_means": paired, "paired_terminal_differences": paired_terminal,
        "matched_classification_changes": classification_changes,
        "scope": "six saved forecast arrays decoded once each; no source/cache/checkpoint, model, RHS, integration, training, or optimizer calls",
        "completed_at_utc": datetime.now(UTC).isoformat()}
    # Check the recorded per-role summary entries without trusting them as the source of recomputation.
    for group in GROUPS:
        for role in ROLES:
            stage = "test" if role == "test" else "fit"
            key = f"{group}_{role}"
            expected = {field: loaded[key][0][field]
                        for field in ("summary", "unstable_prefix", "stability", "task3_six_term_score")}
            mismatches = compare(safe(summaries[stage]["records"][group][role]), safe(expected), f"{key}.summary")
            if mismatches:
                output["status"] = "MISMATCH"
                row = next(r for r in rows if r["key"] == key)
                row["mismatch_count"] += len(mismatches)
                row["mismatches"].extend(mismatches[:100])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(output_path).open("x", encoding="utf-8") as stream:
        json.dump(safe(output), stream, indent=2, allow_nan=False)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = main_run(project_root(), args.plan, args.output)
    print(json.dumps({"status": result["status"], "mismatches": sum(r["mismatch_count"] for r in result["roles"])}, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
