#!/usr/bin/env python3
"""One-use arithmetic audit of saved machine-local hidden-oracle outputs."""
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

EXPECTED_OUTPUT = Path("reports/generated/machine_local_hidden_oracle_20261113/saved-audit.json")
PLAN_SCHEMA = "machine_local_hidden_oracle_saved_audit.v1"
INPUT_FIELDS = {"paths", "coarse_paths", "scale", "history_scale", "mass", "ids", "roles"}


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def safe(value):
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, np.generic):
        return safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def compare(actual, expected, route, mismatches):
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            mismatches.append({"path": route, "reason": "type"})
            return
        if actual.keys() != expected.keys():
            mismatches.append({"path": route, "reason": "keys"})
        for key in actual.keys() & expected.keys():
            compare(actual[key], expected[key], f"{route}.{key}", mismatches)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            mismatches.append({"path": route, "reason": "length_or_type"})
            return
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            compare(left, right, f"{route}[{index}]", mismatches)
    elif isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if type(actual) is not type(expected) or actual != expected:
            mismatches.append({"path": route, "reason": "value", "expected": expected, "actual": actual})
    elif isinstance(expected, (int, float)):
        if not isinstance(actual, (int, float)) or isinstance(actual, bool):
            mismatches.append({"path": route, "reason": "type"})
        elif not math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12):
            mismatches.append({"path": route, "reason": "value", "expected": expected, "actual": actual})
    elif actual != expected:
        mismatches.append({"path": route, "reason": "value"})


def role_npz_names(module) -> list[str]:
    return ["inputs.npz", *(f"{group}_{role}.npz" for group in module.GROUPS for role in module.ROLES)]


def role_fields(group: str) -> set[str]:
    fields = {f"{step}_{name}" for step in ("fine", "refined") for name in ("forecast",)}
    if not group.endswith("_oracle"):
        fields.update(f"{step}_{name}" for step in ("fine", "refined")
                      for name in ("history", "iterates", "increments", "singular", "residuals"))
    return fields


def authorize(root: Path, plan_arg: Path, run_arg: Path, output_arg: Path):
    plan_path = plan_arg.resolve(strict=True)
    if root.resolve() not in plan_path.parents:
        raise RuntimeError("audit plan must be inside the project checkout")
    policy = read_json(root / "configs/research_policy.json")
    plan = read_json(plan_path)
    authority = plan.get("authorization")
    entry = policy.get("saved_evidence_audits", {}).get(authority)
    relative_plan = plan_path.relative_to(root).as_posix()
    plan_hash = sha256_file(plan_path)
    if (plan.get("schema") != PLAN_SCHEMA or not isinstance(entry, dict)
            or entry.get("status") != "active" or entry.get("max_executions") != 1
            or entry.get("retry_allowed") is not False or entry.get("plan_path") != relative_plan
            or entry.get("plan_sha256") != plan_hash or entry.get("max_npz_decodes") != 13
            or entry.get("new_model_calls") != 0):
        raise RuntimeError("saved-array audit authorization is not active and exactly bound")
    review_path = (root / locate_report(root, entry["review_path"], entry.get("review_sha256"))).resolve(strict=True)
    if root.resolve() not in review_path.parents or sha256_file(review_path) != entry.get("review_sha256"):
        raise RuntimeError("saved-array audit review hash differs")
    review = read_json(review_path)
    if (review.get("approved") is not True or review.get("plan_path") != relative_plan
            or review.get("plan_sha256") != plan_hash
            or review.get("implementation_bindings") != plan.get("implementation_bindings")):
        raise RuntimeError("saved-array audit review does not approve this exact plan")
    if plan.get("output_path") != EXPECTED_OUTPUT.as_posix():
        raise RuntimeError("saved-array audit output path differs from its fixed destination")
    run_dir = run_arg.resolve(strict=True)
    expected_run = (root / plan["run_dir"]).resolve(strict=True)
    if (run_dir != expected_run or run_dir.parent != (root / "results").resolve(strict=True)
            or run_dir.name != plan.get("run_id")):
        raise RuntimeError("supplied run directory differs from the bound immutable run")
    output = output_arg.resolve()
    expected_output = (root / EXPECTED_OUTPUT).resolve()
    if output != expected_output or output.exists():
        raise RuntimeError("audit output must be the new registered generated report path")
    for relative, digest in plan.get("implementation_bindings", {}).items():
        path = Path(relative)
        actual = sha256_file(path if path.is_absolute()
                             else root / locate_report(root, path.as_posix(), digest))
        if actual != digest:
            raise RuntimeError(f"saved-audit implementation binding differs: {relative}")
    if not plan.get("implementation_bindings"):
        raise RuntimeError("saved-audit plan lacks implementation bindings")
    if plan.get("npz_decode_limit") != 13:
        raise RuntimeError("saved-audit plan must permit exactly thirteen NPZ decodes")
    return plan, entry, run_dir, output, plan_hash, review


def reserve(root: Path, plan: dict, entry: dict, plan_hash: str, review: dict) -> Path:
    common = Path(subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=root, capture_output=True, text=True, check=True,
    ).stdout.strip()).resolve(strict=True)
    reservation = common / "machine_local_hidden_oracle_saved_audit_budgets" / plan["authorization"]
    reservation.parent.mkdir(parents=True, exist_ok=True)
    reservation.mkdir(exist_ok=False)
    marker = {"authorization": plan["authorization"], "plan_sha256": plan_hash,
              "review_sha256": entry["review_sha256"], "run_id": plan["run_id"],
              "spent": True, "retry_allowed": False, "npz_decode_limit": 13,
              "reserved_at_utc": datetime.now(UTC).isoformat()}
    (reservation / "reservation.json").write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    return reservation


def verify_artifact_manifest(run_dir: Path, plan: dict, allowed_npz: list[str], truth_bytes: bytes):
    manifest_path = run_dir / "artifact_hashes.json"
    raw_manifest = manifest_path.read_bytes()
    if sha256_bytes(raw_manifest) != plan.get("artifact_manifest_sha256"):
        raise RuntimeError("run artifact_hashes.json differs from the saved-audit plan")
    manifest = json.loads(raw_manifest)
    actual_files = {path.name for path in run_dir.iterdir() if path.is_file()} - {"artifact_hashes.json"}
    if actual_files != set(manifest):
        raise RuntimeError("run artifact file set differs from artifact_hashes.json")
    if sha256_bytes(truth_bytes) != manifest.get("truth.json"):
        raise RuntimeError("truth.json hash differs from the run artifact manifest")
    payloads = {}
    for name, expected in manifest.items():
        path = run_dir / name
        if path.parent.resolve() != run_dir.resolve() or not path.is_file():
            raise RuntimeError(f"unsafe or missing run artifact: {name}")
        raw = truth_bytes if name == "truth.json" else path.read_bytes()
        if sha256_bytes(raw) != expected:
            raise RuntimeError(f"run artifact hash mismatch: {name}")
        if name in allowed_npz:
            payloads[name] = raw
    return manifest, payloads, raw_manifest


def decode_once(name: str, raw: bytes, expected_fields: set[str]) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        if set(archive.files) != expected_fields:
            raise ValueError(f"{name}: NPZ fields differ from the frozen contract")
        return {key: archive[key].copy() for key in archive.files}


def frozen_roles(module) -> np.ndarray:
    return np.array(["train"] * len(range(*module.ROLES["train"].indices(64)))
                    + ["val"] * len(range(*module.ROLES["val"].indices(64)))
                    + ["test"] * len(range(*module.ROLES["test"].indices(64)))
                    )


def load_truth_inputs(module, input_arrays: dict[str, np.ndarray], truth_data: dict):
    paths, coarse = input_arrays["paths"], input_arrays["coarse_paths"]
    scale, hscale, mass = input_arrays["scale"], input_arrays["history_scale"], input_arrays["mass"]
    roles = frozen_roles(module)
    checks = []
    if paths.shape != (64, 811, 30) or coarse.shape != (64, 811, 30):
        raise ValueError("frozen truth inputs must be [64,811,30] at both reference steps")
    if scale.shape != (20,) or hscale.shape != (20,) or mass.shape != (10,):
        raise ValueError("frozen scale/mass arrays have invalid shape")
    if input_arrays["ids"].shape != (64,) or not np.array_equal(input_arrays["ids"], np.arange(64)):
        raise ValueError("frozen event IDs differ")
    if input_arrays["roles"].shape != (64,) or not np.array_equal(input_arrays["roles"].astype(str), roles):
        raise ValueError("frozen role allocation differs")
    if not all(np.isfinite(array).all() for array in (paths, coarse, scale, hscale, mass)):
        raise ValueError("frozen truth inputs or scales contain nonfinite values")
    if not np.all(mass > 0) or not np.all(scale > 0) or not np.all(hscale > 0):
        raise ValueError("frozen mass/scales must be positive")
    expected_scale = np.maximum(np.sqrt(np.mean(paths[:32, 11:311, :20] ** 2, axis=(0, 1))), 1e-5)
    expected_hscale = np.maximum(np.sqrt(np.mean(paths[:32, :11, :20] ** 2, axis=(0, 1))), 1e-5)
    checks.append({"name": "train_only_target_scale", "passed": bool(np.array_equal(scale, expected_scale)),
                   "max_abs_difference": float(np.max(np.abs(scale - expected_scale)))})
    checks.append({"name": "train_only_history_scale", "passed": bool(np.array_equal(hscale, expected_hscale)),
                   "max_abs_difference": float(np.max(np.abs(hscale - expected_hscale)))})
    if not all(row["passed"] for row in checks):
        raise ValueError("saved scale arrays do not recompute from frozen train truth only")
    classes = [module.class_for(path, path[10:], mass) for path in paths]
    coarse_classes = [module.class_for(path, path[10:], mass) for path in coarse]
    difference = float(np.max(np.abs((paths[:, :, :20] - coarse[:, :, :20]) / scale)))
    coverage = all({item["label"] for item in classes[part]} == {"stable", "unstable"}
                   for part in module.ROLES.values())
    truth_pass = bool(coverage and difference <= 1e-6 and np.isfinite(paths).all()
        and all(left["label"] == right["label"] and left["trigger_index"] == right["trigger_index"]
                and (left["trigger_index"] is None or left["trigger_index"] > 10)
                for left, right in zip(classes, coarse_classes, strict=True)))
    expected_truth = {"pass": truth_pass, "class_coverage_pass": coverage, "step_difference": difference,
                      "classes": classes, "coarse_classes": coarse_classes}
    mismatches = []
    compare(safe(truth_data), safe(expected_truth), "truth", mismatches)
    return classes, truth_pass, checks, mismatches


def compare_role(module, group: str, role: str, arrays: dict[str, np.ndarray], paths: np.ndarray,
                 classes: list[dict], scale: np.ndarray, hscale: np.ndarray, mass: np.ndarray,
                 saved: dict):
    oracle = group.endswith("_oracle")
    result = module.describe(arrays, paths, classes, scale, hscale, mass, oracle)
    expected = safe(result)
    actual = {key: saved.get(key) for key in expected}
    mismatches = []
    compare(safe(actual), expected, f"{group}_{role}", mismatches)
    return result, mismatches


def recompute(module, input_arrays, truth_data, role_archives, role_jsons, saved_paired, saved_summary):
    truth_classes, truth_pass, scale_checks, mismatches = load_truth_inputs(module, input_arrays, truth_data)
    scale, hscale, mass = input_arrays["scale"], input_arrays["history_scale"], input_arrays["mass"]
    if not truth_pass:
        if role_archives or role_jsons:
            mismatches.append({"path": "candidate_archives", "reason": "truth_gate_requires_no_candidates"})
        paired = {}
        compare(safe(saved_paired), paired, "paired_differences", mismatches)
        compare(safe(saved_summary.get("paired_differences")), paired, "summary.paired_differences", mismatches)
        expected_summary = {"scientific_pass": False, "truth_pass": False, "numerical_pass": False,
            "matched_recovery_pass": False, "evaluation_complete": False, "records": {},
            "paired_differences": {}, "fit_attempts": 0, "noise_injection": 0.0,
            "t0_allowed": False, "next_route": "REVIEW_TRUTH_GATE_FAILURE"}
        compare(safe({key: saved_summary.get(key) for key in expected_summary}),
                safe(expected_summary), "summary", mismatches)
        return {"truth_pass": False, "scale_checks": scale_checks, "role_checks": [],
                "failure_trace_counts": {}, "mismatch_count": len(mismatches),
                "mismatches": mismatches[:500], "records": {}, "paired_differences": {}}
    records, role_checks, trace_counts = {}, [], {}
    for group in module.GROUPS:
        records[group] = {}
        for role, part in module.ROLES.items():
            key = f"{group}_{role}"
            arrays = role_archives[key]
            count = len(range(*part.indices(64)))
            oracle = group.endswith("_oracle")
            required = role_fields(group)
            if set(arrays) != required:
                raise ValueError(f"{key}: saved forecast archive field set differs")
            if arrays["fine_forecast"].shape != (count, 801, 30) or arrays["refined_forecast"].shape != (count, 801, 30):
                raise ValueError(f"{key}: forecast array shapes differ")
            if not oracle and (arrays["fine_history"].shape != (count, 11, 30)
                               or arrays["refined_history"].shape != (count, 11, 30)):
                raise ValueError(f"{key}: estimated-history array shapes differ")
            if not oracle:
                for step in ("fine", "refined"):
                    expected_shapes = {"iterates": (count, 4, 10), "increments": (count, 3, 10),
                                       "singular": (count, 3, 10), "residuals": (count, 3, 200)}
                    for field, shape in expected_shapes.items():
                        if arrays[f"{step}_{field}"].shape != shape:
                            raise ValueError(f"{key}: {step}_{field} shape differs")
                    forecast, history = arrays[f"{step}_forecast"], arrays[f"{step}_history"]
                    iterates, increments = arrays[f"{step}_iterates"], arrays[f"{step}_increments"]
                    for event_index in range(count):
                        state = input_arrays["paths"][part][event_index]
                        if np.isfinite(forecast[event_index]).all():
                            if not np.array_equal(forecast[event_index, 0, :20], state[10, :20]):
                                mismatches.append({"path": f"{key}.{step}.forecast_origin[{event_index}]", "reason": "reset_state"})
                            if np.isfinite(history[event_index, -1, 20:]).all() and not np.array_equal(
                                    forecast[event_index, 0, 20:], history[event_index, -1, 20:]):
                                mismatches.append({"path": f"{key}.{step}.forecast_hidden_origin[{event_index}]", "reason": "state_mismatch"})
                        if np.isfinite(history[event_index]).all():
                            if not np.array_equal(history[event_index, 0, :20], state[0, :20]):
                                mismatches.append({"path": f"{key}.{step}.history_start[{event_index}]", "reason": "state_mismatch"})
                            if not np.array_equal(history[event_index, 0, 20:], iterates[event_index, -1]):
                                mismatches.append({"path": f"{key}.{step}.history_hidden_start[{event_index}]", "reason": "state_mismatch"})
                        if np.isfinite(iterates[event_index]).all() and np.isfinite(increments[event_index]).all():
                            if not np.array_equal(iterates[event_index, 1:],
                                                  iterates[event_index, :-1] + increments[event_index]):
                                mismatches.append({"path": f"{key}.{step}.iterate_update[{event_index}]", "reason": "arithmetic"})
                            if not np.array_equal(iterates[event_index, 0], np.zeros(10)):
                                mismatches.append({"path": f"{key}.{step}.initial_iterate[{event_index}]", "reason": "state_mismatch"})
                        singular = arrays[f"{step}_singular"][event_index]
                        if np.isfinite(singular).all():
                            tolerance = 64*np.finfo(np.float64).eps*200*singular[:, 0]
                            ranks = np.sum(singular > tolerance[:, None], axis=1)
                            conditions = singular[:, 0]/singular[:, -1]
                            if not np.all(ranks == 10) or not np.all(conditions <= 1e8):
                                mismatches.append({"path": f"{key}.{step}.saved_singular_rank[{event_index}]", "reason": "rank_or_condition"})
            else:
                for step in ("fine", "refined"):
                    forecast = arrays[f"{step}_forecast"]
                    finite = np.isfinite(forecast).all(axis=(1, 2))
                    if not np.array_equal(forecast[finite, 0], input_arrays["paths"][part, 10][finite]):
                        mismatches.append({"path": f"{key}.{step}.forecast_origin", "reason": "oracle_reset_state"})
            result, issues = compare_role(module, group, role, arrays, input_arrays["paths"][part],
                truth_classes[part], scale, hscale, mass, role_jsons[key])
            failure_rows = role_jsons[key].get("failures", [])
            summary_row = saved_summary.get("records", {}).get(group, {}).get(role, {})
            compare(safe(summary_row.get("failures")), safe(failure_rows),
                    f"summary.records.{key}.failures", mismatches)
            records[group][role] = {**result, "failures": failure_rows}
            mismatches.extend(issues)
            trace_counts[key] = len(role_jsons[key].get("failures", []))
            role_checks.append({"role_group": key, "allocated": count,
                                "failure_trace_count": trace_counts[key], "mismatch_count": len(issues),
                                "numerical_pass": result["numerical_pass"],
                                "recovery_pass": result["recovery_pass"]})
    paired = module.paired_differences(records) if truth_pass else {}
    compare(safe(saved_paired), safe(paired), "paired_differences", mismatches)
    compare(safe(saved_summary.get("paired_differences")), safe(paired), "summary.paired_differences", mismatches)
    expected_summary = {
        "scientific_pass": bool(truth_pass and all(row["numerical_pass"] for grp in records.values() for row in grp.values())
                                 and all(records[group][role]["recovery_pass"]
                                         for group in ("matched_gn", "matched_oracle") for role in module.ROLES)),
        "truth_pass": truth_pass,
        "numerical_pass": bool(truth_pass and all(row["numerical_pass"] for grp in records.values() for row in grp.values())),
        "matched_recovery_pass": bool(truth_pass and all(records[group][role]["recovery_pass"]
                                        for group in ("matched_gn", "matched_oracle") for role in module.ROLES)),
        "evaluation_complete": truth_pass,
        "records": safe(records),
        "paired_differences": safe(paired),
        "fit_attempts": 0,
        "noise_injection": 0.0,
        "t0_allowed": False,
        "next_route": "REVIEW_INITIALIZATION_DYNAMICS_SEPARATION",
    }
    actual_summary = {key: saved_summary.get(key) for key in expected_summary}
    compare(safe(actual_summary), safe(expected_summary), "summary", mismatches)
    return {"truth_pass": truth_pass, "scale_checks": scale_checks, "role_checks": role_checks,
            "failure_trace_counts": trace_counts, "mismatch_count": len(mismatches),
            "mismatches": mismatches[:500], "records": records, "paired_differences": paired}


def run(root: Path, run_arg: Path, plan_arg: Path, output_arg: Path):
    plan, entry, run_dir, output, plan_hash, review = authorize(root, plan_arg, run_arg, output_arg)
    reservation = reserve(root, plan, entry, plan_hash, review)
    sys.path.insert(0, str(root / "src"))
    import kron_reduce.evaluation.machine_local_hidden_oracle as module

    truth_bytes = (run_dir / "truth.json").read_bytes()
    truth_data = json.loads(truth_bytes)
    role_names = role_npz_names(module)[1:]
    expected_npz_files = {"inputs.npz", "matched_parameters.npz", "mismatch_parameters.npz"}
    if truth_data.get("pass") is True:
        expected_npz_files.update(role_names)
    manifest_path = run_dir / "artifact_hashes.json"
    raw_manifest = manifest_path.read_bytes()
    manifest = json.loads(raw_manifest)
    actual_npz_files = {name for name in manifest if name.endswith(".npz")}
    if actual_npz_files != expected_npz_files:
        raise RuntimeError("NPZ file set differs from the frozen truth-gate branch")
    npz_names = ["inputs.npz", *role_names] if truth_data.get("pass") is True else ["inputs.npz"]
    manifest, payloads, raw_manifest = verify_artifact_manifest(run_dir, plan, npz_names, truth_bytes)
    input_arrays = decode_once("inputs.npz", payloads["inputs.npz"], INPUT_FIELDS)
    role_archives = {}
    if truth_data.get("pass") is True:
        if len(npz_names) != 13 or len(set(npz_names)) != 13:
            raise RuntimeError("successful truth gate must lead to exactly twelve candidate archives")
        for group in module.GROUPS:
            for role in module.ROLES:
                name = f"{group}_{role}.npz"
                role_archives[f"{group}_{role}"] = decode_once(name, payloads[name], role_fields(group))
    if len(payloads) not in (1, 13):
        raise RuntimeError("saved NPZ decode count must be one after truth failure or thirteen after pass")
    saved_paired = read_json(run_dir / "paired_differences.json")
    saved_summary = read_json(run_dir / "summary.json")
    role_jsons = ({f"{group}_{role}": read_json(run_dir / f"{group}_{role}.json")
                  for group in module.GROUPS for role in module.ROLES}
                  if truth_data.get("pass") is True else {})
    frozen = read_json(run_dir / "frozen_inputs_and_models.json")
    if frozen.get("inputs.npz") != manifest["inputs.npz"]:
        raise RuntimeError("frozen input manifest does not bind the audited inputs.npz")
    for parameter_name in ("matched_parameters.npz", "mismatch_parameters.npz"):
        if frozen.get(parameter_name) != manifest.get(parameter_name):
            raise RuntimeError(f"frozen model manifest does not bind {parameter_name}")
    if truth_data.get("pass") is True:
        for name in module.GROUPS:
            for role in module.ROLES:
                npz_name = f"{name}_{role}.npz"
                if npz_name not in manifest:
                    raise RuntimeError(f"run artifact manifest missing {npz_name}")
    result = recompute(module, input_arrays, truth_data, role_archives, role_jsons, saved_paired, saved_summary)
    report = {"schema": "machine_local_hidden_oracle.saved_audit.v1",
        "authorization": plan["authorization"], "status": "PASS" if result["mismatch_count"] == 0 else "MISMATCH",
        "run_id": plan["run_id"], "artifact_manifest_sha256": sha256_bytes(raw_manifest),
        "npz_decodes": len(payloads), "npz_decode_limit": 13,
        "truth_pass_recomputed": result["truth_pass"], "scale_source_checks": result["scale_checks"],
        "roles": result["role_checks"], "failure_trace_counts": result["failure_trace_counts"],
        "mismatch_count": result["mismatch_count"], "mismatches": result["mismatches"],
        "output_scope": "saved-array arithmetic only; no model, RHS, integration, fit, autodiff, SVD, QR, or linear solve calls",
        "completed_at_utc": datetime.now(UTC).isoformat()}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(safe(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8", newline="\n")
    (reservation / "completed.json").write_text(json.dumps({"run_id": plan["run_id"],
        "result_sha256": sha256_file(output), "npz_decodes": len(payloads),
        "mismatches": result["mismatch_count"], "spent": True, "retry_allowed": False},
        indent=2) + "\n", encoding="utf-8", newline="\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--audit-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run(project_root(), args.run_dir, args.audit_plan, args.output)
    print(json.dumps({"status": report["status"], "npz_decodes": report["npz_decodes"],
                      "mismatches": report["mismatch_count"]}, indent=2))


if __name__ == "__main__":
    main()
