#!/usr/bin/env python3
"""Read-only arithmetic audit for one completed state-port synthetic run.

Loads only the four registered evidence NPZ archives, once each, with pickle
disabled. It never imports model, workflow, evaluation, integration, or autodiff
code and never modifies the run directory.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

EXPECTED_LOGICAL = {
    "numpy_rhs": 69017,
    "torch_rhs": 9769,
    "jacobian": 5,
    "scalar_backward": 2,
    "model_constructions": 614,
    "rollout_validations": 158,
}
EXPECTED_OPERATORS = {"probe": 3, "rollouts": 162}
EXPECTED_SCOPES = {
    "setup": {"probe": 1},
    "algebra": {"probe": 1},
    "jacobians": {"probe": 1},
    "paths": {"rollouts": 162},
}
NPZ_NAMES = ("inputs", "algebra", "jacobians", "paths")
FD_EPSILONS = ("1e-05", "1e-06")
RUN_ID = "state_port_synthetic_v1-synthetic_correctness-20261110T161605071933Z-992738b0-967f4fb0"
AUTHORITY = "state_port_saved_review_v1_20261111"
PLAN_RELATIVE = Path("configs/proposals/state_port_saved_review_v1.json")
SOURCES_RELATIVE = Path("configs/proposals/state_port_saved_sources_v1.json")
OUTPUT_RELATIVE = Path("reports/generated/state_port_synthetic_20261111/saved-audit.json")


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict] = []
        self.mismatches: list[dict] = []

    def check(self, name: str, ok: bool, **details) -> None:
        item = {"name": name, "passed": bool(ok), **json_safe(details)}
        self.checks.append(item)
        if not ok:
            self.mismatches.append(item)


def json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def maxabs(value) -> float:
    array = np.asarray(value)
    return float(np.max(np.abs(array))) if array.size else 0.0


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_record(run_dir: Path, name: str):
    """Decode one NPZ exactly once; recursively resolve its JSON array refs."""
    metadata = read_json(run_dir / f"{name}.json")
    with np.load(run_dir / f"{name}.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}

    def resolve(value, route="$"):
        if isinstance(value, dict):
            if "array" in value:
                key = value["array"]
                if not isinstance(key, str) or key not in arrays:
                    raise ValueError(f"{name}:{route}: missing NPZ array {key!r}")
                array = arrays[key]
                if value.get("shape") != list(array.shape):
                    raise ValueError(f"{name}:{route}: stored shape does not match")
                if value.get("dtype") != array.dtype.str:
                    raise ValueError(f"{name}:{route}: stored dtype does not match")
                return array
            return {k: resolve(v, f"{route}.{k}") for k, v in value.items()}
        if isinstance(value, list):
            return [resolve(v, f"{route}[{i}]") for i, v in enumerate(value)]
        return value

    return resolve(metadata), arrays


def audit_manifest(root: Path, run_dir: Path, sources: dict, audit: Audit) -> None:
    files = sources.get("files")
    if not isinstance(files, dict):
        raise TypeError("saved sources manifest must map run-relative paths to records")
    listed = set(files)
    actual = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file()
    }
    file_set_ok = listed == actual
    audit.check("manifest_file_set", file_set_ok,
                unlisted=sorted(actual - listed), missing=sorted(listed - actual))
    mismatched = []
    total_bytes = 0
    for relative, record in sorted(files.items()):
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            mismatched.append({"path": relative, "problem": "unsafe_relative_path"})
            continue
        target = (run_dir / relative).resolve()
        if run_dir.resolve() not in target.parents or not target.is_file():
            mismatched.append({"path": relative, "problem": "missing_or_outside_run"})
            continue
        size = target.stat().st_size
        total_bytes += size
        actual_hash = sha256(target)
        if size != record.get("bytes") or actual_hash != record.get("sha256"):
            mismatched.append({"path": relative, "expected_bytes": record.get("bytes"),
                               "actual_bytes": size, "expected_sha256": record.get("sha256"),
                               "actual_sha256": actual_hash})
    audit.check("manifest_sha256", not mismatched, mismatches=mismatched)
    audit.check("manifest_file_totals", len(files) == 22 and total_bytes == 1305816,
                count=len(files), bytes=total_bytes)
    decoded_bytes = sum(files[f"{name}.npz"]["bytes"] for name in NPZ_NAMES)
    decode_budget_ok = decoded_bytes == 1188578
    audit.check("manifest_npz_decode_budget", decode_budget_ok,
                decoded_npz_bytes=decoded_bytes, limit=1188578)
    summary = files.get("summary.json", {})
    summary_ok = summary.get("bytes") == 90 and isinstance(summary.get("sha256"), str)
    audit.check("manifest_summary_raw_preserved", summary_ok,
                bytes=summary.get("bytes"), sha256=summary.get("sha256"))
    identity_ok = (sources.get("run_id") == RUN_ID
                   and sources.get("source_commit") == "992738b0c4bfbf26e073f40a29f67cea4c4ac429"
                   and sources.get("process_status") == "FAIL_PRESERVED")
    audit.check("manifest_run_identity", identity_ok,
                run_id=sources.get("run_id"), source_commit=sources.get("source_commit"))
    if (mismatched or not file_set_ok or not decode_budget_ok or not summary_ok
            or not identity_ok or len(files) != 22 or total_bytes != 1305816):
        raise RuntimeError("saved-source manifest file verification failed; no NPZ was decoded")


def authorize_and_reserve(root: Path) -> tuple[dict, dict, dict, Path]:
    policy = read_json(root / "configs/research_policy.json")
    entry = policy["saved_evidence_audits"][AUTHORITY]
    if (entry.get("status") != "active" or entry.get("max_executions") != 1
            or entry.get("retry_allowed") is not False or entry.get("run_id") != RUN_ID):
        raise RuntimeError("saved-evidence audit policy is not active for the fixed run")
    plan_path = root / PLAN_RELATIVE
    plan = read_json(plan_path)
    plan_sha = sha256(plan_path)
    sources_path = root / SOURCES_RELATIVE
    sources_sha = sha256(sources_path)
    script_path = Path(__file__).resolve(strict=True)
    script_sha = sha256(script_path)
    protocol_path = root / plan["protocol_path"]
    protocol_sha = sha256(protocol_path)
    source_plan_path = root / plan["source_plan_path"]
    source_plan_sha = sha256(source_plan_path)
    review_path = root / entry["review_path"]
    review_sha = sha256(review_path)
    review = read_json(review_path)
    if (entry.get("plan_path") != PLAN_RELATIVE.as_posix()
            or entry.get("plan_sha256") != plan_sha
            or entry.get("review_sha256") != review_sha
            or plan.get("schema") != "kron_reduce.state_port_saved_review.v1"
            or plan.get("authorization") != AUTHORITY
            or plan.get("run_id") != RUN_ID
            or plan.get("source_commit") != "992738b0c4bfbf26e073f40a29f67cea4c4ac429"
            or plan.get("max_npz_decodes") != 4
            or plan.get("max_executions") != 1
            or plan.get("retry_allowed") is not False
            or plan.get("npz_files") != [f"{name}.npz" for name in NPZ_NAMES]
            or plan.get("npz_decode_limit_per_file") != 1
            or plan.get("model_calls") != 0 or plan.get("rhs_calls") != 0
            or plan.get("integration_calls") != 0 or plan.get("autograd_calls") != 0
            or plan.get("expm_calls") != 0 or plan.get("svd_calls") != 0
            or plan.get("ls_calls") != 0 or plan.get("fit_attempts") != 0
            or plan.get("sources_path") != SOURCES_RELATIVE.as_posix()
            or plan.get("sources_sha256") != sources_sha
            or plan.get("script_path") != Path(__file__).resolve().relative_to(root).as_posix()
            or plan.get("script_sha256") != script_sha
            or plan.get("protocol_sha256") != protocol_sha
            or plan.get("source_plan_sha256") != source_plan_sha
            or plan.get("npz_compressed_bytes") != 1188578
            or plan.get("manifest_file_count") != 22
            or plan.get("manifest_total_bytes") != 1305816
            or plan.get("process_status") != "FAIL_PRESERVED"
            or plan.get("output_path") != OUTPUT_RELATIVE.as_posix()
            or review.get("approved") is not True
            or review.get("plan_path") != PLAN_RELATIVE.as_posix()
            or review.get("plan_sha256") != plan_sha):
        raise RuntimeError("saved-evidence plan, script, sources or review binding mismatch")
    sources = read_json(sources_path)
    if (sources.get("run_id") != RUN_ID
            or sources.get("source_commit") != plan.get("source_commit")):
        raise RuntimeError("saved-source manifest identity mismatch")
    common = Path(subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=root, capture_output=True, text=True, check=True,
    ).stdout.strip()).resolve(strict=True)
    budget_root = common / "state_port_saved_review_budgets"
    budget_root.mkdir(parents=True, exist_ok=True)
    reservation = budget_root / AUTHORITY
    reservation.mkdir(exist_ok=False)
    marker = {
        "authorization": AUTHORITY, "run_id": RUN_ID, "source_commit": plan["source_commit"],
        "plan_sha256": plan_sha, "sources_sha256": sources_sha, "script_sha256": script_sha,
        "review_sha256": review_sha, "spent": True, "retry_allowed": False,
        "reserved_at_utc": datetime.now(UTC).isoformat(),
    }
    write_exclusive_json(reservation / "reservation.json", marker)
    return plan, sources, review, reservation


def write_exclusive_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def audit_inputs(data: dict, audit: Audit) -> None:
    nmat = np.asarray(data["N"], dtype=float)
    pmat = np.asarray(data["P"], dtype=float)
    qmat = np.asarray(data["Q"], dtype=float)
    states = data["states"]
    models = data["models"]
    mass = np.asarray(models["baseline"]["mass"], dtype=float)
    ones = np.ones(10)
    checks = {
        "N_shape": nmat.shape == (10, 9),
        "N_orthonormal": maxabs(nmat.T @ nmat - np.eye(9)) <= 1e-12,
        "N_zero_sum": maxabs(ones @ nmat) <= 1e-12,
        "P_shape": pmat.shape == (10, 10),
        "Q_is_P_transpose": qmat.shape == (10, 10) and maxabs(qmat - pmat.T) <= 1e-14,
        "P_idempotent": maxabs(pmat @ pmat - pmat) <= 1e-12,
        "Q_idempotent": maxabs(qmat @ qmat - qmat) <= 1e-12,
        "P_mass_null": maxabs(pmat @ mass) <= 1e-12,
        "Q_one_null": maxabs(qmat @ ones) <= 1e-12,
        "mass_shape": mass.shape == (10,) and bool(np.all(mass > 0)),
    }
    for name, passed in checks.items():
        audit.check(f"inputs.{name}", passed)
    shared_mass = all(maxabs(np.asarray(model["mass"]) - mass) == 0.0
                      for model in models.values())
    audit.check("inputs.shared_fixed_mass", shared_mass)
    for name in ("x1", "x2", "zero"):
        state = np.asarray(states[name], dtype=float)
        coi = max(maxabs(mass @ state[:10]), maxabs(mass @ state[10:20]))
        audit.check(f"inputs.{name}_coi", state.shape == (22,) and coi <= 1e-10,
                    coi_error=coi)
    # Check projection definitions from the frozen mass rather than trusting labels.
    expected_p = np.eye(10) - np.outer(mass, ones) / mass.sum()
    audit.check("inputs.P_definition", maxabs(pmat - expected_p) <= 1e-14,
                max_error=maxabs(pmat - expected_p))


def audit_algebra(data: dict, audit: Audit) -> None:
    checks = data.get("checks", [])
    expected_rows = {(model, state) for model in
                     ("baseline", "constant", "state", "constant0", "state0")
                     for state in ("zero", "x1", "x2")}
    actual_rows = {(row.get("model"), row.get("state")) for row in checks}
    audit.check("algebra.pointwise_roster",
                len(checks) == 15 and actual_rows == expected_rows,
                count=len(checks), missing=sorted(expected_rows - actual_rows),
                extra=sorted(actual_rows - expected_rows))
    audit.check("algebra.record_pass", data.get("pass") is True)
    finite = True
    max_energy = max_power = max_balance = max_shift = max_backend = 0.0
    for row in checks:
        for key, value in row.items():
            if key not in {"model", "state"}:
                finite &= np.isfinite(value)
        max_energy = max(max_energy, float(row.get("energy_residual", 0.0)))
        max_power = max(max_power, float(row.get("power_residual", 0.0)))
        max_balance = max(max_balance, float(row.get("balanced_port", 0.0)))
        max_shift = max(max_shift, float(row.get("common_angle_error", 0.0)))
        max_backend = max(max_backend, float(row.get("backend_rhs_error", 0.0)))
    audit.check("algebra.finite", bool(finite))
    audit.check("algebra.energy_identity", max_energy <= 1e-11, max_error=max_energy)
    audit.check("algebra.reciprocal_power", max_power <= 1e-11, max_error=max_power)
    audit.check("algebra.balanced_port", max_balance <= 1e-11, max_error=max_balance)
    audit.check("algebra.common_angle_invariance", max_shift <= 1e-11, max_error=max_shift)
    audit.check("algebra.numpy_torch_rhs", max_backend <= 1e-11, max_error=max_backend)
    negative = float(data["negative_one_sided_residual"])
    audit.check("algebra.one_sided_negative_control", negative > 1e-10,
                residual=negative)
    samples = data.get("energy_samples", [])
    audit.check("algebra.energy_sample_roster", len(samples) == 15,
                count=len(samples))
    for index, row in enumerate(samples):
        denominator = float(row["denominator"])
        energy_rate = float(row["energy_rate"])
        supply = float(row["supply_minus_dissipation"])
        identity_error = float(row["identity_error"])
        recomputed_error = abs(energy_rate - supply)
        normalized_error = recomputed_error / denominator
        audit.check(f"algebra.energy_sample_{index}",
                    denominator >= 1 and normalized_error <= 1e-11
                    and abs(recomputed_error - identity_error) <= 1e-14,
                    normalized_error=normalized_error,
                    recomputed_error=recomputed_error,
                    saved_identity_error=identity_error)


def audit_jacobians(data: dict, audit: Audit) -> None:
    records = data["records"]
    expected_records = {"constant", "constant0", "state", "state0"}
    audit.check("jacobians.record_groups", set(records) == expected_records,
                saved=sorted(records), expected=sorted(expected_records))
    audit.check("jacobians.record_pass", data.get("pass") is True)
    for record_name, record in records.items():
        automatic = np.asarray(record["automatic"])
        packed = 18 if record_name.endswith("0") else 40
        audit.check(f"jacobians.{record_name}.automatic_shape_finite",
                    automatic.shape == (22, packed) and np.isfinite(automatic).all(),
                    shape=list(automatic.shape))
        for eps in FD_EPSILONS:
            samples = np.asarray(record["finite_difference_samples"][eps])
            audit.check(f"jacobians.{record_name}.{eps}.samples_shape_finite",
                        samples.shape == (packed, 2, 22) and np.isfinite(samples).all(),
                        shape=list(samples.shape))
            recomputed = ((samples[:, 0, :] - samples[:, 1, :]) / (2 * float(eps))).T
            saved = np.asarray(record["finite_difference"][eps])
            audit.check(f"jacobians.{record_name}.{eps}.fd_from_rhs",
                        recomputed.shape == saved.shape and maxabs(recomputed - saved) == 0.0,
                        max_difference=maxabs(recomputed - saved))
            error = maxabs(automatic - recomputed)
            tolerance = 1e-8 + 2e-6 * maxabs(recomputed)
            comparison = record["comparisons"][eps]
            audit.check(f"jacobians.{record_name}.{eps}.autograd",
                        error <= tolerance and bool(comparison["pass"]),
                        error=error, tolerance=tolerance,
                        saved_error=comparison["error"], saved_tolerance=comparison["tolerance"])
    state_fd = np.asarray(records["state"]["finite_difference"]["1e-06"])[:, :22]
    detached = np.asarray(data["negative_detached_jacobian"])
    audit.check("jacobians.detached_shape_finite",
                detached.shape == (22, 22) and np.isfinite(detached).all(),
                shape=list(detached.shape))
    negative_error = maxabs(detached - state_fd)
    negative_tolerance = 1e-8 + 2e-6 * maxabs(state_fd)
    saved_negative = data["negative_detached_port"]
    audit.check("jacobians.detached_negative_control",
                negative_error > 10 * negative_tolerance
                and abs(negative_error - float(saved_negative["error"])) <= 1e-14,
                error=negative_error, tolerance=negative_tolerance,
                ten_tolerance=10 * negative_tolerance)


def audit_paths(data: dict, inputs: dict, audit: Audit) -> None:
    paths = data["paths"]
    checks = data["checks"]
    models = inputs["models"]
    expected_paths = {f"baseline-{state}-{step}" for state in ("x1", "x2")
                      for step in ("0.005", "0.0025")}
    expected_paths |= {f"{model}-x1-0.0025" for model in ("constant0", "state0")}
    expected_paths |= {f"{model}-{state}-{step}" for model in ("constant", "state")
                       for state in ("x1", "x2") for step in ("0.005", "0.0025")}
    expected_paths |= {"constant-torch", "state-torch"}
    audit.check("paths.long_path_roster",
                len(paths) == 16 and set(paths) == expected_paths,
                count=len(paths), missing=sorted(expected_paths - set(paths)),
                extra=sorted(set(paths) - expected_paths))
    audit.check("paths.record_pass", data.get("pass") is True)
    audit.check("paths.check_roster", set(checks) == set(paths),
                missing=sorted(set(paths) - set(checks)),
                extra=sorted(set(checks) - set(paths)))
    for tag, path_value in paths.items():
        path = np.asarray(path_value, dtype=float)
        audit.check(f"paths.{tag}.shape_finite",
                    path.ndim == 2 and path.shape[1] == 22 and np.isfinite(path).all(),
                    shape=list(path.shape))
        mass = np.asarray(models["baseline" if tag.startswith("baseline") else
                                tag.split("-")[0]]["mass"], dtype=float)
        coi = max(maxabs(path[:, :10] @ mass), maxabs(path[:, 10:20] @ mass))
        audit.check(f"paths.{tag}.coi", coi <= 1e-10, coi_error=coi)

    # Every NumPy path stores pointwise energy-rate, supply-minus-dissipation,
    # and its normalization denominator; recompute the normalized maximum.
    for tag, row in checks.items():
        if "energy_samples" not in row:
            continue
        samples = np.asarray(row["energy_samples"], dtype=float)
        normalized = np.abs(samples[:, 0] - samples[:, 1]) / samples[:, 2]
        recomputed = float(np.max(normalized))
        stored = float(row["energy_residual"])
        audit.check(f"paths.{tag}.energy_samples",
                    samples.shape == (301, 3) and np.isfinite(samples).all()
                    and np.all(samples[:, 2] >= 1) and recomputed <= 1e-11
                    and abs(recomputed - stored) <= 1e-14,
                    recomputed_max=recomputed, stored_max=stored)

    recomputed_comparisons = {}
    for kind in ("constant", "state"):
        recomputed_comparisons[kind + "_torch_numpy"] = maxabs(
            paths[kind + "-torch"] - paths[kind + "-x1-0.0025"])
        recomputed_comparisons[kind + "_zero_baseline"] = maxabs(
            paths[kind + "0-x1-0.0025"] - paths["baseline-x1-0.0025"])
    for name in ("baseline", "constant", "state"):
        for state in ("x1", "x2"):
            recomputed_comparisons[name + "_" + state + "_step_refinement"] = maxabs(
                paths[f"{name}-{state}-0.005"] - paths[f"{name}-{state}-0.0025"])
    saved_comparisons = data["comparisons"]
    for name, error in recomputed_comparisons.items():
        tolerance = 1e-12 if name.endswith("zero_baseline") else (
            1e-10 if name.endswith("torch_numpy") else 1e-6)
        saved_error = float(saved_comparisons.get(name, float("nan")))
        audit.check(f"paths.comparison.{name}",
                    error <= tolerance and abs(error - saved_error) <= 1e-14,
                    recomputed=error, saved=saved_error, tolerance=tolerance)

    gradients = data["trajectory_gradients"]
    audit.check("paths.nominal_short_path_roster", set(gradients) == {"constant", "state"},
                saved=sorted(gradients))
    total_fd_paths = 0
    for kind, record in gradients.items():
        nominal = record["nominal"]
        nominal_path = np.asarray(nominal["path"], dtype=float)
        audit.check(f"paths.{kind}.nominal_path_shape_finite",
                    nominal_path.shape == (11, 22) and np.isfinite(nominal_path).all(),
                    shape=list(nominal_path.shape))
        nominal_loss = float(np.mean(nominal_path[:, :20] ** 2))
        audit.check(f"paths.{kind}.nominal_loss",
                    abs(nominal_loss - float(nominal["loss"])) <= 1e-14,
                    recomputed=nominal_loss, saved=nominal["loss"])
        recomputed_fd = {}
        for eps in FD_EPSILONS:
            samples = record["finite_difference_samples"][eps]
            expected_indices = {(i, j) for i in range(9) for j in range(2)}
            actual_indices = [tuple(int(i) for i in sample["index"]) for sample in samples]
            audit.check(f"paths.{kind}.{eps}.fd_index_roster",
                        len(samples) == 18 and len(set(actual_indices)) == 18
                        and set(actual_indices) == expected_indices,
                        count=len(samples), missing=sorted(expected_indices - set(actual_indices)),
                        duplicates=len(actual_indices) - len(set(actual_indices)))
            gradient = np.zeros((9, 2))
            sample_ok = True
            for sample in samples:
                index = tuple(int(i) for i in sample["index"])
                paths_pair = np.asarray(sample["paths"], dtype=float)
                losses = np.asarray(sample["losses"], dtype=float)
                actual_losses = np.asarray([
                    float(np.mean(path[:, :20] ** 2)) for path in paths_pair
                ])
                sample_ok &= paths_pair.shape == (2, 11, 22)
                sample_ok &= maxabs(actual_losses - losses) <= 1e-14
                sample_ok &= np.isfinite(paths_pair).all() and np.isfinite(losses).all()
                total_fd_paths += int(paths_pair.shape[0])
                gradient[index] = (losses[0] - losses[1]) / (2 * float(eps))
            saved_fd = np.asarray(record["finite_difference"][eps])
            fd_error = maxabs(gradient - saved_fd)
            recomputed_fd[eps] = gradient
            audit.check(f"paths.{kind}.{eps}.fd_from_saved_losses",
                        sample_ok and fd_error <= 1e-14,
                        max_gradient_difference=fd_error)
            automatic = np.asarray(record["automatic"])
            grad_error = maxabs(automatic - gradient)
            tolerance = 1e-8 + 2e-6 * maxabs(gradient)
            saved = record["comparisons"][eps]
            audit.check(f"paths.{kind}.{eps}.autograd",
                        grad_error <= tolerance and bool(saved["pass"]),
                        error=grad_error, tolerance=tolerance,
                        saved_error=saved["error"], saved_tolerance=saved["tolerance"])
    audit.check("paths.finite_difference_short_path_count", total_fd_paths == 144,
                count=total_fd_paths, expected=144)


def audit_failure_records(run_dir: Path, audit: Audit) -> None:
    failure = read_json(run_dir / "failure.json")
    attempted_record = read_json(run_dir / "failed-operation.json")
    audit.check("failure.retry_closed", failure.get("retry_allowed") is False)
    audit.check("failure.completed_scopes",
                failure.get("completed_scopes") == ["algebra", "jacobians", "paths"],
                saved=failure.get("completed_scopes"))
    audit.check("failure.logical_operations",
                failure.get("logical_operations") == EXPECTED_LOGICAL,
                saved=failure.get("logical_operations"), expected=EXPECTED_LOGICAL)
    attempted = attempted_record.get("attempted", {})
    returned = attempted_record.get("returned", {})
    audit.check("failure.operation_attempted", attempted == EXPECTED_LOGICAL,
                saved=attempted, expected=EXPECTED_LOGICAL)
    audit.check("failure.operation_returned", returned == EXPECTED_LOGICAL,
                saved=returned, expected=EXPECTED_LOGICAL)
    audit.check("failure.operations_balanced", attempted == returned)


def audit_journal(run_dir: Path, audit: Audit) -> None:
    journal = run_dir / "journal"
    reservations = {}
    for path in journal.glob("*-reserved.json"):
        reservations[path.stem.removesuffix("-reserved")] = read_json(path)
    closes = {}
    for path in journal.glob("*-closed.json"):
        closes[path.stem.removesuffix("-closed")] = read_json(path)
    audit.check("journal.closed_scope_set", set(closes) == set(EXPECTED_SCOPES),
                scopes=sorted(closes))
    audit.check("journal.reservation_scope_set", set(reservations) == set(EXPECTED_SCOPES),
                scopes=sorted(reservations))
    aggregate = {}
    for scope, expected in EXPECTED_SCOPES.items():
        scope_id = scope.replace("_", "-")
        reservation = reservations.get(scope_id)
        if reservation is not None:
            audit.check(f"journal.{scope}.reservation",
                        reservation.get("spent") is True
                        and reservation.get("retry_allowed") is False
                        and reservation.get("reserved_operation_limits") == expected,
                        reservation=reservation.get("reserved_operation_limits"))
        close = closes.get(scope_id)
        if close is None:
            audit.check(f"journal.{scope}.closed", False)
            continue
        actual = close.get("actual_counters", {})
        attempted, returned, failed = (actual.get(k, {}) for k in
                                       ("attempted", "returned", "failed"))
        audit.check(f"journal.{scope}.status",
                    close.get("status") == "completed_within_budget"
                    and close.get("durably_closed") is True)
        audit.check(f"journal.{scope}.attempted_returned",
                    attempted == expected and returned == expected and not failed,
                    attempted=attempted, returned=returned, failed=failed)
        for key, count in attempted.items():
            aggregate[key] = aggregate.get(key, 0) + count
    audit.check("journal.aggregate_operator_counts", aggregate == EXPECTED_OPERATORS,
                saved=aggregate, expected=EXPECTED_OPERATORS)

    raw = (journal / "calls.jsonl").read_bytes()
    entries = [json.loads(line) for line in raw.splitlines()]
    transitions = {"attempted": {}, "returned": {}, "failed": {}}
    by_identity = {}
    for row in entries:
        transition, category = row["transition"], row["category"]
        transitions[transition][category] = transitions[transition].get(category, 0) + 1
        identity = (row["scope_id"], category, row.get("index"), row.get("tag"))
        by_identity.setdefault(identity, Counter())[transition] += 1
    audit.check("journal.transitions_balanced",
                transitions["attempted"] == transitions["returned"]
                and not transitions["failed"]
                and all(counts == Counter({"attempted": 1, "returned": 1})
                        for counts in by_identity.values()),
                transitions=transitions)
    expected_transition_categories = {"probe": 3, "rollout": 162}
    audit.check("journal.transition_counts",
                transitions["attempted"] == expected_transition_categories,
                attempted=transitions["attempted"], expected=expected_transition_categories)
    ranges = []
    for scope_id, close in closes.items():
        start, end = close["start_byte_offset"], close["end_byte_offset"]
        payload = raw[start:end]
        actual_sha = hashlib.sha256(payload).hexdigest()
        rows = [json.loads(line) for line in payload.splitlines()]
        audit.check(f"journal.{scope_id}.range_hash",
                    len(payload) == end - start and actual_sha == close["range_sha256"],
                    start=start, end=end, expected=close["range_sha256"], actual=actual_sha)
        audit.check(f"journal.{scope_id}.range_scope",
                    bool(rows) and all(row.get("scope_id") == scope_id for row in rows),
                    rows=len(rows))
        scope_counts = {state: {} for state in ("attempted", "returned", "failed")}
        for row in rows:
            state, category = row["transition"], row["category"]
            category = "rollouts" if category == "rollout" else category
            scope_counts[state][category] = scope_counts[state].get(category, 0) + 1
        expected_scope = EXPECTED_SCOPES[scope_id.replace("-", "_")]
        audit.check(f"journal.{scope_id}.range_operations",
                    scope_counts["attempted"] == expected_scope
                    and scope_counts["returned"] == expected_scope
                    and not scope_counts["failed"],
                    counts=scope_counts, expected=expected_scope)
        ranges.append((start, end, scope_id))
    ranges.sort()
    cursor = 0
    for start, end, scope_id in ranges:
        audit.check(f"journal.{scope_id}.range_contiguous", start == cursor,
                    expected_start=cursor, actual_start=start)
        cursor = end
    audit.check("journal.ranges_cover_calls", cursor == len(raw),
                covered_bytes=cursor, calls_bytes=len(raw))


def run_audit(root: Path) -> dict:
    root = root.resolve(strict=True)
    plan, sources, _review, reservation = authorize_and_reserve(root)
    run_dir = (root / "results" / RUN_ID).resolve(strict=True)
    if run_dir.parent != (root / "results").resolve(strict=True):
        raise RuntimeError("saved evidence path escaped results/<fixed-run-id>")
    audit = Audit()
    try:
        audit_manifest(root, run_dir, sources, audit)
        records = {}
        for name in NPZ_NAMES:
            records[name], arrays = decode_record(run_dir, name)
            finite_arrays = all(np.issubdtype(value.dtype, np.number)
                                and np.isfinite(value).all() for value in arrays.values())
            audit.check(f"{name}.all_arrays_finite_numeric", finite_arrays,
                        array_count=len(arrays))
        audit_inputs(records["inputs"], audit)
        audit_algebra(records["algebra"], audit)
        audit_jacobians(records["jacobians"], audit)
        audit_paths(records["paths"], records["inputs"], audit)
        audit_failure_records(run_dir, audit)
        audit_journal(run_dir, audit)
        gate_passes = {name: records[name].get("pass") is True
                       for name in ("algebra", "jacobians", "paths")}
        audit.check("saved_records.pass_gates", all(gate_passes.values()), gates=gate_passes)
        output = root / plan["output_path"]
        report = {
            "schema": "kron_reduce.state_port_saved_arithmetic_audit.v1",
            "run_id": RUN_ID,
            "source_commit": plan["source_commit"],
            "process_status": "FAIL_PRESERVED",
            "scientific_checks": ("PASS_FROM_SAVED_EVIDENCE" if not audit.mismatches
                                  else "FAIL_FROM_SAVED_EVIDENCE"),
            "audit_status": "PASS" if not audit.mismatches else "FAIL",
            "audited_at_utc": datetime.now(UTC).isoformat(),
            "saved_sources_manifest_sha256": plan["sources_sha256"],
            "npz_decode_limit": {"per_file": 1, "files": list(NPZ_NAMES)},
            "scientific_calls": {"model": 0, "rhs": 0, "integration": 0, "autograd": 0},
            "check_count": len(audit.checks),
            "mismatch_count": len(audit.mismatches),
            "checks": audit.checks,
            "mismatches": audit.mismatches,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        write_exclusive_json(output, report)
        write_exclusive_json(reservation / "completed.json", {
            "run_id": RUN_ID, "status": "complete", "authorization_spent": True,
            "retry_allowed": False, "scientific_checks": report["scientific_checks"],
            "result_path": plan["output_path"], "result_sha256": sha256(output),
            "check_count": report["check_count"], "mismatch_count": report["mismatch_count"],
        })
        return report
    except BaseException as error:
        try:
            write_exclusive_json(reservation / "failed.json", {
                "run_id": RUN_ID, "status": "failed", "authorization_spent": True,
                "retry_allowed": False, "error_type": type(error).__name__, "message": str(error),
                "completed_checks": len(audit.checks), "mismatches": audit.mismatches,
            })
        except OSError as persist_error:
            error.add_note(f"Could not persist saved-audit failure marker: {persist_error}")
        raise


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    report = run_audit(root)
    print(json.dumps({key: report[key] for key in
                      ("run_id", "process_status", "scientific_checks",
                       "check_count", "mismatch_count")}, ensure_ascii=False))
    return 0 if report["audit_status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

