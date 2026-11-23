"""One-use arithmetic review of a completed state-port train run.

This program reads only run artifacts and JSON policy/proposal metadata. It does
not call or instantiate models, workflows, RHS functions, integrators, optimizers,
or autodiff. Each run NPZ is decoded at most once with pickle disabled; source
NPZs are never opened.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import numpy as np

from project_harness.report_storage import locate_report

AUTH = "kiki_state_port_train_v1_20261111"
PLAN = "configs/proposals/kiki_state_port_train_v1.json"
SOURCES = "configs/proposals/kiki_state_port_train_sources_v1.json"
PROTOCOL = "docs/protocols/kiki_state_port_train_v1_zh.md"
SCRIPT = "scripts/audit_state_port_train_saved.py"
OUTPUT = "reports/generated/kiki_state_port_train_20261111/saved-audit.json"
SEEDS = (324, 325, 326)
METHODS = ("baseline", "constant", "state")
FIT_METHODS = ("constant", "state")
STEPS = ("0.005", "0.0025")
EXPECTED_NPZ_COUNT = 281
ATOL = 1e-12


class Audit:
    def __init__(self):
        self.checks = []
        self.mismatches = []

    def check(self, name, passed, **details):
        row = {"name": name, "passed": bool(passed), **json_safe(details)}
        self.checks.append(row)
        if not passed:
            self.mismatches.append(row)


def json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_exclusive_json(path: Path, value) -> None:
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def decode_record(run_dir: Path, stem: str, decoded: set[str]):
    """Read one JSON/NPZ pair once and resolve save_record array references."""
    if stem in decoded:
        raise PermissionError(f"run NPZ already decoded: {stem}")
    decoded.add(stem)
    metadata = read_json(run_dir / f"{stem}.json")
    referenced = set()
    declarations = {}

    def resolve(value, route="$" ):
        if isinstance(value, dict):
            if "array" in value:
                key = value["array"]
                if (not isinstance(key, str) or key in declarations
                        or not isinstance(value.get("shape"), list)
                        or not isinstance(value.get("dtype"), str)):
                    raise ValueError(f"{stem}:{route}: invalid or duplicate array reference")
                referenced.add(key)
                declarations[key] = {"shape": value["shape"], "dtype": value["dtype"]}
                return None
            return {key: resolve(item, f"{route}.{key}") for key, item in value.items()}
        if isinstance(value, list):
            return [resolve(item, f"{route}[{index}]") for index, item in enumerate(value)]
        return value

    resolve(metadata)
    npz_path = run_dir / f"{stem}.npz"
    with zipfile.ZipFile(npz_path) as archive:
        names = archive.namelist()
        expected_names = {key + ".npy" for key in referenced}
        if len(names) != len(set(names)) or set(names) != expected_names:
            raise ValueError(f"{stem}: NPZ member roster differs from JSON references")
        for key in referenced:
            with archive.open(key + ".npy") as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _fortran, dtype = np.lib.format.read_array_header_1_0(stream)
                else:
                    shape, _fortran, dtype = np.lib.format.read_array_header_2_0(stream)
            if list(shape) != declarations[key]["shape"] or dtype.str != declarations[key]["dtype"]:
                raise ValueError(f"{stem}:{key}: NPY header differs from JSON declaration")
            if dtype.hasobject or dtype.kind not in "fiu":
                raise ValueError(f"{stem}:{key}: unsafe or nonnumeric dtype")
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    if referenced != set(arrays):
        raise ValueError(f"{stem}: unreferenced NPZ members")
    for key, array in arrays.items():
        if not np.isfinite(array).all():
            raise ValueError(f"{stem}:{key}: nonfinite array")

    def bind(value, route="$" ):
        if isinstance(value, dict):
            if "array" in value:
                return arrays[value["array"]]
            return {key: bind(item, f"{route}.{key}") for key, item in value.items()}
        if isinstance(value, list):
            return [bind(item, f"{route}[{index}]") for index, item in enumerate(value)]
        return value

    resolved = bind(metadata)
    return resolved, arrays


def expected_npz_stems():
    stems = {"common-initial-state", "fixed-targets"}
    for seed in SEEDS:
        for method in FIT_METHODS:
            stems.add(f"{seed}-{method}/final")
            stems.update(f"{seed}-{method}/update-{index:02d}" for index in range(1, 41))
        for method in METHODS:
            stems.update(f"paths-{seed}-{method}-{step}" for step in STEPS)
            stems.add(f"scores-{seed}-{method}")
        for method in FIT_METHODS:
            stems.add(f"numpy-{seed}-{method}")
    return stems


def verify_artifact_manifest(run_dir: Path, audit: Audit):
    manifest = read_json(run_dir / "artifact_hashes.json")
    if not isinstance(manifest, dict):
        raise TypeError("artifact hash manifest must be an object")
    listed, actual = set(manifest), {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*") if path.is_file()
    }
    link_entries = []
    for path in run_dir.rglob("*"):
        if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
            link_entries.append(path.relative_to(run_dir).as_posix())
    file_set_ok = actual == listed | {"artifact_hashes.json"} and not link_entries
    audit.check("run.artifact_file_set", file_set_ok,
                missing=sorted(listed - actual), unlisted=sorted(actual - listed - {"artifact_hashes.json"}),
                links=link_entries)
    mismatches = []
    total_bytes = 0
    for relative, expected in sorted(manifest.items()):
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts or "\\" in relative:
            mismatches.append({"path": relative, "reason": "unsafe_path"})
            continue
        candidate = run_dir.joinpath(*path.parts)
        target = candidate.resolve(strict=False)
        if (target != candidate or run_dir.resolve() not in target.parents or not target.is_file()
                or candidate.is_symlink()):
            mismatches.append({"path": relative, "reason": "missing_or_outside_run"})
            continue
        total_bytes += target.stat().st_size
        actual_hash = sha256(target)
        if actual_hash != expected:
            mismatches.append({"path": relative, "expected": expected, "actual": actual_hash})
    audit.check("run.artifact_sha256", not mismatches, mismatches=mismatches)
    npz_actual = {path.relative_to(run_dir).with_suffix("").as_posix()
                  for path in run_dir.rglob("*.npz")}
    npz_expected = expected_npz_stems()
    audit.check("run.npz_roster", npz_actual == npz_expected and len(npz_actual) == EXPECTED_NPZ_COUNT,
                count=len(npz_actual), missing=sorted(npz_expected - npz_actual),
                unexpected=sorted(npz_actual - npz_expected))
    if (not file_set_ok or mismatches or npz_actual != npz_expected
            or len(npz_actual) != EXPECTED_NPZ_COUNT):
        raise RuntimeError("run artifact manifest or NPZ roster failed before any decode")
    return manifest, total_bytes, npz_expected


def audit_journal(run_dir: Path, plan: dict, audit: Audit, *, expected_scope_count=293):
    journal = run_dir / "operations"
    scope_plan = plan["scope_limits"]
    expected_ids = {name.replace("_", "-") for name in scope_plan}
    reserved_paths = {path.stem.removesuffix("-reserved"): path
                      for path in journal.glob("*-reserved.json")}
    closed_paths = {path.stem.removesuffix("-closed"): path
                    for path in journal.glob("*-closed.json")}
    audit.check("journal.scope_rosters", len(expected_ids) == expected_scope_count
                and set(reserved_paths) == expected_ids and set(closed_paths) == expected_ids,
                expected=expected_scope_count, declared=len(expected_ids),
                reserved=len(reserved_paths), closed=len(closed_paths))
    logical_totals, operator_totals = Counter(), Counter()
    ranges = []
    expected_ops_by_id = {}
    scope_ok = True
    for scope_name, expected in scope_plan.items():
        scope_id = scope_name.replace("_", "-")
        reservation = read_json(reserved_paths[scope_id]) if scope_id in reserved_paths else {}
        closure = read_json(closed_paths[scope_id]) if scope_id in closed_paths else {}
        logical_totals.update(expected.get("logical_limits", {}))
        expected_ops = {key: value for key, value in expected.items() if key in {"probe", "rollouts"}}
        positive_ops = {key: value for key, value in expected_ops.items() if value > 0}
        expected_ops_by_id[scope_id] = positive_ops
        operator_totals.update(expected_ops)
        reserved_ops = reservation.get("reserved_operation_limits", {})
        actual = closure.get("actual_counters", {})
        attempted, returned, failed = (actual.get(k, {}) for k in ("attempted", "returned", "failed"))
        scope_ok &= (reservation.get("scope_id") == scope_id
                     and reservation.get("reservation") == expected
                     and reservation.get("spent") is True
                     and reservation.get("retry_allowed") is False
                     and reserved_ops == expected_ops
                     and closure.get("scope_id") == scope_id
                     and closure.get("status") == "completed_within_budget"
                     and closure.get("durably_closed") is True
                     and attempted == positive_ops and returned == positive_ops and not failed)
        start, end = closure.get("start_byte_offset"), closure.get("end_byte_offset")
        if type(start) is int and type(end) is int:
            ranges.append((start, end, scope_id, closure.get("range_sha256")))
        else:
            scope_ok = False
    audit.check("journal.scope_reservations_and_closures", bool(scope_ok))
    logical_total = {key: value for key, value in logical_totals.items() if value}
    operator_total = {key: value for key, value in operator_totals.items() if value}
    expected_operators = {key: value for key, value in plan["operator_limits"].items() if value}
    audit.check("journal.plan_scope_totals",
                logical_total == plan["logical_operation_limits"]
                and operator_total == expected_operators,
                logical=logical_total, operators=operator_total)

    raw = (journal / "calls.jsonl").read_bytes()
    transitions = {key: Counter() for key in ("attempted", "returned", "failed")}
    by_identity = {}
    for line_number, line in enumerate(raw.splitlines(), start=1):
        row = json.loads(line.decode("utf-8"))
        transition = row["transition"]
        category = row["category"]
        budget_category = "rollouts" if category == "rollout" else category
        transitions[transition][budget_category] += 1
        identity = (row["scope_id"], category, row.get("index"), row.get("tag"))
        by_identity.setdefault(identity, Counter())[transition] += 1
    log_ok = (transitions["attempted"] == transitions["returned"]
              and transitions["attempted"] == Counter(plan["operator_limits"])
              and not transitions["failed"]
              and len(raw.splitlines()) == 2 * sum(plan["operator_limits"].values())
              and all(counts == Counter({"attempted": 1, "returned": 1})
                      for counts in by_identity.values()))
    audit.check("journal.call_transitions", bool(log_ok),
                attempted=dict(transitions["attempted"]), returned=dict(transitions["returned"]),
                failed=dict(transitions["failed"]), identities=len(by_identity))

    ranges.sort()
    range_ok = bool(ranges) and len(ranges) == len(expected_ids)
    cursor = 0
    for start, end, scope_id, expected_hash in ranges:
        payload = raw[start:end]
        range_ok &= (start == cursor and end >= start and len(payload) == end - start
                     and hashlib.sha256(payload).hexdigest() == expected_hash)
        rows = [json.loads(line.decode("utf-8")) for line in payload.splitlines()]
        if expected_ops_by_id.get(scope_id):
            range_ok &= bool(rows) and all(row.get("scope_id") == scope_id for row in rows)
        else:
            range_ok &= not rows and start == end
        cursor = end
    range_ok &= cursor == len(raw)
    audit.check("journal.closed_range_hashes", bool(range_ok), scopes=len(ranges), log_bytes=len(raw))
    return {"scopes": len(expected_ids), "operator_calls": sum(transitions["attempted"].values()),
            "transitions": len(raw.splitlines()), "log_bytes": len(raw)}


def _same(left, right, *, atol=ATOL, rtol=1e-12):
    try:
        return bool(np.allclose(np.asarray(left), np.asarray(right), atol=atol, rtol=rtol,
                                equal_nan=False))
    except (TypeError, ValueError):
        return False


def event_losses(path, truth, counts, scale):
    """Recompute the protocol's equal 1 s/3 s mean per-event normalized MSE."""
    path = np.asarray(path, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    counts = np.asarray(counts)
    scale = np.asarray(scale, dtype=np.float64)
    if (path.ndim != 3 or path.shape[2] < 20 or truth.ndim != 3
            or path.shape[0] != truth.shape[0] or path.shape[0] != len(counts)
            or path.shape[1] < int(counts.max()) + 1 or truth.shape[1] < int(counts.max()) + 1
            or truth.shape[2] != 20 or scale.shape != (20,)):
        raise ValueError("paths, targets, counts and scales are misaligned")
    result = np.empty(len(counts), dtype=np.float64)
    for index, count_value in enumerate(counts):
        count = int(count_value)
        error = (path[index, 1:count + 1, :20] - truth[index, 1:count + 1]) / scale
        point_mse = np.mean(error ** 2, axis=1)
        first = np.mean(point_mse[:min(count, 100)])
        third = np.mean(point_mse[:min(count, 300)])
        result[index] = .5 * (first + third)
    return result


def describe_path(path, truth, counts, scale):
    path = np.asarray(path, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    losses = event_losses(path, truth, counts, scale)
    norm_total = q_total = v_total = 0.0
    point_total = 0
    for index, count_value in enumerate(counts):
        count = int(count_value)
        error = path[index, 1:count + 1, :20] - truth[index, 1:count + 1]
        norm_total += float(np.sum((error / scale) ** 2))
        q_total += float(np.sum(error[:, :10] ** 2))
        v_total += float(np.sum(error[:, 10:20] ** 2))
        point_total += count
    return {
        "event_ell_mean": float(np.mean(losses)),
        "event_ell_median": float(np.median(losses)),
        "event_ell": losses.tolist(),
        "pooled_normalized_mse": norm_total / (20 * point_total),
        "q_rmse_rad": math.sqrt(q_total / (10 * point_total)),
        "v_rmse_rad_per_s": math.sqrt(v_total / (10 * point_total)),
    }


def adam_step_matches(row, previous_m, previous_v, step, *, lr=.01, beta1=.9,
                      beta2=.999, epsilon=1e-8):
    gradient = np.asarray(row["gradient_clipped"], dtype=np.float64)
    expected_m = beta1 * previous_m + (1 - beta1) * gradient
    expected_v = beta2 * previous_v + (1 - beta2) * gradient ** 2
    expected_mhat = expected_m / (1 - beta1 ** step)
    expected_vhat = expected_v / (1 - beta2 ** step)
    expected_theta = row["theta_before"] - lr * expected_mhat / (np.sqrt(expected_vhat) + epsilon)
    adam = row["adam"]
    matches = (_same(adam["exp_avg"], expected_m, atol=2e-12, rtol=2e-11)
               and _same(adam["exp_avg_sq"], expected_v, atol=2e-12, rtol=2e-11)
               and _same(row["theta_after"], expected_theta, atol=2e-12, rtol=2e-11))
    return bool(matches), expected_m, expected_v


def check_physics_record(physics, path, counts, inputs, params):
    mass = params["mass"]
    max_coi = 0.0
    for index, count_value in enumerate(counts):
        count = int(count_value)
        event_path = path[index, :count + 1]
        max_coi = max(max_coi, float(np.max(np.abs(event_path[:, :10] @ mass))),
                      float(np.max(np.abs(event_path[:, 10:20] @ mass))))
    rows = physics.get("points", [])
    if len(rows) != 3 * len(counts):
        return False, {"point_count": len(rows), "max_coi": max_coi}
    expected_points = [(index, point) for index, count in enumerate(counts)
                       for point in (0, int(count) // 2, int(count))]
    actual_points = [(int(row["index"]), int(row["point"])) for row in rows]
    valid = actual_points == expected_points
    max_identity = max_normalized = 0.0
    for row in rows:
        index, point = int(row["index"]), int(row["point"])
        state = path[index, point]
        velocity, latent = state[10:20], state[20:]
        power = inputs["constant_power"][index]
        denominator = max(1.0, abs(float(velocity @ power))
                          + abs(float(velocity @ params["damping"] @ velocity))
                          + abs(float(latent @ params["latent_dissipation"] @ latent)))
        supply = float(velocity @ power - velocity @ params["damping"] @ velocity
                       - latent @ params["latent_dissipation"] @ latent)
        error = abs(float(row["energy_rate"]) - supply)
        normalized = error / denominator
        max_identity = max(max_identity, error)
        max_normalized = max(max_normalized, normalized)
        valid &= math.isfinite(float(row["energy_rate"]))
        valid &= _same(row["supply_minus_dissipation"], supply)
        valid &= _same(row["denominator"], denominator)
        valid &= _same(row["identity_error"], error)
        valid &= _same(row["normalized_error"], normalized)
        valid &= _same(row["hidden_energy"], .5 * float(latent @ latent))
        valid &= math.isfinite(float(row["total_energy"]))
    valid &= max_coi <= 1e-10 and max_identity <= 1e-8
    valid &= _same(physics.get("max_mass_coi"), max_coi, atol=1e-12, rtol=1e-10)
    valid &= physics.get("qualified") is True
    return bool(valid), {"point_count": len(rows), "max_coi": max_coi,
                         "max_identity_error": max_identity,
                         "max_normalized_identity_error": max_normalized}


def audit_records(root: Path, run_dir: Path, plan: dict, frozen_sources: dict, audit: Audit):
    manifest, total_bytes, stems = verify_artifact_manifest(run_dir, audit)
    journal_detail = audit_journal(run_dir, plan, audit)
    decoded = set()

    def record(stem):
        return decode_record(run_dir, stem, decoded)[0]

    common_freeze = read_json(run_dir / "common-initial-state-freeze.json")
    source_access = common_freeze.get("source_access_at_freeze", {})
    phase_expected = plan["source_phase_limits"]["through_common_initial_state_freeze"]
    source_files = frozen_sources["files"]
    a14 = frozen_sources["sources"]["a14"]["run_id"]
    a15 = frozen_sources["sources"]["a15"]["run_id"]
    causal_paths = {
        f"results/{a14}/artifact_hashes.json", f"results/{a14}/allocated_population.json",
        f"results/{a14}/permitted_inputs.npz", f"results/{a15}/artifact_hashes.json",
        f"results/{a15}/allocated_population.json",
    }
    causal_paths.update(path for path in source_files
                        if path.startswith(f"results/{a14}/static-")
                        and path.endswith((".json", ".npz")))
    causal_paths.update(path for path in source_files
                        if path.startswith(f"results/{a14}/seed-")
                        and path.endswith("/history-0.0025.npz"))
    actual_causal = {row.get("path") for row in source_access.get("verified_files", [])}
    causal_members = sorted(
        f"{path}:{member}" for path in causal_paths if source_files[path]["kind"] == "npz"
        for member in source_files[path]["members"]
    )
    no_future_npz = all(a15 not in path or path.endswith(("/artifact_hashes.json", "/allocated_population.json"))
                        for path in actual_causal)
    phase_exact = actual_causal == causal_paths and source_access.get("decoded_members") == causal_members
    audit.check("source.common_freeze_phase", all(source_access.get(k) == v for k, v in phase_expected.items())
                and no_future_npz and phase_exact, source_access={k: source_access.get(k) for k in phase_expected},
                no_A15_payload=no_future_npz)
    source_access_after = read_json(run_dir / "source-access.json")
    limits = plan["source_limits"]
    exact_source_keys = ("files_attempted", "members_attempted", "bytes_read", "logical_bytes")
    source_budget_ok = (all(source_access_after.get(k) == limits[k] for k in exact_source_keys)
                        and source_access_after.get("npy_header_bytes", math.inf)
                        <= limits["npy_header_bytes"])
    audit.check("source.total_budget", source_budget_ok,
                observed={k: source_access_after.get(k) for k in limits}, expected=limits)
    source_inventory = read_json(run_dir / "source-inventory.json")
    expected_reads = {path: spec for path, spec in frozen_sources["files"].items()}
    actual_reads = {row.get("path"): {"sha256": row.get("sha256"), "size_bytes": row.get("size_bytes")}
                    for row in source_access_after.get("verified_files", [])}
    expected_reads = {path: {"sha256": spec["sha256"], "size_bytes": spec["size_bytes"]}
                      for path, spec in expected_reads.items()}
    expected_members = sorted(
        f"{path}:{member}" for path, spec in frozen_sources["files"].items()
        if spec["kind"] == "npz" for member in spec["members"]
    )
    audit.check("source.inventory_binding", source_inventory == frozen_sources
                and actual_reads == expected_reads
                and source_access_after.get("decoded_members") == expected_members,
                file_reads=len(actual_reads), member_reads=len(source_access_after.get("decoded_members", [])))
    pop_path = run_dir / "population.json"
    population = read_json(pop_path)
    audit.check("freeze.population_binding", common_freeze.get("population_sha256") == sha256(pop_path))
    freeze_artifacts = common_freeze.get("artifacts", {})
    common_npz = run_dir / "common-initial-state.npz"
    common_json = run_dir / "common-initial-state.json"
    audit.check("freeze.common_artifacts", freeze_artifacts == {
        "common-initial-state.json": sha256(common_json), "common-initial-state.npz": sha256(common_npz),
    })

    common = record("common-initial-state")
    targets = record("fixed-targets")
    inputs = common["inputs"]
    params = common["parameters"]
    initial = common["initial"]
    truth, counts, weights, omega_s = (targets[k] for k in ("truth", "counts", "weights", "omega_s"))
    rows = population["cache_rows"]
    ids = [row["event_id"] for row in rows]
    input_shapes = (inputs["history"].shape == (258, 11, 20)
                    and inputs["constant_power"].shape == (258, 10)
                    and inputs["equilibrium"].shape == (258, 10)
                    and inputs["mass"].shape == (10,) and inputs["target_scale"].shape == (20,))
    parameter_shapes = True
    for seed in SEEDS:
        model = params[str(seed)]
        parameter_shapes &= (initial[str(seed)].shape == (258, 22)
                             and model["mass"].shape == (10,)
                             and model["edge_stiffness"].shape == (45,)
                             and model["damping"].shape == (10, 10)
                             and model["coupling"].shape == (10, 2)
                             and model["latent_skew"].shape == (2, 2)
                             and model["latent_dissipation"].shape == (2, 2)
                             and np.array_equal(model["mass"], inputs["mass"]))
    audit.check("targets.common_initial_and_model_shapes", input_shapes and parameter_shapes)
    audit.check("targets.population_shapes", len(rows) == 258 and len(population["allocated_slots"]) == 268
                and truth.shape == (258, 301, 20) and counts.shape == (258,)
                and weights.shape == (258,) and omega_s.shape == (258,)
                and _same(counts, plan["future_counts"]) and np.all(weights > 0)
                and abs(float(weights.sum()) - 1) <= 1e-12
                and np.isfinite(truth).all() and np.isfinite(omega_s).all() and np.all(omega_s > 0))
    audit.check("targets.fixed_origin", all(np.array_equal(initial[str(seed)][:, :20], inputs["history"][:, -1])
                and np.array_equal(initial[str(seed)][:, :20], truth[:, 0]) for seed in SEEDS))

    final_records = {
        (str(seed), method): record(f"{seed}-{method}/final")
        for seed in SEEDS for method in FIT_METHODS
    }

    execution_plan = read_json(run_dir / "execution-plan.json")
    audit.check("run.execution_plan", execution_plan == plan)
    groups_summary = read_json(run_dir / "summary.json")
    score_groups = groups_summary.get("groups", {})
    recomputed_groups = {str(seed): {step: {} for step in STEPS} for seed in SEEDS}
    all_path_desc = {}
    all_saved_paths = {}
    selected_paths = {}
    path_checks = 0
    for seed in SEEDS:
        seed_key = str(seed)
        all_path_desc[seed_key] = {step: {} for step in STEPS}
        all_saved_paths[seed_key] = {method: {} for method in METHODS}
        selected_paths[seed_key] = {}
        fixed_params = params[seed_key]
        for method in METHODS:
            selected_paths[seed_key][method] = None
            for step in STEPS:
                stem = f"paths-{seed}-{method}-{step}"
                path_record = record(stem)
                path = path_record["path"]
                all_saved_paths[seed_key][method][step] = path
                saved_counts = path_record["counts"]
                path_valid = (path.shape == (258, 301, 22) and _same(saved_counts, counts)
                              and np.array_equal(path[:, 0], initial[seed_key]))
                max_coi = max(float(np.max(np.abs(path[:, :, :10] @ fixed_params["mass"]))),
                              float(np.max(np.abs(path[:, :, 10:20] @ fixed_params["mass"]))))
                path_valid &= max_coi <= 1e-10
                physics_valid, physics_summary = check_physics_record(
                    path_record["physics"], path, counts, inputs, fixed_params)
                path_valid &= physics_valid
                recomputed = event_losses(path, truth, counts, inputs["target_scale"])
                group = score_groups[seed_key][step][method]
                group_losses = group["event_main_mse"]
                group_valid = len(group_losses) == len(recomputed) and _same(group_losses, recomputed,
                                                                             atol=2e-10, rtol=2e-10)
                path_valid &= group_valid
                desc = describe_path(path, truth, counts, inputs["target_scale"])
                all_path_desc[seed_key][step][method] = desc
                path_checks += 1
                audit.check(f"path.{seed}.{method}.{step}", path_valid, max_coi=max_coi,
                            physics=physics_summary, event_loss_matches_group=group_valid)
                if step == "0.0025":
                    selected_paths[seed_key][method] = path[int(np.flatnonzero(counts == 300)[0])].copy()

    saved_descriptive = read_json(run_dir / "descriptive-scores.json")
    desc_match = compare_tree(saved_descriptive, all_path_desc, atol=2e-10, rtol=2e-10)
    audit.check("scores.descriptive", desc_match)

    line_summary_path = run_dir / "paired-line-summary.json"
    from kron_reduce.evaluation.energy_cap_scores import aggregate_group, score_pair
    from kron_reduce.evaluation.state_port_train_scores import paired_line_summary, verdict

    score_pair_calls, aggregate_calls = 0, 0
    for seed in SEEDS:
        seed_key = str(seed)
        for method in METHODS:
            saved = record(f"scores-{seed}-{method}")
            recomputed_records = {step: [] for step in STEPS}
            refinements = []
            for index, count_value in enumerate(counts):
                count = int(count_value)
                pair = score_pair(
                    history=inputs["history"][index],
                    truth=truth[index, :count + 1],
                    coarse=all_saved_paths[seed_key][method][STEPS[0]][index, :count + 1],
                    fine=all_saved_paths[seed_key][method][STEPS[1]][index, :count + 1],
                    equilibrium=inputs["equilibrium"][index], mass=inputs["mass"],
                    omega_s=float(omega_s[index]), scale=inputs["target_scale"],
                    clock=np.arange(count + 1) * .01,
                )
                score_pair_calls += 1
                refinements.append(pair["refinement"])
                for step in STEPS:
                    recomputed_records[step].append(pair["steps"][step])
            saved_shape = {"steps": saved.get("steps"), "refinements": saved.get("refinements")}
            recomputed_shape = {"steps": recomputed_records, "refinements": refinements}
            pair_ok = compare_tree(saved_shape, recomputed_shape, atol=2e-10, rtol=2e-10)
            audit.check(f"scores.recomputed_pair.{seed}.{method}", pair_ok,
                        event_rows=len(refinements))
            for step in STEPS:
                group = aggregate_group(recomputed_records[step], weights, counts, event_ids=ids)
                aggregate_calls += 1
                recomputed_groups[seed_key][step][method] = group
                saved_group = score_groups[seed_key][step][method]
                audit.check(f"scores.recomputed_aggregate.{seed}.{method}.{step}",
                            compare_tree(saved_group, group, atol=2e-10, rtol=2e-10),
                            event_count=len(recomputed_records[step]))
    audit.check("scores.pure_helper_call_budget", score_pair_calls == 2322 and aggregate_calls == 18,
                score_pair_calls=score_pair_calls, aggregate_calls=aggregate_calls)

    expected_line = paired_line_summary(recomputed_groups, rows)
    saved_line = read_json(line_summary_path)
    audit.check("scores.paired_line_summary", compare_tree(saved_line, expected_line,
                atol=2e-10, rtol=2e-10))

    trace_count, trace_ok = 0, True
    trace_first_events, trace_first_losses = {}, {}
    for seed in SEEDS:
        for method in FIT_METHODS:
            group_key = f"{seed}-{method}"
            previous = None
            previous_m = np.zeros((9, 2))
            previous_v = np.zeros((9, 2))
            for update in range(1, 41):
                row = record(f"{group_key}/update-{update:02d}")
                trace_count += 1
                before, after = row["theta_before"], row["theta_after"]
                grad, clipped = row["gradient_unclipped"], row["gradient_clipped"]
                adam = row["adam"]
                finite = all(np.isfinite(x).all() for x in (before, after, grad, clipped,
                            row["event_loss"], adam["exp_avg"], adam["exp_avg_sq"]))
                norm = float(np.linalg.norm(grad))
                clipnorm = float(np.linalg.norm(clipped))
                clip_ok = clipnorm <= 10 + 1e-10 and _same(norm, row["gradient_norm_unclipped"],
                                                          atol=1e-8, rtol=1e-8)
                clip_scale = min(1.0, 10.0 / (norm + 1e-6))
                clip_ok &= _same(clipped, grad * clip_scale, atol=2e-12, rtol=2e-12)
                step_value = int(np.asarray(adam["step"]).item())
                current_ok = (row["update"] == update and row["objective_parameter_step"] == update - 1
                              and finite and clip_ok and step_value == update
                              and before.shape == (9, 2) and after.shape == (9, 2)
                              and adam["exp_avg"].shape == (9, 2)
                              and adam["exp_avg_sq"].shape == (9, 2)
                              and row["event_loss"].shape == (258,))
                if update == 1:
                    trace_first_events[group_key] = row["event_loss"]
                    trace_first_losses[group_key] = row["loss"]
                    current_ok &= _same(before, np.zeros((9, 2)), atol=0, rtol=0)
                if previous is not None:
                    current_ok &= _same(before, previous, atol=0, rtol=0)
                expected_scalar = float(np.asarray(row["event_loss"]) @ weights)
                current_ok &= _same(row["loss"], expected_scalar, atol=2e-10, rtol=2e-10)
                adam_ok, next_m, next_v = adam_step_matches(row, previous_m, previous_v, update)
                current_ok &= adam_ok
                trace_ok &= bool(current_ok)
                previous = after
                previous_m, previous_v = next_m, next_v
            final = final_records[(str(seed), method)]
            trace_ok &= _same(final["theta"], previous, atol=0, rtol=0)
            scale = float(np.asarray(final["scale"]).item())
            trace_ok &= math.isfinite(scale) and scale > 0
            delta = scale * np.tanh(final["theta"])
            trace_ok &= _same(final["delta"], delta, atol=1e-13, rtol=1e-13)
            expected_scale = float(np.linalg.norm(params[str(seed)]["coupling"]) / math.sqrt(18))
            trace_ok &= _same(scale, expected_scale, atol=1e-12, rtol=1e-12)
    zero_theta_match = True
    for seed in SEEDS:
        left, right = f"{seed}-constant", f"{seed}-state"
        zero_theta_match &= (
            _same(trace_first_events[left], trace_first_events[right], atol=1e-12, rtol=1e-12)
            and _same(trace_first_losses[left], trace_first_losses[right], atol=1e-12, rtol=1e-12)
        )
    audit.check("training.traces", trace_ok and trace_count == 240, updates=trace_count)
    audit.check("training.zero_theta_constant_state", zero_theta_match)

    final_freeze = read_json(run_dir / "final-model-freeze.json")
    expected_final = {}
    for seed in SEEDS:
        for method in FIT_METHODS:
            for name in ("final.json", "final.npz", "update-40.json", "update-40.npz"):
                relative = f"{seed}-{method}/{name}"
                expected_final[relative] = sha256(run_dir / relative)
    audit.check("freeze.final_parameters", final_freeze.get("artifacts") == expected_final
                and final_freeze.get("optimizer_updates") == 240)
    paths_freeze = read_json(run_dir / "future-paths-freeze.json")
    expected_paths = {}
    for seed in SEEDS:
        for method in METHODS:
            for step in STEPS:
                for extension in ("json", "npz"):
                    relative = f"paths-{seed}-{method}-{step}.{extension}"
                    expected_paths[relative] = sha256(run_dir / relative)
    audit.check("freeze.all_paths", paths_freeze.get("artifacts") == expected_paths
                and len(expected_paths) == 36)

    backend_checks = groups_summary.get("backend_checks", [])
    backend_ok = len(backend_checks) == 9
    backend_json_ok = True
    for check in backend_checks:
        seed, method = str(check["seed"]), check["method"]
        backend_json_ok &= compare_tree(read_json(run_dir / f"backend-{seed}-{method}.json"),
                                        check["comparisons"], atol=0, rtol=0)
        comparisons = check["comparisons"]
        if method == "baseline":
            backend_ok &= len(comparisons) == 258
            for index, row in enumerate(comparisons):
                backend_ok &= row["index"] == index and row["qualified"] is True
                backend_ok &= row["visible"] <= 1e-10 and row["hidden"] <= row["hidden_bound"]
        else:
            backend_ok &= len(comparisons) == 1
            row = comparisons[0]
            ref, actual = record(f"numpy-{seed}-{method}"), selected_paths[seed][method]
            ref_path = ref["path"]
            backend_ok &= ref_path.shape == actual.shape == (301, 22)
            backend_ok &= int(np.asarray(ref["index"]).item()) == int(np.flatnonzero(counts == 300)[0])
            bound = 1e-10 * (1 + float(np.max(np.abs(ref_path[:, 20:]))))
            visible = float(np.max(np.abs((actual[:, :20] - ref_path[:, :20]) / inputs["target_scale"])))
            hidden = float(np.max(np.abs(actual[:, 20:] - ref_path[:, 20:])))
            backend_ok &= row["index"] == int(np.asarray(ref["index"]).item())
            backend_ok &= _same(row["visible"], visible, atol=1e-12, rtol=1e-10)
            backend_ok &= _same(row["hidden"], hidden, atol=1e-12, rtol=1e-10)
            backend_ok &= _same(row["hidden_bound"], bound, atol=1e-12, rtol=1e-10)
            backend_ok &= row["qualified"] is (visible <= 1e-10 and hidden <= bound)
    audit.check("backend.numpy_comparisons", bool(backend_ok and backend_json_ok),
                check_groups=len(backend_checks), backend_json_matches=backend_json_ok)

    postrun = plan["postrun_saved_arithmetic_audit"]
    audit.check("scores.plan_budget_binding",
                postrun.get("pure_array_score_pair_calls") == score_pair_calls
                and postrun.get("pure_array_aggregate_group_calls") == aggregate_calls,
                expected_score_pair_calls=postrun.get("pure_array_score_pair_calls"),
                expected_aggregate_calls=postrun.get("pure_array_aggregate_group_calls"))

    calls_expected = plan["logical_operation_limits"]
    operator_expected = plan["operator_limits"]
    operation_ok = (groups_summary.get("logical_operations") == calls_expected
                    and groups_summary.get("logical_returned") == calls_expected
                    and groups_summary.get("operator_counts") == operator_expected
                    and groups_summary.get("operator_returned") == operator_expected)
    audit.check("budget.attempted_returned", operation_ok,
                logical=groups_summary.get("logical_operations"), operators=groups_summary.get("operator_counts"))

    audit.check("summary.process_status", groups_summary.get("process_status") == "PASS")
    stage_evidence = read_json(run_dir / "stage_evidence.json")
    evidence_keys = ("source_commit", "process_status", "data_policy", "scientific_verdict",
                     "promotion_decision", "claim_boundary")
    audit.check("summary.stage_evidence", stage_evidence.get("status") == "PASS"
                and all(stage_evidence.get(key) == groups_summary.get(key) for key in evidence_keys))
    audit.check("run.decoded_npz_once", len(decoded) == EXPECTED_NPZ_COUNT
                and decoded == stems, decoded_count=len(decoded))
    computed_valid = all(check["passed"] for check in audit.checks)
    computed_verdict = verdict(recomputed_groups, valid=computed_valid)
    audit.check("summary.verdict", all(groups_summary.get(key) == value for key, value in computed_verdict.items()
                if key in {"V", "G_baseline", "G_constant", "D_baseline", "D_constant", "scientific_verdict"}),
                expected={key: computed_verdict[key] for key in
                          ("V", "G_baseline", "G_constant", "D_baseline", "D_constant", "scientific_verdict")})
    return {"manifest_files": len(manifest), "manifest_bytes": total_bytes,
            "decoded_npz_count": len(decoded), "trace_updates": trace_count,
            "path_records": path_checks, "score_pair_calls": score_pair_calls,
            "aggregate_group_calls": aggregate_calls,
            "journal": journal_detail,
            "source_access": {k: source_access_after.get(k) for k in limits},
            "scientific_verdict": groups_summary.get("scientific_verdict")}


def compare_tree(left, right, *, atol=1e-12, rtol=1e-12):
    if isinstance(right, dict):
        return isinstance(left, dict) and set(left) == set(right) and all(
            compare_tree(left[key], value, atol=atol, rtol=rtol) for key, value in right.items())
    if isinstance(right, (list, tuple)):
        return isinstance(left, list) and len(left) == len(right) and all(
            compare_tree(a, b, atol=atol, rtol=rtol) for a, b in zip(left, right, strict=True))
    if isinstance(right, (np.ndarray, float, int, np.number)) and not isinstance(right, bool):
        return _same(left, right, atol=atol, rtol=rtol)
    return left == right


def authorize_and_reserve(root: Path):
    root = root.resolve(strict=True)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=root,
                            capture_output=True, text=True, check=True).stdout
    if status.strip():
        raise RuntimeError("saved train audit requires a clean checkout")
    plan_path = root / PLAN
    sources_path = root / SOURCES
    script_path = root / SCRIPT
    plan, sources = read_json(plan_path), read_json(sources_path)
    policy = read_json(root / "configs/research_policy.json")
    approval = policy.get("development_evaluations", {}).get(AUTH, {})
    protocol_path = root / locate_report(root, PROTOCOL, approval.get("protocol_sha256"))
    postrun = plan.get("postrun_saved_arithmetic_audit", {})
    expected = plan.get("scope", {})
    checks = (
        approval.get("status") == "user_approved",
        approval.get("authorization") == AUTH,
        approval.get("plan_path") == PLAN,
        approval.get("plan_sha256") == sha256(plan_path),
        approval.get("protocol_path") == PROTOCOL,
        approval.get("protocol_sha256") == sha256(protocol_path),
        approval.get("sources_path") == SOURCES,
        approval.get("sources_sha256") == sha256(sources_path),
        all(approval.get(k) == v for k, v in expected.items()),
        postrun.get("npz_decode_limit_per_file") == 1,
        postrun.get("model_calls") == postrun.get("rhs_calls") == 0,
        postrun.get("autograd_calls") == postrun.get("optimizer_calls") == 0,
        postrun.get("output_path") == OUTPUT,
        postrun.get("script_path") == SCRIPT,
        postrun.get("script_sha256") == sha256(script_path),
        postrun.get("same_run_json_and_artifact_hash_checks") is True,
        postrun.get("allowed_records") == ["common-initial-state", "fixed-targets", "*-*/update-*",
            "*-*/final", "paths-*", "numpy-*", "scores-*"],
        plan.get("sources_path") == SOURCES,
        plan.get("sources_sha256") == sha256(sources_path),
        plan.get("protocol_path") == PROTOCOL,
        plan.get("protocol_sha256") == sha256(protocol_path),
        sources.get("schema") == "kiki_state_port_train_sources_v1",
    )
    if not all(checks):
        raise RuntimeError("saved train audit plan, approval or source bindings mismatch")
    bindings = approval.get("implementation_bindings", {})
    for relative, expected_hash in bindings.items():
        if sha256(root / locate_report(root, relative, expected_hash)) != expected_hash:
            raise RuntimeError(f"saved train implementation binding mismatch: {relative}")
    if (bindings.get(SCRIPT) != sha256(script_path)
            or bindings.get(SOURCES) != sha256(sources_path)):
        raise RuntimeError("saved train audit script or source manifest is not bound by policy")

    common_git = Path(subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=root,
        capture_output=True, text=True, check=True,
    ).stdout.strip()).resolve(strict=True)
    run_budget = common_git / "state_port_train_budgets" / AUTH
    completion = read_json(run_budget / "completed.json")
    run_id = completion.get("run_id")
    if (not isinstance(run_id, str) or not run_id or "/" in run_id or "\\" in run_id
            or completion.get("status") not in {"complete", "completed"}
            or completion.get("authorization_spent") is not True
            or completion.get("retry_allowed") is not False):
        raise RuntimeError("shared completed marker is not a successful one-use train run")
    source_commit = completion.get("source_commit")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                          capture_output=True, text=True, check=True).stdout.strip()
    if source_commit != head or sources.get("source_commit") not in {None, source_commit}:
        raise RuntimeError("shared completion commit differs from the clean checkout")
    run_dir = (root / "results" / run_id).resolve(strict=True)
    if run_dir.parent != (root / "results").resolve(strict=True):
        raise RuntimeError("completed run path escapes results")
    if completion.get("artifact_manifest_sha256") != sha256(run_dir / "artifact_hashes.json"):
        raise RuntimeError("shared completion does not bind the run artifact manifest")
    if (root / OUTPUT).exists():
        raise FileExistsError(f"saved audit output already exists: {OUTPUT}")

    reservation = common_git / "state_port_train_saved_audit_budgets" / AUTH
    reservation.parent.mkdir(parents=True, exist_ok=True)
    reservation.mkdir(exist_ok=False)
    write_exclusive_json(reservation / "reservation.json", {
        "authorization": AUTH, "run_id": run_id, "source_commit": source_commit,
        "plan_sha256": sha256(plan_path), "sources_sha256": sha256(sources_path),
        "script_sha256": sha256(script_path), "spent": True, "retry_allowed": False,
        "reserved_at_utc": datetime.now(UTC).isoformat(),
    })
    return plan, sources, run_id, run_dir, reservation


def run_audit(root: Path):
    root = root.resolve(strict=True)
    plan, sources, run_id, run_dir, reservation = authorize_and_reserve(root)
    audit = Audit()
    try:
        detail = audit_records(root, run_dir, plan, sources, audit)
        report = {
            "schema": "kron_reduce.state_port_train_saved_audit.v1",
            "authorization": AUTH, "run_id": run_id,
            "source_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                capture_output=True, text=True, check=True).stdout.strip(),
            "process_status": "PASS_PRESERVED",
            "audit_status": "PASS" if not audit.mismatches else "FAIL",
            "scientific_verdict": detail["scientific_verdict"],
            "audited_at_utc": datetime.now(UTC).isoformat(),
            "source_manifest_sha256": plan["sources_sha256"],
            "npz_decode_limit": {"per_file": 1, "count": detail["decoded_npz_count"]},
            "scientific_calls": {"model": 0, "rhs": 0, "autograd": 0, "optimizer": 0},
            "saved_array_rechecks": {"score_pair": detail["score_pair_calls"],
                                      "aggregate_group": detail["aggregate_group_calls"]},
            "details": detail, "check_count": len(audit.checks),
            "mismatch_count": len(audit.mismatches),
            "checks": audit.checks, "mismatches": audit.mismatches,
        }
        report_path = root / OUTPUT
        write_exclusive_json(report_path, report)
        marker = {"run_id": run_id, "status": "complete" if not audit.mismatches else "failed",
                  "authorization_spent": True, "retry_allowed": False, "result_path": OUTPUT,
                  "result_sha256": sha256(report_path), "check_count": len(audit.checks),
                  "mismatch_count": len(audit.mismatches)}
        if audit.mismatches:
            marker["failure_reason"] = "saved evidence arithmetic audit found mismatches"
            write_exclusive_json(reservation / "failed.json", marker)
        else:
            write_exclusive_json(reservation / "completed.json", marker)
        return report
    except BaseException as error:
        try:
            write_exclusive_json(reservation / "failed.json", {
                "run_id": run_id, "status": "failed", "authorization_spent": True,
                "retry_allowed": False, "error_type": type(error).__name__,
                "message": str(error), "completed_checks": len(audit.checks),
                "mismatches": audit.mismatches,
            })
        except OSError as persist_error:
            error.add_note(f"Could not persist failure marker: {persist_error}")
        raise


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    report = run_audit(root)
    print(json.dumps({key: report[key] for key in
                      ("run_id", "process_status", "audit_status", "check_count", "mismatch_count")},
                     ensure_ascii=False))
    return 0 if report["audit_status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
