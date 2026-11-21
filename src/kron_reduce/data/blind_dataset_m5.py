"""Build a truth-only, sealed IEEE39 cohort from one registered candidate list."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np

DATASET = "kiki_ieee39_blind_v2"
HISTORY_COUNTS = {"train": 346, "val": 107, "test": 76}
HISTORY_SOURCES = ("ieee39_postfault", "ieee39_expansion", "ieee39_holdout_supplement")
FRAME_WIDTH = 220
LATE_RANGE_S = (.2, 1.5)
MAX_ATTEMPTS = 210
SIM_FAILURES = {
    "powerflow_failure", "pre_fault_not_steady", "dynamic_initialization_failure", "numerical_TDS_failure",
    "LOS_sampling_would_extrapolate", "invalid_raw_solver_output", "resampling_would_extrapolate",
    "export_LOS_does_not_match_solver_stop", "unconfirmed_physical_outcome",
    "invalid_time_axis", "invalid_sampling_interval", "invalid_full_state", "frozen_or_padded_states",
}


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read(path):
    return json.loads(Path(path).read_bytes())


def _text(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(value)


def _json(path, value):
    _text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _log(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def _duration_key(bus, seconds):
    return int(bus), round(float(seconds) * 1_000_000_000)


def load_history_metadata(source_contract_path):
    """Verify registered CSVs independently of the historical trajectory containers."""
    sources = _read(source_contract_path)["sources"]
    index, hdf_paths = {}, {}
    for source in HISTORY_SOURCES:
        entry = sources["files"][f"index/{source}"]
        path = Path(entry["path"])
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise ValueError(f"historical metadata changed for {source}")
        rows = list(csv.DictReader(io.StringIO(content.decode("utf-8-sig"))))
        index[source] = {row["event_id"]: row for row in rows}
        if len(rows) != len(index[source]):
            raise ValueError(f"duplicate IDs in the historical index {source}")
        hdf_paths[source] = path.with_name("dataset.h5")

    roster = []
    for role, count in HISTORY_COUNTS.items():
        rows = sources["rosters"][role]
        if len(rows) != count:
            raise ValueError(f"unexpected historical {role} roster size")
        for item in rows:
            source, raw_id = item["source_dataset"], item["raw_id"]
            record = index[source][raw_id]
            if (item["id"] != f"{source}/{raw_id}" or not re.fullmatch("[a-z][a-z0-9_]+", raw_id)
                    or record["label"] != item["label"] or record["label"] not in ("stable", "unstable")):
                raise ValueError("historical metadata and source-qualified roster differ")
            roster.append(record | {"id": item["id"], "source": source, "role": role})
    if len({row["id"] for row in roster}) != sum(HISTORY_COUNTS.values()):
        raise ValueError("historical roles are not disjoint")
    configurations = [row for source in HISTORY_SOURCES[:2] for row in index[source].values()]
    if len(configurations) != 591:
        raise ValueError("the two archived configuration indexes must have 591 entries")
    configurations += [r for r in roster if r["source"] == HISTORY_SOURCES[2]]
    return {"roster": roster, "configurations": configurations, "hdf_paths": hdf_paths}


def prepare_candidates(history):
    """Prespecify 210 novel fault conditions using historical metadata only."""
    development = defaultdict(list)
    for row in history["roster"]:
        if row["role"] in ("train", "val"):
            development[int(row["fault_bus"])].append(row)
    locations = {}
    for row in history["configurations"]:
        locations.setdefault(int(row["fault_bus"]), (row["fault_line"], float(row["fault_location"])))
    blocked = {_duration_key(r["fault_bus"], r["fault_duration_s"]) for r in history["configurations"]}
    groups = []
    for bus, records in development.items():
        low_reactance = [r for r in records if float(r["fault_xf"]) == .001]
        stable = [float(r["fault_duration_s"]) for r in low_reactance if r["label"] == "stable"]
        unstable = sorted((float(r["fault_duration_s"]), float(r["event_time_s"]) + float(r["record_start_delay_s"]))
                          for r in low_reactance if r["label"] == "unstable")
        if stable and unstable:
            lo, (hi, los) = max(stable), unstable[0]
            if lo < hi and LATE_RANGE_S[0] <= los <= LATE_RANGE_S[1]:
                groups.append((round((hi - lo) * 1000), abs(los - .7), bus, round(lo * 1000), round(hi * 1000)))
    known = sorted(groups)[:14]
    novel = sorted(set(locations) - development.keys())
    if len(known) != 14 or len(novel) != 7:
        raise ValueError("historical design does not supply 14 transition and seven unseen train/val locations")
    slots = {}
    for _, _, bus, lo, hi in known:
        durations = [lo - offset for offset in (43, 33, 23, 13, 3)] + [lo + round((hi - lo) / 3)]
        durations += [hi + offset for offset in (3, 13, 23, 33)]
        slots[bus] = [(value, (lo, hi)) for value in durations]
    for bus in novel:
        slots[bus] = [(value, None) for value in (71, 111, 161, 211, 281, 341, 411, 471, 541, 591)]
    output = []
    # Alternate unseen buses and transition buses in every pass through the duration grid.
    order = [bus for i, new in enumerate(novel) for bus in (new, known[2*i][2], known[2*i+1][2])]
    for slot in range(10):
        for bus in order:
            millis, bracket = slots[bus][slot]
            while _duration_key(bus, millis / 1000) in blocked:
                millis += 1
            duration = millis / 1000
            blocked.add(_duration_key(bus, duration))
            line, location = locations[bus]
            near = bracket is not None and bracket[0] - 10 <= millis <= bracket[1] + 15
            output.append({"id": f"blind_{len(output)+1:04d}", "bus": bus, "line": line, "location": location,
                           "duration_s": duration, "xf": .001, "unseen_train_val_bus": bus in novel,
                           "near_historical_transition": near})
    return output


def _state_hashes(value):
    row = np.asarray(value, dtype="<f8")
    if row.shape != (FRAME_WIDTH,) or not np.isfinite(row).all():
        raise ValueError("clearing record must contain all 220 finite dynamic states")
    near = np.round(row, decimals=10).astype("<f8", copy=False)
    near[near == 0.] = 0.
    return tuple(hashlib.sha256(array.tobytes(order="C")).hexdigest() for array in (row, near))


def _historical_frames(history, case_sha, signature, mapping, access):
    """Open each shard once and request one row per authorized source-qualified ID."""
    exact, quantized = defaultdict(list), defaultdict(list)
    for source, path in history["hdf_paths"].items():
        items = [r for r in history["roster"] if r["source"] == source]
        binding = history["hdf_bindings"][source]
        _log(access, {"phase": "historical_container_reserved", "source": source, **binding, "rows": len(items)})
        with h5py.File(path, "r") as store:
            if (store.attrs["andes_version"], store.attrs["case_sha256"], store.attrs["postfault_signature"],
                    json.loads(store.attrs["state_mapping"])) != ("2.0.0", case_sha, signature, mapping):
                raise ValueError(f"historical system identity differs for {source}")
            for row in items:
                event = store["events"][row["event_id"]]
                expected = tuple(typ(row[key]) for key, typ in (
                    ("fault_bus", int), ("fault_line", str), ("fault_location", float), ("fault_duration_s", float),
                    ("fault_xf", float), ("label", str)))
                actual = tuple(event.attrs[key] for key in (
                    "fault_bus", "fault_line", "fault_location", "fault_duration_s", "fault_xf", "label"))
                delay = float(event.attrs["record_start_delay_s"])
                values = event["full_state"]
                if (actual != expected or values.shape != (int(row["valid_length"]), FRAME_WIDTH)
                        or values.dtype.kind != "f" or values.dtype.itemsize != 8
                        or delay != float(row["record_start_delay_s"]) or not 0 < delay <= .01):
                    raise ValueError(f"historical clearing-frame metadata differs for {row['id']}")
                scope = {"id": row["id"], "role": row["role"]}
                _log(access, {"phase": "historical_frame_reserved", **scope, "selection": "full_state[0, :]"})
                a, b = _state_hashes(values[0, :])
                exact[a].append(row["id"])
                quantized[b].append(row["id"])
                _log(access, {"phase": "historical_frame_hashed", **scope, "record_start_delay_s": delay,
                              "exact_sha256": a, "round_10_decimals_sha256": b})
    return exact, quantized


def _admission(rows, label, late, bus, novel):
    """Reserve four late-instability slots and stable slots for four unseen buses."""
    totals = Counter(row["label"] for row in rows)
    if label == "stable":
        novel_buses = {row["fault_bus"] for row in rows if row["unseen_train_val_bus"]}
        missing = max(0, 4 - len(novel_buses))
        return totals[label] < 62 and ((novel and bus not in novel_buses) or 62 - totals[label] > missing)
    late_rows = [row for row in rows if row["late_instability"]]
    late_buses = {row["fault_bus"] for row in late_rows}
    missing_late = max(0, 4 - len(late_rows), 3 - len(late_buses))
    gain = late and (bus not in late_buses or len(late_rows) < 4 and len(late_buses) >= 3)
    return totals[label] < 32 and (gain or 32 - totals[label] > missing_late)


def _simulate_call(plan, registry):
    import andes

    config = _read(plan["generator_config"])
    if (andes.__version__, config["andes_version"], config["postfault_horizon_s"], config["sample_interval_s"],
            config["integration_step_s"], config["fault_start_s"]) != ("2.0.0", "2.0.0", 15., .01, .005, 1.):
        raise ValueError("ANDES or the frozen source clock differs")
    case = Path(andes.get_case(config["case"]))
    if _sha(case) != registry["case_sha256"]:
        raise ValueError("ANDES static case does not match the source workbook")
    kiki_src = Path(plan["kiki_src"]).resolve(strict=True)
    sys.path.insert(0, str(kiki_src))
    spec = importlib.util.spec_from_file_location("m5_frozen_source_generator", plan["generator"])
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    for module, relative in (("kiki.data.private_dataset", "data/private_dataset.py"),
                             ("kiki.physics.coordinates", "physics/coordinates.py")):
        if Path(sys.modules[module].__file__).resolve() != kiki_src / "kiki" / relative:
            raise ValueError(f"the frozen generator imported an unregistered helper: {module}")
    logging.getLogger("andes").setLevel(logging.ERROR)
    np.random.seed(config["seed"])
    return generator.simulate, config


def _check_generated(event, arrays, metadata, inertia, mapping, registry):
    label, t = metadata["label"], np.asarray(arrays["t"])
    condition = (metadata["fault_bus"], metadata["fault_line"], float(metadata["fault_location"]),
                 float(metadata["fault_duration_s"]), float(metadata["fault_xf"]))
    if condition != tuple(event[key] for key in ("bus", "line", "location", "duration_s", "xf")):
        raise ValueError("simulator outcome differs from the registered fault condition")
    if (metadata["postfault_signature"] != registry["postfault_signature"]
            or mapping != json.loads(registry["state_mapping"]) or np.shape(inertia) != (10,)
            or not np.isfinite(inertia).all() or np.any(inertia <= 0.)):
        raise ValueError("generated case changed the registered restored system")
    if (label not in ("stable", "unstable") or not metadata["solver_success"] or metadata["failed_steps"] != 0
            or not np.array_equal(t, np.arange(len(t)) * .01)
            or not np.isclose(metadata["record_start_delay_s"], 1e-4, rtol=0, atol=1e-8)
            or len(t) != metadata["valid_length"]
            or np.shape(arrays["full_state"]) != (len(t), FRAME_WIDTH)
            or any(np.shape(arrays[name]) != (len(t), 10) for name in ("delta", "omega", "Pm"))):
        raise ValueError("generated trace violates the frozen simulator output contract")
    if label == "stable" and (len(t) != 1501 or metadata["event_time_s"] != ""
                              or metadata["termination_reason"] != "stable_horizon_reached"):
        raise ValueError("the original 15 s stable outcome must be retained")
    if label == "unstable" and (float(metadata["event_time_s"]) != t[-1]
                                or metadata["termination_reason"] != "physical_loss_of_synchronism"):
        raise ValueError("unstable outcome does not match the original physical stop")


def _attempts(dest, evidence, candidates, history, registry, config, simulate, solver_workdir):
    attempts, selected, primary = Counter(), [], None
    current_exact, current_round = {}, {}
    with ((evidence / "historical_access.jsonl").open("x", encoding="utf-8", newline="\n") as access,
          (evidence / "attempts.jsonl").open("x", encoding="utf-8", newline="\n") as journal,
          h5py.File(dest / "dataset.h5", "x") as store):
        historical, historical_round = _historical_frames(history, registry["case_sha256"],
            registry["postfault_signature"], json.loads(registry["state_mapping"]), access)
        groups = store.create_group("events")
        for event in candidates:
            event_id = event["id"]
            _log(journal, {"phase": "reserved", **event})
            diagnostics = {}
            try:
                arrays, metadata, equilibrium, inertia, state_map, algebraic_names = simulate(
                    event, config, solver_workdir, diagnostics=diagnostics)
                _check_generated(event, arrays, metadata, inertia, state_map, registry)
            except Exception as error:
                reason = str(error)
                _log(journal, {"phase": "returned", "id": event_id, "status": "simulator_failure", "reason": reason,
                               "error_type": type(error).__name__,
                               "converged_steps": diagnostics.get("converged_steps"),
                               "failed_steps": diagnostics.get("failed_steps")})
                attempts["simulator_failure"] += 1
                numeric = isinstance(error, (FloatingPointError, np.linalg.LinAlgError))
                known = isinstance(error, ValueError) and (
                    reason in SIM_FAILURES or re.fullmatch(r"invalid_[a-zA-Z_]+_shape_or_finiteness", reason))
                if not (numeric or known):
                    raise
                continue

            label, delay = metadata["label"], metadata["record_start_delay_s"]
            if primary is None:
                primary = equilibrium, inertia
                store.attrs.update(config=json.dumps(config), dataset_id=DATASET, case_sha256=registry["case_sha256"],
                                   postfault_signature=registry["postfault_signature"], sample_interval_s=.01,
                                   omega_s=2*np.pi*60, mva_base=100., andes_version=config["andes_version"],
                                   state_mapping=json.dumps(state_map),
                                   algebraic_names=json.dumps(algebraic_names), raw_units=registry["raw_units"])
                store.create_dataset("inertia", data=inertia)
                static = store.create_group("equilibrium")
                for key, value in equilibrium.items():
                    static.create_dataset(key, data=value)
            elif (not np.array_equal(primary[1], inertia) or primary[0].keys() != equilibrium.keys()
                  or any(not np.allclose(value, primary[0][key], rtol=0, atol=1e-9)
                         for key, value in equilibrium.items())):
                raise ValueError("generated events differ in the common postfault operating point")
            los = float(metadata["event_time_s"]) + delay if label == "unstable" else None
            late = los is not None and LATE_RANGE_S[0] <= los <= LATE_RANGE_S[1]
            exact, rounded = _state_hashes(arrays["full_state"][0])
            colliding = (historical.get(exact, []) + historical_round.get(rounded, [])
                         + [value for value in (current_exact.get(exact), current_round.get(rounded)) if value])
            status = "accepted"
            if colliding:
                status = "clearing_frame_overlap"
            elif len(arrays["t"]) <= 11:
                status = "insufficient_100ms_window"
            elif not _admission(selected, label, late, event["bus"], event["unseen_train_val_bus"]):
                status = "allocated_quota_full"
            if status == "accepted":
                subgroup = groups.create_group(event_id)
                subgroup.attrs.update(metadata)
                for key, value in arrays.items():
                    subgroup.create_dataset(key, data=value, compression="gzip", shuffle=True)
                store.flush()
                selected.append({"event_id": event_id, **metadata, "los_s_from_clearing": los, "late_instability": late,
                                 "unseen_train_val_bus": event["unseen_train_val_bus"],
                                 "near_historical_transition": event["near_historical_transition"],
                                 "exact_clearing_sha256": exact, "round_10_decimals_clearing_sha256": rounded})
                current_exact[exact], current_round[rounded] = event_id, event_id
            attempts[status] += 1
            _log(journal, {"phase": "returned", "id": event_id, "status": status, "label": label,
                           "los_s_from_clearing": los, "late_instability": late, "record_start_delay_s": delay,
                           "collision_ids": sorted(set(colliding)), "exact_sha256": exact, "round_10_decimals_sha256": rounded})
            if len(selected) == 94:
                break
    return attempts, selected, sum(map(len, historical.values()))


def generate_and_seal(plan, destination, *, solver_workdir, evidence_directory):
    """Run the registered simulator once per attempted condition, then seal the new cohort."""
    bindings = {Path(file).resolve(): digest for file, digest in plan["file_sha256"].items()}
    required = {Path(plan[name]).resolve() for name in ("source_contract", "registry_manifest", "generator", "generator_config")}
    required |= {Path(plan["kiki_src"]).resolve() / "kiki" / path
                 for path in ("data/private_dataset.py", "physics/coordinates.py")}
    if not required <= bindings.keys():
        raise ValueError("the exact generator inputs and imported physics helpers require SHA256 bindings")
    for file, digest in bindings.items():
        if _sha(file) != digest:
            raise ValueError(f"preregistered file changed: {file}")
    history = load_history_metadata(plan["source_contract"])
    shards = _read(plan["registry_manifest"])["shards"]
    registry = shards["ieee39_postfault"]["attributes"]
    history["hdf_bindings"] = plan["history_hdf"]
    if set(history["hdf_bindings"]) != set(HISTORY_SOURCES):
        raise ValueError("the three historical HDF5 containers require explicit registered bindings")
    for source, path in history["hdf_paths"].items():
        bind = history["hdf_bindings"][source]
        if (Path(bind["path"]).resolve() != path.resolve() or bind["bytes"] != path.stat().st_size
                or (source in shards and bind["archive_sha256"] != shards[source]["sha256"]["dataset.h5"])):
            raise ValueError(f"historical HDF5 container differs from its source binding: {source}")
    candidates = plan["candidates"]
    if len(candidates) != MAX_ATTEMPTS or candidates != prepare_candidates(history):
        raise ValueError("candidate list differs from the preregistered deterministic schedule")
    solver_workdir = Path(solver_workdir).resolve()
    if not solver_workdir.is_relative_to((Path.cwd() / "scratch" / "one_off").resolve()):
        raise ValueError("ANDES code generation and outputs must stay in the registered scratch directory")
    solver_workdir.mkdir(parents=True, exist_ok=True)
    simulate, config = _simulate_call(plan, registry)

    dest, evidence = Path(destination), Path(evidence_directory)
    dest.mkdir(parents=True, exist_ok=False)
    attempts, selected, n_history = _attempts(dest, evidence, candidates, history, registry, config, simulate, solver_workdir)
    counts = dict(Counter(row["label"] for row in selected))
    late_rows = [row for row in selected if row["late_instability"]]
    late_buses = sorted({row["fault_bus"] for row in late_rows})
    novel_buses = sorted({row["fault_bus"] for row in selected if row["unseen_train_val_bus"]})
    passed = counts == {"stable": 62, "unstable": 32} and len(late_rows) >= 4 and len(late_buses) >= 3 and len(novel_buses) >= 4
    status = "PASS" if passed else "FAIL"
    qa = {"status": status, "planned_candidates": len(candidates), "attempted": sum(attempts.values()),
          "results": dict(attempts), "accepted_by_label": counts,
          "late_range_s_from_clearing": list(LATE_RANGE_S), "late_instability_count": len(late_rows),
          "late_instability_buses": late_buses, "unseen_train_val_buses": novel_buses,
          "near_historical_transition_count": sum(row["near_historical_transition"] for row in selected),
          "selection_inputs": ["simulator label", "true sustained loss-of-synchronism time", "registered candidate order",
                               "bound duplicate checks and eligibility"], "model_prediction_reads": 0}
    _json(dest / "qa.json", qa)
    _text(dest / "qa_report.md", "# IEEE39 blind-cohort quality audit\n\n"
          f"{status}. The simulator attempted {qa['attempted']} of {len(candidates)} registered conditions. "
          f"The cohort contains {counts.get('stable', 0)} stable and {counts.get('unstable', 0)} unstable outcomes. "
          f"{len(late_rows)} instabilities occurred from 0.2 to 1.5 seconds after clearing across {len(late_buses)} buses. "
          f"{len(novel_buses)} accepted buses were absent from the historical train and val rosters.\n\n"
          f"The first-frame audit covered {n_history} historical trajectories in registered dynamic-state order. "
          "Stable labels use the original 15-second observation. Candidate selection uses the simulator outcomes.\n")
    _json(dest / "disjointness_audit.json", {
        "historical_cohorts": HISTORY_COUNTS, "historical_first_frames": n_history,
        "historical_metadata_configurations": len(history["configurations"]),
        "historical_containers": history["hdf_bindings"],
        "container_identity_check": "Bound path and byte count, archived SHA256 metadata when available, and exact HDF5 attributes.",
        "configuration_key": ["fault_bus", "fault_line", "fault_location", "fault_duration_s"],
        "additional_configuration_key": ["fault_bus", "fault_duration_s"],
        "state_selection": "full_state[0, :] (first sample after the restored-system check)",
        "state_encoding": "220 little-endian float64 values in registered state order",
        "coarse_duplicate_key": "same values rounded to 10 decimal places with canonical signed zero",
        "record_start_delay_s": {"nominal": 1e-4, "absolute_tolerance": 1e-8},
        "accepted_overlap_count": 0, "rejected_state_overlaps": attempts["clearing_frame_overlap"],
        "historical_access_ledger_sha256": _sha(evidence / "historical_access.jsonl"),
    })
    if not passed:
        raise RuntimeError("blind cohort did not reach the registered quotas; see attempt evidence")
    with (dest / "dataset_index.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(selected)
    _json(dest / "splits.json", {"blind": [row["event_id"] for row in selected]})
    _json(dest / "manifest.json", {
        "dataset_id": DATASET, "generation": "truth-only preregistered replay of simulate", "cases": len(selected),
        "labels": counts, "case_sha256": registry["case_sha256"], "restored_system_signature": registry["postfault_signature"],
        "andes_version": config["andes_version"], "sample_interval_s": config["sample_interval_s"],
        "stable_truth_horizon_s": config["postfault_horizon_s"], "model_eval_max_frames": 811,
        "source_contract_sha256": _sha(plan["source_contract"]), "generator_sha256": _sha(plan["generator"]),
        "generator_config_sha256": _sha(plan["generator_config"]), "attempts_sha256": _sha(evidence / "attempts.jsonl"),
        "candidate_plan_sha256": hashlib.sha256(json.dumps(candidates, separators=(",", ":"), sort_keys=True).encode()).hexdigest(),
        "python_executable": sys.executable, "python": sys.version, "files": sorted(p.name for p in dest.iterdir()),
    })
    checksum = dest / "checksums.sha256"
    _text(checksum, "".join(f"{_sha(path)}  {path.name}\n" for path in sorted(dest.iterdir())))
    return {"status": "PASS", "events": len(selected), "labels": counts, "late_instability_count": len(late_rows),
            "attempts": sum(attempts.values()), "manifest_sha256": _sha(dest / "manifest.json"), "checksums_sha256": _sha(checksum)}
