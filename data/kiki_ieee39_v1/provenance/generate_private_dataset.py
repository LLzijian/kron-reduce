"""Small IEEE39 restored-system generator. Run with the installed ANDES Python."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import andes
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kiki.data.private_dataset import check_arrays, los_index
from kiki.physics import coi_project


def physical_parameters(system):
    """Exclude only temporary timed-event devices, not any physical component."""
    values = {}
    for name, model in system.models.items():
        if not model.n or model.group == "TimedEvent":
            continue
        values[name] = {key: np.asarray(param.v).tolist() for key, param in model.params.items()}
    return hashlib.sha256(json.dumps(values, sort_keys=True, default=str).encode()).hexdigest()


def state_mapping(system):
    rows = [None] * system.dae.n
    for name, model in system.models.items():
        for state_name, state in model.states.items():
            for column, device in zip(state.a, model.idx.v, strict=True):
                rows[int(column)] = [int(column), state_name, name, str(device)]
    if any(row is None for row in rows):
        raise ValueError("unmapped_dynamic_state")
    return rows


def _boundary_events(config: dict) -> list[dict]:
    """Validate the explicit TRAIN-only boundary event list.

    Boundary selection is intentionally data/config driven: this helper does
    not search, bisect, or infer events from any split.
    """
    events = [dict(event) for event in config["smoke"]]
    if len(events) != 20:
        raise ValueError(f"boundary stage requires exactly 20 explicit events, got {len(events)}")
    heldout = set(config["validation_buses"]) | set(config["test_buses"])
    overlap = sorted({int(event["bus"]) for event in events} & heldout)
    if overlap:
        raise ValueError(f"boundary events overlap held-out buses: {overlap}")
    physical = [(str(event["line"]), float(event["location"]), int(event["bus"]),
                 str(config["fault_type"]), float(event["xf"]), float(event["duration_s"]))
                for event in events]
    if len(set(physical)) != len(physical):
        raise ValueError("boundary event list contains duplicate physical conditions")
    if sorted(int(event["bus"]) for event in events) != [int(event["bus"]) for event in events]:
        raise ValueError("boundary event list must be sorted by bus")
    return events


def _resolve_solver_workdir(value: str | Path) -> Path:
    candidate = Path(value).resolve()
    allowed = ((ROOT / ".loop-work").resolve(), (ROOT / "private_data").resolve())
    if not any(candidate != root and candidate.is_relative_to(root) for root in allowed):
        raise ValueError("--solver-workdir must be a non-root path under .loop-work or private_data")
    return candidate


def _solver_load_kwargs(solver_workdir: str | Path | None) -> tuple[dict, Path | None]:
    kwargs = {"setup": False, "no_output": True, "default_config": True}
    if solver_workdir is None:
        return kwargs, None
    workdir = Path(solver_workdir)
    kwargs.update({"pycode_path": str(workdir / "andes_pycode"),
                   "output_path": str(workdir / "andes_output"),
                   "no_undill": True, "autogen_stale": False})
    return kwargs, workdir


def simulate(event, config, solver_workdir=None, diagnostics=None):
    """Run one event, optionally filling an observational diagnostics mapping.

    The default path and returned tuple are intentionally unchanged.  The
    diagnostics mapping is populated even when the run later raises, so a
    numerical/physical failure remains an explicit attempted arm.
    """
    capture_diagnostics = diagnostics is not None
    if diagnostics is None:
        diagnostics = {}
    if andes.__version__ != config["andes_version"]:
        raise ValueError("ANDES version differs from frozen config")
    load_kwargs, codegen_workdir = _solver_load_kwargs(solver_workdir)
    ss = andes.load(andes.get_case(config["case"]), **load_kwargs)
    for model in ss.models.values():
        if model.group == "TimedEvent" and model.n:
            model.u.v[:] = [0] * model.n
    if config["disable_pss"]:
        ss.IEEEST.u.v[:] = [0] * ss.IEEEST.n
    uid = ss.Line.idx.v.index(event["line"])
    endpoint = ss.Line.bus1.v[uid] if event["location"] == 0.0 else ss.Line.bus2.v[uid]
    if endpoint != event["bus"] or event["location"] not in (0.0, 1.0):
        raise ValueError("fault_endpoint_mismatch")
    clearing = config["fault_start_s"] + event["duration_s"]
    ss.add("Fault", {"idx": "private_fault", "bus": event["bus"],
                     "tf": config["fault_start_s"], "tc": clearing, "rf": 0., "xf": event["xf"]})
    if codegen_workdir is not None:
        if (codegen_workdir / "andes_pycode" / "__init__.py").is_file():
            ss.codegen._load_calls()
        if not ss.with_calls:
            ss.prepare(quick=True, nomp=True)
    ss.setup()
    if not ss.PFlow.run():
        raise ValueError("powerflow_failure")
    tds = ss.TDS
    tds.config.tstep = config["integration_step_s"]
    tds.config.tol = config["newton_tolerance"]
    tds.config.max_iter = config["newton_max_iterations"]
    tds.config.fixt = 1
    tds.config.shrinkt = 0
    tds.config.criteria = 0
    tds.config.no_tqdm = 1
    tds.config.tf = clearing + config["postfault_horizon_s"] + config["sample_interval_s"]
    tds.init()
    def _residual_info(value):
        values = np.asarray(value, dtype=float)
        return {"finite": bool(np.isfinite(values).all()),
                "max_abs": float(np.max(np.abs(values))) if values.size else 0.0,
                "l2": float(np.linalg.norm(values)) if values.size else 0.0,
                "size": int(values.size)}

    if capture_diagnostics:
        diagnostics.update({
        "initial_f_convention": "stored after TDS.test_init masks no_check_init rows; not an unmasked raw derivative",
        "initial_f": _residual_info(getattr(ss.dae, "f", np.array([]))),
        "initial_g": _residual_info(getattr(ss.dae, "g", np.array([]))),
        "initial_xy_finite": bool(np.isfinite(ss.dae.xy).all()),
        "attempted_steps": 0, "converged_steps": 0, "failed_steps": 0,
        "step_end_times_s": [], "step_sizes_s": [], "scheduled_event_times_s": {
            "fault_start": float(config["fault_start_s"]), "clearing": float(clearing)},
        })
    if not tds.test_ok or not np.isfinite(ss.dae.xy).all():
        raise ValueError("dynamic_initialization_failure")
    inertia = ss.GENROU.M.v.copy()  # ANDES setup converts M to system MVA base.
    equilibrium = {key: getattr(ss.GENROU, variable).v.copy()
                   for key, variable in (("delta", "delta"), ("omega", "omega"), ("Pm", "tm"), ("Pe", "Pe"))}
    equilibrium["full_state"] = ss.dae.x.copy()
    equilibrium["algebraic_state"] = ss.dae.y.copy()
    signature = physical_parameters(ss)
    mapping = state_mapping(ss)
    status = {"failed_step": False, "physical_stop": False, "post_checked": False,
              "pre_checked": False, "post_sample_start": None, "sample_delta": [], "sample_omega": [],
              "previous_time": None, "previous_delta": None, "previous_omega": None,
              "fault_applied": False}
    step = tds.itm_step
    original_do_switch = tds.do_switch

    def checked_step():
        target = float(ss.dae.t)
        if capture_diagnostics:
            diagnostics["attempted_steps"] += 1
        ok = step()
        if capture_diagnostics:
            diagnostics["step_end_times_s"].append(target)
            diagnostics["step_sizes_s"].append(float(getattr(tds, "h", config["integration_step_s"])))
            if config["fault_start_s"] - .02 <= target <= config["fault_start_s"] + 1e-8:
                diagnostics.setdefault("pre_fault_residuals", []).append({
                    "time_s": target, "f": _residual_info(getattr(ss.dae, "f", np.array([]))),
                    "g": _residual_info(getattr(ss.dae, "g", np.array([])))})
            if ok and min(abs(target-config["fault_start_s"]), abs(target-clearing)) <= 2*config["integration_step_s"]+1e-8:
                diagnostics.setdefault("event_neighborhood", []).append({
                    "time_s": target, "fault_active_before_switch": bool(np.any(ss.Fault.uf.v != 0)),
                    "stored_f": _residual_info(ss.dae.f), "stored_g": _residual_info(ss.dae.g)})
        if not ok:
            status["failed_step"] = True
            if capture_diagnostics:
                diagnostics["failed_steps"] += 1
            tds.busted = True
            tds.err_msg = "private dataset rejects any nonconverged integration step"
        else:
            if capture_diagnostics:
                diagnostics["converged_steps"] += 1
        return ok

    def physical_stop():
        now = float(ss.dae.t)
        if not np.isfinite(ss.dae.xy).all():
            status["failed_step"] = True
            tds.busted = True
            return True
        if config["fault_start_s"] + 1e-8 < now < clearing - 1e-8:
            if not np.any(ss.Fault.uf.v != 0):
                raise ValueError("fault_was_not_applied")
            status["fault_applied"] = True
        if config["fault_start_s"] - .02 <= now < config["fault_start_s"] - 1e-8:
            if (np.max(abs(ss.GENROU.omega.v - 1)) > config["pre_fault_speed_tolerance_pu"]
                    or np.max(abs(ss.GENROU.delta.v - equilibrium["delta"])) > config["pre_fault_angle_drift_rad"]):
                raise ValueError("pre_fault_not_steady")
            status["pre_checked"] = True
        if now <= clearing + 1e-8:
            return True
        if not status["post_checked"]:
            if physical_parameters(ss) != signature or np.any(ss.Fault.uf.v != 0):
                raise ValueError("postfault_system_not_restored")
            status["post_checked"] = True
            status["post_sample_start"] = now
        next_sample = status["post_sample_start"] + len(status["sample_delta"]) * config["sample_interval_s"]
        if now >= next_sample - 1e-8:
            previous = status["previous_time"]
            fraction = 1.0 if previous is None else (next_sample-previous)/(now-previous)
            if not -1e-8 <= fraction <= 1+1e-8:
                raise ValueError("LOS_sampling_would_extrapolate")
            for key in ("delta", "omega"):
                current = getattr(ss.GENROU, key).v
                prior = current if previous is None else status[f"previous_{key}"]
                status[f"sample_{key}"].append(prior + fraction*(current-prior))
            hold = config["los_hold_samples"]
            if los_index(status["sample_delta"][-hold:], status["sample_omega"][-hold:], inertia, config) is not None:
                status["physical_stop"] = True
                # Called after a converged, stored step. End normally at this
                # time; never relabel an ANDES numerical failure as instability.
                tds.config.tf = now
        status["previous_time"] = now
        status["previous_delta"] = ss.GENROU.delta.v.copy()
        status["previous_omega"] = ss.GENROU.omega.v.copy()
        return True

    def observed_do_switch():
        if not capture_diagnostics:
            return original_do_switch()
        before = np.asarray(getattr(ss.Fault.uf, "v", np.array([]))).copy()
        before_time = float(ss.dae.t)
        result = original_do_switch()
        after = np.asarray(getattr(ss.Fault.uf, "v", np.array([]))).copy()
        if not np.array_equal(before, after):
            diagnostics.setdefault("switch_callbacks", []).append({
                "time_s": before_time,
                "fault_active_before": bool(np.any(before != 0)),
                "fault_active_after": bool(np.any(after != 0)),
            })
        return result

    tds.itm_step = checked_step
    tds.check_criteria = physical_stop
    if capture_diagnostics:
        tds.do_switch = observed_do_switch
    success = tds.run(no_summary=True)
    endpoints = np.asarray(diagnostics["step_end_times_s"], dtype=float) if capture_diagnostics else np.array([])
    if capture_diagnostics:
      for name, target in diagnostics["scheduled_event_times_s"].items():
        if endpoints.size:
            before = endpoints[endpoints <= target + 1e-8]
            after = endpoints[endpoints >= target - 1e-8]
            diagnostics.setdefault("event_brackets", {})[name] = {
                "target_s": target,
                "present_within_1e-8_s": bool(np.any(np.abs(endpoints - target) <= 1e-8)),
                "before_s": None if before.size == 0 else float(before[-1]),
                "after_s": None if after.size == 0 else float(after[0]),
            }
        else:
            diagnostics.setdefault("event_brackets", {})[name] = {
                "target_s": target, "present_within_1e-8_s": False,
                "before_s": None, "after_s": None,
            }
    if capture_diagnostics and diagnostics["step_sizes_s"]:
        diagnostics["step_size_min_s"] = float(np.min(diagnostics["step_sizes_s"]))
        diagnostics["step_size_max_s"] = float(np.max(diagnostics["step_sizes_s"]))
    if capture_diagnostics:
        diagnostics["solver_final_time_s"] = float(ss.dae.t)
        diagnostics["solver_success"] = bool(success and not tds.busted and not status["failed_step"])
    if not success or tds.busted or status["failed_step"]:
        raise ValueError("numerical_TDS_failure")
    if (not status["pre_checked"] or not status["post_checked"] or not status["fault_applied"]
            or physical_parameters(ss) != signature):
        raise ValueError("missing_baseline_or_restoration_check")
    raw_time = np.asarray(ss.dae.ts.t)
    x, y = np.asarray(ss.dae.ts.x), np.asarray(ss.dae.ts.y)
    if np.any(np.diff(raw_time) <= 0) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("invalid_raw_solver_output")
    start = status["post_sample_start"]
    dt = config["sample_interval_s"]
    count = min(int(np.floor((raw_time[-1]-start)/dt+1e-7))+1,
                round(config["postfault_horizon_s"]/dt)+1)
    grid = start + np.arange(count)*dt
    if grid[0] < raw_time[0] or grid[-1] > raw_time[-1]+1e-8:
        raise ValueError("resampling_would_extrapolate")

    def interpolate(array):
        return np.stack([np.interp(grid, raw_time, array[:, j]) for j in range(array.shape[1])], axis=1)

    full_state, algebraic = interpolate(x), interpolate(y)
    arrays = {"t": np.arange(count)*dt, "full_state": full_state, "algebraic_state": algebraic}
    for key, variable in (("delta", "delta"), ("omega", "omega"), ("Pm", "tm"),
                          ("Pe", "Pe"), ("Qe", "Qe"), ("te", "te")):
        var = getattr(ss.GENROU, variable)
        arrays[key] = full_state[:, var.a] if var.v_code == "x" else algebraic[:, var.a]
    arrays["bus_v"] = algebraic[:, ss.Bus.v.a]
    arrays["bus_a"] = algebraic[:, ss.Bus.a.a]
    arrays["terminal_v"] = np.hypot(algebraic[:, ss.GENROU.vd.a], algebraic[:, ss.GENROU.vq.a])
    check_arrays(arrays, full_state, config)
    trigger = los_index(arrays["delta"], arrays["omega"], inertia, config)
    if status["physical_stop"]:
        if trigger != len(grid)-1:
            raise ValueError("export_LOS_does_not_match_solver_stop")
        label, reason = "unstable", "physical_loss_of_synchronism"
    else:
        tail = max(1, round(config["stable_tail_seconds"]/dt))
        if (trigger is not None or arrays["t"][-1] < config["postfault_horizon_s"]-1e-8
                or np.max(np.ptp(arrays["delta"][-tail:], axis=1)) >= config["los_angle_rad"]
                or np.max(abs(coi_project(arrays["omega"][-tail:], inertia))) >= config["stable_tail_speed_pu"]):
            raise ValueError("unconfirmed_physical_outcome")
        label, reason = "stable", "stable_horizon_reached"
    metadata = {"fault_line": event["line"], "fault_location": event["location"], "fault_bus": event["bus"],
                "fault_type": config["fault_type"], "fault_duration_s": event["duration_s"], "fault_xf": event["xf"],
                "clearing_time_s": clearing, "record_start_delay_s": start-clearing, "label": label,
                "termination_reason": reason, "event_time_s": float(arrays["t"][-1]) if label == "unstable" else "",
                "trajectory_end_s": float(arrays["t"][-1]), "valid_length": len(grid),
                "postfault_signature": signature, "solver_success": True, "failed_steps": 0}
    if capture_diagnostics:
        record_time = float(arrays["t"][-1]) if status["physical_stop"] else None
        diagnostics.update({"label": label, "physical_stop_record_time_s": record_time,
                            "physical_stop_since_clearing_s": None if record_time is None else record_time + float(start-clearing),
                            "physical_stop_absolute_time_s": None if record_time is None else record_time + float(start),
                            "record_start_delay_s": float(start-clearing), "physical_stop": bool(status["physical_stop"])})
    return arrays, metadata, equilibrium, inertia, mapping, ss.dae.y_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "pilot", "expansion", "holdout", "boundary"), default="smoke")
    parser.add_argument("--config", default="configs/private_dataset.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--solver-workdir", default=None,
                        help="optional workspace-local ANDES codegen/output root")
    args = parser.parse_args()
    config = json.loads((ROOT/args.config).read_text())
    output = (ROOT/args.output).resolve()
    if args.solver_workdir is not None:
        if args.workers != 1:
            raise ValueError("--solver-workdir requires --workers 1")
        solver_workdir = _resolve_solver_workdir(args.solver_workdir)
    else:
        solver_workdir = None
    if not output.is_relative_to((ROOT/"private_data").resolve()) or output.exists():
        raise ValueError("output must be a new private_data subdirectory")
    np.random.seed(config["seed"])
    logging.getLogger("andes").setLevel(logging.ERROR)
    events = config["smoke"]
    if args.stage in ("pilot", "expansion", "holdout"):
        ss = andes.load(andes.get_case(config["case"]), setup=False, no_output=True, default_config=True)
        events = []
        for bus in config["pilot_buses"]:
            choices = [(str(line), float(position)) for line, a, b in zip(ss.Line.idx.v, ss.Line.bus1.v, ss.Line.bus2.v)
                       for position, endpoint in enumerate((a,b)) if endpoint == bus]
            line, location = choices[0]
            events.extend({"line": line, "location": location, "bus": bus, "duration_s": duration, "xf": xf}
                          for duration in config["pilot_durations_s"] for xf in config["pilot_fault_reactance_pu"])
    elif args.stage == "boundary":
        events = _boundary_events(config)
    output.mkdir(parents=True)
    rows, rejected = [], Counter()
    start = time.perf_counter()
    pool = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    if pool is not None:
        futures = {pool.submit(simulate,event,config,solver_workdir):(number,event) for number,event in enumerate(events)}
        work = ((futures[f][0], futures[f][1], f) for f in as_completed(futures))
    else:
        work = ((number,event,None) for number,event in enumerate(events))
    with h5py.File(output/"dataset.h5", "x") as file:
        file.attrs["config"] = json.dumps(config)
        file.attrs["stage"] = args.stage
        file.attrs["andes_version"] = andes.__version__
        file.attrs["case_sha256"] = hashlib.sha256(Path(andes.get_case(config["case"])).read_bytes()).hexdigest()
        file.attrs["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        file.attrs["omega_s"] = 2*np.pi*60.
        file.attrs["mva_base"] = 100.
        file.attrs["raw_units"] = "delta rad; omega absolute pu; Pm=GENROU.tm, Pe=GENROU.Pe, te=GENROU.te on system MVA base"
        file.create_group("events")
        for number, event, future in work:
            eid = f"{args.stage}_{number:04d}"
            try:
                result = future.result() if future is not None else simulate(event, config, solver_workdir)
                arrays, metadata, equilibrium, inertia, mapping, ynames = result
                if rows and metadata["postfault_signature"] != file.attrs["postfault_signature"]:
                    raise ValueError("different_postfault_system_between_events")
                if rows:
                    np.testing.assert_allclose(equilibrium["full_state"], file["equilibrium/full_state"][...], atol=1e-9, rtol=0)
                    if json.loads(file.attrs["state_mapping"]) != mapping:
                        raise ValueError("different_state_mapping")
                else:
                    file.attrs["postfault_signature"] = metadata["postfault_signature"]
                    file.attrs["state_mapping"] = json.dumps(mapping)
                    file.attrs["algebraic_names"] = json.dumps(ynames)
                    file.create_dataset("inertia", data=inertia)
                    eq = file.create_group("equilibrium")
                    for key, values in equilibrium.items():
                        eq.create_dataset(key, data=values)
                group = file["events"].create_group(eid)
                for key, values in arrays.items():
                    group.create_dataset(key, data=values, compression="gzip", shuffle=True)
                for key, value in metadata.items():
                    group.attrs[key] = value
                rows.append({"event_id": eid, **metadata})
                print(eid, metadata["label"], metadata["valid_length"], "accepted", flush=True)
            except Exception as exc:  # noqa: BLE001 - discard whole failed simulations
                reason = f"{type(exc).__name__}: {exc}"
                rejected[reason] += 1
                print(eid, "DISCARD", reason, flush=True)
            file.flush()
        file.attrs["discard_counts"] = json.dumps(rejected)
        file.attrs["attempted"] = len(events)
    if pool is not None:
        pool.shutdown()
    rows.sort(key=lambda row: row["event_id"])
    if rows:
        with (output/"dataset_index.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    splits = {role: [] for role in ("train", "validation", "test")}
    for row in rows:
        bus = row["fault_bus"]
        role = "test" if bus in config["test_buses"] else "validation" if bus in config["validation_buses"] else "train"
        splits[role].append(row["event_id"])
    (output/"splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    print("Completed", len(rows), "/", len(events), "discarded", dict(rejected), "seconds", time.perf_counter()-start, flush=True)
    if not rows:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
