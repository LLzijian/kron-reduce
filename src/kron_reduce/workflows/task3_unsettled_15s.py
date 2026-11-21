"""Extend previously unsettled frozen TASK3 forecasts without fitting or new truth reads."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import tomllib
from collections import Counter
from contextlib import closing
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from kron_reduce.data.durable_audit_ledger import AppendOnlyAuditJournal
from kron_reduce.physics.coordinates import coi_project, coi_project_power
from project_harness.evidence import sha256_file
from project_harness.git_state import assert_clean, git_head, verify_bound_file
from project_harness.loop import approve, workflow_lock
from project_harness.manifest import load_manifest, project_root
from project_harness.policy import load_policy, validate_run_policy
from project_harness.runner import _run_experiment

WORKFLOW = "task3_unsettled_15s_v1"
STAGE = "reevaluate"
STEPS = (.0025, .00125)
ZERO = {"fits": 0, "gradients": 0, "optimizer_updates": 0}


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_hash(path, digest, journal=None, **fields):
    if journal is not None:
        journal.append({"action": "verify_file", "path": str(path), "sha256": digest, **fields})
    require(path.is_file() and sha256_file(path) == digest, f"Frozen SHA256 mismatch: {path}")


def history_member(archive, member):
    """Decode eleven rows from a C-order npy member in a sealed cache."""
    with archive.open(member + ".npy") as stream:
        version = np.lib.format.read_magic(stream)
        header = {(1, 0): np.lib.format.read_array_header_1_0,
                  (2, 0): np.lib.format.read_array_header_2_0}[version]
        shape, fortran, dtype = header(stream)
        require(len(shape) == 2 and shape[0] >= 11 and shape[1] == 20 and not fortran
                and dtype.kind == "f", f"Invalid history member: {member}")
        return np.frombuffer(stream.read(11 * 20 * dtype.itemsize), dtype=dtype).reshape(11, 20).astype(float)


def cached_events(plan, static, access):
    def event(item, archive, prefix, role):
        access.append({"action": "read_history", "role": role, "id": item["id"],
                       "members": [prefix + "x[0:11]", prefix + "p0"]})
        with archive.open(prefix + "p0.npy") as stream:
            power = np.lib.format.read_array(stream, allow_pickle=False)
        return item | {"x": history_member(archive, prefix + "x"), "p0": power,
                       "mass": static["mass"], "equilibrium_delta": static["delta_star"],
                       "omega_s": float(static["omega_s"])}

    events = {}
    binding = plan["sources"]["development"]
    path = Path(binding["path"])
    check_hash(path, binding["sha256"], access, role="val")
    with ZipFile(path) as cache:
        for item in plan["events"]["val"]:
            events["val", item["id"]] = event(item, cache, item["raw_id"] + "_", "val")
    for item in plan["events"]["test"]:
        binding = plan["sources"][f"test/{item['raw_id']}"]
        path = Path(binding["path"])
        check_hash(path, binding["sha256"], access, role="test")
        with ZipFile(path) as cache:
            events["test", item["id"]] = event(item, cache, "", "test")
    return events


def blind_events(plan, static, access):
    import h5py

    manifest = read(Path(plan["blind_manifest"]["path"]))
    path = Path(plan["blind_hdf5"]["path"])
    check_hash(path, plan["blind_hdf5"]["sha256"], access, role="blind_v2")
    access.append({"action": "open_sealed_container", "role": "blind_v2", "path": str(path)})
    with h5py.File(path, "r") as store:
        require(store.attrs["dataset_id"] == manifest["dataset_id"] == "kiki_ieee39_blind_v2"
                and store.attrs["case_sha256"] == manifest["case_sha256"]
                and store.attrs["postfault_signature"] == manifest["restored_system_signature"]
                and store.attrs["sample_interval_s"] == .01, "The blind container identity changed")
        frequency = float(store.attrs["omega_s"])
        mass = store["inertia"][...] / frequency
        equilibrium = store["equilibrium/delta"][...]
        power_star = store["equilibrium/Pm"][...]
        for actual, key in ((frequency, "omega_s"), (mass, "mass"),
                            (equilibrium, "delta_star"), (power_star, "pm_star")):
            np.testing.assert_allclose(actual, static[key], rtol=1e-7, atol=1e-9)
        result = {}
        for item in plan["events"]["blind_v2"]:
            access.append({"action": "read_history", "role": "blind_v2", "id": item["id"],
                           "members": ["t[0:11]", "delta[0:11]", "omega[0:11]", "Pm[10]"]})
            source = store["events"][item["raw_id"]]
            require(source.attrs["label"] == item["label"]
                    and np.array_equal(source["t"][:11], np.arange(11) * .01),
                    f"Blind input does not match seal: {item['id']}")
            angle = coi_project(source["delta"][:11] - equilibrium, mass)
            speed = (source["omega"][:11] - 1.) * frequency
            result["blind_v2", item["id"]] = item | {"x": np.c_[angle, speed],
                "p0": source["Pm"][10] - power_star, "mass": mass,
                "equilibrium_delta": equilibrium, "omega_s": frequency}
    return result


def kron_forecaster(model, name, scales, prior, budget):
    from kron_reduce.models.first_swing_fit import predict_first_swing_fit
    from kron_reduce.models.unified_discovery_m3 import predict_m3

    def forecast(events, seconds, step):
        q = np.stack([coi_project(event["x"][:, :10], event["mass"]) for event in events])
        v = np.stack([coi_project(event["x"][:, 10:], event["mass"]) for event in events])
        p0 = np.stack([coi_project_power(event["p0"], event["mass"]) for event in events])
        eq = np.stack([coi_project(event["equilibrium_delta"], event["mass"]) for event in events])
        inputs = (model, q, v, p0, eq, scales["history"], seconds * 100)
        kwargs = {"max_step": step, "memory_radius": scales["memory_radius"],
                  "prior_weight": prior, "spend": budget.spend}
        value = (predict_m3(*inputs, **kwargs) if name == "m3b_g2"
                 else predict_first_swing_fit(*inputs, **kwargs, soft_radius=name == "soft_2r"))
        return value["forecast"][:, :, :20].cpu().numpy()

    return forecast


def classify(paths, event, steps, seconds, *, match_settlement):
    from task3.c2.evaluation import numerical
    from task3.protocol import classify as classify_path

    n = seconds * 100
    paths = [path[:n + 1] for path in paths]
    require(all(path.shape == (n + 1, 20) for path in paths), "Forecast grid is incomplete")
    outcomes = [classify_path(path, event) for path in paths]
    first = outcomes[0]
    categories = [row["category"] for row in outcomes]
    valid = all(category != "numerical_unresolved" for category in categories)
    match = len(set(categories)) == 1
    hits = [row["trigger_index"] for row in outcomes]
    comparison = None
    if len(paths) == 2:
        both_los = all(hit is not None for hit in hits)
        end_frame = max(hits) - 10 if both_los else n
        comparison = numerical(*paths, event, end_frame)
        valid = (valid and comparison["passed"] and (both_los or all(hit is None for hit in hits))
                 and (not both_los or max(hits) - min(hits) <= 2) and (match or not match_settlement))
    return {"status": first["category"] if valid else "numerical_unresolved", "qualified": valid,
            "predicted_los_time_forecast_s": round((hits[0] - 10) * .01, 2) if valid and hits[0] is not None else None,
            "per_step": [{"step_s": step, "category": row["category"], "trigger_index": row["trigger_index"],
                          "finite_through_window": bool(np.isfinite(path).all())}
                         for step, row, path in zip(steps, outcomes, paths, strict=True)],
            "numerical_comparison": comparison}


def archived_forecasts(binding, events, access):
    path = Path(binding["path"])
    access.append({"action": "read_archived_predictions", "path": str(path),
                   "members": [binding["ids"], binding["forecast"]], "ids": [event["id"] for event in events]})
    with np.load(path, allow_pickle=False) as data:
        ids = [str(name).split("/")[-1] for name in data[binding["ids"]]]
        index = {name: i for i, name in enumerate(ids)}
        require(len(index) == len(ids), f"Duplicate archived forecasts: {path}")
        values = data[binding["forecast"]][[index[event["raw_id"]] for event in events]]
    require(values.shape == (len(events), 801, 20), f"Unexpected archived forecast grid: {path}")
    return values


def forecast_asset(group):
    return f"{group['release']}--{group['forecast_model']}--{group['role']}.npz"


def execute(root, directory, plan):
    import torch

    from kron_reduce.evaluation.machine_local_synthetic import Budget
    from kron_reduce.evaluation.task3_frozen_15s import load_forecaster
    from kron_reduce.workflows.postfault_m4_v2 import frozen_models

    torch.set_num_threads(1)
    if not torch.cuda.is_available() or os.getenv("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("The frozen structure models require deterministic CUDA float64")
    torch.use_deterministic_algorithms(True)
    asset_root, task3 = Path(plan["asset_root"]), Path(plan["task3_root"])
    sys.path.insert(0, str(task3 / "src"))
    from task3.c2.evaluation import numerical

    with (closing(AppendOnlyAuditJournal(directory / "access.jsonl")) as access,
          closing(AppendOnlyAuditJournal(directory / "operations.jsonl")) as operations):
        budget = Budget(directory / "kron_operations.jsonl", plan["kron_operations"])
        with closing(budget.stream), torch.no_grad():
            # All checkpoint and code identities are checked before any history decode.
            for binding in plan["frozen_files"]:
                check_hash(Path(binding["path"]), binding["sha256"], access, role="frozen_asset")
            m5 = tomllib.loads((root / "configs/experiments/blind_confirmation_m5_v1.toml").read_text(encoding="utf-8"))
            _, models, scales, _ = frozen_models(asset_root, m5, device="cuda")
            model_names = {"hard_r": "hard_r", "soft_2r": "soft_2r", "m3b_g2": "m3"}
            forecasters = {
                ("kron", name): kron_forecaster(models[key].requires_grad_(False).eval(), name, scales,
                                               m5["prior_weight"], budget)
                for name, key in model_names.items()}
            static_path = Path(plan["sources"]["static"]["path"])
            check_hash(static_path, plan["sources"]["static"]["sha256"], access, role="static")
            with np.load(static_path, allow_pickle=False) as arrays:
                static = dict(arrays)
            for model in models.values():
                np.testing.assert_allclose(static["mass"], model.mass.cpu().numpy(), rtol=1e-12, atol=1e-12)
            for group in plan["groups"]:
                key = group["release"], group["forecast_model"]
                if key not in forecasters and not group.get("reuse"):
                    forecasters[key] = load_forecaster(task3, *key, static)
            all_events = cached_events(plan, static, access) | blind_events(plan, static, access)
            require(len(all_events) == sum(map(len, plan["events"].values())), "Duplicate input histories")
            for compound, event in all_events.items():
                require(event["x"].shape == (11, 20) and event["p0"].shape == (10,)
                        and np.isfinite(event["x"]).all() and np.isfinite(event["p0"]).all(),
                        f"Invalid causal inputs: {compound}")
            references = {forecast_asset(group): archived_forecasts(group["archive"],
                          [all_events[group["role"], name] for name in group["ids"]], access)
                          for group in plan["groups"]}
            rows, attempts, reused = [], 0, 0
            for group in plan["groups"]:
                release, role, model, actual = (group[key] for key in ("release", "role", "model", "forecast_model"))
                events = [all_events[role, name] for name in group["ids"]]
                steps = group["steps_s"]
                args = {"release": release, "model": model, "role": role, "forecast_model": actual}
                artifact = forecast_asset(group)
                if binding := group.get("reuse"):
                    path = Path(binding["path"])
                    access.append({"action": "reuse_completed_15s_forecasts", "path": str(path), **args})
                    with np.load(path, allow_pickle=False) as prior:
                        require(prior["ids"].tolist() == group["ids"] and prior["steps_s"].tolist() == steps,
                                f"The recovered result does not match the registered roster: {path}")
                        paths = prior["forecasts"]
                    require(paths.shape == (len(steps), len(events), 1501, 20), "Recovered forecast shape changed")
                    shutil.copyfile(path, directory / artifact)
                    reused += len(events) * len(steps)
                    operations.append({"action": "forecast_reused", **args, "ids": group["ids"], "steps_s": steps})
                else:
                    paths = []
                    keys = ("id", "x", "p0", "mass", "equilibrium_delta") if release == "kron" else ("id", "x", "p0")
                    inputs = [{key: event[key] for key in keys} for event in events]
                    for step in steps:
                        attempts += len(events)
                        require(attempts <= plan["new_forecast_event_step_attempts"], "Forecast allocation exceeded")
                        operations.append({"action": "forecast_reserved", **args, "step_s": step,
                                           "ids": group["ids"], "horizon_s": 15})
                        paths.append(forecasters[release, actual](inputs, 15, step))
                        operations.append({"action": "forecast_returned", **args, "step_s": step,
                                           "count": len(events)})
                    np.savez_compressed(directory / artifact, ids=np.array(group["ids"]),
                                        steps_s=steps, forecasts=np.stack(paths))
                archived = references[artifact]
                for event, predictions, reference in zip(events, zip(*paths, strict=True), archived, strict=True):
                    before = classify(predictions, event, steps, 8, match_settlement=release == "kron")
                    after = classify(predictions, event, steps, 15, match_settlement=release == "kron")
                    parity = numerical(predictions[0], reference, event, 800)
                    qualified = before["status"] == "window_no_los_unsettled" and parity["passed"]
                    status = after["status"] if qualified else "prefix_mismatch"
                    if actual != model:
                        status = "native_horizon_limited"
                    rows.append(args | {"id": event["id"], "truth_label": event["label"],
                        "forecast_artifact": artifact, "prefix_8s": before, "forecast_15s": after,
                        "archive_parity_8s": parity, "reproduced_8s": qualified, "status": status})
                print(f"{role}: {actual} ({len(events)} events)", flush=True)
            require(attempts == plan["new_forecast_event_step_attempts"]
                    and attempts + reused == plan["forecast_event_step_attempts"] and len(rows) == plan["pairs"],
                    "The extension must account for every registered case and step")
    write(directory / "cases.json", rows)
    return {"pairs": len(rows), "history_reads": {role: len(events) for role, events in plan["events"].items()},
            "raw_future_frames_read": 0, "simulations": 0, "forecast_event_step_attempts": attempts + reused,
            "new_forecast_event_step_attempts": attempts, "reused_forecast_event_steps": reused,
            "kron_operations": dict(budget.counts), "torch": torch.__version__,
            "cuda": torch.version.cuda, "cuda_device": torch.cuda.get_device_name(),
            "status_counts": dict(Counter(row["status"] for row in rows))}


def authorization(root, experiment):
    validate_run_policy(root, WORKFLOW, experiment)
    scope = load_policy(root)["official_test_evaluations"][experiment["test_authorization"]]
    registered_path = scope["plan_path"]
    require(scope["status"] == "active", "A fresh bounded approval is required")
    verify_bound_file(root, registered_path, scope["plan_sha256"])
    plan = read(root / registered_path)
    for path, digest in plan["tracked_sha256"].items():
        verify_bound_file(root, path, digest)
    require(plan["pairs"] == scope["pairs"] and plan["forecast_event_step_attempts"] == scope["forecast_event_step_attempts"]
            and plan["new_forecast_event_step_attempts"] == scope["new_forecast_event_step_attempts"]
            and plan["forecast_horizon_s"] == 15 and plan["inputs"] == "measured_history_100ms_zoh_v1"
            and all(scope[key] == value for key, value in ZERO.items()), "The approval differs from the forecast scope")
    return plan, scope


def run(*, root, experiment, stage, result_dir, run_id):
    plan, scope = authorization(root, experiment)
    require(stage == STAGE and sys.version_info[:2] == (3, 12), "The registered stage requires Python 3.12")
    reservation = root / ".project_state" / f"{Path(scope['plan_path']).stem}-reserved.json"
    write(reservation, {"run_id": run_id, "retry_allowed": False})
    start = time.perf_counter()
    evidence = execute(root, result_dir, plan) | ZERO | {
        "status": "PASS", "process_status": "PASS", "source_commit": git_head(root),
        "claim_boundary": "Follow-up of archived 8 s qualified unsettled forecasts on exposed fixed cohorts.",
        "plan_sha256": scope["plan_sha256"], "run_id": run_id,
        "runtime": {"python": sys.version, "executable": sys.executable, "seconds": time.perf_counter() - start}}
    write(result_dir / "stage_evidence.json", evidence)
    write(result_dir / "artifact_hashes.json", {path.name: sha256_file(path) for path in sorted(result_dir.iterdir())})


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    root = project_root()
    assert_clean(root)
    experiment = load_manifest(root)["experiments"][WORKFLOW]
    authorization(root, experiment)
    approve(root, STAGE, WORKFLOW)
    with workflow_lock(root, WORKFLOW):
        print(_run_experiment(root, WORKFLOW, STAGE, experiment=experiment), flush=True)


if __name__ == "__main__":
    main()
