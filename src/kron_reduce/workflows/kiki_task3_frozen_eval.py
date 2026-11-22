"""One-use 200 ms A03 transfer evaluation of TASK3's sealed kiki caches."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np

from kron_reduce.data.durable_audit_ledger import AppendOnlyAuditJournal
from kron_reduce.data.history_contract import CausalStateHistory
from kron_reduce.models.nonlinear_latent import LearnedNonlinearModel
from kron_reduce.physics.coordinates import coi_project, coi_project_power
from project_harness.evidence import sha256_file, verify_artifact_directory
from project_harness.policy import load_policy
from project_harness.report_storage import locate_report

AUTH = "kiki_task3_frozen_200ms_20261109"
PROPOSAL = "configs/proposals/kiki_task3_frozen_200ms.json"
MODULE = "src/kron_reduce/workflows/kiki_task3_frozen_eval.py"
REQUIRED_BINDINGS = {
    MODULE,
    "src/kron_reduce/models/nonlinear_latent.py",
    "src/kron_reduce/models/learned_latent.py",
    "src/kron_reduce/models/causal_latent.py",
    "src/kron_reduce/data/history_contract.py",
    "src/kron_reduce/physics/coordinates.py",
    "src/kron_reduce/data/durable_audit_ledger.py",
    "src/kron_reduce/data/audit_access_ledger.py",
    "project_harness/evidence.py",
    "project_harness/policy.py",
    "requirements/constraints-py312.txt",
    "docs/protocols/kiki_task3_frozen_200ms_20261109_zh.md",
}
ARCHIVE = "runs/campaign200ms"
CLOCK = np.arange(801) * 0.01
HISTORY = np.arange(-20, 1) * 0.01


def require(condition, reason):
    if not condition:
        raise PermissionError(reason)


def checked_bytes(root, relative, digest):
    path = (root / locate_report(root, relative, digest)).resolve()
    require(path.is_relative_to(root), f"out of scope: {relative}")
    value = path.read_bytes()
    require(hashlib.sha256(value).hexdigest() == digest, f"identity mismatch: {relative}")
    return value


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())


def reviewed_plan(root):
    approval = load_policy(root).get("frozen_retrospective_evaluations", {}).get(AUTH, {})
    require(approval.get("status") == "active", "frozen evaluation is not registered active")
    require(
        approval.get("fit_attempts") == approval.get("optimizer_updates") == 0,
        "only frozen inference is authorized",
    )
    bindings = approval["implementation_bindings"]
    require(REQUIRED_BINDINGS <= bindings.keys(), "missing reviewed implementation bindings")
    for relative, digest in bindings.items():
        checked_bytes(root, relative, digest)
    checked_bytes(root, approval["review_receipt"], approval["review_sha256"])
    plan = json.loads(checked_bytes(root, PROPOSAL, approval["proposal_sha256"]))
    require(
        plan["authorization_id"] == AUTH and plan["dataset"] == "kiki_ieee39_v1",
        "incorrect research scope",
    )
    return plan


def sealed_read(task3, files, relative, journal):
    digest = files[relative]
    journal.append({"stage": "read_reserved", "file": relative, "sha256": digest})
    value = checked_bytes(task3, relative, digest)
    journal.append({"stage": "bytes_verified", "file": relative, "bytes": len(value)})
    return value


def load_events(task3, files, plan, split, static, train_ids, journal):
    def read(name):
        return sealed_read(task3, files, f"{ARCHIVE}/{name}", journal)

    reference = json.loads(read(f"equilibrium/{split}.json"))
    groups = {row["event_id"]: row["group"] for row in reference["records"]}
    labels = {row["event_id"]: row for row in reference["classifications"]}
    count = plan[split]["eligible"]
    require(
        len(labels) == len(reference["classifications"]) == count
        and set(groups) == set(labels)
        and not train_ids & labels.keys(),
        f"{split} roster mismatch or train overlap",
    )
    require(
        sum(r["label"] == "unstable" for r in labels.values()) == plan[split]["unstable"],
        f"{split} label counts differ",
    )
    require(all(re.fullmatch(r"[a-z][a-z0-9_]+", i) for i in labels), "unsafe event ID")

    def event(eid, x, p0):
        require(
            x.ndim == 2
            and x.shape[1] == 20
            and 22 <= len(x) <= 821
            and p0.shape == (10,)
            and np.isfinite(x).all()
            and np.isfinite(p0).all(),
            f"invalid cached arrays for {eid}",
        )
        journal.append({"stage": "decoded", "split": split, "event": eid, "points": len(x)})
        return dict(
            id=eid,
            x=x,
            p0=p0,
            group=groups[eid],
            label=labels[eid]["label"],
            event_time_s=labels[eid]["true_event_time_record_s"],
            **static,
        )

    if split == "val":
        identity = json.loads(read("preflight/identity.json"))
        excluded = identity["excluded"]["dev"]
        require(
            identity["raw"]["dev"] == plan[split]["allocated"]
            and identity["eligible"]["dev"] == count,
            "val allocation differs",
        )
        events = []
        with np.load(io.BytesIO(read("preflight/development.npz")), allow_pickle=False) as arrays:
            for eid in labels:
                journal.append({"stage": "decode_reserved", "split": split, "event": eid})
                x, p0 = arrays[f"{eid}_x"], arrays[f"{eid}_p0"]
                require(
                    hashlib.sha256(x.tobytes() + p0.tobytes()).hexdigest()
                    == identity["payload_sha256"][eid],
                    f"val array binding differs: {eid}",
                )
                events.append(event(eid, x, p0))
    else:
        freeze = json.loads(read("freeze_manifest.json"))
        test = freeze["test_data"]
        access = json.loads(read("test_access.json"))
        require(
            freeze["fitting_closed"]
            and freeze["selection_closed"]
            and test["already_exposed"]
            and access["status"] == "COMPLETE"
            and access["freeze_sha256"] == files[f"{ARCHIVE}/freeze_manifest.json"]
            and test["raw_test"] == plan[split]["allocated"]
            and set(test["eligible_ids"]) == set(access["events"]) == set(labels)
            and sorted((Path(r["path"]).name, len(r["eligible"])) for r in test["roots"])
            == [("ieee39_expansion", 43), ("ieee39_holdout_supplement", 32)],
            "previously exposed test roster is not the exact sealed TASK3 roster",
        )
        excluded = test["excluded"]
        events = []
        for eid in labels:
            name = f"test_cache/{eid}.npz"
            entry = access["events"][eid]
            require(
                entry["status"] == "CACHED" and entry["sha256"] == files[f"{ARCHIVE}/{name}"],
                f"test access receipt differs: {eid}",
            )
            journal.append({"stage": "decode_reserved", "split": split, "event": eid})
            with np.load(io.BytesIO(read(name)), allow_pickle=False) as arrays:
                events.append(event(eid, arrays["x"], arrays["p0"]))
    require(
        len(excluded) == len(set(excluded)) == plan[split]["allocated"] - count
        and not set(excluded) & labels.keys(),
        f"{split} coverage mismatch",
    )
    return events, excluded


def forecast(model, history, observed_origin, equilibrium, step):
    path = model.predict(history, CLOCK, equilibrium, max_step=step)[:, :20].copy()
    path[1:, 10:] = coi_project(path[1:, 10:], model.closure.mass)
    path[0] = observed_origin
    return path


def evaluate_model(model, events, evaluate, plan, budget, journal, name, split, out):
    mass = model.closure.mass
    histories = [
        CausalStateHistory(
            HISTORY,
            coi_project(e["x"][:21, :10], mass),
            coi_project(e["x"][:21, 10:], mass),
            coi_project_power(e["p0"], mass),
        )
        for e in events
    ]
    equilibrium = coi_project(events[0]["equilibrium_delta"], mass)
    failed = []

    def paths(step):
        cost = len(events) * 820 * round(0.01 / step)
        budget["paths_reserved"] += len(events)
        budget["rk4_updates_reserved"] += cost
        require(
            budget["paths_reserved"] <= plan["max_model_paths"]
            and budget["rk4_updates_reserved"] <= plan["max_rk4_updates"]
            and 4 * budget["rk4_updates_reserved"] <= plan["max_rhs_calls"],
            "path budget exhausted",
        )
        journal.append(
            {"stage": "paths_reserved", "model": name, "split": split, "step": step, **budget}
        )
        output = []
        for e, h in zip(events, histories, strict=True):
            try:
                output.append(forecast(model, h, e["x"][20], equilibrium, step))
            except (FloatingPointError, OverflowError, ValueError) as err:
                # shortcut: A03 returns no partial path on failure; upgrade only with a reviewed prefix-preserving solver.
                path = np.full((801, 20), np.nan)
                path[0] = e["x"][20]
                output.append(path)
                failure = {
                    "event": e["id"],
                    "step": step,
                    "error": type(err).__name__,
                    "detail": str(err),
                }
                failed.append(failure)
                journal.append({"stage": "path_failed", "model": name, "split": split, **failure})
        journal.append({"stage": "paths_closed", "model": name, "split": split, "step": step})
        return np.stack(output)

    fine, coarse = paths(0.005), paths(0.01)
    result = evaluate(fine, events, coarse)
    arrays = {"step005": fine, "step010": coarse}
    if result["selection_score"] is None:
        original = {
            "counts": result["counts"],
            "intervals": [
                {key: row[key] for key in ("event_id", "horizon", "numerical", "qualified")}
                for row in result["records"]
            ],
        }
        arrays["step0025"] = paths(0.0025)
        result = evaluate(fine, events, arrays["step0025"])
        result["initial_numerical_qualification"] = original
    result["task3_six_term_score_zero_common"] = result.pop("selection_score")
    result.update(
        model=name,
        split=split,
        path_failures=failed,
        refinement_executed="step0025" in arrays,
        output_convention=plan["output_convention"],
    )
    with (out / f"{split}-{name}-predictions.npz").open("xb") as stream:
        np.savez_compressed(stream, event_ids=np.asarray([e["id"] for e in events]), **arrays)
    return result


def run(root):
    root = root.resolve()
    plan = reviewed_plan(root)
    require(root == Path(plan["checkout"]).resolve(), "evaluation is bound to one checkout")
    spent = root / ".project_state" / f"{AUTH}.spent.json"
    require(not spent.exists(), "evaluation authorization was already spent")
    task3 = (root.parent / "task3_baselines").resolve()
    require(
        not any(k == "task3" or k.startswith("task3.") for k in sys.modules),
        "start in a fresh Python process",
    )
    for relative, digest in plan["task3_code"].items():
        checked_bytes(task3, relative, digest)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(task3 / "src"))
    from task3.c200.evaluation import evaluate
    from task3.protocol import safe

    delivery = json.loads(
        checked_bytes(task3, plan["task3_delivery"], plan["task3_delivery_sha256"])
    )
    files = {name.replace("\\", "/"): digest for name, digest in delivery["files"].items()}
    run_id = AUTH + "-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid4().hex[:8]
    spent.parent.mkdir(exist_ok=True)
    write_json(spent, {"authorization": AUTH, "run_id": run_id})
    out = root / "results" / run_id
    out.mkdir()
    print(f"RUN_ID={run_id}", flush=True)
    write_json(
        out / "pre_access_freeze.json",
        {
            "plan": plan,
            "python": sys.version,
            "executable": sys.executable,
            "numpy": np.__version__,
        },
    )
    journal = AppendOnlyAuditJournal(out / "access.jsonl")
    budget = {"model_events_reserved": 0, "paths_reserved": 0, "rk4_updates_reserved": 0}
    try:
        fit = root / "results" / plan["checkpoint_run"]
        allocation = json.loads(
            checked_bytes(fit, "role_allocation.json", plan["checkpoint_allocation_sha256"])
        )
        train_ids = {r["raw_id"] for r in allocation["rows"]}
        require(len(train_ids) == 369, "A03 original train allocation differs")
        models = {}
        for name, digest in plan["checkpoints"].items():
            relative = f"candidate-{name}.npz"
            models[name] = LearnedNonlinearModel.load(
                io.BytesIO(checked_bytes(fit, relative, digest))
            )
        static_bytes = sealed_read(task3, files, f"{ARCHIVE}/preflight/static.npz", journal)
        with np.load(io.BytesIO(static_bytes), allow_pickle=False) as arrays:
            static = {
                "mass": arrays["mass"],
                "equilibrium_delta": arrays["delta_star"],
                "omega_s": float(arrays["omega_s"]),
            }
        for model in models.values():
            np.testing.assert_allclose(model.closure.mass, static["mass"], rtol=1e-12, atol=1e-12)
        for split in ("val", "test"):
            events, excluded = load_events(task3, files, plan, split, static, train_ids, journal)
            write_json(
                out / f"{split}-coverage.json",
                {"eligible": [e["id"] for e in events], "not_covered": excluded},
            )
            for name, model in models.items():
                budget["model_events_reserved"] += len(events)
                require(
                    budget["model_events_reserved"] <= plan["max_model_event_combinations"],
                    "model-event budget exhausted",
                )
                result = evaluate_model(
                    model, events, evaluate, plan, budget, journal, name, split, out
                )
                write_json(out / f"{split}-{name}.json", safe(result))
                print(f"{split} {name}: {result['counts']}", flush=True)
            if split == "val":
                write_json(
                    out / "val_freeze.json",
                    {p.name: sha256_file(p) for p in sorted(out.glob("val-*"))},
                )
        write_json(
            out / "completion.json",
            dict(
                status="COMPLETE",
                **budget,
                rhs_calls_reserved=4 * budget["rk4_updates_reserved"],
                fit_attempts=0,
                optimizer_updates=0,
                promotion_decision="KEEP_BASELINE",
                independent_test=False,
            ),
        )
    except BaseException as err:
        write_json(out / "failure.json", dict(error=type(err).__name__, message=str(err), **budget))
        raise
    finally:
        journal.close()
    write_json(out / "artifact_hashes.json", {p.name: sha256_file(p) for p in out.iterdir()})
    verify_artifact_directory(out, strict=True)


if __name__ == "__main__":
    run(Path(__file__).resolve().parents[3])
