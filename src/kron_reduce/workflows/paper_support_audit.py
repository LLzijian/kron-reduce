from __future__ import annotations

import csv
import hashlib
import json
import tomllib
from pathlib import Path

from kron_reduce.data import CanonicalStore, load_split


SPLITS = {
    "general_dynamics": "general_line_5fold_v1",
    "binary_stability": "stability_line_5fold_v1",
    "combined_dynamics": "combined_line_5fold_v1",
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _role_rows(store: CanonicalStore, ids: tuple[str, ...], steps: int) -> dict[str, int]:
    eligible = [event_id for event_id in ids if store.metadata(event_id).valid_length >= steps]
    labels = [store.metadata(event_id).outcome_label for event_id in eligible]
    return {
        "events": len(ids),
        "eligible": len(eligible),
        "excluded_short": len(ids) - len(eligible),
        "stable": labels.count("stable"),
        "unstable": labels.count("unstable"),
    }


def run(*, root: Path, experiment: dict, stage: str, result_dir: Path, run_id: str) -> None:
    if stage != "support_audit":
        raise RuntimeError("paper support audit only implements the support_audit stage")
    config = tomllib.loads(
        (root / "configs" / "experiments" / "paper_support_audit.toml").read_text(
            encoding="utf-8"
        )
    )
    dataset_dir = root / "data" / "ieee39_postclear_canonical_v1"
    store = CanonicalStore.open(dataset_dir)
    rows: list[dict[str, object]] = []
    findings: list[dict[str, object]] = []

    for view, split_name in SPLITS.items():
        for fold in config["folds"]:
            split = load_split(
                dataset_dir, view=view, split_name=split_name, fold=fold
            )
            role_ids = {
                "train": split.train_ids,
                "val": split.val_ids,
                "test": split.test_ids,
            }
            role_sets = {name: set(ids) for name, ids in role_ids.items()}
            event_isolated = not any(
                role_sets[left] & role_sets[right]
                for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
            )
            line_sets = {
                name: {store.metadata(event_id).fault_line for event_id in ids}
                for name, ids in role_ids.items()
            }
            line_isolated = not any(
                line_sets[left] & line_sets[right]
                for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
            )
            calibration_ids = tuple(split.disturbance_calibration["training_event_ids"])
            calibration_train_only = (
                set(calibration_ids) == role_sets["train"]
                and len(calibration_ids) == len(split.train_ids)
            )
            findings.append(
                {
                    "view": view,
                    "fold": fold,
                    "event_isolated": event_isolated,
                    "fault_line_isolated": line_isolated,
                    "calibration_train_only": calibration_train_only,
                }
            )
            for steps in config["horizon_steps"]:
                for role, ids in role_ids.items():
                    rows.append(
                        {
                            "view": view,
                            "fold": fold,
                            "role": role,
                            "horizon_steps": steps,
                            **_role_rows(store, ids, steps),
                        }
                    )

    # A prospective prefix risk set is defined without using trajectory length
    # as a feature: the prefix length is fixed, short records fail closed, and
    # events already observed inside the prefix are excluded explicitly.
    risk_rows: list[dict[str, object]] = []
    time_cache: dict[str, object] = {}
    for fold in config["folds"]:
        split = load_split(
            dataset_dir,
            view="binary_stability",
            split_name=SPLITS["binary_stability"],
            fold=fold,
        )
        for steps in config["horizon_steps"]:
            for role, ids in (
                ("train", split.train_ids),
                ("val", split.val_ids),
                ("test", split.test_ids),
            ):
                eligible = 0
                short = 0
                event_in_prefix = 0
                for event_id in ids:
                    meta = store.metadata(event_id)
                    if meta.valid_length <= steps:
                        short += 1
                        continue
                    if event_id not in time_cache:
                        time_cache[event_id] = store.load_event(event_id).t
                    prefix_end = float(time_cache[event_id][steps - 1])
                    if meta.event_observed and meta.event_time_s is not None and meta.event_time_s <= prefix_end:
                        event_in_prefix += 1
                        continue
                    eligible += 1
                risk_rows.append(
                    {
                        "fold": fold,
                        "role": role,
                        "prefix_steps": steps,
                        "eligible": eligible,
                        "excluded_short": short,
                        "excluded_event_in_prefix": event_in_prefix,
                    }
                )

    passed = all(
        row["event_isolated"]
        and row["fault_line_isolated"]
        and row["calibration_train_only"]
        for row in findings
    )
    if not passed:
        raise RuntimeError("support audit found split or calibration leakage")

    with (result_dir / "support_counts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (result_dir / "prospective_risk_sets.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(risk_rows[0]))
        writer.writeheader()
        writer.writerows(risk_rows)
    audit = {
        "status": "PASS",
        "run_id": run_id,
        "evidence_status": "support_audit",
        "performance_metrics_read": False,
        "offline_metadata_roles_inspected": ["train", "val", "test"],
        "split_findings": findings,
        "prospective_definition": {
            "fixed_prefix_steps": config["horizon_steps"],
            "short_trajectory_policy": "fail_closed_and_audit",
            "event_in_prefix_policy": "exclude_from_prospective_risk_set",
        },
    }
    (result_dir / "support_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    artifacts = {
        path.relative_to(result_dir).as_posix(): _sha(path)
        for path in result_dir.rglob("*")
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    (result_dir / "artifact_hashes.json").write_text(
        json.dumps(artifacts, indent=2), encoding="utf-8"
    )
