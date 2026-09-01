"""Read-only P1 execution of shared-pole, excitation-local observability evidence."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from kron_reduce.data.p1_causal import P1RoleManifest, validate_p1_samples
from kron_reduce.evaluation.p1_shared_pole_observability import (
    DMD_RANKS,
    audit_local_observability_events,
    extract_shared_equation_poles,
    summarize_local_observability,
)
from kron_reduce.models.prunable_energy_model import PrunableEnergyModel
from kron_reduce.workflows.p1_counterfactual_execution import (
    _new_output,
    _seal,
    verify_counterfactual_artifacts,
)
from kron_reduce.workflows.p1_five_arm_execution import (
    _read_json,
    _write_json,
    file_sha256,
    verify_five_arm_bundle,
)

_REQUIRED_FILES = {
    "lineage.json",
    "modal_audit_summary.json",
    "modal_bootstrap.json",
    "modal_event_matches.csv",
    "modal_line_support.csv",
    "shared_equation_poles.json",
    "shared_pole_local_observability.csv",
}

_EVENT_FIELDS = (
    "event_id",
    "view_name",
    "fault_line",
    "pole_id",
    "dmd_rank",
    "candidate_count",
    "candidate_modes_json",
    "dmd_error",
    "candidate_index",
    "candidate_frequency_hz",
    "candidate_damping_ratio",
    "frequency_relative_error",
    "damping_absolute_error",
    "mac",
    "assignment_cost",
    "match_passed",
    "match_status",
    "finite_window_energy_share",
    "dmd_rank_passes",
    "registered_dmd_ranks",
    "event_supported",
)
_LINE_FIELDS = (
    "pole_id",
    "view_name",
    "fault_line",
    "event_count",
    "supporting_event_count",
    "event_support_fraction",
    "line_supported",
)
_VIEW_FIELDS = (
    "pole_id",
    "view_name",
    "independent_lines",
    "supporting_lines",
    "line_support_fraction",
    "bootstrap_ci95_lower",
    "bootstrap_ci95_upper",
    "bootstrap_support_probability",
    "bootstrap_draws",
    "bootstrap_seed",
    "cluster_unit",
    "status",
)


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if type(value) is bool:
        return "true" if value else "false"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise ValueError("nonfinite CSV evidence is forbidden")
    return value


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    expected = set(fieldnames)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            if set(row) != expected:
                raise ValueError(f"CSV row does not match the frozen schema for {path.name}")
            writer.writerow({name: _csv_value(row[name]) for name in fieldnames})


def _read_csv_header(path: Path) -> tuple[str, ...]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        try:
            return tuple(next(reader))
        except StopIteration as error:
            raise ValueError(f"empty CSV artifact: {path.name}") from error


def _read_csv_rows(path: Path, fields: Sequence[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != tuple(fields):
            raise ValueError(f"CSV schema changed: {path.name}")
        return list(reader)


def _validate_modal_artifact_contents(directory: Path) -> None:
    directory = Path(directory)
    event_rows = _read_csv_rows(directory / "modal_event_matches.csv", _EVENT_FIELDS)
    line_rows = _read_csv_rows(directory / "modal_line_support.csv", _LINE_FIELDS)
    view_rows = _read_csv_rows(
        directory / "shared_pole_local_observability.csv", _VIEW_FIELDS
    )
    shared = _read_json(directory / "shared_equation_poles.json")
    summary = _read_json(directory / "modal_audit_summary.json")
    lineage = _read_json(directory / "lineage.json")
    bootstrap = _read_json(directory / "modal_bootstrap.json")
    if (
        shared.get("schema") != "kron_reduce.p1_shared_equation_poles.v1"
        or shared.get("trajectory_modes_used_to_define_poles") is not False
        or summary.get("schema") != "kron_reduce.p1_modal_audit_summary.v1"
        or summary.get("scientific_verdict") is not None
        or summary.get("model_unit_gate_passed") is not False
        or summary.get("promotion_decision") != "KEEP_BASELINE"
        or lineage.get("model_count") != 1
        or lineage.get("source_routing") is not False
        or lineage.get("test_ids_read") != []
        or lineage.get("other_folds_read") != []
        or bootstrap.get("draws") != 2000
        or bootstrap.get("seed") != 324
        or bootstrap.get("cluster_unit") != "source_dataset_x_fault_line"
    ):
        raise ValueError("modal observability artifact contract changed")
    if (
        summary.get("base_manifest_sha256") != lineage.get("base_manifest_sha256")
        or summary.get("core_checkpoint_sha256") != shared.get("checkpoint_sha256")
        or summary.get("role_manifest_sha256") != lineage.get("role_manifest_sha256")
        or summary.get("static_signature_sha256") != shared.get("static_signature_sha256")
    ):
        raise ValueError("modal observability lineage binding changed")
    pole_ids = {
        str(row["pole_id"]) for row in shared.get("equation_reportable_modes", [])
    }
    if (
        len(pole_ids) != len(shared.get("equation_reportable_modes", []))
        or pole_ids != set(summary.get("pole_statuses", {}))
        or len(pole_ids) != int(summary.get("equation_reportable_pole_count", -1))
    ):
        raise ValueError("shared-pole identity/count binding changed")
    event_keys = [
        (row["event_id"], row["pole_id"], int(row["dmd_rank"])) for row in event_rows
    ]
    if (
        len(event_keys) != len(set(event_keys))
        or any(rank not in DMD_RANKS for _, _, rank in event_keys)
        or any(pole_id not in pole_ids for _, pole_id, _ in event_keys)
        or len(event_rows) != int(summary.get("event_rank_match_cells", -1))
        or len(event_rows)
        != int(summary.get("event_count", -1)) * len(pole_ids) * len(DMD_RANKS)
    ):
        raise ValueError("modal event match cells are incomplete or duplicated")
    if pole_ids and len({event_id for event_id, _, _ in event_keys}) != int(
        summary.get("event_count", -1)
    ):
        raise ValueError("modal event count binding changed")
    event_poles: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    assigned_candidates: dict[tuple[str, int], list[int]] = defaultdict(list)
    candidate_payloads: dict[tuple[str, int], tuple[int, Any]] = {}
    for row in event_rows:
        if row["match_passed"] not in {"true", "false"} or row["event_supported"] not in {
            "true",
            "false",
        }:
            raise ValueError("modal event boolean encoding changed")
        candidates = json.loads(row["candidate_modes_json"])
        if not isinstance(candidates, list) or len(candidates) != int(row["candidate_count"]):
            raise ValueError("modal candidate list/count binding changed")
        event_rank = (row["event_id"], int(row["dmd_rank"]))
        payload = (int(row["candidate_count"]), candidates)
        if candidate_payloads.setdefault(event_rank, payload) != payload:
            raise ValueError("candidate modes differ across poles in one event/rank")
        if row["candidate_index"]:
            assigned_candidates[event_rank].append(int(row["candidate_index"]))
        if (row["match_status"] == "MATCHED_PASS") != (row["match_passed"] == "true"):
            raise ValueError("modal match status/boolean binding changed")
        event_poles[(row["event_id"], row["pole_id"])].append(row)
    if any(len(indices) != len(set(indices)) for indices in assigned_candidates.values()):
        raise ValueError("one empirical mode was assigned to more than one shared pole")
    if len(event_poles) != int(summary.get("event_pole_cells", -1)):
        raise ValueError("modal event/pole count binding changed")
    derived_events: dict[tuple[str, str, str], list[bool]] = defaultdict(list)
    for (_event_id, pole_id), rows in event_poles.items():
        ranks = {int(row["dmd_rank"]) for row in rows}
        invariant_fields = (
            "view_name",
            "fault_line",
            "finite_window_energy_share",
            "dmd_rank_passes",
            "registered_dmd_ranks",
            "event_supported",
        )
        if ranks != set(DMD_RANKS) or any(
            len({row[field] for row in rows}) != 1 for field in invariant_fields
        ):
            raise ValueError("modal event/pole rank cells are incomplete or inconsistent")
        passed = sum(row["match_passed"] == "true" for row in rows)
        energy = float(rows[0]["finite_window_energy_share"])
        supported = rows[0]["event_supported"] == "true"
        if (
            passed != int(rows[0]["dmd_rank_passes"])
            or int(rows[0]["registered_dmd_ranks"]) != len(DMD_RANKS)
            or supported != (energy >= 0.01 and passed >= 2)
        ):
            raise ValueError("modal event support was not derived from the frozen gates")
        derived_events[(pole_id, rows[0]["view_name"], rows[0]["fault_line"])].append(
            supported
        )
    line_keys = [(row["pole_id"], row["view_name"], row["fault_line"]) for row in line_rows]
    view_keys = [(row["pole_id"], row["view_name"]) for row in view_rows]
    if (
        len(line_keys) != len(set(line_keys))
        or len(view_keys) != len(set(view_keys))
        or any(pole_id not in pole_ids for pole_id, *_ in line_keys)
        or any(pole_id not in pole_ids for pole_id, _ in view_keys)
        or any(row["line_supported"] not in {"true", "false"} for row in line_rows)
        or any(
            row["status"] not in {"SUPPORTED", "UNSUPPORTED", "UNDETERMINED"}
            for row in view_rows
        )
    ):
        raise ValueError("modal line/view cells are malformed or duplicated")
    line_lookup = {
        (row["pole_id"], row["view_name"], row["fault_line"]): row for row in line_rows
    }
    if set(line_lookup) != set(derived_events):
        raise ValueError("modal line table does not cover the derived event cells")
    derived_lines: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for key, flags in sorted(derived_events.items()):
        row = line_lookup[key]
        supporting = int(np.sum(flags))
        fraction = float(np.mean(flags))
        line_supported = fraction >= 0.50
        if (
            int(row["event_count"]) != len(flags)
            or int(row["supporting_event_count"]) != supporting
            or not np.isclose(float(row["event_support_fraction"]), fraction, atol=1e-15)
            or (row["line_supported"] == "true") != line_supported
        ):
            raise ValueError("modal line support was not derived from complete event cells")
        derived_lines[(key[0], key[1])].append(line_supported)
    bootstrap_cells = bootstrap.get("cells", [])
    bootstrap_keys = [
        (str(row.get("pole_id")), str(row.get("view_name"))) for row in bootstrap_cells
    ]
    if bootstrap_keys != view_keys:
        raise ValueError("modal bootstrap cells do not follow the frozen view table")
    view_lookup = {(row["pole_id"], row["view_name"]): row for row in view_rows}
    if set(view_lookup) != set(derived_lines):
        raise ValueError("modal view table does not cover the derived line cells")
    rng = np.random.default_rng(324)
    for cell in bootstrap_cells:
        key = (str(cell.get("pole_id")), str(cell.get("view_name")))
        draws = np.asarray(cell.get("draw_support_fractions"), dtype=float)
        if draws.shape != (2000,) or not np.isfinite(draws).all() or np.any(
            (draws < 0.0) | (draws > 1.0)
        ):
            raise ValueError("modal bootstrap draw vector is invalid")
        flags = np.asarray(derived_lines[key], dtype=float)
        expected_draws = rng.choice(flags, size=(2000, len(flags)), replace=True).mean(axis=1)
        row = view_lookup[key]
        point = float(np.mean(flags))
        lower, upper = (float(value) for value in np.quantile(expected_draws, (0.025, 0.975)))
        probability = float(np.mean(expected_draws >= 0.80))
        expected_status = (
            "UNDETERMINED"
            if len(flags) < 3
            else "SUPPORTED"
            if probability >= 0.80
            else "UNSUPPORTED"
        )
        if (
            not np.array_equal(draws, expected_draws)
            or int(row["independent_lines"]) != len(flags)
            or int(row["supporting_lines"]) != int(np.sum(flags))
            or not np.isclose(float(row["line_support_fraction"]), point, atol=1e-15)
            or not np.isclose(float(row["bootstrap_ci95_lower"]), lower, atol=1e-15)
            or not np.isclose(float(row["bootstrap_ci95_upper"]), upper, atol=1e-15)
            or not np.isclose(
                float(row["bootstrap_support_probability"]), probability, atol=1e-15
            )
            or int(row["bootstrap_draws"]) != 2000
            or int(row["bootstrap_seed"]) != 324
            or row["cluster_unit"] != "source_dataset_x_fault_line"
            or row["status"] != expected_status
        ):
            raise ValueError("modal view/bootstrap evidence was not derived from frozen line cells")


def verify_modal_observability_artifacts(
    directory: Path, expected_manifest_sha256: str
) -> dict[str, str]:
    """Reject changed, missing and newly added modal evidence files."""

    hashes = verify_counterfactual_artifacts(directory, expected_manifest_sha256)
    if set(hashes) != _REQUIRED_FILES:
        raise ValueError("modal observability artifact file set changed")
    if _read_csv_header(Path(directory) / "modal_event_matches.csv") != _EVENT_FIELDS:
        raise ValueError("modal event CSV schema changed")
    if _read_csv_header(Path(directory) / "modal_line_support.csv") != _LINE_FIELDS:
        raise ValueError("modal line CSV schema changed")
    if (
        _read_csv_header(Path(directory) / "shared_pole_local_observability.csv")
        != _VIEW_FIELDS
    ):
        raise ValueError("modal view CSV schema changed")
    _validate_modal_artifact_contents(Path(directory))
    return hashes


def _expected_events(manifest: P1RoleManifest) -> list[dict[str, str]]:
    return [
        {
            "event_id": row.event_id,
            "view_name": row.source_dataset,
            "fault_line": row.fault_line,
        }
        for row in sorted(manifest.records, key=lambda value: value.event_id)
        if row.role == "internal_audit"
    ]


def _event_match_rows(event_audit: Mapping[str, Any]) -> list[dict[str, Any]]:
    event_lookup = {
        (row["event_id"], row["pole_id"]): row for row in event_audit["event_rows"]
    }
    candidates: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in event_audit["candidate_rows"]:
        candidates[(row["event_id"], int(row["dmd_rank"]))].append(
            {
                key: row[key]
                for key in (
                    "candidate_index",
                    "frequency_hz",
                    "damping_ratio",
                    "decay_rate",
                    "strength",
                    "shape_real",
                    "shape_imag",
                )
            }
        )
    output = []
    for match in event_audit["rank_rows"]:
        evidence = event_lookup[(match["event_id"], match["pole_id"])]
        output.append(
            {
                "event_id": match["event_id"],
                "view_name": match["view_name"],
                "fault_line": match["fault_line"],
                "pole_id": match["pole_id"],
                "dmd_rank": match["dmd_rank"],
                "candidate_count": match["candidate_count"],
                "candidate_modes_json": candidates[
                    (match["event_id"], int(match["dmd_rank"]))
                ],
                "dmd_error": match["dmd_error"],
                "candidate_index": match["candidate_index"],
                "candidate_frequency_hz": match["candidate_frequency_hz"],
                "candidate_damping_ratio": match["candidate_damping_ratio"],
                "frequency_relative_error": match["frequency_relative_error"],
                "damping_absolute_error": match["damping_absolute_error"],
                "mac": match["mac"],
                "assignment_cost": match["assignment_cost"],
                "match_passed": match["match_passed"],
                "match_status": match["status"],
                "finite_window_energy_share": evidence["finite_window_energy_share"],
                "dmd_rank_passes": evidence["dmd_rank_passes"],
                "registered_dmd_ranks": evidence["registered_dmd_ranks"],
                "event_supported": evidence["event_supported"],
            }
        )
    return output


def evaluate_saved_modal_observability(
    base_directory: Path,
    base_manifest_sha256: str,
    manifest: P1RoleManifest,
    internal_samples: Sequence[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    """Create immutable local-observability evidence without fitting or reading future Pm."""

    base_directory, output_dir = Path(base_directory), Path(output_dir)
    validate_p1_samples(internal_samples, manifest, "internal_audit")
    base = verify_five_arm_bundle(base_directory, base_manifest_sha256, manifest)
    expected = _expected_events(manifest)
    if [row["event_id"] for row in expected] != list(manifest.ids("internal_audit")):
        raise ValueError("internal-audit manifest ordering changed")
    checkpoint_sha256 = str(base["arms"][2]["checkpoint_hash"])
    _new_output(output_dir, (base_directory,))
    try:
        model = PrunableEnergyModel.load(base_directory / "core3.npz")
        poles, shared = extract_shared_equation_poles(
            model, internal_samples, checkpoint_sha256
        )
        event_audit = audit_local_observability_events(
            internal_samples,
            poles,
            expected_static_signature_sha256=shared["static_signature_sha256"],
            dmd_ranks=DMD_RANKS,
        )
        observability = summarize_local_observability(
            event_audit["event_rows"], expected, [pole.pole_id for pole in poles]
        )
        event_rows = _event_match_rows(event_audit)
        status_counts = Counter(row["status"] for row in observability["view_rows"])
        summary = {
            "schema": "kron_reduce.p1_modal_audit_summary.v1",
            "base_manifest_sha256": base_manifest_sha256,
            "core_checkpoint_sha256": checkpoint_sha256,
            "role_manifest_sha256": manifest.source_sha256,
            "role": "internal_audit",
            "static_signature_sha256": shared["static_signature_sha256"],
            "equation_reportable_pole_count": len(poles),
            "event_count": len(expected),
            "event_pole_cells": len(event_audit["event_rows"]),
            "event_rank_match_cells": len(event_rows),
            "view_status_counts": dict(sorted(status_counts.items())),
            "pole_statuses": observability["pole_statuses"],
            "frequency_source": "shared_equation_poles",
            "trajectory_dmd_role": "local_observability_only",
            "source_labels_entered_model_or_pole_extraction": False,
            "observed_or_future_pm_read": False,
            "scientific_verdict": None,
            "model_unit_gate_passed": False,
            "promotion_decision": "KEEP_BASELINE",
        }
        lineage = {
            "schema": "kron_reduce.p1_modal_observability_lineage.v1",
            "base_manifest_sha256": base_manifest_sha256,
            "core_checkpoint_sha256": checkpoint_sha256,
            "core_npz_sha256": file_sha256(base_directory / "core3.npz"),
            "role_manifest_sha256": manifest.source_sha256,
            "role_id_hash": manifest.role_id_hashes["internal_audit"],
            "role": "internal_audit",
            "model_count": 1,
            "source_routing": False,
            "labels_used": False,
            "future_observed_pm_used": False,
            "shared_spectrum_frozen_before_target_grouping": True,
            "test_ids_read": [],
            "other_folds_read": [],
        }
        _write_json(output_dir / "shared_equation_poles.json", shared)
        _write_csv(output_dir / "modal_event_matches.csv", _EVENT_FIELDS, event_rows)
        _write_csv(
            output_dir / "modal_line_support.csv", _LINE_FIELDS, observability["line_rows"]
        )
        _write_csv(
            output_dir / "shared_pole_local_observability.csv",
            _VIEW_FIELDS,
            observability["view_rows"],
        )
        _write_json(output_dir / "modal_bootstrap.json", observability["bootstrap"])
        _write_json(output_dir / "modal_audit_summary.json", summary)
        _write_json(output_dir / "lineage.json", lineage)
        verify_five_arm_bundle(base_directory, base_manifest_sha256, manifest)
        if {path.name for path in output_dir.iterdir()} != _REQUIRED_FILES:
            raise ValueError("modal observability output file set is incomplete")
        _validate_modal_artifact_contents(output_dir)
        digest = _seal(output_dir)
        verify_modal_observability_artifacts(output_dir, digest)
        return {
            **summary,
            "directory": str(output_dir.resolve()),
            "artifact_manifest_sha256": digest,
        }
    except Exception as error:
        if not (output_dir / "artifact_hashes.json").exists() and not (
            output_dir / "FAILED.json"
        ).exists():
            _write_json(
                output_dir / "FAILED.json",
                {
                    "status": "modal_observability_attempt_failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "scientific_verdict": None,
                    "model_unit_gate_passed": False,
                    "promotion_decision": "KEEP_BASELINE",
                },
            )
        raise
