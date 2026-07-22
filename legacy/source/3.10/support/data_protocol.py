"""Protocol A/B data loading and split definitions."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from support.common import (
    ACTIVE_GROUPS,
    DENSE_DIR,
    HIGHORDER_DIR,
    PROJECT_ROOT,
    V1_DIR,
)

from diag_clean_unstable_trajectories import (
    clean_dataset,
    clean_raw_npz,
    clean_sequence,
    summarize_diagnostics,
)
from coherent_aggregation import aggregate_sequences
from evaluate_8cluster_dataonly_engineering_ieee39 import (
    external_meta_rows,
    external_raw_data,
    make_centered_external_sequence,
)
from graph_sindy_data import M_PRIOR, prepare_centered_dataset


PROTOCOL_B_SPLITS: dict[str, dict[str, list[str]]] = {
    "B1": {
        "test": ["Line_35", "Line_41", "Line_45"],
        "val": ["Line_36"],
        "train": ["Line_2", "Line_37", "Line_42", "Line_43", "Line_44", "Line_46"],
    },
    "B2": {
        "test": ["Line_36", "Line_42", "Line_46"],
        "val": ["Line_35"],
        "train": ["Line_2", "Line_37", "Line_41", "Line_43", "Line_44", "Line_45"],
    },
    "B3": {
        "test": ["Line_37", "Line_43", "Line_44"],
        "val": ["Line_46"],
        "train": ["Line_2", "Line_35", "Line_36", "Line_41", "Line_42", "Line_45"],
    },
    "B4": {
        "test": ["Line_35", "Line_42", "Line_46"],
        "val": ["Line_44"],
        "train": ["Line_2", "Line_36", "Line_37", "Line_41", "Line_43", "Line_45"],
    },
    "B5": {
        "test": ["Line_36", "Line_43", "Line_45"],
        "val": ["Line_37"],
        "train": ["Line_2", "Line_35", "Line_41", "Line_42", "Line_44", "Line_46"],
    },
}


_RAW_FLATLINE_AUDIT_CACHE: dict[str, dict[str, Any]] = {}


def read_case_results(dataset_dir: Path) -> list[dict[str, str]]:
    path = dataset_dir / "case_results.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def label_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    labels = ["stable", "near_critical", "unstable"]
    return {label: sum(1 for r in rows if r.get("label") == label) for label in labels}


def fault_line_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for row in rows:
        line = str(row.get("fault_line", ""))
        out.setdefault(line, {"stable": 0, "near_critical": 0, "unstable": 0, "total": 0})
        lab = str(row.get("label", ""))
        if lab in out[line]:
            out[line][lab] += 1
        out[line]["total"] += 1
    return dict(sorted(out.items()))


def _balanced_indices(rows: list[dict[str, Any]], per_label: int, minimum: int) -> list[int]:
    picked: list[int] = []
    for lab in ("stable", "near_critical", "unstable"):
        for i, row in enumerate(rows):
            if row.get("label") == lab and i not in picked:
                picked.append(i)
                if sum(1 for j in picked if rows[j].get("label") == lab) >= per_label:
                    break
    i = 0
    while len(picked) < min(minimum, len(rows)) and i < len(rows):
        if i not in picked:
            picked.append(i)
        i += 1
    return sorted(picked)


def limit_labeled(rows: list[dict[str, Any]], seqs: list[dict[str, Any]], quick: bool, per_label: int = 8) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not quick:
        return rows, seqs
    idx = _balanced_indices(rows, per_label=per_label, minimum=min(30, len(rows)))
    return [rows[i] for i in idx], [seqs[i] for i in idx]


def load_external_dataset(dataset_dir: Path, clean_flatlines: bool = True) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    seqs: list[dict[str, Any]] = []
    for row in external_meta_rows(dataset_dir):
        npz_path = dataset_dir / "npz" / f"{row['scenario_id']}.npz"
        if not npz_path.exists():
            continue
        row = dict(row)
        if clean_flatlines:
            raw, raw_diag = clean_raw_npz(npz_path, meta=row)
            row["_raw_clean_diag"] = raw_diag
        else:
            raw = external_raw_data(dataset_dir, row["scenario_id"])
            raw_diag = None
        seq = make_centered_external_sequence(row, raw)
        if raw_diag is not None:
            seq["_raw_clean_diag"] = raw_diag
        seq["split"] = "external"
        seq["true_label"] = row.get("label", "")
        rows.append(row)
        seqs.append(seq)
    return rows, seqs


def _tag_cleaning_diagnostics(
    diagnostics: list[dict[str, Any]],
    protocol: str,
    split_id: str,
    data_split: str,
) -> list[dict[str, Any]]:
    return [
        {**diag, "protocol": protocol, "split_id": split_id, "data_split": data_split}
        for diag in diagnostics
    ]


def _clean_rows_and_sequences(
    rows: list[dict[str, Any]],
    seqs: list[dict[str, Any]],
    protocol: str,
    split_id: str,
    data_split: str,
    clean_flatlines: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not clean_flatlines:
        return rows, seqs, []
    cleaned_rows, cleaned_seqs, diagnostics = clean_dataset(rows, seqs)
    return (
        cleaned_rows,
        cleaned_seqs,
        _tag_cleaning_diagnostics(diagnostics, protocol, split_id, data_split),
    )


def _clean_sequences_only(
    seqs: list[dict[str, Any]],
    protocol: str,
    split_id: str,
    data_split: str,
    clean_flatlines: bool,
    default_label: str = "",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not clean_flatlines:
        return seqs, []
    cleaned: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for i, seq in enumerate(seqs):
        cleaned_seq = clean_sequence(seq)
        diag = dict(cleaned_seq.get("_clean_diag", {}))
        diag["scenario_id"] = str(seq.get("scenario_id", seq.get("meta", {}).get("scenario_id", i)))
        diag["label"] = str(seq.get("true_label", seq.get("meta", {}).get("label", default_label)))
        diag["failure_reason"] = str(seq.get("meta", {}).get("failure_reason", ""))
        diag["fault_line"] = str(seq.get("meta", {}).get("fault_line", seq.get("meta", {}).get("line", "")))
        diag["fault_duration"] = seq.get("meta", {}).get("fault_duration", "")
        cleaned.append(cleaned_seq)
        diagnostics.append(diag)
    return cleaned, _tag_cleaning_diagnostics(diagnostics, protocol, split_id, data_split)


def cleaning_summary_rows(diagnostics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for diag in diagnostics:
        key = (
            str(diag.get("protocol", "")),
            str(diag.get("split_id", "")),
            str(diag.get("data_split", "")),
            str(diag.get("label", "")),
        )
        rec = groups.setdefault(
            key,
            {
                "protocol": key[0],
                "split_id": key[1],
                "data_split": key[2],
                "label": key[3],
                "total": 0,
                "flatlined": 0,
                "flatline_fractions": [],
                "valid_steps": [],
                "total_steps": [],
            },
        )
        rec["total"] += 1
        if diag.get("has_flatline"):
            rec["flatlined"] += 1
            rec["flatline_fractions"].append(float(diag.get("flatline_fraction", 0.0)))
        if diag.get("valid_steps") not in ("", None):
            rec["valid_steps"].append(float(diag.get("valid_steps", 0.0)))
        if diag.get("total_steps") not in ("", None):
            rec["total_steps"].append(float(diag.get("total_steps", 0.0)))

    rows: list[dict[str, Any]] = []
    for rec in sorted(groups.values(), key=lambda r: (r["protocol"], r["split_id"], r["data_split"], r["label"])):
        fracs = rec.pop("flatline_fractions")
        valid_steps = rec.pop("valid_steps")
        total_steps = rec.pop("total_steps")
        rec["flatlined_fraction"] = rec["flatlined"] / max(rec["total"], 1)
        rec["mean_flatline_fraction"] = float(np.mean(fracs)) if fracs else 0.0
        rec["max_flatline_fraction"] = float(np.max(fracs)) if fracs else 0.0
        rec["mean_valid_steps"] = float(np.mean(valid_steps)) if valid_steps else 0.0
        rec["mean_total_steps"] = float(np.mean(total_steps)) if total_steps else 0.0
        rows.append(rec)
    return rows


def raw_flatline_audit(dataset_dir: Path) -> dict[str, Any]:
    cache_key = str(dataset_dir.resolve())
    cached = _RAW_FLATLINE_AUDIT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    diagnostics: list[dict[str, Any]] = []
    npz_dir = dataset_dir / "npz"
    meta_dir = dataset_dir / "meta"
    for npz_path in sorted(npz_dir.glob("*.npz")):
        sid = npz_path.stem
        meta_path = meta_dir / f"{sid}.json"
        meta: dict[str, Any] = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        _data, diag = clean_raw_npz(npz_path, meta=meta)
        diag["scenario_id"] = sid
        diag["label"] = str(meta.get("label", ""))
        diag["failure_reason"] = str(meta.get("failure_reason", ""))
        diagnostics.append(diag)
    summary = summarize_diagnostics(diagnostics)
    _RAW_FLATLINE_AUDIT_CACHE[cache_key] = summary
    return summary


def load_highorder_dataset(quick: bool = False) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train, val, test, diag = prepare_centered_dataset()
    if quick:
        train = train[:8]
        val = val[:4]
        test = test[:4]
    return train, val, test, diag


def protocol_a_dataset(quick: bool = False, clean_flatlines: bool = True) -> dict[str, Any]:
    train, val, test, diag = load_highorder_dataset(quick=quick)
    v1_rows, v1_seqs = load_external_dataset(V1_DIR, clean_flatlines=clean_flatlines)
    v1_rows, v1_seqs = limit_labeled(v1_rows, v1_seqs, quick=quick, per_label=8)
    train, train_diag = _clean_sequences_only(train, "A", "A", "train", clean_flatlines, default_label="stable")
    val, val_diag = _clean_sequences_only(val, "A", "A", "val", clean_flatlines, default_label="stable")
    test, test_diag = _clean_sequences_only(test, "A", "A", "test", clean_flatlines, default_label="stable")
    v1_rows, v1_seqs, external_diag = _clean_rows_and_sequences(
        v1_rows, v1_seqs, "A", "A", "external", clean_flatlines
    )
    cleaning_diagnostics = [*train_diag, *val_diag, *test_diag, *external_diag]
    return {
        "protocol": "A",
        "name": "Protocol A Blind External Generalization",
        "clean_flatlines": clean_flatlines,
        "train": train,
        "val": val,
        "test": test,
        "external_rows": v1_rows,
        "external": v1_seqs,
        "diag": diag,
        "cleaning_diagnostics": cleaning_diagnostics,
        "cleaning_summary": summarize_diagnostics(cleaning_diagnostics) if clean_flatlines else {},
    }


def protocol_b_datasets(quick: bool = False, clean_flatlines: bool = True) -> list[dict[str, Any]]:
    dense_rows, dense_seqs = load_external_dataset(DENSE_DIR, clean_flatlines=clean_flatlines)
    by_id = {row["scenario_id"]: (row, seq) for row, seq in zip(dense_rows, dense_seqs)}
    # Preserve case_results order, because meta json ordering can differ on some filesystems.
    ordered: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in read_case_results(DENSE_DIR):
        item = by_id.get(row["scenario_id"])
        if item is not None:
            merged = {**item[0], **row}
            item[1]["true_label"] = merged.get("label", "")
            ordered.append((merged, item[1]))
    if ordered:
        dense_rows = [x[0] for x in ordered]
        dense_seqs = [x[1] for x in ordered]

    selected_splits = ["B1"] if quick else list(PROTOCOL_B_SPLITS)
    out: list[dict[str, Any]] = []
    for split_id in selected_splits:
        spec = PROTOCOL_B_SPLITS[split_id]
        part: dict[str, list[Any]] = {"train_rows": [], "train": [], "val_rows": [], "val": [], "test_rows": [], "test": []}
        for row, seq in zip(dense_rows, dense_seqs):
            line = row.get("fault_line")
            dest = None
            if line in spec["train"]:
                dest = "train"
            elif line in spec["val"]:
                dest = "val"
            elif line in spec["test"]:
                dest = "test"
            if dest is None:
                continue
            part[f"{dest}_rows"].append(row)
            part[dest].append(seq)
        if quick:
            part["train_rows"], part["train"] = limit_labeled(part["train_rows"], part["train"], True, per_label=8)
            part["val_rows"], part["val"] = limit_labeled(part["val_rows"], part["val"], True, per_label=4)
            part["test_rows"], part["test"] = limit_labeled(part["test_rows"], part["test"], True, per_label=8)
        train_rows, train, train_diag = _clean_rows_and_sequences(
            part["train_rows"], part["train"], "B", split_id, "train", clean_flatlines
        )
        val_rows, val, val_diag = _clean_rows_and_sequences(
            part["val_rows"], part["val"], "B", split_id, "val", clean_flatlines
        )
        test_rows, test, test_diag = _clean_rows_and_sequences(
            part["test_rows"], part["test"], "B", split_id, "test", clean_flatlines
        )
        part = {
            **part,
            "train_rows": train_rows,
            "train": train,
            "val_rows": val_rows,
            "val": val,
            "test_rows": test_rows,
            "test": test,
        }
        cleaning_diagnostics = [*train_diag, *val_diag, *test_diag]
        out.append({
            "protocol": "B",
            "name": "Protocol B Boundary Adaptation Dense Fault-Line Group Split",
            "split_id": split_id,
            "split_spec": spec,
            "clean_flatlines": clean_flatlines,
            "cleaning_diagnostics": cleaning_diagnostics,
            "cleaning_summary": summarize_diagnostics(cleaning_diagnostics) if clean_flatlines else {},
            **part,
        })
    return out


def variant_sequences(seqs: list[dict[str, Any]], variant: str) -> list[dict[str, Any]]:
    if variant == "clustered":
        return aggregate_sequences(seqs, ACTIVE_GROUPS, M_PRIOR)
    if variant == "unclustered":
        return seqs
    raise ValueError(f"Unknown input variant: {variant}")


def variant_triplet(train: list[dict[str, Any]], val: list[dict[str, Any]], test: list[dict[str, Any]], variant: str):
    return variant_sequences(train, variant), variant_sequences(val, variant), variant_sequences(test, variant)


def dataset_audit(quick: bool = False) -> dict[str, Any]:
    high_train, high_val, high_test, _ = load_highorder_dataset(quick=False)
    v1_rows = read_case_results(V1_DIR)
    dense_rows = read_case_results(DENSE_DIR)
    v1_ids = {r["scenario_id"] for r in v1_rows}
    dense_ids = {r["scenario_id"] for r in dense_rows}
    high_lines = sorted({
        str(seq.get("meta", {}).get("fault_line", seq.get("meta", {}).get("line", "")))
        for seq in [*high_train, *high_val, *high_test]
        if seq.get("meta")
    })
    external_lines = sorted({str(r.get("fault_line", "")) for r in v1_rows})
    overlap_lines = sorted(set(high_lines) & set(external_lines))
    split_audits = []
    for split_id, spec in PROTOCOL_B_SPLITS.items():
        train = set(spec["train"])
        val = set(spec["val"])
        test = set(spec["test"])
        split_audits.append({
            "split_id": split_id,
            "train_lines": sorted(train),
            "val_lines": sorted(val),
            "test_lines": sorted(test),
            "train_val_overlap": sorted(train & val),
            "train_test_overlap": sorted(train & test),
            "val_test_overlap": sorted(val & test),
        })
    nc_v1 = {r["scenario_id"] for r in v1_rows if r.get("label") == "near_critical"}
    nc_dense = {r["scenario_id"] for r in dense_rows if r.get("label") == "near_critical"}
    return {
        "quick": quick,
        "highorder_path": str(HIGHORDER_DIR),
        "v1_path": str(V1_DIR),
        "dense_path": str(DENSE_DIR),
        "highorder_counts": {"train": len(high_train), "val": len(high_val), "test": len(high_test)},
        "v1_counts": {"total": len(v1_rows), **label_counts(v1_rows)},
        "dense_counts": {"total": len(dense_rows), **label_counts(dense_rows)},
        "dense_contains_v1_cases": len(v1_ids & dense_ids),
        "v1_not_in_dense": len(v1_ids - dense_ids),
        "dense_extra_cases": len(dense_ids - v1_ids),
        "near_critical_overlap_v1_dense": len(nc_v1 & nc_dense),
        "highorder_external_fault_line_overlap": overlap_lines,
        "dense_fault_line_counts": fault_line_counts(dense_rows),
        "flatline_cleaning_default": True,
        "v1_flatline_audit": raw_flatline_audit(V1_DIR),
        "dense_flatline_audit": raw_flatline_audit(DENSE_DIR),
        "cleaning_note": "Protocol A/B loaders truncate machine-precision ANDES flatline tails by default; diagnostics are audit-only and not model inputs.",
        "protocol_b_splits": split_audits,
        "note": "Dense contains all v1 cases, so Protocol B never uses v1 as a held-out test set.",
    }


def assert_protocol_b_disjoint() -> None:
    for split_id, spec in PROTOCOL_B_SPLITS.items():
        train, val, test = map(set, (spec["train"], spec["val"], spec["test"]))
        if train & val or train & test or val & test:
            raise AssertionError(f"Protocol B split {split_id} has overlapping fault lines")
