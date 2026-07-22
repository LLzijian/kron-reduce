from __future__ import annotations

import json
from pathlib import Path

from .schema import CanonicalSplit


VIEW_SPLITS = {
    "general_dynamics": "general_line_5fold_v1",
    "binary_stability": "stability_line_5fold_v1",
    "combined_dynamics": "combined_line_5fold_v1",
}


def load_split(
    dataset_path: str | Path,
    *,
    view: str,
    split_name: str,
    fold: str,
) -> CanonicalSplit:
    dataset_path = Path(dataset_path)
    expected = VIEW_SPLITS.get(view)
    if expected is None or expected != split_name:
        raise ValueError(f"view {view!r} requires split {expected!r}, got {split_name!r}")
    split_data = json.loads(
        (dataset_path / "splits" / f"{split_name}.json").read_text(encoding="utf-8")
    )
    if fold not in split_data:
        raise KeyError(f"unknown fold {fold!r}")
    row = split_data[fold]
    roles = {name: tuple(row[name]) for name in ("train", "val", "test")}
    if any(set(roles[a]) & set(roles[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError(f"event leakage in {split_name}/{fold}")
    prefix = split_name.split("_line_")[0]
    calibration_path = (
        dataset_path / "splits" / "disturbance_calibration" / f"{prefix}_{fold}.json"
    )
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    return CanonicalSplit(
        view=view,
        split_name=split_name,
        fold=fold,
        train_ids=roles["train"],
        val_ids=roles["val"],
        test_ids=roles["test"],
        disturbance_calibration=calibration,
    )
