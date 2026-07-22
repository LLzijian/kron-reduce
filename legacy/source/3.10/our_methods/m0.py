"""M0 backbone fitting and rollout for Protocol A/B."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from support.common import PM_MODE, code_fingerprint, read_json, sequence_signature, write_json
from support.data_protocol import variant_sequences

from evaluate_kron_graph_sindy_ieee39 import fit_kron_graph_model, rollout


def fit_m0(train: list[dict[str, Any]], model_dir: Path, resume: bool, alpha: float = 1e-4, threshold: float = 1e-3):
    """Fit the L0 swing backbone on 7-area clustered training data."""
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.json"
    train_signature = sequence_signature(train)
    code_sig = code_fingerprint()
    if resume and model_path.exists() and model_path.stat().st_size > 0:
        meta = read_json(model_path)
        if meta.get("train_signature") == train_signature and meta.get("pm_mode") == PM_MODE and meta.get("code_fingerprint") == code_sig:
            from evaluate_8cluster_dataonly_engineering_ieee39 import load_model

            return load_model(model_path), model_path
    area_train = variant_sequences(train, "clustered")
    M_area = np.asarray(area_train[0]["M_group"], dtype=np.float64)
    model, stats = fit_kron_graph_model(
        area_train,
        M_area,
        resolution="7-area",
        level="L0",
        alpha=alpha,
        threshold=threshold,
    )
    write_json(model_path, {
        "model": model.to_dict(),
        "fit_stats": stats,
        "pm_mode": PM_MODE,
        "input_form": "clustered",
        "groups": area_train[0].get("groups", []),
        "train_signature": train_signature,
        "code_fingerprint": code_sig,
    })
    return model, model_path


def rollout_m0(model, seq: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return rollout(model, seq)
