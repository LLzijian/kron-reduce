"""M2A training wrapper using the existing 3.7 residual Graph-SINDy trainer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from support.common import PM_MODE, code_fingerprint, sequence_signature, sha256_file, read_json, write_json

import hist_run_two_tier_batch as two_tier
from hist_models_forward_residual_v3 import make_mask
from run_8cluster_forward_residual_sindy_ieee39 import feature_names


def train_or_load_m2a(
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    test: list[dict[str, Any]],
    seed: int,
    model_dir: Path,
    m0_model_path: Path,
    epochs: int,
    resume: bool,
) -> dict[str, Any]:
    model_dir.mkdir(parents=True, exist_ok=True)
    result_path = model_dir / f"seed{seed}_result.json"
    params_path = model_dir / f"seed{seed}_params.npz"
    cache_signature = {
        "train_signature": sequence_signature(train),
        "val_signature": sequence_signature(val),
        "test_signature": sequence_signature(test),
        "m0_sha256": sha256_file(m0_model_path) if m0_model_path.exists() else "",
        "code_fingerprint": code_fingerprint(),
    }
    if resume and result_path.exists() and params_path.exists():
        meta = read_json(result_path)
        if (
            int(meta.get("epochs", -1)) == int(epochs)
            and meta.get("pm_mode") == PM_MODE
            and meta.get("cache_signature") == cache_signature
        ):
            params = np.load(params_path)["params_c4"]
            return {**meta, "params_c4": params}

    old_get_data = two_tier.get_data
    old_m0_path = two_tier.M0_PATH
    try:
        two_tier.get_data = lambda _tier: (train, val, test)
        two_tier.M0_PATH = Path(str(m0_model_path))
        min_epochs = min(300, max(40, epochs // 5))
        patience = max(120, epochs // 2)
        result = two_tier.train_one("M2a", 1, seed, epochs=epochs, min_epochs=min_epochs, patience=patience)
    finally:
        two_tier.get_data = old_get_data
        two_tier.M0_PATH = old_m0_path

    np.savez(params_path, params_c4=result["params_c4"])
    meta = {k: v for k, v in result.items() if k != "params_c4"}
    meta.update({"method": "M2A", "seed": seed, "pm_mode": PM_MODE, "epochs": epochs, "cache_signature": cache_signature})
    write_json(result_path, meta)
    return {**meta, "params_c4": result["params_c4"]}


def rollout_m2a(params: np.ndarray, bm, seq: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return rollout_m2a_prepared(prepare_m2a_rollout(params), bm, seq)


def prepare_m2a_rollout(params: np.ndarray) -> dict[str, Any]:
    fnames = feature_names(two_tier.N_NODES, False)
    mask_full = make_mask(fnames, ["full"])
    theta = np.asarray(params, dtype=np.float64) * mask_full
    node_theta = theta[:two_tier.N_NODES * two_tier.NODE_FEAT_DIM].reshape(
        two_tier.N_NODES, two_tier.NODE_FEAT_DIM
    )
    edge_theta = theta[two_tier.N_NODES * two_tier.NODE_FEAT_DIM:].reshape(
        two_tier.N_EDGES, two_tier.EDGE_FEAT_DIM
    )
    return {
        "B": two_tier.incidence(two_tier.N_NODES),
        "node_theta": node_theta,
        "edge_theta": edge_theta,
    }


def rollout_m2a_prepared(prepared: dict[str, Any], bm, seq: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    B = prepared["B"]
    node_theta = prepared["node_theta"]
    edge_theta = prepared["edge_theta"]
    t = np.asarray(seq["t"], dtype=np.float64)
    da = np.asarray(seq["delta"], dtype=np.float64)
    wa = np.asarray(seq["omega"], dtype=np.float64)
    dpm = np.asarray(seq["tm"], dtype=np.float64) - np.asarray(seq["eq"]["Pm0"], dtype=np.float64)
    s0 = np.asarray(seq["eq"]["sin_eta0"], dtype=np.float64)
    c0_s = np.asarray(seq["eq"]["cos_eta0"], dtype=np.float64)
    dt_actual = float(np.mean(np.diff(t)))
    ddpm = np.gradient(dpm, dt_actual, axis=0, edge_order=2)
    dp = np.zeros_like(da)
    wp = np.zeros_like(wa)
    dp[0], wp[0] = da[0], wa[0]
    for k in range(len(t) - 1):
        dt_v = float(t[k + 1] - t[k])
        eta = dp[k] @ B
        dsin = np.sin(eta) - s0
        dcos = np.cos(eta) - c0_s
        gs = (dsin * bm.k_s) @ B.T
        dw = (two_tier.OMEGA_S * dpm[k] - bm.D * wp[k] - two_tier.OMEGA_S * gs) / bm.M
        wc = np.clip(wp[k], -50, 50)
        node_feat = np.column_stack([wc, np.abs(wc) * wc, wc ** 2, dpm[k], dpm[k] * wc, ddpm[k]])
        dw += np.sum(node_feat * node_theta, axis=-1)
        domega_e = wc @ B
        edge_feat = np.column_stack([
            dsin,
            dcos,
            domega_e,
            np.clip(domega_e * dsin, -100, 100),
            np.clip(domega_e * dcos, -100, 100),
        ])
        dw += np.sum(edge_feat * edge_theta, axis=-1) @ B.T
        wp[k + 1] = wp[k] + dt_v * dw
        dp[k + 1] = dp[k] + 0.5 * dt_v * (wp[k] + wp[k + 1])
        if not np.all(np.isfinite(dp[k + 1])) or not np.all(np.isfinite(wp[k + 1])):
            raise RuntimeError("non-finite")
        if np.max(np.abs(wp[k + 1])) > 1e6 or np.max(np.abs(dp[k + 1])) > 1e6:
            raise RuntimeError("diverged")
    return dp, wp
