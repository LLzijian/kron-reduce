"""M2A-LYAP training wrapper using the existing 3.8.4 implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from support.common import PM_MODE, code_fingerprint, sequence_signature, sha256_file, read_json, write_json

import hist_run_384_m2a_lyapunov as lyap


def train_or_load_m2a_lyap(
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

    old_get_data = lyap.get_data
    old_m0_path = lyap.M0_PATH
    try:
        lyap.get_data = lambda _tier: (train, val, test)
        lyap.M0_PATH = Path(str(m0_model_path))
        min_epochs = min(300, max(40, epochs // 5))
        patience = max(120, epochs // 2)
        result = lyap.train_one(seed, 1, epochs=epochs, min_epochs=min_epochs, patience=patience)
    finally:
        lyap.get_data = old_get_data
        lyap.M0_PATH = old_m0_path

    np.savez(params_path, params_c4=result["params_c4"])
    meta = {k: v for k, v in result.items() if k != "params_c4"}
    meta.update({"method": "M2A-LYAP", "seed": seed, "pm_mode": PM_MODE, "epochs": epochs, "cache_signature": cache_signature})
    write_json(result_path, meta)
    return {**meta, "params_c4": result["params_c4"]}


def rollout_m2a_lyap(params: np.ndarray, bm, seq: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return rollout_m2a_lyap_prepared(prepare_m2a_lyap_rollout(params), bm, seq)


def prepare_m2a_lyap_rollout(params: np.ndarray) -> dict[str, Any]:
    node_theta, edge_theta = lyap.split_theta_384(lyap.torch.tensor(params, dtype=lyap.torch.float64))
    return {
        "B": lyap.incidence(lyap.N_NODES),
        "node_theta": node_theta.detach().numpy(),
        "edge_theta": edge_theta.detach().numpy(),
    }


def rollout_m2a_lyap_prepared(prepared: dict[str, Any], bm, seq: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    B = prepared["B"]
    node_np = prepared["node_theta"]
    edge_np = prepared["edge_theta"]
    t_arr = np.asarray(seq["t"], dtype=np.float64)
    da = np.asarray(seq["delta"], dtype=np.float64)
    wa = np.asarray(seq["omega"], dtype=np.float64)
    tm = np.asarray(seq["tm"], dtype=np.float64)
    pm0 = np.asarray(seq["eq"]["Pm0"], dtype=np.float64)
    dpm = tm - pm0
    s0 = np.asarray(seq["eq"]["sin_eta0"], dtype=np.float64)
    c0 = np.asarray(seq["eq"]["cos_eta0"], dtype=np.float64)
    dt_actual = float(np.mean(np.diff(t_arr)))
    ddpm_arr = np.gradient(dpm, dt_actual, axis=0, edge_order=2)
    dp = np.zeros_like(da)
    wp = np.zeros_like(wa)
    dp[0], wp[0] = da[0], wa[0]

    for k in range(len(t_arr) - 1):
        dt_v = float(t_arr[k + 1] - t_arr[k])
        eta = dp[k] @ B
        es = np.sin(eta) - s0
        domega_e = wp[k] @ B
        gs = (es * bm.k_s) @ B.T
        dw = (lyap.OMEGA_S * dpm[k] - bm.D * wp[k] - lyap.OMEGA_S * gs) / bm.M
        wc = np.clip(wp[k], -50, 50)
        node_feat_np = np.column_stack([
            wc,
            np.abs(wc) * wc,
            wc ** 3,
            dpm[k],
            dpm[k] * wc,
            ddpm_arr[k],
        ])
        edge_feat_np = lyap.build_edge_features_np(eta, domega_e, s0, c0)
        r_node = np.sum(node_feat_np * node_np, axis=-1)
        edge_val = np.sum(edge_feat_np * edge_np, axis=-1)
        edge_force = edge_val @ B.T
        r_total = r_node + edge_force / bm.M
        r_total = r_total - np.sum(bm.M * r_total) / np.sum(bm.M)
        dw += r_total
        wp[k + 1] = wp[k] + dt_v * dw
        dp[k + 1] = dp[k] + 0.5 * dt_v * (wp[k] + wp[k + 1])
        if not np.all(np.isfinite(dp[k + 1])) or not np.all(np.isfinite(wp[k + 1])):
            raise RuntimeError("non-finite")
        if np.max(np.abs(wp[k + 1])) > 1e6 or np.max(np.abs(dp[k + 1])) > 1e6:
            raise RuntimeError("diverged")
    return dp, wp
