"""Shared utilities for the final 3.10 Protocol A/B comparison."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "AGENTS.md").exists() and (parent / "discovery").exists():
            return parent
    raise RuntimeError(f"Could not locate project root from {here}")


PROJECT_ROOT = find_project_root()
DISCOVERY_DIR = PROJECT_ROOT / "discovery"
KRON_DIR = DISCOVERY_DIR / "3.2-kron"
for _path in (DISCOVERY_DIR, KRON_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

RESULT_DIR = PROJECT_ROOT / "results/03_kron_graph_sindy/02_cluster/3.10-final_paper_comparison_protocol_ab"
QUICK_RESULT_DIR = RESULT_DIR.with_name(f"{RESULT_DIR.name}_quick_smoke")
REPORT_PATH = PROJECT_ROOT / "reports/hist_3.10-Final_Paper_Comparison.md"
QUICK_REPORT_PATH = PROJECT_ROOT / "reports/hist_3.10-Final_Paper_Comparison_quick_smoke.md"
V1_DIR = PROJECT_ROOT / "data/external_sim/andes/ieee39_near_critical_unstable_v1"
DENSE_DIR = PROJECT_ROOT / "data/external_sim/andes/ieee39_near_critical_unstable_v1_dense"
HIGHORDER_DIR = PROJECT_ROOT / "data/external_sim/andes/ieee39_highorder_dataset"

SEEDS_FULL = [324, 325, 326, 327, 328]
SEEDS_SMOKE = [324]
WINDOWS_FULL = [20, 40, 100, 200]
WINDOWS_SMOKE = [20, 40]

ACTIVE_GROUPS = [[0, 7], [1, 2], [3], [4], [5, 6], [8], [9]]
PM_MODE = "Pm(t) observed exogenous input"


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return str(obj)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    if fieldnames is None:
        preferred = [
            "protocol", "split_id", "model", "method", "family", "variant", "window_len", "seed",
            "status", "pm_mode", "train_cases", "val_cases", "test_cases",
            "tp", "fn", "fp", "tn", "binary_total", "non_binary_total",
            "FSR", "FUR", "Accuracy", "BalancedAccuracy", "AUC",
            "rollout_failed_cases", "rollout_failure_rate",
            "rollout_r2", "closed_loop_r2", "forecast_r2", "worst_case_r2",
            "mean_delta_r2", "mean_omega_r2", "rmse", "failure_rate",
            "mean_FSR", "std_FSR", "mean_FUR", "std_FUR",
            "mean_Accuracy", "std_Accuracy", "mean_BalancedAccuracy", "std_BalancedAccuracy",
            "mean_rollout_r2", "std_rollout_r2", "mean_rmse", "std_rmse", "n_runs",
        ]
        keys = {k for row in rows for k in row.keys()}
        fieldnames = [k for k in preferred if k in keys] + sorted(keys - set(preferred))
    if not rows:
        tmp_path.write_text(",".join(fieldnames or []) + "\n", encoding="utf-8")
        for attempt in range(5):
            try:
                tmp_path.replace(path)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.5)
        return
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    for attempt in range(5):
        try:
            tmp_path.replace(path)
            break
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.5)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def code_fingerprint() -> str:
    """Fingerprint this harness and the wrapped local trainers that affect results."""
    paths = sorted((PROJECT_ROOT / "discovery/3.2-kron/hist_310_final_paper_comparison").rglob("*.py"))
    paths.extend([
        PROJECT_ROOT / "discovery/3.2-kron/diag_clean_unstable_trajectories.py",
        PROJECT_ROOT / "discovery/3.2-kron/hist_run_two_tier_batch.py",
        PROJECT_ROOT / "discovery/3.2-kron/hist_run_384_m2a_lyapunov.py",
        PROJECT_ROOT / "discovery/3.2-kron/hist_models_forward_residual_v3.py",
        PROJECT_ROOT / "discovery/evaluate_kron_graph_sindy_ieee39.py",
        PROJECT_ROOT / "discovery/3.2-kron/diag_evaluate_external_cct_v3.py",
        PROJECT_ROOT / "discovery/run_8cluster_forward_residual_sindy_ieee39.py",
        PROJECT_ROOT / "discovery/run_forward_residual_graphsindy_v324_ieee39.py",
    ])
    h = hashlib.sha256()
    for path in sorted({p.resolve() for p in paths if p.exists() and p.is_file()}, key=lambda p: str(p)):
        h.update(str(path.relative_to(PROJECT_ROOT)).replace("\\", "/").encode("utf-8"))
        h.update(b"\0")
        h.update(sha256_file(path).encode("ascii"))
        h.update(b"\0")
    return h.hexdigest()


def dataset_fingerprint() -> str:
    """Cheap fingerprint for dataset inventories used by Protocol A/B."""
    paths = [
        V1_DIR / "case_results.csv",
        V1_DIR / "dataset_summary.json",
        DENSE_DIR / "case_results.csv",
        DENSE_DIR / "dataset_summary.json",
        HIGHORDER_DIR / "dataset_summary.json",
        HIGHORDER_DIR / "generation_log.txt",
    ]
    h = hashlib.sha256()
    for path in paths:
        h.update(str(path.relative_to(PROJECT_ROOT)).replace("\\", "/").encode("utf-8"))
        h.update(b"\0")
        if path.exists() and path.is_file():
            h.update(str(path.stat().st_size).encode("ascii"))
            h.update(b":")
            h.update(str(int(path.stat().st_mtime_ns)).encode("ascii"))
            h.update(b":")
            h.update(sha256_file(path).encode("ascii"))
        else:
            h.update(b"missing")
        h.update(b"\0")
    return h.hexdigest()


def sequence_signature(seqs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sig = []
    for i, seq in enumerate(seqs):
        t = np.asarray(seq.get("t", []), dtype=np.float64)
        delta = np.asarray(seq.get("delta", []))
        omega = np.asarray(seq.get("omega", []))
        sig.append({
            "scenario_id": str(seq.get("scenario_id", seq.get("meta", {}).get("scenario_id", i))),
            "n_t": int(len(t)),
            "t0": float(t[0]) if len(t) else None,
            "t1": float(t[-1]) if len(t) else None,
            "delta_shape": list(delta.shape),
            "omega_shape": list(omega.shape),
        })
    return sig


def cache_token(parts: Iterable[Any]) -> str:
    raw = json.dumps(list(parts), sort_keys=True, default=json_default)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def load_or_run_json(cache_dir: Path, parts: Iterable[Any], resume: bool, compute):
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{cache_token(parts)}.json"
    label = "/".join(str(p) for p in parts)
    if resume and path.exists() and path.stat().st_size > 0:
        print(f"[cache] {label}", flush=True)
        return read_json(path)
    print(f"[run] {label}", flush=True)
    result = compute()
    write_json(path, result)
    return result


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        val = float(value)
        return val if math.isfinite(val) else default
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    n = min(len(y_true), len(y_pred))
    if n <= 2:
        return float("nan")
    y_true = y_true[:n]
    y_pred = y_pred[:n]
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true, axis=0, keepdims=True)) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-15)


def rmse_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    n = min(len(y_true), len(y_pred))
    if n <= 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true[:n] - y_pred[:n]) ** 2)))


def state_array(seq: dict[str, Any], include_pm: bool = False) -> np.ndarray:
    delta = np.asarray(seq["delta"], dtype=np.float64)
    omega = np.asarray(seq["omega"], dtype=np.float64)
    if not include_pm:
        return np.column_stack([delta, omega])
    pm = dpm_array(seq)
    return np.column_stack([delta, omega, pm])


def dpm_array(seq: dict[str, Any]) -> np.ndarray:
    tm = np.asarray(seq.get("tm"), dtype=np.float64)
    pm0 = np.asarray(seq["eq"]["Pm0"], dtype=np.float64)
    return tm - pm0[np.newaxis, :]


def trajectory_metrics(delta_true: np.ndarray, omega_true: np.ndarray, delta_pred: np.ndarray, omega_pred: np.ndarray) -> dict[str, float]:
    n = min(len(delta_true), len(delta_pred), len(omega_true), len(omega_pred))
    dt_true = np.asarray(delta_true[:n], dtype=np.float64)
    wt_true = np.asarray(omega_true[:n], dtype=np.float64)
    dt_pred = np.asarray(delta_pred[:n], dtype=np.float64)
    wt_pred = np.asarray(omega_pred[:n], dtype=np.float64)
    return {
        "delta_r2": r2_score_np(dt_true, dt_pred),
        "omega_r2": r2_score_np(wt_true, wt_pred),
        "state_r2": r2_score_np(np.column_stack([dt_true, wt_true]), np.column_stack([dt_pred, wt_pred])),
        "delta_rmse": rmse_np(dt_true, dt_pred),
        "omega_rmse": rmse_np(wt_true, wt_pred),
        "state_rmse": rmse_np(np.column_stack([dt_true, wt_true]), np.column_stack([dt_pred, wt_pred])),
    }


def binary_label(label: str | None) -> int | None:
    if label == "unstable":
        return 1
    if label == "stable":
        return 0
    return None


def binary_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    total = tp + fn + fp + tn
    tpr = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "binary_total": total,
        "FSR": fn / max(tp + fn, 1),
        "FUR": fp / max(fp + tn, 1),
        "Accuracy": (tp + tn) / max(total, 1),
        "BalancedAccuracy": 0.5 * (tpr + tnr),
    }


def auc_score(scores: list[float], labels: list[int]) -> float:
    score_arr = np.asarray(scores, dtype=np.float64)
    label_arr = np.asarray(labels, dtype=np.int64)
    pos = score_arr[label_arr == 1]
    neg = score_arr[label_arr == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = 0.0
    for p in pos:
        wins += float(np.sum(p > neg)) + 0.5 * float(np.sum(p == neg))
    return wins / max(len(pos) * len(neg), 1)


def aggregate_runs(rows: list[dict[str, Any]], group_keys: list[str], metric_keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") not in ("ok", "degraded"):
            continue
        key = tuple(row.get(k, "") for k in group_keys)
        groups.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for key, items in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        base = {k: v for k, v in zip(group_keys, key)}
        base["n_runs"] = len(items)
        for metric in metric_keys:
            vals = [safe_float(r.get(metric)) for r in items]
            vals = [v for v in vals if math.isfinite(v)]
            if vals:
                base[f"mean_{metric}"] = float(np.mean(vals))
                base[f"std_{metric}"] = float(np.std(vals, ddof=0))
                base[f"min_{metric}"] = float(np.min(vals))
                base[f"max_{metric}"] = float(np.max(vals))
            else:
                base[f"mean_{metric}"] = float("nan")
                base[f"std_{metric}"] = float("nan")
        out.append(base)
    return out


def fmt_num(value: Any, digits: int = 4) -> str:
    val = safe_float(value)
    if not math.isfinite(val):
        return "-"
    if abs(val) >= 1.0e4 or (0 < abs(val) < 1.0e-3):
        return f"{val:.{digits}e}"
    return f"{val:.{digits}f}"


def mean_std_text(row: dict[str, Any], metric: str) -> str:
    return f"{fmt_num(row.get(f'mean_{metric}'))} ± {fmt_num(row.get(f'std_{metric}'))}"
