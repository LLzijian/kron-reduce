"""Window-based supervised classifiers for Protocol B only."""

from __future__ import annotations

import random
from typing import Any

import numpy as np

from support.common import PM_MODE, binary_label, binary_metrics, dpm_array, state_array

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


if torch is not None:
    class WindowClassifier(nn.Module):
        def __init__(self, model_type: str, n_ch: int, win_len: int):
            super().__init__()
            self.model_type = model_type
            if model_type == "CNN":
                self.net = nn.Sequential(
                    nn.Conv1d(n_ch, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
                    nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
                )
                self.head = nn.Linear(64, 2)
            elif model_type == "LSTM":
                self.rnn = nn.LSTM(n_ch, 64, 2, batch_first=True, bidirectional=True, dropout=0.1)
                self.head = nn.Linear(128, 2)
            elif model_type == "CNN+LSTM":
                self.conv = nn.Sequential(nn.Conv1d(n_ch, 32, 5, padding=2), nn.ReLU())
                self.rnn = nn.LSTM(32, 64, 1, batch_first=True)
                self.head = nn.Linear(64, 2)
            elif model_type == "Transformer":
                d_model = 64
                self.proj = nn.Linear(n_ch, d_model)
                self.pos = nn.Parameter(torch.randn(1, win_len, d_model) * 0.02)
                layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True, dropout=0.1)
                self.enc = nn.TransformerEncoder(layer, num_layers=2)
                self.head = nn.Linear(d_model, 2)
            else:
                raise ValueError(model_type)

        def forward(self, x):
            if self.model_type == "CNN":
                return self.head(self.net(x).squeeze(-1))
            if self.model_type == "LSTM":
                h, _ = self.rnn(x.permute(0, 2, 1))
                return self.head(h[:, -1])
            if self.model_type == "CNN+LSTM":
                z = self.conv(x).permute(0, 2, 1)
                h, _ = self.rnn(z)
                return self.head(h[:, -1])
            z = x.permute(0, 2, 1)
            h = self.enc(self.proj(z) + self.pos[:, : z.shape[1]])
            return self.head(h[:, -1])


def _window(seq: dict[str, Any], window_len: int) -> np.ndarray:
    x = state_array(seq)
    u = dpm_array(seq)
    z = np.column_stack([x, u])
    win = z[:window_len].T
    return ((win - win.mean(axis=1, keepdims=True)) / (win.std(axis=1, keepdims=True) + 1e-8)).astype(np.float32)


def _cleaning_fields(row: dict[str, Any]) -> dict[str, Any]:
    diag = row.get("_clean_diag", {})
    if not isinstance(diag, dict):
        diag = {}
    return {
        "cleaned_has_flatline": int(bool(diag.get("has_flatline", False))),
        "cleaned_valid_steps": diag.get("valid_steps", ""),
        "cleaned_total_steps": diag.get("total_steps", ""),
        "cleaned_divergence_time": diag.get("divergence_time", ""),
        "cleaned_flatline_fraction": diag.get("flatline_fraction", ""),
    }


def _dataset(rows: list[dict[str, Any]], seqs: list[dict[str, Any]], window_len: int):
    xs: list[np.ndarray] = []
    ys: list[int] = []
    ids: list[str] = []
    kept_rows: list[dict[str, Any]] = []
    for row, seq in zip(rows, seqs):
        lab = binary_label(row.get("label"))
        if lab is None or len(seq["t"]) < window_len:
            continue
        xs.append(_window(seq, window_len))
        ys.append(int(lab))
        ids.append(row["scenario_id"])
        kept_rows.append(row)
    return xs, ys, ids, kept_rows


def run_classifier(
    model_type: str,
    train_rows: list[dict[str, Any]],
    train_seqs: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    test_seqs: list[dict[str, Any]],
    protocol: str,
    split_id: str,
    variant: str,
    seed: int,
    window_len: int,
    quick: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if torch is None:
        return {
            "protocol": protocol, "split_id": split_id, "method": model_type, "family": "direct_classifier",
            "variant": variant, "window_len": window_len, "seed": seed, "status": "skipped",
            "pm_mode": PM_MODE, "error": "torch unavailable",
        }, []
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        xs_tr, ys_tr, _ids_tr, _rows_tr = _dataset(train_rows, train_seqs, window_len)
        xs_te, ys_te, ids_te, rows_te = _dataset(test_rows, test_seqs, window_len)
        binary_train = [r for r in train_rows if binary_label(r.get("label")) is not None]
        binary_test = [r for r in test_rows if binary_label(r.get("label")) is not None]
        dropped_train_short = len(binary_train) - len(xs_tr)
        dropped_test_short = len(binary_test) - len(xs_te)
        if len(xs_tr) < 4 or len(xs_te) < 1 or len(set(ys_tr)) < 2 or len(set(ys_te)) < 2:
            return {
                "protocol": protocol, "split_id": split_id, "method": model_type, "family": "direct_classifier",
                "variant": variant, "window_len": window_len, "seed": seed, "status": "skipped",
                "pm_mode": PM_MODE, "train_cases": len(binary_train), "test_cases": len(binary_test),
                "evaluated_train_cases": len(xs_tr), "evaluated_test_cases": len(xs_te),
                "dropped_train_short_cases": dropped_train_short,
                "dropped_test_short_cases": dropped_test_short,
                "error": "insufficient binary train/test coverage after valid-window truncation",
            }, []
        Xtr = torch.tensor(np.stack(xs_tr), dtype=torch.float32)
        ytr = torch.tensor(np.asarray(ys_tr, dtype=np.int64))
        Xte = torch.tensor(np.stack(xs_te), dtype=torch.float32)
        model = WindowClassifier(model_type, Xtr.shape[1], window_len)
        counts = np.bincount(ys_tr, minlength=2)
        weights = torch.tensor(len(ys_tr) / (2 * np.maximum(counts, 1)), dtype=torch.float32)
        crit = nn.CrossEntropyLoss(weight=weights)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        epochs = 8 if quick else 30
        idx_all = torch.arange(len(Xtr), dtype=torch.long)
        for _ in range(epochs):
            perm = idx_all[torch.randperm(len(idx_all))]
            for b in range(0, len(perm), 32):
                idx = perm[b:b + 32]
                loss = crit(model(Xtr[idx]), ytr[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
        with torch.no_grad():
            logits = model(Xte)
            probs = torch.softmax(logits, dim=1)[:, 1].numpy()
            preds = logits.argmax(1).numpy().astype(int).tolist()
        metrics = binary_metrics(ys_te, preds)
        per_case = []
        for row, sid, true, pred, prob in zip(rows_te, ids_te, ys_te, preds, probs):
            per_case.append({
                "protocol": protocol,
                "split_id": split_id,
                "method": model_type,
                "family": "direct_classifier",
                "variant": variant,
                "window_len": window_len,
                "seed": seed,
                "pm_mode": PM_MODE,
                "eval_kind": "stability",
                "scenario_id": sid,
                "fault_line": row.get("fault_line", ""),
                "fault_duration": row.get("fault_duration", ""),
                "true_label": "unstable" if true == 1 else "stable",
                "pred_label": "unstable" if pred == 1 else "stable",
                "score_unstable": float(prob),
                **_cleaning_fields(row),
            })
        summary = {
            "protocol": protocol,
            "split_id": split_id,
            "method": model_type,
            "family": "direct_classifier",
            "variant": variant,
            "window_len": window_len,
            "seed": seed,
            "status": "ok",
            "pm_mode": PM_MODE,
            "train_cases": len(binary_train),
            "test_cases": len(binary_test),
            "evaluated_train_cases": len(xs_tr),
            "evaluated_test_cases": len(xs_te),
            "dropped_train_short_cases": dropped_train_short,
            "dropped_test_short_cases": dropped_test_short,
            "non_binary_total": sum(1 for r in test_rows if binary_label(r.get("label")) is None),
            **metrics,
        }
        return summary, per_case
    except Exception as exc:
        return {
            "protocol": protocol, "split_id": split_id, "method": model_type, "family": "direct_classifier",
            "variant": variant, "window_len": window_len, "seed": seed, "status": "failed",
            "pm_mode": PM_MODE, "error": str(exc)[:240],
        }, []
