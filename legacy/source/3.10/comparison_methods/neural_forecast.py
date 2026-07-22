"""LSTM rollout baseline with observed future Pm(t) as exogenous input."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from support.common import dpm_array, r2_score_np, rmse_np, state_array

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


if torch is not None:
    class ExogenousLstmRollout(nn.Module):
        def __init__(self, x_dim: int, u_dim: int, hidden: int = 64):
            super().__init__()
            self.encoder = nn.LSTM(x_dim + u_dim, hidden, batch_first=True)
            self.cell = nn.LSTMCell(x_dim + u_dim, hidden)
            self.head = nn.Linear(hidden, x_dim)

        def forward(self, prefix_x, prefix_u, future_u):
            enc_in = torch.cat([prefix_x, prefix_u], dim=-1)
            _, (h, c) = self.encoder(enc_in)
            h = h[-1]
            c = c[-1]
            x = prefix_x[:, -1, :]
            ys = []
            for k in range(future_u.shape[1]):
                inp = torch.cat([x, future_u[:, k]], dim=-1)
                h, c = self.cell(inp, (h, c))
                dx = self.head(h)
                x = x + dx
                ys.append(x)
            return torch.stack(ys, dim=1)


@dataclass
class LstmBundle:
    model: Any
    x_mean: Any
    x_std: Any
    u_mean: Any
    u_std: Any
    x_dim: int
    u_dim: int


def fit_lstm_rollout(train: list[dict[str, Any]], seed: int, window_len: int, quick: bool) -> LstmBundle:
    if torch is None:
        raise RuntimeError("torch unavailable")
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    xs = [torch.tensor(state_array(s), dtype=torch.float32) for s in train]
    us = [torch.tensor(dpm_array(s), dtype=torch.float32) for s in train]
    x_all = torch.cat(xs[: min(len(xs), 20)], dim=0)
    u_all = torch.cat(us[: min(len(us), 20)], dim=0)
    x_mean, x_std = x_all.mean(0), x_all.std(0) + 1e-6
    u_mean, u_std = u_all.mean(0), u_all.std(0) + 1e-6
    hidden = 32 if quick else 64
    model = ExogenousLstmRollout(xs[0].shape[1], us[0].shape[1], hidden=hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    epochs = 8 if quick else 30
    for _ in range(epochs):
        seq_idx = random.randrange(len(xs))
        x = xs[seq_idx]
        u = us[seq_idx]
        if len(x) < window_len + 12:
            continue
        horizon = min(random.randint(8, 45), len(x) - window_len - 1)
        start = random.randrange(0, len(x) - window_len - horizon)
        px = (x[start:start + window_len] - x_mean) / x_std
        pu = (u[start:start + window_len] - u_mean) / u_std
        fu = (u[start + window_len:start + window_len + horizon] - u_mean) / u_std
        target = (x[start + window_len:start + window_len + horizon] - x_mean) / x_std
        pred = model(px.unsqueeze(0), pu.unsqueeze(0), fu.unsqueeze(0)).squeeze(0)
        loss = torch.mean((pred - target) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
    return LstmBundle(model=model, x_mean=x_mean, x_std=x_std, u_mean=u_mean, u_std=u_std, x_dim=xs[0].shape[1], u_dim=us[0].shape[1])


def rollout_lstm(bundle: LstmBundle, seq: dict[str, Any], window_len: int) -> tuple[np.ndarray, np.ndarray]:
    if torch is None:
        raise RuntimeError("torch unavailable")
    x_true = torch.tensor(state_array(seq), dtype=torch.float32)
    u_true = torch.tensor(dpm_array(seq), dtype=torch.float32)
    start = min(int(window_len), len(x_true) - 2)
    horizon = len(x_true) - start
    with torch.no_grad():
        px = (x_true[:start] - bundle.x_mean) / bundle.x_std
        pu = (u_true[:start] - bundle.u_mean) / bundle.u_std
        fu = (u_true[start:] - bundle.u_mean) / bundle.u_std
        pred_future = bundle.model(px.unsqueeze(0), pu.unsqueeze(0), fu.unsqueeze(0)).squeeze(0)
        pred_future = pred_future * bundle.x_std + bundle.x_mean
    pred = x_true.clone()
    pred[start:] = pred_future
    arr = pred.numpy()
    n = arr.shape[1] // 2
    if not np.all(np.isfinite(arr)) or np.max(np.abs(arr)) > 1e6:
        raise RuntimeError("LSTM rollout diverged")
    return arr[:, :n], arr[:, n:]


def rollout_lstm_many(
    bundle: LstmBundle,
    seqs: list[dict[str, Any]],
    window_len: int,
    batch_size: int = 64,
) -> list[tuple[np.ndarray, np.ndarray] | Exception]:
    """Batch LSTM rollout evaluation for equal-length trajectory groups."""
    if torch is None:
        raise RuntimeError("torch unavailable")
    out: list[tuple[np.ndarray, np.ndarray] | Exception] = [RuntimeError("not evaluated")] * len(seqs)
    groups: dict[tuple[int, int, int], list[int]] = {}
    for i, seq in enumerate(seqs):
        try:
            x = state_array(seq)
            start = min(int(window_len), len(x) - 2)
            groups.setdefault((len(x), start, x.shape[1]), []).append(i)
        except Exception as exc:
            out[i] = exc
    bundle.model.eval()
    with torch.no_grad():
        for _key, idxs in groups.items():
            for b0 in range(0, len(idxs), batch_size):
                batch_ids = idxs[b0:b0 + batch_size]
                try:
                    xs = [state_array(seqs[i]) for i in batch_ids]
                    us = [dpm_array(seqs[i]) for i in batch_ids]
                    x_true = torch.tensor(np.stack(xs), dtype=torch.float32)
                    u_true = torch.tensor(np.stack(us), dtype=torch.float32)
                    start = min(int(window_len), x_true.shape[1] - 2)
                    px = (x_true[:, :start] - bundle.x_mean) / bundle.x_std
                    pu = (u_true[:, :start] - bundle.u_mean) / bundle.u_std
                    fu = (u_true[:, start:] - bundle.u_mean) / bundle.u_std
                    pred_future = bundle.model(px, pu, fu)
                    pred_future = pred_future * bundle.x_std + bundle.x_mean
                    pred = x_true.clone()
                    pred[:, start:] = pred_future
                    arr = pred.numpy()
                    n = arr.shape[2] // 2
                    for local_i, global_i in enumerate(batch_ids):
                        one = arr[local_i]
                        if not np.all(np.isfinite(one)) or np.max(np.abs(one)) > 1e6:
                            out[global_i] = RuntimeError("LSTM rollout diverged")
                        else:
                            out[global_i] = (one[:, :n], one[:, n:])
                except Exception as exc:
                    for global_i in batch_ids:
                        out[global_i] = exc
    return out
