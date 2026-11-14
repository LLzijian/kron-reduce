"""Small batched RK4 and future loss for the paired state-port fit."""
from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import torch

from kron_reduce.models.state_port import StatePortClosure

DT = 0.01
HORIZONS = (1.0, 3.0)


def _float64_tensor(value, name, *, device="cpu"):
    if isinstance(value, torch.Tensor):
        if value.dtype != torch.float64 or value.device != torch.device(device):
            location = "CPU" if torch.device(device).type == "cpu" else str(device)
            raise ValueError(f"{name} must be {location} float64")
        return value
    return torch.as_tensor(value, dtype=torch.float64, device=device)


def _call_rhs(field, state, power, equilibrium, calls, location):
    if calls is not None:
        calls("attempted", batch_size=len(state), state=state.detach().numpy().copy(), **location)
    try:
        result = field(state, power, equilibrium)
    except BaseException as error:
        if calls is not None:
            calls("failed", batch_size=len(state), state=state.detach().numpy().copy(), error=error, **location)
        raise
    if calls is not None:
        calls("returned", batch_size=len(state), state=state.detach().numpy().copy(),
              result=result.detach().numpy().copy(), **location)
    return result


def rollout(model: StatePortClosure, initial, power, equilibrium, counts, *, max_step,
            delta=None, calls: Callable | None = None) -> torch.Tensor:
    """Roll out shared initial states, skipping each sample after its last target."""
    if not isinstance(model, StatePortClosure):
        raise TypeError("rollout requires a StatePortClosure")
    initial = _float64_tensor(initial, "initial")
    power = _float64_tensor(power, "power")
    equilibrium = _float64_tensor(equilibrium, "equilibrium")
    counts = np.asarray(counts)
    n, r = model.n_nodes, model.latent_dimension
    if (initial.ndim != 2 or initial.shape[1] != 2 * n + r or not len(initial)
            or power.shape != (len(initial), n) or equilibrium.shape != (len(initial), n)
            or counts.shape != (len(initial),) or counts.dtype.kind not in "iu"
            or np.any(counts < 1)):
        raise ValueError("initial, inputs, and positive integer counts must align by sample")
    if (not torch.isfinite(initial).all() or not torch.isfinite(power).all()
            or not torch.isfinite(equilibrium).all()):
        raise ValueError("rollout inputs must be finite")
    if torch.max(torch.abs(power.sum(dim=1))) > 1e-10:
        raise ValueError("power must be balanced for every sample")
    if not np.isfinite(max_step) or not 0 < max_step <= DT:
        raise ValueError("max_step must be finite and in (0, 0.01]")
    if calls is not None and not callable(calls):
        raise TypeError("calls must be callable")

    max_count = int(counts.max())
    clock = np.arange(max_count + 1, dtype=float) * DT
    expected_substeps = {0.01: 1, 0.005: 2, 0.0025: 4}.get(max_step)
    if expected_substeps is None or any(
            math.ceil(interval / max_step - 1e-12) != expected_substeps for interval in np.diff(clock)):
        raise ValueError("clock does not match the frozen RK4 substep counts")
    field = model.torch_vector_field(delta)
    mass = torch.tensor(model.mass, dtype=torch.float64)
    state = initial
    outputs = [state]
    for output_index, interval in enumerate(np.diff(clock), start=1):
        active_np = np.flatnonzero(counts >= output_index)
        active = torch.as_tensor(active_np, dtype=torch.long)
        current = state[active]
        count = max(1, math.ceil(interval / max_step - 1e-12))
        h = interval / count
        for substep in range(count):
            p, eq = power[active], equilibrium[active]
            location = {"active_indices": active_np, "output_index": output_index, "substep": substep, "h": h}
            k1 = _call_rhs(field, current, p, eq, calls, location | {"rk_stage": 1})
            k2 = _call_rhs(field, current + h * k1 / 2, p, eq, calls, location | {"rk_stage": 2})
            k3 = _call_rhs(field, current + h * k2 / 2, p, eq, calls, location | {"rk_stage": 3})
            k4 = _call_rhs(field, current + h * k3, p, eq, calls, location | {"rk_stage": 4})
            current = current + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            q = current[:, :n]
            v = current[:, n : 2 * n]
            q = q - (q @ mass)[:, None] / mass.sum()
            v = v - (v @ mass)[:, None] / mass.sum()
            current = torch.cat((q, v, current[:, 2 * n :]), dim=1)
            if not torch.isfinite(current).all() or torch.max(torch.abs(current)) > 1e12:
                raise FloatingPointError("nonfinite or divergent state-port rollout")
        state = state.index_copy(0, active, current)
        outputs.append(state)
    return torch.stack(outputs, dim=1)


def loss(path, target, weights, scale, counts, *, return_events=False):
    """Equal-weight 1 s/3 s event losses, with no score after each true prefix."""
    if type(return_events) is not bool:
        raise TypeError("return_events must be a bool")
    device = path.device if isinstance(path, torch.Tensor) else torch.device("cpu")
    path = _float64_tensor(path, "path", device=device)
    target = _float64_tensor(target, "target", device=device)
    weights = _float64_tensor(weights, "weights", device=device)
    scale = _float64_tensor(scale, "scale", device=device)
    counts = np.asarray(counts)
    if (path.ndim != 3 or target.ndim != 3 or path.shape[0] < 1
            or path.shape[:2] != target.shape[:2]
            or path.shape[0] != len(weights) or counts.shape != (len(weights),)
            or counts.dtype.kind not in "iu" or np.any(counts < 1)
            or np.any(counts > round(HORIZONS[-1] / DT))
            or target.shape[1] != int(counts.max()) + 1
            or path.shape[2] < target.shape[2] or target.shape[2] != scale.numel()
            or weights.shape != (path.shape[0],) or scale.ndim != 1
            or torch.any(weights < 0) or not torch.any(weights > 0)
            or not torch.isfinite(path).all() or not torch.isfinite(target).all()
            or not torch.isfinite(weights).all() or not torch.isfinite(scale).all()
            or torch.any(scale <= 0)):
        raise ValueError("path, target, weights, scale, and counts are not aligned finite inputs")
    visible = path[:, :, : target.shape[2]]
    normalized = ((visible[:, 1:] - target[:, 1:]) / scale).square().mean(dim=2)
    time_index = torch.arange(1, target.shape[1], dtype=torch.int64, device=device)
    count_tensor = torch.as_tensor(counts, dtype=torch.int64, device=device)
    event_loss = path.new_zeros(path.shape[0])
    for horizon_s in HORIZONS:
        horizon_count = round(horizon_s / DT)
        lengths = torch.clamp(count_tensor, max=horizon_count)
        mask = time_index[None, :] <= lengths[:, None]
        event_loss = event_loss + (
            normalized * mask / lengths[:, None] / len(HORIZONS)
        ).sum(dim=1)
    objective = torch.sum(weights * event_loss)
    return (objective, event_loss) if return_events else objective
