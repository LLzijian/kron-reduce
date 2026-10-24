"""Fixed-temperature temporal risk aggregation for float64 training losses."""
from __future__ import annotations

import math

import torch


def temporal_logsumexp_risk(
    point_mse: torch.Tensor,
    time_weights: torch.Tensor,
    temperature: float,
    *,
    return_time_risk_weights: bool = False,
):
    """Return one weighted log-sum-exp risk per event.

    ``point_mse`` and ``time_weights`` have shape ``(event, time)``. The
    nonnegative weights are fixed and sum to one within each event. Zero
    weights exclude unavailable times from the softmax and have zero gradient.
    The optional time risk weights are detached diagnostics; they do not
    reweight events.
    """
    if not isinstance(point_mse, torch.Tensor) or not isinstance(time_weights, torch.Tensor):
        raise TypeError("point_mse and time_weights must be torch tensors")
    if point_mse.dtype != torch.float64 or time_weights.dtype != torch.float64:
        raise TypeError("point_mse and time_weights must use torch.float64")
    if point_mse.ndim != 2 or point_mse.numel() == 0:
        raise ValueError("point_mse must be a nonempty (event, time) matrix")
    if time_weights.shape != point_mse.shape:
        raise ValueError("time_weights shape must match point_mse")
    if point_mse.device != time_weights.device:
        raise ValueError("point_mse and time_weights must share a device")
    if type(temperature) not in (int, float):
        raise TypeError("temperature must be a fixed Python int or float")
    tau = float(temperature)
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("temperature must be finite and positive")
    if time_weights.requires_grad:
        raise ValueError("time_weights must be fixed and cannot require gradients")
    if not bool(torch.isfinite(point_mse).all()):
        raise ValueError("point_mse must be finite")
    if bool((point_mse < 0).any()):
        raise ValueError("point_mse must be nonnegative")
    if not bool(torch.isfinite(time_weights).all()):
        raise ValueError("time_weights must be finite")
    if bool((time_weights < 0).any()):
        raise ValueError("time_weights must be nonnegative")
    sums = time_weights.sum(dim=1)
    if not bool(torch.allclose(sums, torch.ones_like(sums), rtol=0.0, atol=1e-12)):
        raise ValueError("time_weights must sum to one for every event")

    active = time_weights > 0
    safe_weights = torch.where(active, time_weights, torch.ones_like(time_weights))
    logits = torch.where(
        active,
        torch.log(safe_weights) + point_mse / tau,
        torch.full_like(point_mse, -torch.inf),
    )
    risk = tau * torch.logsumexp(logits, dim=1)
    if not return_time_risk_weights:
        return risk
    time_risk_weights = torch.softmax(logits, dim=1)
    time_risk_weights = torch.where(active, time_risk_weights, torch.zeros_like(time_risk_weights))
    return risk, time_risk_weights.detach()
