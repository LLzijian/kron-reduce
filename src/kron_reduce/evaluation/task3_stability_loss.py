"""Differentiable five-point loss-of-synchronism margin for training."""

import math

import torch


def loss_of_sync_margin(path, equilibrium, mass, omega_s, counts):
    """Return the maximum five-point margin whose window contains a future point.

    Path rows contain 11 observed points followed by the forecast. Counts are
    the numbers of genuine future points per event, excluding right padding.
    Angles are in radians and speeds are in rad/s.
    """
    if path.ndim != 3:
        raise ValueError("expected a batch of visible paths")
    batch, frames, channels = path.shape
    if (not batch or channels != 20 or frames < 12 or equilibrium.shape != (batch, 10)
            or mass.shape != equilibrium.shape or omega_s.shape != (batch, 1, 1)
            or counts.shape != (batch,) or counts.dtype != torch.int64):
        raise ValueError("input shapes must align with 11 observations and visible future counts")
    if torch.any((counts < 1) | (counts > frames - 11)):
        raise ValueError("every event must contain at least one genuine future point")
    if (not all(torch.isfinite(value).all() for value in (path, equilibrium, mass, omega_s))
            or torch.any(mass <= 0) or torch.any(omega_s <= 0)):
        raise ValueError("visible paths and equilibrium must be finite, with positive mass and frequency")

    theta = path[..., :10] + equilibrium[:, None, :]
    speed_pu = path[..., 10:] / omega_s
    mass = mass[:, None, :]
    relative = speed_pu - (speed_pu * mass).sum(dim=-1, keepdim=True) / mass.sum(dim=-1, keepdim=True)
    angle_margin = (theta.amax(dim=-1) - theta.amin(dim=-1)) / math.pi - 1.
    speed_margin = relative.abs().amax(dim=-1) / .02 - 1.
    point_margin = torch.minimum(angle_margin, speed_margin)
    windows = point_margin.unfold(1, 5, 1).amin(dim=-1)[:, 7:]
    valid = torch.arange(frames - 11, device=path.device) < counts[:, None]
    return windows.masked_fill(~valid, -torch.inf).amax(dim=1)
