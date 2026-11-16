"""TASK3 adapter for the topology-conditioned local-memory model."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from kron_reduce.evaluation.kiki_machine_local_full import inputs
from kron_reduce.models.machine_local_topology import (
    TopologyMemoryModel,
    predict_from_topology_history,
)
from kron_reduce.models.topology_memory_rom import TopologyMemoryROM


def make_model(backbone, support: str, *, train_scales: bool = False, device: str = "cpu",
               learn_harmonics: bool = False, memory_saturation: bool = False) -> TopologyMemoryModel:
    if support not in ("T0M1", "T1M1") or backbone.n_nodes != 10:
        raise ValueError("TASK3 bridge supports ten-machine T0M1 or T1M1 only")
    if (learn_harmonics or memory_saturation) and support != "T0M1":
        raise ValueError("a restoring or memory modification requires fixed-topology T0M1 support")
    mass = backbone.mass
    projection = np.eye(10) - np.outer(mass, np.ones(10)) / mass.sum()
    amplitude = np.linalg.norm(backbone.coupling) / np.linalg.norm(projection)
    if not np.isfinite(amplitude) or amplitude <= 0:
        raise ValueError("backbone port scale must be finite and positive")
    port, tau = np.full(10, amplitude), np.geomspace(.06, 1.6, 10)
    if learn_harmonics or memory_saturation:
        model = TopologyMemoryROM(backbone, port=port, tau=tau, train_scales=train_scales,
                                  learn_harmonics=learn_harmonics, saturation=memory_saturation)
    else:
        model = TopologyMemoryModel(backbone, topology=support == "T1M1", port=port, tau=tau,
                                    train_scales=train_scales)
    return model.to(device)


def load_snapshot(model: TopologyMemoryModel, path: Path) -> TopologyMemoryModel:
    """Reload weights only against exactly the same backbone and model structure."""
    current = model.state_dict()
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != set(current):
            raise ValueError("snapshot state keys differ from model structure")
        values = {}
        for name, original in current.items():
            value = saved[name]
            if value.shape != tuple(original.shape) or value.dtype != np.float64 or not np.isfinite(value).all():
                raise ValueError(f"invalid snapshot tensor: {name}")
            if name not in dict(model.named_parameters()) and not np.array_equal(value, original.cpu().numpy()):
                raise ValueError(f"snapshot backbone or topology differs: {name}")
            values[name] = torch.from_numpy(value.copy()).to(original.device)
    model.load_state_dict(values, strict=True)
    return model.requires_grad_(False).eval()


def predict(model: TopologyMemoryModel, events, scale, step, spend, *, count: int = 800,
            memory_radius: float = 1., prior_weight: float = 0.) -> dict[str, torch.Tensor]:
    result = predict_from_topology_history(model, *inputs(events, scale), count,
                                           max_step=step, spend=spend, memory_radius=memory_radius,
                                           prior_weight=prior_weight)
    n = model.n_nodes
    memory_end = 2*n + model.memory_dimension
    return {
        "forecast": result["forecast"][:, :, :2*n],
        "history": result["history"][:, :, :memory_end],
        "hidden_start": result["hidden_start"][:, :model.memory_dimension],
        "gn_singular": result["singular"][None],
    }
