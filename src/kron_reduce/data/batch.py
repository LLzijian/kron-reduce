from __future__ import annotations

from typing import Any

import numpy as np


def collate_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("cannot collate an empty batch")
    static = {
        key: np.stack([sample["model_inputs"]["static"][key] for sample in samples])
        for key in ("delta0", "omega0", "Pm_equilibrium", "Pm_clear", "Pm0")
    }
    lengths = [len(sample["training_context"]["targets"]["t"]) for sample in samples]
    width = max(lengths)
    n_nodes = static["delta0"].shape[-1]
    targets = {
        "delta": np.zeros((len(samples), width, n_nodes), dtype=np.float64),
        "omega": np.zeros((len(samples), width, n_nodes), dtype=np.float64),
        "mask": np.zeros((len(samples), width), dtype=bool),
    }
    for index, sample in enumerate(samples):
        length = lengths[index]
        targets["delta"][index, :length] = sample["training_context"]["targets"]["delta"]
        targets["omega"][index, :length] = sample["training_context"]["targets"]["omega"]
        targets["mask"][index, :length] = True
    return {
        "model_inputs": {"static": static},
        "training_context": {"targets": targets, "lengths": np.asarray(lengths)},
    }
