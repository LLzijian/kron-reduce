from __future__ import annotations

from typing import Any

from .contracts import InputContract


def audit_model_sample(sample: dict[str, Any], contract: InputContract | None = None) -> None:
    contract = contract or InputContract()
    model_inputs = sample["model_inputs"]
    visible = set(model_inputs)
    for block in model_inputs.values():
        if isinstance(block, dict):
            visible.update(block)
    leaked = visible & set(contract.forbidden)
    if leaked:
        raise AssertionError(f"forbidden model-visible fields: {sorted(leaked)}")
    if "label" in model_inputs or "metadata" in model_inputs:
        raise AssertionError("label/metadata blocks must remain in training_context")
