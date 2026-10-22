"""Prospective research policy; historical evidence retains its original contract."""
from __future__ import annotations

import json
from pathlib import Path

from .evidence import sha256_file


def load_policy(root: Path) -> dict:
    path = root / "configs/research_policy.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def validate_run_policy(
    root: Path, name: str, experiment: dict, *, historical_reproduction: bool = False,
) -> None:
    policy = load_policy(root)
    if not policy:
        return
    dataset = experiment.get("dataset")
    if experiment.get("execution_mode") == "historical_reproduction_only":
        if not historical_reproduction:
            raise RuntimeError("HISTORICAL_ONLY: use --historical-reproduction for registered replay")
        registry = json.loads((root / policy["historical_registry"]).read_text(encoding="utf-8"))
        frozen = registry.get(name)
        if frozen is None or dataset != frozen["dataset"]:
            raise RuntimeError("HISTORICAL_IDENTITY: unregistered historical replay")
        for relative, digest in frozen["bindings"].items():
            if sha256_file(root / relative) != digest:
                raise RuntimeError(f"HISTORICAL_CONTRACT_CHANGED: {relative}")
        return
    if historical_reproduction:
        raise RuntimeError("HISTORICAL_IDENTITY: replay is restricted to the frozen registry")
    if dataset in policy["synthetic_datasets"]:
        return
    if dataset != policy["primary_real_dataset"]:
        raise RuntimeError("DATASET_POLICY: new real-data experiments must use registered kiki data")
    if experiment.get("spatial_node_count") != 10 or experiment.get("spatial_clustering") is not False:
        raise RuntimeError("SPATIAL_POLICY: retain ten nodes; spatial clustering is paused")
    contracts = json.loads((root / policy["input_contracts"]).read_text(encoding="utf-8"))
    if experiment.get("input_contract_id") not in contracts:
        raise RuntimeError("INPUT_CONTRACT: declare forecast origin and observation budget")
    if experiment.get("reads_test_performance", False):
        raise RuntimeError(
            "TEST_EXPOSED: existing kiki test is not fresh; register independent kiki test evidence first"
        )


def audit_research_policy(root: Path, manifest: dict) -> list[dict]:
    policy = load_policy(root)
    if not policy:
        return []
    messages = []
    if manifest.get("default_dataset") != policy["primary_real_dataset"]:
        messages.append("default dataset differs from the active kiki policy")
    entry = manifest.get("datasets", {}).get(policy["primary_real_dataset"])
    if entry is None:
        messages.append("primary kiki data registry is missing")
    else:
        from .data_check import _verify_checksum_manifest

        try:
            checks = _verify_checksum_manifest(
                root / entry["path"], entry["checksum_manifest"],
                dataset_name=policy["primary_real_dataset"],
            )
            if not checks or not all(row["pass"] for row in checks):
                messages.append("frozen kiki registry metadata failed checksum verification")
        except (OSError, ValueError, KeyError) as error:
            messages.append(f"kiki metadata registry: {error}")
    for name, experiment in manifest["experiments"].items():
        try:
            validate_run_policy(
                root, name, experiment,
                historical_reproduction=experiment.get("execution_mode") == "historical_reproduction_only",
            )
        except (RuntimeError, OSError, KeyError, ValueError) as error:
            messages.append(f"{name}: {error}")
    return [{"severity": "ERROR", "check": "research_policy", "message": text}
            for text in messages]
