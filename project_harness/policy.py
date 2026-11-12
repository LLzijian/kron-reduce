"""Prospective research policy; historical evidence retains its original contract."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .evidence import _safe_relative, sha256_file


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
    legacy = policy.get("legacy_real_experiments", {})
    if name in legacy:
        encoded = json.dumps(experiment, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        if hashlib.sha256(encoded).hexdigest() != legacy[name]:
            raise RuntimeError("LEGACY_EXPERIMENT_CHANGED: register a new official-split experiment")
    elif policy.get("default_real_data_policy") == "kiki_official_train_test_v1":
        population = experiment.get("test_population")
        if population != "task3_campaign2_final_test":
            raise RuntimeError("OFFICIAL_SPLIT_POLICY: reuse TASK3's frozen 76-event test roster")
        expected = {"data_policy": "kiki_official_train_test_v1", "training_split": "train",
                    "internal_resplit": False,
                    "execution_mode": "bounded_kiki_official_cli_only",
                    "split_reference_sha256": policy["official_split_reference_sha256"]}
        if any(type(experiment.get(k)) is not type(v) or experiment[k] != v for k, v in expected.items()):
            raise RuntimeError("OFFICIAL_SPLIT_POLICY: new experiments must use the frozen official train/test contract")
        if experiment.get("input_contract_id") != "measured_history_100ms_zoh_v1":
            raise RuntimeError("TASK3_INPUT_CONTRACT: benchmark uses eleven real 10 ms observations")
    if experiment.get("reads_test_performance", False):
        auth = policy.get("official_test_evaluations", {}).get(experiment.get("test_authorization"), {})
        if (name in legacy or auth.get("status") != "active" or auth.get("experiment") != name
                or type(auth.get("fit_attempts")) is not int or auth["fit_attempts"] != 0
                or auth.get("test_population") != experiment.get("test_population")
                or auth.get("split_reference_sha256") != policy.get("official_split_reference_sha256")
                or type(auth.get("max_executions")) is not int or auth["max_executions"] != 1
                or auth.get("retry_allowed") is not False):
            raise RuntimeError("TEST_EXPOSED: register the frozen comparative test scope before access")
        binding = auth.get("frozen_model_manifest", {})
        if not isinstance(binding, dict) or not isinstance(binding.get("path"), str):
            raise RuntimeError("TEST_MODEL_FREEZE: frozen model manifest is missing")
        path = _safe_relative(root, binding.get("path"))
        if not path.is_file() or sha256_file(path) != binding.get("sha256"):
            raise RuntimeError("TEST_MODEL_FREEZE: frozen model manifest is missing or changed")


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
