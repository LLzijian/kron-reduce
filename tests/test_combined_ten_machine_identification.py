from __future__ import annotations

import json

import numpy as np

from kron_reduce.data import CanonicalStore, load_split
from kron_reduce.models import EDGE_MEMORY, CandidateBlockInstance, PrunableEnergyModel
from kron_reduce.workflows.combined_ten_machine_identification import (
    DENSE_SOURCE,
    HIGHER_ORDER_SOURCE,
    _verify_upstream,
    combined_role,
    validate_role_manifest,
)


def test_registered_combined_roles_match_frozen_corpus(root) -> None:
    store = CanonicalStore.open(root / "data" / "ieee39_postclear_canonical_v1")
    split = load_split(
        store.dataset_dir,
        view="combined_dynamics",
        split_name="combined_line_5fold_v1",
        fold="F1",
    )
    rows = []
    for canonical_role, event_ids in (("train", split.train_ids), ("val", split.val_ids)):
        for event_id in event_ids:
            metadata = store.metadata(event_id)
            rows.append(
                {
                    "event_id": event_id,
                    "source_dataset": metadata.source_dataset,
                    "fault_line": metadata.fault_line,
                    "role": combined_role(
                        metadata.source_dataset,
                        metadata.fault_line,
                        metadata.valid_length,
                        canonical_role,
                    ),
                }
            )
    audit = validate_role_manifest(rows)
    assert audit["exact_expected_counts"] is True
    assert audit["counts"]["fit"] == 110
    assert audit["counts"]["selection"] == 25
    assert audit["counts"]["internal_audit"] == 43
    assert audit["short_safety_events"] == 40
    assert audit["fault_line_role_leakage"] is False


def test_short_trajectories_can_only_enter_postfreeze_safety() -> None:
    assert combined_role(DENSE_SOURCE, "Line_2", 30, "train") == "postfreeze_safety_train"
    assert (
        combined_role(DENSE_SOURCE, "Line_36", 30, "val")
        == "postfreeze_safety_validation"
    )
    assert combined_role(HIGHER_ORDER_SOURCE, "Line_10", 1001, "train") == "fit"


def test_low_rank_instance_round_trip_preserves_reciprocal_state_dimension(tmp_path) -> None:
    instance = CandidateBlockInstance(
        "edge_memory:0:tau=0.2",
        EDGE_MEMORY,
        time_constant=0.2,
        factor=((0.4,), (-0.1,), (0.2,)),
        port_rank=1,
        source_support=(("dense", 0.9), ("higher", 0.85)),
        energy_constraint="reciprocal_real_pole_storage",
    )
    model = PrunableEnergyModel(block_instances=(instance,))
    model.node_damping = np.asarray([0.2, 0.25, 0.3])
    model.coupling = np.asarray([1.0, 0.8, 0.9])
    model.potential_coefficients = np.zeros((3, 3))
    model.edge_damping_coefficients = np.zeros((3, 3))
    model.node_high_order_gain = np.zeros(3)
    model.memory_factors = [np.asarray(instance.factor)]
    model.memory_gains = np.diag(model.memory_factors[0] @ model.memory_factors[0].T)[None, :]
    model.resonant_factors = []
    model.resonant_gains = np.zeros((0, 3))
    path = tmp_path / "low_rank.npz"
    model.save(path)
    loaded = PrunableEnergyModel.load(path)
    assert loaded.active_instances == (instance,)
    assert loaded.reciprocal_connections()[EDGE_MEMORY][0].shape == (3, 1)
    assert loaded._hidden_shapes(3, 3)["memory"] == (1,)


def test_frozen_cluster_refusal_hash_is_verified(root) -> None:
    config_path = (
        root
        / "configs"
        / "experiments"
        / "paper_combined_ten_machine_identification_f1_seed324.toml"
    )
    import tomllib

    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    evidence = _verify_upstream(root, config)
    assert evidence["verdict"] == "VALIDATED_REFUSAL"
    assert evidence["cluster_decision"]["n_clusters"] == 10
    assert len(evidence["artifact_hash"]) == 64


def test_combined_workflow_is_registered_as_one_shared_model(root) -> None:
    manifest = json.loads((root / "configs" / "project_manifest.json").read_text(encoding="utf-8"))
    experiment = manifest["experiments"]["paper_combined_ten_machine_identification_f1_seed324"]
    assert experiment["workflow"] == "combined_ten_machine_identification_v1"
    workflow = json.loads(
        (root / "configs" / "workflows" / "combined_ten_machine_identification_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert workflow["scientific_target"].startswith("one_frozen_ten_machine")
    assert "source_routing" in workflow["locked_claims"]


def test_pruning_protects_only_data_supported_modes() -> None:
    def mode(frequency: float) -> dict[str, object]:
        return {
            "frequency_hz": frequency,
            "damping_ratio": 0.1,
            "shape_real": [1.0, -1.0],
            "shape_imag": [0.0, 0.0],
        }

    reference = [mode(0.5), mode(2.0)]
    candidate = [mode(0.5)]
    mass = np.ones(2)
    assert PrunableEnergyModel._modal_noninferior(
        reference,
        candidate,
        mass,
        required_modes=[mode(0.5)],
    )
    assert not PrunableEnergyModel._modal_noninferior(
        reference,
        candidate,
        mass,
        required_modes=[mode(2.0)],
    )
