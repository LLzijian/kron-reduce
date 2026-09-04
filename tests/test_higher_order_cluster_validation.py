from __future__ import annotations

from copy import deepcopy

from kron_reduce.data import load_split
from kron_reduce.selection import audit_cluster_decision, data_level_group_split, decide_clustering
from kron_reduce.workflows.higher_order_cluster_validation import (
    HIGHER_ORDER_SOURCE,
    _adapt_samples,
    select_higher_order_subsets,
    validation_verdict,
)


def _higher_order_rows(store, dataset_dir) -> list[dict[str, str]]:
    split = load_split(
        dataset_dir,
        view="combined_dynamics",
        split_name="combined_line_5fold_v1",
        fold="F1",
    )
    return [
        {"event_id": event_id, "fault_line": store.metadata(event_id).fault_line}
        for event_id in split.train_ids
        if store.metadata(event_id).source_dataset == HIGHER_ORDER_SOURCE
    ]


def _decision(use_clustering: bool, selected_rank: int = 10):
    from kron_reduce.selection import ClusterDecision

    n_clusters = selected_rank if use_clustering else 10
    partition = tuple((index,) for index in range(10))
    return ClusterDecision(
        use_clustering=use_clustering,
        n_original=10,
        n_clusters=n_clusters,
        partition=partition,
        score=0.5,
        components={},
        bootstrap_support=1.0,
        selected_rank=selected_rank,
        candidate_scores=(),
        spectrum=(),
        roles={"fit": (), "selection": (), "audit": ()},
    )


def test_higher_order_hash_subsets_are_deterministic_and_line_complete(store, dataset_dir) -> None:
    rows = _higher_order_rows(store, dataset_dir)
    first, first_manifest = select_higher_order_subsets(rows, seed=324, main_per_line=2)
    second, second_manifest = select_higher_order_subsets(rows, seed=324, main_per_line=2)
    assert first == second
    assert first_manifest == second_manifest
    assert {name: len(values) for name, values in first.items()} == {
        "main": 31,
        "small": 16,
        "full": 58,
    }
    lines = {row["fault_line"] for row in rows}
    main_lines = {
        row["fault_line"] for row in first_manifest if bool(row["in_main_31"])
    }
    assert len(lines) == 16
    assert main_lines == lines
    assert all(not row["selection_uses_label"] for row in first_manifest)
    assert all(not row["selection_uses_response"] for row in first_manifest)


def test_all_role_seeds_keep_fault_lines_disjoint(store, dataset_dir) -> None:
    rows = _higher_order_rows(store, dataset_dir)
    subsets, _ = select_higher_order_subsets(rows, seed=324, main_per_line=2)
    line_by_id = {row["event_id"]: row["fault_line"] for row in rows}
    samples = [
        {
            "event_id": event_id,
            "training_context": {"metadata": {"fault_line": line_by_id[event_id]}},
        }
        for event_id in subsets["main"]
    ]
    for seed in (324, 1324, 2324, 3324, 4324):
        _, roles = data_level_group_split(samples, seed=seed)
        role_sets = {name: set(groups) for name, groups in roles.items()}
        assert role_sets["fit"].isdisjoint(role_sets["selection"])
        assert role_sets["fit"].isdisjoint(role_sets["audit"])
        assert role_sets["selection"].isdisjoint(role_sets["audit"])
        assert set().union(*role_sets.values()) == set(line_by_id.values())


def test_validation_verdict_covers_accept_refuse_failure_and_inconclusive_states() -> None:
    refusal = _decision(False)
    rejection_rows = [{"n_clusters": rank, "passes_all": False} for rank in range(2, 10)]
    stable_refusal = [refusal] * 5
    assert validation_verdict(
        refusal, rejection_rows, stable_refusal, (refusal, refusal)
    )["verdict"] == "VALIDATED_REFUSAL"
    false_rejection_rows = deepcopy(rejection_rows)
    false_rejection_rows[-1]["passes_all"] = True
    assert validation_verdict(
        refusal, false_rejection_rows, stable_refusal, (refusal, refusal)
    )["verdict"] == "FAILED_FALSE_REJECTION"

    acceptance = _decision(True, 7)
    acceptance_rows = [
        {"n_clusters": rank, "passes_all": rank == 7} for rank in range(2, 10)
    ]
    assert validation_verdict(
        acceptance, acceptance_rows, [acceptance] * 5, (acceptance, acceptance)
    )["verdict"] == "VALIDATED_ACCEPTANCE"
    acceptance_rows[5]["passes_all"] = False
    assert validation_verdict(
        acceptance, acceptance_rows, [acceptance] * 5, (acceptance, acceptance)
    )["verdict"] == "FAILED_FALSE_ACCEPTANCE"
    assert validation_verdict(
        refusal,
        rejection_rows,
        [refusal, refusal, _decision(True, 7), _decision(True, 7), _decision(True, 7)],
        (refusal, refusal),
    )["verdict"] == "INCONCLUSIVE"


def test_frozen_audit_does_not_mutate_decision_and_reconstructs_ten_nodes(
    store, dataset_dir
) -> None:
    rows = _higher_order_rows(store, dataset_dir)
    subsets, _ = select_higher_order_subsets(rows, seed=324, main_per_line=1)
    samples_by_id = _adapt_samples(store, subsets["small"])
    samples = [samples_by_id[event_id] for event_id in subsets["small"]]
    decision = decide_clustering(
        samples,
        seed=324,
        bootstrap_draws=0,
        max_steps=101,
        delays=4,
    )
    frozen = deepcopy(decision.to_dict())
    audit = audit_cluster_decision(
        samples,
        decision,
        seed=324,
        max_steps=101,
        rollout_horizons_s=(0.05,),
        cpig_bootstrap_draws=4,
    )
    assert decision.to_dict() == frozen
    assert audit.decision_unchanged is True
    assert len(audit.candidate_audits) == 8
    for row in audit.candidate_audits:
        assert sorted(node for group in row["partition"] for node in group) == list(range(10))
        assert row["rollout_failure_count"] == 0
