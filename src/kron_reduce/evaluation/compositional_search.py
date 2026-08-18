from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, replace
from itertools import combinations
from typing import Any


@dataclass(frozen=True)
class SearchState:
    """A hashable AFSD v4 structure-search state."""

    spatial_partition: tuple[tuple[int, ...], ...]
    internal_modes: int = 0
    equivalent_graph: tuple[tuple[int, int], ...] = ()
    active_modules: tuple[str, ...] = ()
    lineage: tuple[str, ...] = ()

    @property
    def key(self) -> tuple[Any, ...]:
        return (
            self.spatial_partition,
            self.internal_modes,
            self.equivalent_graph,
            self.active_modules,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def add_module(state: SearchState, name: str) -> SearchState:
    modules = tuple(sorted({*state.active_modules, str(name)}))
    return replace(state, active_modules=modules, lineage=(*state.lineage, f"add_module:{name}"))


def remove_module(state: SearchState, name: str) -> SearchState:
    modules = tuple(item for item in state.active_modules if item != name)
    return replace(state, active_modules=modules, lineage=(*state.lineage, f"remove_module:{name}"))


def add_edge(state: SearchState, edge: tuple[int, int]) -> SearchState:
    canonical = tuple(sorted((int(edge[0]), int(edge[1]))))
    edges = tuple(sorted({*state.equivalent_graph, canonical}))
    return replace(state, equivalent_graph=edges, lineage=(*state.lineage, f"add_edge:{canonical}"))


def _sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    metrics = row["metrics"]
    hard_pass = metrics.get("hard_gate") == "PASS"
    return (
        0 if hard_pass else 1,
        float(metrics.get("heldout_error", float("inf"))),
        int(metrics.get("description_length", 10**9)),
        row["state"].key,
    )


def deterministic_beam_search(
    initial: SearchState,
    action_fn: Callable[[SearchState], Iterable[SearchState]],
    evaluate_fn: Callable[[SearchState, int], dict[str, Any]],
    *,
    beam_width: int,
    max_depth: int,
) -> dict[str, Any]:
    """Tiered deterministic Beam search with complete lineage and fit counts."""

    if beam_width <= 0 or max_depth < 0:
        raise ValueError("beam_width must be positive and max_depth nonnegative")
    cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    tree: list[dict[str, Any]] = []
    fit_counts = {"tier_1": 0, "tier_2": 0, "tier_3": 0}

    def evaluate(state: SearchState, tier: int) -> dict[str, Any]:
        cache_key = (*state.key, int(tier))
        if cache_key not in cache:
            cache[cache_key] = dict(evaluate_fn(state, tier))
            fit_counts[f"tier_{tier}"] += 1
        return cache[cache_key]

    initial_row = {"state": initial, "metrics": evaluate(initial, 1), "depth": 0}
    beam = [initial_row]
    all_rows = [initial_row]
    for depth in range(1, int(max_depth) + 1):
        candidates: dict[tuple[Any, ...], dict[str, Any]] = {}
        for parent in beam:
            for child in action_fn(parent["state"]):
                if child.key in candidates:
                    continue
                metrics = evaluate(child, 1)
                row = {"state": child, "metrics": metrics, "depth": depth}
                candidates[child.key] = row
                tree.append(
                    {
                        "parent": parent["state"].to_dict(),
                        "child": child.to_dict(),
                        "depth": depth,
                        "tier_1": metrics,
                    }
                )
        eligible = [row for row in candidates.values() if row["metrics"].get("hard_gate") == "PASS"]
        if not eligible:
            break
        beam = sorted(eligible, key=_sort_key)[: int(beam_width)]
        for row in beam:
            row["metrics"] = evaluate(row["state"], 2)
        beam = sorted(
            [row for row in beam if row["metrics"].get("hard_gate") == "PASS"],
            key=_sort_key,
        )[: int(beam_width)]
        all_rows.extend(beam)
        if not beam:
            break

    finalists = sorted(all_rows, key=_sort_key)[: int(beam_width)]
    for row in finalists:
        row["metrics"] = evaluate(row["state"], 3)
    eligible_final = [row for row in finalists if row["metrics"].get("hard_gate") == "PASS"]
    if not eligible_final:
        return {
            "status": "SEARCH_METHOD_INADEQUATE",
            "selected": None,
            "tree": tree,
            "fit_counts": fit_counts,
        }
    selected = min(eligible_final, key=_sort_key)
    return {
        "status": "completed",
        "selected": selected["state"].to_dict(),
        "selected_metrics": selected["metrics"],
        "tree": tree,
        "fit_counts": fit_counts,
        "evaluated_unique_structures": len({row["state"].key for row in all_rows}),
    }


def exhaustive_oracle(
    states: Sequence[SearchState],
    evaluate_fn: Callable[[SearchState, int], dict[str, Any]],
) -> dict[str, Any]:
    rows = [
        {"state": state, "metrics": dict(evaluate_fn(state, 3)), "depth": len(state.lineage)}
        for state in states
    ]
    eligible = [row for row in rows if row["metrics"].get("hard_gate") == "PASS"]
    if not eligible:
        return {"status": "MODEL_CLASS_INADEQUATE", "selected": None, "full_fits": len(rows)}
    selected = min(eligible, key=_sort_key)
    return {
        "status": "completed",
        "selected": selected["state"].to_dict(),
        "selected_metrics": selected["metrics"],
        "full_fits": len(rows),
    }


def module_subset_states(initial: SearchState, modules: Sequence[str]) -> list[SearchState]:
    result = []
    for count in range(len(modules) + 1):
        for subset in combinations(sorted(modules), count):
            result.append(replace(initial, active_modules=tuple(subset)))
    return result


def oracle_agreement(beam: dict[str, Any], oracle: dict[str, Any]) -> dict[str, Any]:
    beam_selected = beam.get("selected")
    oracle_selected = oracle.get("selected")
    structural_fields = (
        "spatial_partition",
        "internal_modes",
        "equivalent_graph",
        "active_modules",
    )
    agrees = (
        beam_selected is not None
        and oracle_selected is not None
        and all(
            beam_selected.get(field) == oracle_selected.get(field) for field in structural_fields
        )
    )
    beam_full = int(beam.get("fit_counts", {}).get("tier_3", 0))
    oracle_full = int(oracle.get("full_fits", 0))
    return {
        "status": "PASS" if agrees else "FAIL",
        "same_structure": agrees,
        "beam_full_closed_loop_fits": beam_full,
        "oracle_full_closed_loop_fits": oracle_full,
        "full_fit_reduction": oracle_full - beam_full,
    }


def paired_module_mcts_trigger(
    initial: SearchState,
    modules: Sequence[str],
    evaluate_fn: Callable[[SearchState, int], dict[str, Any]],
    *,
    beam_widths: tuple[int, ...] = (3, 5, 10),
) -> dict[str, Any]:
    """Detect pair synergy that repeated finite Beam search cannot recover."""

    oracle = exhaustive_oracle(module_subset_states(initial, modules), evaluate_fn)

    def actions(state: SearchState) -> Iterable[SearchState]:
        for module in sorted(set(modules) - set(state.active_modules)):
            yield add_module(state, module)

    rows = []
    for width in beam_widths:
        beam = deterministic_beam_search(
            initial,
            actions,
            evaluate_fn,
            beam_width=int(width),
            max_depth=len(modules),
        )
        rows.append({"beam_width": int(width), "agreement": oracle_agreement(beam, oracle)})
    missed = all(row["agreement"]["status"] != "PASS" for row in rows)
    return {
        "status": "MCTS_TRIGGERED" if missed else "BEAM_ADEQUATE",
        "reason": "paired-module synergy missed by every preregistered Beam width"
        if missed
        else "a preregistered Beam width matches exhaustive oracle",
        "beam_width_audit": rows,
        "oracle": oracle,
    }
