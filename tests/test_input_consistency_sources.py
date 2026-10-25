"""Synthetic ancestry checks for the A17 metadata source binder."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from kron_reduce.evaluation.input_consistency_plan import SOURCE_MANIFESTS, SOURCE_RUNS
from kron_reduce.evaluation.input_consistency_sources import verify_source_documents

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "configs/proposals/kiki_a17_input_consistency_reads.json"


def _hex(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _fp(descriptor: dict) -> dict:
    return {key: descriptor[key] for key in ("shape", "dtype", "sha256_c_order_bytes")}


def _new_doc(documents: dict, path: str) -> dict:
    return documents[path]["document"]


def _build_synthetic_documents(plan: dict) -> dict:
    """Build authenticated-shaped envelopes from synthetic metadata only."""
    rows = {item["path"]: item for item in plan["metadata_files"]}
    docs = {
        path: {"document": {}, "size_bytes": row["size_bytes"], "sha256": row["sha256"]}
        for path, row in rows.items()
    }

    def put(path: str, doc: dict) -> None:
        _new_doc(docs, path).update(doc)

    a14, a16 = SOURCE_RUNS["A14"], SOURCE_RUNS["A16"]
    roots = {name: f"results/{run}/" for name, run in SOURCE_RUNS.items()}
    manifests = {name: {} for name in SOURCE_RUNS}

    # Source artifact manifests bind every planned source JSON and NPZ identity.
    for row in plan["contexts"]:
        source = row["source"]
        prefix = f"results/{SOURCE_RUNS[source]}/"
        manifests[source][row["path"][len(prefix):]] = row["sha256"]
    for path, row in rows.items():
        for source, run in SOURCE_RUNS.items():
            prefix = f"results/{run}/"
            if path.startswith(prefix) and not path.endswith("/artifact_hashes.json"):
                manifests[source][path[len(prefix):]] = row["sha256"]
    # A09/A10 bulk ancestors are referenced by A15's fingerprint document.
    for seed in (324, 325, 326):
        manifests["A09"][f"seed-{seed}-h005/paths.npz"] = _hex(f"A09:{seed}")
        manifests["A10"][f"seed-{seed}-h0025/paths.npz"] = _hex(f"A10:{seed}")
    manifests["A10"]["real_train_arrays.npz"] = _hex("A10:real-train")

    def doc_path(row: dict) -> str:
        return row["path"].removesuffix(".npz") + ".json"

    # Per-file saved JSON descriptors mirror the frozen plan; values remain absent.
    for row in plan["contexts"]:
        target = _new_doc(docs, doc_path(row))
        context_id = row["context_id"]
        if context_id == "inputs":
            target.update({member["name"]: member for member in row["members"]})
        elif context_id.startswith("static-"):
            parameters = target.setdefault("parameters", {})
            for member in row["members"]:
                parameters[member["name"].removeprefix("parameters__")] = member
        elif context_id.startswith("integral-"):
            target.update({member["name"]: member for member in row["members"]})
            target.update({"status": "complete", "rank": 2, "tol": 1e-12,
                           "natural_norm": 1.5, "svd": {"rank": 2, "threshold": 1e-10}})
        elif context_id.startswith("weighted-"):
            target.setdefault("values", {})["weighted_impulse"] = {
                **row["members"][0], "array_member": row["members"][0]["name"],
                "columns": ["point", "lower", "upper"]
            }
        elif context_id.startswith("impulse-"):
            target.setdefault("u11", [])
            target["u11"] = [member for member in row["members"]]
            target["d"] = [
                {"array_member": f"root__d__{i}", "shape": [10, 3], "dtype": "<f8",
                 "sha256_c_order_bytes": _hex(f"d:{context_id}:{i}"),
                 "columns": ["point", "lower", "upper"]}
                for i in range(10)
            ]
        else:
            target.update({member["name"]: member for member in row["members"]})

    # A16 slot JSON binds the same-slot weighted impulse, Gram descriptors and intervals.
    for row in plan["contexts"]:
        if not row["context_id"].startswith("weighted-"):
            continue
        _, seed_text, index_text = row["context_id"].split("-")
        index, seed = int(index_text), int(seed_text)
        path = f"{roots['A16']}slot-{seed}-{index:03d}.json"
        slot_status = "resolved_nonzero" if index % 2 == 0 else "resolved_zero"
        put(path, {
            "status": slot_status,
            "complete": True,
            "values": {
                "weighted_impulse": {
                    **row["members"][0], "array_member": row["members"][0]["name"],
                    "columns": ["point", "lower", "upper"]
                },
                "gram": [
                    {"array_member": f"gram-{seed}-{index}-0", "shape": [2, 3], "dtype": "<f8",
                     "sha256_c_order_bytes": _hex(f"gram:{seed}:{index}:0"),
                     "columns": ["point", "lower", "upper"]},
                    {"array_member": f"gram-{seed}-{index}-1", "shape": [2, 3], "dtype": "<f8",
                     "sha256_c_order_bytes": _hex(f"gram:{seed}:{index}:1"),
                     "columns": ["point", "lower", "upper"]},
                ],
                "trace": {"point": 2.0, "lower": 1.9, "upper": 2.1},
                "determinant": {"point": 3.0, "lower": 2.9, "upper": 3.1},
            },
        })

    population = plan["population"]
    a14_population = {key: population[key] for key in
                      ("allocated_slots", "cache_rows", "weights", "seed_event_slots")}
    put(roots["A14"] + "allocated_population.json", a14_population)
    put(roots["A15"] + "allocated_population.json", {
        **a14_population,
        "future_point_counts": population["future_point_counts"],
        "future_offsets": [0] * 259,
    })

    # A14's permitted inputs trace to synthetic A10/A06 fingerprints.
    inputs = _new_doc(docs, roots["A14"] + "permitted_inputs.json")
    for member in next(row for row in plan["contexts"] if row["context_id"] == "inputs")["members"]:
        inputs[member["name"]] = member
    a14_source_members = {"A10/real_train_arrays.npz": {
        name: _fp(inputs[name]) | {"array_member": f"A10/real_train_arrays.npz__{name}"}
        for name in inputs
    }}
    for seed in (324, 325, 326):
        static = _new_doc(docs, roots["A14"] + f"static-{seed}.json")
        source_key = f"A06/checkpoint-240-{seed}/model.npz"
        a14_source_members[source_key] = {}
        for name, descriptor in static["parameters"].items():
            a14_source_members[source_key][name] = _fp(descriptor) | {
                "array_member": f"{source_key}__{name}"
            }
    for seed in (324, 325, 326):
        a14_source_members[f"A09/seed-{seed}-h005/paths.npz"] = {"history": {
            "shape": [258, 11, 22], "dtype": "<f8", "sha256_c_order_bytes": _hex(f"a09-history-{seed}")
        }}
        a14_source_members[f"A10/seed-{seed}-h0025/paths.npz"] = {"history": {
            "shape": [258, 11, 22], "dtype": "<f8", "sha256_c_order_bytes": _hex(f"a10-history-{seed}")
        }}
    put(roots["A14"] + "source_members.json", a14_source_members)

    # A15 fingerprints preserve the coarse/fine bulk paths and A10 truth/clock ancestry.
    a15_fingerprints = {}
    for seed in (324, 325, 326):
        a15_fingerprints[f"A09/seed-{seed}-h005/paths.npz"] = {
            "future": {"shape": [72_405, 22], "dtype": "<f8", "sha256_c_order_bytes": _hex(f"coarse-{seed}")},
            "offsets": {"shape": [259], "dtype": "<i8", "sha256_c_order_bytes": _hex("coarse-offsets")},
        }
        a15_fingerprints[f"A10/seed-{seed}-h0025/paths.npz"] = {
            "future": {"shape": [72_405, 22], "dtype": "<f8", "sha256_c_order_bytes": _hex(f"fine-{seed}")},
            "offsets": {"shape": [259], "dtype": "<i8", "sha256_c_order_bytes": _hex("fine-offsets")},
        }
    a15_fingerprints["A10/real_train_arrays.npz"] = {
        "future": {"shape": [72_405, 20], "dtype": "<f8", "sha256_c_order_bytes": _hex("truth")},
        "forecast_time": {"shape": [72_405], "dtype": "<f8", "sha256_c_order_bytes": _hex("time")},
        "offsets": {"shape": [259], "dtype": "<i8", "sha256_c_order_bytes": _hex("truth-offsets")},
        "omega_s": {"shape": [258], "dtype": "<f8", "sha256_c_order_bytes": _hex("omega")},
    }
    put(roots["A15"] + "source_member_fingerprints.json", a15_fingerprints)

    # Receipts bind the already-consumed outcomes without changing them.
    put(roots["A14"] + "summary.json", {
        "execution_completed": True, "V_pass": True, "I_pass": True,
        "promotion_decision": "KEEP_BASELINE", "scientific_verdict": "ACCEPT",
        "comparison_complete": True, "completed_slots": 774, "seed_event_slots": 774,
    })
    put(roots["A15"] + "summary.json", {
        "execution_completed": True, "V_pass": True, "I_pass": False, "D_pass": False,
        "scientific_verdict": "REJECT", "promotion_decision": "KEEP_BASELINE",
        "complete_population": True, "completed_slots": 774, "seed_event_slots": 774,
    })
    put(roots["A16"] + "summary.json", {
        "authorization": "kiki_a16_development_20261025",
        "source_commit": "45a2d989c6351d50e3b835b70b12cd4429f2182b",
        "diagnostic_status": "COMPLETE", "promotion_decision": "KEEP_BASELINE",
    })
    put("docs/reports/kiki_a16_pm_impulse_20261025_evidence.json", {
        "run_id": a16, "result_artifact_manifest_sha256": SOURCE_MANIFESTS["A16"],
        "source_commit": "45a2d989c6351d50e3b835b70b12cd4429f2182b",
        "authorization": "kiki_a16_development_20261025",
    })

    control_plan = {
        "kind": "A16_finite_history_power_impulse_reads_v1",
        "authorization": "kiki_a16_development_20261025",
        "source_run_id": a14,
        "source_manifest_sha256": SOURCE_MANIFESTS["A14"],
        "events": population["cache_rows"],
    }
    put("configs/proposals/kiki_a16_pm_impulse_reads.json", control_plan)
    put(roots["A16"] + "frozen_plan.json", {"read_plan": control_plan})
    original_population = {key: population[key] for key in
                           ("allocated_slots", "cache_rows", "weights", "seed_event_slots")}
    diagnostic_slots = []
    for event in population["seed_event_slots"]:
        diagnostic_slots.append({
            **event,
            "status": "resolved_nonzero" if event["cache_index"] % 2 == 0 else "resolved_zero",
            "complete": True,
        })
    put(roots["A16"] + "allocated_slots.json", {
        "original_population": original_population,
        "diagnostic_slots": diagnostic_slots,
    })

    # Bind every source-run JSON's frozen fingerprint in its synthetic manifest.
    for path, row in rows.items():
        for source, run_id in SOURCE_RUNS.items():
            prefix = f"results/{run_id}/"
            if path.startswith(prefix) and not path.endswith("/artifact_hashes.json"):
                manifests[source][path[len(prefix):]] = row["sha256"]
    for source, manifest in manifests.items():
        put(roots[source] + "artifact_hashes.json", manifest)
    return docs


@pytest.fixture(scope="module")
def fixture_bundle() -> tuple[dict, dict]:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    return plan, _build_synthetic_documents(plan)


def test_synthetic_ancestry_returns_all_slots_and_compact_fingerprints(fixture_bundle) -> None:
    plan, docs = fixture_bundle
    result = verify_source_documents(plan, docs)

    assert len(result["slot_bindings"]) == 774
    assert set(result["slot_bindings"]) == {
        f"{seed}-{index:03d}" for seed in (324, 325, 326) for index in range(258)
    }
    assert result["checks_summary"]["a16_slots_retained_without_status_filter"] is True
    assert result["checks_summary"]["a15_verdict"] == "REJECT"
    assert result["checks_summary"]["promotion_decision"] == "KEEP_BASELINE"
    assert result["slot_bindings"]["324-000"]["integral_parameters"] == {
        "rank": 2, "threshold": 1e-10, "tol": 1e-12, "natural_norm": 1.5
    }
    assert result["slot_bindings"]["324-000"]["weighted_impulse"]["array_member"] == (
        "root__values__weighted_impulse"
    )
    assert result["slot_bindings"]["324-000"]["gram"][0]["columns"] == [
        "point", "lower", "upper"
    ]
    input_ancestry = result["ancestry_fingerprints"]["A14_inputs"]["history"]
    assert set(input_ancestry) == {"a14_saved_member", "ancestor_member"}
    assert len(result["ancestry_fingerprints"]["E"]) == 6
    assert len(result["ancestry_fingerprints"]["clock"]) == 774
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("tamper", ["metadata_sha", "manifest_link", "source_descriptor", "population", "summary"])
def test_source_binding_mutations_rejected(fixture_bundle, tamper: str) -> None:
    plan, original = fixture_bundle
    docs = copy.deepcopy(original)
    if tamper == "metadata_sha":
        docs[next(path for path in docs if path.endswith("/summary.json"))]["sha256"] = "0" * 64
    elif tamper == "manifest_link":
        path = next(path for path in docs if path.endswith("/static-324.json"))
        docs["results/" + SOURCE_RUNS["A14"] + "/artifact_hashes.json"]["document"]["static-324.npz"] = "0" * 64
    elif tamper == "source_descriptor":
        path = f"results/{SOURCE_RUNS['A14']}/source_members.json"
        docs[path]["document"]["A06/checkpoint-240-324/model.npz"]["mass"]["sha256_c_order_bytes"] = "0" * 64
    elif tamper == "population":
        path = f"results/{SOURCE_RUNS['A15']}/allocated_population.json"
        docs[path]["document"]["weights"][0] = 0.0
    else:
        path = f"results/{SOURCE_RUNS['A15']}/summary.json"
        docs[path]["document"]["scientific_verdict"] = "ACCEPT"
    with pytest.raises(ValueError):
        verify_source_documents(plan, docs)


def test_bad_slot_completion_or_interval_fails_without_status_filter(fixture_bundle) -> None:
    plan, original = fixture_bundle
    docs = copy.deepcopy(original)
    allocation_path = f"results/{SOURCE_RUNS['A16']}/allocated_slots.json"
    docs[allocation_path]["document"]["diagnostic_slots"][0]["complete"] = False
    with pytest.raises(ValueError):
        verify_source_documents(plan, docs)

    docs = copy.deepcopy(original)
    slot_path = f"results/{SOURCE_RUNS['A16']}/slot-324-000.json"
    docs[slot_path]["document"]["values"]["determinant"]["lower"] = -1.0
    with pytest.raises(ValueError):
        verify_source_documents(plan, docs)


def test_mappingproxy_and_tuple_envelopes_are_supported(fixture_bundle) -> None:
    from types import MappingProxyType

    plan, docs = fixture_bundle

    def freeze(value):
        if isinstance(value, dict):
            return MappingProxyType({key: freeze(item) for key, item in value.items()})
        if isinstance(value, list):
            return tuple(freeze(item) for item in value)
        return value

    frozen_plan = freeze(plan)
    frozen_docs = MappingProxyType({path: freeze(envelope) for path, envelope in docs.items()})
    result = verify_source_documents(frozen_plan, frozen_docs)
    assert len(result["slot_bindings"]) == 774


def test_corrupt_actual_read_envelope_is_rejected(fixture_bundle) -> None:
    plan, original = fixture_bundle
    docs = copy.deepcopy(original)
    path = f"results/{SOURCE_RUNS['A14']}/seed-324-event-000/integral.json"
    docs[path]["size_bytes"] += 1
    with pytest.raises(ValueError, match="bounded-read metadata fingerprint"):
        verify_source_documents(plan, docs)
