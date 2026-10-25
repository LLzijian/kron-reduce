"""Authenticate A17 source-document ancestry using already-read JSON envelopes.

This module performs metadata identity checks only. It has no filesystem or
array-reading code; callers must pass bounded, hash-verified JSON envelopes.
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence

from kron_reduce.evaluation.input_consistency_plan import (
    CONTROL_PLAN_PATH,
    SOURCE_MANIFESTS,
    SOURCE_RUNS,
    validate_production_plan,
)

HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_DESCRIPTOR_FIELDS = ("array_member", "shape", "dtype", "sha256_c_order_bytes")
_FINGERPRINT_FIELDS = ("shape", "dtype", "sha256_c_order_bytes")
_POPULATION_FIELDS = ("allocated_slots", "cache_rows", "weights", "seed_event_slots")


def _fail(message: str) -> None:
    raise ValueError(f"invalid A17 source documents: {message}")


def _dict(value: object, where: str) -> dict:
    if not isinstance(value, Mapping):
        _fail(f"{where} must be a JSON object")
    return value


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _doc(documents: Mapping, path: str) -> dict:
    try:
        envelope = documents[path]
    except KeyError as error:
        raise ValueError(f"invalid A17 source documents: missing document {path}") from error
    envelope = _dict(envelope, f"envelope {path}")
    if set(envelope) != {"document", "size_bytes", "sha256"}:
        _fail(f"document envelope fields differ for {path}")
    document = _dict(envelope["document"], f"document {path}")
    if type(envelope["size_bytes"]) is not int or envelope["size_bytes"] <= 0:
        _fail(f"document byte count is invalid for {path}")
    if not isinstance(envelope["sha256"], str) or HEX64.fullmatch(envelope["sha256"]) is None:
        _fail(f"document SHA is malformed for {path}")
    return document


def _fingerprint(descriptor: object, where: str) -> dict:
    descriptor = _dict(descriptor, where)
    for key in ("shape", "dtype", "sha256_c_order_bytes"):
        if key not in descriptor:
            _fail(f"{where} misses {key}")
    shape = descriptor["shape"]
    if (not isinstance(shape, Sequence) or isinstance(shape, (str, bytes))
            or any(type(dim) is not int or dim < 0 for dim in shape)
            or descriptor["dtype"] not in {"<f8", "<i8"}
            or not isinstance(descriptor["sha256_c_order_bytes"], str)
            or HEX64.fullmatch(descriptor["sha256_c_order_bytes"]) is None):
        _fail(f"{where} descriptor is malformed")
    if "array_member" in descriptor and not isinstance(descriptor["array_member"], str):
        _fail(f"{where} member name is malformed")
    return {key: _plain(descriptor[key]) for key in _FINGERPRINT_FIELDS}


def _descriptor_record(descriptor: object, where: str) -> dict:
    _fingerprint(descriptor, where)
    descriptor = _dict(descriptor, where)
    keys = [key for key in (*_DESCRIPTOR_FIELDS, "columns") if key in descriptor]
    return {key: _plain(descriptor[key]) for key in keys}


def _same_descriptor(actual: object, expected: object, where: str) -> dict:
    actual = _dict(actual, where)
    expected = _dict(expected, f"expected {where}")
    left = _fingerprint(actual, where)
    right = _fingerprint(expected, f"expected {where}")
    if left != right:
        _fail(f"descriptor fingerprint differs for {where}")
    if ("array_member" in actual and "array_member" in expected
            and actual["array_member"] != expected["array_member"]):
        _fail(f"array member differs for {where}")
    return _descriptor_record(actual, where)


def _same_fingerprint(actual: object, expected: object, where: str) -> dict:
    left = _fingerprint(actual, where)
    right = _fingerprint(expected, f"expected {where}")
    if left != right:
        _fail(f"source ancestry fingerprint differs for {where}")
    return _descriptor_record(actual, where)


def _manifest_for(plan: dict, documents: Mapping, source: str) -> dict:
    path = f"results/{SOURCE_RUNS[source]}/artifact_hashes.json"
    row = next((item for item in plan["metadata_files"] if item["path"] == path), None)
    if row is None:
        _fail(f"missing planned source manifest {source}")
    envelope = _dict(documents.get(path), f"envelope {path}")
    if envelope.get("size_bytes") != row["size_bytes"] or envelope.get("sha256") != row["sha256"]:
        _fail(f"source manifest envelope differs for {source}")
    manifest = _doc(documents, path)
    if not manifest or any(not isinstance(key, str) or not isinstance(value, str)
                           or HEX64.fullmatch(value) is None for key, value in manifest.items()):
        _fail(f"source manifest malformed for {source}")
    if row["sha256"] != SOURCE_MANIFESTS[source]:
        _fail(f"source manifest binding differs for {source}")
    return manifest


def _manifest_file(manifest: dict, run_id: str, path: str, digest: str) -> None:
    prefix = f"results/{run_id}/"
    if not path.startswith(prefix):
        _fail(f"path is outside source run: {path}")
    relative = path[len(prefix):]
    if manifest.get(relative) != digest:
        _fail(f"artifact manifest does not bind {relative}")


def _context(plan: dict, context_id: str) -> dict:
    for row in plan["contexts"]:
        if row["context_id"] == context_id:
            return row
    _fail(f"missing frozen context {context_id}")


def _metadata_document_path(npz_path: str) -> str:
    return npz_path.removesuffix(".npz") + ".json"


def _check_context_document(plan: dict, documents: Mapping, row: dict,
                            manifest: dict) -> dict:
    _manifest_file(manifest, SOURCE_RUNS[row["source"]], row["path"], row["sha256"])
    document_path = _metadata_document_path(row["path"])
    document = _doc(documents, document_path)
    for member in row["members"]:
        name = member["name"]
        if row["context_id"].startswith("weighted-"):
            actual = document.get("values", {}).get("weighted_impulse")
        elif row["context_id"].startswith("impulse-"):
            suffix = name.removeprefix("root__u11__")
            actual = document.get("u11", [])[int(suffix)]
        elif row["context_id"].startswith("static-"):
            actual = document.get("parameters", {}).get(name.removeprefix("parameters__"))
        else:
            actual = document.get(name)
        _same_descriptor(actual, member, f"{row['context_id']}:{name}")
    return document


def _positive_interval(value: object, where: str) -> dict:
    value = _dict(value, where)
    if set(value) != {"point", "lower", "upper"}:
        _fail(f"interval fields differ for {where}")
    lower, point, upper = value["lower"], value["point"], value["upper"]
    if (any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
            for x in (lower, point, upper)) or lower <= 0 or not lower <= point <= upper):
        _fail(f"interval must be finite, positive, and ordered for {where}")
    return {"point": point, "lower": lower, "upper": upper}


def _population_projection(population: dict) -> dict:
    return {key: _plain(population[key]) for key in _POPULATION_FIELDS}


def _complete_descriptor_map(source_members: dict, source_key: str,
                             local: dict, where: str) -> dict:
    source = _dict(source_members.get(source_key), f"source member {source_key}")
    output = {}
    for name, descriptor in local.items():
        if name not in source:
            _fail(f"source ancestry misses {source_key}:{name}")
        _same_fingerprint(descriptor, source[name], f"{where}:{name}")
        output[name] = {
            "a14_saved_member": _descriptor_record(descriptor, f"{where}:{name}"),
            "ancestor_member": _descriptor_record(source[name], f"{where}:{name}"),
        }
    return output


def verify_source_documents(plan: dict, documents: Mapping) -> dict:
    """Verify authenticated source ancestry and return compact slot bindings.

    `documents` must map every frozen metadata path to an envelope containing
    the parsed object and the actual bounded-read byte count/SHA recorded by
    the reader. No filesystem access occurs here.
    """
    plan = _plain(plan)
    validate_production_plan(plan)
    if not isinstance(documents, Mapping):
        _fail("documents must be a path-to-envelope mapping")
    metadata_rows = {row["path"]: row for row in plan["metadata_files"]}
    contexts_by_id = {row["context_id"]: row for row in plan["contexts"]}
    if set(documents) != set(metadata_rows):
        _fail("document envelope set must exactly match the frozen metadata whitelist")
    for path, row in metadata_rows.items():
        envelope = _dict(documents[path], f"envelope {path}")
        if envelope.get("size_bytes") != row["size_bytes"] or envelope.get("sha256") != row["sha256"]:
            _fail(f"bounded-read metadata fingerprint differs for {path}")

    manifests = {name: _manifest_for(plan, documents, name) for name in SOURCE_RUNS}
    # Bind every source-run JSON document through that run's sealed manifest.
    for path, row in metadata_rows.items():
        for source, run_id in SOURCE_RUNS.items():
            prefix = f"results/{run_id}/"
            if path.startswith(prefix) and not path.endswith("/artifact_hashes.json"):
                _manifest_file(manifests[source], run_id, path, row["sha256"])

    context_documents = {
        row["context_id"]: _check_context_document(plan, documents, row, manifests[row["source"]])
        for row in plan["contexts"]
    }

    a14 = SOURCE_RUNS["A14"]
    a15 = SOURCE_RUNS["A15"]
    a16 = SOURCE_RUNS["A16"]
    a14_root, a15_root, a16_root = (f"results/{run}/" for run in (a14, a15, a16))
    a14_population = _doc(documents, a14_root + "allocated_population.json")
    a15_population = _doc(documents, a15_root + "allocated_population.json")
    if (_population_projection(a14_population) != _population_projection(plan["population"])
            or _population_projection(a15_population) != _population_projection(plan["population"])):
        _fail("A14/A15 frozen population differs from A17")
    if (_plain(a15_population.get("future_point_counts")) != plan["population"].get("future_point_counts")
        or len(_plain(a15_population.get("future_offsets", []))) != 259):
        _fail("A15 future counts/offset metadata differs")

    a14_summary = _doc(documents, a14_root + "summary.json")
    a15_summary = _doc(documents, a15_root + "summary.json")
    a16_summary = _doc(documents, a16_root + "summary.json")
    if not (a14_summary.get("execution_completed") is True
            and a14_summary.get("V_pass") is True and a14_summary.get("I_pass") is True
            and a14_summary.get("comparison_complete") is True
            and a14_summary.get("completed_slots") == 774
            and a14_summary.get("seed_event_slots") == 774
            and a14_summary.get("promotion_decision") == "KEEP_BASELINE"):
        _fail("A14 completed V/I KEEP_BASELINE receipt differs")
    if not (a15_summary.get("execution_completed") is True
            and a15_summary.get("V_pass") is True and a15_summary.get("I_pass") is False
            and a15_summary.get("D_pass") is False and a15_summary.get("scientific_verdict") == "REJECT"
            and a15_summary.get("complete_population") is True
            and a15_summary.get("completed_slots") == 774
            and a15_summary.get("seed_event_slots") == 774
            and a15_summary.get("promotion_decision") == "KEEP_BASELINE"):
        _fail("A15 V-pass/I-D-fail REJECT KEEP_BASELINE receipt differs")
    if (a16_summary.get("authorization") != "kiki_a16_development_20261025"
            or a16_summary.get("source_commit") != "45a2d989c6351d50e3b835b70b12cd4429f2182b"
            or a16_summary.get("diagnostic_status") != "COMPLETE"
            or a16_summary.get("promotion_decision") != "KEEP_BASELINE"):
        _fail("A16 consumed run receipt identity differs")

    a16_report_path = "docs/reports/kiki_a16_pm_impulse_20261025_evidence.json"
    a16_report = _doc(documents, a16_report_path)
    if (a16_report.get("run_id") != a16
            or a16_report.get("result_artifact_manifest_sha256") != SOURCE_MANIFESTS["A16"]
            or a16_report.get("source_commit") != a16_summary["source_commit"]
            or a16_report.get("authorization") != a16_summary["authorization"]):
        _fail("A16 completed evidence receipt does not bind run/manifest/commit")

    control_plan = _doc(documents, CONTROL_PLAN_PATH)
    frozen_plan_path = a16_root + "frozen_plan.json"
    frozen_plan = _doc(documents, frozen_plan_path)
    if frozen_plan.get("read_plan") != control_plan:
        _fail("A16 frozen read plan differs from its controlled config")
    if (control_plan.get("source_run_id") != a14
            or control_plan.get("source_manifest_sha256") != SOURCE_MANIFESTS["A14"]
            or control_plan.get("authorization") != "kiki_a16_development_20261025"):
        _fail("A16 control plan does not bind the consumed A14 source")
    if _plain(frozen_plan.get("read_plan", {}).get("events")) != plan["population"]["cache_rows"]:
        _fail("A16 frozen event order differs from the eligible cache population")

    a16_allocated = _doc(documents, a16_root + "allocated_slots.json")
    if _population_projection(a16_allocated.get("original_population", {})) != _population_projection(
        plan["population"]
    ):
        _fail("A16 allocated population differs from the frozen A17 population")
    slots = a16_allocated.get("diagnostic_slots")
    expected_slots = plan["population"]["seed_event_slots"]
    if (not isinstance(slots, Sequence) or isinstance(slots, (str, bytes))
            or len(slots) != 774):
        _fail("A16 must retain all 774 diagnostic slots")
    slot_by_key = {}
    slot_order = []
    for row in slots:
        row = _dict(row, "A16 diagnostic slot")
        if (type(row.get("seed")) is not int or type(row.get("cache_index")) is not int
                or not isinstance(row.get("event_id"), str)
                or not isinstance(row.get("status"), str) or not row["status"]):
            _fail("A16 diagnostic slot identity/status fields are malformed")
        key = f"{row.get('seed')}-{row.get('cache_index', -1):03d}"
        if key in slot_by_key:
            _fail(f"duplicate A16 slot {key}")
        slot_by_key[key] = row
        slot_order.append(key)
    expected_keys = []
    for expected_slot in expected_slots:
        key = f"{expected_slot['seed']}-{expected_slot['cache_index']:03d}"
        expected_keys.append(key)
        row = slot_by_key.get(key)
        if (row is None or row.get("event_id") != expected_slot["event_id"]
                or row.get("complete") is not True):
            _fail(f"A16 slot identity/completion differs for {key}")
    if slot_order != expected_keys:
        _fail("A16 diagnostic slots are not in original seed/event order")

    a14_members = _doc(documents, a14_root + "source_members.json")
    a15_members = _doc(documents, a15_root + "source_member_fingerprints.json")
    expected_a14_source_keys = {
        "A10/real_train_arrays.npz",
        *(f"A09/seed-{seed}-h005/paths.npz" for seed in (324, 325, 326)),
        *(f"A10/seed-{seed}-h0025/paths.npz" for seed in (324, 325, 326)),
        *(f"A06/checkpoint-240-{seed}/model.npz" for seed in (324, 325, 326)),
    }
    if set(a14_members) != expected_a14_source_keys:
        _fail("A14 source_members provenance key set differs")
    for source, keys in (
        ("A09", [f"seed-{seed}-h005/paths.npz" for seed in (324, 325, 326)]),
        ("A10", ["real_train_arrays.npz", *[
            f"seed-{seed}-h0025/paths.npz" for seed in (324, 325, 326)
        ]]),
    ):
        if any(key not in manifests[source] for key in keys):
            _fail(f"A14 source ancestry file identities missing from {source} manifest")
    input_doc = _doc(documents, a14_root + "permitted_inputs.json")
    input_source = _complete_descriptor_map(
        a14_members, "A10/real_train_arrays.npz", input_doc, "A14 permitted inputs"
    )
    lineage_static = {}
    slot_bindings = {}
    ancestry_e = {}
    ancestry_truth = {}
    ancestry_clock = {}

    for seed in (324, 325, 326):
        static_path = f"{a14_root}static-{seed}.json"
        static_doc = _doc(documents, static_path)
        static_params = _dict(static_doc.get("parameters"), f"A14 static {seed} parameters")
        source_key = f"A06/checkpoint-240-{seed}/model.npz"
        source_map = _dict(a14_members.get(source_key), f"A14 ancestry {source_key}")
        static_descriptors = {}
        for name in ("mass", "edge_stiffness", "damping", "coupling", "latent_skew", "latent_dissipation"):
            descriptor = static_params.get(name)
            source_desc = source_map.get(name)
            static_descriptors[name] = _same_fingerprint(
                descriptor, source_desc, f"A14/A06 {seed}:{name}"
            )
        lineage_static[str(seed)] = {
            "source_key": source_key,
            "descriptors": {name: _descriptor_record(desc, f"static {seed}:{name}")
                            for name, desc in static_descriptors.items()},
            "source_member_descriptors": {
                name: _descriptor_record(source_map[name], f"A06 source {seed}:{name}")
                for name in static_descriptors
            },
        }

    # A15 upstream bulk fingerprints are returned for root's later bounded
    # slice/concatenation SHA checks; no values are decoded here.
    for seed in (324, 325, 326):
        coarse_key = f"A09/seed-{seed}-h005/paths.npz"
        fine_key = f"A10/seed-{seed}-h0025/paths.npz"
        coarse = _dict(a15_members.get(coarse_key), coarse_key)
        fine = _dict(a15_members.get(fine_key), fine_key)
        for label, item in (("coarse", coarse), ("fine", fine)):
            future_fp = _fingerprint(item.get("future"), f"{label} future {seed}")
            offsets_fp = _fingerprint(item.get("offsets"), f"{label} offsets {seed}")
            if future_fp["shape"] != [72_405, 22] or future_fp["dtype"] != "<f8":
                _fail(f"A15 {label} E source shape/dtype differs for seed {seed}")
            if offsets_fp["shape"] != [259] or offsets_fp["dtype"] != "<i8":
                _fail(f"A15 {label} offset source shape/dtype differs for seed {seed}")
        if (f"seed-{seed}-h005/paths.npz" not in manifests["A09"]
                or f"seed-{seed}-h0025/paths.npz" not in manifests["A10"]):
            _fail(f"A09/A10 bulk source files are absent from sealed manifests for seed {seed}")
        ancestry_e[f"{seed}-h005"] = {"source": coarse_key,
                                      "file_sha256": manifests["A09"].get(f"seed-{seed}-h005/paths.npz"),
                                      "future": _fingerprint(coarse["future"], coarse_key + ":future"),
                                      "offsets": _fingerprint(coarse["offsets"], coarse_key + ":offsets")}
        ancestry_e[f"{seed}-h0025"] = {"source": fine_key,
                                        "file_sha256": manifests["A10"].get(f"seed-{seed}-h0025/paths.npz"),
                                        "future": _fingerprint(fine["future"], fine_key + ":future"),
                                        "offsets": _fingerprint(fine["offsets"], fine_key + ":offsets")}

    truth_key = "A10/real_train_arrays.npz"
    truth_source = _dict(a15_members.get(truth_key), truth_key)
    for name in ("future", "forecast_time", "offsets", "omega_s"):
        if name not in truth_source:
            _fail(f"A15 truth/clock ancestry misses {name}")
    truth_shapes = {"future": ([72_405, 20], "<f8"),
                    "forecast_time": ([72_405], "<f8"),
                    "offsets": ([259], "<i8"), "omega_s": ([258], "<f8")}
    for name, (shape, dtype) in truth_shapes.items():
        fp = _fingerprint(truth_source[name], truth_key + ":" + name)
        if fp["shape"] != shape or fp["dtype"] != dtype:
            _fail(f"A15 source truth/clock shape/dtype differs for {name}")
    if "real_train_arrays.npz" not in manifests["A10"]:
        _fail("A10 truth source file is absent from its sealed manifest")
    ancestry_truth[truth_key] = {
        "file_sha256": manifests["A10"].get("real_train_arrays.npz"),
        "members": {
            name: _fingerprint(truth_source[name], truth_key + ":" + name)
            for name in ("future", "forecast_time", "offsets", "omega_s")
        },
    }

    for index, expected_slot in enumerate(expected_slots):
        seed, cache_index = expected_slot["seed"], expected_slot["cache_index"]
        key = f"{seed}-{cache_index:03d}"
        slot_row = slot_by_key[key]
        slot_doc_path = f"{a16_root}slot-{seed}-{cache_index:03d}.json"
        slot_doc = context_documents[f"weighted-{seed}-{cache_index:03d}"]
        impulse_doc_path = f"{a16_root}impulse-{cache_index:03d}.json"
        impulse_doc = context_documents[f"impulse-{cache_index:03d}"]
        if slot_doc.get("complete") is not True or slot_doc.get("status") != slot_row.get("status"):
            _fail(f"A16 slot result is incomplete for {key}")
        values = _dict(slot_doc.get("values"), f"A16 slot values {key}")
        gram = values.get("gram")
        if (not isinstance(gram, Sequence) or isinstance(gram, (str, bytes))
                or len(gram) != 2):
            _fail(f"A16 Gram descriptors missing for {key}")
        gram_fingerprints = []
        for part, descriptor in enumerate(gram):
            fp = _fingerprint(descriptor, f"A16 Gram {key}:{part}")
            if _fingerprint(descriptor, f"A16 Gram {key}:{part}")["shape"] != [2, 3] or descriptor.get("dtype") != "<f8":
                _fail(f"A16 Gram shape/dtype differs for {key}:{part}")
            if _plain(descriptor.get("columns")) != ["point", "lower", "upper"]:
                _fail(f"A16 Gram interval columns differ for {key}:{part}")
            gram_fingerprints.append(_descriptor_record(descriptor, f"A16 Gram {key}:{part}"))
        trace = _positive_interval(values.get("trace"), f"A16 trace {key}")
        determinant = _positive_interval(values.get("determinant"), f"A16 determinant {key}")

        integral_path = f"{a14_root}seed-{seed}-event-{cache_index:03d}/integral.json"
        integral_doc = _doc(documents, integral_path)
        integral_context = _context(plan, f"integral-{seed}-{cache_index:03d}")
        integral_desc = {}
        for member in integral_context["members"]:
            name = member["name"]
            integral_desc[name] = _same_descriptor(integral_doc.get(name), member,
                                                   f"A14 integral {key}:{name}")
        rank, threshold, tol, natural_norm = (integral_doc.get("rank"),
                                               integral_doc.get("svd", {}).get("threshold"),
                                               integral_doc.get("tol"),
                                               integral_doc.get("natural_norm"))
        if (type(rank) is not int or rank <= 0
                or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
                       for x in (threshold, tol, natural_norm))
                or threshold < 0 or tol < 0 or natural_norm <= 0
                or integral_doc.get("status") != "complete"):
            _fail(f"A14 rank/threshold/tolerance/natural norm invalid for {key}")
        if integral_doc.get("svd", {}).get("rank") != rank:
            _fail(f"A14 rank differs from SVD record for {key}")

        impulse_descriptors = {}
        for name in ("u11", "d"):
            sequence = impulse_doc.get(name)
            required = 11 if name == "u11" else 10
            if (not isinstance(sequence, Sequence) or isinstance(sequence, (str, bytes))
                    or len(sequence) != required):
                _fail(f"A16 impulse descriptor count differs for {key}:{name}")
            if name == "d":
                for i, desc in enumerate(sequence):
                    fp = _fingerprint(desc, f"A16 impulse {key}:d:{i}")
                    if _fingerprint(desc, f"A16 impulse {key}:d:{i}")["shape"] != [10, 3] or desc.get("dtype") != "<f8":
                        _fail(f"A16 d descriptor shape/dtype differs for {key}:{i}")
            impulse_descriptors[name] = [
                _descriptor_record(item, f"A16 impulse {key}:{name}:{i}")
                for i, item in enumerate(sequence)
            ]
        ancestry_clock[key] = _descriptor_record(
            _doc(documents, _metadata_document_path(_context(plan, f"clock-{cache_index:03d}")["path"])).get("clock"),
            f"A15 clock {key}",
        )
        record_descriptors = {}
        for suffix in ("h005", "h0025"):
            reference_key = f"reference-{seed}-{cache_index:03d}-{suffix}"
            reference_doc = context_documents[reference_key]
            record_descriptors[suffix] = {
                name: _descriptor_record(reference_doc[name], f"A15 record {key}:{suffix}:{name}")
                for name in ("candidate", "baseline", "reset")
            }
            if "truth" in reference_doc:
                record_descriptors[suffix]["truth"] = _descriptor_record(
                    reference_doc["truth"], f"A15 saved truth {key}:{suffix}"
                )
                ancestry_truth[key] = record_descriptors[suffix]["truth"]

        slot_bindings[key] = {
            "seed": seed,
            "cache_index": cache_index,
            "event_id": expected_slot["event_id"],
            "status": slot_row["status"],
            "complete": True,
            "integral_parameters": {"rank": rank, "threshold": threshold,
                                    "tol": tol, "natural_norm": natural_norm},
            "integral_descriptors": {name: _descriptor_record(desc, f"integral {key}:{name}")
                                     for name, desc in integral_desc.items()},
            "weighted_impulse": _descriptor_record(values["weighted_impulse"], f"weighted impulse {key}"),
            "gram": gram_fingerprints,
            "trace": trace,
            "determinant": determinant,
            "impulse": impulse_descriptors,
            "source_paths": {"integral": integral_path, "slot": slot_doc_path,
                             "impulse": impulse_doc_path},
            "future_records": record_descriptors,
        }

    # Also validate A14 source NPZ descriptors for the six permitted inputs and
    # six static parameters against their A17 per-file descriptors and lineage.
    for name, descriptor in input_doc.items():
        planned = contexts_by_id["inputs"]
        planned_member = next(item for item in planned["members"] if item["name"] == name)
        _same_descriptor(descriptor, planned_member, f"A14 planned input:{name}")
    for seed in (324, 325, 326):
        static_doc = _doc(documents, f"{a14_root}static-{seed}.json")
        for name in ("mass", "edge_stiffness", "damping", "coupling", "latent_skew", "latent_dissipation"):
            descriptor = static_doc["parameters"][name]
            planned_name = f"parameters__{name}"
            planned = contexts_by_id[f"static-{seed}"]
            _same_descriptor(descriptor, next(m for m in planned["members"] if m["name"] == planned_name),
                             f"A14 planned static {seed}:{name}")

    # Return compact ancestry descriptors; the root owns any later bounded
    # array slicing/concatenation and all scientific gates.
    return {
        "source_runs": dict(SOURCE_RUNS),
        "source_manifest_sha256": dict(SOURCE_MANIFESTS),
        "slot_bindings": slot_bindings,
        "ancestry_fingerprints": {
            "E": ancestry_e,
            "truth": ancestry_truth,
            "clock": ancestry_clock,
            "A14_inputs": input_source,
            "A14_static": lineage_static,
        },
        "checks_summary": {
            "metadata_envelopes": len(documents),
            "a16_complete_slots": len(slot_bindings),
            "a16_slots_retained_without_status_filter": True,
            "a14_verdict": "ACCEPT",
            "a15_verdict": "REJECT",
            "promotion_decision": "KEEP_BASELINE",
            "npz_manifest_bindings": len(plan["contexts"]),
            "metadata_manifest_bindings": sum(
                path.startswith(tuple(f"results/{run}/" for run in SOURCE_RUNS.values()))
                and not path.endswith("/artifact_hashes.json") for path in metadata_rows
            ),
        },
    }
