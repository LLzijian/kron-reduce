"""Strict metadata-only validation for the frozen A17 production read plan."""
from __future__ import annotations

import hashlib
import json
import math
import re

from kron_reduce.data.input_consistency_reader import RUNS, validate_saved_plan

AUTH = "kiki_a17_development_20261025"
PLAN_KIND = "A17_finite_saved_input_consistency_reads_v1"
POPULATION_SHA256 = "2daa8571fb10321b0853ebe3342e533e4381ab356c549bfc860f5aa25a2f474a"
CONTROL_PLAN_PATH = "configs/proposals/kiki_a16_pm_impulse_reads.json"
CONTROL_PLAN_SIZE = 1_989_555
CONTROL_PLAN_SHA256 = "ba39f016dbb6ef468be09e68af595501dd69cf283c26acc9fbdb128fb2642bd0"
SEEDS = (324, 325, 326)
STEPS = (("h005", "0.005"), ("h0025", "0.0025"))
EXPECTED_LIMITS = {
    "npz_unique_files": 3358,
    "npz_contexts": 3616,
    "descriptors": 20406,
    "arrays": 20406,
    "array_elements": 21486372,
    "array_logical_hash_bytes": 171890976,
    "opaque_hash_passes": 6716,
    "opaque_hash_bytes": 641698014,
    "metadata_files": 3376,
    "metadata_bytes": 72922783,
    "npy_header_bytes_max": 204060000,
    "npy_header_per_member_cap_including_magic_and_length": 10000,
}
SOURCE_MANIFESTS = {
    "A09": "ae87ac95d337a9c8a1d4fdadd305701190e704cf9a85923f0ab9e27dbbfd4573",
    "A10": "41741c4a3ab0973be2053cb23e29efd06ceaac3af02239e72f2cda140dccbc86",
    "A14": "7438242d358244dece184a393ef871374322d04415237fff733ecd7c01767938",
    "A15": "e67dbb7c2d62ac82017e9d1049fd7a01c492c6b9c3587aa841d7bb89d247f1c4",
    "A16": "568a6271fd09dd98fb15c9c18e6d9a4c22799d639817f3fc988326dfd7134f7d",
}
SOURCE_RUNS = {
    "A09": "paper_kiki_a09_train_mechanism-train_mechanism-20261024T054046076318Z-ff5ae772-e3d8db5f",
    "A10": "paper_kiki_a10_train_refinement-train_refinement-20261024T062534123221Z-161cae52-515f5625",
    **RUNS,
}
CONTROL_METADATA = {
    f"results/{SOURCE_RUNS['A09']}/artifact_hashes.json": (2296, SOURCE_MANIFESTS["A09"]),
    f"results/{SOURCE_RUNS['A10']}/artifact_hashes.json": (1784, SOURCE_MANIFESTS["A10"]),
    f"results/{RUNS['A14']}/artifact_hashes.json": (1_128_155, SOURCE_MANIFESTS["A14"]),
    f"results/{RUNS['A15']}/artifact_hashes.json": (1_131_620, SOURCE_MANIFESTS["A15"]),
    f"results/{RUNS['A16']}/artifact_hashes.json": (189_054, SOURCE_MANIFESTS["A16"]),
    f"results/{RUNS['A14']}/allocated_population.json": (
        252_865, "6bfc5c51a5fb35ed011c7e49b2df9d126ffc443ecb0f114be607833c264089db"
    ),
    f"results/{RUNS['A14']}/source_members.json": (
        394_385, "2fc7469ded60ec250d766f564be88e8c962eadabbfaeea671c28eaf265daa821"
    ),
    f"results/{RUNS['A15']}/source_member_fingerprints.json": (
        654_495, "0b2d60eff347802d5241ad4fe89ebee33ba5600f361ac432311237c05fa52327"
    ),
    f"results/{RUNS['A15']}/allocated_population.json": (
        258_219, "598309c5a69426b4efa7373335599c30002df66f725b8a8829b2a96cf666ac05"
    ),
    f"results/{SOURCE_RUNS['A09']}/scientific_plan.json": (
        24_259, "3862836e3b4596c429a4bbd645b2da62efe5657f2392e8695da867f42c1a8865"
    ),
    f"results/{SOURCE_RUNS['A10']}/scientific_plan.json": (
        25_414, "5b320b07f4034c9ddff253b827dd2eac000bdadf12fb126bfb98aa80aea64646"
    ),
    CONTROL_PLAN_PATH: (
        CONTROL_PLAN_SIZE,
        CONTROL_PLAN_SHA256,
    ),
    f"results/{RUNS['A16']}/frozen_plan.json": (
        2_156_571, "48e1beb23bd0cebdeb8e96eb76169feac129bd9f61d27e73dcfa50cdd1ddd8ac"
    ),
    f"results/{RUNS['A16']}/allocated_slots.json": (
        402_014, "c48f6c7a6503825f6c669e601b78c01b3124855fa5c39c86cf5c52f654e695c0"
    ),
    f"results/{RUNS['A16']}/summary.json": (
        1_164, "25d6528eff5325b6efa87197ddac120624689a1bffa5ba7ee8b326e3bd876136"
    ),
    f"results/{RUNS['A14']}/summary.json": (
        2_076, "9cf1ab27c4d0935f6f013c14e143c76e7ea324d5b44057bd7266c30a62f91ba8"
    ),
    f"results/{RUNS['A15']}/summary.json": (
        2_091, "6a52e96793c202d24e40bcddfe76f08bf218c060b631c885627a5a17dbc18fc0"
    ),
    "docs/reports/kiki_a16_pm_impulse_20261025_evidence.json": (
        15_561, "8d77a73edcd35bedde19959ec7f0eba3f41fa39a9460043871ea1acbc69fc4c8"
    ),
}
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def _fail(message: str) -> None:
    raise ValueError(f"invalid A17 production read plan: {message}")


def _safe_relative(path: object) -> bool:
    return (isinstance(path, str) and bool(path) and "\\" not in path and ":" not in path
            and not path.startswith("/") and "//" not in path
            and all(part not in {"", ".", ".."} for part in path.split("/")))


def _add_context(expected: dict, identity: str, source: str, phase: int, path: str,
                 members: list[tuple[str, tuple[int, ...]]]) -> None:
    if identity in expected:
        _fail(f"duplicate generated context identity {identity}")
    expected[identity] = {
        "source": source,
        "phase": phase,
        "path": path,
        "members": members,
    }


def _expected_contexts(population: dict) -> dict[str, dict]:
    expected = {}
    a14 = f"results/{RUNS['A14']}"
    a15 = f"results/{RUNS['A15']}"
    a16 = f"results/{RUNS['A16']}"
    _add_context(expected, "inputs", "A14", 1, f"{a14}/permitted_inputs.npz", [
        ("history", (258, 11, 20)), ("history_time", (258, 11)),
        ("constant_power", (258, 10)), ("equilibrium", (258, 10)),
        ("mass", (10,)), ("target_scale", (20,)),
    ])
    static_members = [
        ("parameters__mass", (10,)), ("parameters__edge_stiffness", (45,)),
        ("parameters__damping", (10, 10)), ("parameters__coupling", (10, 2)),
        ("parameters__latent_skew", (2, 2)), ("parameters__latent_dissipation", (2, 2)),
    ]
    integral_members = [
        ("Ltilde", (100, 2)), ("U", (100, 2)), ("s", (2,)), ("Vh", (2, 2)),
        ("btilde4", (100,)), ("c", (10,)), ("S", (10,)), ("h", (10,)),
        ("time", (11,)), ("zh2", (2,)), ("zh4", (2,)), ("zTobs2", (2,)),
        ("zTobs4", (2,)), ("PhiT", (2, 2)), ("wT", (2,)),
    ]
    for seed in SEEDS:
        _add_context(expected, f"static-{seed}", "A14", 1,
                     f"{a14}/static-{seed}.npz", static_members)
        for index in range(258):
            _add_context(expected, f"integral-{seed}-{index:03d}", "A14", 1,
                         f"{a14}/seed-{seed}-event-{index:03d}/integral.npz", integral_members)
    for seed in SEEDS:
        for index in range(258):
            _add_context(expected, f"weighted-{seed}-{index:03d}", "A16", 1,
                         f"{a16}/slot-{seed}-{index:03d}.npz",
                         [("root__values__weighted_impulse", (100, 3))])
    impulse_members = [(f"root__u11__{index}", (10, 3)) for index in range(11)]
    for index in range(258):
        _add_context(expected, f"impulse-{index:03d}", "A16", 1,
                     f"{a16}/impulse-{index:03d}.npz", impulse_members)
    for index, point_count in enumerate(population["future_point_counts"]):
        path = f"{a15}/seed-324-event-{index:03d}/future-0.005.npz"
        _add_context(expected, f"clock-{index:03d}", "A15", 2, path,
                     [("clock", (point_count + 1,))])
    for seed in SEEDS:
        for index, point_count in enumerate(population["future_point_counts"]):
            for suffix, step in STEPS:
                members = [("candidate", (point_count + 1, 22)),
                           ("baseline", (point_count + 1, 22)), ("reset", (22,))]
                if seed == 324 and suffix == "h005":
                    members.append(("truth", (point_count + 1, 20)))
                _add_context(expected, f"reference-{seed}-{index:03d}-{suffix}", "A15", 3,
                             f"{a15}/seed-{seed}-event-{index:03d}/future-{step}.npz", members)
    return expected


def _expected_freeze_slots() -> list[str]:
    return [f"{seed}-{index:03d}-{suffix}" for seed in SEEDS for index in range(258)
            for suffix, _step in STEPS]


def _validate_population(population: object) -> str:
    if not isinstance(population, dict):
        _fail("population must be a mapping")
    try:
        canonical = json.dumps(population, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("invalid A17 production read plan: population is not canonical JSON") from error
    digest = hashlib.sha256(canonical).hexdigest()
    if digest != POPULATION_SHA256:
        _fail("canonical population SHA differs")
    required = {"allocated_slots", "cache_rows", "weights", "seed_event_slots", "future_point_counts"}
    if set(population) != required:
        _fail("population field set differs")
    allocated, cache, weights, slots, counts = (population[key] for key in (
        "allocated_slots", "cache_rows", "weights", "seed_event_slots", "future_point_counts"
    ))
    if (len(allocated) != 268 or len(cache) != 258 or len(weights) != 258
            or len(slots) != 774 or len(counts) != 258
            or any(type(value) is not int for value in counts)
            or sum(counts) != 72_147):
        _fail("fixed allocated/cache/seed-slot/N_i population counts differ")
    cache_ids = [row.get("event_id") for row in cache]
    if len(set(cache_ids)) != 258:
        _fail("cache event identities are not unique")
    eligible = [row for row in allocated if row.get("input_status") == "eligible"]
    if eligible != cache:
        _fail("cache rows do not equal the ordered eligible allocation")
    for index, row in enumerate(cache):
        if (row.get("cache_index") != index or row.get("role") != "train"
                or row.get("source_role") != "train" or type(row.get("declared_length")) is not int
                or row["declared_length"] < 12 or row.get("event_id") != f"{row.get('source_dataset')}/{row.get('raw_id')}"):
            _fail("eligible cache row identity/role/order differs")
    expected_slots = [
        {"seed": seed, "cache_index": index, "event_id": row["event_id"]}
        for seed in SEEDS for index, row in enumerate(cache)
    ]
    if slots != expected_slots:
        _fail("seed/event allocation differs from fixed seed-major order")
    if any(not isinstance(weight, (int, float)) or isinstance(weight, bool)
           or not math.isfinite(weight) or weight <= 0 for weight in weights):
        _fail("original weights must be finite positive numbers")
    return digest


def validate_production_plan(plan: dict) -> dict:
    """Check only the frozen A17 selector/metadata structure; never opens files."""
    try:
        validate_saved_plan(plan)
    except (KeyError, TypeError, ValueError, PermissionError) as error:
        raise ValueError(f"invalid A17 production read plan: base reader contract: {error}") from error
    if (plan.get("kind") != PLAN_KIND or plan.get("authorization") != AUTH
            or plan.get("execution_authorized") is not False):
        _fail("plan kind, authorization, or inactive execution flag differs")
    if plan.get("phase_order") != {
        "1": "all 1548 candidate origins frozen",
        "2": "all 1548 candidate futures frozen",
        "3": "I/E/truth decode and fixed scoring only",
    }:
        _fail("phase order differs")
    if plan.get("source_runs") != SOURCE_RUNS or plan.get("source_manifest_sha256") != SOURCE_MANIFESTS:
        _fail("the five sealed source run identities differ")
    population_sha = _validate_population(plan.get("population"))
    if population_sha != plan.get("freeze_contract", {}).get("population_sha256"):
        _fail("freeze contract population identity differs")

    freeze = plan["freeze_contract"]
    expected_slots = _expected_freeze_slots()
    if (freeze.get("slot_ids") != expected_slots or len(expected_slots) != 1548
            or freeze.get("max_bytes") != 1_048_576
            or freeze.get("phase2_filename") != "origins_freeze.json"
            or freeze.get("phase3_filename") != "futures_freeze.json"):
        _fail("ordered origin/future freeze contract differs")

    expected = _expected_contexts(plan["population"])
    rows = plan["contexts"]
    if len(rows) != EXPECTED_LIMITS["npz_contexts"] or len(expected) != len(rows):
        _fail("context total differs")
    rows_by_id = {}
    unique_files = {}
    selector_count = element_count = 0
    expected_metadata_paths = set(CONTROL_METADATA)
    for row in rows:
        identity = row.get("context_id")
        if identity in rows_by_id or identity not in expected:
            _fail(f"unknown or duplicate context identity {identity!r}")
        rows_by_id[identity] = row
        spec = expected[identity]
        if (row.get("source") != spec["source"] or row.get("phase") != spec["phase"]
                or row.get("path") != spec["path"] or not _safe_relative(row.get("path"))):
            _fail(f"context source/phase/path differs for {identity}")
        members = row.get("members")
        if not isinstance(members, list) or len(members) != len(spec["members"]):
            _fail(f"member count differs for {identity}")
        for member, (name, shape) in zip(members, spec["members"], strict=True):
            if (member.get("name") != name or member.get("shape") != list(shape)
                    or member.get("dtype") != "<f8"
                    or not isinstance(member.get("sha256_c_order_bytes"), str)
                    or HEX64.fullmatch(member["sha256_c_order_bytes"]) is None):
                _fail(f"member identity/shape/dtype/SHA differs for {identity}:{name}")
            selector_count += 1
            element_count += math.prod(shape)
        path, size, digest = row["path"], row.get("size_bytes"), row.get("sha256")
        if (type(size) is not int or size <= 0 or not isinstance(digest, str)
                or HEX64.fullmatch(digest) is None):
            _fail(f"opaque file size/SHA is invalid for {identity}")
        previous = unique_files.get(path)
        if previous is not None and previous != (size, digest):
            _fail(f"duplicate context file identity conflicts for {path}")
        unique_files[path] = (size, digest)
        expected_metadata_paths.add(path.removesuffix(".npz") + ".json")
    if set(rows_by_id) != set(expected):
        _fail("one or more frozen contexts are missing")

    if (len(unique_files) != EXPECTED_LIMITS["npz_unique_files"]
            or selector_count != EXPECTED_LIMITS["descriptors"]
            or selector_count != EXPECTED_LIMITS["arrays"]
            or element_count != EXPECTED_LIMITS["array_elements"]
            or element_count * 8 != EXPECTED_LIMITS["array_logical_hash_bytes"]
            or sum(size for size, _sha in unique_files.values()) * 2
            != EXPECTED_LIMITS["opaque_hash_bytes"]):
        _fail("derived unique-file/selector/array/opaque-byte counts differ")

    metadata = plan["metadata_files"]
    if len(metadata) != EXPECTED_LIMITS["metadata_files"]:
        _fail("metadata file count differs")
    metadata_by_path = {}
    for row in metadata:
        path, size, digest = row.get("path"), row.get("size_bytes"), row.get("sha256")
        if (not _safe_relative(path) or not path.endswith(".json") or path in metadata_by_path
                or type(size) is not int or size <= 0 or not isinstance(digest, str)
                or HEX64.fullmatch(digest) is None):
            _fail("metadata identity is unsafe, duplicate, or malformed")
        metadata_by_path[path] = row
    if set(metadata_by_path) != expected_metadata_paths:
        _fail("metadata path whitelist does not exactly cover NPZ companions and fixed controls")
    for path, (size, digest) in CONTROL_METADATA.items():
        row = metadata_by_path[path]
        if row["size_bytes"] != size or row["sha256"] != digest:
            _fail(f"fixed metadata control identity differs for {path}")
    if sum(row["size_bytes"] for row in metadata) != EXPECTED_LIMITS["metadata_bytes"]:
        _fail("metadata byte total differs")
    if EXPECTED_LIMITS["npy_header_bytes_max"] != selector_count * 10_000:
        _fail("bounded NPY header total differs")
    if plan.get("limits") != EXPECTED_LIMITS:
        _fail("frozen operation and byte limits differ")

    return {
        "authorization": AUTH,
        "execution_authorized": False,
        "population_sha256": population_sha,
        "source_runs": dict(SOURCE_RUNS),
        "source_manifest_sha256": dict(SOURCE_MANIFESTS),
        "counts": dict(EXPECTED_LIMITS),
        "npz_identities": {path: {"size_bytes": size, "sha256": sha}
                            for path, (size, sha) in unique_files.items()},
        "metadata_identities": {path: {"size_bytes": row["size_bytes"], "sha256": row["sha256"]}
                                for path, row in metadata_by_path.items()},
    }
