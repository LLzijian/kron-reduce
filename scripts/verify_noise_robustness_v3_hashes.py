"""Compare the two noise manifests after the registered one-shot V3 holdout."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from kron_reduce.data.task3_measurement_noise import SCENARIOS
from kron_reduce.workflows.noise_robustness_esrom_v3 import (
    REPORT_DIR,
    SEALED_FREEZE,
    _phase,
    _plans,
)
from project_harness.git_state import verify_bound_file
from project_harness.manifest import project_root

ROSTERS = ("val", "test", "blind_v2")
EXPECTED = {
    (roster, scenario, rep)
    for roster in ROSTERS for scenario in SCENARIOS
    for rep in range(1 if scenario == "clean" else 5)
}


def _digests(document):
    if isinstance(document, dict):
        document = document.get("rosters", document.get("entries", document))
    if isinstance(document, list):
        rows = ((row["roster"], row["scenario"], row["rep"], row["sha256"]) for row in document)
    else:
        rows = ((roster, scenario, rep, digest)
                for roster, conditions in document.items()
                for scenario, repeats in conditions.items()
                for rep, digest in repeats.items())
    output = {}
    for roster, scenario, rep, digest in rows:
        if isinstance(digest, dict):
            digest = digest["sha256"]
        if (not isinstance(roster, str) or not isinstance(scenario, str)
                or not isinstance(rep, (int, str)) or not re.fullmatch(r"[0-9]+", str(rep))
                or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)):
            raise ValueError("invalid repetition or SHA-256 in the noise manifest")
        key = (roster, scenario, int(rep))
        if key in output:
            raise ValueError(f"duplicate noise manifest key: {key}")
        output[key] = digest
    return output


def compare(local, peer):
    own, other = _digests(local), _digests(peer)
    if len(EXPECTED) != 138 or set(own) != EXPECTED:
        raise ValueError("the local manifest must cover the exact 138 registered groups")
    missing = sorted(EXPECTED - set(other))
    extra = sorted(set(other) - EXPECTED)
    wrong = [{"group": key, "local": own[key], "peer": other[key]}
             for key in sorted(EXPECTED & set(other)) if own[key] != other[key]]
    return {
        "status": "FAIL" if missing or extra or wrong else "PASS",
        "allocated": len(EXPECTED), "matched": len(EXPECTED) - len(missing) - len(wrong),
        "missing": missing, "unexpected": extra, "mismatches": wrong,
    }


def main():
    root = project_root()
    plan, _, digest = _plans(root)
    _, freeze = _phase(root, "freeze", digest)
    heldout, evidence = _phase(root, "blind", digest)
    bound_freeze = freeze["frozen_model_manifest"]["sha256"]
    verify_bound_file(root, SEALED_FREEZE, bound_freeze)
    if evidence["frozen_model_manifest"]["sha256"] != bound_freeze:
        raise PermissionError("the one-shot holdout must match the accepted V3 freeze")
    local = (heldout / "noise_manifest.json").read_bytes()
    local_sha = hashlib.sha256(local).hexdigest()
    if local_sha != evidence["noise_manifest_sha256"]:
        raise ValueError("the local noise manifest differs from the held-out receipt")
    peer_path = Path(plan["counterpart"]) / "noise_manifest.json"
    peer = peer_path.read_bytes()
    result = compare(json.loads(local), json.loads(peer))
    result.update({
        "holdout_run": evidence["artifact_manifest"], "plan_sha256": digest,
        "local_sha256": local_sha, "peer_sha256": hashlib.sha256(peer).hexdigest(),
        "peer_path": peer_path.as_posix(),
    })
    destination = root / REPORT_DIR / "_sealed" / "noise_cross_repository.json"
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
