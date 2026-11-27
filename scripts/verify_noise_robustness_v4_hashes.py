"""Compare the 138 noise groups after the registered V4 holdout."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from verify_noise_robustness_v3_hashes import compare

from kron_reduce.workflows.noise_robustness_esrom_v4 import (
    REPORT_DIR,
    SEALED_FREEZE,
    _phase,
    _plans,
)
from project_harness.git_state import verify_bound_file
from project_harness.manifest import project_root


def main():
    root = project_root()
    plan, _, digest = _plans(root)
    _, freeze = _phase(root, "freeze", digest)
    heldout, evidence = _phase(root, "blind", digest)
    bound_freeze = freeze["frozen_model_manifest"]["sha256"]
    verify_bound_file(root, SEALED_FREEZE, bound_freeze)
    if evidence["frozen_model_manifest"]["sha256"] != bound_freeze:
        raise PermissionError("the one-shot holdout must match the accepted V4 freeze")
    local = (heldout / "noise_manifest.json").read_bytes()
    local_sha = hashlib.sha256(local).hexdigest()
    if local_sha != evidence["noise_manifest_sha256"]:
        raise ValueError("the local noise manifest differs from the held-out receipt")
    peer_path = Path(plan["counterpart"]) / "noise_manifest.json"
    peer = peer_path.read_bytes()
    publisher = json.loads(peer)
    if (publisher["namespace"] != "esrom_task3_noise_v1" or publisher["dtype"] != "<f8"
            or publisher["bit_generator"] != "PCG64"
            or publisher["layout"] != ["eps_delta[11,10]", "eps_omega[11,10]", "eps_pm[10]"]):
        raise ValueError("peer noise generator metadata differs from the TASK3 protocol")
    records = publisher["records"]
    if any(row["n_events"] != plan["counts"].get(row["roster"]) for row in records):
        raise ValueError("peer noise groups do not bind the complete registered rosters")
    result = compare(json.loads(local), records)
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
