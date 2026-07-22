from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-commit")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--refresh-legacy-manifest", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "src"))
    from project_harness.data_check import verify_data

    result = verify_data(root)
    if result["status"] != "PASS":
        failures = [row for row in result["findings"] if not row["pass"]]
        raise SystemExit(f"canonical copy verification failed: {failures[:3]}")
    dataset_dir = root / "data" / "ieee39_postclear_canonical_v1"
    legacy = root / "legacy"
    manifest_path = legacy / "source_manifest.sha256"
    if args.verify_only and args.refresh_legacy_manifest:
        parser.error("--verify-only and --refresh-legacy-manifest are mutually exclusive")
    if not args.refresh_legacy_manifest:
        legacy_failures = []
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            expected, relative = line.split("  ", 1)
            path = legacy / relative
            if not path.is_file() or sha256_file(path) != expected:
                legacy_failures.append(relative)
        if legacy_failures:
            raise SystemExit(f"legacy snapshot verification failed: {legacy_failures[:3]}")
    if args.verify_only:
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "canonical_files": result["files"],
                    "canonical_events": result["event_count"],
                    "legacy_files": len(manifest_path.read_text(encoding="utf-8").splitlines()),
                },
                indent=2,
            )
        )
        return
    if not args.source_commit:
        parser.error("--source-commit is required unless --verify-only is used")
    provenance = {
        "source_repository": "swing_equation",
        "source_git_commit": args.source_commit,
        "copied_at": datetime.now(timezone.utc).isoformat(),
        "copy_policy": "byte_for_byte_no_materialization",
        "dataset_revision": result["dataset_revision"],
        "build_id": result["build_id"],
        "verified_files": result["files"],
        "verified_events": result["event_count"],
        "h5_sha256": sha256_file(dataset_dir / "ieee39_postclear_canonical_v1.h5"),
    }
    (dataset_dir / "COPY_PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    lines = []
    for path in sorted(p for p in legacy.rglob("*") if p.is_file() and p != manifest_path):
        lines.append(f"{sha256_file(path)}  {path.relative_to(legacy).as_posix()}")
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
