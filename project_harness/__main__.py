from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit import audit_project
from .data_check import verify_data
from .git_state import assert_clean, status_lines
from .loop import approve, load_state, next_stage, record_stage, run_one
from .manifest import load_manifest, project_root
from .runner import run_experiment


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="project_harness")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("list", "audit", "git-check", "data-check"): sub.add_parser(name)
    run_parser = sub.add_parser("run"); run_parser.add_argument("experiment"); run_parser.add_argument("--stage", required=True)
    loop_parser = sub.add_parser("loop"); loop_parser.add_argument("--workflow", default="paper_v1"); loop_sub = loop_parser.add_subparsers(dest="loop_command", required=True)
    loop_sub.add_parser("status"); loop_sub.add_parser("next"); loop_sub.add_parser("run-one")
    approve_parser = loop_sub.add_parser("approve"); approve_parser.add_argument("gate")
    advance_parser = loop_sub.add_parser("advance")
    advance_parser.add_argument("stage")
    advance_parser.add_argument("--evidence-file", required=True)
    args = parser.parse_args(argv); root = project_root()
    if args.command == "list": print(json.dumps(load_manifest(root), indent=2))
    elif args.command == "audit":
        result=audit_project(root); print(json.dumps(result,indent=2)); raise SystemExit(0 if result["status"]=="PASS" else 1)
    elif args.command == "data-check":
        result=verify_data(root); printable={**result,"findings":[row for row in result["findings"] if not row["pass"]]}; print(json.dumps(printable,indent=2)); raise SystemExit(0 if result["status"]=="PASS" else 1)
    elif args.command == "git-check":
        assert_clean(root); print(json.dumps({"status":"PASS","dirty":status_lines(root)},indent=2))
    elif args.command == "run": print(run_experiment(root,args.experiment,args.stage))
    elif args.loop_command == "status": print(json.dumps(load_state(root,args.workflow),indent=2))
    elif args.loop_command == "next": print(next_stage(root,args.workflow))
    elif args.loop_command == "run-one": print(json.dumps(run_one(root,args.workflow),indent=2))
    elif args.loop_command == "approve": print(json.dumps(approve(root,args.gate,args.workflow),indent=2))
    elif args.loop_command == "advance":
        evidence_path = Path(args.evidence_file)
        if not evidence_path.is_absolute():
            evidence_path = root / evidence_path
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        print(json.dumps(record_stage(root,args.stage,evidence,args.workflow),indent=2))


if __name__ == "__main__": main()
