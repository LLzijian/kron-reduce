from __future__ import annotations

import ast
import json
from pathlib import Path

from .manifest import load_manifest


REQUIRED = (
    "AGENTS.md", "README.md", "pyproject.toml", "configs/project_manifest.json",
    "src/kron_reduce/data", "src/kron_reduce/models", "project_harness", "tests", "paper",
)


def audit_project(root: Path) -> dict:
    findings = []
    for relative in REQUIRED:
        if not (root / relative).exists():
            findings.append({"severity": "ERROR", "check": "required_path", "message": f"missing {relative}"})
    manifest = load_manifest(root)
    if manifest["default_dataset"] not in manifest["datasets"]:
        findings.append({"severity": "ERROR", "check": "manifest", "message": "default dataset is not registered"})
    for path in (root / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "E:\\03-nn_Transfer_Func\\swing_equation" in text or "E:/03-nn_Transfer_Func/swing_equation" in text:
            findings.append({"severity": "ERROR", "check": "absolute_legacy_path", "message": str(path.relative_to(root))})
        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            module = None
            if isinstance(node, ast.ImportFrom): module = node.module
            elif isinstance(node, ast.Import):
                for name in node.names:
                    if name.name.startswith("legacy"):
                        findings.append({"severity": "ERROR", "check": "legacy_import", "message": str(path.relative_to(root))})
            if module and (module.startswith("legacy") or "hist_3" in module):
                findings.append({"severity": "ERROR", "check": "legacy_import", "message": str(path.relative_to(root))})
    status = "PASS" if not any(row["severity"] == "ERROR" for row in findings) else "FAIL"
    return {"status": status, "errors": sum(row["severity"] == "ERROR" for row in findings), "findings": findings}
