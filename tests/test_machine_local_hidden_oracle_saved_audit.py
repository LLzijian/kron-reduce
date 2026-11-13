"""Small arithmetic-only checks for the saved-array audit helpers."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SCRIPT = ROOT / "scripts" / "audit_machine_local_hidden_oracle_saved.py"
spec = importlib.util.spec_from_file_location("saved_audit", SCRIPT)
audit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = audit
spec.loader.exec_module(audit)
from kron_reduce.evaluation import machine_local_hidden_oracle as module


def test_synthetic_exact_and_mutated_metrics():
    path = np.zeros((1, 811, 30), dtype=np.float64)
    forecast = np.zeros((1, 801, 30), dtype=np.float64)
    arrays = {"fine_forecast": forecast.copy(), "refined_forecast": forecast.copy()}
    labels = [module.class_for(path[0], path[0, 10:], np.ones(10))]
    saved = module.describe(arrays, path, labels, np.ones(20), np.ones(20), np.ones(10), True)
    _, exact = audit.compare_role(module, "matched_oracle", "train", arrays, path,
                                  labels, np.ones(20), np.ones(20), np.ones(10), saved)
    assert exact == []
    altered = {**saved, "allocated": 2}
    _, mismatch = audit.compare_role(module, "matched_oracle", "train", arrays, path,
                                     labels, np.ones(20), np.ones(20), np.ones(10), altered)
    assert any(row["path"].endswith(".allocated") for row in mismatch)


def test_synthetic_failed_truth_gate_is_detected_without_candidates():
    paths = np.zeros((64, 811, 30), dtype=np.float64)
    classes = [module.class_for(path, path[10:], np.ones(10)) for path in paths]
    scale = np.maximum(np.sqrt(np.mean(paths[:32, 11:311, :20] ** 2, axis=(0, 1))), 1e-5)
    hscale = np.maximum(np.sqrt(np.mean(paths[:32, :11, :20] ** 2, axis=(0, 1))), 1e-5)
    arrays = {"paths": paths, "coarse_paths": paths.copy(), "scale": scale,
              "history_scale": hscale, "mass": np.ones(10), "ids": np.arange(64),
              "roles": np.array(["train"]*32+["val"]*16+["test"]*16)}
    truth = {"pass": False, "class_coverage_pass": False, "step_difference": 0.0,
             "classes": classes, "coarse_classes": classes}
    computed, passed, checks, mismatches = audit.load_truth_inputs(module, arrays, truth)
    assert len(computed) == 64 and not passed and not mismatches
    assert all(check["passed"] for check in checks)
    summary = {"scientific_pass": False, "truth_pass": False, "numerical_pass": False,
               "matched_recovery_pass": False, "evaluation_complete": False, "records": {},
               "paired_differences": {}, "fit_attempts": 0, "noise_injection": 0.0,
               "t0_allowed": False, "next_route": "REVIEW_TRUTH_GATE_FAILURE"}
    result = audit.recompute(module, arrays, truth, {}, {}, {}, summary)
    assert result["mismatch_count"] == 0

