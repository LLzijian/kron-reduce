from __future__ import annotations

import pytest

from project_harness import runner


@pytest.mark.parametrize("historical_reproduction", [False, True])
def test_generic_runner_refuses_before_any_integrity_or_workflow_access(
    tmp_path, monkeypatch, historical_reproduction,
):
    experiment = {"execution_mode": "bounded_input_audit_cli_only", "dataset": "kiki_ieee39_v1"}
    monkeypatch.setattr(runner, "load_manifest", lambda _root: {"experiments": {"audit": experiment}})
    touched = []

    def forbidden(*_args, **_kwargs):
        touched.append(True)
        raise AssertionError("the generic entry must not reach any data or workflow operation")

    for name in ("validate_run_policy", "_verify_development_text_bindings", "verify_data",
                 "_run_experiment", "load_policy"):
        monkeypatch.setattr(runner, name, forbidden)
    with pytest.raises(RuntimeError, match="BOUNDED_INPUT_AUDIT_CLI_ONLY"):
        runner.run_experiment(
            tmp_path, "audit", "input_audit", historical_reproduction=historical_reproduction,
        )
    assert touched == []
