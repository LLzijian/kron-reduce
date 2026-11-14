import math
from importlib.metadata import version

import pytest
import torch

from kron_reduce.evaluation import kiki_machine_local_r20_r21 as paired
from kron_reduce.workflows import kiki_machine_local_r20_r21 as workflow


class _FakeModel(torch.nn.Module):
    def __init__(self, name, trainable_stiffness):
        super().__init__()
        self.name = name
        self.value = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.log_stiffness_factor = (
            torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
            if trainable_stiffness else None
        )
        self.step = 0


class _Budget:
    def spend(self, *_args):
        pass


def _models():
    return {"r20": _FakeModel("r20", False), "r21": _FakeModel("r21", True)}


def _patch_training(monkeypatch, values):
    def train_updates(model, _train, _scales, _budget, _target, settings, **_kwargs):
        for update in range(1, settings["updates"]+1):
            model.step = update
            with torch.no_grad():
                model.value.fill_(update)
                if model.log_stiffness_factor is not None:
                    model.log_stiffness_factor.fill_(update/100)
            yield {"update": update}

    def monitor(model, *_args):
        value = values[model.name][model.step]
        return {"nmse": value, "event_nmse": [], "qualified": 0, "failures": []}

    monkeypatch.setattr(paired, "train_updates", train_updates)
    monkeypatch.setattr(paired, "monitor", monitor)


def test_pair_stops_on_common_patience_and_reloads_independent_earliest_minima(tmp_path, monkeypatch):
    values = {
        "r20": {10: 1., 20: .5, 30: .5, 40: .6, 50: .7},
        "r21": {10: 2., 20: 1.5, 30: 1., 40: 1.1, 50: 1.2},
    }
    _patch_training(monkeypatch, values)
    models = _models()
    result = paired.fit_pair(
        models, (), (), {}, _Budget(), tmp_path,
        {"updates": 80, "train_batch": 2, "val_interval": 10, "val_patience": 2},
    )

    selection = paired.read_json(tmp_path/"selection.json")
    assert selection["stop_reason"] == "joint_val_plateau"
    assert [row["update"] for row in selection["checkpoints"]] == [10, 20, 30, 40, 50]
    assert result["r20"]["selected_update"] == 20
    assert result["r21"]["selected_update"] == 30
    assert result["r20"]["selected_val_nmse"] == .5
    assert result["r21"]["selected_val_nmse"] == 1.
    assert models["r20"].value.item() == 20
    assert models["r21"].value.item() == 30
    assert math.exp(models["r21"].log_stiffness_factor.item()) == math.exp(.3)


def test_null_checkpoint_does_not_advance_joint_patience(tmp_path, monkeypatch):
    values = {
        "r20": {10: 1., 20: None, 30: 1., 40: 1.},
        "r21": {10: 2., 20: 2., 30: 2., 40: 2.},
    }
    _patch_training(monkeypatch, values)
    result = paired.fit_pair(
        _models(), (), (), {}, _Budget(), tmp_path,
        {"updates": 40, "train_batch": 2, "val_interval": 10, "val_patience": 2},
    )

    selection = paired.read_json(tmp_path/"selection.json")
    assert selection["stop_reason"] == "joint_val_plateau"
    assert [row["update"] for row in selection["checkpoints"]] == [10, 20, 30, 40]
    assert [row["joint_patience_count"] for row in selection["checkpoints"]] == [0, 0, 1, 2]
    assert result["r20"]["selected_update"] == 10
    assert result["r21"]["selected_update"] == 10


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_authorization_uses_the_registered_experiment_and_approval(tmp_path, monkeypatch):
    import json

    name, authorization = "paired_cuda_fit_fixture", "paired_cuda_fit_fixture_approval"
    entry = {"module": workflow.MODULE, "workflow": "paired_cuda_fixture", "stage": "fit",
             "development_authorization_id": authorization}
    plan = {"stage": "fit", "device": "cuda:0", "protocol_path": "protocol.md",
            "protocol_sha256": "fixture", "operation_limits": {"fit": {}},
            "settings": {"updates": 80, "train_batch": 64, "eval_batch": 16,
                         "val_interval": 10, "val_patience": 3},
            "implementation_bindings": {}, "backbone": {},
            "source_contract": {"path": "source.json", "sha256": "fixture"}}
    auth = {"status": "active", "max_executions": 1, "retry_allowed": False, "experiment": name,
            "fit_attempts": 2, "plan_path": "plan.json", "plan_sha256": "fixture",
            "review_path": "review.json", "review_sha256": "fixture", "operation_limits": {}}
    for filename, data in (("plan.json", plan), ("source.json", {"sources": {}, "backbone": {},
                            "task3_root": "fixture", "external_code": {}}),
                           ("review.json", {"approved": True, "plan_sha256": "fixture",
                                            "implementation_bindings": {}})):
        (tmp_path / filename).write_text(json.dumps(data), encoding="utf-8")
    (tmp_path / "requirements").mkdir()
    pins = [f"{package}=={version(package).split('+')[0]}"
            for package in ("numpy", "scipy", "torch", "filelock", "h5py")]
    (tmp_path / "requirements/constraints-py312.txt").write_text("\n".join(pins), encoding="utf-8")
    monkeypatch.setattr(workflow, "assert_clean", lambda *_: None)
    monkeypatch.setattr(workflow, "verify_bound_file", lambda *_: None)
    monkeypatch.setattr(workflow, "validate_run_policy", lambda *_: None)
    monkeypatch.setattr(workflow, "load_manifest", lambda _: {"experiments": {name: entry}})
    monkeypatch.setattr(workflow, "load_policy", lambda _: {"development_evaluations": {authorization: auth}})
    monkeypatch.setattr(workflow, "load_state", lambda _, workflow_name: workflow_name)
    monkeypatch.setattr(workflow, "approval_recorded", lambda state, stage: (state, stage) == (entry["workflow"], "fit"))
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    actual_name, actual_entry, actual_plan, actual_authorization = workflow.authorize(tmp_path, "fit", name)
    assert (actual_name, actual_entry, actual_authorization) == (name, entry, authorization)
    assert actual_plan["device"] == "cuda:0"
    assert actual_plan["sources"]["allowed_roles"] == ["train", "val"]
    auth["status"] = "consumed"
    with pytest.raises(RuntimeError, match="scope not active"):
        workflow.authorize(tmp_path, "fit", name)
