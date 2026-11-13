import math

import torch

from kron_reduce.evaluation import kiki_machine_local_r20_r21 as paired


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
