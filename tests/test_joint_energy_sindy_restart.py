import json
from types import SimpleNamespace

import numpy as np
import torch

from kron_reduce.evaluation import joint_energy_sindy_synthetic as synthetic


def test_fit_saves_model_and_adam_state_at_validation_checkpoints(monkeypatch, tmp_path):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.))

        def group_norms(self):
            return {}

    def objective(model, *args, training):
        with torch.set_grad_enabled(training):
            return model.weight.square()

    monkeypatch.setattr(synthetic, "trajectory_objective", objective)
    settings = {"updates": 3, "learning_rate": .1, "train_frames": 1, "val_frames": 1,
                "val_interval": 2, "save_checkpoints": True}
    model = Model()
    budget = SimpleNamespace(spend=lambda _: None)
    fit = synthetic._fit(model, None, np.array(["train", "val"]), None, None, None, None,
                         settings, budget, directory=tmp_path)

    assert fit["selected_update"] == 3
    checkpoints = tmp_path / "checkpoints"
    assert sorted(path.name for path in checkpoints.glob("*.pt")) == [
        "update-000.pt", "update-002.pt", "update-003.pt",
    ]

    initial = torch.load(checkpoints / "update-000.pt", weights_only=True)
    torch.testing.assert_close(initial["model"]["weight"], torch.tensor(1.))
    assert initial["update"] == 0 and not initial["optimizer"]["state"]

    restored = Model()
    optimizer = torch.optim.Adam(restored.parameters())
    for update in (2, 3):
        checkpoint = torch.load(checkpoints / f"update-{update:03d}.pt", weights_only=True)
        assert checkpoint["update"] == update
        restored.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        assert 0. < restored.weight.item() < 1.
        state = optimizer.state[restored.weight]
        assert state["step"].item() == update
        assert state["exp_avg"].item() > 0.
    torch.testing.assert_close(restored.weight, model.weight)


def test_fit_resume_matches_uninterrupted_fit_from_checkpoint(monkeypatch, tmp_path):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.))

        def group_norms(self):
            return {}

    def objective(model, observed, rows, *args, training):
        with torch.set_grad_enabled(training):
            return model.weight.square() * (1. + float(len(rows)))

    monkeypatch.setattr(synthetic, "trajectory_objective", objective)
    settings = {"updates": 4, "learning_rate": .1, "train_frames": 1, "val_frames": 1,
                "val_interval": 2, "save_checkpoints": True, "train_batch": 1}
    roles = np.array(["train", "train", "val"])
    budget = SimpleNamespace(spend=lambda _: None)
    uninterrupted_dir = tmp_path / "uninterrupted"
    uninterrupted_dir.mkdir()
    uninterrupted_model = Model()
    uninterrupted = synthetic._fit(uninterrupted_model, None, roles, None, None, None, None,
                                   settings, budget, directory=uninterrupted_dir)

    source = tmp_path / "interrupted"
    source.mkdir()
    step = torch.optim.Adam.step
    calls = 0

    def interrupt_after_fourth_update(optimizer, *args, **kwargs):
        nonlocal calls
        calls += 1
        result = step(optimizer, *args, **kwargs)
        if calls == 4:
            raise RuntimeError("simulated interruption after update 4")
        return result

    monkeypatch.setattr(torch.optim.Adam, "step", interrupt_after_fourth_update)
    with np.testing.assert_raises_regex(RuntimeError, "simulated interruption"):
        synthetic._fit(Model(), None, roles, None, None, None, None, settings, budget, directory=source)
    monkeypatch.setattr(torch.optim.Adam, "step", step)

    resumed_dir = tmp_path / "resumed"
    resumed_dir.mkdir()
    resumed_model = Model()
    resumed = synthetic._fit(resumed_model, None, roles, None, None, None, None, settings, budget,
                             directory=resumed_dir, resume=(source, 2))

    assert resumed["actual_updates"] == uninterrupted["actual_updates"] == 4
    assert resumed["selected_update"] == uninterrupted["selected_update"]
    assert resumed["train_curve"] == uninterrupted["train_curve"]
    assert resumed["val_curve"] == uninterrupted["val_curve"]
    torch.testing.assert_close(resumed_model.weight, uninterrupted_model.weight)
    resumed_checkpoint = torch.load(resumed_dir / "checkpoints" / "update-004.pt", weights_only=True)
    full_checkpoint = torch.load(uninterrupted_dir / "checkpoints" / "update-004.pt", weights_only=True)
    torch.testing.assert_close(resumed_checkpoint["model"]["weight"], full_checkpoint["model"]["weight"])
    for key in ("step", "exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(resumed_checkpoint["optimizer"]["state"][0][key],
                                   full_checkpoint["optimizer"]["state"][0][key])

    train_rows = [json.loads(line) for line in (resumed_dir / "training.jsonl").read_text().splitlines()]
    full_rows = [json.loads(line) for line in (uninterrupted_dir / "training.jsonl").read_text().splitlines()]
    assert train_rows == full_rows
    assert [row["actual_updates"] for row in train_rows] == [0, 1, 2, 2, 3, 4, 4]
