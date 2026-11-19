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
