import numpy as np
import pytest
import torch

from kron_reduce.evaluation.first_swing_probes import speed_peak_gap, zero_history_reference


class ConstantHistory:
    n_nodes = 10
    latent_dimension = 10

    @staticmethod
    def _tensor(value):
        return torch.as_tensor(value, dtype=torch.float64)

    @staticmethod
    def rollout(initial, power, equilibrium, count, *, max_step, spend):
        assert count == 10 and max_step == .0025
        spend("rollouts")
        return initial[:, None].expand(-1, count+1, -1)


def event(event_id, amplitude, frames=11):
    x = np.zeros((frames, 20))
    x[1:, 10:12] = amplitude, -amplitude
    return {"id": event_id, "x": x, "role": "train", "p0": np.zeros(10),
            "equilibrium": np.zeros(10), "mass": np.ones(10)}


def test_reference_uses_ten_residual_frames_and_full_train_median():
    events = [event("a", 0), event("b", 2), event("c", 4)]
    spent = []
    actual = zero_history_reference(ConstantHistory(), events, np.ones(20), spent.append,
                                    batch_size=2, floor=.001)
    assert actual["reference"] == pytest.approx(2/np.sqrt(10))
    assert actual["zero_history_nrmse"]["c"] == pytest.approx(4/np.sqrt(10))
    assert spent == ["rollouts", "rollouts"]
    with pytest.raises(ValueError, match="train roster"):
        zero_history_reference(ConstantHistory(), [events[0] | {"role": "val"}], np.ones(20),
                               spent.append, batch_size=2, floor=.001)


def test_speed_peak_gap_uses_shared_post_origin_horizon_and_coi():
    truth = event("a", 2, frames=65)
    truth["x"][61:, 10:12] = [10., -10.]
    predicted = np.zeros((801, 20))
    predicted[:, 10:] = 7.
    predicted[1:51, 10:12] = [8.5, 5.5]
    predicted[51:, 10] = 500.
    result = speed_peak_gap(truth, predicted)
    assert result == {"points": 50, "true_peak_rad_s": 2., "predicted_peak_rad_s": 1.5, "gap_rad_s": .5}
    assert speed_peak_gap(truth, predicted[:2]) is None
