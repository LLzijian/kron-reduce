import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from kron_reduce.evaluation import kiki_machine_local_full as full


def _event(event_id, length=15):
    x = np.arange(length * 20, dtype=float).reshape(length, 20)
    return {
        "id": event_id,
        "x": x,
        "raw_x": x,
        "p0": np.zeros(10),
        "equilibrium": np.zeros(10),
        "mass": np.ones(10),
        "label": "stable",
    }


def test_evaluate_scores_full_roster_after_batch_failure_and_keeps_invalids(
    tmp_path, monkeypatch,
):
    events = [_event("a"), _event("b"), _event("c")]
    predict_calls = []

    def predict(_model, delta, omega, p0, equilibrium, scale, count, *, max_step, spend):
        predict_calls.append((len(delta), max_step))
        if len(delta) > 1 or np.all(delta[0] == events[1]["x"][:11, :10]):
            raise FloatingPointError("mock numerical failure")
        batch = len(delta)
        visible = torch.zeros((batch, 801, 20), dtype=torch.float64)
        return {
            "forecast": torch.cat((visible, torch.zeros((batch, 801, 10))), dim=2),
            "hidden_start": torch.zeros((batch, 10)),
            "history": torch.zeros((batch, 11, 30)),
            "gn_singular": torch.ones((1, batch, 1)),
        }

    seen = {}

    def score(predictions, scoring_events, comparison):
        seen["ids"] = [event["id"] for event in scoring_events]
        seen["invalid_b"] = bool(np.isnan(predictions[1]).any())
        assert len(comparison) == len(events)
        return {
            "summary": {},
            "classifications": [
                {"predicted_los": False, "window_settled": True, "qualified": i != 1}
                for i in range(len(scoring_events))
            ],
            "selection_score": None,
        }

    task3 = ModuleType("task3")
    task3.__path__ = []
    c2 = ModuleType("task3.c2")
    c2.__path__ = []
    protocol = ModuleType("task3.protocol")
    protocol.classify = lambda *_args: {
        "predicted_los": False, "window_settled": True,
    }
    evaluator = ModuleType("task3.c2.evaluation")
    evaluator.evaluate = score
    monkeypatch.setitem(sys.modules, "task3", task3)
    monkeypatch.setitem(sys.modules, "task3.c2", c2)
    monkeypatch.setitem(sys.modules, "task3.protocol", protocol)
    monkeypatch.setitem(sys.modules, "task3.c2.evaluation", evaluator)
    monkeypatch.setattr(full, "predict_from_history", predict)
    monkeypatch.setattr(full, "write_json", lambda *_args: None)
    monkeypatch.setattr(full.np, "savez_compressed", lambda *_args, **_kwargs: None)

    result = full.evaluate(object(), events, np.ones(20), SimpleNamespace(spend=lambda *_a, **_k: None),
                           tmp_path, "train", {"eval_batch": 3})

    assert seen == {"ids": ["a", "b", "c"], "invalid_b": True}
    assert len(predict_calls) == 8  # failed batch and per-event retries at both step sizes
    assert result["task3_six_term_score"] is None
    assert result["stability"]["numerical_failures"] == 1
    assert result["stability"]["accuracy"] is None
    assert sum(failure.get("id") == "b" for failure in result["failures"]) == 2
    assert all("error" in failure for failure in result["failures"])


def test_train_uses_actual_prefix_counts_edge_padding_and_full_dataset_weights(
    tmp_path, monkeypatch,
):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.port_parameter = torch.nn.Parameter(torch.tensor(0.1, dtype=torch.float64))
            self.rate_parameter = torch.nn.Parameter(torch.tensor(0.2, dtype=torch.float64))

    model = Model()
    events = [_event("a", 15), _event("b", 13), _event("c", 14), _event("d", 12)]
    calls = []

    def predict(_model, delta, omega, p0, equilibrium, scale, count, *, max_step, spend):
        batch, visible = len(delta), 20
        value = (_model.port_parameter + _model.rate_parameter).expand(batch, count + 1, visible)
        return {"forecast": value}

    original_loss = full.loss

    def record_loss(path, target, weights, scale, counts):
        calls.append((np.asarray(target).copy(), np.asarray(counts).copy(),
                      np.asarray(weights).copy()))
        return original_loss(path, target, weights, scale, counts)

    monkeypatch.setattr(full, "predict_from_history", predict)
    monkeypatch.setattr(full, "loss", record_loss)
    monkeypatch.setattr(full, "snapshot", lambda module: {k: v.detach().clone() for k, v in module.state_dict().items()})
    monkeypatch.setattr(full, "write_json", lambda *_args: None)
    monkeypatch.setattr(full, "NUMERICAL_ERRORS", (FloatingPointError, torch.linalg.LinAlgError))

    class Budget:
        def spend(self, *_args, **_kwargs):
            pass

    (tmp_path / "training.jsonl").touch()
    (tmp_path / "training.jsonl").unlink()
    result = full.train(model, events, {"history": np.ones(20), "target": np.ones(20)},
                        Budget(), tmp_path, {"updates": 1, "train_batch": 2})

    assert result["completed_updates"] == 1
    assert [counts.tolist() for _target, counts, _weights in calls] == [[4, 2], [3, 1]]
    for target, _counts, weights in calls:
        np.testing.assert_allclose(weights, np.full(2, 1 / len(events)))
        assert target.shape[1] == max(_counts) + 1
    assert calls[0][0][1, 3, 0] == events[1]["x"][12, 0]
    assert calls[1][0][1, 2, 0] == events[3]["x"][11, 0]


def test_full_dataset_loss_gradient_matches_weighted_microbatches():
    counts = np.array([4, 2, 3, 1])
    target = np.stack([
        np.pad(np.full((count + 1, 20), i + 1., dtype=float),
               ((0, counts.max() - count), (0, 0)), mode="edge")
        for i, count in enumerate(counts)
    ])
    parameter = torch.nn.Parameter(torch.tensor([.1, .2, .3, .4], dtype=torch.float64))
    weights = np.full(len(counts), 1 / len(counts))
    path = parameter[:, None, None].expand(len(counts), counts.max() + 1, 20)
    full_objective = full.loss(path, target, weights, np.ones(20), counts)
    full_gradient, = torch.autograd.grad(full_objective, parameter, retain_graph=True)

    microbatch_objective = sum(
        full.loss(
            parameter[start:end, None, None].expand(end - start, int(counts[start:end].max()) + 1, 20),
            target[start:end, :int(counts[start:end].max()) + 1],
            weights[start:end], np.ones(20), counts[start:end],
        )
        for start, end in ((0, 2), (2, 4))
    )
    microbatch_gradient, = torch.autograd.grad(microbatch_objective, parameter)

    torch.testing.assert_close(microbatch_gradient, full_gradient)


def test_predict_checked_only_translates_the_frozen_nonfinite_input_error(monkeypatch):
    def fail(message):
        raise ValueError(message)

    monkeypatch.setattr(full, "predict_from_history", lambda *_a, **_k: fail("model inputs must be finite"))
    with pytest.raises(FloatingPointError, match="nonfinite intermediate RK4 state"):
        full.predict_checked(object())

    monkeypatch.setattr(full, "predict_from_history", lambda *_a, **_k: fail("history shape differs"))
    with pytest.raises(ValueError, match="history shape differs"):
        full.predict_checked(object())
