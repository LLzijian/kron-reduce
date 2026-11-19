import pytest
import torch

from kron_reduce.evaluation.machine_local_topology import make_model
from kron_reduce.evaluation.machine_local_topology_truth import parameters
from kron_reduce.models.first_swing_gn_grid import predict_gn_grid
from kron_reduce.models.machine_local_topology import predict_from_topology_history


def _unmetered(*_):
    pass


@pytest.fixture
def causal_batch():
    model = make_model(parameters(), "T0M1", "cpu")
    n = model.n_nodes
    state = model.mass.new_zeros((3, 3*n))
    state[1:, 2*n] = state.new_tensor([.35, .8])
    state[1:, 2*n+1] = -state[1:, 2*n]
    power = state.new_zeros((3, n))
    power[:, :2] = state.new_tensor([.05, -.05])
    equilibrium = torch.zeros_like(power)
    scale = state.new_full((2*n,), .01)
    observed = model.rollout(state, power, equilibrium, 10, max_step=.0025, spend=_unmetered)
    zero = model.rollout(torch.zeros_like(state), power, equilibrium, 10, max_step=.0025, spend=_unmetered)
    primary_r0 = ((observed[:, 1:, :2*n]-zero[:, 1:, :2*n])/scale).square().mean((1, 2)).sqrt()
    return (model, observed[:, :, :n], observed[:, :, n:2*n], power, equilibrium, scale), primary_r0


def _modes(reference):
    return [{"name": name, "prior": prior, "radius_multiplier": multiplier, "reference": reference}
            for name, prior, multiplier in (("fixed", "fixed", 1.), ("quadratic", "quadratic", 2.),
                                            ("hinge", "hinge", 1.6))]


@torch.no_grad()
def test_grid_reproduces_legacy_controls_and_hinge_boundary(causal_batch):
    args, primary = causal_batch
    model, q, v, *_ = args
    n, r = model.n_nodes, model.memory_dimension
    settings = {"max_step": .0025, "spend": _unmetered, "memory_radius": .1, "prior_weight": 1e-6}
    reference = primary[1].item()
    assert primary[0] < reference < primary[2]
    grid = predict_gn_grid(*args, **settings, modes=_modes(reference), primary_r0=primary, count=2)
    old_fixed = predict_from_topology_history(*args, 2, **settings)
    old_soft = predict_from_topology_history(*args, 2, **settings, adaptive_prior_reference=reference)

    assert grid["forecast"].shape == (3, 3, 3, 2*n)
    assert grid["history"].shape == (3, 3, 11, 2*n+r)
    assert grid["hidden_start"].shape == grid["singular"].shape == (3, 3, r)
    for i, old in enumerate((old_fixed, old_soft)):
        torch.testing.assert_close(grid["forecast"][i], old["forecast"][..., :2*n], atol=1e-11, rtol=1e-10)
        for key in ("history", "hidden_start", "singular"):
            torch.testing.assert_close(grid[key][i], old[key], atol=1e-11, rtol=1e-10)
    torch.testing.assert_close(grid["zero_history_nrmse"], primary, atol=1e-12, rtol=0.)
    torch.testing.assert_close(grid["prior_weight"][1], old_soft["adaptive_prior_weight"], atol=1e-13, rtol=1e-10)

    for key in ("forecast", "history", "hidden_start", "singular"):
        torch.testing.assert_close(grid[key][2, :2], grid[key][0, :2], atol=1e-11, rtol=0.)
    assert grid["soft_active"].tolist() == [[False]*3, [True]*3, [False, False, True]]
    limits = primary.new_tensor([[.1]*3, [.2]*3, [.1, .1, .16]])
    torch.testing.assert_close(grid["radius_limit"], limits)
    norms = grid["hidden_start"].norm(dim=-1)
    assert torch.all(norms <= limits+1e-13)
    assert .1 < norms[2, 2] < .16
    hinge = (primary/reference-1.).clamp_min(0.)
    torch.testing.assert_close(grid["prior_weight"][2], 1e-6/(1.+hinge.square()), atol=1e-13, rtol=1e-10)
    origin = torch.cat((q[:, -1], v[:, -1]), dim=-1)
    torch.testing.assert_close(grid["forecast"][:, :, 0], origin.expand(3, -1, -1), atol=0., rtol=0.)


@torch.no_grad()
def test_refined_hinge_keeps_pinned_primary_gate_and_rejects_future_frames(causal_batch):
    args, primary = causal_batch
    model, q, v, power, equilibrium, scale = args
    reference = primary[1].item()
    # An intentionally reversed gate distinguishes the pinned input from local-step r0.
    pinned = primary.new_tensor([3.*reference, reference, 0.])
    modes = _modes(reference)
    modes = [modes[0], modes[2]]
    settings = {"max_step": .00125, "spend": _unmetered, "memory_radius": .1, "prior_weight": 1e-6,
                "modes": modes, "primary_r0": pinned, "count": 0}
    grid = predict_gn_grid(*args, **settings)
    assert grid["zero_history_nrmse"][0] < reference < grid["zero_history_nrmse"][2]
    assert grid["soft_active"][1].tolist() == [True, False, False]
    torch.testing.assert_close(grid["prior_weight"][1], primary.new_tensor([.2e-6, 1e-6, 1e-6]), atol=1e-13, rtol=1e-10)
    torch.testing.assert_close(grid["hidden_start"][1, 1:], grid["hidden_start"][0, 1:], atol=1e-11, rtol=0.)
    assert grid["forecast"].shape[2] == 1

    future_q, future_v = (torch.cat((history, history[:, -1:]), dim=1) for history in (q, v))
    with pytest.raises(ValueError):
        predict_gn_grid(model, future_q, future_v, power, equilibrium, scale, **settings)
    with pytest.raises(ValueError):
        predict_gn_grid(*args, **(settings | {"primary_r0": None}))
def test_registered_gate_requires_both_steps_even_with_eleven_detections():
    from kron_reduce.evaluation.first_swing_gn_decomposition import _gate

    candidate = "candidate"
    stability = {"numerical_failures": 0, "uncovered": 0, "tp": 11, "fp": 1}
    scores = {role: {candidate: {"task3_six_term_score": .22, "stability": dict(stability),
                   "stable_complete": {"8": {"qualified": count, "mean": .24}}}}
              for role, count in (("train", 311), ("val", 95))}
    witness = {dt: {"summary": {"fixed": {"signed_median": 2.5, "absolute_median": 3.},
               candidate: {"signed_median": -2., "absolute_median": 2.5, "qualified": 33}}}
               for dt in ("0.0025", "0.00125")}
    peaks = {"summary": {dt: {"fixed": {"positive_median": 2.},
                         candidate: {"positive_median": 1.6, "qualified": 12}}
                         for dt in witness}}
    settings = {"relative_limit": .8, "stable_8s_mean_max": .24, "false_alarms_max": 1}
    assert _gate(scores, witness, peaks, settings, candidate)["pass"]
    peaks["summary"]["0.00125"][candidate]["positive_median"] = 1.61
    assert not _gate(scores, witness, peaks, settings, candidate)["criteria"]["recall_or_peak"]
    scores["val"][candidate]["stability"]["tp"] = 12
    assert _gate(scores, witness, peaks, settings, candidate)["pass"]
    witness["0.00125"]["summary"][candidate]["absolute_median"] = 3.01
    assert not _gate(scores, witness, peaks, settings, candidate)["criteria"]["paired_witnesses"]
