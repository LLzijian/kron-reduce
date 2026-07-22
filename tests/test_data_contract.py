from __future__ import annotations

import pickle

import numpy as np
import pytest

from kron_reduce.data import ModelDataRequest, adapt_event, load_split
from kron_reduce.data.audit import audit_model_sample
from kron_reduce.data.contracts import IneligibleEventError, ShortSequenceError


def test_store_views_and_lazy_pickle(store):
    assert len(store.event_ids()) == 315
    assert len(store.event_ids("general_dynamics")) == 100
    assert len(store.event_ids("binary_stability")) == 215
    restored = pickle.loads(pickle.dumps(store))
    assert restored.event_ids() == store.event_ids()


def test_exact_clearing_and_variable_length(store):
    event = store.load_event(store.event_ids()[0])
    assert event.t[0] == 0.0
    assert np.all(np.diff(event.t) > 0)
    assert event.delta.shape == (len(event.t), 10)
    lengths = {store.metadata(eid).valid_length for eid in store.event_ids()}
    assert min(lengths) == 2
    assert max(lengths) > 1000


def test_split_is_disjoint_and_bound_to_view(dataset_dir):
    split = load_split(dataset_dir, view="combined_dynamics", split_name="combined_line_5fold_v1", fold="F1")
    assert not set(split.train_ids) & set(split.val_ids)
    assert not set(split.train_ids) & set(split.test_ids)
    with pytest.raises(ValueError):
        load_split(dataset_dir, view="binary_stability", split_name="combined_line_5fold_v1", fold="F1")


def test_zoh_and_observed_pm_are_separate(store):
    eid = next(e for e in store.event_ids() if store.metadata(e).valid_length >= 20)
    event = store.load_event(eid); inertia = store.dataset_metadata.M_prior
    zoh = adapt_event(event, inertia_prior=inertia, request=ModelDataRequest.initial_state_rollout("zoh"))
    observed = adapt_event(event, inertia_prior=inertia, request=ModelDataRequest.initial_state_rollout("observed"))
    assert np.all(zoh["model_inputs"]["exogenous"]["Pm"] == zoh["model_inputs"]["exogenous"]["Pm"][0])
    assert np.array_equal(observed["model_inputs"]["exogenous"]["Pm"], observed["training_context"]["Pm_observed"])
    assert zoh["training_context"]["claim_type"] == "causal_zoh_rollout"
    audit_model_sample(zoh)


def test_clustered_shape_and_prefix_rules(store):
    inertia = store.dataset_metadata.M_prior
    long_id = next(e for e in store.event_ids() if store.metadata(e).valid_length >= 20)
    sample = adapt_event(store.load_event(long_id), inertia_prior=inertia, request=ModelDataRequest.prefix_steps(20), representation="clustered_7")
    assert sample["model_inputs"]["observed_context"]["delta"].shape == (20, 7)
    short_id = next(e for e in store.event_ids() if store.metadata(e).valid_length == 2)
    with pytest.raises(ShortSequenceError):
        adapt_event(store.load_event(short_id), inertia_prior=inertia, request=ModelDataRequest.prefix_steps(20))


def test_event_inside_prefix_is_ineligible(store):
    inertia = store.dataset_metadata.M_prior
    candidates = [eid for eid in store.event_ids() if store.metadata(eid).event_observed and store.metadata(eid).valid_length >= 2]
    event = store.load_event(candidates[0])
    request = ModelDataRequest.prefix_steps(2, prospective=True)
    if event.metadata.event_time_s <= event.t[1]:
        with pytest.raises(IneligibleEventError):
            adapt_event(event, inertia_prior=inertia, request=request)
