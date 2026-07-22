from __future__ import annotations

import numpy as np

from kron_reduce.data import ModelDataRequest, adapt_event, load_split
from kron_reduce.models import (
    AutoregressiveLSTMModel,
    EnergyM2AModel,
    M0Model,
    M2AModel,
    PINNModel,
    RunContext,
    SINDYcModel,
)


def _samples(store, dataset_dir):
    split=load_split(dataset_dir,view="combined_dynamics",split_name="combined_line_5fold_v1",fold="F1")
    train_ids=[eid for eid in split.train_ids if store.metadata(eid).valid_length>=10][:1]
    val_ids=[eid for eid in split.val_ids if store.metadata(eid).valid_length>=10][:1]
    req=ModelDataRequest.initial_state_rollout("zoh"); inertia=store.dataset_metadata.M_prior
    build=lambda ids:[adapt_event(store.load_event(eid),inertia_prior=inertia,request=req) for eid in ids]
    return build(train_ids),build(val_ids)


def test_main_six_models_real_fit_rollout_save_load(store,dataset_dir,tmp_path):
    train,val=_samples(store,dataset_dir); context=RunContext(epochs=1,horizon_steps=8,max_cases=1)
    m0=M0Model(); result=m0.fit(train,val,context); assert result.status=="completed"
    models = [
        ("M0", m0, result, M0Model, "npz"),
        ("M2A", M2AModel(M0Model(m0.damping, m0.coupling)), None, M2AModel, "npz"),
        ("M2A-energy", EnergyM2AModel(M0Model(m0.damping, m0.coupling)), None, EnergyM2AModel, "npz"),
        ("SINDYc", SINDYcModel(), None, SINDYcModel, "npz"),
        ("PINN", PINNModel(M0Model(m0.damping, m0.coupling)), None, PINNModel, "pt"),
        ("LSTM", AutoregressiveLSTMModel(), None, AutoregressiveLSTMModel, "pt"),
    ]
    target=val[0]["training_context"]["targets"]
    for name, model, fit, loader, extension in models:
        fit = fit or model.fit(train, val, context)
        assert fit.status == "completed"
        path = tmp_path / f"{name}.{extension}"
        model.save(path)
        loaded = loader.load(path)
        dp,wp=loaded.rollout(val[0]["model_inputs"]["static"],target["t"][:8],{"Pm":val[0]["model_inputs"]["exogenous"]["Pm"][:8]})
        assert dp.shape==wp.shape==(8,7)
        assert np.isfinite(dp).all() and np.isfinite(wp).all()
