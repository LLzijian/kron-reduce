from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from kron_reduce.physics.coordinates import coi_project

from .base import DynamicsModel, FitResult, RunContext
from .m0 import M0Model, _trapz
from .m2a import M2AModel


class DynamicM2AModel(DynamicsModel):
    """Validation-gated first-order edge-memory extension of static M2A."""

    method_name = "M2A-dynamic"

    def __init__(self, static_model=None, tau=None, a=None, b=None, adopted=False):
        self.static_model = static_model or M2AModel()
        self.tau = None if tau is None else np.asarray(tau, dtype=np.float64)
        self.a = None if a is None else np.asarray(a, dtype=np.float64)
        self.b = None if b is None else np.asarray(b, dtype=np.float64)
        self.c = None if a is None else np.ones_like(self.a)
        self.adopted = bool(adopted)
        self.diagnostics = {}

    @staticmethod
    def _filter_inputs(t, dsin, edge_speed, tau):
        z1, z2 = np.empty_like(dsin), np.empty_like(edge_speed)
        z1[0], z2[0] = dsin[0], edge_speed[0]
        for k in range(len(t)-1):
            dt=float(t[k+1]-t[k]); decay=np.exp(-dt/tau)
            z1[k+1]=decay*z1[k]+(1-decay)*0.5*(dsin[k]+dsin[k+1])
            z2[k+1]=decay*z2[k]+(1-decay)*0.5*(edge_speed[k]+edge_speed[k+1])
        return z1,z2

    def _weak_memory_system(self, samples, context, tau):
        rows, targets=[] ,[]
        model=self.static_model
        for sample in samples[:context.max_cases]:
            target=sample["training_context"]["targets"]; t=np.asarray(target["t"]); d=np.asarray(target["delta"]); w=np.asarray(target["omega"]); pm=np.asarray(sample["training_context"]["Pm_observed"]); static=sample["model_inputs"]["static"]
            inc=np.asarray(static["incidence"]); mass=np.asarray(static["mass"]); eta=d@inc; eta0=inc.T@np.asarray(static["equilibrium_delta"]); dsin=np.sin(eta)-np.sin(eta0); ew=w@inc
            limit=min(len(t),max(context.horizon_steps*5,max(context.integral_windows)+1)); z1,z2=self._filter_inputs(t[:limit],dsin[:limit],ew[:limit],tau)
            designs=np.stack([np.column_stack((inc*z1[k],inc*z2[k])) for k in range(limit)])
            force=np.stack([model.force(d[k],w[k],pm[k],static) for k in range(limit)])
            for width in context.integral_windows:
                if width<=0 or width>=limit:continue
                for start in range(limit-width):
                    stop=start+width; seg=slice(start,stop+1)
                    rows.append(_trapz(designs[seg],t[seg])); targets.append(mass*(w[stop]-w[start])-_trapz(force[seg],t[seg]))
        return np.concatenate(rows),np.concatenate(targets)

    def fit(self,train,val,run_context):
        if self.static_model.node_theta is None:self.static_model.fit(train,val,run_context)
        base_loss=self.static_model._validation_loss(val,run_context); e=train[0]["model_inputs"]["static"]["incidence"].shape[1]; candidates=[]
        for tau in (0.02,0.05,0.1,0.2,0.5):
            matrix,vector=self._weak_memory_system(train,run_context,tau); scale=np.sqrt(np.mean(matrix**2,axis=0)); scale=np.where(scale>1e-14,scale,1.0); x=matrix/scale
            beta=np.linalg.lstsq(np.vstack((x,1e-3*np.eye(x.shape[1]))),np.concatenate((vector,np.zeros(x.shape[1]))),rcond=None)[0]; theta=beta/scale
            self.tau=np.full(e,tau); self.a=theta[:e]; self.b=theta[e:]; self.c=np.ones(e)
            try:loss=self._validation_loss(val,run_context)
            except (ValueError,FloatingPointError):loss=float("inf")
            candidates.append((loss,tau,theta.copy(),float(np.sqrt(np.mean((matrix@theta-vector)**2)))))
        loss,tau,theta,residual=min(candidates,key=lambda item:item[0]); improvement=(base_loss-loss)/max(base_loss,np.finfo(float).eps); self.adopted=bool(np.isfinite(loss) and improvement>=run_context.refinement_min_improvement)
        self.tau=np.full(e,tau); self.a=theta[:e] if self.adopted else np.zeros(e); self.b=theta[e:] if self.adopted else np.zeros(e); self.c=np.ones(e)
        final_loss=loss if self.adopted else base_loss; self.diagnostics={"objective":"single_integral_weak_memory_solve","epochs":0,"selected_tau":tau,"weak_residual_rmse":residual,"static_validation_loss":base_loss,"dynamic_validation_loss":loss,"relative_improvement":improvement,"adopted":self.adopted,"z0_policy":"clearing_quasi_steady"}
        return FitResult(self.method_name,"completed",tuple(s["event_id"] for s in train[:run_context.max_cases]),tuple(s["event_id"] for s in val[:run_context.max_cases]),float(final_loss),self.diagnostics)

    def _validation_loss(self,val,context):
        losses=[]
        for sample in val[:context.max_cases]:
            target=sample["training_context"]["targets"]; steps=min(len(target["t"]),context.horizon_steps); d,w=self.rollout(sample["model_inputs"]["static"],target["t"][:steps],{"Pm":sample["model_inputs"]["exogenous"]["Pm"][:steps]}); inc=np.asarray(sample["model_inputs"]["static"]["incidence"])
            losses.append(np.mean((((d-target["delta"][:steps])@inc)/self.static_model.base.validation_scales["edge_angle"])**2)+np.mean(((w-target["omega"][:steps])/self.static_model.base.validation_scales["omega"])**2))
        return float(np.mean(losses))

    def rollout(self,initial_state,time_grid,exogenous):
        if self.a is None:raise RuntimeError("dynamic M2A is not fitted")
        if not self.adopted:return self.static_model.rollout(initial_state,time_grid,exogenous)
        t=np.asarray(time_grid); pm=np.asarray(exogenous["Pm"]); mass=np.asarray(initial_state["mass"]); inc=np.asarray(initial_state["incidence"]); eq=np.asarray(initial_state["equilibrium_delta"]); d=coi_project(np.asarray(initial_state["delta0"]).reshape(-1),mass); w=coi_project(np.asarray(initial_state["omega0"]).reshape(-1),mass); dsin=np.sin(inc.T@d)-np.sin(inc.T@eq); ew=inc.T@w; z=self.a*dsin+self.b*ew; od,ow=[d.copy()],[w.copy()]
        def field(di,wi,zi,p):
            ds=np.sin(inc.T@di)-np.sin(inc.T@eq); es=inc.T@wi; force=self.static_model.force(di,wi,p,initial_state)+inc@(self.c*zi); return wi,force/mass,(-zi+self.a*ds+self.b*es)/self.tau
        for k in range(len(t)-1):
            interval=float(t[k+1]-t[k]); sub=max(1,int(np.ceil(interval/self.static_model.base.max_internal_step))); h=interval/sub
            for j in range(sub):
                q0=j/sub; qh=(j+.5)/sub; q1=(j+1)/sub; p=lambda q: pm[k] if exogenous.get("interpolation","auto")=="zoh" or np.allclose(pm,pm[[0]]) else pm[k]+q*(pm[k+1]-pm[k]); k1=field(d,w,z,p(q0)); k2=field(d+.5*h*k1[0],w+.5*h*k1[1],z+.5*h*k1[2],p(qh)); k3=field(d+.5*h*k2[0],w+.5*h*k2[1],z+.5*h*k2[2],p(qh)); k4=field(d+h*k3[0],w+h*k3[1],z+h*k3[2],p(q1)); d=coi_project(d+h*(k1[0]+2*k2[0]+2*k3[0]+k4[0])/6,mass); w=coi_project(w+h*(k1[1]+2*k2[1]+2*k3[1]+k4[1])/6,mass); z=z+h*(k1[2]+2*k2[2]+2*k3[2]+k4[2])/6
            if not np.isfinite(d).all() or not np.isfinite(w).all():raise FloatingPointError(f"nonfinite dynamic M2A rollout at {k+1}")
            od.append(d.copy());ow.append(w.copy())
        return np.stack(od),np.stack(ow)

    def save(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,damping=self.static_model.base.damping,coupling=self.static_model.base.coupling,node_theta=self.static_model.node_theta,edge_theta=self.static_model.edge_theta,tau=self.tau,a=self.a,b=self.b,adopted=self.adopted);path.with_suffix(".json").write_text(json.dumps({"method":self.method_name,"diagnostics":self.diagnostics},indent=2),encoding="utf-8")

    @classmethod
    def load(cls,path):
        with np.load(path) as data:
            static=M2AModel(M0Model(data["damping"],data["coupling"]),data["node_theta"],data["edge_theta"]);return cls(static,data["tau"],data["a"],data["b"],bool(data["adopted"]))

    def physics_audit(self):return {"status":"pass" if self.a is not None else "not_fitted","causal_z0":True,"future_state_initialization":False,"adopted":self.adopted,"internal_force_balance_by_construction":True}
