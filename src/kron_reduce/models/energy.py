from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

from kron_reduce.physics.coordinates import coi_project, coi_project_power

from .base import DynamicsModel, FitResult, RunContext
from .m0 import M0Model, _pm_equilibrium, _trapz


class EnergyM2AModel(DynamicsModel):
    """Port-Hamiltonian edge-memory model with an auditable supply identity."""

    method_name = "M2A-energy"

    def __init__(
        self, base=None, node_damping=None, edge_damping=None, kappa=None,
        decay=None, memory_coupling=None, reported_name=None,
    ):
        self.base = base or M0Model()
        self.node_damping = None if node_damping is None else np.asarray(node_damping, dtype=np.float64)
        self.edge_damping = None if edge_damping is None else np.asarray(edge_damping, dtype=np.float64)
        self.kappa = None if kappa is None else np.asarray(kappa, dtype=np.float64)
        self.decay = None if decay is None else np.asarray(decay, dtype=np.float64)
        self.memory_coupling = None if memory_coupling is None else np.asarray(memory_coupling, dtype=np.float64)
        self.reported_name = reported_name or "M2A-energy-regularized"
        self.diagnostics = {}

    def fit(self, train, val, run_context):
        if self.base.damping is None:
            self.base.fit(train, val, run_context)
        self.base.max_internal_step = run_context.max_internal_step
        n = train[0]["model_inputs"]["static"]["mass"].size
        e = train[0]["model_inputs"]["static"]["incidence"].shape[1]
        floor = 1e-10
        self.node_damping = np.maximum(self.base.damping.copy(), floor)
        self.kappa = np.maximum(self.base.coupling.copy(), floor)
        self.decay = np.ones(e, dtype=np.float64)
        self.memory_coupling = np.zeros(e, dtype=np.float64)
        rows, targets = [], []
        for sample in train[:run_context.max_cases]:
            target=sample["training_context"]["targets"]; t=np.asarray(target["t"]); d=np.asarray(target["delta"]); w=np.asarray(target["omega"]); pm=np.asarray(sample["training_context"]["Pm_observed"]); static=sample["model_inputs"]["static"]; mass=np.asarray(static["mass"]); inc=np.asarray(static["incidence"]); eq=np.asarray(static["equilibrium_delta"]); eta0=inc.T@eq
            limit=min(len(t),max(run_context.horizon_steps*5,max(run_context.integral_windows)+1))
            conservative=[]; designs=[]
            for k in range(limit):
                dsin=np.sin(inc.T@d[k])-np.sin(eta0); ew=inc.T@w[k]; dpm=pm[k]-_pm_equilibrium(static)
                conservative.append(coi_project_power(dpm-self.node_damping*w[k]-inc@(self.kappa*dsin),mass))
                designs.append(-inc*ew)
            conservative=np.asarray(conservative); designs=np.asarray(designs)
            for width in run_context.integral_windows:
                if width<=0 or width>=limit:continue
                for start in range(limit-width):
                    stop=start+width; seg=slice(start,stop+1); rows.append(_trapz(designs[seg],t[seg])); targets.append(mass*(w[stop]-w[start])-_trapz(conservative[seg],t[seg]))
        matrix,vector=np.concatenate(rows),np.concatenate(targets); scale=np.sqrt(np.mean(matrix**2,axis=0));scale=np.where(scale>1e-14,scale,1.0);x=matrix/scale
        result=lsq_linear(np.vstack((x,1e-3*np.eye(e))),np.concatenate((vector,np.zeros(e))),bounds=(floor,np.inf),max_iter=2000);self.edge_damping=result.x/scale
        losses=[]
        for sample in val[:run_context.max_cases]:
            target=sample["training_context"]["targets"];steps=min(len(target["t"]),run_context.horizon_steps);dp,wp=self.rollout(sample["model_inputs"]["static"],target["t"][:steps],{"Pm":sample["model_inputs"]["exogenous"]["Pm"][:steps],"interpolation":"zoh"});inc=np.asarray(sample["model_inputs"]["static"]["incidence"])
            losses.append(np.mean((((dp-target["delta"][:steps])@inc)/self.base.validation_scales["edge_angle"])**2)+np.mean(((wp-target["omega"][:steps])/self.base.validation_scales["omega"])**2))
        loss=float(np.mean(losses));self.diagnostics={"objective":"single_structured_integral_weak_solve","epochs":0,"solver_iterations":int(result.nit),"weak_residual_rmse":float(np.sqrt(np.mean((matrix@self.edge_damping-vector)**2))),"port_hamiltonian_memory_state":True,"trajectory_refinement_adopted":False}
        return FitResult(self.method_name,"completed",tuple(s["event_id"] for s in train[:run_context.max_cases]),tuple(s["event_id"] for s in val[:run_context.max_cases]),loss,{"reported_name":self.reported_name,**self.diagnostics})

    def initial_memory(self, delta, omega, static):
        edge_speed=np.asarray(static["incidence"]).T@omega
        return -self.memory_coupling*edge_speed/self.decay

    def vector_field_augmented(self, delta, omega, z, pm, static):
        if self.node_damping is None:raise RuntimeError("energy model is not fitted")
        inc=np.asarray(static["incidence"]);mass=np.asarray(static["mass"]);eta=inc.T@delta;eta0=inc.T@np.asarray(static["equilibrium_delta"]);edge_speed=inc.T@omega;dpm=np.asarray(pm).reshape(-1)-_pm_equilibrium(static)
        force=dpm-inc@(self.kappa*(np.sin(eta)-np.sin(eta0)))-self.node_damping*omega-inc@(self.edge_damping*edge_speed)+inc@(self.memory_coupling*z)
        force=coi_project_power(force,mass)
        return omega,force/mass,-self.decay*z-self.memory_coupling*edge_speed

    def vector_field(self,delta,omega,pm,static):
        z=self.initial_memory(delta,omega,static);dd,dw,_=self.vector_field_augmented(delta,omega,z,pm,static);return dd,dw

    def _rollout_augmented(self,initial_state,time_grid,pm,*,interpolation="auto",max_step=None):
        t=np.asarray(time_grid,dtype=np.float64);pm=np.asarray(pm,dtype=np.float64);mass=np.asarray(initial_state["mass"]);d=coi_project(np.asarray(initial_state["delta0"]).reshape(-1),mass);w=coi_project(np.asarray(initial_state["omega0"]).reshape(-1),mass);z=self.initial_memory(d,w,initial_state);od,ow,oz=[d.copy()],[w.copy()],[z.copy()];step=self.base.max_internal_step if max_step is None else max_step;mode="zoh" if interpolation=="zoh" or (interpolation=="auto" and np.allclose(pm,pm[[0]])) else "linear"
        for k in range(len(t)-1):
            interval=float(t[k+1]-t[k]);
            if interval<=0:raise ValueError("rollout time must be strictly increasing")
            sub=max(1,int(np.ceil(interval/step)));h=interval/sub
            for j in range(sub):
                def p(q):return pm[k] if mode=="zoh" else pm[k]+q*(pm[k+1]-pm[k])
                q0=j/sub;qh=(j+.5)/sub;q1=(j+1)/sub;k1=self.vector_field_augmented(d,w,z,p(q0),initial_state);k2=self.vector_field_augmented(d+.5*h*k1[0],w+.5*h*k1[1],z+.5*h*k1[2],p(qh),initial_state);k3=self.vector_field_augmented(d+.5*h*k2[0],w+.5*h*k2[1],z+.5*h*k2[2],p(qh),initial_state);k4=self.vector_field_augmented(d+h*k3[0],w+h*k3[1],z+h*k3[2],p(q1),initial_state);d=coi_project(d+h*(k1[0]+2*k2[0]+2*k3[0]+k4[0])/6,mass);w=coi_project(w+h*(k1[1]+2*k2[1]+2*k3[1]+k4[1])/6,mass);z=z+h*(k1[2]+2*k2[2]+2*k3[2]+k4[2])/6
            if not np.isfinite(d).all() or not np.isfinite(w).all() or not np.isfinite(z).all():raise FloatingPointError(f"nonfinite port-Hamiltonian rollout at {k+1}")
            od.append(d.copy());ow.append(w.copy());oz.append(z.copy())
        return np.stack(od),np.stack(ow),np.stack(oz)

    def rollout(self,initial_state,time_grid,exogenous):
        d,w,_=self._rollout_augmented(initial_state,time_grid,exogenous["Pm"],interpolation=exogenous.get("interpolation","auto"));return d,w

    def storage(self,delta,omega,static,z=None):
        if self.kappa is None:raise RuntimeError("energy model is not fitted")
        inc=np.asarray(static["incidence"]);eta=inc.T@delta;eta0=inc.T@np.asarray(static["equilibrium_delta"]);shifted=np.cos(eta0)-np.cos(eta)-np.sin(eta0)*(eta-eta0);memory=0.0 if z is None else 0.5*np.sum(np.asarray(z)**2)
        return float(0.5*np.sum(np.asarray(static["mass"])*np.asarray(omega)**2)+np.sum(self.kappa*shifted)+memory)

    def storage_gradient(self,delta,omega,z,static):
        inc=np.asarray(static["incidence"]);eta=inc.T@delta;eta0=inc.T@np.asarray(static["equilibrium_delta"]);return inc@(self.kappa*(np.sin(eta)-np.sin(eta0))),np.asarray(static["mass"])*omega,z

    def supply_rate(self,omega,z,pm,static):
        edge_speed=np.asarray(static["incidence"]).T@omega;dpm=np.asarray(pm).reshape(-1)-_pm_equilibrium(static);return float(omega@dpm-omega@(self.node_damping*omega)-edge_speed@(self.edge_damping*edge_speed)-z@(self.decay*z))

    def energy_step_audit(self,static,pm,interval=0.04):
        errors=[]
        for divisor in (1,2,4):
            grid=np.linspace(0.0,interval,divisor+1);schedule=np.repeat(np.asarray(pm).reshape(1,-1),len(grid),axis=0);d,w,z=self._rollout_augmented(static,grid,schedule,interpolation="zoh",max_step=interval/divisor);hvals=np.asarray([self.storage(d[i],w[i],static,z[i]) for i in range(len(grid))]);rates=np.asarray([self.supply_rate(w[i],z[i],schedule[i],static) for i in range(len(grid))]);balance=(hvals[-1]-hvals[0])-float(np.trapezoid(rates,grid));errors.append(abs(balance))
        return {"energy_balance_dt":errors[0],"energy_balance_dt2":errors[1],"energy_balance_dt4":errors[2],"discrete_energy_convergent":bool(errors[2]<=errors[1]+1e-12<=errors[0]+2e-12)}

    def save(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,damping=self.base.damping,coupling=self.base.coupling,node_damping=self.node_damping,edge_damping=self.edge_damping,kappa=self.kappa,decay=self.decay,memory_coupling=self.memory_coupling,max_internal_step=self.base.max_internal_step);path.with_suffix(".json").write_text(json.dumps({"method":self.method_name,"reported_name":self.reported_name,"diagnostics":self.diagnostics},indent=2),encoding="utf-8")

    @classmethod
    def load(cls,path):
        path=Path(path);meta=json.loads(path.with_suffix(".json").read_text(encoding="utf-8")) if path.with_suffix(".json").exists() else {}
        with np.load(path) as data:
            base=M0Model(data["damping"],data["coupling"],max_internal_step=float(data["max_internal_step"]) if "max_internal_step" in data else .01);model=cls(base,data["node_damping"],data["edge_damping"],data["kappa"],data["decay"],data["memory_coupling"],meta.get("reported_name"));model.diagnostics=meta.get("diagnostics",{});return model

    def physics_audit(self):
        if self.node_damping is None:return {"status":"not_fitted"}
        positive=all(np.all(x>0) for x in (self.node_damping,self.edge_damping,self.kappa,self.decay))
        return {"status":"pending_certificate_gate" if positive else "fail","minimum_node_damping":float(np.min(self.node_damping)),"minimum_edge_damping":float(np.min(self.edge_damping)),"minimum_kappa":float(np.min(self.kappa)),"minimum_decay":float(np.min(self.decay)),"positive_dissipation_parameters":bool(positive),"port_hamiltonian_cancellation_by_construction":True,"reported_name":self.reported_name}
