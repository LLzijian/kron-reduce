from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from kron_reduce.physics.coordinates import coi_basis


@dataclass
class EnergyGateResult:
    reported_name: str
    certificate_passed: bool
    storage_passed: bool
    centering_error: float
    gradient_norm: float
    minimum_projected_hessian: float
    maximum_chain_rule_error: float
    maximum_supply_violation: float
    artificial_violation_loss: float
    maximum_analytical_identity_error: float = float("nan")
    evaluated_points: int = 0
    discrete_energy_convergent: bool = False
    energy_balance_dt: float = float("nan")
    energy_balance_dt2: float = float("nan")
    energy_balance_dt4: float = float("nan")
    accuracy_not_worse_than_m0: bool = False
    rollout_failure_rate: float = 1.0
    model_mean_rmse: float = float("nan")
    m0_mean_rmse: float = float("nan")

    def to_dict(self):
        return asdict(self)


def _gradient(function, x, epsilon=1e-6):
    result = np.empty_like(x, dtype=np.float64)
    for index in range(len(x)):
        direction = np.zeros_like(x); direction[index] = epsilon
        result[index] = (function(x + direction) - function(x - direction)) / (2 * epsilon)
    return result


def _hessian(function, x, epsilon=2e-5):
    hessian = np.empty((len(x), len(x)), dtype=np.float64)
    for index in range(len(x)):
        direction = np.zeros_like(x); direction[index] = epsilon
        hessian[:, index] = (_gradient(function, x + direction) - _gradient(function, x - direction)) / (2 * epsilon)
    return 0.5 * (hessian + hessian.T)


def supply_violation_loss(dot_h: float, external_supply: float, tolerance: float = 0.0) -> float:
    return float(max(0.0, dot_h - external_supply - tolerance) ** 2)


def evaluate_energy_gate(model, samples: list[dict], tolerance: float = 1e-6) -> EnergyGateResult:
    if not samples:
        raise ValueError("energy gate requires train/val samples")
    sample = samples[0]; static = sample["model_inputs"]["static"]
    eq = np.asarray(static["equilibrium_delta"], dtype=np.float64); zeros = np.zeros_like(eq)
    n = len(eq); e = np.asarray(static["incidence"]).shape[1]; z0 = np.zeros(e)
    function = lambda x: float(model.storage(x[:n], x[n:2*n], static, x[2*n:]))
    xeq = np.concatenate((eq, zeros, z0)); center = abs(function(xeq)); gradient = _gradient(function, xeq)
    basis = coi_basis(np.asarray(static["mass"])); projection = np.block([
        [basis, np.zeros_like(basis), np.zeros((n, e))],
        [np.zeros_like(basis), basis, np.zeros((n, e))],
        [np.zeros((e, n-1)), np.zeros((e, n-1)), np.eye(e)],
    ])
    projected = projection.T @ _hessian(function, xeq) @ projection
    minimum_hessian = float(np.min(np.linalg.eigvalsh(projected)))
    chain_errors=[]; identity_errors=[]; supply_violations=[]; evaluated=0
    for candidate in samples:
        target=candidate["training_context"]["targets"]; pm=candidate["model_inputs"]["exogenous"]["Pm"]; static_i=candidate["model_inputs"]["static"]
        indices=np.unique(np.linspace(0,len(target["t"])-1,min(8,len(target["t"])),dtype=int))
        for index in indices:
            d,w=target["delta"][index],target["omega"][index]; z=model.initial_memory(d,w,static_i); x=np.concatenate((d,w,z)); analytical=np.concatenate(model.storage_gradient(d,w,z,static_i)); fd,fw,fz=model.vector_field_augmented(d,w,z,pm[index],static_i); vector=np.concatenate((fd,fw,fz)); dot_h=float(analytical@vector); eps=1e-6; directional=(float(model.storage(x[:n]+eps*fd,x[n:2*n]+eps*fw,static_i,x[2*n:]+eps*fz))-float(model.storage(x[:n]-eps*fd,x[n:2*n]-eps*fw,static_i,x[2*n:]-eps*fz)))/(2*eps); expected=model.supply_rate(w,z,pm[index],static_i); external=float(w@(np.asarray(pm[index]).reshape(-1)-np.asarray(static_i["Pm_equilibrium"]).reshape(-1))); chain_errors.append(abs(dot_h-directional));identity_errors.append(abs(dot_h-expected));supply_violations.append(dot_h-external);evaluated+=1
    artificial_loss=supply_violation_loss(1.0,0.0)
    step=model.energy_step_audit(static,np.asarray(sample["model_inputs"]["exogenous"]["Pm"])[0])
    model_rmse=[];base_rmse=[];failures=0
    for candidate in samples:
        target=candidate["training_context"]["targets"];steps=min(100,len(target["t"]));static_i=candidate["model_inputs"]["static"];schedule=np.asarray(candidate["model_inputs"]["exogenous"]["Pm"][:steps])
        try:
            md,mw=model.rollout(static_i,target["t"][:steps],{"Pm":schedule,"interpolation":"zoh"});bd,bw=model.base.rollout(static_i,target["t"][:steps],{"Pm":schedule,"interpolation":"zoh"});model_error=np.concatenate(((md-target["delta"][:steps]).ravel(),(mw-target["omega"][:steps]).ravel()));base_error=np.concatenate(((bd-target["delta"][:steps]).ravel(),(bw-target["omega"][:steps]).ravel()));model_rmse.append(float(np.sqrt(np.mean(model_error**2))));base_rmse.append(float(np.sqrt(np.mean(base_error**2))))
        except (ValueError,FloatingPointError):failures+=1
    model_mean=float(np.mean(model_rmse)) if model_rmse else float("inf");base_mean=float(np.mean(base_rmse)) if base_rmse else float("inf");failure_rate=failures/len(samples);accuracy_pass=bool(failures==0 and model_mean<=base_mean*(1.0+1e-8)+1e-12)
    maximum_chain=float(max(chain_errors,default=np.inf));maximum_identity=float(max(identity_errors,default=np.inf));maximum_supply=float(max(supply_violations,default=np.inf))
    storage_passed=bool(center<=tolerance and np.linalg.norm(gradient)<=tolerance*10 and minimum_hessian>1e-9 and maximum_chain<=tolerance*10 and step["discrete_energy_convergent"])
    certificate_passed=bool(storage_passed and maximum_identity<=tolerance*10 and maximum_supply<=tolerance and artificial_loss>0 and accuracy_pass)
    name="M2A-LYAP" if certificate_passed else ("M2A-storage" if storage_passed else "M2A-energy-regularized");model.reported_name=name
    return EnergyGateResult(name,certificate_passed,storage_passed,center,float(np.linalg.norm(gradient)),minimum_hessian,maximum_chain,maximum_supply,artificial_loss,maximum_identity,evaluated,bool(step["discrete_energy_convergent"]),float(step["energy_balance_dt"]),float(step["energy_balance_dt2"]),float(step["energy_balance_dt4"]),accuracy_pass,float(failure_rate),model_mean,base_mean)
