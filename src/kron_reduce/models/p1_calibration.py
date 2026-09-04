"""All-case, controlled loss calibration for P1, not scientific promotion.

The legacy refiner remains untouched. Both loss arms here share the same
residual cells, initialization, physical group coordinates, and solver budget.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from kron_reduce.data.p1_causal import P1RoleManifest, validate_p1_samples
from kron_reduce.evaluation.pm_closure_diagnostics import (
    FinalLossCellWeight,
    GradientAudit,
    GradientMeasurement,
    SharedF1EvaluationContract,
)
from kron_reduce.models.base import sample_rollout_inputs
from kron_reduce.models.prunable_energy_model import (
    EDGE_DAMPING,
    EDGE_MEMORY,
    NODE_HIGH_ORDER,
    NONLINEAR_POTENTIAL,
    RESONANT_MEMORY,
    PrunableEnergyModel,
)

_BLOCK_PARAMETERS = {
    NONLINEAR_POTENTIAL: "potential_coefficients",
    EDGE_DAMPING: "edge_damping_coefficients",
    NODE_HIGH_ORDER: "node_high_order_gain",
    EDGE_MEMORY: "memory_port_operator",
    RESONANT_MEMORY: "resonant_port_operator",
}
_ARRAYS = ("node_damping", "coupling", "potential_coefficients", "edge_damping_coefficients", "node_high_order_gain", "memory_gains", "resonant_gains")


@dataclass(frozen=True)
class P1CalibrationSettings:
    max_nfev: int = 16
    ftol: float = 1e-8
    xtol: float = 1e-8
    gtol: float = 1e-8
    lower_scale: float = 0.25
    upper_scale: float = 4.0

    def __post_init__(self) -> None:
        if type(self.max_nfev) is not int or self.max_nfev < 1:
            raise ValueError("a positive common solver budget is required")
        if (self.ftol, self.xtol, self.gtol, self.lower_scale, self.upper_scale) != (1e-8, 1e-8, 1e-8, 0.25, 4.0):
            raise ValueError("P1 calibration tolerances and physical scale bounds are fixed")


class CalibrationRolloutError(RuntimeError):
    """An event failed; no residual cells may be removed or replaced by zero."""


def _squared_loss(residual: np.ndarray) -> float:
    try:
        with np.errstate(over="raise", invalid="raise"):
            value = float(residual @ residual)
    except FloatingPointError as error:
        raise CalibrationRolloutError("nonfinite squared objective") from error
    if not np.isfinite(value):
        raise CalibrationRolloutError("nonfinite squared objective")
    return value


class P1CalibrationObjective:
    def __init__(self, samples: Sequence[dict[str, Any]], manifest: P1RoleManifest, *, balanced: bool):
        if type(balanced) is not bool:
            raise TypeError("balanced must be explicit bool")
        validate_p1_samples(samples, manifest, "fit")
        self.samples = tuple(copy.deepcopy(sample) for sample in sorted(samples, key=lambda item: item["event_id"]))
        self.contract = SharedF1EvaluationContract(manifest.role_id_hashes)
        self.manifest_sha256 = manifest.source_sha256
        self.balanced = balanced
        # Same legacy scale definition, estimated only on the complete fit role.
        try:
            with np.errstate(over="raise", invalid="raise"):
                self.scales = PrunableEnergyModel._training_scales(self.samples)
        except FloatingPointError as error:
            raise ValueError("invalid training scales") from error
        if any(not np.isfinite(value) or value < 1e-8 for value in self.scales.values()):
            raise ValueError("invalid training scales")
        keys: list[tuple[str, str, str, str, str]] = []
        dimensions = {}
        for sample in self.samples:
            metadata = sample["training_context"]["metadata"]
            static = sample["model_inputs"]["static"]
            dimensions[sample["event_id"]] = {"delta": static["incidence"].shape[1], "omega": len(static["mass"])}
            for horizon in self.contract.horizons_seconds:
                for output in ("delta", "omega"):
                    keys.append((metadata["source_dataset"], metadata["fault_line"], sample["event_id"], str(horizon), output))
        children: dict[tuple[str, ...], set[str]] = defaultdict(set)
        for key in keys:
            for index, value in enumerate(key):
                children[key[:index]].add(value)
        self.cells = []
        self.slices = []
        offset = 0
        for key in keys:
            if balanced:
                weight = 1.0
                for index in range(len(key)):
                    weight /= len(children[key[:index]])
            else:
                dims = dimensions[key[2]]
                weight = dims[key[4]] / sum(dims.values()) / len(self.samples) / len(self.contract.horizons_seconds)
            cell_id = json.dumps(key, separators=(",", ":"))
            self.cells.append(FinalLossCellWeight(cell_id, *key, weight))
            size = (int(key[3]) * 100 + 1) * dimensions[key[2]][key[4]]
            self.slices.append(slice(offset, offset + size))
            offset += size
        self.cells, self.slices = tuple(self.cells), tuple(self.slices)
        self.residual_size = offset

    @property
    def common_design(self) -> dict[str, Any]:
        return {
            "contract": asdict(self.contract), "role_manifest_sha256": self.manifest_sha256,
            "fit_ids": [sample["event_id"] for sample in self.samples],
            "scales": dict(self.scales), "angle_coordinates": "edge_angle",
            "registered_cell_keys": [list(cell.registered_key) for cell in self.cells],
            "scale_role": "fit", "parameter_calibration_role": "fit",
            "selection_used_for_calibration": False,
        }

    def assemble(self, predictions: dict[str, tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        if set(predictions) != {sample["event_id"] for sample in self.samples}:
            raise CalibrationRolloutError("predictions must cover every registered fit event")
        errors = {}
        for sample in self.samples:
            event_id = sample["event_id"]
            target = sample["training_context"]["targets"]
            delta, omega = predictions[event_id]
            if any(np.shape(value) != np.shape(target[name]) or not np.isfinite(value).all() for name, value in (("delta", delta), ("omega", omega))):
                raise CalibrationRolloutError(f"{event_id}: nonfinite or incomplete prediction")
            incidence = sample["model_inputs"]["static"]["incidence"]
            errors[event_id] = {
                "delta": (np.asarray(delta) - target["delta"]) @ incidence / self.scales["edge_angle"],
                "omega": (np.asarray(omega) - target["omega"]) / self.scales["omega"],
            }
        residual = np.empty(self.residual_size)
        for cell, section in zip(self.cells, self.slices, strict=True):
            stop = int(cell.horizon) * 100 + 1
            values = errors[cell.event][cell.output][:stop]
            residual[section] = values.reshape(-1) * np.sqrt(cell.weight / values.size)
        if not np.isfinite(residual).all():
            raise CalibrationRolloutError("nonfinite complete residual")
        return residual

    def evaluate(self, model: PrunableEnergyModel) -> np.ndarray:
        if model.max_internal_step != self.contract.max_step:
            raise ValueError("model integrator differs from the frozen common contract")
        predictions = {}
        for sample in self.samples:
            static, time, exogenous = sample_rollout_inputs(sample)
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    predictions[sample["event_id"]] = model.rollout(static, time, exogenous)
            except (FloatingPointError, ValueError) as error:
                raise CalibrationRolloutError(f"{sample['event_id']}: {error}") from error
        return self.assemble(predictions)


def core_fingerprint(model: PrunableEnergyModel) -> str:
    """Equation identity, not a substitute for an immutable file manifest."""
    description = {
        "blocks": [row.to_dict() for row in model.active_instances],
        "node_tau": model.node_tau, "potential_scale": model.potential_scale,
        "max_internal_step": model.max_internal_step,
        "memory_taus": list(model.memory_taus), "resonances": list(model.resonances),
    }
    digest = hashlib.sha256(json.dumps(description, sort_keys=True).encode())
    arrays = [(name, getattr(model, name)) for name in _ARRAYS]
    arrays += [(f"{name}_{index}", value) for name in ("memory_factors", "resonant_factors") for index, value in enumerate(getattr(model, name))]
    for name, value in arrays:
        array = np.asarray(value, dtype="<f8")
        if not np.isfinite(array).all():
            raise ValueError(f"nonfinite initial parameters: {name}")
        digest.update(name.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


class PhysicalGroupScaling:
    """A private copy of the starting core; caller's model is never modified."""

    def __init__(self, initial_model: PrunableEnergyModel):
        initial_model._require_fitted()
        core_fingerprint(initial_model)
        if any(np.any(np.asarray(getattr(initial_model, name)) < 0) for name in _ARRAYS):
            raise ValueError("physical group scaling requires nonnegative initial coefficients")
        self.model = copy.deepcopy(initial_model)
        self.reference = copy.deepcopy(initial_model)
        self.names = ("node_damping", "coupling", *(_BLOCK_PARAMETERS[block] for block in initial_model.active_blocks))

    def assign(self, log_scales: np.ndarray) -> PrunableEnergyModel:
        values = np.asarray(log_scales)
        if values.shape != (len(self.names),) or not np.isfinite(values).all():
            raise ValueError("invalid group scale vector")
        for name, log_scale in zip(self.names, values, strict=True):
            scale = float(np.exp(log_scale))
            if name in {"memory_port_operator", "resonant_port_operator"}:
                family = "memory" if name.startswith("memory") else "resonant"
                setattr(self.model, f"{family}_factors", [factor * np.sqrt(scale) for factor in getattr(self.reference, f"{family}_factors")])
                setattr(self.model, f"{family}_gains", getattr(self.reference, f"{family}_gains") * scale)
            else:
                setattr(self.model, name, np.asarray(getattr(self.reference, name)) * scale)
        return self.model


def measure_gradient_audit(
    objective: P1CalibrationObjective,
    residual_function: Callable[[np.ndarray], np.ndarray],
    coordinates: np.ndarray,
    group_names: tuple[str, ...],
    *,
    step: float = 1e-5,
) -> tuple[GradientAudit, dict[str, Any]]:
    """Cell gradients and independent whole-residual/whole-objective checks."""
    point = np.asarray(coordinates, dtype=float)
    if point.shape != (len(group_names),) or not np.isfinite(point).all() or step <= 0 or not np.isfinite(step):
        raise ValueError("invalid gradient audit coordinates")
    try:
        base = residual_function(point)
        jacobian = np.empty((objective.residual_size, len(point)))
        objective_gradient = []
        for index in range(len(point)):
            perturbation = np.zeros_like(point)
            perturbation[index] = step
            plus, minus = residual_function(point + perturbation), residual_function(point - perturbation)
            jacobian[:, index] = (plus - minus) / (2 * step)
            objective_gradient.append((_squared_loss(plus) - _squared_loss(minus)) / (2 * step))
        # Compute totals directly from complete arrays, not by summing cell records.
        total = 2 * jacobian.T @ base
        measurements = []
        for cell, section in zip(objective.cells, objective.slices, strict=True):
            local = 2 * jacobian[section].T @ base[section]
            measurements.extend((cell.cell_id, group, float(value)) for group, value in zip(group_names, local, strict=True))
        squared_total = sum(value**2 for _cell, _group, value in measurements)
        audit = GradientAudit(
            expected_cell_ids=tuple(cell.cell_id for cell in objective.cells),
            expected_parameter_group_ids=group_names,
            measurements=tuple(GradientMeasurement(cell, group, (value,), value**2 / squared_total if squared_total else 0.0) for cell, group, value in measurements),
            total_gradients={name: (float(value),) for name, value in zip(group_names, total, strict=True)},
        )
        relative_error = float(np.linalg.norm(total - objective_gradient) / max(np.linalg.norm(total), np.linalg.norm(objective_gradient), 1e-12))
        return audit, {"whole_objective_difference_gradient": objective_gradient, "relative_error": relative_error, "step": step, "method": "central_difference_complete_residual_and_complete_scalar_objective"}
    finally:
        residual_function(point)  # Restore the private model to the audited point.


def calibrate_loss_pair(
    initial_model: PrunableEnergyModel,
    fit_samples: Sequence[dict[str, Any]],
    manifest: P1RoleManifest,
    *,
    settings: P1CalibrationSettings | None = None,
    loss_modes: tuple[bool, ...] = (False, True),
) -> tuple[tuple[PrunableEnergyModel, dict[str, Any]], ...]:
    """Compare losses, without certifying prior training lineage or selecting a winner."""
    settings = settings or P1CalibrationSettings()
    if loss_modes not in {(False,), (True,), (False, True)} or any(type(mode) is not bool for mode in loss_modes):
        raise ValueError("use the registered pair or one registered loss for a controlled arm")
    objectives = [P1CalibrationObjective(fit_samples, manifest, balanced=flag) for flag in loss_modes]
    if any(objective.common_design != objectives[0].common_design for objective in objectives):
        raise RuntimeError("loss arms must share the complete design")
    common = {**objectives[0].common_design, "solver": asdict(settings)}
    common_hash = hashlib.sha256(json.dumps(common, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    initial_hash = core_fingerprint(initial_model)
    output = []
    for objective in objectives:
        coordinates = PhysicalGroupScaling(initial_model)
        initial = np.zeros(len(coordinates.names))
        calls = 0

        def residual(values: np.ndarray, *, objective: P1CalibrationObjective = objective, coordinates: PhysicalGroupScaling = coordinates) -> np.ndarray:
            nonlocal calls
            calls += 1
            return objective.evaluate(coordinates.assign(values))

        record: dict[str, Any] = {
            "loss": "balanced" if objective.balanced else "original_all_case",
            "common_design": common, "common_design_sha256": common_hash,
            "initial_core_fingerprint": initial_hash,
            "parameter_groups": list(coordinates.names),
            "final_cell_weights": [asdict(cell) for cell in objective.cells],
            "prior_fit_only_lineage_proven": False,
            "scientific_acceptance": False, "eligible_for_promotion": False,
        }
        accepted = False
        final_coordinates = initial
        try:
            baseline = residual(initial)
            baseline_loss = _squared_loss(baseline)
            result = least_squares(
                residual, initial, bounds=(np.log(settings.lower_scale), np.log(settings.upper_scale)),
                max_nfev=settings.max_nfev, ftol=settings.ftol, xtol=settings.xtol, gtol=settings.gtol,
            )
            candidate_coordinates = np.asarray(result.x)
            if candidate_coordinates.shape != initial.shape or not np.isfinite(candidate_coordinates).all() or np.any(candidate_coordinates < np.log(settings.lower_scale)) or np.any(candidate_coordinates > np.log(settings.upper_scale)):
                raise CalibrationRolloutError("solver returned parameters outside registered bounds")
            refined = residual(result.x)
            candidate_loss = _squared_loss(refined)
            accepted = bool(result.success and candidate_loss <= baseline_loss)
            final_coordinates = result.x if accepted else initial
            record.update({"baseline_loss": baseline_loss, "candidate_loss": candidate_loss, "solver_success": bool(result.success), "solver_nfev": int(result.nfev)})
        except CalibrationRolloutError as error:
            record.update({"failure": str(error), "solver_success": False})
        model = coordinates.assign(final_coordinates) if accepted else copy.deepcopy(initial_model)
        if accepted:
            model.validation_scales = dict(objective.scales)
            model._refresh_parameter_diagnostics()
        record.update({"status": "calibration_accepted" if accepted else "calibration_rejected", "objective_calls": calls, "final_group_scales": dict(zip(coordinates.names, np.exp(final_coordinates).tolist(), strict=True)), "final_core_fingerprint": core_fingerprint(model)})
        output.append((model, record))
    if core_fingerprint(initial_model) != initial_hash:
        raise RuntimeError("calibration mutated the starting core")
    return tuple(output)
