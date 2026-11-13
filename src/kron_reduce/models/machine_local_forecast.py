"""Frozen public prediction entry point for the nonlinear machine-local model."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from kron_reduce.data.history_contract import CausalStateHistory
from kron_reduce.models.machine_local import DiagonalMemoryModel
from kron_reduce.models.nonlinear_history import predict_from_history
from kron_reduce.models.nonlinear_latent import NonlinearPassiveClosure


class FrozenMachineLocalForecaster(torch.nn.Module):
    """Public prediction and checkpoint contract for a frozen machine-local model.

    Checkpoints include model parameters, the fixed history normalization, and
    the explicit numerical entry contract in ``state_dict`` extra state.
    """

    INITIALIZER = "nonlinear_history_gauss_newton_v1"
    GN_UPDATES = 3
    HISTORY_DT = 0.01
    HISTORY_FRAMES = 11
    CONTRACT_VERSION = 1

    def __init__(self, model: DiagonalMemoryModel, history_scale, *, max_step=0.005):
        super().__init__()
        if not isinstance(model, DiagonalMemoryModel) or model.backbone_kind != "sine":
            raise TypeError("frozen forecast requires a sine DiagonalMemoryModel")
        scale = np.asarray(history_scale, dtype=np.float64)
        if (scale.shape != (2*model.n_nodes,) or not np.isfinite(scale).all()
                or np.any(scale <= 0)):
            raise ValueError("history_scale must contain positive finite delta and omega scales")
        if max_step not in (0.005, 0.0025, 0.00125):
            raise ValueError("max_step must use a registered nonlinear-history RK4 step")
        self.model = model
        self.register_buffer("history_scale", torch.tensor(scale, dtype=torch.float64))
        self.max_step = float(max_step)

    def get_extra_state(self):
        return {
            "contract_version": self.CONTRACT_VERSION,
            "initializer": self.INITIALIZER,
            "gauss_newton_updates": self.GN_UPDATES,
            "history_dt_seconds": self.HISTORY_DT,
            "history_frames": self.HISTORY_FRAMES,
            "history_scale": self.history_scale.detach().cpu().tolist(),
            "max_step_seconds": self.max_step,
            "visible_reset": "measured_last_frame_before_forecast",
            "effective_stiffness": self.model.stiffness.detach().cpu().tolist(),
            "port_kind": self.model.port_kind,
            "backbone_kind": self.model.backbone_kind,
        }

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("checkpoint forecast contract does not match this forecaster")

    def save_checkpoint(self, path) -> None:
        """Write a new weights-only checkpoint."""
        if self.model.log_stiffness_factor is not None:
            raise ValueError("fold the effective stiffness into the model before export")
        with Path(path).open("xb") as stream:
            torch.save(self.state_dict(), stream)

    @classmethod
    def load_checkpoint(cls, path):
        """Rebuild a forecaster with the restricted weights-only loader."""
        state = torch.load(Path(path), map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not isinstance(state.get("_extra_state"), dict):
            raise TypeError("checkpoint has no frozen forecast contract")
        contract = state["_extra_state"]
        required = {"contract_version", "initializer", "gauss_newton_updates", "history_dt_seconds",
                    "history_frames", "history_scale", "max_step_seconds", "effective_stiffness",
                    "visible_reset",
                    "port_kind", "backbone_kind"}
        if (set(contract) != required or contract["contract_version"] != cls.CONTRACT_VERSION
                or contract["initializer"] != cls.INITIALIZER
                or contract["gauss_newton_updates"] != cls.GN_UPDATES
                or contract["history_dt_seconds"] != cls.HISTORY_DT
                or contract["history_frames"] != cls.HISTORY_FRAMES
                or contract["visible_reset"] != "measured_last_frame_before_forecast"
                or contract["port_kind"] not in {"local", "global"}
                or contract["backbone_kind"] != "sine"):
            raise ValueError("unsupported frozen forecast contract")
        prefix = "model."
        try:
            take = lambda name: state[prefix+name].detach().cpu().numpy()
            mass, stiffness, damping = take("mass"), take("stiffness"), take("damping")
            encoder, raw_port, raw_rate = take("encoder"), take("port_parameter"), take("rate_parameter")
            feature_scale = take("feature_scale")
        except (KeyError, AttributeError) as exc:
            raise ValueError("checkpoint is missing model tensors") from exc
        n, r = len(mass), encoder.shape[0]
        if encoder.shape != (r, 8*n) or raw_rate.shape != (r,):
            raise ValueError("checkpoint model tensor shapes are invalid")
        tau = np.exp(np.log(.02)+np.log(5./.02)/(1+np.exp(-raw_rate)))
        port = np.exp(raw_port) if contract["port_kind"] == "local" else raw_port
        coupling = np.zeros((n, r))
        backbone = NonlinearPassiveClosure(mass, stiffness, damping, coupling,
                                           np.zeros((r, r)), np.eye(r), "sine")
        model = DiagonalMemoryModel(backbone, port_kind=contract["port_kind"], port=port,
                                    tau=tau, encoder=encoder, feature_scale=feature_scale)
        forecaster = cls(model, contract["history_scale"], max_step=contract["max_step_seconds"])
        for name, expected in forecaster.named_buffers():
            value = state.get(name)
            if (not isinstance(value, torch.Tensor) or value.dtype != expected.dtype
                    or not torch.equal(value, expected)):
                raise ValueError(f"checkpoint buffer differs from its model contract: {name}")
        forecaster.load_state_dict(state, strict=True)
        return forecaster

    def predict_batch(self, histories, equilibria, count: int, *, spend=None):
        """Run the shared three-update initializer and return forecast plus traces."""
        histories = list(histories)
        if not histories or any(not isinstance(item, CausalStateHistory) for item in histories):
            raise TypeError("prediction requires a nonempty sequence of CausalStateHistory")
        if any(len(item.time) != self.HISTORY_FRAMES
               or not np.allclose(np.diff(item.time), self.HISTORY_DT, atol=1e-12, rtol=0)
               for item in histories):
            raise ValueError("prediction requires eleven frames on the 10 ms history clock")
        equilibria = np.asarray(equilibria, dtype=np.float64)
        n = self.model.n_nodes
        if equilibria.shape != (len(histories), n) or not np.isfinite(equilibria).all():
            raise ValueError("equilibria must be finite and match the history batch")
        if type(count) is not int or count < 0:
            raise ValueError("count must be a nonnegative integer")
        with torch.no_grad():
            return predict_from_history(
                self.model,
                torch.as_tensor(np.stack([item.delta for item in histories]), dtype=torch.float64),
                torch.as_tensor(np.stack([item.omega for item in histories]), dtype=torch.float64),
                torch.as_tensor(np.stack([item.constant_power for item in histories]), dtype=torch.float64),
                torch.as_tensor(equilibria, dtype=torch.float64),
                self.history_scale,
                count,
                max_step=self.max_step,
                spend=spend or (lambda *_: None),
            )

    def predict(self, history: CausalStateHistory, equilibrium, count: int) -> np.ndarray:
        """Predict visible and hidden states from measured history to forecast origin."""
        result = self.predict_batch([history], np.asarray(equilibrium, dtype=np.float64)[None], count)
        return result["forecast"][0].cpu().numpy()
