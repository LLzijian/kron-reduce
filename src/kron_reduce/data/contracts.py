from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PmMode(str, Enum):
    ZOH = "zoh"
    OBSERVED = "observed"


class RequestMode(str, Enum):
    INITIAL_STATE_ROLLOUT = "initial_state_rollout"
    PREFIX_STEPS = "prefix_steps"


@dataclass(frozen=True)
class InputContract:
    name: str = "causal_postclear_v1"
    allowed_static: tuple[str, ...] = (
        "delta0", "omega0", "Pm_equilibrium", "Pm_clear", "Pm0",
        "equilibrium_delta", "mass", "incidence"
    )
    allowed_exogenous: tuple[str, ...] = ("t", "Pm")
    forbidden: tuple[str, ...] = (
        "Pe", "bus_v", "bus_a", "label", "outcome_label", "event_time_s",
        "valid_length", "valid_mask", "fault_line", "termination_reason",
    )


@dataclass(frozen=True)
class ModelDataRequest:
    mode: RequestMode
    pm_mode: PmMode
    n_steps: int | None = None
    prospective: bool = False

    @classmethod
    def initial_state_rollout(cls, pm_mode: str = "zoh") -> "ModelDataRequest":
        return cls(RequestMode.INITIAL_STATE_ROLLOUT, PmMode(pm_mode))

    @classmethod
    def prefix_steps(
        cls, n_steps: int, prospective: bool = True, pm_mode: str = "zoh"
    ) -> "ModelDataRequest":
        if n_steps < 2:
            raise ValueError("prefix requests require at least two real samples")
        return cls(RequestMode.PREFIX_STEPS, PmMode(pm_mode), n_steps, prospective)


class ShortSequenceError(ValueError):
    pass


class IneligibleEventError(ValueError):
    pass
