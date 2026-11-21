"""Forward-only adapters for the frozen TASK3 trajectory endpoints."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import numpy as np


LASS_LAST_BLOCK_HOLD = "lass_small_p0_last_block_hold"
HISTORICAL = ("m3_pm", "C3000", "F23_0101", "S23_0101")
REFERENCES = ("equilibrium", "persistence")
CAMPAIGN1 = ("hankel_dmd", "gru_seq2seq", "latent_neural_ode", "koopman_shred")
CAMPAIGN2 = (
    "graph_gru_extension1",
    "gru_seq2seq_extension1",
    "latent_neural_ode_extension1",
    "small_transformer",
    "koopman_shred_faithful",
    "hankel_dmd",
    "sindy_shred_faithful_revision1",
    "sindy_shred_authorized_retry",
    "TASK2_F23_P0_extension1",
    "TASK2_S23_P0_extension1",
    *HISTORICAL,
    *REFERENCES,
)
OTHERS = ("actnet_history_p0", "stan_rotor_p0", "i_ikno_p0_inferencefix", "lass_small_p0")
RELEASES = {
    "campaign1": CAMPAIGN1,
    "campaign2": CAMPAIGN2,
    "latest_algorithm_extension": ("history_deeponet_p0",),
    "latest_algorithm_others": (*OTHERS, LASS_LAST_BLOCK_HOLD),
}
BLIND_FREEZE = "runs/kron_blind_100ms_20261121/freeze_manifest.json"
FINAL_FREEZE = "paper/campaign2_final/freeze_manifest.json"
FIRST_FREEZE = "runs/campaign2/preflight/campaign1_readonly_manifest.json"
FREEZE_DIGESTS = {
    BLIND_FREEZE: "ab9257a0dfe996f6b4c9d4ceb7f9e761749fb42216a29a6decd2703f46554491",
    FINAL_FREEZE: "180fa58db64cb78848ba09d3423430d8ffb4f2ac7ada1f47e009a574a11a09fa",
    FIRST_FREEZE: "b48d3cc69cdcd7417963ed3c13f05a7edd7a7fb8f113aae1e19a925a58d7e586",
}


class UnsupportedHorizon(ValueError):
    """A frozen model has no forecast definition at the requested horizon."""


def _source_name(release: str, name: str) -> str:
    if name not in RELEASES.get(release, ()):
        raise ValueError(f"Unknown frozen TASK3 trajectory endpoint: {release}/{name}")
    return "lass_small_p0" if name == LASS_LAST_BLOCK_HOLD else name


def _folder(root: Path, release: str, name: str) -> Path:
    if release == "campaign1":
        return root / "runs" / f"campaign1_{name}"
    return root / "runs" / release / name


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _kiki_root(root: Path) -> Path:
    config = _read(root / "configs/dataset.json")
    default = root / config["fallback_kiki_root"]
    chosen = os.environ.get(config["kiki_root_env"])
    return (Path(chosen).expanduser() if chosen else default).resolve()


def _kiki_archive(root: Path) -> Path:
    return root.parent / "grid-transient-rom-curated-20261008"


def _historical_checkpoint(root: Path, registration: dict) -> Path:
    relative = "private_data/" + registration["checkpoint"].replace("\\", "/").split("/private_data/", 1)[1]
    original = _kiki_root(root) / relative
    if original.is_file() or not relative.endswith("structured_numeric_export.json"):
        return original
    return _kiki_archive(root) / "_selected_private_data" / relative


def bindings(root: Path, release: str, name: str) -> dict[str, str]:
    """Verify forecast code and model assets against the existing archive pins."""
    root = root.resolve()
    name = _source_name(release, name)
    bound = {}

    def bind(path: Path, expected: str) -> None:
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Frozen TASK3 SHA256 mismatch: {path}")
        bound[Path(os.path.relpath(path, root)).as_posix()] = actual

    for relative, expected in FREEZE_DIGESTS.items():
        bind(root / relative, expected)
    final = _read(root / FINAL_FREEZE)
    pins = {
        key.replace("\\", "/"): value
        for key, value in (_read(root / FIRST_FREEZE) | final["files"]).items()
    }
    for key, value in _read(root / BLIND_FREEZE)["frozen_files"].items():
        relative = key.replace("\\", "/").split("/task3_baselines/", 1)[-1]
        if relative in pins and pins[relative] != value:
            raise ValueError(f"Conflicting TASK3 archive pins: {relative}")
        pins[relative] = value

    def local(relative: str) -> None:
        bind(root / relative, pins[relative])

    local("src/task3/models.py")
    local("src/task3/protocol.py")
    if release != "campaign1":
        for relative in ("c2/models.py", "c2/safety.py", "c2/evaluation.py"):
            local(f"src/task3/{relative}")
        local("runs/campaign2/preflight/static.npz")

    folder = _folder(root, release, name)
    prefix = folder.relative_to(root).as_posix()
    if release == "campaign1":
        local("src/task3/campaign.py")
        for file in ("result.json", "normalization.npz", "model.npz" if name == "hankel_dmd" else "checkpoint.pt"):
            local(f"{prefix}/{file}")
    elif name in HISTORICAL:
        local("src/task3/c2/historical.py")
        local(f"{prefix}/registration.json")
    elif name in REFERENCES:
        local("src/task3/c2/final_release.py")
    elif release == "campaign2":
        local(f"{prefix}/{'model.npz' if name == 'hankel_dmd' else 'selected.pt'}")
        if name == "hankel_dmd":
            local("src/task3/c2/dmd.py")
    else:
        local(f"{prefix}/endpoint_freeze.json")
        seal = _read(folder / "endpoint_freeze.json")
        bind(folder / "selected.pt", seal["selected_sha256"])
        for key, expected in seal["source_hashes"].items():
            bind(root / key.replace("\\", "/"), expected)

    if name.startswith(("TASK2_F23_", "TASK2_S23_")) or name in ("F23_0101", "S23_0101"):
        local("src/task3/c2/p0_kernel.py")
        for key in pins:
            if key.startswith("src/task3/c2/task2_math/") and key.endswith(".py"):
                local(key)
    if name in HISTORICAL:
        registration = _read(folder / "registration.json")
        checkpoint = _historical_checkpoint(root, registration)
        bind(checkpoint, registration["sha256"])
        if name == "C3000":
            kiki = _kiki_root(root)
            source = kiki / "private_data/task2_long_0060_5s"
            metadata = checkpoint.with_name("checkpoint.json")
            for key, expected in final["external_dependencies"].items():
                relative = key.replace("\\", "/").split("/kiki/", 1)[1]
                if relative.startswith(("experiments/", "src/")):
                    bind(_kiki_archive(root) / relative, expected)
                elif relative in {
                    metadata.relative_to(kiki).as_posix(),
                    "private_data/task2_long_0060_5s/model.pt",
                }:
                    bind(kiki / relative, expected)
            if Path(_read(metadata)["source"]).resolve() != source:
                raise ValueError("C3000 observer checkpoint points outside the verified source")
    return bound


def steps(release: str, name: str, evaluation_json: dict) -> tuple[float, float | None]:
    """Take the solver step and any comparison step from the original score."""
    name = _source_name(release, name)
    score = evaluation_json.get("evaluation", evaluation_json)
    if release == "campaign1" and name in ("latent_neural_ode", "koopman_shred"):
        return .005, .01
    reference = score.get("forecast_reference_dt", .01)
    if reference == .005:
        return reference, .0025 if score["refinement_executed"] else .01
    if reference == .01:
        return reference, None
    raise ValueError(f"Unexpected frozen TASK3 solver step: {reference}")


def _last_block_lass(static: dict):
    import torch
    from task3.extensions.other_models import TokenLinearODE

    class LastBlockHold(TokenLinearODE):
        def forward(self, history, power, times):
            late = times > 8
            if not bool(late.any()):
                return super().forward(history, power, times)
            intermediates = {}
            with self.parameters_head.register_forward_hook(
                lambda module, inputs, output: intermediates.update(parameters=output)
            ), self.initial.register_forward_hook(
                lambda module, inputs, output: intermediates.update(initial=output)
            ):
                original = super().forward(history, power, times[~late])
            parameters = intermediates["parameters"][:, :160].reshape(len(history), 20, 8, 8)
            initial = intermediates["initial"].reshape(len(history), 20, 2, 2)
            z0 = torch.complex(initial[..., 0], initial[..., 1])
            z = z0
            for q in parameters.unbind(dim=2):
                a = torch.complex(q[..., :2].tanh(), torch.nn.functional.softplus(q[..., 2:4]) + .1)
                force = torch.complex(q[..., 4:6], q[..., 6:8])
                z = torch.exp(a) * z + torch.expm1(a) / a * force
            elapsed = (times[late] - 8)[None, :, None, None]
            exponent = a[:, None] * elapsed
            delta = torch.exp(exponent) * z[:, None] + torch.expm1(exponent) / a[:, None] * force[:, None] - z0[:, None]
            real = torch.stack((delta.real, delta.imag), -1).flatten(-2)
            tail = history[:, -1, None] + (real * self.decoder[None, None]).sum(-1) * self.scale
            return torch.cat((original, tail), dim=1)

    return LastBlockHold(static)


def _restore_c3000(root: Path, checkpoint: Path, static: dict):
    import torch
    from task3.c2.historical import HistoricalC
    from task3.c2.models import Base

    kiki = _kiki_archive(root)
    sys.path[:0] = [str(kiki / "experiments"), str(kiki / "src")]
    import kiki as kiki_package
    import train_joint_latent
    import train_structured_rom

    if Path(kiki_package.__file__).resolve().parent != kiki / "src/kiki":
        raise RuntimeError("The imported KIKI package does not match the verified historical source")
    for module in (train_joint_latent, train_structured_rom):
        if Path(module.__file__).resolve().parent != kiki / "experiments":
            raise RuntimeError(f"Unverified imported historical model module {module.__name__}")
    source = Path(_read(checkpoint.with_name("checkpoint.json"))["source"]) / "model.pt"
    source_state = torch.load(source, map_location="cpu", weights_only=True)

    def array(key):
        return source_state[key].numpy()

    # The unavailable network file only seeded these buffers, already persisted in the frozen source model.
    base_keys = ("mass", "incidence", "delta_star", "coupling", "damping", "rate", "gain")
    base = SimpleNamespace(**{key: array(key) for key in base_keys})
    electrical_keys = ("z_scale", "voltage_star", "load_positions", "load_g", "load_star")
    core = train_joint_latent.JointROM(
        base, array("ports"), array("input_scale"), array("feature_scale"),
        float(array("force_scale")), float(array("load_scale")),
        electrical=tuple(array(key) for key in electrical_keys),
        latent_dim=source_state["c"].shape[1],
    )
    for key in (*base_keys, *electrical_keys, "angle_incidence", "ports", "input_scale",
                "feature_scale", "force_scale", "load_scale", "metric_scale"):
        if not torch.equal(core.state_dict()[key], source_state[key]):
            raise ValueError(f"C3000 restored source buffer differs from its checkpoint: {key}")
    core.load_state_dict(source_state, strict=True)
    core.eval()

    class RestoredC3000(HistoricalC):
        def __init__(self):
            Base.__init__(self, static)
            self.core = train_structured_rom.CFreeROM(core)
            self.core.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)

    return RestoredC3000()


def load_forecaster(root: Path, release: str, name: str, static: dict) -> Callable:
    """Return a CPU predictor restricted to 11 observed frames and constant origin power."""
    root = root.resolve()
    source_name = _source_name(release, name)
    bindings(root, release, name)
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    import task3

    if Path(task3.__file__).resolve().parent != root / "src/task3":
        raise RuntimeError("The imported TASK3 package does not match the verified source")
    import torch
    from task3.protocol import inputs

    folder = _folder(root, release, source_name)
    if release == "campaign1":
        from task3.campaign import tensor
        from task3.models import HankelDMD, LatentModel, Seq2Seq

        with np.load(folder / "normalization.npz", allow_pickle=False) as archive:
            scale, pscale = archive["scale"], archive["pscale"]
        if name == "hankel_dmd":
            model = HankelDMD()
            with np.load(folder / "model.npz", allow_pickle=False) as archive:
                model.a, model.u = archive["a"], archive["u"]
        else:
            config = _read(folder / "result.json")["config"]
            model = Seq2Seq(config) if name == "gru_seq2seq" else LatentModel(name, config)
            state = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True)
            model.load_state_dict(state["model"])
            model.eval()

        def predict(events, seconds, step):
            history, power = inputs(events, scale, pscale)
            if name == "hankel_dmd":
                return model.predict(history, power, scale, pscale, seconds=seconds)
            stride = round(.01 / step)
            with torch.no_grad():
                chunks = [
                    model(tensor(history[i:i + 32]), tensor(power[i:i + 32]), round(seconds / step), step)
                    .numpy()[:, ::stride] * scale
                    for i in range(0, len(events), 32)
                ]
            return np.concatenate(chunks)

    elif name in REFERENCES:
        def predict(events, seconds, step):
            if name == "equilibrium":
                return np.zeros((len(events), round(seconds * 100) + 1, 20))
            origin = np.stack([event["x"][-1] for event in events])
            return np.repeat(origin[:, None], round(seconds * 100) + 1, axis=1)
    elif name == "hankel_dmd":
        with np.load(folder / "model.npz", allow_pickle=False) as archive:
            u, a, b = archive["u"], archive["a"], archive["b"]

        def predict(events, seconds, step):
            scale = static["scale"]
            history, power = inputs(events, scale, static["pscale"])
            z = history.reshape(len(events), -1) @ u
            output = [history[:, -1] * scale]
            forcing = power @ b.T
            for _ in range(round(seconds * 100)):
                z = z @ a.T + forcing
                output.append(z @ u[200:220].T * scale)
            return np.stack(output, axis=1)
    else:
        from task3.c2.models import create
        from task3.c2.safety import forecast

        if name in HISTORICAL:
            from task3.c2.historical import HistoricalFS, M3History

            checkpoint = _historical_checkpoint(root, _read(folder / "registration.json"))
            if name == "C3000":
                model = _restore_c3000(root, checkpoint, static)
            elif name == "m3_pm":
                model = M3History(static, checkpoint)
            else:
                model = HistoricalFS(static, name, checkpoint)
        else:
            if release == "campaign2":
                kind = "sindy_shred_faithful" if name == "sindy_shred_authorized_retry" else source_name
                model = create(kind.removesuffix("_revision1").removesuffix("_extension1"), static)
            elif name == "history_deeponet_p0":
                from task3.extensions.deeponet import HistoryDeepONet

                model = HistoryDeepONet(static)
            elif name == "i_ikno_p0_inferencefix":
                from task3.extensions.iikno_physical_rollout import PhysicalRolloutKoopman

                model = PhysicalRolloutKoopman(static)
            elif name == LASS_LAST_BLOCK_HOLD:
                model = _last_block_lass(static)
            else:
                from task3.extensions.other_models import MODELS

                model = MODELS[name](static)
            state = torch.load(folder / "selected.pt", map_location="cpu", weights_only=True)
            model.load_state_dict(state)

        def predict(events, seconds, step):
            if name == "lass_small_p0" and seconds > 8:
                raise UnsupportedHorizon("The frozen Small LASS P0 supplies eight time blocks, through 8 s")
            prediction, _ = forecast(model, events, dt=step, seconds=seconds)
            return prediction

    def predict_causal(events: list[dict], seconds: int, step: float) -> np.ndarray:
        if not events or any(event["x"].shape != (11, 20) or event["p0"].shape != (10,) for event in events):
            raise ValueError("TASK3 forecasts require exactly 11 observed 20-channel frames and 10 origin powers")
        result = predict(events, seconds, step)
        if result.shape != (len(events), round(seconds * 100) + 1, 20):
            raise ValueError(f"Unexpected TASK3 forecast shape: {result.shape}")
        return result

    return predict_causal
