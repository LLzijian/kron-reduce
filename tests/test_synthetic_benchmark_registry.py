from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from kron_reduce.evaluation import (
    bootstrap_m0_physical_modes,
    cross_cluster_modal_consensus,
    exact_dmd_modes,
    finite_difference_augmented_modes,
    linearized_passive_network_modes,
    match_dmd_modes,
    physical_model_modes,
)
from kron_reduce.models import M0Model, RunContext
from kron_reduce.synthetic import benchmark_registry_v1

ROOT = Path(__file__).resolve().parents[1]


def _modal_ringdowns(*, samples_per_mode: int = 4) -> tuple[object, list[dict]]:
    benchmark = benchmark_registry_v1()["swing4_b2_modal"]
    time = np.linspace(0.0, 6.0, 601)
    samples: list[dict] = []
    for mode_index in range(len(benchmark.ground_truth_modes())):
        for replicate in range(samples_per_mode):
            initial = benchmark.modal_initial_state(
                [mode_index],
                amplitude=0.012 + 0.001 * replicate,
                phase=0.29 * replicate,
            )
            trajectory = benchmark.simulate(
                event_id=f"mode-{mode_index}-event-{replicate}",
                fault_line=f"mode-{mode_index}-line-{replicate}",
                regime=f"mode_{mode_index}_only",
                time=time,
                delta0=initial[0],
                omega0=initial[1],
            )
            samples.append(benchmark.as_model_sample(trajectory))
    return benchmark, samples


def test_registry_has_locked_modal_and_stability_truth() -> None:
    registry = benchmark_registry_v1()
    assert set(registry) == {
        "swing2_known_separatrix",
        "swing4_b2_modal",
        "swing4_reciprocal_port_r1",
    }
    assert len(registry["swing4_b2_modal"].ground_truth_modes()) == 3
    assert len(registry["swing4_reciprocal_port_r1"].ground_truth_modes()) == 4
    assert registry["swing2_known_separatrix"].stability_barrier == 2.0
    for benchmark in registry.values():
        for mode in benchmark.ground_truth_modes():
            assert mode.frequency_hz > 0.0
            assert mode.decay_rate < 0.0
            assert mode.damping_ratio > 0.0


def test_checksum_locked_registry_matches_generator() -> None:
    locked = json.loads(
        (ROOT / "data" / "synthetic_benchmark_registry_v1" / "registry.json").read_text(
            encoding="utf-8"
        )
    )["benchmarks"]
    generated = benchmark_registry_v1()
    assert set(locked) == set(generated)
    for name, benchmark in generated.items():
        expected = locked[name]
        assert expected["latent_rank"] == benchmark.rank
        assert np.array_equal(expected["mass"], benchmark.mass)
        assert np.array_equal(expected["incidence"], benchmark.incidence)
        assert np.array_equal(expected["coupling"], benchmark.coupling)
        truth = benchmark.ground_truth_modes()
        assert len(expected["ground_truth_modes"]) == len(truth)
        for locked_mode, generated_mode in zip(expected["ground_truth_modes"], truth, strict=True):
            assert np.isclose(locked_mode["frequency_hz"], generated_mode.frequency_hz)
            assert np.isclose(locked_mode["damping_ratio"], generated_mode.damping_ratio)
            assert np.allclose(locked_mode["shape_real"], generated_mode.shape_real)


def test_finite_difference_augmented_linearization_recovers_exact_port_modes() -> None:
    benchmark = benchmark_registry_v1()["swing4_reciprocal_port_r1"]
    hidden = np.zeros(2 * benchmark.rank)

    def field(
        delta: np.ndarray,
        omega: np.ndarray,
        hidden_state: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rank = benchmark.rank
        derivative = benchmark._field(
            (delta, omega, hidden_state[:rank], hidden_state[rank:]),
            np.zeros(len(benchmark.mass)),
        )
        return derivative[0], derivative[1], np.concatenate(derivative[2:])

    numerical, diagnostics = finite_difference_augmented_modes(
        benchmark.mass,
        np.zeros_like(benchmark.mass),
        hidden,
        field,
    )
    matches = match_dmd_modes(
        benchmark.ground_truth_modes(),
        numerical,
        max_relative_frequency_error=1e-5,
    )
    assert len(matches) == 4
    assert max(item["relative_frequency_error"] for item in matches) < 1e-8
    assert max(item["absolute_damping_ratio_error"] for item in matches) < 1e-8
    assert min(item["mac"] for item in matches) > 1.0 - 1e-8
    assert diagnostics["all_poles_stable"] is True


def test_physical_model_modal_registry_uses_fitted_equation() -> None:
    benchmark = benchmark_registry_v1()["swing4_b2_modal"]
    model = M0Model(
        damping=benchmark.node_damping,
        coupling=benchmark.coupling,
    )
    static = {
        "mass": benchmark.mass,
        "incidence": benchmark.incidence,
        "equilibrium_delta": np.zeros_like(benchmark.mass),
        "Pm_equilibrium": np.zeros_like(benchmark.mass),
    }
    modes, diagnostics = physical_model_modes(model, static)
    matches = match_dmd_modes(
        benchmark.ground_truth_modes(), modes, max_relative_frequency_error=1e-10
    )
    assert len(matches) == 3
    assert diagnostics["linearization"] == "analytic_COI_quotient"
    assert diagnostics["all_poles_stable"] is True


def test_passive_benchmark_supply_identity_and_energy_decay() -> None:
    benchmark = benchmark_registry_v1()["swing4_reciprocal_port_r1"]
    initial = benchmark.modal_initial_state([0, 3], amplitude=0.02, phase=0.37)
    state = initial
    field = benchmark._field(state, np.zeros(len(benchmark.mass)))
    edge_speed = benchmark.incidence.T @ state[1]
    angular = 2.0 * np.pi * float(benchmark.modal_frequency_hz)
    analytic_energy_rate = (
        np.sum(benchmark.mass * state[1] * field[1])
        + np.sum(benchmark.coupling * np.sin(benchmark.incidence.T @ state[0]) * edge_speed)
        + np.sum(field[2] * angular**2 * state[2] + state[3] * field[3])
    )
    expected_rate = -float(
        np.sum(benchmark.node_damping * state[1] ** 2)
        + np.sum(benchmark.edge_damping * edge_speed**2)
        + 2.0 * float(benchmark.modal_damping_ratio) * angular * np.sum(state[3] ** 2)
    )
    assert abs(analytic_energy_rate - expected_rate) < 1e-12

    trajectory = benchmark.simulate(
        event_id="passive-ringdown",
        fault_line="synthetic-line",
        regime="mixed_modal",
        time=np.linspace(0.0, 4.0, 1601),
        delta0=initial[0],
        omega0=initial[1],
        hidden_position0=initial[2],
        hidden_velocity0=initial[3],
    )
    energy = benchmark.energy(trajectory)
    assert np.max(np.diff(energy)) < 1e-11
    assert energy[-1] < energy[0]


def test_shared_weak_model_recovers_modes_that_do_not_repeat_per_fault_line() -> None:
    benchmark, samples = _modal_ringdowns()
    modes_by_fault_line = {}
    for sample in samples:
        target = sample["training_context"]["targets"]
        mass_weighted_speed = target["omega"] * np.sqrt(benchmark.mass)
        modes_by_fault_line[sample["training_context"]["metadata"]["fault_line"]] = exact_dmd_modes(
            mass_weighted_speed,
            float(target["t"][1] - target["t"][0]),
            rank=4,
            min_frequency_hz=0.02,
            max_frequency_hz=3.0,
        )
    assert all(modes_by_fault_line.values())
    assert (
        cross_cluster_modal_consensus(
            modes_by_fault_line,
            minimum_support_fraction=0.8,
        )
        == []
    )

    train = [sample for index, sample in enumerate(samples) if index % 4 != 3]
    validation = [sample for index, sample in enumerate(samples) if index % 4 == 3]
    model = M0Model()
    model.fit(
        train,
        validation,
        RunContext(
            horizon_steps=601,
            max_cases=len(train),
            integral_windows=(1, 5, 20, 80),
            ridge_values=(1e-10, 1e-8, 1e-6),
            fit_max_steps=601,
            max_internal_step=0.01,
        ),
    )
    fitted_modes = linearized_passive_network_modes(
        benchmark.mass,
        benchmark.incidence,
        np.zeros_like(benchmark.mass),
        model.damping,
        model.coupling,
    )
    matches = match_dmd_modes(
        benchmark.ground_truth_modes(),
        fitted_modes,
        max_relative_frequency_error=0.1,
    )
    assert len(matches) == 3
    assert np.median([item["relative_frequency_error"] for item in matches]) < 0.02
    assert np.median([item["absolute_damping_ratio_error"] for item in matches]) < 0.02
    assert np.median([item["mac"] for item in matches]) > 0.95
    assert model.diagnostics["effective_rank"] == model.diagnostics["n_parameters"]
    assert model.diagnostics["matrix_condition_number"] < 1e8


def test_fault_line_bootstrap_supports_shared_physical_modes() -> None:
    benchmark, samples = _modal_ringdowns(samples_per_mode=3)
    train = [sample for index, sample in enumerate(samples) if index % 3 != 2]
    validation = [sample for index, sample in enumerate(samples) if index % 3 == 2]
    model = M0Model()
    model.fit(
        train,
        validation,
        RunContext(
            horizon_steps=601,
            max_cases=len(train),
            integral_windows=(1, 5, 20, 80),
            ridge_values=(1e-10,),
            fit_max_steps=601,
        ),
    )
    supported, diagnostics = bootstrap_m0_physical_modes(
        model,
        train,
        draws=40,
        seed=324,
        windows=(1, 5, 20, 80),
        horizon_steps=601,
        max_fit_steps=601,
    )
    assert len(supported) == len(benchmark.ground_truth_modes())
    assert min(mode.support_fraction for mode in supported) >= 0.8
    assert min(mode.median_mac for mode in supported) > 0.95
    assert diagnostics["successful_draw_fraction"] == 1.0
    assert diagnostics["full_rank_draw_fraction"] >= 0.8


def test_known_separatrix_energy_is_exact() -> None:
    benchmark = benchmark_registry_v1()["swing2_known_separatrix"]
    time = np.asarray([0.0, 0.01])
    at_equilibrium = benchmark.simulate(
        event_id="equilibrium",
        fault_line="synthetic-line",
        regime="equilibrium",
        time=time,
        delta0=np.zeros(2),
        omega0=np.zeros(2),
    )
    at_saddle = benchmark.simulate(
        event_id="saddle",
        fault_line="synthetic-line",
        regime="separatrix",
        time=time,
        delta0=np.asarray([0.5 * np.pi, -0.5 * np.pi]),
        omega0=np.zeros(2),
    )
    assert benchmark.energy(at_equilibrium)[0] == 0.0
    assert np.isclose(benchmark.energy(at_saddle)[0], benchmark.stability_barrier)
