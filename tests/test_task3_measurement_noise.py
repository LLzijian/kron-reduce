import numpy as np

from kron_reduce.data.task3_measurement_noise import (
    draw_task3_noise,
    generate_noise_bank,
    inject_raw_history,
)
from kron_reduce.physics.coordinates import coi_project


def test_noise_draw_is_deterministic_and_clean_is_zero():
    first = draw_task3_noise("val", "gauss_dse_dynamic_pm", 3, "ieee39_postfault/raw")
    second = draw_task3_noise("val", "gauss_dse_dynamic_pm", 3, "ieee39_postfault/raw")
    np.testing.assert_array_equal(first.eps_delta, second.eps_delta)
    np.testing.assert_array_equal(first.eps_omega, second.eps_omega)
    np.testing.assert_array_equal(first.eps_pm, second.eps_pm)
    clean = draw_task3_noise("val", "clean", 0, "ieee39_postfault/raw")
    assert clean.clean
    assert clean.endpoint_index is None


def test_noise_bank_digest_uses_sorted_paired_draws():
    ids = ["ieee39_expansion/raw_b", "ieee39_postfault/raw_a"]
    first = generate_noise_bank("test", "nongauss_laplace", 0, ids)
    second = generate_noise_bank("test", "nongauss_laplace", 0, reversed(ids))
    assert first.sha256 == second.sha256
    assert list(first.draws) == sorted(ids)


def test_injection_changes_only_raw_history_before_projection():
    mass = np.arange(1, 11, dtype=np.float64)
    raw_x = np.arange(16 * 20, dtype=np.float64).reshape(16, 20)
    raw_p0 = np.arange(10, dtype=np.float64)
    draw = draw_task3_noise("blind_v2", "gauss_dse_dynamic", 1,
                            "kiki_ieee39_blind_v2/raw_case")
    noisy_x, noisy_p0 = inject_raw_history(raw_x, raw_p0, mass, draw)
    assert not np.shares_memory(noisy_x, raw_x)
    np.testing.assert_array_equal(noisy_x[11:], raw_x[11:])
    np.testing.assert_array_equal(noisy_x[:11, :10],
                                  raw_x[:11, :10] + coi_project(draw.eps_delta, mass))
    np.testing.assert_array_equal(noisy_x[:11, 10:], raw_x[:11, 10:] + draw.eps_omega)
    np.testing.assert_array_equal(noisy_p0, raw_p0)

    clean = draw_task3_noise("blind_v2", "clean", 0, "kiki_ieee39_blind_v2/raw_case")
    clean_x, clean_p0 = inject_raw_history(raw_x, raw_p0, mass, clean)
    np.testing.assert_array_equal(clean_x, raw_x)
    np.testing.assert_array_equal(clean_p0, raw_p0)


def test_outlier_endpoint_records_selected_history_frame():
    draw = draw_task3_noise("val", "outlier_endpoint_10sigma", 4,
                            "ieee39_holdout_supplement/raw_case")
    assert draw.endpoint_index in (0, 10)

