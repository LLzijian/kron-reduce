import numpy as np

from kron_reduce.evaluation.machine_local_gaussian_oracle import expected_risks


def test_expected_risks_separate_full_variance_reset_bias_horizons_and_channels():
    variance = np.empty((800, 20))
    variance[:, :10] = 4.0
    variance[:, 10:] = 36.0
    bias = np.zeros((64, 801, 30))
    bias[:32, 1:, :10] = 2.0
    bias[:32, 1:, 10:20] = -3.0
    bias[32:48, 1:, :10] = 4.0
    bias[32:48, 1:, 10:20] = -6.0
    bias[48:, 1:, :10] = 6.0
    bias[48:, 1:, 10:20] = -9.0
    scale = np.r_[np.full(10, 2.0), np.full(10, 3.0)]

    result = expected_risks(variance, bias, scale)

    assert set(result) == {"train", "val", "test"}
    role_bias = {"train": 1.0, "val": 4.0, "test": 9.0}
    for role, extra in role_bias.items():
        row = result[role]
        for seconds in (1, 3, 8):
            assert row[f"full_nmse_{seconds}s_q"] == 1.0
            assert row[f"full_nmse_{seconds}s_v"] == 4.0
            assert row[f"full_nmse_{seconds}s_all"] == 2.5
            assert row[f"reset_extra_nmse_{seconds}s_q"] == extra
            assert row[f"reset_extra_nmse_{seconds}s_v"] == extra
            assert row[f"reset_extra_nmse_{seconds}s_all"] == extra
            assert row[f"reset_nmse_{seconds}s_q"] == 1.0+extra
            assert row[f"reset_nmse_{seconds}s_v"] == 4.0+extra
            assert row[f"reset_nmse_{seconds}s_all"] == 2.5+extra
        assert row["full_main_nmse"] == 2.5
        assert row["reset_extra_main_nmse"] == extra
        assert row["reset_main_nmse"] == 2.5+extra
