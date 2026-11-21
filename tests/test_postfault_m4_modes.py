import json

import numpy as np
import pytest

from kron_reduce.evaluation.postfault_m4_modes import evaluate_modes
from kron_reduce.models.machine_local_topology import TopologyMemoryModel
from kron_reduce.models.nonlinear_latent import NonlinearPassiveClosure
from kron_reduce.models.unified_discovery_m3 import UnifiedDiscoveryModel


def test_two_area_ringdown_and_unidentified_event_coverage():
    # A balanced two-area mode has network curvature 10*k_cross on this complete graph.
    mass = np.ones(10)
    partition = np.repeat([1., -1.], 5)
    cross, damping, gain, tau = .94, .45, .55, .60
    edges = np.array([cross if partition[i] != partition[j] else 3.1 for i in range(10) for j in range(i+1, 10)])
    empty = np.empty((0, 0))
    backbone = NonlinearPassiveClosure(mass, edges, damping*(np.eye(10)-np.ones((10, 10))/10.),
                                       np.empty((10, 0)), empty, empty)
    options = {"port": np.full(10, gain), "tau": np.full(10, tau)}
    hard = TopologyMemoryModel(backbone, topology=False, **options)
    hard.set_angle_damping(2/3., spatial_mix=1.)
    m3 = UnifiedDiscoveryModel(backbone, support=("G2",), **options)
    pole = next(w for w in np.linalg.eigvals([[0., 1., 0.], [-10.*cross, -damping, -gain],
                                             [0., gain, -1./tau]]) if w.imag > 0.)
    time = np.arange(801)*.01
    drift = np.linspace(-.12, .08, 10)
    speed = np.exp(pole*time[:, None]).real*partition+drift
    base = {"role": "val", "label": "stable", "mass": mass, "equilibrium": np.zeros(10)}
    events = [base | {"id": "oscillation", "target_omega": speed, "forecast_time": time},
              base | {"id": "quiescent", "x": np.zeros((811, 20))},
              base | {"id": "short", "target_omega": speed[:-1]}]
    report = evaluate_modes({"m3": m3, "hard_r": hard}, events)
    json.dumps(report, allow_nan=False)
    assert report["summary"]["status_counts"] == {
        "identified": 1, "no_stabilized_mode": 1, "incomplete_window": 1,
        "nonfinite_window": 0, "identification_failed": 0}
    assert report["summary"]["distinct_static_references"] == 1
    observed, = report["events"]["oscillation"]["identification"]["retained"]
    assert observed["frequency_hz"] == pytest.approx(pole.imag/(2*np.pi), abs=1e-9)
    assert observed["damping_ratio"] == pytest.approx(-pole.real/abs(pole), abs=1e-9)
    for name in ("hard_r", "m3"):
        linearization = report["references"]["0"][name]
        assert len(linearization["matrix"]) == len(linearization["all_quotient_poles_per_s"]) == 28
        assert linearization["autodiff_max_abs_error"] < 1e-10
        match = report["events"]["oscillation"]["comparisons"][name]["dominant"]
        assert match["strong"] and match["mac"] > .9999999
        assert abs(match["frequency_error_hz"]) < 1e-9 and abs(match["damping_ratio_error"]) < 1e-9
        assert report["summary"]["models"][name]["dominant_strong_fraction_all_events"] == 1/3
