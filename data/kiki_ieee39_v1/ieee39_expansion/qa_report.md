# Private dataset QA

PASS: 230/240 accepted; discarded 10.

Outcomes: {'stable': 192, 'unstable': 38}. Dynamic states: 220.

Discard reasons: {'ValueError: numerical_TDS_failure': 9, 'ValueError: unconfirmed_physical_outcome': 1}.

Stable raw Pm peak-to-peak max: 4.4336801 system pu.
Maximum COI residual: 1.17e-13.

Clearing-state PCA fractions (standardized, descriptive only): [0.3779195231275069, 0.13405226830331218, 0.11756713370440068].
Nearest-neighbor standardized distance min/median: 0.115602/0.749731.

Split counts: {'train': 142, 'validation': 43, 'test': 45}. Bus/line groups disjoint.
Pilot sufficient for M2/M3-Pm: True (train groups 9, validation groups 3).

Checks: restored-system signature, finite arrays, strict uniform time, no repeated/frozen states, core shapes, state mapping, uniform physical outcome, normal solver at LOS, COI, group isolation.
Holdout accessed only by mechanical data QA/coverage, not by any model or selection routine.
