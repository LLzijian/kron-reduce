# Private dataset QA

PASS: 361/384 accepted; discarded 23.

Outcomes: {'stable': 305, 'unstable': 56}. Dynamic states: 220.

Discard reasons: {'ValueError: numerical_TDS_failure': 22, 'ValueError: unconfirmed_physical_outcome': 1}.

Stable raw Pm peak-to-peak max: 4.6683446 system pu.
Maximum COI residual: 1.19e-13.

Clearing-state PCA fractions (standardized, descriptive only): [0.49881223531677304, 0.15411813373747021, 0.11959848455828428].
Nearest-neighbor standardized distance min/median: 0.00983044/0.659099.

Split counts: {'train': 227, 'validation': 74, 'test': 60}. Bus/line groups disjoint.
Pilot sufficient for M2/M3-Pm: True (train groups 15, validation groups 5).

Checks: restored-system signature, finite arrays, strict uniform time, no repeated/frozen states, core shapes, state mapping, uniform physical outcome, normal solver at LOS, COI, group isolation.
Holdout accessed only by mechanical data QA/coverage, not by any model or selection routine.
