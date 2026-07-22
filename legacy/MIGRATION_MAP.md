# Migration map

| Historical concept | Clean module |
|---|---|
| 3.10 `support.data_protocol` | `kron_reduce.data` |
| 3.10 `our_methods.m0` | `kron_reduce.models.m0` |
| 3.10 `our_methods.m2a` | `kron_reduce.models.m2a` |
| 3.10 `our_methods.m2a_lyap` | `kron_reduce.models.energy` + energy gate |
| 3.10 PINN/LSTM/comparisons | `kron_reduce.models.pinn/lstm/operators/classifiers` |
| 3.11 calibration | exploratory legacy snapshot only |
| 3.37r1 prospective cohort | `kron_reduce.evaluation.cohorts` |
| 3.37r1 physics/energy audits | `kron_reduce.evaluation.physics/energy` |
