# IEEE39 post-clear canonical dataset v1

Revision: `ieee39_postclear_canonical_v1`

Content revision: `postclear_provenance_v2`

Harness contract: `causal_windowed_v2`

This directory is the canonicalized, exact-clearing, variable-length data body
for the historical IEEE39 high-order and Protocol A/B comparison families. It
is **not** the current fixed-T0 Protocol C/D mainline dataset unless the project
registry is explicitly changed.

## Safe use

- Open the HDF5 lazily through `discovery/3.2-kron/harness`.
- Select one declared view and its bound line-grouped split.
- Convert an event through an `InputContract` and `ModelDataRequest`.
- M0 rollout receives only `delta[0]`, `omega[0]`, equilibrium/physical priors,
  and stepwise `Pm(t)`.
- Prefix models request a fixed `n_steps` or `horizon_s`; short events fail
  closed by default and must never be padded into model-visible inputs. For
  safety/stability metrics, use fixed `n_steps`; a physical-time horizon is
  rejected unless an explicit leakage-audited resampling policy is introduced.
- `Pe`, voltage, labels, event time, trajectory length, masks, and disturbance
  ranks are offline/training-controller data, not causal model inputs.

The HDF5 is intentionally not duplicated in Git. `checksums.sha256`, source
SHA256 fields, event content signatures, and the Git-sealed source archive
manifest provide reproducibility.
