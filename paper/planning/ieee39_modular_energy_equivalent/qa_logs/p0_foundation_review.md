# P0 foundation review

Review date: 2026-08-28
Review type: internal foundation-file audit before algorithm development

| Dimension | Score / 10 | Finding |
|---|---:|---|
| Scope and exclusions | 9.0 | IEEE39, synthetic evidence role, single-model rule and publication-only metadata are explicit. |
| Research canon | 8.5 | Current accepted, rejected and unresolved facts are separated. |
| Evidence table | 8.5 | Claims carry evidence strength and misuse risks; failed-module evidence was added. |
| Argument map | 8.0 | Main contribution and dependency order are explicit; final results remain hypotheses. |
| Section contracts | 8.0 | Each section has purpose, evidence need and refusal boundary. |
| Terminology and style | 8.5 | Naming gates and plain-language reporting rules are fixed. |
| State and version control | 8.0 | Phase, locked claims and next transition are machine-readable. |
| Claim safety | 9.0 | Level 6, modal and causal-input overclaims are explicitly blocked. |

Mean foundation score: **8.44 / 10**.

The foundation is sufficient to begin P1, not to begin a manuscript results section. The main unresolved items are the causal source of long-horizon speed error, scale-aware module contribution, a fair causal baseline contract, and a registered independent IEEE39 boundary-audit dataset.

Validation at freeze:

- `project_harness audit`: PASS, 0 errors.
- `project_harness data-check`: PASS, canonical 315 events; synthetic registry PASS.
- JSON parse: PASS.
- Ruff on changed Python/test files: PASS.
- Full pytest suite: 196 passed.
