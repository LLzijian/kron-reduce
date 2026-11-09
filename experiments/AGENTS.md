# AGENTS.md

Experiment cards declare falsifiable hypotheses, data revision, split, input
contract, methods, selection budget, metrics and stopping gates. Implementation
belongs in `src/kron_reduce/`.

Bind each executable experiment to its registered workflow, stage and entry
point. Use `loop plan` to review the stage and evidence contract, then confirm
the authorized budget and data access in `configs/research_policy.json`.
Fresh `test` access requires a registered independent dataset, frozen
selection and explicit approval through the test gate. Any retrospective
exception follows its own one-use policy and protocol.
