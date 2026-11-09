# AGENTS.md

This package owns the registry, Git state, data verification, immutable run
creation, audits and workflow gates. Numerical models, feature libraries and
stability classifiers belong in `src/kron_reduce/`.

- Keep `loop plan`, `loop status` and `loop next` read-only. A preview uses
  registry metadata and saved workflow state.
- Use the same per-workflow lock for approvals, manual evidence, built-in
  checks and experiment dispatch. Hold it from the state check through the
  evidence write. Use the shared transition and evidence validators when
  advancing and when replaying history during audit.
- Check the registered stage, workflow approval and scientific authority
  before source verification or dispatch. Dedicated one-use audits follow
  their registered CLI and bounded read procedure. The generic runner must
  reject those registrations before generic data verification.
- Assign a new run ID to every experiment attempt. Retain failure evidence and
  keep spent one-use authorizations closed.
- Validate harness changes with `python -m project_harness audit` and the
  harness regression tests. Run source checksum verification as a separate
  authorized `data-check`.
