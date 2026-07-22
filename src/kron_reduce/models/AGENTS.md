# AGENTS.md

Models implement the common dynamics interface. They may consume only the
registered `model_inputs` blocks. Test data must never enter `fit`. Stability
classifiers are supplementary and must not masquerade as dynamics models.
