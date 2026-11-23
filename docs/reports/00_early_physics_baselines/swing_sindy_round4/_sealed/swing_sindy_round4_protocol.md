# Round 4: direct SINDy in a second-order swing shell

## Question

This validation-only ablation tests whether a mostly original continuous-time
SINDy regression becomes competitive once the exact second-order swing
kinematics are retained.  It uses the conditional observed mechanical-power
schedule and therefore remains a non-causal upper-bound experiment.

## Registered model

The model enforces

\[
\dot\delta=\omega,\qquad
m\dot\omega=\Pi_f\Theta(q,\omega,\Delta P_m)\Xi,
\]

where \(q\) is the COI-projected angle displacement and \(\Pi_f\) projects the
learned force to zero total force.  \(\Theta\) contains every first- and
second-degree monomial of \((q,\omega,\Delta P_m)\), with no constant term so
the declared equilibrium is anchored.  Only the generalized force is learned.

The acceleration target is obtained by a second-order numerical derivative of
the measured speed.  Coefficients are solved once by normalized STRidge for
the registered ridge and threshold grid.  Candidates are shortlisted by
validation derivative error, then selected by H=100 closed-loop standardized
angle-and-frequency loss.  This is not epoch training.

## Evidence and gate

- Seed 324, all five registered fault-line folds.
- Training and model selection use train/validation only.
- Observed Pm is linearly interpolated at every RK4 stage.
- H=100 includes only trajectories with at least 100 real samples.
- Test remains locked unless pooled omega R2 is greater than 0.9, mean RMSE
  improves on frozen M0 by at least 2%, rollout failure rate is zero, and the
  fault-line cluster bootstrap upper 95% endpoint is below zero.
- No Pm ODE is fitted in this round.
