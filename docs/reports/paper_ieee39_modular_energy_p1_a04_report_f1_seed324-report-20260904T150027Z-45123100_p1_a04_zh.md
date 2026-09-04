# IEEE39 模块化能量等值 P1-A04 诊断报告

## 结论

- 科学结论：`ACCEPT`
- 模型晋级：`KEEP_BASELINE`
- 边界：Accepted P1 diagnosis on F1 fit/selection/internal-audit evidence only. The candidate model is not promoted; there is no test/F2-F5 performance, identified-frequency, stability-certificate, or Level 6 claim.

## 三个问题

### Q1_modules_change_prediction_or_degenerate_to_B2

The jointly deleted model is statistically worse in at least one registered cell, so the whole candidate does not collapse to B2.  However, no named module has independent predictive support; the observed module-status counts are {'UNIDENTIFIABLE': 7}. The candidate therefore cannot be promoted as a validated module-union model.

边界：Joint-B2 refutation is not individual-module support. Nonzero coefficients are not independent contributions; zero coefficients and failed profile gates remain negative evidence.

### Q2_why_long_horizon_speed_error_is_large

Numerical integration and COI preservation are not the cause. A causal mechanical-power closure improves the registered point error, but 10 s pooled speed R2 remains -0.225783. The weak local objective improves only 0.673%, all three closed-loop calibrations are rejected after solver failure, and the overall fault-line interval is unavailable. The evidence identifies an unresolved mixture of forcing/dynamic closure and objective-calibration mismatch, not one proven cause.

边界：Point improvements do not replace a missing overall confidence interval. The observed-Pm arm is noncausal and parameter-confounded, and the causal closure has not achieved positive 10 s speed R2.

### Q3_modal_audit_shared_poles_and_local_observability

The implementation follows the v8 semantics: poles come from the one frozen shared equation and trajectory DMD only measures excitation-local observability. On IEEE39, none of the 9 reportable poles is observed; view statuses are {'UNDETERMINED': 9, 'UNSUPPORTED': 9}. No identified-frequency claim is available.

边界：Internal equation poles are not identified IEEE39 oscillation frequencies unless an eligible excitation view supplies the preregistered support.

## 下一步

进入 P2 公平基线与评价合同冻结；当前候选不作为已接受部署模型。
