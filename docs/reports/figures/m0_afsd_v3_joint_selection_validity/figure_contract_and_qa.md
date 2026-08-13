# Figure 1 contract and QA

## Contract

- Core conclusion: v2 supplies conditional fixed-support rollout evidence for dissipative-memory closure, but the current F1 source-by-excitation design is not eligible for a joint adaptive-order claim.
- Archetype: schematic-led composite with one source-coverage heatmap and one quantitative rollout panel.
- Hero panel: panel b, because the source-by-excitation overlap failure determines whether the model lattice may be fitted.
- Statistical unit: fault-line cluster for panel b; panel c is a descriptive aggregate over 34 F1 validation events and is not presented as v3 confirmatory inference.
- Claim boundary: F1, seed324, observed `Pm(t)`, no test, no F2--F5, no autonomous mechanical-power prediction.

## Source data

- `source_excitation_fault_lines.csv`: immutable snapshot of v3 `source_excitation_table.csv`.
- `v2_conditional_rollout_metrics.csv`: immutable snapshot of v2 `aggregate_metrics.csv`, restricted to C3-to-C3 B2 and SS-DM at H=100/500/1000.
- The plotting script asserts the 95-event total, the decisive 11-versus-1 overlap cell, and both 10 s headline RMSE values before rendering.

## Visual QA

- [x] Panel letters are ordered and aligned.
- [x] All labels are legible at the final 183 mm width.
- [x] Color is not the sole carrier of meaning: every heatmap cell is numerically labelled and both rollout series use distinct markers.
- [x] Red outlines consistently denote cells below the preregistered five-cluster minimum; the orange outline denotes the only cross-source overlap cell.
- [x] Panel c explicitly separates trajectory improvement from failed absolute modal gates.
- [x] No error bars are implied where no corresponding horizon-wise cluster interval was registered.
- [x] The figure contains no test result and no physical-line interpretation.
- [x] PNG was inspected at original resolution after final rendering; no text overlap, clipping, or illegible annotation remained.

## Exports

- Editable/vector: `fig1_afsd_v3_evidence.svg`, `fig1_afsd_v3_evidence.pdf`.
- Raster: `fig1_afsd_v3_evidence.png`, `fig1_afsd_v3_evidence.tiff` at 600 dpi; TIFF uses LZW compression.
- Fonts are kept as text in SVG and embedded as TrueType in PDF.

