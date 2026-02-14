# Chapter 5 — Results and Discussion

## Experiments described

- **Spatial-temporal feature impact**: baseline (`V* + Time + Amount`) vs enhanced feature set.
- **Balancing-method effectiveness** across multiple samplers.
- **Model comparison** across classical, boosting, and neural/deep variants.
- **Cost-benefit analysis** for operational fraud handling.
- **Diagnostic analysis** through ROC and confusion-matrix-style interpretation.

## Implementation check

- Implemented and produced by pipeline outputs:
  - `model_results.csv` with per-run metrics,
  - `results_summary.md` with top-performing runs,
  - plots for feature-impact and balancing comparisons,
  - benchmark savings figure for operational value.
- Added explicit model-family study in this update so Chapter 5 comparison logic is traceable as its own experiment block.
