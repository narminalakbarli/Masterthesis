# Chapter 2 — Literature Review

## Techniques and methods described

- **Rule-based detection systems**: deterministic thresholds/if-else logic as legacy baseline.
- **Supervised ML methods**: logistic regression, decision tree, random forest, XGBoost/CatBoost.
- **Imbalance mitigation**:
  - random over/under-sampling,
  - SMOTE-family methods,
  - ADASYN,
  - hybrid sampling variants.
- **Deep-learning lines**: feed-forward neural nets and sequence-focused models (LSTM/CNN/attention concepts).
- **Unsupervised/semi-supervised lines**: autoencoder-style anomaly scoring, isolation forest, one-class approaches.
- **Spatial-temporal context**: temporal cadence and context-aware features as key gaps in prior work.

## Experiment implications from Chapter 2

- Compare model families under a consistent evaluation protocol.
- Include balancing-method ablations to quantify gains and trade-offs.
- Include anomaly-method results, not only supervised classifiers.
- Track not only recall but also precision/F1/AUC and operational-cost perspectives.

## Implementation check

- All above families are now explicitly represented in configurable studies:
  - `balancing_methods`, `model_family_comparison`, `ensemble_comparison`, and `anomaly_methods`.
- The previous pipeline already had most models/samplers, but **did not expose dedicated study tracks for model-family, ensemble-only, and anomaly-only comparisons**.
- This gap has been implemented in this change by adding those study blocks and execution paths.
