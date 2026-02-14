# Chapter 3 — Research Questions and Data

## Techniques, methods, and data experiments described

- **Research-question-to-feature mapping**:
  - spatial-temporal features tied to fraud-pattern hypotheses,
  - context attributes tied to channel/geographic risk behavior.
- **Data construction**:
  - primary ULB/Kaggle dataset,
  - optional contextual augmentation corpus,
  - derived engineered variables for temporal and regional behavior.
- **Class imbalance protocol**: explicit handling of extreme fraud rarity.
- **Split/evaluation protocol**:
  - chronological split to avoid temporal leakage,
  - train/validation/test separation,
  - exploratory visual checks before modeling.

## Implementation check

- Implemented:
  - robust dataset loading path (local/Kaggle/public mirror/synthetic fallback),
  - context-prior alignment and synthetic fill-in,
  - chronological split with validation subset,
  - engineered temporal/context features used by enhanced experiments.
- Snippet coverage exists for split, feature engineering, and preprocessing and aligns with this chapter.
- No additional code gap detected for Chapter 3 requirements.
