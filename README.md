# Masterthesis
Machine Learning Methods in Credit Card Fraud Detection

This repository contains a reproducible Python pipeline aligned with the thesis methodology.

## What is implemented

- Baseline fraud models on Kaggle credit-card-fraud style data.
- Thesis-inspired spatial-temporal feature engineering (`hour`, `day`, inter-transaction time, region distance/change).
- Synthetic contextual variables (channel, same-state, card-present, merchant category) when original fields are unavailable.
- Class imbalance handling with SMOTE (applied leakage-safe on training subset only).
- Model comparison across logistic regression, random forest, XGBoost, and a stacking ensemble.
- Rule-based benchmark and cost-benefit comparison.
- Comparison of **baseline feature set** vs **enhanced feature set**.

## 1) Environment setup

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
```

## 2) Extract chapter notes from thesis

```bash
python scripts/extract_thesis_chapters.py
```

Generated output:

- `docs/thesis_chapter_notes.md`

## 3) Run reproducibility pipeline

```bash
python -m src.thesis_repro.run_pipeline --sample-size 80000
```

Generated outputs:

- `outputs/model_results.csv`
- `outputs/results_summary.md`

## 4) Data source behavior

The pipeline first attempts to download `creditcard.csv` from TensorFlow's public mirror of the commonly used Kaggle credit-card-fraud dataset.
If downloading is unavailable, it generates a synthetic fallback dataset with similarly severe class imbalance.

## 5) How this mirrors the thesis

- Chapter 3: Kaggle-style base data + synthetic operational variables.
- Chapter 4: spatial-temporal feature engineering and SMOTE balancing.
- Chapters 5-6: model-family comparison (recall/precision/F1/F2/AUC) + rule-based benchmark and cost modeling.
- Chapter 7: summary-ready outputs in markdown/CSV for interpretation and limitations discussion.
