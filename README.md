# Masterthesis
Machine Learning Methods in Credit Card Fraud Detection

This repository contains a reproducible Python pipeline aligned with the thesis methodology and includes a paper-vs-code experiment coverage audit.

## What is implemented

- Baseline fraud models on Kaggle credit-card-fraud style data.
- Thesis-inspired spatial-temporal feature engineering (`hour`, `day`, inter-transaction time, region distance/change).
- Synthetic contextual variables (channel, same-state, card-present, merchant category) when original fields are unavailable.
- Class imbalance studies across: none, random oversampling, SMOTE, SMOTE-ENN, random undersampling, cost-sensitive learning.
- Model-family comparison across logistic regression, decision tree, random forest, XGBoost, MLP, and a stacking ensemble.
- Rule-based benchmark and cost-benefit comparison.
- Plot generation and a paper target validation report.

## 1) Environment setup

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
```

## 2) Download thesis data (Kaggle first)

The thesis dataset is publicly available on Kaggle as:

- `mlg-ulb/creditcardfraud`

Run:

```bash
python scripts/download_kaggle_data.py --out-dir data
```

Behavior:
- If Kaggle API credentials are configured (`~/.kaggle/kaggle.json` or env vars), the script downloads from Kaggle.
- If credentials are missing, it falls back to a public mirror of the same `creditcard.csv` file.

## 3) Extract chapter notes from thesis

```bash
python scripts/extract_thesis_chapters.py
```

Generated output:

- `docs/thesis_chapter_notes.md`

## 4) Run reproducibility pipeline

```bash
python -m src.thesis_repro.run_pipeline --sample-size 60000
```

Generated outputs:

- `outputs/model_results.csv`
- `outputs/results_summary.md`
- `outputs/paper_target_validation.csv`
- `outputs/paper_experiment_inventory.csv`
- `outputs/plots/spatiotemporal_recall.png`
- `outputs/plots/spatiotemporal_f1.png`
- `outputs/plots/balancing_heatmap_f1.png`
- `outputs/plots/benchmark_savings.png`

## 5) Data source behavior in code

`src/thesis_repro/data.py` uses this order:
1. Local `data/creditcard.csv` (if already present)
2. Kaggle dataset download (`mlg-ulb/creditcardfraud`)
3. Public mirror download
4. Synthetic fallback generation

## 6) How this mirrors the thesis

- Chapter 3: Kaggle-style base data + synthetic operational variables.
- Chapter 4: spatial-temporal feature engineering and balancing strategies.
- Chapters 5-6: model-family comparison (recall/precision/F1/AUC), balancing comparison, and rule-based benchmark with cost modeling.
- Chapter 7: results summary + explicit validation of paper target values and coverage status.

## 7) Current replication gaps

The thesis also discusses sequential deep models (LSTM/attention/CNN), hybrid XGBoost-LSTM, and SHAP/attention interpretability analysis.
Those are tracked in `outputs/paper_experiment_inventory.csv` as not-yet-replicated and can be added in a next step.
