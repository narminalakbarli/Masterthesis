# Chapter 1 — Introduction

## Techniques, methods, and experiment framing described

- **Domain framing**: growth of card payments and fraud losses as motivation for real-time risk scoring.
- **Fraud typology framing**: card-not-present fraud, account takeover, synthetic identity fraud, and merchant/refund abuse are introduced as distinct fraud mechanisms.
- **Regulatory/operational framing**: pressure for explainable and auditable systems is established.
- **Method families previewed**:
  - Classical ML (logistic regression, decision trees, random forests, gradient boosting).
  - Deep learning (LSTM/CNN/attention-style models).
  - Hybrid and anomaly-detection paths (autoencoder and isolation-style methods).
- **Core experiment motivation**:
  - evaluate whether richer spatial-temporal/context features improve detection quality,
  - compare balancing strategies for highly imbalanced labels,
  - benchmark data-driven approaches vs rule-based fraud screening.

## Implementation check in this repository

- The repository implements this chapter's planned method families and framing through configurable studies and model banks:
  - classical models, deep proxies, anomaly models, and ensembles are in `src/thesis_repro/experiments.py`.
  - spatial-temporal/context feature engineering is in `src/thesis_repro/data.py`.
  - rule-based benchmark + cost comparison are in the experiment runner.
- No direct code gaps were found for Chapter 1 framing after this update.
