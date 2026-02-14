# Code 4.5: Training, threshold tuning, and evaluation

```python
# src/thesis_repro/experiments.py

model.fit(X_fit, y_fit)

val_score = model.predict_proba(X_val_proc)[:, 1]
thr = _best_threshold(y_val, val_score)

test_score = model.predict_proba(X_test_proc)[:, 1]
pred = (test_score >= thr).astype(int)

result = EvalResult(
    recall=recall_score(y_test, pred, zero_division=0),
    precision=precision_score(y_test, pred, zero_division=0),
    f1=f1_score(y_test, pred, zero_division=0),
    f2=fbeta_score(y_test, pred, beta=2, zero_division=0),
    auc_roc=roc_auc_score(y_test, test_score),
    auc_pr=average_precision_score(y_test, test_score),
)
```

This fragment shows the complete supervised evaluation loop: fit, tune threshold on validation set, and report metrics on unseen test data.
Threshold optimization on validation (instead of test) keeps performance estimates unbiased.
