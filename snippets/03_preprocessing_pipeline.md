# Code 4.3: Mixed-type preprocessing pipeline

```python
# src/thesis_repro/experiments.py

numeric_cols = X_tr.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = [c for c in X_tr.columns if c not in numeric_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)

X_tr_proc = preprocessor.fit_transform(X_tr)
X_val_proc = preprocessor.transform(X_val)
X_test_proc = preprocessor.transform(X_test)
```

This preprocessing pipeline standardizes numerical features and one-hot encodes categorical context variables.
`fit_transform` is called only on training data, while validation/test use `transform`, which preserves leakage-free evaluation.
