# Code 4.1: Time-aware split with leakage prevention

```python
# src/thesis_repro/experiments.py

def _prepare_splits(df: pd.DataFrame, features: list[str]):
    df = df.sort_values("Time").reset_index(drop=True)
    y = df["Class"].to_numpy()
    split_idx = int(len(df) * 0.8)

    X_train_all = df.iloc[:split_idx][features]
    y_train_all = y[:split_idx]
    X_test = df.iloc[split_idx:][features]
    y_test = y[split_idx:]

    val_split = int(len(X_train_all) * 0.8)
    X_tr = X_train_all.iloc[:val_split]
    y_tr = y_train_all[:val_split]
    X_val = X_train_all.iloc[val_split:]
    y_val = y_train_all[val_split:]
```

This fragment performs a chronological split (first train/validation, then test), preventing future transactions from leaking into the training phase.
That design is important for fraud detection because production scoring is always forward in time.
