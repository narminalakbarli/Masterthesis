# Code 4.2: Spatial-temporal feature engineering

```python
# src/thesis_repro/data.py

def add_spatiotemporal_and_synthetic_features(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    out = df.copy()

    seconds = out["Time"].to_numpy()
    out["hour"] = ((seconds // 3600) % 24).astype(int)
    out["day"] = ((seconds // (3600 * 24)) % 7).astype(int)
    out = out.sort_values("Time").reset_index(drop=True)
    out["inter_txn_seconds"] = out["Time"].diff().fillna(out["Time"].median()).clip(lower=0)
    out["amount_log"] = np.log1p(out["Amount"])

    out["card_present"] = (out["channel"] != "online").astype(int)
    out["region_changed"] = (out["merchant_region"] != out["home_region"]).astype(int)
    return out
```

This snippet adds the core engineered variables used in enhanced experiments: temporal cadence (`hour`, `day`, `inter_txn_seconds`), amount scaling (`amount_log`), and contextual risk signals (`card_present`, `region_changed`).
These features model behavioral and geo-context patterns beyond raw PCA components.
