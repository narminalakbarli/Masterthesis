# Code 4.4: Class imbalance handling with SMOTE

```python
# src/thesis_repro/experiments.py

def _sampler(name: str, random_state: int = 42):
    if name == "smote":
        return SMOTE(sampling_strategy=0.15, random_state=random_state, k_neighbors=3)
    if name == "smoteenn":
        return SMOTEENN(random_state=random_state, sampling_strategy=0.15)
    if name == "cost_sensitive":
        return None

sampler = _sampler(balancing, random_state=random_state)
X_fit, y_fit = X_tr_proc, y_tr
if sampler is not None:
    X_fit, y_fit = sampler.fit_resample(X_tr_proc, y_tr)
```

The code applies balancing only to the training fold before model fitting.
Using `random_state` keeps resampling deterministic and reproducible across runs.
The same function supports multiple balancing strategies used in the thesis experiments.
