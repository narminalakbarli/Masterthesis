# Code 4.6: Full experiment configuration and seeded execution

```python
# src/thesis_repro/run_pipeline.py

cfg = _load_config(args.config)
results = run(
    sample_size=args.sample_size,   # optional CLI override
    random_state=args.seed,         # optional CLI override
    config=cfg,
)
```

```json
// config/experiment_config.json (excerpt)
{
  "sample_size": 80000,
  "random_state": 42,
  "splits": {"test_size": 0.2, "validation_size_within_train": 0.2},
  "threshold": {"min": 0.1, "max": 0.9, "num": 17, "beta": 2.0},
  "costs": {"fraud_cost": 200.0, "review_cost": 2.0},
  "samplers": {"smote": {"sampling_strategy": 0.15, "k_neighbors": 3}},
  "models": {"RandomForest": {"n_estimators": 220, "max_depth": 14}},
  "studies": {"balancing_methods": {"models": ["LogisticRegression", "XGBoost"]}}
}
```

This setup makes the pipeline fully configuration-driven: experiment matrix, model hyperparameters, balancing settings, threshold tuning, costs, and outputs are all tracked in one versioned file.
CLI arguments are used only as explicit overrides for ad-hoc runs.
