# Code 4.7: Thesis experiment matrix coverage in config

```json
// config/experiment_config.json (excerpt)
{
  "studies": {
    "spatiotemporal_impact": {"enabled": true, "balancing": "smote"},
    "balancing_methods": {"enabled": true, "balancing_methods": ["smote", "adasyn", "random_undersampling"]},
    "model_family_comparison": {
      "enabled": true,
      "balancing": "smote",
      "feature_set": "enhanced_features",
      "models": ["LogisticRegression", "DecisionTree", "RandomForest", "XGBoost", "CatBoost", "MLP", "LSTMProxy", "CNNProxy", "AttentionProxy"]
    },
    "ensemble_comparison": {
      "enabled": true,
      "balancing": "smote",
      "models": ["StackingEnsemble", "HybridXGBLSTM"]
    },
    "anomaly_methods": {
      "enabled": true,
      "balancing": "none",
      "models": ["AutoencoderProxy", "IsolationForest", "OneClassSVM"]
    }
  }
}
```

This configuration block makes each major thesis experiment family directly runnable and auditable.
It prevents “implicit” coverage by forcing explicit study scopes in the experiment registry.
