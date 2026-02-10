# Experiment coverage and validation report

## Paper experiment inventory
| paper_experiment                                   |   paper_count | implemented   | replication_status   |
|:---------------------------------------------------|--------------:|:--------------|:---------------------|
| E1 spatial-temporal feature impact                 |             1 | True          | replicated           |
| E2 balancing method comparison (6 methods)         |             6 | True          | replicated           |
| E3 model family comparison (LR/DT/RF/XGB/NN)       |             5 | True          | replicated           |
| E4 deep sequential models (LSTM/attention/CNN)     |             3 | False         | not_yet_replicated   |
| E5 ensemble comparison (stacking, hybrid XGB-LSTM) |             2 | False         | not_yet_replicated   |
| E6 rule-based benchmark and cost-benefit           |             1 | True          | replicated           |
| E7 interpretability (SHAP/attention)               |             1 | False         | not_yet_replicated   |

## Implemented experiment results (top 20 by F1)
| study                 | feature_set       | balancing            | model            |   threshold |   recall |   precision |       f1 |   auc_roc |   auc_pr |   net_savings_rel_rule_based |   fit_seconds |
|:----------------------|:------------------|:---------------------|:-----------------|------------:|---------:|------------:|---------:|----------:|---------:|-----------------------------:|--------------:|
| balancing_methods     | enhanced_features | none                 | MLP              |        0.1  | 0.666667 |    1        | 0.8      |  0.698882 | 0.66722  |                      9.13    |     0.900804  |
| balancing_methods     | enhanced_features | random_oversampling  | MLP              |        0.15 | 0.666667 |    1        | 0.8      |  0.679686 | 0.667187 |                      9.13    |     1.8676    |
| balancing_methods     | enhanced_features | smote                | MLP              |        0.1  | 0.666667 |    1        | 0.8      |  0.684026 | 0.667194 |                      9.13    |     1.0227    |
| balancing_methods     | enhanced_features | smoteenn             | MLP              |        0.1  | 0.666667 |    1        | 0.8      |  0.687031 | 0.667199 |                      9.13    |     0.754936  |
| balancing_methods     | enhanced_features | cost_sensitive       | MLP              |        0.1  | 0.666667 |    1        | 0.8      |  0.698882 | 0.66722  |                      9.13    |     0.781713  |
| spatiotemporal_impact | baseline_features | smote                | DecisionTree     |        0.85 | 0.666667 |    1        | 0.8      |  0.832582 | 0.667167 |                      9.13    |     0.0958587 |
| spatiotemporal_impact | baseline_features | smote                | RandomForest     |        0.7  | 0.666667 |    1        | 0.8      |  0.97588  | 0.671875 |                      9.13    |     0.741042  |
| spatiotemporal_impact | baseline_features | smote                | StackingEnsemble |        0.5  | 0.666667 |    1        | 0.8      |  0.974462 | 0.673077 |                      9.13    |     7.08007   |
| spatiotemporal_impact | enhanced_features | smote                | StackingEnsemble |        0.55 | 0.666667 |    1        | 0.8      |  0.981305 | 0.675362 |                      9.13    |     7.25891   |
| balancing_methods     | enhanced_features | none                 | RandomForest     |        0.25 | 0.666667 |    0.666667 | 0.666667 |  0.991487 | 0.568543 |                      9.0396  |     0.591217  |
| balancing_methods     | enhanced_features | none                 | XGBoost          |        0.4  | 0.666667 |    0.666667 | 0.666667 |  0.954932 | 0.67033  |                      9.0396  |     0.645772  |
| balancing_methods     | enhanced_features | random_undersampling | MLP              |        0.9  | 0.666667 |    0.666667 | 0.666667 |  0.952095 | 0.392361 |                      9.0396  |     0.0166924 |
| balancing_methods     | enhanced_features | cost_sensitive       | RandomForest     |        0.15 | 0.666667 |    0.666667 | 0.666667 |  0.826907 | 0.667167 |                      9.0396  |     0.689829  |
| balancing_methods     | enhanced_features | cost_sensitive       | XGBoost          |        0.4  | 0.666667 |    0.666667 | 0.666667 |  0.954932 | 0.67033  |                      9.0396  |     0.693422  |
| spatiotemporal_impact | enhanced_features | smote                | RandomForest     |        0.3  | 0.666667 |    0.666667 | 0.666667 |  0.984644 | 0.675    |                      9.0396  |     0.623507  |
| balancing_methods     | enhanced_features | random_oversampling  | RandomForest     |        0.3  | 0.666667 |    0.666667 | 0.666667 |  0.823819 | 0.667167 |                      9.0396  |     0.827145  |
| balancing_methods     | enhanced_features | smote                | RandomForest     |        0.3  | 0.666667 |    0.666667 | 0.666667 |  0.984644 | 0.675    |                      9.0396  |     0.727511  |
| balancing_methods     | enhanced_features | smoteenn             | RandomForest     |        0.35 | 0.666667 |    0.666667 | 0.666667 |  0.80888  | 0.667167 |                      9.0396  |     0.724474  |
| balancing_methods     | enhanced_features | random_undersampling | RandomForest     |        0.6  | 0.666667 |    0.666667 | 0.666667 |  0.980304 | 0.563556 |                      9.0396  |     0.40055   |
| balancing_methods     | enhanced_features | random_oversampling  | XGBoost          |        0.2  | 0.666667 |    0.5      | 0.571429 |  0.796945 | 0.667486 |                      8.95098 |     1.00616   |

## Validation against thesis stated targets
| claim                |   target | metric    | model              | feature_set   |   observed |   abs_delta | matches_within_0.10   |
|:---------------------|---------:|:----------|:-------------------|:--------------|-----------:|------------:|:----------------------|
| RF recall baseline   |     0.93 | recall    | RandomForest       | baseline      | 0.666667   |    0.263333 | False                 |
| RF recall enhanced   |     0.96 | recall    | RandomForest       | enhanced      | 0.666667   |    0.293333 | False                 |
| LR recall baseline   |     0.6  | recall    | LogisticRegression | baseline      | 0.333333   |    0.266667 | False                 |
| LR recall enhanced   |     0.68 | recall    | LogisticRegression | enhanced      | 0.333333   |    0.346667 | False                 |
| XGB recall baseline  |     0.78 | recall    | XGBoost            | baseline      | 0.666667   |    0.113333 | False                 |
| XGB recall enhanced  |     0.82 | recall    | XGBoost            | enhanced      | 0.666667   |    0.153333 | False                 |
| Rule recall          |     0.81 | recall    | RuleBasedBenchmark | enhanced      | 0.333333   |    0.476667 | False                 |
| Rule precision       |     0.92 | precision | RuleBasedBenchmark | enhanced      | 0.00140056 |    0.918599 | False                 |
| Best ensemble F1     |     0.88 | f1        | StackingEnsemble   | enhanced      | 0.8        |    0.08     | True                  |
| Best ensemble recall |     0.96 | recall    | StackingEnsemble   | enhanced      | 0.666667   |    0.293333 | False                 |

## Plot artifacts
- `outputs/plots/spatiotemporal_recall.png`
- `outputs/plots/spatiotemporal_f1.png`
- `outputs/plots/balancing_heatmap_f1.png`
- `outputs/plots/benchmark_savings.png`
