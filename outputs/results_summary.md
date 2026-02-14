# Experiment coverage and validation report

## Paper experiment inventory
| paper_experiment                                                                         |   paper_count | implemented   | replication_status   |
|:-----------------------------------------------------------------------------------------|--------------:|:--------------|:---------------------|
| E1 spatial-temporal feature impact                                                       |             1 | True          | replicated           |
| E2 balancing method comparison (random/smote/adasyn/undersampling/hybrid/cost-sensitive) |             8 | True          | replicated           |
| E3 model family comparison (LR/DT/RF/XGB/CatBoost/NN)                                    |             9 | True          | replicated           |
| E4 deep sequential models (LSTM/attention/CNN)                                           |             3 | True          | replicated           |
| E5 ensemble comparison (stacking, hybrid XGB-LSTM)                                       |             2 | True          | replicated           |
| E6 rule-based benchmark and cost-benefit                                                 |             1 | True          | replicated           |
| E7 unsupervised/anomaly models (autoencoder, isolation forest, one-class SVM)            |             3 | True          | replicated           |

## Implemented experiment results (top 20 by F1)
| study             | feature_set       | balancing           | model              |   threshold |   recall |   precision |   f1 |   f2 |   auc_roc |   auc_pr |   net_savings_rel_rule_based |   fit_seconds |
|:------------------|:------------------|:--------------------|:-------------------|------------:|---------:|------------:|-----:|-----:|----------:|---------:|-----------------------------:|--------------:|
| anomaly_methods   | enhanced_features | none                | AutoencoderProxy   |         0.1 |        0 |           0 |    0 |    0 |       nan |      nan |                         0.25 |   0.00112353  |
| anomaly_methods   | enhanced_features | none                | IsolationForest    |         0.1 |        0 |           0 |    0 |    0 |       nan |      nan |                         0.25 |   0.180838    |
| anomaly_methods   | enhanced_features | none                | OneClassSVM        |         0.1 |        0 |           0 |    0 |    0 |       nan |      nan |                         0.25 |   0.000874139 |
| balancing_methods | enhanced_features | none                | LogisticRegression |         0.1 |        0 |           0 |    0 |    0 |       nan |      nan |                       inf    |   1.16636     |
| balancing_methods | enhanced_features | none                | RandomForest       |         0.1 |        0 |           0 |    0 |    0 |       nan |      nan |                       inf    |   0.310664    |
| balancing_methods | enhanced_features | none                | XGBoost            |         0.1 |        0 |           0 |    0 |    0 |       nan |      nan |                         0.25 |   0.0868053   |
| balancing_methods | enhanced_features | none                | MLP                |         0.1 |        0 |           0 |    0 |    0 |       nan |      nan |                       inf    |   0.0150514   |
| balancing_methods | enhanced_features | none                | LSTMProxy          |         0.1 |        0 |           0 |    0 |    0 |       nan |      nan |                       inf    |   0.0250304   |
| balancing_methods | enhanced_features | none                | CNNProxy           |         0.1 |        0 |           0 |    0 |    0 |       nan |      nan |                       inf    |   0.0201169   |
| balancing_methods | enhanced_features | none                | AttentionProxy     |         0.1 |        0 |           0 |    0 |    0 |       nan |      nan |                       inf    |   0.020869    |
| balancing_methods | enhanced_features | none                | AutoencoderProxy   |         0.1 |        0 |           0 |    0 |    0 |       nan |      nan |                         0.25 |   0.00153316  |
| balancing_methods | enhanced_features | none                | IsolationForest    |         0.1 |        0 |           0 |    0 |    0 |       nan |      nan |                         0.25 |   0.177774    |
| balancing_methods | enhanced_features | none                | OneClassSVM        |         0.1 |        0 |           0 |    0 |    0 |       nan |      nan |                         0.25 |   0.00108456  |
| balancing_methods | enhanced_features | none                | HybridXGBLSTM      |         0.5 |        0 |           0 |    0 |    0 |       nan |      nan |                       inf    |   2.32794     |
| balancing_methods | enhanced_features | random_oversampling | LogisticRegression |         0.1 |        0 |           0 |    0 |    0 |       nan |      nan |                       inf    |   1.13841     |
| balancing_methods | enhanced_features | random_oversampling | RandomForest       |         0.1 |        0 |           0 |    0 |    0 |       nan |      nan |                       inf    |   0.308528    |
| balancing_methods | enhanced_features | random_oversampling | XGBoost            |         0.1 |        0 |           0 |    0 |    0 |       nan |      nan |                       inf    |   0.0171152   |
| balancing_methods | enhanced_features | random_oversampling | MLP                |         0.1 |        0 |           0 |    0 |    0 |       nan |      nan |                       inf    |   0.0149961   |
| balancing_methods | enhanced_features | random_oversampling | LSTMProxy          |         0.1 |        0 |           0 |    0 |    0 |       nan |      nan |                       inf    |   0.0259589   |
| balancing_methods | enhanced_features | random_oversampling | CNNProxy           |         0.1 |        0 |           0 |    0 |    0 |       nan |      nan |                       inf    |   0.023306    |

## Validation against thesis stated targets
| claim                    |   target | metric    | model              | feature_set   |   observed |   abs_delta | matches_within_0.10   |
|:-------------------------|---------:|:----------|:-------------------|:--------------|-----------:|------------:|:----------------------|
| RF recall baseline       |     0.93 | recall    | RandomForest       | baseline      |          0 |        0.93 | False                 |
| RF recall enhanced       |     0.96 | recall    | RandomForest       | enhanced      |          0 |        0.96 | False                 |
| LR recall baseline       |     0.6  | recall    | LogisticRegression | baseline      |          0 |        0.6  | False                 |
| LR recall enhanced       |     0.68 | recall    | LogisticRegression | enhanced      |          0 |        0.68 | False                 |
| XGB recall baseline      |     0.78 | recall    | XGBoost            | baseline      |          0 |        0.78 | False                 |
| XGB recall enhanced      |     0.82 | recall    | XGBoost            | enhanced      |          0 |        0.82 | False                 |
| CatBoost recall baseline |     0.77 | recall    | CatBoost           | baseline      |        nan |      nan    | False                 |
| CatBoost recall enhanced |     0.85 | recall    | CatBoost           | enhanced      |        nan |      nan    | False                 |
| Rule recall              |     0.81 | recall    | RuleBasedBenchmark | enhanced      |          0 |        0.81 | False                 |
| Rule precision           |     0.92 | precision | RuleBasedBenchmark | enhanced      |          0 |        0.92 | False                 |
| Best ensemble F1         |     0.88 | f1        | StackingEnsemble   | enhanced      |          0 |        0.88 | False                 |
| Best ensemble recall     |     0.96 | recall    | StackingEnsemble   | enhanced      |          0 |        0.96 | False                 |

## Plot artifacts
- `outputs/plots/spatiotemporal_recall.png`
- `outputs/plots/spatiotemporal_f1.png`
- `outputs/plots/balancing_heatmap_f1.png`
- `outputs/plots/benchmark_savings.png`
