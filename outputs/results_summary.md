# Experiment results

## Model-level metrics
| experiment        | model                    |   threshold |   recall |   precision |         f1 |         f2 |   auc_roc |     auc_pr |   net_savings_rel_rule_based |
|:------------------|:-------------------------|------------:|---------:|------------:|-----------:|-----------:|----------:|-----------:|-----------------------------:|
| baseline_features | RandomForest+SMOTE       |        0.25 | 0.833333 |  0.769231   | 0.8        | 0.819672   |  0.993584 | 0.863895   |                      17.6158 |
| baseline_features | Stacking(RF+XGB)+SMOTE   |        0.1  | 0.75     |  0.818182   | 0.782609   | 0.762712   |  0.993563 | 0.830896   |                      11.8411 |
| baseline_features | XGBoost+SMOTE            |        0.25 | 0.833333 |  0.625      | 0.714286   | 0.78125    |  0.988128 | 0.815047   |                      17.3592 |
| baseline_features | LogisticRegression+SMOTE |        0.9  | 0.833333 |  0.4        | 0.540541   | 0.684932   |  0.978301 | 0.707786   |                      16.6326 |
| baseline_features | RuleBasedBenchmark       |        0.5  | 0.333333 |  0.00143885 | 0.00286533 | 0.00707214 |  0.492906 | 0.00147962 |                       1      |
| enhanced_features | Stacking(RF+XGB)+SMOTE   |        0.1  | 0.833333 |  1          | 0.909091   | 0.862069   |  0.956476 | 0.862219   |                      17.88   |
| enhanced_features | RandomForest+SMOTE       |        0.2  | 0.833333 |  0.666667   | 0.740741   | 0.793651   |  0.943201 | 0.857492   |                      17.4439 |
| enhanced_features | XGBoost+SMOTE            |        0.1  | 0.833333 |  0.625      | 0.714286   | 0.78125    |  0.964029 | 0.867574   |                      17.3592 |
| enhanced_features | LogisticRegression+SMOTE |        0.9  | 0.75     |  0.333333   | 0.461538   | 0.6        |  0.972782 | 0.673504   |                      11.2453 |
| enhanced_features | RuleBasedBenchmark       |        0.5  | 0.333333 |  0.00143885 | 0.00286533 | 0.00707214 |  0.492906 | 0.00147962 |                       1      |

## Average by feature set (non-rule models)
| experiment        |   recall |   precision |       f1 |       f2 |   auc_roc |   auc_pr |
|:------------------|---------:|------------:|---------:|---------:|----------:|---------:|
| baseline_features |   0.8125 |    0.653103 | 0.709359 | 0.762141 |  0.988394 | 0.804406 |
| enhanced_features |   0.8125 |    0.65625  | 0.706414 | 0.759242 |  0.959122 | 0.815197 |
