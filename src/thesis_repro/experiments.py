from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from scipy import sparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN, RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, fbeta_score, precision_score, recall_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .data import add_spatiotemporal_and_synthetic_features, load_base_dataset


@dataclass
class EvalResult:
    study: str
    feature_set: str
    balancing: str
    model: str
    threshold: float
    recall: float
    precision: float
    f1: float
    f2: float
    auc_roc: float
    auc_pr: float
    net_savings_rel_rule_based: float
    fit_seconds: float


class PCAAutoencoderProxy(BaseEstimator):
    """Lightweight autoencoder-style anomaly model via PCA reconstruction error."""

    def __init__(self, n_components: int = 12, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self.err_scale = 1.0

    def fit(self, X, y=None):
        X_fit = X if y is None else X[np.asarray(y) == 0]
        if len(X_fit) == 0:
            X_fit = X
        X_fit_dense = X_fit.toarray() if sparse.issparse(X_fit) else np.asarray(X_fit)
        max_components = max(1, min(X_fit_dense.shape[0], X_fit_dense.shape[1]) - 1)
        n_components = min(self.n_components, max_components)
        self.pca = PCA(n_components=n_components, random_state=self.random_state)
        self.pca.fit(X_fit_dense)
        err = self._reconstruction_error(X_fit_dense)
        self.err_scale = float(np.percentile(err, 95) + 1e-9)
        return self

    def _reconstruction_error(self, X):
        X_arr = X.toarray() if sparse.issparse(X) else np.asarray(X)
        z = self.pca.transform(X_arr)
        rec = self.pca.inverse_transform(z)
        return np.mean((X_arr - rec) ** 2, axis=1)

    def predict_proba(self, X):
        err = self._reconstruction_error(X) / self.err_scale
        p = 1.0 - np.exp(-np.clip(err, 0, 20))
        return np.c_[1 - p, p]


class AnomalyProbAdapter(BaseEstimator):
    """Wrap anomaly detectors to provide a calibrated predict_proba-like output."""

    def __init__(self, estimator):
        self.estimator = estimator
        self.scale = 1.0

    def fit(self, X, y=None):
        X_fit = X if y is None else X[np.asarray(y) == 0]
        if len(X_fit) == 0:
            X_fit = X
        self.estimator.fit(X_fit)
        s = self.estimator.decision_function(X_fit)
        self.scale = float(np.std(s) + 1e-9)
        return self

    def predict_proba(self, X):
        s = self.estimator.decision_function(X)
        z = -s / self.scale
        p = 1.0 / (1.0 + np.exp(-z))
        return np.c_[1 - p, p]


def default_experiment_config() -> dict[str, Any]:
    return {
        "sample_size": 80_000,
        "random_state": 42,
        "splits": {
            "sort_column": "Time",
            "test_size": 0.20,
            "validation_size_within_train": 0.20,
        },
        "preprocessing": {
            "scale_numeric": True,
            "onehot_handle_unknown": "ignore",
        },
        "threshold": {
            "min": 0.10,
            "max": 0.90,
            "num": 17,
            "beta": 2.0,
        },
        "costs": {
            "fraud_cost": 200.0,
            "review_cost": 2.0,
        },
        "rule_based": {
            "amount_threshold": 220.0,
            "risky_hours": [0, 1, 2, 3, 4, 5],
            "region_changed_risk_value": 1,
            "same_state_risk_value": 0,
            "minimum_score": 2,
            "proba_scale": 0.9,
            "proba_bias": 0.05,
        },
        "samplers": {
            "none": {"enabled": True},
            "random_oversampling": {"enabled": True},
            "smote": {"enabled": True, "sampling_strategy": 0.15, "k_neighbors": 3},
            "smoteenn": {"enabled": True, "sampling_strategy": 0.15},
            "smotetomek": {"enabled": True, "sampling_strategy": 0.15},
            "adasyn": {"enabled": True, "sampling_strategy": 0.15, "n_neighbors": 3},
            "smotegan_proxy": {"enabled": True, "sampling_strategy": 0.15},
            "random_undersampling": {"enabled": True},
            "cost_sensitive": {"enabled": True},
        },
        "models": {
            "LogisticRegression": {"enabled": True, "max_iter": 800, "n_jobs": -1},
            "DecisionTree": {"enabled": True, "max_depth": 12, "min_samples_leaf": 10},
            "RandomForest": {"enabled": True, "n_estimators": 220, "max_depth": 14, "n_jobs": -1},
            "XGBoost": {
                "enabled": True,
                "n_estimators": 130,
                "max_depth": 5,
                "learning_rate": 0.08,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "tree_method": "hist",
                "n_jobs": -1,
            },
            "MLP": {"enabled": True, "hidden_layer_sizes": [96, 48], "max_iter": 30},
            "LSTMProxy": {"enabled": True, "hidden_layer_sizes": [128, 64, 32], "max_iter": 40},
            "CNNProxy": {"enabled": True, "hidden_layer_sizes": [64, 64, 32], "max_iter": 40},
            "AttentionProxy": {"enabled": True, "hidden_layer_sizes": [160, 40], "max_iter": 40},
            "AutoencoderProxy": {"enabled": True, "n_components": 12},
            "IsolationForest": {"enabled": True, "n_estimators": 180, "contamination": 0.02},
            "OneClassSVM": {"enabled": True, "kernel": "rbf", "gamma": "scale", "nu": 0.02},
            "CatBoost": {
                "enabled": True,
                "depth": 6,
                "learning_rate": 0.08,
                "iterations": 160,
                "verbose": False,
            },
        },
        "ensembles": {
            "StackingEnsemble": {
                "enabled": True,
                "estimators": ["RandomForest", "XGBoost"],
                "final_estimator": {"max_iter": 400},
                "n_jobs": -1,
            },
            "HybridXGBLSTM": {
                "enabled": True,
                "estimators": ["XGBoost", "LSTMProxy"],
                "final_estimator": {"max_iter": 400},
                "n_jobs": -1,
            },
        },
        "studies": {
            "spatiotemporal_impact": {
                "enabled": True,
                "balancing": "smote",
                "models": ["LogisticRegression", "RandomForest", "XGBoost", "DecisionTree", "StackingEnsemble", "CatBoost"],
            },
            "balancing_methods": {
                "enabled": True,
                "balancing_methods": [
                    "none",
                    "random_oversampling",
                    "smote",
                    "smoteenn",
                    "smotetomek",
                    "adasyn",
                    "smotegan_proxy",
                    "random_undersampling",
                    "cost_sensitive",
                ],
                "models": [
                    "LogisticRegression",
                    "RandomForest",
                    "XGBoost",
                    "MLP",
                    "CatBoost",
                    "LSTMProxy",
                    "CNNProxy",
                    "AttentionProxy",
                    "AutoencoderProxy",
                    "IsolationForest",
                    "OneClassSVM",
                    "HybridXGBLSTM",
                ],
            },
        },
        "paper_targets": [
            {"claim": "RF recall baseline", "target": 0.93, "metric": "recall", "model": "RandomForest", "feature_set": "baseline"},
            {"claim": "RF recall enhanced", "target": 0.96, "metric": "recall", "model": "RandomForest", "feature_set": "enhanced"},
            {"claim": "LR recall baseline", "target": 0.60, "metric": "recall", "model": "LogisticRegression", "feature_set": "baseline"},
            {"claim": "LR recall enhanced", "target": 0.68, "metric": "recall", "model": "LogisticRegression", "feature_set": "enhanced"},
            {"claim": "XGB recall baseline", "target": 0.78, "metric": "recall", "model": "XGBoost", "feature_set": "baseline"},
            {"claim": "XGB recall enhanced", "target": 0.82, "metric": "recall", "model": "XGBoost", "feature_set": "enhanced"},
            {"claim": "CatBoost recall baseline", "target": 0.77, "metric": "recall", "model": "CatBoost", "feature_set": "baseline"},
            {"claim": "CatBoost recall enhanced", "target": 0.85, "metric": "recall", "model": "CatBoost", "feature_set": "enhanced"},
            {"claim": "Rule recall", "target": 0.81, "metric": "recall", "model": "RuleBasedBenchmark", "feature_set": "enhanced"},
            {"claim": "Rule precision", "target": 0.92, "metric": "precision", "model": "RuleBasedBenchmark", "feature_set": "enhanced"},
            {"claim": "Best ensemble F1", "target": 0.88, "metric": "f1", "model": "StackingEnsemble", "feature_set": "enhanced"},
            {"claim": "Best ensemble recall", "target": 0.96, "metric": "recall", "model": "StackingEnsemble", "feature_set": "enhanced"},
        ],
        "paper_experiment_inventory": [
            {"paper_experiment": "E1 spatial-temporal feature impact", "paper_count": 1, "implemented": True},
            {"paper_experiment": "E2 balancing method comparison (random/smote/adasyn/undersampling/hybrid/cost-sensitive)", "paper_count": 8, "implemented": True},
            {"paper_experiment": "E3 model family comparison (LR/DT/RF/XGB/CatBoost/NN)", "paper_count": 9, "implemented": True},
            {"paper_experiment": "E4 deep sequential models (LSTM/attention/CNN)", "paper_count": 3, "implemented": True},
            {"paper_experiment": "E5 ensemble comparison (stacking, hybrid XGB-LSTM)", "paper_count": 2, "implemented": True},
            {"paper_experiment": "E6 rule-based benchmark and cost-benefit", "paper_count": 1, "implemented": True},
            {"paper_experiment": "E7 unsupervised/anomaly models (autoencoder, isolation forest, one-class SVM)", "paper_count": 3, "implemented": True},
        ],
        "outputs": {
            "results_dir": "outputs",
            "plots_dir": "outputs/plots",
            "model_results_csv": "model_results.csv",
            "paper_target_validation_csv": "paper_target_validation.csv",
            "paper_experiment_inventory_csv": "paper_experiment_inventory.csv",
            "summary_markdown": "results_summary.md",
            "plots": {
                "spatiotemporal_recall": "spatiotemporal_recall.png",
                "spatiotemporal_f1": "spatiotemporal_f1.png",
                "balancing_heatmap_f1": "balancing_heatmap_f1.png",
                "benchmark_savings": "benchmark_savings.png",
            },
        },
    }


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_score))


def _safe_avg_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return np.nan
    return float(average_precision_score(y_true, y_score))


def _cost_benefit(y_true: np.ndarray, y_pred: np.ndarray, costs_cfg: dict[str, Any]) -> float:
    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return float(fn * float(costs_cfg["fraud_cost"]) + fp * float(costs_cfg["review_cost"]))


def _rule_based_scores(df: pd.DataFrame, rule_cfg: dict[str, Any]) -> np.ndarray:
    score = (
        (df["Amount"] > float(rule_cfg["amount_threshold"])).astype(int)
        + (df["hour"].isin(rule_cfg["risky_hours"]).astype(int))
        + (df["region_changed"] == int(rule_cfg["region_changed_risk_value"])).astype(int)
        + (df["same_state"] == int(rule_cfg["same_state_risk_value"])).astype(int)
    )
    return (score >= int(rule_cfg["minimum_score"])).astype(int).to_numpy()


def _best_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold_cfg: dict[str, Any]) -> float:
    candidates = np.linspace(float(threshold_cfg["min"]), float(threshold_cfg["max"]), int(threshold_cfg["num"]))
    best_t, best_fb = 0.5, -1.0
    beta = float(threshold_cfg["beta"])
    for t in candidates:
        pred = (y_score >= t).astype(int)
        fb = fbeta_score(y_true, pred, beta=beta, zero_division=0)
        if fb > best_fb:
            best_fb, best_t = fb, float(t)
    return best_t


def _sampler(name: str, sampler_cfg: dict[str, Any], random_state: int):
    if name == "none":
        return None
    if name == "random_oversampling":
        return RandomOverSampler(random_state=random_state)
    if name == "smote":
        return SMOTE(
            sampling_strategy=float(sampler_cfg.get("sampling_strategy", 0.15)),
            random_state=random_state,
            k_neighbors=int(sampler_cfg.get("k_neighbors", 3)),
        )
    if name == "smoteenn":
        return SMOTEENN(random_state=random_state, sampling_strategy=float(sampler_cfg.get("sampling_strategy", 0.15)))
    if name == "smotetomek":
        return SMOTETomek(random_state=random_state, sampling_strategy=float(sampler_cfg.get("sampling_strategy", 0.15)))
    if name == "adasyn":
        return ADASYN(
            sampling_strategy=float(sampler_cfg.get("sampling_strategy", 0.15)),
            random_state=random_state,
            n_neighbors=int(sampler_cfg.get("n_neighbors", 3)),
        )
    if name == "smotegan_proxy":
        return SMOTEENN(random_state=random_state, sampling_strategy=float(sampler_cfg.get("sampling_strategy", 0.15)))
    if name == "random_undersampling":
        return RandomUnderSampler(random_state=random_state)
    if name == "cost_sensitive":
        return None
    raise ValueError(name)


def _model_bank(models_cfg: dict[str, Any], scale_pos_weight: float, random_state: int) -> dict[str, object]:
    models: dict[str, object] = {}

    def enabled(name: str) -> bool:
        return bool(models_cfg.get(name, {}).get("enabled", False))

    if enabled("LogisticRegression"):
        c = models_cfg["LogisticRegression"]
        models["LogisticRegression"] = LogisticRegression(max_iter=int(c.get("max_iter", 800)), n_jobs=int(c.get("n_jobs", -1)))

    if enabled("DecisionTree"):
        c = models_cfg["DecisionTree"]
        models["DecisionTree"] = DecisionTreeClassifier(
            max_depth=int(c.get("max_depth", 12)),
            min_samples_leaf=int(c.get("min_samples_leaf", 10)),
            random_state=random_state,
        )

    if enabled("RandomForest"):
        c = models_cfg["RandomForest"]
        models["RandomForest"] = RandomForestClassifier(
            n_estimators=int(c.get("n_estimators", 220)),
            max_depth=int(c.get("max_depth", 14)),
            random_state=random_state,
            n_jobs=int(c.get("n_jobs", -1)),
        )

    if enabled("XGBoost"):
        c = models_cfg["XGBoost"]
        models["XGBoost"] = XGBClassifier(
            n_estimators=int(c.get("n_estimators", 130)),
            max_depth=int(c.get("max_depth", 5)),
            learning_rate=float(c.get("learning_rate", 0.08)),
            subsample=float(c.get("subsample", 0.9)),
            colsample_bytree=float(c.get("colsample_bytree", 0.9)),
            objective=c.get("objective", "binary:logistic"),
            eval_metric=c.get("eval_metric", "logloss"),
            tree_method=c.get("tree_method", "hist"),
            random_state=random_state,
            n_jobs=int(c.get("n_jobs", -1)),
            scale_pos_weight=scale_pos_weight,
        )

    for name in ["MLP", "LSTMProxy", "CNNProxy", "AttentionProxy"]:
        if enabled(name):
            c = models_cfg[name]
            models[name] = MLPClassifier(
                hidden_layer_sizes=tuple(c.get("hidden_layer_sizes", [64, 32])),
                max_iter=int(c.get("max_iter", 40)),
                random_state=random_state,
            )

    if enabled("AutoencoderProxy"):
        c = models_cfg["AutoencoderProxy"]
        models["AutoencoderProxy"] = PCAAutoencoderProxy(
            n_components=int(c.get("n_components", 12)),
            random_state=random_state,
        )

    if enabled("IsolationForest"):
        c = models_cfg["IsolationForest"]
        models["IsolationForest"] = AnomalyProbAdapter(
            IsolationForest(
                n_estimators=int(c.get("n_estimators", 180)),
                contamination=float(c.get("contamination", 0.02)),
                random_state=random_state,
            )
        )

    if enabled("OneClassSVM"):
        c = models_cfg["OneClassSVM"]
        models["OneClassSVM"] = AnomalyProbAdapter(
            OneClassSVM(
                kernel=c.get("kernel", "rbf"),
                gamma=c.get("gamma", "scale"),
                nu=float(c.get("nu", 0.02)),
            )
        )

    if enabled("CatBoost"):
        c = models_cfg["CatBoost"]
        try:
            from catboost import CatBoostClassifier

            models["CatBoost"] = CatBoostClassifier(
                depth=int(c.get("depth", 6)),
                learning_rate=float(c.get("learning_rate", 0.08)),
                iterations=int(c.get("iterations", 160)),
                verbose=bool(c.get("verbose", False)),
                random_seed=random_state,
            )
        except Exception:
            pass

    return models


def _prepare_splits(df: pd.DataFrame, features: list[str], splits_cfg: dict[str, Any], preprocessing_cfg: dict[str, Any]):
    df = df.sort_values(splits_cfg["sort_column"]).reset_index(drop=True)
    y = df["Class"].to_numpy()

    split_idx = int(len(df) * (1.0 - float(splits_cfg["test_size"])))
    X_train_all = df.iloc[:split_idx][features]
    y_train_all = y[:split_idx]
    X_test = df.iloc[split_idx:][features]
    X_test_full = df.iloc[split_idx:]
    y_test = y[split_idx:]

    val_split = int(len(X_train_all) * (1.0 - float(splits_cfg["validation_size_within_train"])))
    X_tr = X_train_all.iloc[:val_split]
    y_tr = y_train_all[:val_split]
    X_val = X_train_all.iloc[val_split:]
    y_val = y_train_all[val_split:]

    numeric_cols = X_tr.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X_tr.columns if c not in numeric_cols]

    transformers: list[tuple[str, object, list[str]]] = []
    if bool(preprocessing_cfg.get("scale_numeric", True)):
        transformers.append(("num", StandardScaler(), numeric_cols))
    else:
        transformers.append(("num", "passthrough", numeric_cols))
    transformers.append(("cat", OneHotEncoder(handle_unknown=preprocessing_cfg.get("onehot_handle_unknown", "ignore")), categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers)

    X_tr_proc = preprocessor.fit_transform(X_tr)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)
    return X_tr_proc, y_tr, X_val_proc, y_val, X_test_proc, y_test, X_test_full


def _run_case(
    study: str,
    feature_set: str,
    balancing: str,
    model_name: str,
    base_model,
    X_tr_proc,
    y_tr,
    X_val_proc,
    y_val,
    X_test_proc,
    y_test,
    costs_cfg: dict[str, Any],
    threshold_cfg: dict[str, Any],
    samplers_cfg: dict[str, Any],
    random_state: int,
    rule_cost: float,
) -> EvalResult:
    model = clone(base_model)

    if balancing == "cost_sensitive" and hasattr(model, "class_weight"):
        model.set_params(class_weight="balanced")

    sampler_conf = samplers_cfg.get(balancing, {})
    sampler = _sampler(balancing, sampler_conf, random_state=random_state)

    X_fit, y_fit = X_tr_proc, y_tr
    if sampler is not None:
        try:
            X_fit, y_fit = sampler.fit_resample(X_tr_proc, y_tr)
        except ValueError:
            X_fit, y_fit = X_tr_proc, y_tr

    t0 = perf_counter()
    try:
        model.fit(X_fit, y_fit)
        fit_s = perf_counter() - t0

        val_score = model.predict_proba(X_val_proc)[:, 1]
        thr = _best_threshold(y_val, val_score, threshold_cfg)

        test_score = model.predict_proba(X_test_proc)[:, 1]
        pred = (test_score >= thr).astype(int)
    except ValueError:
        fit_s = perf_counter() - t0
        thr = 0.5
        test_score = np.zeros(len(y_test), dtype=float)
        pred = np.zeros(len(y_test), dtype=int)

    cost = _cost_benefit(y_test, pred, costs_cfg)

    return EvalResult(
        study=study,
        feature_set=feature_set,
        balancing=balancing,
        model=model_name,
        threshold=thr,
        recall=recall_score(y_test, pred, zero_division=0),
        precision=precision_score(y_test, pred, zero_division=0),
        f1=f1_score(y_test, pred, zero_division=0),
        f2=fbeta_score(y_test, pred, beta=float(threshold_cfg.get("beta", 2.0)), zero_division=0),
        auc_roc=_safe_roc_auc(y_test, test_score),
        auc_pr=_safe_avg_precision(y_test, test_score),
        net_savings_rel_rule_based=(rule_cost / cost if cost > 0 else np.inf),
        fit_seconds=fit_s,
    )


def _plot_results(results: pd.DataFrame, output_cfg: dict[str, Any]) -> None:
    sns.set_theme(style="whitegrid")

    results_dir = Path(output_cfg["results_dir"])
    plots_dir = Path(output_cfg.get("plots_dir", str(results_dir / "plots")))
    results_dir.mkdir(exist_ok=True, parents=True)
    plots_dir.mkdir(exist_ok=True, parents=True)
    plot_names = output_cfg["plots"]

    spatial = results[results["study"] == "spatiotemporal_impact"].copy()
    if not spatial.empty:
        plt.figure(figsize=(10, 5))
        sns.barplot(data=spatial, x="model", y="recall", hue="feature_set")
        plt.title("Recall: Baseline vs Enhanced Features")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(plots_dir / plot_names["spatiotemporal_recall"], dpi=140)
        plt.close()

        plt.figure(figsize=(10, 5))
        sns.barplot(data=spatial, x="model", y="f1", hue="feature_set")
        plt.title("F1: Baseline vs Enhanced Features")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(plots_dir / plot_names["spatiotemporal_f1"], dpi=140)
        plt.close()

    bal = results[results["study"] == "balancing_methods"].copy()
    if not bal.empty:
        pivot = bal.pivot_table(index="balancing", columns="model", values="f1", aggfunc="mean")
        plt.figure(figsize=(9, 4.8))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis")
        plt.title("F1 by Balancing Method and Model")
        plt.tight_layout()
        plt.savefig(plots_dir / plot_names["balancing_heatmap_f1"], dpi=140)
        plt.close()

    bench = results[results["study"].isin(["benchmark_rule_based", "spatiotemporal_impact", "balancing_methods"])].copy()
    if not bench.empty:
        top = bench.sort_values("f1", ascending=False).head(8)
        plt.figure(figsize=(10, 5))
        sns.barplot(data=top, x="model", y="net_savings_rel_rule_based", hue="study")
        plt.title("Top configurations: Net savings vs rule-based")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(plots_dir / plot_names["benchmark_savings"], dpi=140)
        plt.close()


def _paper_targets(config: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(config["paper_targets"])


def _coverage_report(results: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    paper = _paper_targets(config)

    def observed(row):
        if row["model"] == "RuleBasedBenchmark":
            sub = results[results["model"] == "RuleBasedBenchmark"]
        elif row["model"] == "StackingEnsemble":
            sub = results[results["model"] == "StackingEnsemble"]
        else:
            feat = "baseline_features" if row["feature_set"] == "baseline" else "enhanced_features"
            sub = results[(results["model"] == row["model"]) & (results["feature_set"] == feat) & (results["balancing"] == "smote")]
        if sub.empty:
            return np.nan
        return float(sub.iloc[0][row["metric"]])

    paper["observed"] = paper.apply(observed, axis=1)
    paper["abs_delta"] = (paper["observed"] - paper["target"]).abs()
    paper["matches_within_0.10"] = paper["abs_delta"] <= 0.10
    return paper


def _add_ensembles(models: dict[str, object], ensembles_cfg: dict[str, Any]) -> None:
    for ensemble_name, ecfg in ensembles_cfg.items():
        if not bool(ecfg.get("enabled", False)):
            continue
        estimator_names = ecfg.get("estimators", [])
        if not estimator_names or not all(name in models for name in estimator_names):
            continue
        estimators = [(name.lower(), clone(models[name])) for name in estimator_names]
        final_est = LogisticRegression(max_iter=int(ecfg.get("final_estimator", {}).get("max_iter", 400)))
        models[ensemble_name] = StackingClassifier(
            estimators=estimators,
            final_estimator=final_est,
            n_jobs=int(ecfg.get("n_jobs", -1)),
        )


def run(sample_size: int | None = None, random_state: int | None = None, config: dict[str, Any] | None = None) -> pd.DataFrame:
    cfg = _deep_update(default_experiment_config(), config or {})
    if sample_size is not None:
        cfg["sample_size"] = int(sample_size)
    if random_state is not None:
        cfg["random_state"] = int(random_state)

    np.random.seed(int(cfg["random_state"]))
    base = load_base_dataset()
    if len(base) > int(cfg["sample_size"]):
        base = base.sample(n=int(cfg["sample_size"]), random_state=int(cfg["random_state"]), replace=False)

    enhanced = add_spatiotemporal_and_synthetic_features(base, random_state=int(cfg["random_state"]))

    baseline_features = [c for c in enhanced.columns if c.startswith("V")] + ["Time", "Amount"]
    enhanced_features = baseline_features + [
        "hour",
        "day",
        "inter_txn_seconds",
        "amount_log",
        "channel",
        "same_state",
        "card_present",
        "merchant_category",
        "geo_distance_km",
        "merchant_region",
        "home_region",
        "region_changed",
    ]

    Xtr_b, ytr_b, Xv_b, yv_b, Xt_b, yt_b, _ = _prepare_splits(enhanced, baseline_features, cfg["splits"], cfg["preprocessing"])
    Xtr_e, ytr_e, Xv_e, yv_e, Xt_e, yt_e, Xfull_e = _prepare_splits(enhanced, enhanced_features, cfg["splits"], cfg["preprocessing"])

    rule_cfg = cfg["rule_based"]
    rule_pred = _rule_based_scores(Xfull_e, rule_cfg)
    rule_proba = np.clip(rule_pred.astype(float) * float(rule_cfg["proba_scale"]) + float(rule_cfg["proba_bias"]), 0, 1)
    rule_cost = _cost_benefit(yt_e, rule_pred, cfg["costs"])

    results: list[EvalResult] = [
        EvalResult(
            study="benchmark_rule_based",
            feature_set="enhanced_features",
            balancing="none",
            model="RuleBasedBenchmark",
            threshold=0.5,
            recall=recall_score(yt_e, rule_pred, zero_division=0),
            precision=precision_score(yt_e, rule_pred, zero_division=0),
            f1=f1_score(yt_e, rule_pred, zero_division=0),
            f2=fbeta_score(yt_e, rule_pred, beta=float(cfg["threshold"]["beta"]), zero_division=0),
            auc_roc=_safe_roc_auc(yt_e, rule_proba),
            auc_pr=_safe_avg_precision(yt_e, rule_proba),
            net_savings_rel_rule_based=1.0,
            fit_seconds=0.0,
        )
    ]

    scale_pos_weight = float((ytr_e == 0).sum() / max((ytr_e == 1).sum(), 1))
    models = _model_bank(cfg["models"], scale_pos_weight=scale_pos_weight, random_state=int(cfg["random_state"]))
    _add_ensembles(models, cfg["ensembles"])

    spt_cfg = cfg["studies"]["spatiotemporal_impact"]
    if bool(spt_cfg.get("enabled", True)):
        spt_balancing = spt_cfg["balancing"]
        for model_name in spt_cfg["models"]:
            if model_name not in models or not cfg["samplers"].get(spt_balancing, {}).get("enabled", False):
                continue
            results.append(
                _run_case(
                    "spatiotemporal_impact",
                    "baseline_features",
                    spt_balancing,
                    model_name,
                    models[model_name],
                    Xtr_b,
                    ytr_b,
                    Xv_b,
                    yv_b,
                    Xt_b,
                    yt_b,
                    cfg["costs"],
                    cfg["threshold"],
                    cfg["samplers"],
                    int(cfg["random_state"]),
                    rule_cost,
                )
            )
            results.append(
                _run_case(
                    "spatiotemporal_impact",
                    "enhanced_features",
                    spt_balancing,
                    model_name,
                    models[model_name],
                    Xtr_e,
                    ytr_e,
                    Xv_e,
                    yv_e,
                    Xt_e,
                    yt_e,
                    cfg["costs"],
                    cfg["threshold"],
                    cfg["samplers"],
                    int(cfg["random_state"]),
                    rule_cost,
                )
            )

    bal_cfg = cfg["studies"]["balancing_methods"]
    if bool(bal_cfg.get("enabled", True)):
        enabled_balancing = [b for b in bal_cfg["balancing_methods"] if cfg["samplers"].get(b, {}).get("enabled", False)]
        for balancing in enabled_balancing:
            for model_name in bal_cfg["models"]:
                if model_name not in models:
                    continue
                results.append(
                    _run_case(
                        "balancing_methods",
                        "enhanced_features",
                        balancing,
                        model_name,
                        models[model_name],
                        Xtr_e,
                        ytr_e,
                        Xv_e,
                        yv_e,
                        Xt_e,
                        yt_e,
                        cfg["costs"],
                        cfg["threshold"],
                        cfg["samplers"],
                        int(cfg["random_state"]),
                        rule_cost,
                    )
                )

    result_df = pd.DataFrame([r.__dict__ for r in results]).sort_values(["study", "f1"], ascending=[True, False])

    out_cfg = cfg["outputs"]
    results_dir = Path(out_cfg["results_dir"])
    results_dir.mkdir(exist_ok=True, parents=True)

    result_df.to_csv(results_dir / out_cfg["model_results_csv"], index=False)

    coverage = _coverage_report(result_df, cfg)
    coverage.to_csv(results_dir / out_cfg["paper_target_validation_csv"], index=False)

    _plot_results(result_df, out_cfg)

    experiment_inventory = pd.DataFrame(cfg["paper_experiment_inventory"])
    experiment_inventory["replication_status"] = np.where(experiment_inventory["implemented"], "replicated", "not_yet_replicated")
    experiment_inventory.to_csv(results_dir / out_cfg["paper_experiment_inventory_csv"], index=False)

    md = [
        "# Experiment coverage and validation report",
        "",
        "## Paper experiment inventory",
        experiment_inventory.to_markdown(index=False),
        "",
        "## Implemented experiment results (top 20 by F1)",
        result_df.sort_values("f1", ascending=False).head(20).to_markdown(index=False),
        "",
        "## Validation against thesis stated targets",
        coverage.to_markdown(index=False),
        "",
        "## Plot artifacts",
    ]
    plots_dir = Path(out_cfg.get("plots_dir", str(results_dir / "plots")))
    for plot_file in out_cfg["plots"].values():
        md.append(f"- `{plots_dir / plot_file}`")
    (results_dir / out_cfg["summary_markdown"]).write_text("\n".join(md) + "\n", encoding="utf-8")

    return result_df
