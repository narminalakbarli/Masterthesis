from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .data import add_spatiotemporal_and_synthetic_features, load_base_dataset

RESULTS_DIR = Path("outputs")
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


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
    auc_roc: float
    auc_pr: float
    net_savings_rel_rule_based: float
    fit_seconds: float


def _cost_benefit(y_true: np.ndarray, y_pred: np.ndarray, fraud_cost: float = 200.0, review_cost: float = 2.0) -> float:
    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return float(fn * fraud_cost + fp * review_cost)


def _rule_based_scores(df: pd.DataFrame) -> np.ndarray:
    score = (
        (df["Amount"] > 220).astype(int)
        + (df["hour"].isin([0, 1, 2, 3, 4, 5]).astype(int))
        + (df["region_changed"] == 1).astype(int)
        + (df["same_state"] == 0).astype(int)
    )
    return (score >= 2).astype(int).to_numpy()


def _best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    candidates = np.linspace(0.1, 0.9, 17)
    best_t, best_f1 = 0.5, -1.0
    for t in candidates:
        pred = (y_score >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def _sampler(name: str):
    if name == "none":
        return None
    if name == "random_oversampling":
        return RandomOverSampler(random_state=42)
    if name == "smote":
        return SMOTE(sampling_strategy=0.15, random_state=42, k_neighbors=3)
    if name == "smoteenn":
        return SMOTEENN(random_state=42, sampling_strategy=0.15)
    if name == "random_undersampling":
        return RandomUnderSampler(random_state=42)
    if name == "cost_sensitive":
        return None
    raise ValueError(name)


def _model_bank(scale_pos_weight: float) -> dict[str, object]:
    return {
        "LogisticRegression": LogisticRegression(max_iter=800, n_jobs=-1),
        "DecisionTree": DecisionTreeClassifier(max_depth=12, min_samples_leaf=10, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=220, max_depth=14, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=130,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight,
        ),
        "MLP": MLPClassifier(hidden_layer_sizes=(96, 48), max_iter=30, random_state=42),
    }


def _prepare_splits(df: pd.DataFrame, features: list[str]):
    df = df.sort_values("Time").reset_index(drop=True)
    y = df["Class"].to_numpy()
    split_idx = int(len(df) * 0.8)

    X_train_all = df.iloc[:split_idx][features]
    y_train_all = y[:split_idx]
    X_test = df.iloc[split_idx:][features]
    X_test_full = df.iloc[split_idx:]
    y_test = y[split_idx:]

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_all, y_train_all, test_size=0.2, random_state=42, stratify=y_train_all
    )

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
    rule_cost: float,
) -> EvalResult:
    model = clone(base_model)

    if balancing == "cost_sensitive":
        if hasattr(model, "class_weight"):
            model.set_params(class_weight="balanced")
    sampler = _sampler(balancing)

    X_fit, y_fit = X_tr_proc, y_tr
    if sampler is not None:
        X_fit, y_fit = sampler.fit_resample(X_tr_proc, y_tr)

    t0 = perf_counter()
    model.fit(X_fit, y_fit)
    fit_s = perf_counter() - t0

    val_score = model.predict_proba(X_val_proc)[:, 1]
    thr = _best_threshold(y_val, val_score)

    test_score = model.predict_proba(X_test_proc)[:, 1]
    pred = (test_score >= thr).astype(int)
    cost = _cost_benefit(y_test, pred)

    return EvalResult(
        study=study,
        feature_set=feature_set,
        balancing=balancing,
        model=model_name,
        threshold=thr,
        recall=recall_score(y_test, pred, zero_division=0),
        precision=precision_score(y_test, pred, zero_division=0),
        f1=f1_score(y_test, pred, zero_division=0),
        auc_roc=roc_auc_score(y_test, test_score),
        auc_pr=average_precision_score(y_test, test_score),
        net_savings_rel_rule_based=(rule_cost / cost if cost > 0 else np.inf),
        fit_seconds=fit_s,
    )


def _plot_results(results: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    # 1) Spatial-temporal impact per model.
    spatial = results[results["study"] == "spatiotemporal_impact"].copy()
    if not spatial.empty:
        plt.figure(figsize=(10, 5))
        sns.barplot(data=spatial, x="model", y="recall", hue="feature_set")
        plt.title("Recall: Baseline vs Enhanced Features (SMOTE)")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "spatiotemporal_recall.png", dpi=140)
        plt.close()

        plt.figure(figsize=(10, 5))
        sns.barplot(data=spatial, x="model", y="f1", hue="feature_set")
        plt.title("F1: Baseline vs Enhanced Features (SMOTE)")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "spatiotemporal_f1.png", dpi=140)
        plt.close()

    # 2) Balancing impact heatmap on enhanced features.
    bal = results[results["study"] == "balancing_methods"].copy()
    if not bal.empty:
        pivot = bal.pivot_table(index="balancing", columns="model", values="f1", aggfunc="mean")
        plt.figure(figsize=(9, 4.8))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis")
        plt.title("F1 by Balancing Method and Model (Enhanced Features)")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "balancing_heatmap_f1.png", dpi=140)
        plt.close()

    # 3) Benchmark comparison (best models + rule).
    bench = results[results["study"].isin(["benchmark_rule_based", "spatiotemporal_impact", "balancing_methods"])].copy()
    if not bench.empty:
        top = bench.sort_values("f1", ascending=False).head(8)
        plt.figure(figsize=(10, 5))
        sns.barplot(data=top, x="model", y="net_savings_rel_rule_based", hue="study")
        plt.title("Top configurations: Net savings vs rule-based")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "benchmark_savings.png", dpi=140)
        plt.close()


def _paper_targets() -> pd.DataFrame:
    # Extracted from thesis narrative sections.
    return pd.DataFrame(
        [
            {"claim": "RF recall baseline", "target": 0.93, "metric": "recall", "model": "RandomForest", "feature_set": "baseline"},
            {"claim": "RF recall enhanced", "target": 0.96, "metric": "recall", "model": "RandomForest", "feature_set": "enhanced"},
            {"claim": "LR recall baseline", "target": 0.60, "metric": "recall", "model": "LogisticRegression", "feature_set": "baseline"},
            {"claim": "LR recall enhanced", "target": 0.68, "metric": "recall", "model": "LogisticRegression", "feature_set": "enhanced"},
            {"claim": "XGB recall baseline", "target": 0.78, "metric": "recall", "model": "XGBoost", "feature_set": "baseline"},
            {"claim": "XGB recall enhanced", "target": 0.82, "metric": "recall", "model": "XGBoost", "feature_set": "enhanced"},
            {"claim": "Rule recall", "target": 0.81, "metric": "recall", "model": "RuleBasedBenchmark", "feature_set": "enhanced"},
            {"claim": "Rule precision", "target": 0.92, "metric": "precision", "model": "RuleBasedBenchmark", "feature_set": "enhanced"},
            {"claim": "Best ensemble F1", "target": 0.88, "metric": "f1", "model": "StackingEnsemble", "feature_set": "enhanced"},
            {"claim": "Best ensemble recall", "target": 0.96, "metric": "recall", "model": "StackingEnsemble", "feature_set": "enhanced"},
        ]
    )


def _coverage_report(results: pd.DataFrame) -> pd.DataFrame:
    paper = _paper_targets()

    # map our model names
    def observed(row):
        if row["model"] == "RuleBasedBenchmark":
            sub = results[results["model"] == "RuleBasedBenchmark"]
        elif row["model"] == "StackingEnsemble":
            sub = results[results["model"] == "StackingEnsemble"]
        else:
            feat = "baseline_features" if row["feature_set"] == "baseline" else "enhanced_features"
            sub = results[
                (results["model"] == row["model"])
                & (results["feature_set"] == feat)
                & (results["balancing"] == "smote")
            ]
        if sub.empty:
            return np.nan
        return float(sub.iloc[0][row["metric"]])

    paper["observed"] = paper.apply(observed, axis=1)
    paper["abs_delta"] = (paper["observed"] - paper["target"]).abs()
    paper["matches_within_0.10"] = paper["abs_delta"] <= 0.10
    return paper


def run(sample_size: int = 60_000) -> pd.DataFrame:
    base = load_base_dataset()
    if len(base) > sample_size:
        base = base.sample(n=sample_size, random_state=42, replace=False)

    enhanced = add_spatiotemporal_and_synthetic_features(base)

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

    # Prepare splits once per feature-set.
    Xtr_b, ytr_b, Xv_b, yv_b, Xt_b, yt_b, Xfull_b = _prepare_splits(enhanced, baseline_features)
    Xtr_e, ytr_e, Xv_e, yv_e, Xt_e, yt_e, Xfull_e = _prepare_splits(enhanced, enhanced_features)

    # Rule-based benchmark (on enhanced test set context).
    rule_pred = _rule_based_scores(Xfull_e)
    rule_proba = np.clip(rule_pred.astype(float) * 0.9 + 0.05, 0, 1)
    rule_cost = _cost_benefit(yt_e, rule_pred)

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
            auc_roc=roc_auc_score(yt_e, rule_proba),
            auc_pr=average_precision_score(yt_e, rule_proba),
            net_savings_rel_rule_based=1.0,
            fit_seconds=0.0,
        )
    ]

    scale_pos_weight = float((ytr_e == 0).sum() / max((ytr_e == 1).sum(), 1))
    models = _model_bank(scale_pos_weight=scale_pos_weight)

    # Add thesis-mentioned ensemble (stacking of tree-based models).
    models["StackingEnsemble"] = StackingClassifier(
        estimators=[("rf", clone(models["RandomForest"])), ("xgb", clone(models["XGBoost"]))],
        final_estimator=LogisticRegression(max_iter=400),
        n_jobs=-1,
    )

    # STUDY 1: Spatial-temporal impact (SMOTE, baseline vs enhanced).
    for model_name in ["LogisticRegression", "RandomForest", "XGBoost", "DecisionTree", "StackingEnsemble"]:
        results.append(
            _run_case(
                "spatiotemporal_impact",
                "baseline_features",
                "smote",
                model_name,
                models[model_name],
                Xtr_b,
                ytr_b,
                Xv_b,
                yv_b,
                Xt_b,
                yt_b,
                rule_cost,
            )
        )
        results.append(
            _run_case(
                "spatiotemporal_impact",
                "enhanced_features",
                "smote",
                model_name,
                models[model_name],
                Xtr_e,
                ytr_e,
                Xv_e,
                yv_e,
                Xt_e,
                yt_e,
                rule_cost,
            )
        )

    # STUDY 2: Balancing methods on enhanced features.
    balancing_methods = [
        "none",
        "random_oversampling",
        "smote",
        "smoteenn",
        "random_undersampling",
        "cost_sensitive",
    ]
    for balancing in balancing_methods:
        for model_name in ["LogisticRegression", "RandomForest", "XGBoost", "MLP"]:
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
                    rule_cost,
                )
            )

    result_df = pd.DataFrame([r.__dict__ for r in results]).sort_values(["study", "f1"], ascending=[True, False])
    result_df.to_csv(RESULTS_DIR / "model_results.csv", index=False)

    coverage = _coverage_report(result_df)
    coverage.to_csv(RESULTS_DIR / "paper_target_validation.csv", index=False)

    _plot_results(result_df)

    # Enumerate paper experiments and what we replicated.
    experiment_inventory = pd.DataFrame(
        [
            {"paper_experiment": "E1 spatial-temporal feature impact", "paper_count": 1, "implemented": True},
            {"paper_experiment": "E2 balancing method comparison (6 methods)", "paper_count": 6, "implemented": True},
            {"paper_experiment": "E3 model family comparison (LR/DT/RF/XGB/NN)", "paper_count": 5, "implemented": True},
            {"paper_experiment": "E4 deep sequential models (LSTM/attention/CNN)", "paper_count": 3, "implemented": False},
            {"paper_experiment": "E5 ensemble comparison (stacking, hybrid XGB-LSTM)", "paper_count": 2, "implemented": False},
            {"paper_experiment": "E6 rule-based benchmark and cost-benefit", "paper_count": 1, "implemented": True},
            {"paper_experiment": "E7 interpretability (SHAP/attention)", "paper_count": 1, "implemented": False},
        ]
    )
    experiment_inventory["replication_status"] = np.where(
        experiment_inventory["implemented"], "replicated", "not_yet_replicated"
    )
    experiment_inventory.to_csv(RESULTS_DIR / "paper_experiment_inventory.csv", index=False)

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
        "- `outputs/plots/spatiotemporal_recall.png`",
        "- `outputs/plots/spatiotemporal_f1.png`",
        "- `outputs/plots/balancing_heatmap_f1.png`",
        "- `outputs/plots/benchmark_savings.png`",
    ]
    (RESULTS_DIR / "results_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    return result_df
