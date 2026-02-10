from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    fbeta_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .data import add_spatiotemporal_and_synthetic_features, load_base_dataset

RESULTS_DIR = Path("outputs")
RESULTS_DIR.mkdir(exist_ok=True)


@dataclass
class EvalResult:
    experiment: str
    model: str
    threshold: float
    recall: float
    precision: float
    f1: float
    f2: float
    auc_roc: float
    auc_pr: float
    net_savings_rel_rule_based: float


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
    best_t = 0.5
    best_f2 = -1.0
    for t in candidates:
        pred = (y_score >= t).astype(int)
        f2 = fbeta_score(y_true, pred, beta=2, zero_division=0)
        if f2 > best_f2:
            best_f2 = f2
            best_t = float(t)
    return best_t


def _run_single_experiment(df: pd.DataFrame, feature_set_name: str, features: list[str]) -> list[EvalResult]:
    # Temporal ordering to reduce leakage risk.
    df = df.sort_values("Time").reset_index(drop=True)
    y = df["Class"].to_numpy()

    split_idx = int(len(df) * 0.8)
    X_train = df.iloc[:split_idx][features]
    y_train = y[:split_idx]
    X_test = df.iloc[split_idx:][features]
    X_test_full = df.iloc[split_idx:]
    y_test = y[split_idx:]

    # Carve out validation from train for threshold tuning.
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
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

    # Leakage-safe balancing (only training subset).
    smote = SMOTE(sampling_strategy=0.15, random_state=42, k_neighbors=3)
    X_tr_bal, y_tr_bal = smote.fit_resample(X_tr_proc, y_tr)

    models = {
        "LogisticRegression+SMOTE": LogisticRegression(max_iter=800, n_jobs=-1),
        "RandomForest+SMOTE": RandomForestClassifier(
            n_estimators=220, max_depth=14, random_state=42, n_jobs=-1
        ),
        "XGBoost+SMOTE": XGBClassifier(
            n_estimators=120,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        ),
    }

    models["Stacking(RF+XGB)+SMOTE"] = StackingClassifier(
        estimators=[("rf", models["RandomForest+SMOTE"]), ("xgb", models["XGBoost+SMOTE"])],
        final_estimator=LogisticRegression(max_iter=400),
        n_jobs=-1,
    )

    rule_pred = _rule_based_scores(X_test_full)
    rule_cost = _cost_benefit(y_test, rule_pred)
    rule_proba = np.clip(rule_pred.astype(float) * 0.9 + 0.05, 0, 1)

    results: list[EvalResult] = [
        EvalResult(
            experiment=feature_set_name,
            model="RuleBasedBenchmark",
            threshold=0.5,
            recall=recall_score(y_test, rule_pred, zero_division=0),
            precision=precision_score(y_test, rule_pred, zero_division=0),
            f1=f1_score(y_test, rule_pred, zero_division=0),
            f2=fbeta_score(y_test, rule_pred, beta=2, zero_division=0),
            auc_roc=roc_auc_score(y_test, rule_proba),
            auc_pr=average_precision_score(y_test, rule_proba),
            net_savings_rel_rule_based=1.0,
        )
    ]

    for name, model in models.items():
        model.fit(X_tr_bal, y_tr_bal)
        val_score = model.predict_proba(X_val_proc)[:, 1]
        best_t = _best_threshold(y_val, val_score)

        test_score = model.predict_proba(X_test_proc)[:, 1]
        pred = (test_score >= best_t).astype(int)
        cost = _cost_benefit(y_test, pred)
        savings = rule_cost / cost if cost > 0 else np.inf

        results.append(
            EvalResult(
                experiment=feature_set_name,
                model=name,
                threshold=best_t,
                recall=recall_score(y_test, pred, zero_division=0),
                precision=precision_score(y_test, pred, zero_division=0),
                f1=f1_score(y_test, pred, zero_division=0),
                f2=fbeta_score(y_test, pred, beta=2, zero_division=0),
                auc_roc=roc_auc_score(y_test, test_score),
                auc_pr=average_precision_score(y_test, test_score),
                net_savings_rel_rule_based=float(savings),
            )
        )

    return results


def run(sample_size: int = 80_000) -> pd.DataFrame:
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

    all_results: list[EvalResult] = []
    all_results.extend(_run_single_experiment(enhanced, "baseline_features", baseline_features))
    all_results.extend(_run_single_experiment(enhanced, "enhanced_features", enhanced_features))

    result_df = pd.DataFrame([r.__dict__ for r in all_results])
    result_df = result_df.sort_values(["experiment", "f1"], ascending=[True, False]).reset_index(drop=True)
    result_df.to_csv(RESULTS_DIR / "model_results.csv", index=False)

    # Aggregate view for easier comparison.
    summary = (
        result_df[result_df["model"] != "RuleBasedBenchmark"]
        .groupby("experiment", as_index=False)[["recall", "precision", "f1", "f2", "auc_roc", "auc_pr"]]
        .mean()
    )

    md = [
        "# Experiment results",
        "",
        "## Model-level metrics",
        result_df.to_markdown(index=False),
        "",
        "## Average by feature set (non-rule models)",
        summary.to_markdown(index=False),
    ]
    (RESULTS_DIR / "results_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    return result_df
