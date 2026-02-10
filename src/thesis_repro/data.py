from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.datasets import make_classification

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
RAW_PATH = DATA_DIR / "creditcard.csv"
THESIS_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"


def _download_creditcard_csv(path: Path = RAW_PATH) -> bool:
    try:
        response = requests.get(THESIS_URL, timeout=60)
        response.raise_for_status()
        path.write_bytes(response.content)
        return True
    except Exception:
        return False


def _generate_synthetic_creditcard(n_samples: int = 200_000, random_state: int = 42) -> pd.DataFrame:
    X, y = make_classification(
        n_samples=n_samples,
        n_features=30,
        n_informative=12,
        n_redundant=10,
        n_repeated=0,
        n_classes=2,
        weights=[0.9983, 0.0017],
        class_sep=1.2,
        random_state=random_state,
    )
    cols = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
    df = pd.DataFrame(X, columns=cols)
    df["Time"] = np.abs(df["Time"] * 3600)
    df["Amount"] = np.abs(df["Amount"] * 80)
    df["Class"] = y
    return df


def load_base_dataset() -> pd.DataFrame:
    if RAW_PATH.exists():
        df = pd.read_csv(RAW_PATH)
        return df

    ok = _download_creditcard_csv(RAW_PATH)
    if ok:
        return pd.read_csv(RAW_PATH)

    return _generate_synthetic_creditcard()


def add_spatiotemporal_and_synthetic_features(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    out = df.copy()

    # Temporal features inspired by thesis.
    seconds = out["Time"].to_numpy()
    out["hour"] = ((seconds // 3600) % 24).astype(int)
    out["day"] = ((seconds // (3600 * 24)) % 7).astype(int)
    out = out.sort_values("Time").reset_index(drop=True)
    out["inter_txn_seconds"] = out["Time"].diff().fillna(out["Time"].median()).clip(lower=0)
    out["amount_log"] = np.log1p(out["Amount"])

    # Synthetic transactional context (channel, same_state, card_present, merchant category).
    n = len(out)
    out["channel"] = rng.choice(["chip", "swipe", "online"], size=n, p=[0.744, 0.146, 0.11])
    out["same_state"] = rng.choice([1, 0], size=n, p=[0.6383, 0.3617])
    out["card_present"] = (out["channel"] != "online").astype(int)
    out["merchant_category"] = rng.choice(
        ["grocery", "fuel", "entertainment", "travel", "retail", "services"],
        size=n,
        p=[0.29, 0.12, 0.08, 0.07, 0.34, 0.10],
    )

    # Approximate geolocation distance (km) and region-change signal.
    regional_centers = np.array([
        [52.52, 13.41],
        [48.14, 11.58],
        [50.11, 8.68],
        [45.46, 9.19],
        [41.39, 2.17],
        [48.86, 2.35],
    ])
    merchant_idx = rng.integers(0, len(regional_centers), size=n)
    home_idx = rng.integers(0, len(regional_centers), size=n)
    merchant = regional_centers[merchant_idx]
    home = regional_centers[home_idx]

    # Lightweight distance approximation (euclidean on lat/lon scaled).
    out["geo_distance_km"] = np.sqrt(((merchant[:, 0] - home[:, 0]) * 111) ** 2 + ((merchant[:, 1] - home[:, 1]) * 73) ** 2)
    out["merchant_region"] = merchant_idx
    out["home_region"] = home_idx
    out["region_changed"] = (out["merchant_region"] != out["home_region"]).astype(int)

    return out
