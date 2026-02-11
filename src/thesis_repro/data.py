from __future__ import annotations

import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.datasets import make_classification

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
RAW_PATH = DATA_DIR / "creditcard.csv"
AURELIUS_PRIOR_PATH = DATA_DIR / "aurelius_context_prior.csv"
KAGGLE_DATASET = "mlg-ulb/creditcardfraud"
KAGGLE_ZIP = DATA_DIR / "creditcardfraud.zip"
LOCAL_ARCHIVE = Path("kaggle.zip")
PUBLIC_MIRROR_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"


DATASET_REGISTRY = {
    "ULB_Kaggle_CreditCardFraud": {
        "type": "real",
        "source": "mlg-ulb/creditcardfraud",
        "local_path": str(RAW_PATH),
    },
    "PublicMirror_CreditCard": {
        "type": "real_mirror",
        "source": PUBLIC_MIRROR_URL,
        "local_path": str(RAW_PATH),
    },
    "AureliusQuantizedContext_Augmented": {
        "type": "synthetic_feature_layer",
        "source": "generated via add_spatiotemporal_and_synthetic_features",
        "local_path": "in-memory augmentation",
    },
    "AureliusQuantizedContext_ExternalPrior": {
        "type": "synthetic_external",
        "source": "Aurelius Quantized Transactional Context Corpus (optional local CSV prior)",
        "local_path": str(AURELIUS_PRIOR_PATH),
    },
    "SyntheticCreditCardFallback": {
        "type": "synthetic_full",
        "source": "sklearn.make_classification",
        "local_path": "generated on demand",
    },
}


def _extract_creditcard_csv(zip_path: Path, path: Path = RAW_PATH) -> bool:
    if not zip_path.exists():
        return False

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            csv_members = [name for name in zf.namelist() if name.lower().endswith("creditcard.csv")]
            if not csv_members:
                csv_members = [name for name in zf.namelist() if name.lower().endswith(".csv")]
            if not csv_members:
                return False

            extracted = Path(zf.extract(csv_members[0], DATA_DIR))

        if extracted != path:
            if path.exists():
                path.unlink()
            extracted.replace(path)
            try:
                extracted.parent.rmdir()
            except OSError:
                pass

        return path.exists()
    except Exception:
        return False


def _load_from_local_archive(path: Path = RAW_PATH) -> bool:
    candidates = [
        DATA_DIR / LOCAL_ARCHIVE.name,
        Path.cwd() / LOCAL_ARCHIVE.name,
    ]
    for candidate in candidates:
        if _extract_creditcard_csv(candidate, path=path):
            return True
    return False


def _download_from_kaggle(path: Path = RAW_PATH) -> bool:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception:
        return False

    has_auth = (Path.home() / ".kaggle" / "kaggle.json").exists() or (
        os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")
    )
    if not has_auth:
        return False

    try:
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(KAGGLE_DATASET, path=str(DATA_DIR), quiet=True, unzip=False)
        if KAGGLE_ZIP.exists():
            ok = _extract_creditcard_csv(KAGGLE_ZIP, path=path)
            KAGGLE_ZIP.unlink(missing_ok=True)
            return ok
        return path.exists()
    except Exception:
        return False


def _download_from_public_mirror(path: Path = RAW_PATH) -> bool:
    try:
        response = requests.get(PUBLIC_MIRROR_URL, timeout=120)
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
        return pd.read_csv(RAW_PATH)

    # Prefer local Kaggle archive, then Kaggle API (same source as thesis), then public mirror, then synthetic.
    if _load_from_local_archive(RAW_PATH):
        return pd.read_csv(RAW_PATH)

    if _download_from_kaggle(RAW_PATH):
        return pd.read_csv(RAW_PATH)

    if _download_from_public_mirror(RAW_PATH):
        return pd.read_csv(RAW_PATH)

    return _generate_synthetic_creditcard()


def add_spatiotemporal_and_synthetic_features(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    out = df.copy()

    seconds = out["Time"].to_numpy()
    out["hour"] = ((seconds // 3600) % 24).astype(int)
    out["day"] = ((seconds // (3600 * 24)) % 7).astype(int)
    out = out.sort_values("Time").reset_index(drop=True)
    out["inter_txn_seconds"] = out["Time"].diff().fillna(out["Time"].median()).clip(lower=0)
    out["amount_log"] = np.log1p(out["Amount"])

    n = len(out)

    # If an external Aurelius prior file is available locally, borrow its
    # categorical context profile and align rows by quantized (Time, Amount).
    # Otherwise, fall back to thesis-inspired synthetic generation.
    if AURELIUS_PRIOR_PATH.exists():
        prior_df = pd.read_csv(AURELIUS_PRIOR_PATH)
        required = {"Time", "Amount", "channel", "same_state", "merchant_category"}
        if required.issubset(prior_df.columns):
            key_self = pd.DataFrame(
                {
                    "Time_key": out["Time"].round(0).astype(int),
                    "Amount_key": out["Amount"].round(2),
                }
            )
            key_prior = pd.DataFrame(
                {
                    "Time_key": prior_df["Time"].round(0).astype(int),
                    "Amount_key": prior_df["Amount"].round(2),
                    "channel": prior_df["channel"],
                    "same_state": prior_df["same_state"],
                    "merchant_category": prior_df["merchant_category"],
                }
            )
            merged = key_self.merge(key_prior, on=["Time_key", "Amount_key"], how="left")
            out["channel"] = merged["channel"]
            out["same_state"] = merged["same_state"]
            out["merchant_category"] = merged["merchant_category"]
        else:
            out["channel"] = np.nan
            out["same_state"] = np.nan
            out["merchant_category"] = np.nan
    else:
        out["channel"] = np.nan
        out["same_state"] = np.nan
        out["merchant_category"] = np.nan

    out["channel"] = out["channel"].fillna(
        pd.Series(rng.choice(["chip", "swipe", "online"], size=n, p=[0.744, 0.146, 0.11]))
    )
    out["same_state"] = out["same_state"].fillna(pd.Series(rng.choice([1, 0], size=n, p=[0.6383, 0.3617]))).astype(int)
    out["merchant_category"] = out["merchant_category"].fillna(
        pd.Series(
            rng.choice(
                ["grocery", "fuel", "entertainment", "travel", "retail", "services"],
                size=n,
                p=[0.29, 0.12, 0.08, 0.07, 0.34, 0.10],
            )
        )
    )
    out["card_present"] = (out["channel"] != "online").astype(int)

    regional_centers = np.array(
        [[52.52, 13.41], [48.14, 11.58], [50.11, 8.68], [45.46, 9.19], [41.39, 2.17], [48.86, 2.35]]
    )
    merchant_idx = rng.integers(0, len(regional_centers), size=n)
    home_idx = rng.integers(0, len(regional_centers), size=n)
    merchant = regional_centers[merchant_idx]
    home = regional_centers[home_idx]

    out["geo_distance_km"] = np.sqrt(((merchant[:, 0] - home[:, 0]) * 111) ** 2 + ((merchant[:, 1] - home[:, 1]) * 73) ** 2)
    out["merchant_region"] = merchant_idx
    out["home_region"] = home_idx
    out["region_changed"] = (out["merchant_region"] != out["home_region"]).astype(int)

    return out
