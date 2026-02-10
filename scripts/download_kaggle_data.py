from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path

import requests

DATASET = "mlg-ulb/creditcardfraud"
DEFAULT_OUT = Path("data")
KAGGLE_ZIP = "creditcardfraud.zip"
MIRROR_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"


def _download_via_kaggle_api(out_dir: Path) -> bool:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception:
        return False

    if not (Path.home() / ".kaggle" / "kaggle.json").exists() and not (
        os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")
    ):
        return False

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(DATASET, path=str(out_dir), quiet=False, unzip=False)

    zip_path = out_dir / KAGGLE_ZIP
    if not zip_path.exists():
        return False

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    zip_path.unlink(missing_ok=True)

    return (out_dir / "creditcard.csv").exists()


def _download_via_public_mirror(out_dir: Path) -> bool:
    try:
        r = requests.get(MIRROR_URL, timeout=120)
        r.raise_for_status()
        (out_dir / "creditcard.csv").write_bytes(r.content)
        return True
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Download thesis credit-card data.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT), help="Output directory for dataset files.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    kaggle_ok = _download_via_kaggle_api(out_dir)
    if kaggle_ok:
        print(f"Downloaded {DATASET} via Kaggle API -> {out_dir / 'creditcard.csv'}")
        return

    mirror_ok = _download_via_public_mirror(out_dir)
    if mirror_ok:
        print(
            "Kaggle download unavailable (likely missing API credentials). "
            f"Downloaded public mirror instead -> {out_dir / 'creditcard.csv'}"
        )
        return

    raise SystemExit(
        "Failed to download data from both Kaggle API and public mirror. "
        "Check network or Kaggle credentials."
    )


if __name__ == "__main__":
    main()
