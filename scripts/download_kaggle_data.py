from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path

import requests

DATASET = "mlg-ulb/creditcardfraud"
DEFAULT_OUT = Path("data")
KAGGLE_ZIP = "creditcardfraud.zip"
LOCAL_ARCHIVE = Path("kaggle.zip")
MIRROR_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"


def _extract_creditcard_csv(zip_path: Path, out_dir: Path) -> bool:
    if not zip_path.exists():
        return False

    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_members = [name for name in zf.namelist() if name.lower().endswith("creditcard.csv")]
        if not csv_members:
            csv_members = [name for name in zf.namelist() if name.lower().endswith(".csv")]
        if not csv_members:
            return False

        extracted = Path(zf.extract(csv_members[0], out_dir))

    target = out_dir / "creditcard.csv"
    if extracted != target:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.unlink()
        extracted.replace(target)
        try:
            extracted.parent.rmdir()
        except OSError:
            pass

    return target.exists()


def _extract_local_archive(out_dir: Path) -> bool:
    candidates = [
        out_dir / LOCAL_ARCHIVE.name,
        Path.cwd() / LOCAL_ARCHIVE.name,
    ]
    for candidate in candidates:
        if _extract_creditcard_csv(candidate, out_dir):
            return True
    return False


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

    extracted = _extract_creditcard_csv(zip_path, out_dir)
    zip_path.unlink(missing_ok=True)
    return extracted


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

    local_archive_ok = _extract_local_archive(out_dir)
    if local_archive_ok:
        print(f"Extracted local Kaggle archive -> {out_dir / 'creditcard.csv'}")
        return

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
        "Failed to obtain data from local archive, Kaggle API, and public mirror. "
        "Check network or Kaggle credentials."
    )


if __name__ == "__main__":
    main()
