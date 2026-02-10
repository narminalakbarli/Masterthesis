# Reproducibility Tutorial

This tutorial explains how to reproduce the thesis-aligned experiments and regenerate all plots.

## 1) Prerequisites

- Linux/macOS shell (or WSL on Windows)
- Python 3.10+
- Internet access for dependency installation and dataset download

## 2) Clone and enter the repository

```bash
git clone <your-fork-or-repo-url>
cd Masterthesis
```

## 3) Create the virtual environment and install dependencies

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
```

This installs all packages from `requirements.txt` into `.venv`.

## 4) Download the fraud dataset

```bash
python scripts/download_kaggle_data.py --out-dir data
```

### Data source behavior
The downloader/pipeline will use this order:
1. `data/creditcard.csv` (if already present)
2. Kaggle API (`mlg-ulb/creditcardfraud`)
3. Public mirror fallback
4. Synthetic fallback data

> If Kaggle credentials are missing (`~/.kaggle/kaggle.json`), the script still works using the public mirror.

## 5) Extract thesis chapter notes (optional but recommended)

```bash
python scripts/extract_thesis_chapters.py
```

This creates:
- `docs/thesis_chapter_notes.md`

## 6) Run the full reproduction pipeline

### Recommended thesis-like run
```bash
python -m src.thesis_repro.run_pipeline --sample-size 80000
```

### Faster smoke-test run
```bash
python -m src.thesis_repro.run_pipeline --sample-size 20000
```

## 7) Verify generated outputs

After the pipeline finishes, confirm these files exist:

- `outputs/model_results.csv`
- `outputs/results_summary.md`
- `outputs/paper_target_validation.csv`
- `outputs/paper_experiment_inventory.csv`
- `outputs/plots/spatiotemporal_recall.png`
- `outputs/plots/spatiotemporal_f1.png`
- `outputs/plots/balancing_heatmap_f1.png`
- `outputs/plots/benchmark_savings.png`

Example check:

```bash
find outputs -maxdepth 3 -type f | sort
```

## 8) Interpreting results quickly

- `model_results.csv`: all experiment-level metrics (recall, precision, F1, AUC, cost-based metric).
- `results_summary.md`: narrative summary of best-performing approaches.
- `paper_target_validation.csv`: compares achieved metrics to thesis target values.
- `paper_experiment_inventory.csv`: coverage audit of paper experiments that are/aren't currently implemented.
- `outputs/plots/*.png`: publication-friendly figures for key comparisons.

## 9) Common issues and fixes

- **Long runtime**: reduce `--sample-size` for faster checks.
- **Kaggle auth errors**: rely on the built-in public mirror fallback.
- **MLP convergence warning**: expected for short training budgets; does not block output generation.
- **No plot files**: ensure pipeline completed successfully and the process was not interrupted.

## 10) Minimal command checklist

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
python scripts/download_kaggle_data.py --out-dir data
python scripts/extract_thesis_chapters.py
python -m src.thesis_repro.run_pipeline --sample-size 80000
find outputs -maxdepth 3 -type f | sort
```
