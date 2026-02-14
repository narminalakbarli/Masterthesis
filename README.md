# Masterthesis Reproduction: Credit Card Fraud Detection

Reproducible implementation of the thesis **"Machine Learning Methods in Credit Card Fraud Detection"**.

The project runs end-to-end experiments on the ULB/Kaggle credit card fraud dataset, compares feature sets and balancing methods, and generates evaluation tables plus publication-style plots.

## What this repository includes

- **Data ingestion and fallback logic** (local file → Kaggle API → public mirror → synthetic fallback).
- **Feature engineering** inspired by thesis chapters (including spatial-temporal features).
- **Experiment tracks** for:
  - baseline vs enhanced features,
  - imbalance handling methods,
  - model-family comparison,
  - rule-based benchmark and cost comparison.
- **Thesis model coverage** including classical ML (LR/DT/RF/XGBoost/CatBoost), deep proxies (LSTM/CNN/Attention),
  ensemble methods (stacking + hybrid XGB-LSTM), and anomaly models (autoencoder proxy, isolation forest, one-class SVM).
- **Dataset coverage registry** in code for ULB/Kaggle source, public mirror, optional **Aurelius Quantized Transactional Context Corpus** prior (`data/aurelius_context_prior.csv`), Aurelius context augmentation, and synthetic fallback.
- **Generated artifacts**:
  - metrics tables (`model_results.csv`),
  - summary markdown,
  - paper target validation and experiment coverage,
  - plots for recall/F1/balancing/benchmark savings.


## Aurelius prior (naming + rationale)

To avoid person-specific naming and make the method explicit, the optional context dataset is now called the
**Aurelius Quantized Transactional Context Corpus (AQTCC)**.

Why this name:
- **Quantized**: records are aligned by quantized keys `(⌈Time⌉, round(Amount, 2))`.
- **Transactional Context**: it contributes contextual categorical priors (`channel`, `same_state`, `merchant_category`).
- **Corpus/Prior**: mathematically it behaves as a prior distribution layer for categorical context, i.e.
  \( p(c \mid t, a) \) used when key-matched rows exist, with fallback sampling from calibrated categorical marginals.

## Quickstart

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
python scripts/download_kaggle_data.py --out-dir data
python scripts/extract_thesis_chapters.py
python -m src.thesis_repro.run_pipeline
```

If Kaggle credentials are not configured, the downloader automatically uses a public mirror.

The experiment config in `config/experiment_config.json` now contains split settings, thresholding, costs, sampler parameters, model hyperparameters, ensemble setup, study matrices, runtime (CPU/GPU) toggles, and output paths.

### Optional GPU acceleration

GPU usage is **disabled by default** and can be toggled in `config/experiment_config.json`:

```json
"runtime": {
  "gpu": {
    "enabled": true,
    "device": "cuda",
    "catboost_devices": "0"
  }
}
```

- `runtime.gpu.enabled: true` enables GPU mode for XGBoost and CatBoost.
- If GPU is unavailable, the pipeline automatically falls back to CPU for those models.

---


## Thesis code snippets

For thesis-ready code excerpts (text blocks with short explanations), see:

- **[`snippets/`](snippets/README.md)**

## Reproducibility & usage guides

For detailed end-to-end usage and reproducibility instructions, see:

- **[`USAGE.md`](USAGE.md)**
- **[`docs/reproducibility_tutorial.md`](docs/reproducibility_tutorial.md)**

---

## Repository structure

- `scripts/` – environment setup, data download, thesis chapter extraction.
- `src/thesis_repro/` – core data, experiment, and pipeline code.
- `docs/` – generated thesis chapter notes and reproducibility documentation.
- `outputs/` – generated experiment artifacts and plots.

## Main output files

After a successful run, the pipeline writes:

- `outputs/model_results.csv`
- `outputs/results_summary.md`
- `outputs/paper_target_validation.csv`
- `outputs/paper_experiment_inventory.csv`
- `outputs/plots/spatiotemporal_recall.png`
- `outputs/plots/spatiotemporal_f1.png`
- `outputs/plots/balancing_heatmap_f1.png`
- `outputs/plots/benchmark_savings.png`

## Notes

- The project is designed to be reproducible without paid services.
- Full-size runs may take significantly longer than smoke tests depending on CPU/RAM.
- Some models (e.g., MLP) may emit convergence warnings under short iteration budgets; this is expected for quick reproducibility runs.
