# USAGE

This document explains the full end-to-end workflow for this repository:
- how to set up the environment,
- how data is sourced,
- how experiment configuration works,
- how to run the pipeline,
- where outputs/plots are written,
- and what to expect from the generated artifacts.

---

## 1) What this repository does

The project reproduces thesis-aligned credit-card fraud experiments using:
- baseline features (`V1..V28`, `Time`, `Amount`),
- enhanced spatial-temporal/context features,
- multiple balancing strategies,
- a model bank (classical ML, neural proxies, anomaly proxies),
- ensemble variants,
- and cost/benchmark comparisons.

Main execution entrypoint:

```bash
python -m src.thesis_repro.run_pipeline
```

---

## 2) Environment setup

From repository root:

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
```

This creates `.venv` and installs dependencies from `requirements.txt`.

---

## 3) Data sources and fallback order

The pipeline resolves data in this order:

1. `data/creditcard.csv` if already available.
2. Local archive (`kaggle.zip` / extracted CSV) if present.
3. Kaggle dataset (`mlg-ulb/creditcardfraud`) via credentials.
4. Public mirror fallback.
5. Synthetic fallback dataset if all remote/local sources fail.

You can pre-download explicitly:

```bash
python scripts/download_kaggle_data.py --out-dir data
```

### Optional external context prior

If available, place:

```text
data/aurelius_context_prior.csv
```

Expected columns used for context alignment:
- `Time`
- `Amount`
- `channel`
- `same_state`
- `merchant_category`

If missing, the pipeline still runs and generates synthetic context features.

---

## 4) Configuration model

Primary config file:

```text
config/experiment_config.json
```

The repository is config-driven. This JSON controls:
- global run settings (`sample_size`, `random_state`),
- split strategy (`splits`),
- preprocessing behavior (`preprocessing`),
- threshold search space and beta (`threshold`),
- cost model (`costs`),
- rule-based benchmark rules (`rule_based`),
- sampler enable flags + hyperparameters (`samplers`),
- model enable flags + hyperparameters (`models`),
- ensemble composition (`ensembles`),
- experiment matrix (`studies`),
- reference tables (`paper_targets`, `paper_experiment_inventory`),
- output paths and filenames (`outputs`).


### Runtime acceleration (CPU/GPU toggle)

You can enable GPU acceleration from config:

```json
"runtime": {
  "gpu": {
    "enabled": true,
    "device": "cuda",
    "catboost_devices": "0"
  }
}
```

- `runtime.gpu.enabled=false` (default): run all models on CPU.
- `runtime.gpu.enabled=true`: enables GPU mode for XGBoost (`device`) and CatBoost (`task_type=GPU`, `devices`).
- If a compatible GPU is not available, the pipeline retries those models on CPU automatically.

### CLI overrides

`run_pipeline` accepts optional overrides:

```bash
python -m src.thesis_repro.run_pipeline \
  --config config/experiment_config.json \
  --sample-size 40000 \
  --seed 42
```

- If `--config` is omitted, default is `config/experiment_config.json`.
- `--sample-size` and `--seed` override config values only for that run.

---

## 5) Run the pipeline (end-to-end)

Recommended full workflow:

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
python scripts/download_kaggle_data.py --out-dir data
python scripts/extract_thesis_chapters.py
python -m src.thesis_repro.run_pipeline
```

Fast validation run:

```bash
python -m src.thesis_repro.run_pipeline --sample-size 4000 --seed 42
```

---

## 6) Output locations

By default (from config), outputs are written under:

```text
outputs/
outputs/plots/
```

Expected artifacts:

- `outputs/model_results.csv`
- `outputs/paper_target_validation.csv`
- `outputs/paper_experiment_inventory.csv`
- `outputs/results_summary.md`
- `outputs/plots/spatiotemporal_recall.png`
- `outputs/plots/spatiotemporal_f1.png`
- `outputs/plots/balancing_heatmap_f1.png`
- `outputs/plots/benchmark_savings.png`

Quick existence check:

```bash
python - <<'PY'
from pathlib import Path
files = [
    'outputs/model_results.csv',
    'outputs/paper_target_validation.csv',
    'outputs/paper_experiment_inventory.csv',
    'outputs/results_summary.md',
    'outputs/plots/spatiotemporal_recall.png',
    'outputs/plots/spatiotemporal_f1.png',
    'outputs/plots/balancing_heatmap_f1.png',
    'outputs/plots/benchmark_savings.png',
]
for f in files:
    p = Path(f)
    print(f"{f}: {'OK' if p.exists() else 'MISSING'}")
PY
```

---

## 7) What to expect in outputs

### `model_results.csv`
Contains one row per executed experiment case with metrics such as:
- `recall`, `precision`, `f1`, `f2`,
- `auc_roc`, `auc_pr`,
- `threshold`,
- `net_savings_rel_rule_based`,
- timing (`fit_seconds`).

### `paper_target_validation.csv`
Compares observed metrics from replicated experiments against thesis target values.

### `paper_experiment_inventory.csv`
Machine-readable inventory of intended paper experiments and replication status.

### `results_summary.md`
Human-readable summary including inventory table, top-performing runs, validation table, and plot references.

### Plot files
Visual summaries for:
- baseline vs enhanced feature impact,
- balancing strategy comparison,
- benchmark savings comparison.

---

## 8) Typical warnings and behavior

- MLP convergence warnings can appear in smaller/faster runs; this is expected with limited iteration budgets.
- Very small samples may produce unstable minority-class metrics.
- If online data sources are unavailable, synthetic fallback allows the pipeline to complete.

---

## 9) Thesis snippet materials

For thesis-ready text snippets and short explanations, see:

```text
snippets/
```

These files are prepared for “Code X.Y” insertion into your thesis document.

---

## 10) Troubleshooting checklist

1. Confirm venv is active (`which python` should point to `.venv`).
2. Confirm config path is valid (`config/experiment_config.json`).
3. Confirm `data/creditcard.csv` exists if you want deterministic local data source.
4. Re-run with smaller `--sample-size` if runtime is too long.
5. Check `outputs/results_summary.md` for a quick sanity check of generated run content.
