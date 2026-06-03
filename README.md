# CREL-Census

Training data and scripts for establishment closure models (Charles River Economics Labs / Census Bureau project).

## Contents

| Path | Description |
|------|-------------|
| `data/establishment_quarter_panel.parquet` | Establishment-quarter panel used to train all benchmark models (~27M rows, 63 features, 2019Q3--2025Q2). **Git LFS** (~1.7 GB). |
| `data/feature_categories.json` | Feature groupings (static / quarterly / lag) for model specs |
| `scripts/train_closure_benchmark.py` | Train CatBoost, XGBoost, and NGBoost; writes metrics and models under `outputs/benchmark/` |
| `scripts/train_combined_catboost.py` | Parquet streaming and sample-cache helpers (imported by the benchmark script) |

## Setup

```bash
git lfs install
git clone https://github.com/water6973/CREL-Census.git
cd CREL-Census
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train models

```bash
python scripts/train_closure_benchmark.py
```

Optional flags: `--hp-trials 8`, `--output-dir outputs/benchmark`, `--parquet data/establishment_quarter_panel.parquet`.

First run builds row samples from the parquet and caches them under `outputs/sample_cache/` (large; created locally, not committed).

## Outputs

Written to `outputs/benchmark/` by default: trained models, `metrics.json`, precision--recall plots, and `results_table_prf.csv` per run.
