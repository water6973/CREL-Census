# CREL-Census

Replication package for the establishment closure benchmarks (CREL / U.S. Census Bureau).

## What you run

One command trains **all 24 models** from **one dataset**:

```bash
python scripts/train_closure_benchmark.py --v2
```

That reads a single file, `data/establishment_quarter_panel.parquet`, and fits:

| | |
|--|--|
| Outcomes | `panel_period` (closure in window), `next_quarter` (`CLOSED_NEXT_QUARTER`) |
| Feature specs | `static_only`, `quarterly_only`, `quarterly_lagged`, `full` |
| Algorithms | CatBoost, XGBoost, NGBoost |

$2 \times 4 \times 3 = 24$ models. The `--v2` flag matches the paper (seasonal quarter encoding, drop categorical `QUARTER`, eight hyperparameter trials, train through 2023Q4 / validate through 2024Q2).

## Why two Python files?

| File | Role |
|------|------|
| **`train_closure_benchmark.py`** | Entry point. Training loop, metrics, models, PR plots. |
| **`train_combined_catboost.py`** | **Not a second training pipeline.** Shared code for reading the parquet in chunks, building train/val/test row samples, and caching them under `outputs/sample_cache/`. `train_closure_benchmark.py` imports it; you do not run it separately. |

The second file is named for an older internal script. Everything still uses the same `establishment_quarter_panel.parquet`.

## Files in the repo

- `data/establishment_quarter_panel.parquet` — merged panel (Git LFS)
- `data/feature_categories.json` — column lists for the four feature specs
- `requirements.txt`

## Setup

```bash
git lfs install
git clone https://github.com/water6973/CREL-Census.git
cd CREL-Census
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/train_closure_benchmark.py --v2
```

Outputs: `outputs/benchmark/{target}/{feature_set}/{algo}/`
