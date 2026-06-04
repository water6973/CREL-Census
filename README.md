# CREL-Census

Replication package for the establishment closure benchmarks (CREL / U.S. Census Bureau).

## What you run

One command trains **all 24 models** from **one dataset**:

```bash
python scripts/train_closure_models.py --v2
```

That reads `data/establishment_quarter_panel.parquet` and fits:

| | |
|--|--|
| Outcomes | `panel_period`, `next_quarter` |
| Feature specs | `static_only`, `quarterly_only`, `quarterly_lagged`, `full` |
| Algorithms | CatBoost, XGBoost, NGBoost |

$2 \times 4 \times 3 = 24$ models. `--v2` matches the paper settings.

**Full panel:** Every row in each split is used. There are no row caps, no negative downsampling, and no chunked scoring options. The script loads the complete train, validation, and test splits into memory, fits each model once, and scores every held-out row.

| Split | Panel-period (establishment holdout) | Next-quarter (temporal) |
|-------|--------------------------------------|-------------------------|
| Train | ~803k establishments | ~20.5M establishment-quarters |
| Validation | ~172k | ~2.2M |
| Test | ~172k | ~4.3M (2024Q3--2025Q2) |

Expect substantial RAM for the next-quarter task (tens of millions of rows). A machine with at least 64GB is recommended for the full next-quarter pipeline; panel-period alone is much smaller (~1.15M establishments).

Use `--force-resample` to rebuild split caches after updating the parquet or split logic.

## Scripts

| File | Role |
|------|------|
| **`train_closure_models.py`** | Entry point: training loop, metrics, saved models. |
| **`panel_sampling.py`** | Parquet streaming into full train/val/test caches (imported by the trainer). |

## Files

- `data/establishment_quarter_panel.parquet` (Git LFS)
- `data/feature_categories.json`
- `requirements.txt`

## Setup

```bash
git lfs install
git clone https://github.com/water6973/CREL-Census.git
cd CREL-Census
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/train_closure_models.py --v2 --force-resample
```

Outputs: `outputs/benchmark_v2/{target}/{feature_set}/{algo}/`
