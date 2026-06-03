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

## Scripts

| File | Role |
|------|------|
| **`train_closure_models.py`** | Entry point: training loop, metrics, saved models. |
| **`panel_sampling.py`** | Parquet streaming and train/val/test sample caches (imported by the trainer; not run directly). |

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
python scripts/train_closure_models.py --v2
```

Outputs: `outputs/benchmark/{target}/{feature_set}/{algo}/`
