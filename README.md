# CREL-Census

Training panel and scripts to re-estimate establishment closure models (CREL / U.S. Census Bureau pilot, Spring 2026).

## What is in the repo

| File | Why it is here |
|------|----------------|
| `data/establishment_quarter_panel.parquet` | Single merged establishment-quarter file (~27M rows, 63 columns, 2019Q3--2025Q2). All external sources are already joined here; you do not need raw SafeGraph or Rhetorik drops to rerun the benchmarks. **Git LFS** (~1.7 GB). |
| `data/feature_categories.json` | Lists which columns are `STATIC`, `QUARTERLY`, or `LAGGED`. The training script builds four feature blocks from these lists. |
| `scripts/train_closure_benchmark.py` | **Entry point.** Trains CatBoost, XGBoost, and NGBoost for two targets × four feature blocks; writes models and metrics under `outputs/benchmark/`. |
| `scripts/train_combined_catboost.py` | **Not a second pipeline.** Parquet streaming, train/val/test sampling, and sample-cache I/O imported by `train_closure_benchmark.py`. Run `train_closure_benchmark.py` only. |

Raw vendor files are omitted for licensing. The parquet is the analysis copy built from SafeGraph Places, Rhetorik (Dewey), BLS LAUS, FHFA HPI, BEA county GDP/income, Census BFS, Census QWI, and SBA 7(a)/504 FOIA loans.

## Feature blocks (matches the paper and `train_closure_benchmark.py`)

These are **separate** specs, not “static, then add quarterly, then add lags”:

| Script name | Columns used |
|-------------|----------------|
| `static_only` | `STATIC` only |
| `quarterly_only` | `QUARTERLY` only (county macro, QWI, age, etc.) |
| `quarterly_lagged` | `QUARTERLY` + `LAGGED` (eight-quarter rolling means) |
| `full` | `STATIC` + `QUARTERLY` + `LAGGED` |

## Setup and train

```bash
git lfs install
git clone https://github.com/water6973/CREL-Census.git
cd CREL-Census
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/train_closure_benchmark.py
```

First run scans the parquet and writes sample caches under `outputs/sample_cache/` (large; gitignored).

Optional: `--hp-trials 8`, `--output-dir outputs/benchmark`.

## Outputs

`outputs/benchmark/{panel_period,next_quarter}/{feature_set}/{algo}/` — `metrics.json`, models, PR plots. A summary `results_table_prf.csv` is written at the end of a full run.
