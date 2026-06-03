# CREL-Census

Replication package for the establishment closure benchmarks in *Closing Time* (CREL / U.S. Census Bureau, Spring 2026).

## Files

- `data/establishment_quarter_panel.parquet` — merged training panel (Git LFS, ~1.7 GB)
- `data/feature_categories.json` — `STATIC`, `QUARTERLY`, and `LAGGED` column lists
- `scripts/train_closure_benchmark.py` — trains all models
- `scripts/train_combined_catboost.py` — imported sampling utilities (not run directly)
- `requirements.txt`

## Reproduce the twenty-four paper models

```bash
git lfs install
git clone https://github.com/water6973/CREL-Census.git
cd CREL-Census
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/train_closure_benchmark.py --v2
```

This runs 2 outcomes $\times$ 4 feature specifications (`static_only`, `quarterly_only`, `quarterly_lagged`, `full`) $\times$ 3 algorithms (CatBoost, XGBoost, NGBoost). The `--v2` preset matches the paper (seasonal quarter encoding, exclude categorical `QUARTER`, eight hyperparameter trials, temporal split through 2024Q2 validation).

Outputs: `outputs/benchmark/`. Sample caches: `outputs/sample_cache/` (created locally, gitignored).
