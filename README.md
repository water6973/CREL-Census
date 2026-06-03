# CREL-Census

Training panel and scripts to re-estimate establishment closure models (CREL / U.S. Census Bureau pilot, Spring 2026). Raw vendor files are omitted for licensing. The parquet is the analysis copy built from SafeGraph Places, Rhetorik (Dewey), BLS LAUS, FHFA HPI, BEA county GDP/income, Census BFS, Census QWI, and SBA 7(a)/504 FOIA loans.

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
