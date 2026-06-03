# CREL-Census replication

Minimal artifacts to reproduce the **results tables and precision--recall figure** in:

*Closing Time: Predicting Next-Quarter Establishment Closures for the Census Bureau* (Charles River Economics Labs, Spring 2026).

## Files

- `results_table_prf.csv` — test precision, recall, and $F_1$ (validation-tuned threshold) for all models and feature sets
- `figures/pr_overlay_6panel.pdf` — six-panel precision--recall figure in the paper
- `reproduce_results.py` — writes `results_tables.tex` for inclusion in the LaTeX paper

Raw licensed inputs (SafeGraph, Rhetorik/Dewey, etc.) and retraining code are not included.

## Usage

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python reproduce_results.py
```

Copy `results_tables.tex` and `figures/pr_overlay_6panel.pdf` into the paper project, then compile `paper/main.tex`.
