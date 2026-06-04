# CREL-Census

Replication package for the establishment closure benchmarks (CREL / U.S. Census Bureau).

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
