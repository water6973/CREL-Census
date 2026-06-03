#!/usr/bin/env python3
"""
Benchmark closure models: CatBoost, XGBoost, NGBoost × 2 targets × 4 feature sets.

Targets:
  - panel_period   — CLOSED_ON set (establishment holdout)
  - next_quarter   — CLOSED_NEXT_QUARTER (temporal holdout)

Feature sets (rows in results table):
  - static_only, quarterly_only, quarterly_lagged, full

Results table columns (left → right):
  panel: XGBoost | NGBoost | CatBoost | next_quarter: XGBoost | NGBoost | CatBoost

Resume: skips runs already marked done in manifest.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier, Pool
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import ParameterSampler, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import compute_sample_weight
from xgboost import XGBClassifier

# scripts/ngboost.py shadows the ngboost package when running from this directory.
_SCRIPTS = Path(__file__).resolve().parent
_sp = str(_SCRIPTS)
if _sp in sys.path:
    sys.path.remove(_sp)
from ngboost import NGBClassifier  # noqa: E402
if _sp not in sys.path:
    sys.path.insert(0, _sp)

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = _SCRIPTS

from train_combined_catboost import (  # noqa: E402
    CAT_FEATURES,
    DEFAULT_SAMPLE_CACHE,
    LEAK_COLS,
    LAG_SUFFIXES,
    load_sample_cache,
    sample_cache_fingerprint,
    save_sample_cache,
    stream_next_quarter_sample,
    stream_panel_period_sample,
)

DEFAULT_PARQUET = ROOT / "data/establishment_quarter_panel.parquet"
DEFAULT_OUT = ROOT / "outputs/benchmark"
DEFAULT_OUT_NO_QUARTER = ROOT / "outputs/benchmark_no_quarter"
DEFAULT_OUT_V2 = ROOT / "outputs/benchmark"
SEASONAL_COLS = ("QUARTER_SIN", "QUARTER_COS")
TOP_K_FRACS = (0.01, 0.02, 0.05)
CATEGORIES_JSON = ROOT / "data/feature_categories.json"

TARGETS = ("panel_period", "next_quarter")
# Cache folder names match train_combined_catboost.py for resume/reuse.
CACHE_TARGET = {"panel_period": "close_in_panel_period", "next_quarter": "closed_next_quarter"}
ALGOS = ("xgboost", "ngboost", "catboost")
FEATURE_SETS = ("static_only", "quarterly_only", "quarterly_lagged", "full")
COMBINED_CACHE = ROOT / "outputs/sample_cache"

CAT_COLS = CAT_FEATURES


@dataclass
class RunMetrics:
    target: str
    feature_set: str
    algorithm: str
    test_ap: float
    test_prevalence: float
    val_tuned_threshold: float
    test_precision: float
    test_recall: float
    test_f1: float
    n_train: int
    n_test: int
    n_features: int
    train_seconds: float


def load_feature_groups(*, exclude_quarter: bool = False, seasonal_encoding: bool = False) -> dict[str, list[str]]:
    meta = json.loads(CATEGORIES_JSON.read_text(encoding="utf-8"))
    c = meta["categories"]
    static = list(c["STATIC"]["columns"])
    quarterly = list(c["QUARTERLY"]["columns"])
    lagged = list(c["LAGGED"]["columns"])
    if exclude_quarter:
        quarterly = [x for x in quarterly if x != "QUARTER"]
    seasonal = list(SEASONAL_COLS) if seasonal_encoding else []

    def with_seasonal(cols: list[str]) -> list[str]:
        return cols + seasonal if seasonal else cols

    return {
        "static_only": static,
        "quarterly_only": with_seasonal(quarterly),
        "quarterly_lagged": with_seasonal(quarterly + lagged),
        "full": with_seasonal(quarterly + static + lagged),
    }


def add_seasonal_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclical quarter-of-year from QUARTER label (does not use QUARTER as a feature)."""
    if "QUARTER" not in df.columns:
        return df
    out = df.copy()
    qnum = out["QUARTER"].astype(str).str.extract(r"Q([1-4])", expand=False).astype(float)
    angle = 2.0 * np.pi * (qnum - 1.0) / 4.0
    out["QUARTER_SIN"] = np.sin(angle)
    out["QUARTER_COS"] = np.cos(angle)
    return out


def cat_columns(*, exclude_quarter: bool) -> frozenset[str]:
    if exclude_quarter:
        return frozenset(c for c in CAT_COLS if c != "QUARTER")
    return CAT_COLS


def run_key(
    target: str,
    feature_set: str,
    algo: str,
    *,
    exclude_quarter: bool,
    seasonal_encoding: bool,
    hp_trials: int,
) -> str:
    parts = [target, feature_set, algo]
    if exclude_quarter:
        parts.append("no_quarter")
    if seasonal_encoding:
        parts.append("seasonal")
    if hp_trials > 0:
        parts.append(f"hp{hp_trials}")
    return "|".join(parts)


def load_manifest(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"completed": {}, "failed": {}}


def save_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def ensure_samples(
    parquet: Path,
    cache_root: Path,
    target: str,
    *,
    seed: int,
    train_end: str,
    val_end: str,
    max_train: int,
    max_val: int,
    max_test: int,
    neg_ratio: int,
    force: bool,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    sampling = {
        "max_train": max_train,
        "max_val": max_val,
        "max_test": max_test,
        "neg_ratio": neg_ratio,
    }
    if target == "next_quarter":
        sampling["train_end"] = train_end
        sampling["val_end"] = val_end

    cache_name = CACHE_TARGET.get(target, target)
    fp = sample_cache_fingerprint(parquet, target=cache_name, seed=seed, sampling=sampling)
    if not force:
        loaded = load_sample_cache(cache_root, cache_name, fp)
        if loaded is None and cache_root != COMBINED_CACHE:
            loaded = load_sample_cache(COMBINED_CACHE, cache_name, fp)
        if loaded:
            return loaded

    import pyarrow.parquet as pq

    all_cols = pq.ParquetFile(parquet).schema_arrow.names
    feature_cols = [c for c in all_cols if c not in LEAK_COLS]
    cat_set = set(CAT_COLS)

    if target == "panel_period":
        tr, ytr, va, yva, te, yte, n_in = stream_panel_period_sample(
            parquet,
            feature_cols=feature_cols,
            cat_cols=cat_set,
            max_train=max_train,
            max_val=max_val,
            max_test=max_test,
            neg_ratio=neg_ratio,
            seed=seed,
            checkpoint_path=cache_root / cache_name / fp / "scan_checkpoint.pkl",
        )
    else:
        tr, ytr, va, yva, te, yte, n_in = stream_next_quarter_sample(
            parquet,
            feature_cols=feature_cols,
            cat_cols=cat_set,
            train_end=train_end,
            val_end=val_end,
            max_train=max_train,
            max_val=max_val,
            max_test=max_test,
            neg_ratio=neg_ratio,
            seed=seed,
            checkpoint_path=cache_root / cache_name / fp / "scan_checkpoint.pkl",
        )

    save_sample_cache(
        cache_root,
        cache_name,
        fp,
        train_df=tr,
        y_train=ytr,
        val_df=va,
        y_val=yva,
        test_df=te,
        y_test=yte,
        meta={"n_rows_scanned": n_in, **sampling},
    )
    return tr, ytr, va, yva, te, yte


def subset_features(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    use = [c for c in cols if c in train.columns]
    missing = set(cols) - set(use)
    if missing:
        logging.warning("missing feature cols: %s", sorted(missing)[:5])
    return train[use].copy(), val[use].copy(), test[use].copy(), use


def encode_dataframes(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    cat_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    encoders: dict[str, LabelEncoder] = {}
    out_tr, out_va, out_te = train.copy(), val.copy(), test.copy()
    for c in cat_cols:
        if c not in train.columns:
            continue
        le = LabelEncoder()
        combined = (
            train[c].astype(str)
            .tolist()
            + val[c].astype(str).tolist()
            + test[c].astype(str).tolist()
        )
        le.fit(combined)
        encoders[c] = le
        for frame in (out_tr, out_va, out_te):
            frame[c] = le.transform(frame[c].astype(str))
    for c in out_tr.columns:
        if c not in cat_cols:
            for frame in (out_tr, out_va, out_te):
                frame[c] = pd.to_numeric(frame[c], errors="coerce")
    return out_tr, out_va, out_te, encoders


def class_weight_ratio(y: pd.Series) -> float:
    pos = float(y.sum())
    neg = float(len(y) - pos)
    return (neg / pos) if pos > 0 else 1.0


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k_frac: float) -> float:
    n = len(y_true)
    k = max(1, int(n * k_frac))
    idx = np.argpartition(-y_score, k - 1)[:k]
    return float(y_true[idx].mean())


def calibrate_isotonic(y_val: np.ndarray, p_val: np.ndarray, p_test: np.ndarray) -> np.ndarray:
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_val, y_val)
    return iso.predict(p_test).astype(np.float64)


def extended_test_metrics(
    y_test: np.ndarray,
    p_test: np.ndarray,
    y_val: np.ndarray,
    p_val: np.ndarray,
) -> dict:
    p_cal = calibrate_isotonic(y_val, p_val, p_test)
    f1m = best_f1_on_val(y_val, p_val, p_test, y_test)
    out: dict = {
        "test_ap": float(average_precision_score(y_test, p_test)),
        "test_ap_calibrated": float(average_precision_score(y_test, p_cal)),
        "brier_raw": float(brier_score_loss(y_test, p_test)),
        "brier_calibrated": float(brier_score_loss(y_test, p_cal)),
        **f1m,
        "test_precision_calibrated_at_val_thr": float(
            precision_score(y_test, (p_cal >= f1m["val_tuned_threshold"]).astype(int), zero_division=0)
        ),
        "test_recall_calibrated_at_val_thr": float(
            recall_score(y_test, (p_cal >= f1m["val_tuned_threshold"]).astype(int), zero_division=0)
        ),
    }
    for kf in TOP_K_FRACS:
        pct = int(kf * 100)
        out[f"precision_at_top_{pct}pct_raw"] = precision_at_k(y_test, p_test, kf)
        out[f"precision_at_top_{pct}pct_calibrated"] = precision_at_k(y_test, p_cal, kf)
    return out, p_cal


def plot_calibration(y_test: np.ndarray, p_test: np.ndarray, p_cal: np.ndarray, out_png: Path) -> None:
    plt.figure(figsize=(6, 5))
    prob_true, prob_pred = calibration_curve(y_test, p_test, n_bins=20, strategy="quantile")
    plt.plot(prob_pred, prob_true, "s-", label="raw")
    prob_true_c, prob_pred_c = calibration_curve(y_test, p_cal, n_bins=20, strategy="quantile")
    plt.plot(prob_pred_c, prob_true_c, "s-", label="isotonic")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration (test)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def best_f1_on_val(y_val: np.ndarray, p_val: np.ndarray, p_test: np.ndarray, y_test: np.ndarray) -> dict:
    prec, rec, thr = precision_recall_curve(y_val, p_val)
    f1 = (2 * prec[:-1] * rec[:-1]) / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    bi = int(np.nanargmax(f1))
    t = float(thr[bi])
    yp = (p_test >= t).astype(int)
    return {
        "val_tuned_threshold": t,
        "test_precision": float(precision_score(y_test, yp, zero_division=0)),
        "test_recall": float(recall_score(y_test, yp, zero_division=0)),
        "test_f1": float(
            2
            * precision_score(y_test, yp, zero_division=0)
            * recall_score(y_test, yp, zero_division=0)
            / max(
                precision_score(y_test, yp, zero_division=0) + recall_score(y_test, yp, zero_division=0),
                1e-12,
            )
        ),
    }


def plot_pr(y_test: np.ndarray, proba: np.ndarray, out_png: Path, title: str) -> None:
    ap = average_precision_score(y_test, proba)
    prec, rec, _ = precision_recall_curve(y_test, proba)
    plt.figure(figsize=(7, 5))
    plt.plot(rec, prec, lw=2, label=f"AP={ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def shap_summary(
    model_kind: str,
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cat_cols: list[str],
    out_dir: Path,
    *,
    max_rows: int,
) -> None:
    if len(X) > max_rows and y.nunique() >= 2:
        Xs, _, ys, _ = train_test_split(X, y, train_size=max_rows, stratify=y, random_state=42)
    else:
        Xs, ys = X, y

    names = list(X.columns)
    if model_kind == "catboost":
        pool = Pool(Xs, ys, cat_features=[c for c in cat_cols if c in Xs.columns])
        sv = model.get_feature_importance(pool, type="ShapValues")[:, :-1]
    elif model_kind == "xgboost":
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(Xs)
        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]
    else:
        # NGBoost: mean absolute gradient-based importance proxy via predict_proba sensitivity
        base = model.predict_proba(Xs)[:, 1]
        mean_abs = np.zeros(len(names))
        for j, col in enumerate(names):
            Xp = Xs.copy()
            Xp[col] = np.nan
            try:
                pert = model.predict_proba(Xp)[:, 1]
                mean_abs[j] = float(np.mean(np.abs(pert - base)))
            except Exception:
                mean_abs[j] = 0.0
        detail = {
            "algorithm": "ngboost",
            "note": "permutation-style mean |Δp| with NaN impute (NGBoost has no native SHAP)",
            "features": [
                {"feature": names[j], "mean_abs_shap": float(mean_abs[j])}
                for j in np.argsort(-mean_abs)
            ],
        }
        (out_dir / "shap_statistics.json").write_text(json.dumps(detail, indent=2), encoding="utf-8")
        top = detail["features"][:20][::-1]
        plt.figure(figsize=(8, 5))
        plt.barh([t["feature"] for t in top], [t["mean_abs_shap"] for t in top])
        plt.xlabel("mean |Δp|")
        plt.title("NGBoost feature sensitivity")
        plt.tight_layout()
        plt.savefig(out_dir / "shap_bar.png", dpi=160)
        plt.close()
        return

    mean_abs = np.mean(np.abs(sv), axis=0)
    order = np.argsort(-mean_abs)
    detail = {
        "algorithm": model_kind,
        "n_shap_samples": int(sv.shape[0]),
        "features": [
            {
                "feature": names[i],
                "mean_abs_shap": float(mean_abs[i]),
                "is_lag": any(names[i].endswith(s) for s in LAG_SUFFIXES),
            }
            for i in order
        ],
    }
    (out_dir / "shap_statistics.json").write_text(json.dumps(detail, indent=2), encoding="utf-8")
    topk = min(20, len(names))
    idx = order[:topk][::-1]
    plt.figure(figsize=(8, 5))
    plt.barh([names[i] for i in idx], mean_abs[idx])
    plt.xlabel("mean(|SHAP|)")
    plt.title(f"SHAP — {model_kind}")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_bar.png", dpi=160)
    plt.close()


def hp_param_space(algo: str) -> dict:
    if algo == "catboost":
        return {
            "depth": [4, 5, 6, 7, 8],
            "learning_rate": [0.03, 0.05, 0.08, 0.1],
            "l2_leaf_reg": [1.0, 3.0, 5.0, 10.0],
            "min_data_in_leaf": [1, 5, 15, 40],
        }
    if algo == "xgboost":
        return {
            "max_depth": [4, 5, 6, 7, 8],
            "learning_rate": [0.03, 0.05, 0.08, 0.1],
            "reg_lambda": [1.0, 3.0, 5.0, 10.0],
            "subsample": [0.7, 0.85, 1.0],
        }
    return {
        "n_estimators": [300, 400, 600],
        "learning_rate": [0.03, 0.05, 0.08],
        "minibatch_frac": [0.6, 0.8, 1.0],
    }


def hp_search(
    algo: str,
    train: pd.DataFrame,
    y_train: pd.Series,
    val: pd.DataFrame,
    y_val: pd.Series,
    cat_cols: list[str],
    *,
    trials: int,
    hp_iterations: int,
    ngboost_max_train: int,
    seed: int,
) -> tuple[dict, list[dict]]:
    space = hp_param_space(algo)
    best_ap, best_params = -1.0, {}
    records: list[dict] = []
    for i, params in enumerate(ParameterSampler(space, n_iter=trials, random_state=seed), 1):
        if algo == "catboost":
            _, p_val, _ = train_catboost(
                train, y_train, val, y_val, val, y_val, cat_cols, params=params, iterations=hp_iterations
            )
        elif algo == "xgboost":
            _, p_val, _ = train_xgboost(
                train, y_train, val, y_val, val, y_val, params=params, n_estimators=hp_iterations
            )
        else:
            _, p_val, _ = train_ngboost(
                train,
                y_train,
                val,
                y_val,
                val,
                y_val,
                max_train=ngboost_max_train,
                params=params,
            )
        ap = float(average_precision_score(y_val, p_val))
        records.append({"trial": i, "val_ap": ap, **params})
        logging.info("HP %s trial %s val_AP=%.5f", algo, i, ap)
        if ap > best_ap:
            best_ap, best_params = ap, dict(params)
    return best_params, records


def train_catboost(
    train: pd.DataFrame,
    y_train: pd.Series,
    val: pd.DataFrame,
    y_val: pd.Series,
    test: pd.DataFrame,
    y_test: pd.Series,
    cat_cols: list[str],
    *,
    params: dict | None = None,
    iterations: int = 800,
) -> tuple[CatBoostClassifier, np.ndarray, np.ndarray]:
    w = class_weight_ratio(y_train)
    cat_use = [c for c in cat_cols if c in train.columns]
    p = {
        "depth": 7,
        "learning_rate": 0.08,
        "l2_leaf_reg": 5,
        "min_data_in_leaf": 5,
    }
    if params:
        p.update(params)
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="PRAUC",
        iterations=iterations,
        random_seed=42,
        class_weights=[1.0, w],
        allow_writing_files=False,
        od_type="Iter",
        od_wait=60,
        verbose=0,
        **p,
    )
    model.fit(
        Pool(train, y_train, cat_features=cat_use),
        eval_set=Pool(val, y_val, cat_features=cat_use),
        use_best_model=True,
    )
    p_val = model.predict_proba(Pool(val, y_val, cat_features=cat_use))[:, 1]
    p_test = model.predict_proba(Pool(test, y_test, cat_features=cat_use))[:, 1]
    return model, p_val, p_test


def train_xgboost(
    train: pd.DataFrame,
    y_train: pd.Series,
    val: pd.DataFrame,
    y_val: pd.Series,
    test: pd.DataFrame,
    y_test: pd.Series,
    *,
    params: dict | None = None,
    n_estimators: int = 600,
) -> tuple[XGBClassifier, np.ndarray, np.ndarray]:
    w = class_weight_ratio(y_train)
    p = {
        "max_depth": 7,
        "learning_rate": 0.08,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 5,
    }
    if params:
        p.update(params)
    model = XGBClassifier(
        n_estimators=n_estimators,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=w,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50,
        **p,
    )
    model.fit(
        train,
        y_train,
        eval_set=[(val, y_val)],
        verbose=False,
    )
    p_val = model.predict_proba(val)[:, 1]
    p_test = model.predict_proba(test)[:, 1]
    return model, p_val, p_test


def train_ngboost(
    train: pd.DataFrame,
    y_train: pd.Series,
    val: pd.DataFrame,
    y_val: pd.Series,
    test: pd.DataFrame,
    y_test: pd.Series,
    *,
    max_train: int,
    params: dict | None = None,
) -> tuple[NGBClassifier, np.ndarray, np.ndarray]:
    if len(train) > max_train and y_train.nunique() >= 2:
        tr, _, yr, _ = train_test_split(train, y_train, train_size=max_train, stratify=y_train, random_state=42)
    else:
        tr, yr = train, y_train
    w = compute_sample_weight("balanced", yr)
    p = {"n_estimators": 400, "learning_rate": 0.05, "minibatch_frac": 0.8}
    if params:
        p.update(params)
    model = NGBClassifier(verbose=False, **p)
    model.fit(tr, yr, sample_weight=w)
    p_val = model.predict_proba(val)[:, 1]
    p_test = model.predict_proba(test)[:, 1]
    return model, p_val, p_test


def save_model(algo: str, model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if algo == "catboost":
        model.save_model(path)
    elif algo == "xgboost":
        model.save_model(path)
    else:
        import pickle

        with path.open("wb") as f:
            pickle.dump(model, f)


def run_one(
    *,
    target: str,
    feature_set: str,
    algo: str,
    train_df: pd.DataFrame,
    y_train: pd.Series,
    val_df: pd.DataFrame,
    y_val: pd.Series,
    test_df: pd.DataFrame,
    y_test: pd.Series,
    feature_groups: dict[str, list[str]],
    out_root: Path,
    shap_max_rows: int,
    ngboost_max_train: int,
    cat_cols: frozenset[str],
    seasonal_encoding: bool,
    hp_trials: int,
    hp_iterations: int,
    final_iterations: int,
    seed: int,
) -> RunMetrics:
    t0 = time.time()
    run_dir = out_root / target / feature_set / algo
    run_dir.mkdir(parents=True, exist_ok=True)

    if seasonal_encoding:
        train_df = add_seasonal_columns(train_df)
        val_df = add_seasonal_columns(val_df)
        test_df = add_seasonal_columns(test_df)

    tr, va, te, feat_cols = subset_features(train_df, val_df, test_df, feature_groups[feature_set])
    cat_in = [c for c in cat_cols if c in feat_cols]

    if algo in ("xgboost", "ngboost"):
        tr, va, te, enc = encode_dataframes(tr, va, te, cat_in)
        (run_dir / "label_encoders.json").write_text(
            json.dumps({k: list(v.classes_) for k, v in enc.items()}, indent=2),
            encoding="utf-8",
        )
        cat_for_shap: list[str] = []
    else:
        for c in cat_in:
            tr[c] = tr[c].astype(str).fillna("__MISSING__")
            va[c] = va[c].astype(str).fillna("__MISSING__")
            te[c] = te[c].astype(str).fillna("__MISSING__")
        cat_for_shap = cat_in

    best_params: dict = {}
    hp_records: list[dict] = []
    if hp_trials > 0:
        best_params, hp_records = hp_search(
            algo,
            tr,
            y_train,
            va,
            y_val,
            cat_in,
            trials=hp_trials,
            hp_iterations=hp_iterations,
            ngboost_max_train=ngboost_max_train,
            seed=seed,
        )
        (run_dir / "hyperparameter_search.json").write_text(
            json.dumps({"best_params": best_params, "trials": hp_records}, indent=2),
            encoding="utf-8",
        )

    if algo == "catboost":
        model, p_val, p_test = train_catboost(
            tr, y_train, va, y_val, te, y_test, cat_in, params=best_params, iterations=final_iterations
        )
        save_model(algo, model, run_dir / "model.cbm")
    elif algo == "xgboost":
        model, p_val, p_test = train_xgboost(
            tr, y_train, va, y_val, te, y_test, params=best_params, n_estimators=final_iterations
        )
        save_model(algo, model, run_dir / "model.json")
    else:
        model, p_val, p_test = train_ngboost(
            tr, y_train, va, y_val, te, y_test, max_train=ngboost_max_train, params=best_params or None
        )
        save_model(algo, model, run_dir / "model.pkl")

    y_te = y_test.to_numpy()
    ext, p_cal = extended_test_metrics(y_te, p_test, y_val.to_numpy(), p_val)
    plot_pr(y_te, p_test, run_dir / "pr_curve_raw.png", f"{algo} raw | {target} | {feature_set}")
    plot_pr(y_te, p_cal, run_dir / "pr_curve_calibrated.png", f"{algo} calibrated | {target} | {feature_set}")
    plot_calibration(y_te, p_test, p_cal, run_dir / "calibration_curve.png")
    shap_summary(algo, model, te, y_test, cat_for_shap, run_dir, max_rows=shap_max_rows)

    metrics = {
        "target": target,
        "feature_set": feature_set,
        "algorithm": algo,
        "test_prevalence": float(y_test.mean()),
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "n_features": len(feat_cols),
        "train_seconds": time.time() - t0,
        "best_hp_params": best_params,
        **ext,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return RunMetrics(
        target=target,
        feature_set=feature_set,
        algorithm=algo,
        test_ap=metrics["test_ap"],
        test_prevalence=metrics["test_prevalence"],
        val_tuned_threshold=metrics["val_tuned_threshold"],
        test_precision=metrics["test_precision"],
        test_recall=metrics["test_recall"],
        test_f1=metrics["test_f1"],
        n_train=metrics["n_train"],
        n_test=metrics["n_test"],
        n_features=metrics["n_features"],
        train_seconds=metrics["train_seconds"],
    )


def build_results_table(out_root: Path) -> pd.DataFrame:
    rows = []
    for fs in FEATURE_SETS:
        row: dict[str, object] = {"feature_set": fs}
        for target, prefix in (("panel_period", "panel"), ("next_quarter", "next_q")):
            for algo in ALGOS:
                mpath = out_root / target / fs / algo / "metrics.json"
                if mpath.exists():
                    m = json.loads(mpath.read_text(encoding="utf-8"))
                    row[f"{prefix}_{algo}_ap"] = round(m["test_ap"], 4)
                    row[f"{prefix}_{algo}_ap_cal"] = round(m.get("test_ap_calibrated", m["test_ap"]), 4)
                    row[f"{prefix}_{algo}_p@1_cal"] = round(m.get("precision_at_top_1pct_calibrated", 0), 4)
                    row[f"{prefix}_{algo}_p@5_cal"] = round(m.get("precision_at_top_5pct_calibrated", 0), 4)
                else:
                    for suf in ("_ap", "_ap_cal", "_p@1_cal", "_p@5_cal"):
                        row[f"{prefix}_{algo}{suf}"] = None
        rows.append(row)
    cols = ["feature_set"]
    for prefix in ("panel", "next_q"):
        for algo in ALGOS:
            cols.extend([f"{prefix}_{algo}_ap", f"{prefix}_{algo}_ap_cal", f"{prefix}_{algo}_p@1_cal", f"{prefix}_{algo}_p@5_cal"])
    return pd.DataFrame(rows)[cols]


def build_results_table_prf(out_root: Path) -> pd.DataFrame:
    """Precision / recall / F1 at validation-tuned threshold (raw + calibrated if present)."""
    rows = []
    for fs in FEATURE_SETS:
        row: dict[str, object] = {"feature_set": fs}
        for target, prefix in (("panel_period", "panel"), ("next_quarter", "next_q")):
            for algo in ALGOS:
                mpath = out_root / target / fs / algo / "metrics.json"
                if not mpath.exists():
                    continue
                m = json.loads(mpath.read_text(encoding="utf-8"))
                row[f"{prefix}_{algo}_prec"] = round(m["test_precision"], 4)
                row[f"{prefix}_{algo}_rec"] = round(m["test_recall"], 4)
                row[f"{prefix}_{algo}_f1"] = round(m["test_f1"], 4)
                row[f"{prefix}_{algo}_thr"] = round(m.get("val_tuned_threshold", 0), 4)
                if "test_precision_calibrated_at_val_thr" in m:
                    pc = m["test_precision_calibrated_at_val_thr"]
                    rc = m["test_recall_calibrated_at_val_thr"]
                    row[f"{prefix}_{algo}_prec_cal"] = round(pc, 4)
                    row[f"{prefix}_{algo}_rec_cal"] = round(rc, 4)
                    row[f"{prefix}_{algo}_f1_cal"] = round(
                        2 * pc * rc / (pc + rc) if pc + rc else 0.0, 4
                    )
        rows.append(row)
    return pd.DataFrame(rows)


def write_results_html(df: pd.DataFrame, path: Path, *, title: str = "Closure benchmark") -> None:
    styled = df.to_html(index=False, na_rep="—")
    path.write_text(
        "<html><head><meta charset='utf-8'><style>"
        "table{border-collapse:collapse;font-family:sans-serif;font-size:13px}"
        "th,td{border:1px solid #ccc;padding:6px 10px}"
        "th{background:#eee}</style></head><body>"
        f"<h2>{title}</h2>"
        f"{styled}</body></html>",
        encoding="utf-8",
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument(
        "--exclude-quarter",
        action="store_true",
        help="Drop QUARTER from quarterly feature sets (recommended for temporal next-quarter task)",
    )
    ap.add_argument(
        "--seasonal-encoding",
        action="store_true",
        help="Add QUARTER_SIN / QUARTER_COS (cyclical seasonality) to non-static feature sets",
    )
    ap.add_argument("--hp-trials", type=int, default=0, help="Random-search trials per run (0 = fixed defaults)")
    ap.add_argument("--hp-iterations", type=int, default=400, help="Max trees/iterations during HP search")
    ap.add_argument("--final-iterations", type=int, default=800, help="Trees/iterations for final model after HP")
    ap.add_argument(
        "--v2",
        action="store_true",
        help="Preset: --exclude-quarter --seasonal-encoding --hp-trials 8 -> benchmark_closure_models_v2",
    )
    ap.add_argument(
        "--sample-cache-dir",
        type=Path,
        default=ROOT / "outputs/sample_cache",
        help="Write new caches here; also reads combined_catboost_models/sample_cache if present",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-end", default="2023Q4")
    ap.add_argument("--val-end", default="2024Q2")
    ap.add_argument("--max-train-rows", type=int, default=600_000)
    ap.add_argument("--max-val-rows", type=int, default=120_000)
    ap.add_argument("--max-test-rows", type=int, default=180_000)
    ap.add_argument("--neg-ratio", type=int, default=20)
    ap.add_argument("--shap-max-rows", type=int, default=12_000)
    ap.add_argument("--ngboost-max-train", type=int, default=150_000)
    ap.add_argument("--force-resample", action="store_true")
    ap.add_argument("--only", type=str, default="", help="Run key prefix filter, e.g. panel_period|full|")
    args = ap.parse_args()

    if args.v2:
        args.exclude_quarter = True
        args.seasonal_encoding = True
        if args.hp_trials == 0:
            args.hp_trials = 8
        if args.output_dir is None:
            args.output_dir = DEFAULT_OUT_V2

    if args.output_dir is None:
        args.output_dir = DEFAULT_OUT_NO_QUARTER if args.exclude_quarter else DEFAULT_OUT

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest = load_manifest(manifest_path)
    feature_groups = load_feature_groups(
        exclude_quarter=args.exclude_quarter,
        seasonal_encoding=args.seasonal_encoding,
    )
    cat_cols = cat_columns(exclude_quarter=args.exclude_quarter)
    if args.exclude_quarter:
        logging.info("Excluded QUARTER from quarterly feature sets and categoricals")
    if args.seasonal_encoding:
        logging.info("Added cyclical QUARTER_SIN / QUARTER_COS to non-static feature sets")
    if args.hp_trials > 0:
        logging.info("HP search: %s trials, %s iters (final=%s)", args.hp_trials, args.hp_iterations, args.final_iterations)

    for target in TARGETS:
        logging.info("=== Samples: %s ===", target)
        samples = ensure_samples(
            args.parquet,
            args.sample_cache_dir,
            target,
            seed=args.seed,
            train_end=args.train_end,
            val_end=args.val_end,
            max_train=args.max_train_rows,
            max_val=args.max_val_rows,
            max_test=args.max_test_rows,
            neg_ratio=args.neg_ratio,
            force=args.force_resample,
        )
        tr, ytr, va, yva, te, yte = samples

        for feature_set in FEATURE_SETS:
            for algo in ALGOS:
                key = run_key(
                    target,
                    feature_set,
                    algo,
                    exclude_quarter=args.exclude_quarter,
                    seasonal_encoding=args.seasonal_encoding,
                    hp_trials=args.hp_trials,
                )
                if args.only and not key.startswith(args.only):
                    continue
                if key in manifest.get("completed", {}):
                    logging.info("skip done: %s", key)
                    continue
                logging.info("RUN %s", key)
                try:
                    metrics = run_one(
                        target=target,
                        feature_set=feature_set,
                        algo=algo,
                        train_df=tr,
                        y_train=ytr,
                        val_df=va,
                        y_val=yva,
                        test_df=te,
                        y_test=yte,
                        feature_groups=feature_groups,
                        out_root=args.output_dir,
                        shap_max_rows=args.shap_max_rows,
                        ngboost_max_train=args.ngboost_max_train,
                        cat_cols=cat_cols,
                        seasonal_encoding=args.seasonal_encoding,
                        hp_trials=args.hp_trials,
                        hp_iterations=args.hp_iterations,
                        final_iterations=args.final_iterations,
                        seed=args.seed,
                    )
                    mpath = args.output_dir / target / feature_set / algo / "metrics.json"
                    manifest.setdefault("completed", {})[key] = json.loads(mpath.read_text(encoding="utf-8"))
                    save_manifest(manifest_path, manifest)
                    logging.info("done %s AP=%.4f", key, metrics.test_ap)
                except Exception as exc:
                    logging.exception("failed %s: %s", key, exc)
                    manifest.setdefault("failed", {})[key] = str(exc)
                    save_manifest(manifest_path, manifest)

    df = build_results_table(args.output_dir)
    df.to_csv(args.output_dir / "results_table.csv", index=False)
    write_results_html(df, args.output_dir / "results_table.html", title="Closure benchmark (test AP)")
    (args.output_dir / "results_table.json").write_text(df.to_json(orient="records", indent=2), encoding="utf-8")
    logging.info("Results table -> %s", args.output_dir / "results_table.csv")

    df_prf = build_results_table_prf(args.output_dir)
    df_prf.to_csv(args.output_dir / "results_table_prf.csv", index=False)
    write_results_html(
        df_prf,
        args.output_dir / "results_table_prf.html",
        title="Closure benchmark (test P/R/F1 @ val-tuned threshold)",
    )
    logging.info("P/R/F1 table -> %s", args.output_dir / "results_table_prf.csv")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
