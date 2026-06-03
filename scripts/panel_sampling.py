#!/usr/bin/env python3
"""
Parquet streaming and sample-cache helpers for train_closure_models.py.

Imported by the training script (row sampling, LEAK_COLS, CAT_FEATURES,
stream_panel_period_sample, stream_next_quarter_sample, load/save_sample_cache).
Run train_closure_models.py as the entry point.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import ParameterSampler, train_test_split

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARQUET = ROOT / "data/establishment_quarter_panel.parquet"
DEFAULT_OUT = ROOT / "outputs/sample_cache"
DEFAULT_SAMPLE_CACHE = DEFAULT_OUT

QUARTER_RE = re.compile(r"^(\d{4})Q([1-4])$")
LAG_SUFFIXES = ("delta_1q", "delta_4q", "mean_last8")

LEAK_COLS = frozenset({"CLOSED_NEXT_QUARTER", "STATUS", "CLOSED_ON"})
DROP_EMPTY = frozenset({"sz_Payroll", "ag_Payroll"})
CAT_FEATURES = frozenset(
    {
        "COUNTY_FIPS",
        "ENTITY_TYPE",
        "NAICS_FIRST_DIGIT",
        "NAICS_FIRST_TWO",
        "loanstatus",
        "ENCLOSED",
        "INCLUDES_PARKING_LOT",
        "QUARTER",
        "approvalfy",
        "naicscode",
    }
)
NUMERIC_HINTS = frozenset(
    {
        "LATITUDE",
        "LONGITUDE",
        "REVENUE_IN_USD",
        "NUMBER_OF_EMPLOYEES",
        "WKT_AREA_SQ_METERS",
        "grossapproval",
        "AGE",
        "businessage",
    }
)


def quarter_index(label: str) -> int:
    m = QUARTER_RE.match((label or "").strip())
    if not m:
        return -1
    return int(m.group(1)) * 4 + int(m.group(2)) - 1


def split_name(q_label: str, *, train_end: str, val_end: str) -> str | None:
    qi = quarter_index(q_label)
    if qi < 0:
        return None
    te = quarter_index(train_end)
    ve = quarter_index(val_end)
    if qi <= te:
        return "train"
    if qi <= ve:
        return "val"
    return "test"


def est_id(row: dict) -> str:
    key = "|".join(
        [
            str(row.get("LATITUDE", "") or "").strip(),
            str(row.get("LONGITUDE", "") or "").strip(),
            str(row.get("DATE_FOUNDED", "") or "").strip(),
            str(row.get("NAICS_FIRST_TWO", "") or "").strip(),
            str(row.get("COUNTY_FIPS", "") or "").strip(),
        ]
    )
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:20]


def est_split(est: str, *, seed: int) -> str:
    h = hashlib.sha256(f"{seed}:{est}".encode()).hexdigest()
    bucket = int(h[:8], 16) / 0xFFFFFFFF
    if bucket < 0.70:
        return "train"
    if bucket < 0.85:
        return "val"
    return "test"


def build_feature_lists(columns: list[str]) -> tuple[list[str], list[str]]:
    drop = LEAK_COLS | DROP_EMPTY
    feature_cols = [c for c in columns if c not in drop]
    cat_cols = sorted(CAT_FEATURES & set(feature_cols))
    return feature_cols, cat_cols


def row_to_features(row: pd.Series, feature_cols: list[str], cat_cols: set[str]) -> dict[str, object]:
    out: dict[str, object] = {}
    for c in feature_cols:
        raw = row.get(c)
        if c in cat_cols:
            if pd.isna(raw):
                out[c] = "__MISSING__"
            else:
                s = str(raw).strip()
                out[c] = s if s else "__MISSING__"
        else:
            if pd.isna(raw):
                out[c] = np.nan
            elif isinstance(raw, (bool, np.bool_)):
                out[c] = float(raw)
            elif isinstance(raw, (int, np.integer)):
                out[c] = float(raw)
            elif isinstance(raw, (float, np.floating)):
                out[c] = float(raw)
            else:
                s = str(raw).strip()
                if not s or s.lower() in {"none", "null", "nan", "[]"}:
                    out[c] = np.nan
                else:
                    try:
                        out[c] = float(s.replace(",", ""))
                    except ValueError:
                        out[c] = np.nan
    return out


@dataclass
class SplitBucket:
    pos_rows: list[dict[str, object]]
    neg_rows: list[dict[str, object]]
    neg_seen: int = 0


def _add_to_bucket(
    bucket: SplitBucket,
    feats: dict[str, object],
    y: int,
    *,
    cap: int,
    neg_ratio: int,
    rng: np.random.Generator,
) -> None:
    if y == 1:
        bucket.pos_rows.append(feats)
        return
    n_pos = len(bucket.pos_rows)
    max_neg = min(cap - n_pos, max(1, n_pos * neg_ratio)) if n_pos else min(cap, max(1, neg_ratio * 10))
    if max_neg <= 0:
        return
    bucket.neg_seen += 1
    if len(bucket.neg_rows) < max_neg:
        bucket.neg_rows.append(feats)
        return
    j = int(rng.integers(0, bucket.neg_seen))
    if j < max_neg:
        bucket.neg_rows[j] = feats


def bucket_to_frame(b: SplitBucket) -> tuple[pd.DataFrame, pd.Series]:
    rows = b.pos_rows + b.neg_rows
    labels = [1] * len(b.pos_rows) + [0] * len(b.neg_rows)
    return pd.DataFrame(rows), pd.Series(labels, dtype=np.int8, name="y")


def sample_cache_fingerprint(parquet: Path, *, target: str, seed: int, sampling: dict) -> str:
    """Stable cache id from parquet mtime + sampling hyperparameters."""
    payload = {
        "target": target,
        "parquet": str(parquet.resolve()),
        "parquet_mtime": parquet.stat().st_mtime,
        "parquet_size": parquet.stat().st_size,
        "seed": seed,
        **sampling,
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def save_sample_cache(
    cache_root: Path,
    target: str,
    fingerprint: str,
    *,
    train_df: pd.DataFrame,
    y_train: pd.Series,
    val_df: pd.DataFrame,
    y_val: pd.Series,
    test_df: pd.DataFrame,
    y_test: pd.Series,
    meta: dict,
) -> Path:
    out = cache_root / target / fingerprint
    out.mkdir(parents=True, exist_ok=True)
    for name, X, y in (
        ("train", train_df, y_train),
        ("val", val_df, y_val),
        ("test", test_df, y_test),
    ):
        frame = X.copy()
        frame["__y__"] = y.values
        frame.to_parquet(out / f"{name}.parquet", index=False, compression="snappy")
    meta = {**meta, "fingerprint": fingerprint, "target": target}
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logging.info("Wrote sample cache -> %s", out)
    return out


def scan_checkpoint_path(cache_root: Path, target: str, fingerprint: str) -> Path:
    return cache_root / target / fingerprint / "scan_checkpoint.pkl"


def save_scan_checkpoint(
    path: Path,
    *,
    buckets: dict[str, SplitBucket],
    n_in: int,
    extra: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"buckets": buckets, "n_in": n_in, **extra}
    tmp = path.with_suffix(".pkl.tmp")
    with tmp.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def load_scan_checkpoint(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("rb") as f:
        return pickle.load(f)


def load_sample_cache(
    cache_root: Path,
    target: str,
    fingerprint: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series] | None:
    root = cache_root / target / fingerprint
    if not (root / "meta.json").exists():
        return None
    splits: list[tuple[pd.DataFrame, pd.Series]] = []
    for name in ("train", "val", "test"):
        path = root / f"{name}.parquet"
        if not path.exists():
            return None
        frame = pd.read_parquet(path)
        if "__y__" not in frame.columns:
            return None
        y = frame.pop("__y__").astype(np.int8)
        splits.append((frame, y))
    logging.info("Loaded sample cache from %s", root)
    return (
        splits[0][0],
        splits[0][1],
        splits[1][0],
        splits[1][1],
        splits[2][0],
        splits[2][1],
    )


def class_weight_pos(y: pd.Series) -> float:
    pos = float(y.sum())
    neg = float(len(y) - pos)
    return (neg / pos) if pos > 0 else 1.0


def train_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cat_cols: list[str],
    params: dict,
    iterations: int,
    verbose: int,
) -> CatBoostClassifier:
    w_pos = class_weight_pos(y_train)
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="PRAUC",
        random_seed=42,
        class_weights=[1.0, w_pos],
        allow_writing_files=False,
        od_type="Iter",
        od_wait=80,
        iterations=iterations,
        verbose=verbose,
        **params,
    )
    train_pool = Pool(X_train, y_train, cat_features=cat_cols)
    eval_pool = Pool(X_val, y_val, cat_features=cat_cols)
    model.fit(train_pool, eval_set=eval_pool, use_best_model=True)
    return model


def _best_f1_point(y: np.ndarray, proba: np.ndarray) -> tuple[float, float, float, float]:
    precision, recall, thresholds = precision_recall_curve(y, proba)
    f1 = (2 * precision[:-1] * recall[:-1]) / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    best_idx = int(np.nanargmax(f1))
    return (
        float(thresholds[best_idx]),
        float(f1[best_idx]),
        float(precision[best_idx]),
        float(recall[best_idx]),
    )


def compute_pr_artifacts(
    y: np.ndarray,
    proba: np.ndarray,
    out: Path,
    *,
    title: str,
    stem: str,
    y_val: np.ndarray | None = None,
    proba_val: np.ndarray | None = None,
) -> dict:
    ap_score = float(average_precision_score(y, proba))
    precision, recall, thresholds = precision_recall_curve(y, proba)

    # Report threshold tuned on validation (honest) when available; else test-only.
    if y_val is not None and proba_val is not None:
        best_thr, best_f1, best_p, best_r = _best_f1_point(y_val, proba_val)
        thr_note = "val-tuned F1"
    else:
        best_thr, best_f1, best_p, best_r = _best_f1_point(y, proba)
        thr_note = "test-tuned F1 (optimistic)"

    test_thr, test_f1, test_p, test_r = _best_f1_point(y, proba)
    y_pred = (proba >= best_thr).astype(np.int32)

    plt.figure(figsize=(7.5, 5.5))
    plt.plot(recall, precision, lw=2, label=f"CatBoost (AP={ap_score:.4f})")
    scatter_r = float(recall_score(y, y_pred, zero_division=0))
    scatter_p = float(precision_score(y, y_pred, zero_division=0))
    plt.scatter([scatter_r], [scatter_p], s=40, c="red", label=f"{thr_note} thr={best_thr:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / f"pr_curve_{stem}.png", dpi=200)
    plt.close()

    step = max(1, len(recall) // 400)
    pr_dump = {
        "average_precision": ap_score,
        "prevalence_positive": float(y.mean()),
        "n_scored": int(len(y)),
        "threshold_tuning": thr_note,
        "best_f1_threshold": best_thr,
        "best_f1": best_f1,
        "precision_at_best_f1": scatter_p,
        "recall_at_best_f1": scatter_r,
        "sklearn_precision_at_best_f1_threshold": float(precision_score(y, y_pred, zero_division=0)),
        "sklearn_recall_at_best_f1_threshold": float(recall_score(y, y_pred, zero_division=0)),
        "test_tuned_f1_threshold_optimistic": test_thr,
        "test_tuned_recall_optimistic": test_r,
        "test_tuned_precision_optimistic": test_p,
        "recall_curve_subsampled": recall[::step].tolist(),
        "precision_curve_subsampled": precision[::step].tolist(),
    }
    (out / f"pr_curve_{stem}.json").write_text(json.dumps(pr_dump, indent=2), encoding="utf-8")
    return pr_dump


def compute_shap(
    model: CatBoostClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    cat_cols: list[str],
    out: Path,
    *,
    stem: str,
    title: str,
    note: str,
) -> dict:
    pool = Pool(X, y, cat_features=cat_cols)
    shap_values = model.get_feature_importance(pool, type="ShapValues")
    shap_vals = shap_values[:, :-1]
    names = list(X.columns)
    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    order = np.argsort(-mean_abs)

    detail = {
        "n_shap_samples": int(shap_vals.shape[0]),
        "expected_value_mean": float(shap_values[:, -1].mean()),
        "note": note,
        "features": [],
    }
    for j, name in enumerate(names):
        col = shap_vals[:, j].astype(np.float64)
        detail["features"].append(
            {
                "feature": name,
                "mean_abs_shap": float(np.mean(np.abs(col))),
                "std_shap": float(np.std(col)),
                "mean_shap": float(np.mean(col)),
                "is_lag": any(name.endswith(s) for s in LAG_SUFFIXES),
            }
        )
    detail["features"].sort(key=lambda d: d["mean_abs_shap"], reverse=True)
    lag_only = [f for f in detail["features"] if f["is_lag"]]
    detail["lag_features_ranked"] = lag_only
    (out / f"shap_statistics_{stem}.json").write_text(json.dumps(detail, indent=2), encoding="utf-8")

    topk = min(30, len(names))
    idx = order[:topk][::-1]
    plt.figure(figsize=(9, 6))
    plt.barh([names[i] for i in idx], mean_abs[idx])
    plt.xlabel("mean(|SHAP|)")
    plt.title(f"SHAP — {title}")
    plt.tight_layout()
    plt.savefig(out / f"shap_summary_bar_{stem}.png", dpi=200)
    plt.close()

    lag_order = sorted(lag_only, key=lambda d: d["mean_abs_shap"], reverse=True)
    top_lag = min(30, len(lag_order))
    if top_lag:
        plt.figure(figsize=(9, 6))
        sub = lag_order[:top_lag][::-1]
        plt.barh([d["feature"] for d in sub], [d["mean_abs_shap"] for d in sub])
        plt.xlabel("mean(|SHAP|)")
        plt.title(f"SHAP — lag features only ({title})")
        plt.tight_layout()
        plt.savefig(out / f"shap_lag_only_bar_{stem}.png", dpi=200)
        plt.close()

    return detail


def _est_ids_frame(df: pd.DataFrame) -> pd.Series:
    key = (
        df["LATITUDE"].astype(str).str.strip()
        + "|"
        + df["LONGITUDE"].astype(str).str.strip()
        + "|"
        + df["DATE_FOUNDED"].astype(str).str.strip()
        + "|"
        + df["NAICS_FIRST_TWO"].astype(str).str.strip()
        + "|"
        + df["COUNTY_FIPS"].astype(str).str.strip()
    )
    return key.map(lambda s: hashlib.sha1(s.encode("utf-8")).hexdigest()[:20])


def _features_from_batch(df: pd.DataFrame, feature_cols: list[str], cat_cols: set[str]) -> pd.DataFrame:
    out = df[feature_cols].copy()
    for c in feature_cols:
        if c in cat_cols:
            out[c] = out[c].astype("string").fillna("__MISSING__").replace("", "__MISSING__")
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def stream_panel_period_sample(
    path: Path,
    *,
    feature_cols: list[str],
    cat_cols: set[str],
    max_train: int,
    max_val: int,
    max_test: int,
    neg_ratio: int,
    seed: int,
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 2_000_000,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, int]:
    """Establishment-level split; first row per establishment only."""
    rng = np.random.default_rng(seed)
    buckets = {
        "train": SplitBucket([], []),
        "val": SplitBucket([], []),
        "test": SplitBucket([], []),
    }
    caps = {"train": max_train, "val": max_val, "test": max_test}
    seen_est: set[str] = set()
    n_in = 0
    resume_batch = 0

    if checkpoint_path is not None:
        ck = load_scan_checkpoint(checkpoint_path)
        if ck is not None:
            buckets = ck["buckets"]
            n_in = int(ck["n_in"])
            seen_est = set(ck.get("seen_est", []))
            resume_batch = int(ck.get("resume_batch", n_in // 250_000))
            logging.info(
                "Resuming panel scan from row %s (batch %s) | train pos=%s",
                f"{n_in:,}",
                resume_batch,
                len(buckets["train"].pos_rows),
            )

    cols = list(
        dict.fromkeys(
            feature_cols
            + list(LEAK_COLS)
            + ["LATITUDE", "LONGITUDE", "DATE_FOUNDED", "NAICS_FIRST_TWO", "COUNTY_FIPS"]
        )
    )
    pf = pq.ParquetFile(path)
    batch_size = 250_000
    for batch_i, batch in enumerate(pf.iter_batches(batch_size=batch_size, columns=cols)):
        if batch_i < resume_batch:
            continue
        df = batch.to_pandas()
        n_in += len(df)
        closed = df["CLOSED_ON"].notna() & (df["CLOSED_ON"].astype(str).str.strip() != "")
        df = df.assign(_y=closed.astype(np.int8), _eid=_est_ids_frame(df))
        df = df.loc[~df["_eid"].isin(seen_est)]
        if df.empty:
            continue
        seen_est.update(df["_eid"].tolist())
        df = df.drop_duplicates(subset=["_eid"], keep="first")
        Xb = _features_from_batch(df, feature_cols, cat_cols)
        for i in range(len(df)):
            eid = df["_eid"].iat[i]
            sp = est_split(eid, seed=seed)
            if sp not in buckets:
                continue
            _add_to_bucket(
                buckets[sp],
                Xb.iloc[i].to_dict(),
                int(df["_y"].iat[i]),
                cap=caps[sp],
                neg_ratio=neg_ratio,
                rng=rng,
            )
        if n_in % 2_000_000 == 0:
            logging.info(
                "panel scan %s | train pos=%s | val pos=%s | test pos=%s",
                f"{n_in:,}",
                len(buckets["train"].pos_rows),
                len(buckets["val"].pos_rows),
                len(buckets["test"].pos_rows),
            )
        if checkpoint_path is not None and n_in % checkpoint_every == 0:
            save_scan_checkpoint(
                checkpoint_path,
                buckets=buckets,
                n_in=n_in,
                extra={"seen_est": seen_est, "resume_batch": batch_i + 1},
            )

    if checkpoint_path is not None and checkpoint_path.exists():
        checkpoint_path.unlink()

    train_df, y_train = bucket_to_frame(buckets["train"])
    val_df, y_val = bucket_to_frame(buckets["val"])
    test_df, y_test = bucket_to_frame(buckets["test"])
    logging.info(
        "Panel-period sample: train=%s pos=%s | val=%s pos=%s | test=%s pos=%s | est=%s",
        len(train_df),
        int(y_train.sum()),
        len(val_df),
        int(y_val.sum()),
        len(test_df),
        int(y_test.sum()),
        len(seen_est),
    )
    return train_df, y_train, val_df, y_val, test_df, y_test, n_in


def stream_next_quarter_sample(
    path: Path,
    *,
    feature_cols: list[str],
    cat_cols: set[str],
    train_end: str,
    val_end: str,
    max_train: int,
    max_val: int,
    max_test: int,
    neg_ratio: int,
    seed: int,
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 2_000_000,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, int]:
    rng = np.random.default_rng(seed)
    buckets = {
        "train": SplitBucket([], []),
        "val": SplitBucket([], []),
        "test": SplitBucket([], []),
    }
    caps = {"train": max_train, "val": max_val, "test": max_test}
    n_in = 0
    n_skip = int(0)
    resume_batch = 0

    if checkpoint_path is not None:
        ck = load_scan_checkpoint(checkpoint_path)
        if ck is not None:
            buckets = ck["buckets"]
            n_in = int(ck["n_in"])
            n_skip = int(ck.get("n_skip", 0))
            resume_batch = int(ck.get("resume_batch", n_in // 250_000))
            logging.info(
                "Resuming next-q scan from row %s (batch %s) | train pos=%s",
                f"{n_in:,}",
                resume_batch,
                len(buckets["train"].pos_rows),
            )

    cols = list(dict.fromkeys(feature_cols + list(LEAK_COLS) + ["QUARTER"]))
    pf = pq.ParquetFile(path)
    batch_size = 250_000
    for batch_i, batch in enumerate(pf.iter_batches(batch_size=batch_size, columns=cols)):
        if batch_i < resume_batch:
            continue
        df = batch.to_pandas()
        n_in += len(df)
        valid = df["CLOSED_NEXT_QUARTER"].notna()
        n_skip += int((~valid).sum())
        df = df.loc[valid]
        if df.empty:
            continue
        y = df["CLOSED_NEXT_QUARTER"].map(lambda v: 1 if v is True or str(v).lower() in ("true", "1") else 0)
        splits = df["QUARTER"].astype(str).map(lambda q: split_name(q, train_end=train_end, val_end=val_end))
        df = df.assign(_y=y.astype(np.int8), _sp=splits)
        df = df.loc[df["_sp"].isin(buckets.keys())]
        if df.empty:
            continue
        Xb = _features_from_batch(df, feature_cols, cat_cols)
        for i in range(len(df)):
            sp = df["_sp"].iat[i]
            _add_to_bucket(
                buckets[sp],
                Xb.iloc[i].to_dict(),
                int(df["_y"].iat[i]),
                cap=caps[sp],
                neg_ratio=neg_ratio,
                rng=rng,
            )
        if n_in % 2_000_000 == 0:
            logging.info(
                "next-q scan %s | train pos=%s | val pos=%s | test pos=%s",
                f"{n_in:,}",
                len(buckets["train"].pos_rows),
                len(buckets["val"].pos_rows),
                len(buckets["test"].pos_rows),
            )
        if checkpoint_path is not None and n_in % checkpoint_every == 0:
            save_scan_checkpoint(
                checkpoint_path,
                buckets=buckets,
                n_in=n_in,
                extra={"n_skip": n_skip, "resume_batch": batch_i + 1},
            )

    if checkpoint_path is not None and checkpoint_path.exists():
        checkpoint_path.unlink()

    train_df, y_train = bucket_to_frame(buckets["train"])
    val_df, y_val = bucket_to_frame(buckets["val"])
    test_df, y_test = bucket_to_frame(buckets["test"])
    logging.info(
        "Next-quarter sample: train=%s pos=%s | val=%s | test=%s | skipped=%s",
        len(train_df),
        int(y_train.sum()),
        len(val_df),
        len(test_df),
        n_skip,
    )
    return train_df, y_train, val_df, y_val, test_df, y_test, n_in


def hp_search(
    train_df: pd.DataFrame,
    y_train: pd.Series,
    val_df: pd.DataFrame,
    y_val: pd.Series,
    cat_cols: list[str],
    *,
    trials: int,
    iterations: int,
    seed: int,
) -> tuple[dict, list[dict]]:
    param_space = {
        "depth": [4, 5, 6, 7, 8],
        "learning_rate": [0.03, 0.05, 0.07, 0.1],
        "l2_leaf_reg": [1.0, 3.0, 5.0, 10.0],
        "bagging_temperature": [0.0, 0.5, 1.0],
        "random_strength": [0.5, 1.0, 2.0],
        "min_data_in_leaf": [1, 5, 15, 40],
    }
    best_ap, best_params = -1.0, {}
    records: list[dict] = []
    for i, params in enumerate(ParameterSampler(param_space, n_iter=trials, random_state=seed), 1):
        model = train_catboost(train_df, y_train, val_df, y_val, cat_cols, params, iterations, 0)
        proba = model.predict_proba(Pool(val_df, y_val, cat_features=cat_cols))[:, 1]
        ap_score = float(average_precision_score(y_val, proba))
        records.append({"trial": i, "val_ap": ap_score, **params})
        logging.info("HP trial %s val_AP=%.5f", i, ap_score)
        if ap_score > best_ap:
            best_ap, best_params = ap_score, dict(params)
    return best_params, records


def run_model(
    *,
    name: str,
    train_df: pd.DataFrame,
    y_train: pd.Series,
    val_df: pd.DataFrame,
    y_val: pd.Series,
    test_df: pd.DataFrame,
    y_test: pd.Series,
    cat_cols: list[str],
    out_dir: Path,
    hp_trials: int,
    hp_iterations: int,
    final_iterations: int,
    shap_max_rows: int,
    seed: int,
    label_note: str,
) -> None:
    out = out_dir / name
    out.mkdir(parents=True, exist_ok=True)

    best_params, trials = hp_search(
        train_df, y_train, val_df, y_val, cat_cols, trials=hp_trials, iterations=hp_iterations, seed=seed
    )
    (out / "hyperparameter_search.json").write_text(
        json.dumps({"best_params": best_params, "trials": trials}, indent=2),
        encoding="utf-8",
    )

    fit_df = pd.concat([train_df, val_df], ignore_index=True)
    y_fit = pd.concat([y_train, y_val], ignore_index=True)
    holdout = test_df
    y_holdout = y_test
    if shap_max_rows > 0 and len(test_df) > shap_max_rows and y_test.nunique() >= 2:
        holdout, _, y_holdout, _ = train_test_split(
            test_df, y_test, train_size=shap_max_rows, stratify=y_test, random_state=seed
        )

    model = train_catboost(fit_df, y_fit, holdout, y_holdout, cat_cols, best_params, final_iterations, 100)
    model.save_model(out / "model.cbm")

    proba_val = model.predict_proba(Pool(val_df, y_val, cat_features=cat_cols))[:, 1]
    proba = model.predict_proba(Pool(test_df, y_test, cat_features=cat_cols))[:, 1]
    pr = compute_pr_artifacts(
        y_test.to_numpy(),
        proba,
        out,
        title=f"Precision–Recall — {label_note}",
        stem=name,
        y_val=y_val.to_numpy(),
        proba_val=proba_val,
    )
    shap = compute_shap(
        model,
        holdout.reset_index(drop=True),
        y_holdout.reset_index(drop=True),
        cat_cols,
        out,
        stem=name,
        title=label_note,
        note=f"SHAP on {len(holdout):,} test rows (stratified sample)",
    )

    summary = {
        "target": label_note,
        "n_train_sample": int(len(train_df)),
        "n_val_sample": int(len(val_df)),
        "n_test_sample": int(len(test_df)),
        "train_positive_rate": float(y_train.mean()),
        "test_positive_rate": float(y_test.mean()),
        "best_params": best_params,
        **pr,
        "top_10_features_shap": [f["feature"] for f in shap["features"][:10]],
        "top_15_lag_features_shap": [f["feature"] for f in shap["lag_features_ranked"][:15]],
    }
    (out / f"metrics_{name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logging.info("%s test AP=%.5f", name, pr["average_precision"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-end", type=str, default="2023Q4")
    ap.add_argument("--val-end", type=str, default="2024Q2")
    ap.add_argument("--max-train-rows", type=int, default=600_000)
    ap.add_argument("--max-val-rows", type=int, default=150_000)
    ap.add_argument("--max-test-rows", type=int, default=200_000)
    ap.add_argument("--neg-ratio", type=int, default=20)
    ap.add_argument("--hp-trials", type=int, default=12)
    ap.add_argument("--hp-iterations", type=int, default=600)
    ap.add_argument("--final-iterations", type=int, default=1200)
    ap.add_argument("--shap-max-rows", type=int, default=40_000)
    ap.add_argument("--skip-panel", action="store_true")
    ap.add_argument("--skip-next-quarter", action="store_true")
    ap.add_argument("--sample-cache-dir", type=Path, default=DEFAULT_SAMPLE_CACHE)
    ap.add_argument(
        "--force-resample",
        action="store_true",
        help="Ignore cached stratified samples and re-scan the full parquet",
    )
    ap.add_argument(
        "--sample-only",
        action="store_true",
        help="Only build/load sample cache; skip CatBoost training",
    )
    args = ap.parse_args()

    if not args.parquet.exists():
        raise SystemExit(f"missing parquet: {args.parquet}")

    schema_cols = pq.ParquetFile(args.parquet).schema_arrow.names
    feature_cols, cat_cols = build_feature_lists(schema_cols)
    cat_set = set(cat_cols)
    logging.info("Features: %s (%s categorical)", len(feature_cols), len(cat_cols))
    lag_n = sum(1 for c in feature_cols if any(c.endswith(s) for s in LAG_SUFFIXES))
    logging.info("Lagged feature columns: %s", lag_n)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_panel:
        logging.info("=== Model 1: close in panel period (CLOSED_ON) ===")
        panel_sampling = {
            "max_train": args.max_train_rows,
            "max_val": args.max_val_rows,
            "max_test": args.max_test_rows,
            "neg_ratio": args.neg_ratio,
        }
        panel_fp = sample_cache_fingerprint(
            args.parquet, target="close_in_panel_period", seed=args.seed, sampling=panel_sampling
        )
        loaded = None if args.force_resample else load_sample_cache(
            args.sample_cache_dir, "close_in_panel_period", panel_fp
        )
        ck_path = scan_checkpoint_path(args.sample_cache_dir, "close_in_panel_period", panel_fp)
        if args.force_resample and ck_path.exists():
            ck_path.unlink()
        if loaded:
            tr, ytr, va, yva, te, yte = loaded
        else:
            tr, ytr, va, yva, te, yte, n_in = stream_panel_period_sample(
                args.parquet,
                feature_cols=feature_cols,
                cat_cols=cat_set,
                max_train=args.max_train_rows,
                max_val=args.max_val_rows,
                max_test=args.max_test_rows,
                neg_ratio=args.neg_ratio,
                seed=args.seed,
                checkpoint_path=None if args.force_resample else ck_path,
            )
            save_sample_cache(
                args.sample_cache_dir,
                "close_in_panel_period",
                panel_fp,
                train_df=tr,
                y_train=ytr,
                val_df=va,
                y_val=yva,
                test_df=te,
                y_test=yte,
                meta={"n_panel_rows_scanned": n_in, **panel_sampling},
            )
        if not args.sample_only:
            run_model(
                name="close_in_panel_period",
                train_df=tr,
                y_train=ytr,
                val_df=va,
                y_val=yva,
                test_df=te,
                y_test=yte,
                cat_cols=cat_cols,
                out_dir=args.output_dir,
                hp_trials=args.hp_trials,
                hp_iterations=args.hp_iterations,
                final_iterations=args.final_iterations,
                shap_max_rows=args.shap_max_rows,
                seed=args.seed,
                label_note="closure during panel (CLOSED_ON set)",
            )

    if not args.skip_next_quarter:
        logging.info("=== Model 2: CLOSED_NEXT_QUARTER ===")
        nq_sampling = {
            "max_train": args.max_train_rows,
            "max_val": args.max_val_rows,
            "max_test": args.max_test_rows,
            "neg_ratio": args.neg_ratio,
            "train_end": args.train_end,
            "val_end": args.val_end,
        }
        nq_fp = sample_cache_fingerprint(
            args.parquet, target="closed_next_quarter", seed=args.seed, sampling=nq_sampling
        )
        loaded = None if args.force_resample else load_sample_cache(
            args.sample_cache_dir, "closed_next_quarter", nq_fp
        )
        ck_path = scan_checkpoint_path(args.sample_cache_dir, "closed_next_quarter", nq_fp)
        if args.force_resample and ck_path.exists():
            ck_path.unlink()
        if loaded:
            tr, ytr, va, yva, te, yte = loaded
        else:
            tr, ytr, va, yva, te, yte, n_in = stream_next_quarter_sample(
                args.parquet,
                feature_cols=feature_cols,
                cat_cols=cat_set,
                train_end=args.train_end,
                val_end=args.val_end,
                max_train=args.max_train_rows,
                max_val=args.max_val_rows,
                max_test=args.max_test_rows,
                neg_ratio=args.neg_ratio,
                seed=args.seed,
                checkpoint_path=None if args.force_resample else ck_path,
            )
            save_sample_cache(
                args.sample_cache_dir,
                "closed_next_quarter",
                nq_fp,
                train_df=tr,
                y_train=ytr,
                val_df=va,
                y_val=yva,
                test_df=te,
                y_test=yte,
                meta={"n_panel_rows_scanned": n_in, **nq_sampling},
            )
        if not args.sample_only:
            run_model(
                name="closed_next_quarter",
                train_df=tr,
                y_train=ytr,
                val_df=va,
                y_val=yva,
                test_df=te,
                y_test=yte,
                cat_cols=cat_cols,
                out_dir=args.output_dir,
                hp_trials=args.hp_trials,
                hp_iterations=args.hp_iterations,
                final_iterations=args.final_iterations,
                shap_max_rows=args.shap_max_rows,
                seed=args.seed,
                label_note="CLOSED_NEXT_QUARTER (temporal test split)",
            )

    logging.info("Done. Outputs in %s", args.output_dir)


if __name__ == "__main__":
    raise SystemExit(
        "Use train_closure_models.py --v2 to train all 24 paper models. "
        "This file is imported for parquet sampling only."
    )
