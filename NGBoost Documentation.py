import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ngboost import NGBClassifier
from sklearn.metrics import (
    classification_report, precision_recall_curve,
    average_precision_score, brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import compute_sample_weight
import shap
import itertools
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("/Users/parthhalani/Downloads/POI_data_quarterly.csv")
df = df.sort_values('CURRENT_QUARTER_START')

split_index = int(len(df) * 0.8)
train = df.iloc[:split_index]
test  = df.iloc[split_index:]

drop_cols = [
    'CLOSED_NEXT_QUARTER', 'NAME', 'BRAND', 'OPENED_ON',
    'CLOSED_ON', 'ADDRESS', 'CURRENT_QUARTER',
    'CURRENT_QUARTER_START', 'COUNTY', 'NAICS_CODE', 'NAICS_CODE_2022'
]
X_train = train.drop(columns=drop_cols)
Y_train = train['CLOSED_NEXT_QUARTER']
X_test  = test.drop(columns=drop_cols)
Y_test  = test['CLOSED_NEXT_QUARTER']

neg = (Y_train == 0).sum()
pos = (Y_train == 1).sum()
ratio = int(neg / pos)
print(f"Class ratio: {ratio}x  (neg={neg}, pos={pos})")

weights = compute_sample_weight(class_weight={0: 1, 1: ratio}, y=Y_train)

print("=" * 60)
print("MANUAL HYPERPARAMETER SEARCH")
print("=" * 60)

param_grid = {
    'n_estimators':   [100, 300],
    'learning_rate':  [0.05, 0.1],
    'minibatch_frac': [0.5, 1.0],
}

cv   = StratifiedKFold(n_splits=3, shuffle=False)
X_tr_arr = X_train.values
Y_tr_arr = Y_train.values

best_score  = -np.inf
best_params = None
keys   = list(param_grid.keys())
combos = list(itertools.product(*param_grid.values()))
print(f"Testing {len(combos)} combos x 3 folds...\n")

for i, combo in enumerate(combos):
    params = dict(zip(keys, combo))
    fold_scores = []
    for tr_idx, val_idx in cv.split(X_tr_arr, Y_tr_arr):
        X_f, X_v = X_tr_arr[tr_idx], X_tr_arr[val_idx]
        y_f, y_v = Y_tr_arr[tr_idx], Y_tr_arr[val_idx]
        w_f = weights[tr_idx]
        try:
            clf = NGBClassifier(verbose=False, **params)
            clf.fit(X_f, y_f, sample_weight=w_f)
            prob = clf.predict_proba(X_v)[:, 1]
            fold_scores.append(average_precision_score(y_v, prob))
        except Exception:
            fold_scores.append(0.0)
    mean_score = np.mean(fold_scores)
    print(f"  [{i+1}/{len(combos)}] {params}  PR-AUC={mean_score:.4f}")
    if mean_score > best_score:
        best_score  = mean_score
        best_params = params

print(f"\nBest params : {best_params}")
print(f"Best PR-AUC : {best_score:.4f}")

print("\nRetraining on full training set...")
best_ngb = NGBClassifier(verbose=True, **best_params)
best_ngb.fit(X_train.values, Y_train.values, sample_weight=weights)

Y_prob = best_ngb.predict_proba(X_test.values)[:, 1]

precision_arr, recall_arr, thresholds_arr = precision_recall_curve(Y_test, Y_prob)
f1_scores = np.where(
    (precision_arr[:-1] + recall_arr[:-1]) > 0,
    2 * precision_arr[:-1] * recall_arr[:-1] / (precision_arr[:-1] + recall_arr[:-1]),
    0
)
best_thresh_idx = np.argmax(f1_scores)
best_thresh     = thresholds_arr[best_thresh_idx]

print(f"\nBest threshold: {best_thresh:.4f}  F1={f1_scores[best_thresh_idx]:.4f}  "
      f"P={precision_arr[best_thresh_idx]:.4f}  R={recall_arr[best_thresh_idx]:.4f}")

Y_pred = (Y_prob >= best_thresh).astype(int)
print("\nClassification Report:")
print(classification_report(Y_test, Y_pred))

print("=" * 60)
print("PROBABILITY SPREAD")
print("=" * 60)
for k, v in [('min', Y_prob.min()), ('p10', np.percentile(Y_prob,10)),
             ('p25', np.percentile(Y_prob,25)), ('median', np.median(Y_prob)),
             ('mean', Y_prob.mean()), ('p75', np.percentile(Y_prob,75)),
             ('p90', np.percentile(Y_prob,90)), ('max', Y_prob.max()),
             ('std', Y_prob.std())]:
    print(f"  {k:>8}: {v:.4f}")

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(Y_prob, bins=50, color='steelblue', edgecolor='white', linewidth=0.5)
ax.axvline(Y_prob.mean(), color='red',   linestyle='--', label=f"Mean={Y_prob.mean():.3f}")
ax.axvline(best_thresh,   color='green', linestyle='-',  label=f"Threshold={best_thresh:.3f}")
ax.set_xlabel("Predicted Probability of Closure")
ax.set_ylabel("Count")
ax.set_title("Probability Spread — Business Closure Risk (NGBoost)")
ax.legend()
plt.tight_layout()
plt.savefig("probability_spread.png", dpi=150)
plt.show()

avg_precision = average_precision_score(Y_test, Y_prob)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(recall_arr, precision_arr, color='darkorange', lw=2,
        label=f"PR curve (AP={avg_precision:.3f})")
ax.axhline(Y_test.mean(), color='navy', linestyle='--',
           label=f"Baseline (prev={Y_test.mean():.3f})")
ax.scatter(recall_arr[best_thresh_idx], precision_arr[best_thresh_idx],
           color='green', zorder=5, label=f"Threshold={best_thresh:.3f}")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve (NGBoost)")
ax.legend(); ax.set_xlim([0,1]); ax.set_ylim([0,1.05])
plt.tight_layout()
plt.savefig("precision_recall_curve.png", dpi=150)
plt.show()
print(f"\nPR-AUC: {avg_precision:.4f}")

fop, mpv = calibration_curve(Y_test, Y_prob, n_bins=10, strategy='quantile')
brier = brier_score_loss(Y_test, Y_prob)
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot([0,1],[0,1],'k--', label='Perfect')
ax.plot(mpv, fop, 's-', color='darkorange', lw=2, label=f'NGBoost (Brier={brier:.4f})')
ax.set_xlabel("Mean Predicted Probability"); ax.set_ylabel("Fraction of Positives")
ax.set_title("Calibration Curve (NGBoost)"); ax.legend()
plt.tight_layout()
plt.savefig("calibration_curve.png", dpi=150)
plt.show()
print(f"Brier Score: {brier:.4f}")

print("\nSHAP VALUES")
explainer   = shap.TreeExplainer(best_ngb)
sample_size = min(2000, len(X_test))
X_shap      = X_test.iloc[:sample_size]
shap_values = explainer.shap_values(X_shap)
sv = shap_values[1] if isinstance(shap_values, list) else shap_values

plt.figure(figsize=(9, 6))
shap.summary_plot(sv, X_shap, plot_type='bar', show=False)
plt.title("SHAP Feature Importance")
plt.tight_layout(); plt.savefig("shap_importance_bar.png", dpi=150); plt.show()

plt.figure(figsize=(9, 6))
shap.summary_plot(sv, X_shap, show=False)
plt.title("SHAP Summary Plot")
plt.tight_layout(); plt.savefig("shap_summary_dot.png", dpi=150); plt.show()

highest_risk_idx = int(np.argmax(Y_prob[:sample_size]))
exp = shap.Explanation(
    values=sv[highest_risk_idx],
    base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list)
                else explainer.expected_value,
    data=X_shap.iloc[highest_risk_idx].values,
    feature_names=X_shap.columns.tolist(),
)
shap.plots.waterfall(exp, show=False)
plt.title(f"SHAP Waterfall — Top Risk (p={Y_prob[highest_risk_idx]:.3f})")
plt.tight_layout(); plt.savefig("shap_waterfall_top_risk.png", dpi=150); plt.show()

mean_abs_shap = pd.Series(np.abs(sv).mean(axis=0), index=X_shap.columns).sort_values(ascending=False)
print("\nTop-10 features by mean |SHAP|:")
print(mean_abs_shap.head(10).to_string())
