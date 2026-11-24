"""05 — Conformal Prediction 校准异常检测。

对比:
  1. One-class SVM (原始二分类)
  2. Split Conformal OC-SVM (带 p-value 和覆盖保证)

输出:
  results/figures/conformal_pvalues.png        — p-value 分布
  results/figures/conformal_coverage.png        — 覆盖率 vs 显著性水平
  results/tables/conformal_comparison.csv       — 方法对比表
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["figure.dpi"] = 150

from data.synthetic import generate_dataset
from preprocessing.signal import pipeline as preprocess_pipeline
from preprocessing.windows import sliding_windows, window_features
from models.oneclass import fit_oneclass, score as oc_score
from models.conformal import split_conformal_anomaly, conformal_with_multiple_alphas

FIG_DIR = ROOT / "results" / "figures"
TBL_DIR = ROOT / "results" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

# ── 1. 生成数据 ──────────────────────────────────────────────
print("Generating synthetic data...")
normal_data = generate_dataset(n_normal=80, n_repetitive=0, seed=42)
anomaly_data = generate_dataset(n_normal=0, n_repetitive=40, seed=99)

def extract_features(dataset):
    feats = []
    for s in dataset["samples"]:
        kp = s["kp"]
        kp_clean = preprocess_pipeline(kp)
        wins = sliding_windows(kp_clean, win=60, stride=30)
        for w in wins:
            f = window_features(w)
            if not np.any(np.isnan(f)):
                feats.append(f)
    return np.array(feats)

print("Extracting features...")
X_normal = extract_features(normal_data)
X_anomaly = extract_features(anomaly_data)

if len(X_normal) == 0 or len(X_anomaly) == 0:
    print("ERROR: No features extracted. Check synthetic data.")
    sys.exit(1)

print(f"  Normal windows: {len(X_normal)}")
print(f"  Anomaly windows: {len(X_anomaly)}")

# ── 2. 数据划分 ──────────────────────────────────────────────
n = len(X_normal)
idx = np.random.permutation(n)
n_train = int(n * 0.5)
n_cal = int(n * 0.25)

X_train = X_normal[idx[:n_train]]
X_cal = X_normal[idx[n_train:n_train + n_cal]]
X_test_normal = X_normal[idx[n_train + n_cal:]]
X_test = np.vstack([X_test_normal, X_anomaly])
y_test = np.concatenate([np.ones(len(X_test_normal)), -np.ones(len(X_anomaly))])

print(f"\n  Train: {len(X_train)}, Cal: {len(X_cal)}, "
      f"Test: {len(X_test_normal)} normal + {len(X_anomaly)} anomaly")

# ── 3. Conformal prediction ─────────────────────────────────
print("\nRunning conformal prediction at multiple α levels...")
results = conformal_with_multiple_alphas(
    X_train, X_cal, X_test, y_test=y_test,
    alphas=(0.01, 0.05, 0.1, 0.15, 0.2),
)

rows = []
for alpha, r in sorted(results.items()):
    print(f"  α={alpha:.2f}: guarantee={r['guarantee']:.0%}, "
          f"empirical_coverage={r.get('empirical_coverage', 'N/A')}, "
          f"detection_rate={r.get('detection_rate', 'N/A')}, "
          f"anomaly_rate={r['anomaly_rate']:.1%}")
    rows.append({
        "alpha": alpha,
        "coverage_guarantee": r["guarantee"],
        "empirical_coverage": r.get("empirical_coverage"),
        "detection_rate": r.get("detection_rate"),
        "anomaly_rate": r["anomaly_rate"],
        "n_flagged_anomaly": r["n_anomaly"],
    })

comp_df = pd.DataFrame(rows)

# ── 4. 对比: 原始 OC-SVM ────────────────────────────────────
print("\nBaseline: One-class SVM (no conformal calibration)...")
oc_model = fit_oneclass(X_train, nu=0.1, gamma="scale")
oc_scores = oc_score(oc_model, X_test)
oc_preds = oc_model.predict(X_test)  # 1=normal, -1=anomaly
normal_mask = y_test == 1
anomaly_mask = y_test == -1

oc_coverage = float((oc_preds[normal_mask] == 1).mean())
oc_detection = float((oc_preds[anomaly_mask] == -1).mean())
print(f"  OC-SVM coverage (normal): {oc_coverage:.1%}")
print(f"  OC-SVM detection (anomaly): {oc_detection:.1%}")

comp_df = pd.concat([comp_df, pd.DataFrame([{
    "alpha": "OC-SVM",
    "coverage_guarantee": "none",
    "empirical_coverage": oc_coverage,
    "detection_rate": oc_detection,
    "anomaly_rate": float((oc_preds == -1).mean()),
    "n_flagged_anomaly": int((oc_preds == -1).sum()),
}])], ignore_index=True)

comp_df.to_csv(TBL_DIR / "conformal_comparison.csv", index=False)
print(f"\nSaved: conformal_comparison.csv")

# ── 5. 图1: P-value 分布 ────────────────────────────────────
cr = results[0.1]["result"]
p_normal = cr.p_values[normal_mask]
p_anomaly = cr.p_values[anomaly_mask]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.hist(p_normal, bins=20, alpha=0.7, color="#4CAF50", label="Normal", density=True)
ax.hist(p_anomaly, bins=20, alpha=0.7, color="#F44336", label="Anomaly", density=True)
ax.axvline(0.1, color="black", linestyle="--", linewidth=1.5, label="α=0.10")
ax.set_xlabel("Conformal p-value", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title("P-value Distribution by True Label")
ax.legend()

ax = axes[1]
# P-value vs decision function (sorted)
sort_idx = np.argsort(cr.p_values)
colors = np.where(y_test[sort_idx] == 1, "#4CAF50", "#F44336")
ax.scatter(range(len(cr.p_values)), cr.p_values[sort_idx],
           c=colors, s=15, alpha=0.6)
ax.axhline(0.1, color="black", linestyle="--", linewidth=1.5, label="α=0.10")
ax.axhline(0.05, color="gray", linestyle=":", linewidth=1, label="α=0.05")
ax.set_xlabel("Sample index (sorted by p-value)", fontsize=11)
ax.set_ylabel("Conformal p-value", fontsize=11)
ax.set_title("Sorted P-values (green=normal, red=anomaly)")
ax.legend()

fig.suptitle("Conformal Prediction: Calibrated Anomaly Detection", fontsize=13)
fig.tight_layout()
fig.savefig(FIG_DIR / "conformal_pvalues.png", bbox_inches="tight")
plt.close()
print("Saved: conformal_pvalues.png")

# ── 6. 图2: Coverage vs α ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

alphas_plot = sorted([a for a in results.keys()])
guarantees = [1 - a for a in alphas_plot]
empirical = [results[a].get("empirical_coverage", np.nan) for a in alphas_plot]
detections = [results[a].get("detection_rate", np.nan) for a in alphas_plot]

ax = axes[0]
ax.plot(alphas_plot, guarantees, "k--", linewidth=2, label="Guarantee (1-α)")
ax.plot(alphas_plot, empirical, "o-", color="#2196F3", linewidth=2,
        markersize=8, label="Empirical coverage")
ax.fill_between(alphas_plot, guarantees, 1.0, alpha=0.1, color="green")
ax.set_xlabel("Significance level α", fontsize=11)
ax.set_ylabel("Coverage of normal samples", fontsize=11)
ax.set_title("Coverage Validity\n(empirical should be ≥ guarantee)")
ax.legend()
ax.set_ylim(0.5, 1.02)
ax.grid(alpha=0.3)

# OC-SVM baseline
ax.axhline(oc_coverage, color="#FF9800", linestyle="-.", linewidth=1.5,
           label=f"OC-SVM ({oc_coverage:.0%})")
ax.legend()

ax = axes[1]
ax.plot(alphas_plot, detections, "o-", color="#F44336", linewidth=2,
        markersize=8, label="Conformal detection rate")
ax.axhline(oc_detection, color="#FF9800", linestyle="-.",
           linewidth=1.5, label=f"OC-SVM ({oc_detection:.0%})")
ax.set_xlabel("Significance level α", fontsize=11)
ax.set_ylabel("Anomaly detection rate", fontsize=11)
ax.set_title("Detection Power\n(higher α → more detections but less coverage)")
ax.legend()
ax.grid(alpha=0.3)

fig.suptitle("Conformal Prediction vs One-class SVM", fontsize=13)
fig.tight_layout()
fig.savefig(FIG_DIR / "conformal_coverage.png", bbox_inches="tight")
plt.close()
print("Saved: conformal_coverage.png")

# ── 7. 汇总 ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("CONFORMAL PREDICTION SUMMARY")
print("=" * 60)
print(f"\nAt α=0.10 (90% coverage guarantee):")
r10 = results[0.1]
print(f"  Empirical coverage: {r10.get('empirical_coverage', 0):.1%} (guarantee: 90%)")
print(f"  Detection rate: {r10.get('detection_rate', 0):.1%}")
print(f"\nBaseline OC-SVM:")
print(f"  Coverage: {oc_coverage:.1%} (no guarantee)")
print(f"  Detection rate: {oc_detection:.1%}")
print(f"\nKey advantage: Conformal prediction provides a finite-sample")
print(f"coverage guarantee — the false positive rate is controlled at ≤ α")
print(f"regardless of the underlying distribution. This is critical for")
print(f"clinical applications where false alarms have real costs.")
