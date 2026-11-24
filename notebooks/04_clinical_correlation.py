"""04 — ADOS 临床评分与计算运动特征关联分析。

核心问题：骨架数据中提取的运动特征能否预测 ASD 严重度（ADOS 评分）？

输出:
  results/figures/ados_correlation_heatmap.png   — 特征-评分 Spearman 相关矩阵
  results/figures/ados_scatter_matrix.png         — 关键特征 vs ADOS 散点图
  results/figures/ados_regression.png             — 多元回归预测 vs 真实值
  results/figures/ados_severity_groups.png        — 按严重度分组的运动特征分布
  results/tables/ados_correlation_table.csv       — 完整相关表
  results/tables/regression_results.csv           — 回归系数表
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["figure.dpi"] = 150

from scipy import stats
from data.mmasd_loader import discover_samples, load_2d_sequence, select_target_person
from preprocessing.signal import pipeline as preprocess_pipeline
from analysis.clinical_correlation import (
    load_ados_scores,
    compute_child_features,
    correlate_features_with_ados,
    regression_analysis,
    ChildProfile,
)

MMASD_ROOT = ROOT / "data" / "raw" / "MMASD"
FIG_DIR = ROOT / "results" / "figures"
TBL_DIR = ROOT / "results" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. 加载 ADOS 评分 ────────────────────────────────────────
print("Loading ADOS scores...")
ados_df = load_ados_scores(MMASD_ROOT / "ADOS_rating.xlsx")
print(f"  {len(ados_df)} entries, columns: {ados_df.columns.tolist()}")

# ── 2. 加载骨架数据并按儿童聚合 ──────────────────────────────
print("\nDiscovering skeleton samples...")
samples = discover_samples(MMASD_ROOT, modality="2d")
print(f"  {len(samples)} total samples")

# 按 child_id 分组
from collections import defaultdict
child_samples = defaultdict(list)
for s in samples:
    child_samples[s.child_id].append(s)

# 只保留有 ADOS 评分的儿童
ados_ids = set(ados_df["child_id"].values)
matched_children = {cid: samps for cid, samps in child_samples.items() if cid in ados_ids}
print(f"  {len(matched_children)} children with both skeleton data and ADOS scores")

# ── 3. 提取每个儿童的运动特征 ─────────────────────────────────
print("\nExtracting movement features per child...")
profiles = []

for cid, samps in sorted(matched_children.items()):
    sequences = []
    n_loaded = 0
    for s in samps[:30]:  # 每个儿童最多用 30 个片段
        try:
            pid = select_target_person(s.path, n_sample=3)
            kp = load_2d_sequence(s.path, pid)
            kp_clean = preprocess_pipeline(kp)
            if kp_clean.shape[0] >= 10:
                sequences.append(kp_clean)
                n_loaded += 1
        except Exception:
            continue

    if not sequences:
        print(f"  Child {cid}: no valid sequences, skipping")
        continue

    feats = compute_child_features(cid, sequences)
    if not feats:
        continue

    # 匹配 ADOS 评分
    row = ados_df[ados_df["child_id"] == cid].iloc[0]

    p = ChildProfile(
        child_id=cid,
        n_samples=feats["n_samples"],
        mean_speed=feats["mean_speed"],
        std_speed=feats["std_speed"],
        mean_accel=feats["mean_accel"],
        dominant_freq=feats["dominant_freq"],
        movement_variability=feats["movement_variability"],
        high_freq_power=feats["high_freq_power"],
        sa_total=row["sa_total"] if pd.notna(row["sa_total"]) else None,
        rrb_total=row["rrb_total"] if pd.notna(row["rrb_total"]) else None,
        overall_total=row["overall_total"] if pd.notna(row["overall_total"]) else None,
        comparison_score=row["comparison_score"] if pd.notna(row["comparison_score"]) else None,
    )
    profiles.append(p)
    print(f"  Child {cid}: {n_loaded} clips, speed={feats['mean_speed']:.2f}, "
          f"ADOS={row['overall_total']}")

print(f"\n{len(profiles)} children with complete profiles")

if len(profiles) < 5:
    print("ERROR: Too few profiles for analysis. Check data paths.")
    sys.exit(1)

# ── 4. Spearman 相关分析 ──────────────────────────────────────
print("\nComputing Spearman correlations...")
corr_df = correlate_features_with_ados(profiles)
corr_df.to_csv(TBL_DIR / "ados_correlation_table.csv", index=False)
print(corr_df.to_string())

sig = corr_df[corr_df["significant"]]
if len(sig) > 0:
    print(f"\n  {len(sig)} significant correlations (p < 0.05):")
    for _, r in sig.iterrows():
        print(f"    {r['feature']} ↔ {r['ados_subscale']}: "
              f"ρ={r['spearman_rho']:.3f}, p={r['p_value']:.4f}")
else:
    print("\n  No significant correlations at p < 0.05 (expected with small N)")

# ── 5. 相关热力图 ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

feature_labels = {
    "mean_speed": "Mean Speed",
    "std_speed": "Speed Std",
    "mean_accel": "Mean Accel",
    "dominant_freq": "Dom. Freq",
    "movement_variability": "Mov. Variability",
    "high_freq_power": "HF Power (2-6Hz)",
}
ados_labels = {
    "sa_total": "Social Affect",
    "rrb_total": "RRB",
    "overall_total": "Overall Total",
    "comparison_score": "Comparison Score",
}

feat_order = list(feature_labels.keys())
ados_order = list(ados_labels.keys())

rho_matrix = np.full((len(feat_order), len(ados_order)), np.nan)
p_matrix = np.full((len(feat_order), len(ados_order)), np.nan)

for _, row in corr_df.iterrows():
    if row["feature"] in feat_order and row["ados_subscale"] in ados_order:
        fi = feat_order.index(row["feature"])
        ai = ados_order.index(row["ados_subscale"])
        rho_matrix[fi, ai] = row["spearman_rho"]
        p_matrix[fi, ai] = row["p_value"]

im = ax.imshow(rho_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax, label="Spearman ρ", shrink=0.8)

for i in range(len(feat_order)):
    for j in range(len(ados_order)):
        if not np.isnan(rho_matrix[i, j]):
            color = "white" if abs(rho_matrix[i, j]) > 0.5 else "black"
            star = "*" if p_matrix[i, j] < 0.05 else ""
            star = "**" if p_matrix[i, j] < 0.01 else star
            ax.text(j, i, f"{rho_matrix[i, j]:.2f}{star}",
                    ha="center", va="center", fontsize=9, color=color)

ax.set_xticks(range(len(ados_order)))
ax.set_xticklabels([ados_labels[a] for a in ados_order], rotation=30, ha="right")
ax.set_yticks(range(len(feat_order)))
ax.set_yticklabels([feature_labels[f] for f in feat_order])
ax.set_title("Movement Features ↔ ADOS Clinical Scores\n(Spearman correlation, *p<.05 **p<.01)")
fig.tight_layout()
fig.savefig(FIG_DIR / "ados_correlation_heatmap.png", bbox_inches="tight")
plt.close()
print("\nSaved: ados_correlation_heatmap.png")

# ── 6. 散点图矩阵 ────────────────────────────────────────────
key_features = ["mean_speed", "high_freq_power", "movement_variability"]
key_ados = ["rrb_total", "comparison_score"]

fig, axes = plt.subplots(len(key_features), len(key_ados),
                         figsize=(10, 12), squeeze=False)

for i, feat in enumerate(key_features):
    for j, ados in enumerate(key_ados):
        ax = axes[i, j]
        x = [getattr(p, feat) for p in profiles if getattr(p, ados) is not None]
        y = [getattr(p, ados) for p in profiles if getattr(p, ados) is not None]
        ids = [p.child_id for p in profiles if getattr(p, ados) is not None]

        x, y = np.array(x), np.array(y)
        valid = ~(np.isnan(x) | np.isnan(y))
        x, y = x[valid], y[valid]
        ids_valid = [ids[k] for k in range(len(ids)) if valid[k]]

        ax.scatter(x, y, c="#2196F3", s=50, alpha=0.7, edgecolors="white", linewidth=0.5)

        # 标注 child ID
        for k, cid in enumerate(ids_valid):
            ax.annotate(cid, (x[k], y[k]), fontsize=6, alpha=0.6,
                        xytext=(3, 3), textcoords="offset points")

        # 趋势线
        if len(x) >= 5:
            rho, pval = stats.spearmanr(x, y)
            z = np.polyfit(x, y, 1)
            xline = np.linspace(x.min(), x.max(), 50)
            ax.plot(xline, np.polyval(z, xline), "r--", alpha=0.5, linewidth=1.5)
            ax.set_title(f"ρ={rho:.2f}, p={pval:.3f}", fontsize=10)

        ax.set_xlabel(feature_labels.get(feat, feat))
        ax.set_ylabel(ados_labels.get(ados, ados))

fig.suptitle("Key Movement Features vs ADOS Subscales", fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(FIG_DIR / "ados_scatter_matrix.png", bbox_inches="tight")
plt.close()
print("Saved: ados_scatter_matrix.png")

# ── 7. 多元回归 ──────────────────────────────────────────────
print("\nMultiple regression: Movement features → ADOS Comparison Score")
reg = regression_analysis(profiles, target="comparison_score")

if "error" not in reg:
    print(f"  R² = {reg['r_squared']:.3f}")
    print(f"  Adjusted R² = {reg['adj_r_squared']:.3f}")
    print(f"  LOO-CV R² = {reg['loo_r_squared']:.3f}")
    print(f"  N = {reg['n_children']}")
    print("  Coefficients:")
    for name, coef in reg["coefficients"].items():
        print(f"    {name}: {coef:.4f}")

    # 保存回归系数表
    reg_df = pd.DataFrame([
        {"feature": k, "coefficient": v}
        for k, v in reg["coefficients"].items()
    ])
    reg_df.to_csv(TBL_DIR / "regression_results.csv", index=False)

    # 预测 vs 真实图
    fig, ax = plt.subplots(figsize=(6, 6))
    y_true = np.array(reg["y_true"])
    y_pred = np.array(reg["y_pred"])

    ax.scatter(y_true, y_pred, c="#4CAF50", s=60, alpha=0.7,
               edgecolors="white", linewidth=0.5)
    lims = [min(y_true.min(), y_pred.min()) - 0.5,
            max(y_true.max(), y_pred.max()) + 0.5]
    ax.plot(lims, lims, "k--", alpha=0.3, label="Perfect prediction")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("True ADOS Comparison Score", fontsize=11)
    ax.set_ylabel("Predicted Score (from movement features)", fontsize=11)
    ax.set_title(
        f"Movement → ADOS Prediction\n"
        f"R²={reg['r_squared']:.3f}, LOO-CV R²={reg['loo_r_squared']:.3f}",
        fontsize=12,
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ados_regression.png", bbox_inches="tight")
    plt.close()
    print("Saved: ados_regression.png")
else:
    print(f"  Regression failed: {reg['error']}")

# ── 8. 按严重度分组对比 ──────────────────────────────────────
print("\nSeverity group comparison...")

# 按 comparison score 分组: mild (1-4), moderate (5-7), severe (8-10)
groups = {"Mild (1-4)": [], "Moderate (5-7)": [], "Severe (8-10)": []}
for p in profiles:
    if p.comparison_score is None or np.isnan(p.comparison_score):
        continue
    s = p.comparison_score
    if s <= 4:
        groups["Mild (1-4)"].append(p)
    elif s <= 7:
        groups["Moderate (5-7)"].append(p)
    else:
        groups["Severe (8-10)"].append(p)

print(f"  Group sizes: {', '.join(f'{k}: {len(v)}' for k, v in groups.items())}")

plot_features = [
    ("mean_speed", "Mean Movement Speed"),
    ("mean_accel", "Mean Acceleration"),
    ("high_freq_power", "HF Power (2-6 Hz)"),
    ("movement_variability", "Movement Variability"),
]

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
colors = {"Mild (1-4)": "#4CAF50", "Moderate (5-7)": "#FF9800", "Severe (8-10)": "#F44336"}

for idx, (feat, label) in enumerate(plot_features):
    ax = axes.flat[idx]
    group_data = []
    group_names = []
    group_colors = []

    for gname, gprofiles in groups.items():
        vals = [getattr(p, feat) for p in gprofiles]
        if vals:
            group_data.append(vals)
            group_names.append(f"{gname}\n(n={len(vals)})")
            group_colors.append(colors[gname])

    if group_data:
        bp = ax.boxplot(group_data, labels=group_names, patch_artist=True,
                        widths=0.6, showmeans=True,
                        meanprops=dict(marker="D", markerfacecolor="black", markersize=5))
        for patch, color in zip(bp["boxes"], group_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)

        # 叠加散点
        for gi, vals in enumerate(group_data):
            jitter = np.random.default_rng(42).normal(0, 0.05, len(vals))
            ax.scatter(np.full(len(vals), gi + 1) + jitter, vals,
                       c=group_colors[gi], s=30, alpha=0.7, zorder=5,
                       edgecolors="white", linewidth=0.3)

    ax.set_ylabel(label)
    ax.set_title(label)

    # Kruskal-Wallis test (non-parametric, for small samples)
    if len(group_data) >= 2 and all(len(g) >= 2 for g in group_data):
        try:
            h_stat, kw_p = stats.kruskal(*group_data)
            ax.text(0.95, 0.95, f"KW p={kw_p:.3f}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))
        except Exception:
            pass

fig.suptitle("Movement Features by ADOS Severity Group", fontsize=13)
fig.tight_layout()
fig.savefig(FIG_DIR / "ados_severity_groups.png", bbox_inches="tight")
plt.close()
print("Saved: ados_severity_groups.png")

# ── 9. 汇总 ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("CLINICAL CORRELATION ANALYSIS SUMMARY")
print("=" * 60)
print(f"Children analyzed: {len(profiles)}")
print(f"ADOS score range: {min(p.overall_total for p in profiles if p.overall_total):.0f} - "
      f"{max(p.overall_total for p in profiles if p.overall_total):.0f}")

if len(sig) > 0:
    print(f"\nSignificant correlations ({len(sig)}):")
    for _, r in sig.iterrows():
        direction = "↑" if r["spearman_rho"] > 0 else "↓"
        print(f"  {r['feature']} {direction} {r['ados_subscale']}: "
              f"ρ={r['spearman_rho']:.3f} (p={r['p_value']:.4f})")
else:
    print("\nNo significant individual correlations (small N = expected)")
    print("Trend-level correlations (p < 0.15):")
    trends = corr_df[corr_df["p_value"] < 0.15]
    for _, r in trends.iterrows():
        direction = "↑" if r["spearman_rho"] > 0 else "↓"
        print(f"  {r['feature']} {direction} {r['ados_subscale']}: "
              f"ρ={r['spearman_rho']:.3f} (p={r['p_value']:.4f})")

if "error" not in reg:
    print(f"\nMultivariate prediction R² = {reg['r_squared']:.3f} "
          f"(LOO-CV: {reg['loo_r_squared']:.3f})")
    if reg["loo_r_squared"] > 0:
        print("  → Positive LOO-CV R²: movement features carry predictive signal")
    else:
        print("  → Negative LOO-CV R²: model overfits (more children needed)")

print(f"\nFigures: {FIG_DIR}/ados_*.png")
print(f"Tables:  {TBL_DIR}/ados_*.csv, regression_results.csv")
