"""03_full_evaluation.py — 全方法对比与评估
==========================================

在合成数据上验证所有方法并生成对比表/图:
1. 预处理消融实验
2. HMM 多状态分段
3. 层级贝叶斯模型
4. 姿态几何 vs One-class SVM 检测对比
5. 方法汇总表

用法: python -I notebooks/03_full_evaluation.py
"""
# %%
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.ndimage import uniform_filter1d

from src.data.synthetic import generate_sequence, generate_dataset
from src.preprocessing.signal import (
    mask_low_confidence, interpolate_missing, despike, lowpass, normalize, pipeline
)
from src.preprocessing.windows import sliding_windows, window_features
from src.models.oneclass import fit_oneclass, score
from src.models.hmm import fit_hmm, find_state_boundaries, compute_dwell_times
from src.models.bayes_changepoint import single_changepoint
from src.models.bayes_hierarchical import (
    hierarchical_movement_model, shrinkage_plot_data
)
from src.models.vlm_analyzer import (
    pose_geometry_analysis, detect_arm_flapping, detect_rocking
)
from src.eval.metrics import segment_f1, boundary_error, oneclass_report

RESULTS = pathlib.Path("results/figures")
TABLES = pathlib.Path("results/tables")
RESULTS.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# 1. 预处理消融实验
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("1. PREPROCESSING ABLATION")
print("=" * 60)

rep = generate_sequence(300, mode="repetitive", seed=1)
kp_raw = rep["kp"]
conf = rep["conf"]
kp3 = np.concatenate([kp_raw, np.zeros_like(kp_raw[:, :, :1])], axis=-1)

# 不同预处理组合
ablation_configs = {
    "Raw (no preprocessing)":       lambda k, c: k.copy(),
    "Interpolation only":           lambda k, c: interpolate_missing(mask_low_confidence(k, c)),
    "Interp + Despike":             lambda k, c: despike(interpolate_missing(mask_low_confidence(k, c))),
    "Interp + Despike + Lowpass":   lambda k, c: lowpass(despike(interpolate_missing(mask_low_confidence(k, c)))),
    "Full pipeline":                lambda k, c: pipeline(k, c),
}

# 对每种配置提取特征,训练 one-class SVM
dataset = generate_dataset(n_normal=15, n_repetitive=10, T=300, seed=42)
ablation_results = []

for config_name, preprocess_fn in ablation_configs.items():
    all_feats = []
    all_labels = []
    for s, lab in zip(dataset["samples"], dataset["labels"]):
        kp = s["kp"]
        c = s["conf"]
        kp3_s = np.concatenate([kp, np.zeros_like(kp[:, :, :1])], axis=-1)
        try:
            cleaned = preprocess_fn(kp3_s, c)
            wins = sliding_windows(cleaned, win=60, stride=30)
            for w in wins:
                f = window_features(w)
                if np.isfinite(f).all():
                    all_feats.append(f)
                    all_labels.append(lab)
        except:
            pass

    X = np.array(all_feats)
    y = np.array(all_labels)

    if (y == 0).sum() > 5:
        model = fit_oneclass(X[y == 0], nu=0.1)
        preds = (model.predict(X) == -1).astype(int)
        from sklearn.metrics import f1_score, precision_score, recall_score
        ablation_results.append({
            "Config": config_name,
            "F1": f1_score(y, preds),
            "Precision": precision_score(y, preds),
            "Recall": recall_score(y, preds),
            "N_windows": len(y),
        })

df_ablation = pd.DataFrame(ablation_results)
print(df_ablation.to_string(index=False))
df_ablation.to_csv(TABLES / "preprocessing_ablation.csv", index=False)
print(f"Saved {TABLES / 'preprocessing_ablation.csv'}")

# 可视化
fig, ax = plt.subplots(figsize=(10, 5))
x = range(len(df_ablation))
width = 0.25
ax.bar([i - width for i in x], df_ablation["Precision"], width, label="Precision", color="steelblue")
ax.bar(x, df_ablation["F1"], width, label="F1", color="coral")
ax.bar([i + width for i in x], df_ablation["Recall"], width, label="Recall", color="seagreen")
ax.set_xticks(x)
ax.set_xticklabels([c.replace(" ", "\n") for c in df_ablation["Config"]], fontsize=8)
ax.set_ylabel("Score")
ax.set_title("Preprocessing Ablation — One-class SVM Detection")
ax.legend()
ax.set_ylim(0, 1.1)
fig.tight_layout()
fig.savefig(RESULTS / "preprocessing_ablation.png", dpi=150)
print("Saved preprocessing_ablation.png")

# ══════════════════════════════════════════════════════════════
# 2. HMM 多状态分段
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. HMM MULTI-STATE SEGMENTATION")
print("=" * 60)

cp_data = generate_sequence(300, mode="changepoint", seed=2)
kp_cp = cp_data["kp"]
vel = np.linalg.norm(np.diff(kp_cp[:, 4, :], axis=0), axis=-1) * 30
vel_smooth = uniform_filter1d(vel, 10)

# 拟合 3-state HMM
hmm_result = fit_hmm(vel_smooth, n_states=3, n_iter=100, seed=42)
print(f"HMM converged. Log-likelihood: {hmm_result.log_likelihood:.1f}")
print(f"State means: {hmm_result.means}")
print(f"Transition matrix:\n{np.round(hmm_result.trans_mat, 3)}")

# 停留时间
dwells = compute_dwell_times(hmm_result.state_seq, fs=30.0)
for k, v in dwells.items():
    print(f"  State {k}: {v['count']} segments, mean={v['mean_duration']:.2f}s")

# 可视化
state_names = ["Low activity", "Medium activity", "High activity"]
state_colors = ["#4CAF50", "#FF9800", "#F44336"]

fig, axes = plt.subplots(3, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [2, 1, 1]})
t = np.arange(len(vel_smooth)) / 30.0

# 信号 + 状态着色
axes[0].plot(t, vel_smooth, color="gray", lw=0.5, alpha=0.5)
for k in range(3):
    mask = hmm_result.state_seq == k
    axes[0].fill_between(t, 0, vel_smooth, where=mask, alpha=0.3,
                         color=state_colors[k], label=state_names[k])
axes[0].set_ylabel("RWrist Speed")
axes[0].set_title("HMM 3-State Segmentation")
axes[0].legend(loc="upper right")
axes[0].axvline(150 / 30, color="black", ls=":", alpha=0.5, label="True changepoint")

# 状态后验概率
for k in range(3):
    axes[1].plot(t, hmm_result.posteriors[:, k], color=state_colors[k],
                 label=state_names[k], alpha=0.8)
axes[1].set_ylabel("P(state)")
axes[1].set_title("State Posterior Probabilities")
axes[1].legend(loc="upper right", fontsize=8)

# 转移矩阵热力图
im = axes[2].imshow(hmm_result.trans_mat, cmap="Blues", vmin=0, vmax=1)
for i in range(3):
    for j in range(3):
        axes[2].text(j, i, f"{hmm_result.trans_mat[i,j]:.2f}",
                     ha="center", va="center", fontsize=10)
axes[2].set_xticks(range(3))
axes[2].set_yticks(range(3))
axes[2].set_xticklabels(state_names, fontsize=8)
axes[2].set_yticklabels(state_names, fontsize=8)
axes[2].set_title("Transition Matrix")
plt.colorbar(im, ax=axes[2], fraction=0.03)

fig.tight_layout()
fig.savefig(RESULTS / "hmm_segmentation.png", dpi=150)
print("Saved hmm_segmentation.png")

# HMM 边界检测评估
hmm_boundaries = np.array(find_state_boundaries(hmm_result.state_seq))
true_boundary = np.array([150])  # 真实变点
seg_f1 = segment_f1(hmm_boundaries, true_boundary, tolerance=15)
bnd_err = boundary_error(hmm_boundaries, true_boundary)
print(f"HMM segment F1 (tol=15): {seg_f1:.3f}")
print(f"HMM boundary error: {bnd_err:.1f} frames ({bnd_err/30:.2f}s)")

# ══════════════════════════════════════════════════════════════
# 3. 层级贝叶斯模型
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3. HIERARCHICAL BAYESIAN MODEL")
print("=" * 60)

# 模拟 6 个"儿童",每人 5-15 个窗口
rng = np.random.default_rng(42)
features_by_child = {}
for i in range(6):
    child_mean = rng.normal(1.0, 0.3)
    n_windows = rng.integers(5, 16)
    features_by_child[f"child_{i:02d}"] = rng.normal(child_mean, 0.2, n_windows)

print(f"Simulated {len(features_by_child)} children")
for cid, vals in features_by_child.items():
    print(f"  {cid}: n={len(vals)}, mean={vals.mean():.3f}")

print("\nFitting hierarchical model...")
idata, model = hierarchical_movement_model(features_by_child, draws=500, tune=500, seed=42)

# Shrinkage plot
shrink = shrinkage_plot_data(idata, features_by_child)

fig, ax = plt.subplots(figsize=(10, 6))
pop_mean = list(shrink.values())[0]["pop_mean"]
ax.axhline(pop_mean, color="red", ls="--", alpha=0.5, label=f"Population mean: {pop_mean:.3f}")

for i, (cid, s) in enumerate(shrink.items()):
    ax.plot([i, i], [s["raw_mean"], s["posterior_mean"]], "k-", alpha=0.3)
    ax.scatter(i, s["raw_mean"], c="steelblue", s=s["n_obs"] * 10, alpha=0.6,
               edgecolors="k", zorder=5)
    ax.scatter(i, s["posterior_mean"], c="coral", s=s["n_obs"] * 10, alpha=0.8,
               edgecolors="k", zorder=5, marker="D")
    ax.errorbar(i, s["posterior_mean"], yerr=s["posterior_std"],
                fmt="none", color="coral", alpha=0.5, capsize=3)

ax.set_xticks(range(len(shrink)))
ax.set_xticklabels(list(shrink.keys()), rotation=30, fontsize=8)
ax.set_ylabel("Mean Movement Speed")
ax.set_title("Hierarchical Bayesian Shrinkage\n(circles=raw, diamonds=posterior)")
ax.scatter([], [], c="steelblue", label="Raw sample mean", edgecolors="k")
ax.scatter([], [], c="coral", marker="D", label="Posterior mean (shrunk)", edgecolors="k")
ax.legend()
fig.tight_layout()
fig.savefig(RESULTS / "hierarchical_shrinkage.png", dpi=150)
print("Saved hierarchical_shrinkage.png")

# 后验汇总
mu_pop_samples = idata.posterior["mu_pop"].values.flatten()
sigma_pop_samples = idata.posterior["sigma_pop"].values.flatten()
print(f"\nPopulation mean: {mu_pop_samples.mean():.3f} ± {mu_pop_samples.std():.3f}")
print(f"Population SD: {sigma_pop_samples.mean():.3f} ± {sigma_pop_samples.std():.3f}")

# ══════════════════════════════════════════════════════════════
# 4. 姿态几何检测 vs One-class SVM
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("4. POSE GEOMETRY vs ONE-CLASS SVM")
print("=" * 60)

dataset2 = generate_dataset(n_normal=20, n_repetitive=15, T=300, seed=99)

# 方法 A: One-class SVM
svm_feats = []
svm_labels = []
for s, lab in zip(dataset2["samples"], dataset2["labels"]):
    kp = s["kp"]
    kp3_s = np.concatenate([kp, np.zeros_like(kp[:, :, :1])], axis=-1)
    cleaned = pipeline(kp3_s, s["conf"])
    wins = sliding_windows(cleaned, win=60, stride=30)
    for w in wins:
        f = window_features(w)
        if np.isfinite(f).all():
            svm_feats.append(f)
            svm_labels.append(lab)

X_svm = np.array(svm_feats)
y_svm = np.array(svm_labels)
svm_model = fit_oneclass(X_svm[y_svm == 0], nu=0.1)
svm_preds = (svm_model.predict(X_svm) == -1).astype(int)

# 方法 B: 姿态几何
geo_preds_per_sample = []
geo_labels_per_sample = []
for s, lab in zip(dataset2["samples"], dataset2["labels"]):
    kp = s["kp"]
    kp3_s = np.concatenate([kp, np.zeros_like(kp[:, :, :1])], axis=-1)
    cleaned = pipeline(kp3_s, s["conf"])
    result = pose_geometry_analysis(cleaned)
    geo_preds_per_sample.append(1 if result.is_repetitive else 0)
    geo_labels_per_sample.append(lab)

geo_preds_arr = np.array(geo_preds_per_sample)
geo_labels_arr = np.array(geo_labels_per_sample)

# 汇总
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

comparison = pd.DataFrame([
    {
        "Method": "One-class SVM (window-level)",
        "Accuracy": accuracy_score(y_svm, svm_preds),
        "Precision": precision_score(y_svm, svm_preds),
        "Recall": recall_score(y_svm, svm_preds),
        "F1": f1_score(y_svm, svm_preds),
        "Granularity": "window (2s)",
    },
    {
        "Method": "Pose Geometry Rules (sample-level)",
        "Accuracy": accuracy_score(geo_labels_arr, geo_preds_arr),
        "Precision": precision_score(geo_labels_arr, geo_preds_arr),
        "Recall": recall_score(geo_labels_arr, geo_preds_arr),
        "F1": f1_score(geo_labels_arr, geo_preds_arr),
        "Granularity": "sample (10s)",
    },
])

print(comparison.to_string(index=False))
comparison.to_csv(TABLES / "method_comparison.csv", index=False)

# ══════════════════════════════════════════════════════════════
# 5. 贝叶斯变点 vs HMM 分段精度
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("5. BAYESIAN CHANGEPOINT vs HMM SEGMENTATION")
print("=" * 60)

# 多次实验取平均
cp_results = []
for trial_seed in range(5):
    cp_data = generate_sequence(300, mode="changepoint", seed=100 + trial_seed)
    kp_cp = cp_data["kp"]
    vel = np.linalg.norm(np.diff(kp_cp[:, 4, :], axis=0), axis=-1) * 30
    vel_smooth = uniform_filter1d(vel, 10)

    # Bayesian
    idata_cp = single_changepoint(vel_smooth, draws=300, tune=300, seed=trial_seed)
    tau_median = np.median(idata_cp.posterior["tau"].values.flatten())
    bayes_err = abs(tau_median - 150)

    # HMM
    hmm_res = fit_hmm(vel_smooth, n_states=2, n_iter=80, seed=trial_seed)
    hmm_bounds = np.array(find_state_boundaries(hmm_res.state_seq))
    if len(hmm_bounds) > 0:
        hmm_err = min(abs(hmm_bounds - 150))
    else:
        hmm_err = 150

    cp_results.append({
        "Trial": trial_seed,
        "Bayes_error_frames": bayes_err,
        "Bayes_error_sec": bayes_err / 30,
        "HMM_error_frames": hmm_err,
        "HMM_error_sec": hmm_err / 30,
    })

df_cp = pd.DataFrame(cp_results)
print(df_cp.to_string(index=False))
print(f"\nMean boundary error — Bayes: {df_cp['Bayes_error_sec'].mean():.3f}s, "
      f"HMM: {df_cp['HMM_error_sec'].mean():.3f}s")
df_cp.to_csv(TABLES / "changepoint_comparison.csv", index=False)

# ══════════════════════════════════════════════════════════════
# 6. 综合方法对比表
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("6. COMPREHENSIVE METHOD COMPARISON TABLE")
print("=" * 60)

summary_table = pd.DataFrame([
    {
        "Method": "Bayesian Changepoint",
        "Task": "Activity segmentation",
        "Key metric": f"Boundary error: {df_cp['Bayes_error_sec'].mean():.3f}s",
        "Uncertainty": "Yes (posterior CI)",
        "Supervision": "Unsupervised",
        "Complexity": "O(T·draws)",
    },
    {
        "Method": "HMM (3-state)",
        "Task": "Multi-state segmentation",
        "Key metric": f"Boundary error: {df_cp['HMM_error_sec'].mean():.3f}s",
        "Uncertainty": "Yes (state posteriors)",
        "Supervision": "Unsupervised",
        "Complexity": "O(T·K²·iter)",
    },
    {
        "Method": "One-class SVM",
        "Task": "Anomaly detection",
        "Key metric": f"F1: {comparison.iloc[0]['F1']:.3f}",
        "Uncertainty": "No (decision boundary)",
        "Supervision": "Semi-supervised (normal only)",
        "Complexity": "O(n²·d)",
    },
    {
        "Method": "Pose Geometry Rules",
        "Task": "Behavior classification",
        "Key metric": f"F1: {comparison.iloc[1]['F1']:.3f}",
        "Uncertainty": "No",
        "Supervision": "Unsupervised (domain knowledge)",
        "Complexity": "O(T·J)",
    },
    {
        "Method": "Hierarchical Bayesian",
        "Task": "Individual differences",
        "Key metric": f"Pop σ: {sigma_pop_samples.mean():.3f}",
        "Uncertainty": "Yes (shrinkage + CI)",
        "Supervision": "Unsupervised",
        "Complexity": "O(N·draws)",
    },
    {
        "Method": "VLM Contextual",
        "Task": "Scene understanding",
        "Key metric": "Qualitative (needs API key)",
        "Uncertainty": "No (prompt-dependent)",
        "Supervision": "Zero-shot",
        "Complexity": "O(frames · API_call)",
    },
])

print(summary_table.to_string(index=False))
summary_table.to_csv(TABLES / "comprehensive_comparison.csv", index=False)
print(f"\nSaved to {TABLES / 'comprehensive_comparison.csv'}")

# ══════════════════════════════════════════════════════════════
# 7. 最终结果可视化: 时间轴对比图
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("7. TIMELINE COMPARISON FIGURE")
print("=" * 60)

cp_data = generate_sequence(300, mode="changepoint", seed=2)
kp_cp = cp_data["kp"]
vel = np.linalg.norm(np.diff(kp_cp[:, 4, :], axis=0), axis=-1) * 30
vel_smooth = uniform_filter1d(vel, 10)
t = np.arange(len(vel_smooth)) / 30.0

# 三种方法的分段结果
hmm_res = fit_hmm(vel_smooth, n_states=3, n_iter=100, seed=42)
idata_cp = single_changepoint(vel_smooth, draws=500, tune=500, seed=42)
tau_samples = idata_cp.posterior["tau"].values.flatten()

fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

# Raw signal
axes[0].plot(t, vel_smooth, color="steelblue", lw=0.8)
axes[0].axvline(150/30, color="red", ls="--", alpha=0.5)
axes[0].set_ylabel("Speed")
axes[0].set_title("Activity Timeline — Method Comparison")

# HMM
for k in range(3):
    mask = hmm_res.state_seq == k
    axes[1].fill_between(t, 0, 1, where=mask, alpha=0.5,
                         color=state_colors[k], label=state_names[k])
axes[1].set_ylabel("HMM State")
axes[1].set_yticks([])
axes[1].legend(loc="upper right", fontsize=8)

# Bayesian changepoint
axes[2].hist(tau_samples / 30, bins=50, density=True, color="coral", alpha=0.7)
axes[2].axvline(150/30, color="red", ls="--")
axes[2].axvline(np.median(tau_samples)/30, color="blue", ls="--",
                label=f"median: {np.median(tau_samples)/30:.1f}s")
axes[2].set_ylabel("τ posterior")
axes[2].legend(fontsize=8)

# Pose geometry detection (per window)
kp3_cp = np.concatenate([kp_cp, np.zeros_like(kp_cp[:, :, :1])], axis=-1)
cleaned_cp = pipeline(kp3_cp, cp_data["conf"])
wins_cp = sliding_windows(cleaned_cp, win=60, stride=15)
geo_scores = []
for w in wins_cp:
    flap = detect_arm_flapping(w)
    geo_scores.append(1 if flap["is_detected"] else 0)

win_times = np.arange(len(geo_scores)) * 15 / 30 + 1  # center of window
axes[3].bar(win_times, geo_scores, width=0.4, color="seagreen", alpha=0.6)
axes[3].set_ylabel("Flapping\ndetected")
axes[3].set_xlabel("Time (s)")
axes[3].set_yticks([0, 1])

fig.tight_layout()
fig.savefig(RESULTS / "timeline_comparison.png", dpi=150)
print("Saved timeline_comparison.png")

print("\n" + "=" * 60)
print("✅ FULL EVALUATION COMPLETE")
print("=" * 60)
print(f"\nFigures: {list(RESULTS.glob('*.png'))}")
print(f"Tables: {list(TABLES.glob('*.csv'))}")
