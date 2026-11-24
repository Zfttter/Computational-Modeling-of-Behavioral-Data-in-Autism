"""02_pipeline_real_data.py — 真实 MMASD 数据全流程
================================================

数据下载后运行: python -I notebooks/02_pipeline_real_data.py

覆盖简历三条:
1. 贝叶斯推断 → 变点检测 + 后验不确定性
2. 骨架预处理 → 滤波/插值/归一化
3. 检测方法 → one-class SVM + 姿态特征
"""
# %%
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.data.mmasd_loader import (
    discover_samples, load_2d_sequence, select_target_person,
    parse_sample_id, ACTIVITY_MAP, LABEL_MAP, BODY_25_JOINTS, BODY_25_SKELETON,
)
from src.preprocessing.signal import pipeline as preprocess_pipeline
from src.preprocessing.windows import sliding_windows, window_features
from src.models.oneclass import fit_oneclass, score

RESULTS = pathlib.Path("results/figures")
RESULTS.mkdir(parents=True, exist_ok=True)
DATA_ROOT = pathlib.Path("data/raw/MMASD")

# ── 1. 发现数据 ──────────────────────────────────────────────
print("Discovering samples...")
samples = discover_samples(DATA_ROOT, modality="2d")
print(f"Found {len(samples)} 2D skeleton samples")

if len(samples) == 0:
    print("❌ 没有找到数据。请先运行 scripts/01_download_data.sh 查看下载说明。")
    sys.exit(1)

# 统计
df = pd.DataFrame([{
    "activity": ACTIVITY_MAP.get(s.activity, s.activity),
    "child": s.child_id,
    "day": s.day,
    "label": LABEL_MAP.get(s.label, s.label),
} for s in samples])

print("\n活动分布:")
print(df["activity"].value_counts())
print(f"\n儿童数: {df['child'].nunique()}")
print(f"标签分布:\n{df['label'].value_counts()}")

# ── 2. 加载 + 预处理 ────────────────────────────────────────
print("\nLoading and preprocessing...")
processed = []
meta = []
for s in tqdm(samples[:200]):  # 先加载前200个
    try:
        pid = select_target_person(s.path, n_sample=3)
        kp = load_2d_sequence(s.path, pid)
        # 添加 z=0 维度
        kp3 = np.concatenate([kp[:, :, :2], np.zeros((*kp.shape[:2], 1))], axis=-1)
        conf = kp[:, :, 2]
        cleaned = preprocess_pipeline(kp3, conf, fs=30.0)
        processed.append({"raw": kp, "cleaned": cleaned, "sample": s})
        meta.append({
            "activity": s.activity,
            "label": s.label,
            "child": s.child_id,
            "n_frames": kp.shape[0],
            "mean_conf": conf.mean(),
            "missing_rate": (conf < 0.3).mean(),
        })
    except Exception as e:
        print(f"  Skip {s.path.name}: {e}")

df_meta = pd.DataFrame(meta)
print(f"\nSuccessfully loaded: {len(processed)} samples")
print(f"Mean frames: {df_meta['n_frames'].mean():.0f}")
print(f"Mean confidence: {df_meta['mean_conf'].mean():.3f}")
print(f"Mean missing rate: {df_meta['missing_rate'].mean():.1%}")

# ── 3. 预处理可视化 ──────────────────────────────────────────
if len(processed) > 0:
    s0 = processed[0]
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    ji = 4  # RWrist
    t = np.arange(s0["raw"].shape[0]) / 30.0
    axes[0].plot(t, s0["raw"][:, ji, 1], alpha=0.5, label="Raw Y")
    axes[0].plot(t, s0["cleaned"][:, ji, 1], label="Cleaned Y")
    axes[0].set_ylabel("RWrist Y")
    axes[0].legend()
    axes[0].set_title(f"Sample: {s0['sample'].path.name}")

    axes[1].plot(t, s0["raw"][:, ji, 2], color="orange")
    axes[1].set_ylabel("Confidence")
    axes[1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(RESULTS / "real_preprocessing.png", dpi=150)
    print("Saved real_preprocessing.png")

# ── 4. 特征提取 ─────────────────────────────────────────────
print("\nExtracting features...")
all_feats = []
all_labels = []
all_activities = []

for p in tqdm(processed):
    wins = sliding_windows(p["cleaned"], win=60, stride=30)
    for w in wins:
        try:
            f = window_features(w)
            if np.isfinite(f).all():
                all_feats.append(f)
                all_labels.append(p["sample"].label)
                all_activities.append(p["sample"].activity)
        except:
            pass

X = np.array(all_feats)
y_label = np.array(all_labels)
y_activity = np.array(all_activities)
print(f"Total windows: {X.shape[0]}, Feature dim: {X.shape[1]}")

# ── 5. One-class SVM ────────────────────────────────────────
# 用 "pass (y)" 标签的样本训练,看 "fail (n)" 是否被检测为异常
mask_pass = y_label == "y"
mask_fail = y_label == "n"

if mask_pass.sum() > 10 and mask_fail.sum() > 5:
    model = fit_oneclass(X[mask_pass], nu=0.1)
    scores_pass = score(model, X[mask_pass])
    scores_fail = score(model, X[mask_fail])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(scores_pass, bins=30, alpha=0.6, label="Pass", color="steelblue", density=True)
    ax.hist(scores_fail, bins=30, alpha=0.6, label="Fail", color="salmon", density=True)
    ax.axvline(0, color="k", ls="--", label="Decision boundary")
    ax.set_xlabel("Decision function score")
    ax.set_title("One-class SVM: Pass vs Fail")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "real_oneclass.png", dpi=150)
    print(f"\nAnomaly rate — Pass: {(scores_pass<0).mean():.1%}, Fail: {(scores_fail<0).mean():.1%}")

# ── 6. 活动分类特征嵌入 ──────────────────────────────────────
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# PCA
pca = PCA(n_components=2)
X2 = pca.fit_transform(X)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 按活动着色
activities_unique = np.unique(y_activity)
colors = plt.cm.tab10(np.linspace(0, 1, len(activities_unique)))
for a, c in zip(activities_unique, colors):
    mask = y_activity == a
    name = ACTIVITY_MAP.get(a, a)
    axes[0].scatter(X2[mask, 0], X2[mask, 1], c=[c], label=name, alpha=0.4, s=15)
axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
axes[0].set_title("PCA by Activity")
axes[0].legend(fontsize=7, ncol=2)

# 按标签着色
for lab, name, c in [("y", "Pass", "steelblue"), ("n", "Fail", "salmon"), ("i", "Intermediate", "gray")]:
    mask = y_label == lab
    if mask.any():
        axes[1].scatter(X2[mask, 0], X2[mask, 1], c=c, label=name, alpha=0.4, s=15)
axes[1].set_xlabel(f"PC1")
axes[1].set_ylabel(f"PC2")
axes[1].set_title("PCA by Label")
axes[1].legend()

fig.tight_layout()
fig.savefig(RESULTS / "real_embedding.png", dpi=150)
print("Saved real_embedding.png")

# ── 7. 贝叶斯变点 ───────────────────────────────────────────
print("\nRunning Bayesian changepoint on a sample...")
from src.models.bayes_changepoint import single_changepoint
from scipy.ndimage import uniform_filter1d

# 选一个较长的样本
longest = max(processed, key=lambda p: p["cleaned"].shape[0])
kp = longest["cleaned"]
vel = np.linalg.norm(np.diff(kp[:, 4, :2], axis=0), axis=-1) * 30
vel_smooth = uniform_filter1d(vel, 15)

idata = single_changepoint(vel_smooth, draws=1000, tune=1000, seed=42)
tau_samples = idata.posterior["tau"].values.flatten()
tau_median = np.median(tau_samples)

fig, axes = plt.subplots(2, 1, figsize=(14, 6))
t = np.arange(len(vel_smooth)) / 30.0
axes[0].plot(t, vel_smooth, color="steelblue", lw=0.8)
axes[0].axvline(tau_median / 30, color="red", ls="--", alpha=0.8,
                label=f"τ posterior median: {tau_median/30:.1f}s")
axes[0].fill_betweenx(
    [vel_smooth.min(), vel_smooth.max()],
    np.percentile(tau_samples, 5) / 30,
    np.percentile(tau_samples, 95) / 30,
    alpha=0.15, color="red", label="90% CI"
)
axes[0].set_ylabel("RWrist Speed")
axes[0].set_title(f"Bayesian Changepoint — {longest['sample'].path.name}")
axes[0].legend()

axes[1].hist(tau_samples / 30, bins=60, density=True, color="coral", alpha=0.7)
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Posterior density")
fig.tight_layout()
fig.savefig(RESULTS / "real_bayesian_changepoint.png", dpi=150)
print(f"Changepoint τ: {tau_median/30:.2f}s (90% CI: [{np.percentile(tau_samples,5)/30:.2f}, {np.percentile(tau_samples,95)/30:.2f}])")

# ── 8. 统计汇总 ─────────────────────────────────────────────
summary = {
    "Total samples": len(samples),
    "Loaded samples": len(processed),
    "Total windows": X.shape[0],
    "Feature dim": X.shape[1],
    "Activities": len(activities_unique),
    "Children": df_meta["child"].nunique(),
    "Mean confidence": f"{df_meta['mean_conf'].mean():.3f}",
    "Mean missing rate": f"{df_meta['missing_rate'].mean():.1%}",
}
print("\n" + "=" * 50)
print("Dataset Summary")
print("=" * 50)
for k, v in summary.items():
    print(f"  {k:25s}: {v}")

print("\n✅ Real data pipeline complete.")
