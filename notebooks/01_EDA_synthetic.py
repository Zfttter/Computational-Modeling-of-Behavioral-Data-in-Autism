"""01_EDA_synthetic.py — 探索性数据分析（合成数据版）
==============================================

用 `python -I notebooks/01_EDA_synthetic.py` 运行,
或在 Jupyter 中逐 cell 执行（`# %%` 分隔）。

目的: 在等待真实数据的同时,验证全部预处理 + 建模 pipeline。
"""
# %% [markdown]
# # 1. 生成合成数据

# %%
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无 GUI 后端
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.synthetic import generate_sequence, generate_dataset
from src.data.mmasd_loader import BODY_25_JOINTS, BODY_25_SKELETON
from src.preprocessing.signal import pipeline as preprocess_pipeline
from src.preprocessing.windows import sliding_windows, window_features

RESULTS = pathlib.Path("results/figures")
RESULTS.mkdir(parents=True, exist_ok=True)

# %% 生成三种模式
normal = generate_sequence(300, mode="normal", seed=0)
repetitive = generate_sequence(300, mode="repetitive", seed=1)
changepoint = generate_sequence(300, mode="changepoint", seed=2)

print("Normal shape:", normal["kp"].shape)
print("Repetitive shape:", repetitive["kp"].shape)

# %% [markdown]
# # 2. 可视化原始骨架

# %%
def plot_skeleton_frame(kp_frame, conf_frame=None, ax=None, title=""):
    """画一帧骨架。kp_frame: (25, 2), conf_frame: (25,)"""
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 5))
    for i, j in BODY_25_SKELETON:
        if conf_frame is not None and (conf_frame[i] < 0.3 or conf_frame[j] < 0.3):
            continue
        ax.plot([kp_frame[i, 0], kp_frame[j, 0]],
                [kp_frame[i, 1], kp_frame[j, 1]], "b-", lw=1.5)
    colors = conf_frame if conf_frame is not None else "steelblue"
    sc = ax.scatter(kp_frame[:, 0], kp_frame[:, 1], c=colors,
                    cmap="RdYlGn", vmin=0, vmax=1, s=30, zorder=5)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    return sc


fig, axes = plt.subplots(1, 3, figsize=(12, 5))
for ax, (data, name) in zip(axes, [
    (normal, "Normal"), (repetitive, "Repetitive"), (changepoint, "Changepoint")
]):
    t = data["kp"].shape[0] // 2
    plot_skeleton_frame(data["kp"][t], data["conf"][t], ax, f"{name} (frame {t})")
fig.tight_layout()
fig.savefig(RESULTS / "skeleton_frames.png", dpi=150)
print("Saved skeleton_frames.png")

# %% [markdown]
# # 3. 关节轨迹对比

# %%
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
t = np.arange(300) / 30.0  # 秒

for ax, joint_name, ji in [(axes[0], "RWrist", 4), (axes[1], "LWrist", 7)]:
    ax.plot(t, normal["kp"][:, ji, 1], label="Normal", alpha=0.8)
    ax.plot(t, repetitive["kp"][:, ji, 1], label="Repetitive", alpha=0.8)
    ax.plot(t, changepoint["kp"][:, ji, 1], label="Changepoint", alpha=0.8)
    ax.set_ylabel(f"{joint_name} Y")
    ax.legend(fontsize=8)

axes[1].set_xlabel("Time (s)")
fig.suptitle("Raw Joint Trajectories — Y coordinate", y=1.02)
fig.tight_layout()
fig.savefig(RESULTS / "joint_trajectories.png", dpi=150)
print("Saved joint_trajectories.png")

# %% [markdown]
# # 4. 预处理前后对比

# %%
raw = repetitive["kp"]
conf = repetitive["conf"]

# 添加第三维 (z=0) 以匹配 pipeline 的 (T,J,C=3) 要求
kp3 = np.concatenate([raw, np.zeros_like(raw[:, :, :1])], axis=-1)
conf_full = conf

cleaned = preprocess_pipeline(kp3, conf_full, fs=30.0)

fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
ji = 4  # RWrist
axes[0].plot(raw[:, ji, 1], label="Raw", alpha=0.7)
axes[0].plot(cleaned[:, ji, 1], label="Cleaned", alpha=0.9)
axes[0].set_ylabel("RWrist Y")
axes[0].legend()
axes[0].set_title("Preprocessing: Raw vs Cleaned (Repetitive)")

axes[1].plot(conf[:, ji], color="orange")
axes[1].set_ylabel("Confidence")
axes[1].set_xlabel("Frame")
fig.tight_layout()
fig.savefig(RESULTS / "preprocessing_comparison.png", dpi=150)
print("Saved preprocessing_comparison.png")

# %% [markdown]
# # 5. 滑窗特征与频谱

# %%
from src.preprocessing.windows import sliding_windows, window_features

windows_norm = sliding_windows(
    np.concatenate([normal["kp"], np.zeros_like(normal["kp"][:, :, :1])], axis=-1),
    win=60, stride=30
)
windows_rep = sliding_windows(
    np.concatenate([repetitive["kp"], np.zeros_like(repetitive["kp"][:, :, :1])], axis=-1),
    win=60, stride=30
)

feats_norm = np.array([window_features(w) for w in windows_norm])
feats_rep = np.array([window_features(w) for w in windows_rep])

print(f"Feature vector dim: {feats_norm.shape[1]}")
print(f"Normal windows: {feats_norm.shape[0]}, Repetitive windows: {feats_rep.shape[0]}")

# %% 频谱对比
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, (data, name) in zip(axes, [(normal, "Normal"), (repetitive, "Repetitive")]):
    ji = 4
    y = data["kp"][:, ji, 1]
    freqs = np.fft.rfftfreq(len(y), 1 / 30.0)
    spec = np.abs(np.fft.rfft(y - y.mean()))
    ax.plot(freqs, spec)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"{name} — RWrist Y Spectrum")
    ax.set_xlim(0, 10)
fig.tight_layout()
fig.savefig(RESULTS / "frequency_spectrum.png", dpi=150)
print("Saved frequency_spectrum.png")

# %% [markdown]
# # 6. One-class SVM

# %%
from src.models.oneclass import fit_oneclass, score

# 用 normal 训练,看 repetitive 是否被检测为异常
model = fit_oneclass(feats_norm, nu=0.1)
scores_norm = score(model, feats_norm)
scores_rep = score(model, feats_rep)

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(scores_norm, bins=15, alpha=0.6, label="Normal", color="steelblue")
ax.hist(scores_rep, bins=15, alpha=0.6, label="Repetitive", color="salmon")
ax.axvline(0, color="k", ls="--", label="Decision boundary")
ax.set_xlabel("Decision function score")
ax.set_ylabel("Count")
ax.set_title("One-class SVM: Normal vs Repetitive Windows")
ax.legend()
fig.tight_layout()
fig.savefig(RESULTS / "oneclass_svm.png", dpi=150)
print("Saved oneclass_svm.png")

anomaly_rate_rep = (scores_rep < 0).mean()
anomaly_rate_norm = (scores_norm < 0).mean()
print(f"Anomaly rate — Normal: {anomaly_rate_norm:.1%}, Repetitive: {anomaly_rate_rep:.1%}")

# %% [markdown]
# # 7. 贝叶斯变点检测

# %%
print("Running Bayesian changepoint detection...")
from src.models.bayes_changepoint import single_changepoint

# 用 changepoint 数据:速度信号
cp_data = changepoint["kp"]
vel = np.linalg.norm(np.diff(cp_data[:, 4, :], axis=0), axis=-1) * 30  # RWrist speed
# 平滑一下
from scipy.ndimage import uniform_filter1d
vel_smooth = uniform_filter1d(vel, 10)

idata = single_changepoint(vel_smooth, draws=500, tune=500, seed=42)

import arviz as az
tau_samples = idata.posterior["tau"].values.flatten()

fig, axes = plt.subplots(2, 1, figsize=(12, 6))
t = np.arange(len(vel_smooth)) / 30.0

axes[0].plot(t, vel_smooth, color="steelblue")
axes[0].axvline(150 / 30, color="red", ls="--", label="True changepoint")
axes[0].set_ylabel("RWrist Speed (px/s)")
axes[0].set_title("Bayesian Changepoint Detection")
axes[0].legend()

axes[1].hist(tau_samples / 30, bins=50, density=True, color="coral", alpha=0.7)
axes[1].axvline(150 / 30, color="red", ls="--", label="True")
axes[1].axvline(np.median(tau_samples) / 30, color="blue", ls="--",
                label=f"Posterior median: {np.median(tau_samples)/30:.1f}s")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Posterior density")
axes[1].set_title("Changepoint τ Posterior")
axes[1].legend()

fig.tight_layout()
fig.savefig(RESULTS / "bayesian_changepoint.png", dpi=150)
print("Saved bayesian_changepoint.png")
print(f"True changepoint: 5.0s | Posterior median: {np.median(tau_samples)/30:.2f}s")

# %% [markdown]
# # 8. 多样本 one-class SVM 数据集评估

# %%
dataset = generate_dataset(n_normal=20, n_repetitive=10, T=300, seed=42)

def extract_features(sample):
    kp = sample["kp"]
    kp3 = np.concatenate([kp, np.zeros_like(kp[:, :, :1])], axis=-1)
    wins = sliding_windows(kp3, win=60, stride=30)
    return np.array([window_features(w) for w in wins])

all_feats = []
all_labels = []
for s, lab in zip(dataset["samples"], dataset["labels"]):
    f = extract_features(s)
    all_feats.append(f)
    all_labels.extend([lab] * len(f))

X = np.vstack(all_feats)
y = np.array(all_labels)

# 只用 normal (y==0) 训练
X_train = X[y == 0]
model2 = fit_oneclass(X_train, nu=0.05)
preds = model2.predict(X)  # +1 normal, -1 anomaly

from sklearn.metrics import classification_report
# 把 one-class SVM 的 -1 映射到 "repetitive (1)"
pred_labels = (preds == -1).astype(int)
print("\nClassification Report (One-class SVM on synthetic dataset):")
print(classification_report(y, pred_labels, target_names=["Normal", "Repetitive"]))

# %% [markdown]
# # 9. 特征 2D 嵌入

# %%
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X2 = pca.fit_transform(X)

fig, ax = plt.subplots(figsize=(8, 6))
for lab, name, color in [(0, "Normal", "steelblue"), (1, "Repetitive", "salmon")]:
    mask = y == lab
    ax.scatter(X2[mask, 0], X2[mask, 1], c=color, label=name, alpha=0.5, s=20)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
ax.set_title("Window Feature Embedding (PCA)")
ax.legend()
fig.tight_layout()
fig.savefig(RESULTS / "feature_embedding.png", dpi=150)
print("Saved feature_embedding.png")

# %% [markdown]
# # 总结
#
# 在合成数据上,整个 pipeline 工作正常:
# - 预处理成功降噪 + 插值缺失
# - 频谱分析可清晰区分 normal vs repetitive (~3.5Hz peak)
# - One-class SVM 仅用正常数据训练,可检出重复运动
# - 贝叶斯变点检测能恢复真实变点位置,并给出后验不确定性
# - PCA 嵌入显示两类可分
#
# 下一步: 用真实 MMASD 数据替换 synthetic 生成器。

print("\n✅ EDA complete. All figures saved to results/figures/")
