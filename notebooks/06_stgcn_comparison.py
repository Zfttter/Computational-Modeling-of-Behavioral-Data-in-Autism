"""06 — ST-GCN vs 经典方法对比实验。

任务: 二分类 — pass (y) vs fail (n) 活动片段
对比:
  1. SVM + 滑窗特征 (经典 baseline)
  2. ST-GCN (深度学习)

输出:
  results/figures/stgcn_training.png          — 训练曲线
  results/figures/stgcn_confusion.png         — 混淆矩阵对比
  results/figures/stgcn_comparison_bar.png    — 方法对比柱状图
  results/tables/stgcn_comparison.csv         — 方法对比表
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

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from data.mmasd_loader import discover_samples, load_2d_sequence, select_target_person
from preprocessing.signal import pipeline as preprocess_pipeline
from preprocessing.windows import sliding_windows, window_features

FIG_DIR = ROOT / "results" / "figures"
TBL_DIR = ROOT / "results" / "tables"
MMASD_ROOT = ROOT / "data" / "raw" / "MMASD"

# ── 1. 加载数据 ──────────────────────────────────────────────
print("Discovering samples...")
all_samples = discover_samples(MMASD_ROOT, modality="2d")
print(f"  Total: {len(all_samples)}")

# 只用 pass (y) 和 fail (n), 跳过 intermediate (i)
binary_samples = [s for s in all_samples if s.label in ("y", "n")]
print(f"  Binary (pass/fail): {len(binary_samples)}")

# 加载骨架序列
print("\nLoading skeleton sequences (up to 300 samples)...")
sequences = []
labels = []
child_ids = []
loaded = 0

np.random.seed(42)
indices = np.random.permutation(len(binary_samples))

for idx in indices:
    if loaded >= 300:
        break
    s = binary_samples[idx]
    try:
        pid = select_target_person(s.path, n_sample=3)
        kp = load_2d_sequence(s.path, pid)
        kp_clean = preprocess_pipeline(kp)
        if kp_clean.shape[0] >= 30:
            xy = np.nan_to_num(kp_clean[:, :, :2], nan=0.0)
            sequences.append(xy)
            labels.append(0 if s.label == "y" else 1)  # 0=pass, 1=fail
            child_ids.append(s.child_id)
            loaded += 1
            if loaded % 50 == 0:
                print(f"  Loaded {loaded}...")
    except Exception:
        continue

labels = np.array(labels)
child_ids = np.array(child_ids)
print(f"\nLoaded {len(sequences)} sequences")
print(f"  Pass: {(labels == 0).sum()}, Fail: {(labels == 1).sum()}")
print(f"  Sequence lengths: {min(s.shape[0] for s in sequences)} - {max(s.shape[0] for s in sequences)} frames")

# ── 2. 按儿童划分训练/验证集 (防止数据泄露) ─────────────────
unique_children = np.unique(child_ids)
np.random.shuffle(unique_children)
n_train_children = int(len(unique_children) * 0.7)
train_children = set(unique_children[:n_train_children])
val_children = set(unique_children[n_train_children:])

train_mask = np.array([c in train_children for c in child_ids])
val_mask = ~train_mask

train_seqs = [sequences[i] for i in range(len(sequences)) if train_mask[i]]
train_labels = labels[train_mask]
val_seqs = [sequences[i] for i in range(len(sequences)) if val_mask[i]]
val_labels = labels[val_mask]

print(f"\nChild-wise split:")
print(f"  Train: {len(train_seqs)} samples from {len(train_children)} children")
print(f"  Val: {len(val_seqs)} samples from {len(val_children)} children")
print(f"  Train labels: pass={sum(train_labels==0)}, fail={sum(train_labels==1)}")
print(f"  Val labels: pass={sum(val_labels==0)}, fail={sum(val_labels==1)}")

# ── 3. Baseline: SVM + 滑窗特征 ─────────────────────────────
print("\n" + "=" * 50)
print("Method 1: SVM + Hand-crafted Features")
print("=" * 50)

def extract_clip_features(seq):
    """从整个片段提取一个特征向量 (聚合所有窗口)。"""
    wins = sliding_windows(seq, win=60, stride=30)
    feats = []
    for w in wins:
        try:
            f = window_features(w)
            if np.isfinite(f).all():
                feats.append(f)
        except Exception:
            continue
    if not feats:
        return None
    return np.mean(feats, axis=0)  # 窗口平均

X_train_svm, y_train_svm = [], []
for seq, label in zip(train_seqs, train_labels):
    f = extract_clip_features(seq)
    if f is not None:
        X_train_svm.append(f)
        y_train_svm.append(label)

X_val_svm, y_val_svm = [], []
for seq, label in zip(val_seqs, val_labels):
    f = extract_clip_features(seq)
    if f is not None:
        X_val_svm.append(f)
        y_val_svm.append(label)

X_train_svm = np.array(X_train_svm)
y_train_svm = np.array(y_train_svm)
X_val_svm = np.array(X_val_svm)
y_val_svm = np.array(y_val_svm)

print(f"  Feature extraction: train={len(X_train_svm)}, val={len(X_val_svm)}")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train_svm)
X_val_s = scaler.transform(X_val_svm)

svm = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")
svm.fit(X_train_s, y_train_svm)
svm_preds = svm.predict(X_val_s)
svm_acc = accuracy_score(y_val_svm, svm_preds)
svm_f1 = f1_score(y_val_svm, svm_preds, average="weighted")
svm_cm = confusion_matrix(y_val_svm, svm_preds)

print(f"\n  SVM Results:")
print(f"    Accuracy: {svm_acc:.3f}")
print(f"    Weighted F1: {svm_f1:.3f}")
print(classification_report(y_val_svm, svm_preds, target_names=["Pass", "Fail"]))

# ── 4. ST-GCN ────────────────────────────────────────────────
print("=" * 50)
print("Method 2: ST-GCN (Spatial-Temporal Graph ConvNet)")
print("=" * 50)

import torch
from models.stgcn import train_stgcn

# 确定统一长度
seq_lens = [s.shape[0] for s in sequences]
target_len = int(np.percentile(seq_lens, 50))
target_len = max(60, min(target_len, 300))
print(f"  Target sequence length: {target_len} frames")
print(f"  Model: 2-layer ST-GCN, hidden_dim=64")

result = train_stgcn(
    train_seqs, train_labels.tolist(),
    val_seqs, val_labels.tolist(),
    num_classes=2,
    in_channels=2,
    hidden_dim=64,
    target_len=target_len,
    epochs=80,
    batch_size=16,
    lr=5e-4,
    device="cpu",
)

stgcn_preds = result["val_preds"]
stgcn_labels = result["val_labels"]
stgcn_acc = accuracy_score(stgcn_labels, stgcn_preds)
stgcn_f1 = f1_score(stgcn_labels, stgcn_preds, average="weighted")
stgcn_cm = confusion_matrix(stgcn_labels, stgcn_preds)

print(f"\n  ST-GCN Results:")
print(f"    Accuracy: {stgcn_acc:.3f}")
print(f"    Weighted F1: {stgcn_f1:.3f}")
print(f"    Parameters: {result['num_params']:,}")
print(classification_report(stgcn_labels, stgcn_preds, target_names=["Pass", "Fail"]))

# ── 5. 图1: 训练曲线 ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

h = result["history"]
epochs_range = range(1, len(h["train_loss"]) + 1)

axes[0].plot(epochs_range, h["train_loss"], label="Train", color="#2196F3")
axes[0].plot(epochs_range, h["val_loss"], label="Val", color="#F44336")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("ST-GCN Training Loss")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(epochs_range, h["train_acc"], label="Train", color="#2196F3")
axes[1].plot(epochs_range, h["val_acc"], label="Val", color="#F44336")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("ST-GCN Accuracy")
axes[1].legend()
axes[1].grid(alpha=0.3)

fig.tight_layout()
fig.savefig(FIG_DIR / "stgcn_training.png", bbox_inches="tight")
plt.close()
print("\nSaved: stgcn_training.png")

# ── 6. 图2: 混淆矩阵对比 ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
class_names = ["Pass", "Fail"]

for ax, cm, title in [
    (axes[0], svm_cm, f"SVM (Acc={svm_acc:.1%}, F1={svm_f1:.3f})"),
    (axes[1], stgcn_cm, f"ST-GCN (Acc={stgcn_acc:.1%}, F1={stgcn_f1:.3f})"),
]:
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=14, color=color)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

fig.suptitle("Confusion Matrix: SVM vs ST-GCN (Pass/Fail Classification)", fontsize=12)
fig.tight_layout()
fig.savefig(FIG_DIR / "stgcn_confusion.png", bbox_inches="tight")
plt.close()
print("Saved: stgcn_confusion.png")

# ── 7. 图3: 方法对比柱状图 ──────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

methods = ["SVM\n(hand-crafted features)", "ST-GCN\n(learned features)"]
accs = [svm_acc, stgcn_acc]
f1s = [svm_f1, stgcn_f1]

x = np.arange(len(methods))
width = 0.3

bars1 = ax.bar(x - width / 2, accs, width, label="Accuracy", color="#2196F3", alpha=0.8)
bars2 = ax.bar(x + width / 2, f1s, width, label="Weighted F1", color="#4CAF50", alpha=0.8)

for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}",
                ha="center", va="bottom", fontsize=10)

ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.set_ylabel("Score")
ax.set_ylim(0, 1.15)
ax.set_title("Pass/Fail Classification: Classical vs Deep Learning")
ax.legend()
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig(FIG_DIR / "stgcn_comparison_bar.png", bbox_inches="tight")
plt.close()
print("Saved: stgcn_comparison_bar.png")

# ── 8. 结果表 ────────────────────────────────────────────────
comparison = pd.DataFrame([
    {
        "Method": "SVM + Hand-crafted Features",
        "Type": "Classical ML",
        "Input": "100-dim window features",
        "Accuracy": f"{svm_acc:.3f}",
        "Weighted_F1": f"{svm_f1:.3f}",
        "Parameters": "~100 (SVM support vectors)",
        "Interpretability": "High",
        "Training_Time": "< 1s",
    },
    {
        "Method": "ST-GCN (2-layer)",
        "Type": "Deep Learning",
        "Input": "Raw skeleton (T×25×2)",
        "Accuracy": f"{stgcn_acc:.3f}",
        "Weighted_F1": f"{stgcn_f1:.3f}",
        "Parameters": f"{result['num_params']:,}",
        "Interpretability": "Low",
        "Training_Time": f"~80 epochs (CPU)",
    },
])
comparison.to_csv(TBL_DIR / "stgcn_comparison.csv", index=False)
print("\nSaved: stgcn_comparison.csv")

# ── 9. 汇总 ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ST-GCN vs CLASSICAL METHOD COMPARISON")
print("=" * 60)
print(f"\nTask: Binary classification (Pass vs Fail)")
print(f"Data: {len(sequences)} MMASD clips, child-wise train/val split")
print(f"\n{'Method':<35} {'Accuracy':>10} {'F1':>10}")
print("-" * 55)
print(f"{'SVM + Hand-crafted Features':<35} {svm_acc:>10.3f} {svm_f1:>10.3f}")
print(f"{'ST-GCN (2-layer, 64-dim)':<35} {stgcn_acc:>10.3f} {stgcn_f1:>10.3f}")

print(f"\nST-GCN parameters: {result['num_params']:,}")
print(f"\nTrade-off analysis:")
if stgcn_acc > svm_acc:
    print(f"  ST-GCN outperforms SVM by {(stgcn_acc - svm_acc)*100:.1f}pp accuracy")
    print(f"  But: more parameters, less interpretable, longer training")
elif svm_acc > stgcn_acc:
    print(f"  SVM outperforms ST-GCN by {(svm_acc - stgcn_acc)*100:.1f}pp accuracy")
    print(f"  Classical features are competitive on this small dataset")
    print(f"  ST-GCN may benefit from more data / pretraining")
else:
    print(f"  Methods are comparable in accuracy")

print(f"\nFor clinical deployment:")
print(f"  → SVM preferred for interpretability + small data")
print(f"  → ST-GCN preferred when more data available + accuracy critical")
