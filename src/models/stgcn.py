"""Spatial-Temporal Graph Convolutional Network (ST-GCN) 用于骨架行为分类。

轻量级实现，适合 CPU 训练 + 小规模数据集 (MMASD ~1693 samples)。

参考:
  - Yan et al., "Spatial Temporal Graph Convolutional Networks for
    Skeleton-Based Action Recognition", AAAI 2018
  - Chen et al., "Channel-wise Topology Refinement Graph Convolution
    for Skeleton-Based Action Recognition", ICCV 2021 (CTR-GCN)

设计选择:
  - 2 层 ST-GCN block (轻量，避免小数据集过拟合)
  - 时间卷积核 = 9 (覆盖 ~0.3s)
  - 空间邻接矩阵基于 BODY_25 骨骼连接
  - Global Average Pooling → FC 分类头
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


def build_adjacency_matrix(num_joints: int = 25) -> np.ndarray:
    """构建 BODY_25 的归一化邻接矩阵 (含自连接)。"""
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (1, 5), (5, 6), (6, 7),
        (1, 8), (8, 9), (9, 10), (10, 11),
        (8, 12), (12, 13), (13, 14),
        (0, 15), (0, 16), (15, 17), (16, 18),
        (14, 19), (19, 20), (14, 21),
        (11, 22), (22, 23), (11, 24),
    ]
    A = np.eye(num_joints, dtype=np.float32)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    # 度归一化: D^{-1/2} A D^{-1/2}
    D = np.diag(A.sum(axis=1) ** (-0.5))
    A_norm = D @ A @ D
    return A_norm


class STGCNBlock(nn.Module):
    """单个 ST-GCN block: 空间图卷积 + 时间卷积。"""

    def __init__(self, in_channels: int, out_channels: int,
                 A: torch.Tensor, kernel_size: int = 9, stride: int = 1):
        super().__init__()
        self.A = A  # (V, V)

        # 空间图卷积
        self.gcn = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn_gcn = nn.BatchNorm2d(out_channels)

        # 时间卷积
        padding = (kernel_size - 1) // 2
        self.tcn = nn.Conv2d(out_channels, out_channels,
                             kernel_size=(kernel_size, 1),
                             padding=(padding, 0), stride=(stride, 1))
        self.bn_tcn = nn.BatchNorm2d(out_channels)

        # 残差连接
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()

        self.dropout = nn.Dropout(0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, C, T, V)"""
        res = self.residual(x)

        # 空间: x @ A
        A = self.A.to(x.device)
        x = torch.einsum("nctv,vw->nctw", x, A)
        x = self.bn_gcn(self.gcn(x))
        x = F.relu(x)

        # 时间
        x = self.bn_tcn(self.tcn(x))
        x = self.dropout(x)

        return F.relu(x + res)


class STGCN(nn.Module):
    """轻量 ST-GCN 分类器。

    输入: (N, C_in, T, V) — C_in=2 (x,y) 或 3 (x,y,z), V=25 joints
    输出: (N, num_classes) logits
    """

    def __init__(self, in_channels: int = 2, num_classes: int = 2,
                 hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        A = torch.from_numpy(build_adjacency_matrix())

        self.bn_input = nn.BatchNorm1d(in_channels * 25)

        layers = []
        dims = [in_channels] + [hidden_dim] * num_layers
        for i in range(num_layers):
            stride = 2 if i == num_layers - 1 else 1
            layers.append(STGCNBlock(dims[i], dims[i + 1], A, stride=stride))
        self.layers = nn.ModuleList(layers)

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, C, T, V)"""
        N, C, T, V = x.shape

        # Input BN — reshape to (N*T, C*V) for batch norm
        x_flat = x.permute(0, 2, 1, 3).contiguous().reshape(N * T, C * V)
        x_flat = self.bn_input(x_flat)
        x_flat = torch.clamp(x_flat, -10, 10)
        x = x_flat.reshape(N, T, C, V).permute(0, 2, 1, 3)

        for layer in self.layers:
            x = layer(x)

        # Global average pooling over T and V
        x = x.mean(dim=[2, 3])  # (N, hidden_dim)
        return self.fc(x)


class SkeletonDataset(torch.utils.data.Dataset):
    """骨架序列数据集，支持固定长度裁剪/填充。"""

    def __init__(self, sequences, labels, target_len: int = 150):
        self.sequences = sequences
        self.labels = labels
        self.target_len = target_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx].copy()  # (T, J, C)
        label = self.labels[idx]

        # 清除 NaN/Inf
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)

        T, J, C = seq.shape

        # 裁剪或填充到 target_len
        if T >= self.target_len:
            start = np.random.randint(0, T - self.target_len + 1)
            seq = seq[start:start + self.target_len]
        else:
            pad = np.zeros((self.target_len - T, J, C), dtype=seq.dtype)
            seq = np.concatenate([seq, pad], axis=0)

        # (T, J, C) → (C, T, J)
        x = torch.from_numpy(seq).float().permute(2, 0, 1)
        return x, label


def train_stgcn(
    train_seqs, train_labels, val_seqs, val_labels,
    num_classes: int = 2,
    in_channels: int = 2,
    hidden_dim: int = 64,
    target_len: int = 150,
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-3,
    device: str = "cpu",
) -> dict:
    """训练 ST-GCN 并返回结果。"""
    train_ds = SkeletonDataset(train_seqs, train_labels, target_len)
    val_ds = SkeletonDataset(val_seqs, val_labels, target_len)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)

    model = STGCN(in_channels=in_channels, num_classes=num_classes,
                  hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        # Train
        model.train()
        total_loss, correct, total = 0, 0, 0
        for x, y in train_loader:
            x = x.to(device)
            y = torch.as_tensor(y, dtype=torch.long, device=device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)

        history["train_loss"].append(total_loss / total)
        history["train_acc"].append(correct / total)

        # Val
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y_t = torch.as_tensor(y, dtype=torch.long, device=device)
                logits = model(x)
                loss = criterion(logits, y_t)
                val_loss += loss.item() * len(y_t)
                preds = logits.argmax(1)
                val_correct += (preds == y_t).sum().item()
                val_total += len(y_t)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y)

        history["val_loss"].append(val_loss / val_total)
        history["val_acc"].append(val_correct / val_total)

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}: "
                  f"train_loss={history['train_loss'][-1]:.4f} "
                  f"train_acc={history['train_acc'][-1]:.1%} | "
                  f"val_loss={history['val_loss'][-1]:.4f} "
                  f"val_acc={history['val_acc'][-1]:.1%}")

    return {
        "model": model,
        "history": history,
        "val_preds": np.array(all_preds),
        "val_labels": np.array(all_labels),
        "num_params": sum(p.numel() for p in model.parameters()),
    }
