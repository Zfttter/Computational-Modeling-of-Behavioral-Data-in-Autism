"""合成骨架数据生成器，用于在没有真实数据时测试整个 pipeline。

生成两种模式:
1. "normal": 平稳运动（模拟典型发育 TD）
2. "repetitive": 叠加高频周期运动（模拟刻板行为 — arm flapping / rocking）
"""
import numpy as np

# 简化版 BODY_25 "静息姿态"（归一化坐标,朝前站立）
_REST_POSE = np.array([
    [0.0,  -0.50, 0],   # 0  Nose
    [0.0,  -0.35, 0],   # 1  Neck
    [-0.15, -0.35, 0],  # 2  RShoulder
    [-0.15, -0.15, 0],  # 3  RElbow
    [-0.15,  0.00, 0],  # 4  RWrist
    [0.15, -0.35, 0],   # 5  LShoulder
    [0.15, -0.15, 0],   # 6  LElbow
    [0.15,  0.00, 0],   # 7  LWrist
    [0.0,   0.00, 0],   # 8  MidHip
    [-0.10,  0.00, 0],  # 9  RHip
    [-0.10,  0.25, 0],  # 10 RKnee
    [-0.10,  0.50, 0],  # 11 RAnkle
    [0.10,  0.00, 0],   # 12 LHip
    [0.10,  0.25, 0],   # 13 LKnee
    [0.10,  0.50, 0],   # 14 LAnkle
    [-0.03, -0.53, 0],  # 15 REye
    [0.03, -0.53, 0],   # 16 LEye
    [-0.06, -0.50, 0],  # 17 REar
    [0.06, -0.50, 0],   # 18 LEar
    [0.12,  0.55, 0],   # 19 LBigToe
    [0.13,  0.55, 0],   # 20 LSmallToe
    [0.10,  0.53, 0],   # 21 LHeel
    [-0.12, 0.55, 0],   # 22 RBigToe
    [-0.13, 0.55, 0],   # 23 RSmallToe
    [-0.10, 0.53, 0],   # 24 RHeel
], dtype=np.float32)  # (25, 3)


def generate_sequence(T: int = 300, fs: float = 30.0,
                      mode: str = "normal", seed: int = 0) -> dict:
    """生成一段合成 2D 骨架序列。

    Returns dict with:
        kp:   (T, 25, 2)  xy 坐标
        conf: (T, 25)     置信度
        meta: dict        描述信息
    """
    rng = np.random.default_rng(seed)
    rest = _REST_POSE[:, :2].copy()  # 只取 xy
    J = 25

    # 基线随机漫步（慢速全身晃动）
    drift = np.cumsum(rng.normal(0, 0.001, (T, J, 2)), axis=0)
    kp = rest[None, :, :] + drift

    # 高斯噪声
    noise = rng.normal(0, 0.003, (T, J, 2))
    kp += noise

    if mode == "repetitive":
        # 在手腕和肘部叠加高频正弦（模拟 arm flapping）
        t = np.arange(T) / fs
        freq = 3.5 + rng.uniform(-0.5, 0.5)  # ~3-4 Hz
        amplitude = 0.05
        for j in [3, 4, 6, 7]:  # R/L elbow + wrist
            kp[:, j, 1] += amplitude * np.sin(2 * np.pi * freq * t)
            kp[:, j, 0] += amplitude * 0.3 * np.sin(2 * np.pi * freq * t + np.pi / 4)

    elif mode == "changepoint":
        # 前半段平静,后半段高活动（测试变点检测）
        cp = T // 2
        t2 = np.arange(cp, T) / fs
        freq = 3.0
        for j in [3, 4, 6, 7]:
            kp[cp:, j, 1] += 0.06 * np.sin(2 * np.pi * freq * t2)

    # 模拟遮挡 / 低置信度
    conf = np.ones((T, J), dtype=np.float32) * 0.9
    # 随机 5% 帧 × 关节 置信度降低
    mask = rng.random((T, J)) < 0.05
    conf[mask] = rng.uniform(0.0, 0.3, mask.sum())
    kp[mask] += rng.normal(0, 0.1, (mask.sum(), 2))  # 低置信度点坐标不准

    return {
        "kp": kp.astype(np.float32),
        "conf": conf,
        "meta": {"T": T, "fs": fs, "mode": mode, "seed": seed, "n_joints": J},
    }


def generate_dataset(n_normal: int = 20, n_repetitive: int = 10,
                     T: int = 300, seed: int = 42) -> dict:
    """生成一个小型合成数据集，返回 X, y, meta。"""
    samples = []
    labels = []
    for i in range(n_normal):
        s = generate_sequence(T, mode="normal", seed=seed + i)
        samples.append(s)
        labels.append(0)
    for i in range(n_repetitive):
        s = generate_sequence(T, mode="repetitive", seed=seed + 1000 + i)
        samples.append(s)
        labels.append(1)
    return {"samples": samples, "labels": np.array(labels)}
