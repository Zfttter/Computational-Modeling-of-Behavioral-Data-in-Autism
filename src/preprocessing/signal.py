"""骨架时间序列预处理:缺失插值、滤波、去噪、归一化。

输入形状约定: (T, J, C)  T=帧数, J=关节数, C=2 或 3
低置信度/缺失关节用 NaN 表示。
"""
import numpy as np
from scipy.signal import butter, filtfilt, medfilt


def mask_low_confidence(kp, conf, thr=0.3):
    """conf: (T, J);低于阈值置 NaN。"""
    kp = kp.copy()
    kp[conf < thr] = np.nan
    return kp


def interpolate_missing(kp, max_gap=15):
    """对每个关节/坐标线性插值;长度超过 max_gap 的缺口保持 NaN。"""
    out = kp.copy()
    T, J, C = kp.shape
    t = np.arange(T)
    for j in range(J):
        for c in range(C):
            y = out[:, j, c]
            bad = np.isnan(y)
            if bad.all() or not bad.any():
                continue
            y[bad] = np.interp(t[bad], t[~bad], y[~bad])
            # 还原过长缺口
            run = 0
            for i in range(T):
                run = run + 1 if bad[i] else 0
                if run > max_gap:
                    y[i - run + 1:i + 1] = np.nan
            out[:, j, c] = y
    return out


def despike(kp, k=5):
    """中值滤波去除脉冲噪声(k 为奇数)。"""
    out = kp.copy()
    for j in range(kp.shape[1]):
        for c in range(kp.shape[2]):
            out[:, j, c] = medfilt(kp[:, j, c], k)
    return out


def lowpass(kp, fs=30.0, cutoff=6.0, order=4):
    """零相位 Butterworth 低通。人体运动能量多在 <6Hz。"""
    b, a = butter(order, cutoff / (fs / 2), btype="low")
    out = kp.copy()
    for j in range(kp.shape[1]):
        for c in range(kp.shape[2]):
            out[:, j, c] = filtfilt(b, a, kp[:, j, c])
    return out


def normalize(kp, root=0, ref=(1, 8)):
    """以根关节为原点,按参考骨长(默认颈-髋)缩放,去掉相机位置/远近差异。"""
    kp = kp - kp[:, root:root + 1, :]
    scale = np.linalg.norm(kp[:, ref[0]] - kp[:, ref[1]], axis=-1).mean()
    return kp / (scale + 1e-8)


def pipeline(kp, conf=None, fs=30.0):
    if conf is not None:
        kp = mask_low_confidence(kp, conf)
    kp = interpolate_missing(kp)
    kp = despike(kp)
    kp = lowpass(kp, fs=fs)
    return normalize(kp)
