"""滑窗 + 运动特征(速度/加速度/周期性),供 one-class SVM 与贝叶斯模型使用。"""
import numpy as np


def sliding_windows(kp, win=60, stride=30):
    T = kp.shape[0]
    return [kp[s:s + win] for s in range(0, T - win + 1, stride)]


def window_features(w, fs=30.0):
    """w: (win, J, C) → 1D 特征向量。"""
    vel = np.diff(w, axis=0) * fs
    acc = np.diff(vel, axis=0) * fs
    speed = np.linalg.norm(vel, axis=-1)            # (win-1, J)
    # 主频(检测拍手/摇摆类刻板运动的周期性)
    spec = np.abs(np.fft.rfft(speed - speed.mean(0), axis=0))
    freqs = np.fft.rfftfreq(speed.shape[0], 1 / fs)
    dom = freqs[spec.argmax(0)]
    return np.concatenate([
        speed.mean(0), speed.std(0),
        np.linalg.norm(acc, axis=-1).mean(0),
        dom,
    ])
