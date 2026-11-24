"""ADOS 临床评分与计算运动特征的关联分析。

核心研究问题：从骨架数据中提取的运动特征能否预测 ADOS 临床评分？
这直接检验计算方法作为临床评估辅助工具的潜力。

ADOS-2 子项:
  - Social Affect (SA): 社会情感，0-28
  - Restricted & Repetitive Behavior (RRB): 限制性重复行为，0-16
  - Overall Total: SA + RRB
  - Comparison Score: 1-10 严重度标准化分
"""

import pathlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ChildProfile:
    """单个儿童的运动特征 + 临床评分。"""
    child_id: str
    n_samples: int
    # 运动特征 (从所有片段聚合)
    mean_speed: float
    std_speed: float
    mean_accel: float
    dominant_freq: float
    movement_variability: float  # 跨片段速度标准差
    high_freq_power: float       # 2-6 Hz 频段功率比（刻板行为频段）
    # ADOS 评分
    sa_total: Optional[float] = None
    rrb_total: Optional[float] = None
    overall_total: Optional[float] = None
    comparison_score: Optional[float] = None


def load_ados_scores(xlsx_path: pathlib.Path) -> pd.DataFrame:
    """加载并清洗 ADOS 评分表。"""
    df = pd.read_excel(xlsx_path, engine="openpyxl")

    clean = pd.DataFrame()
    clean["child_id"] = df["ID#"].astype(str).str.strip()
    clean["gender"] = df["Gender"]
    clean["age_raw"] = df["Chronological Age"]
    clean["module"] = df["Module"]
    clean["sa_total"] = pd.to_numeric(df["Social Affect Total "], errors="coerce")
    clean["rrb_total"] = pd.to_numeric(
        df["Restricted and Repetitive Behavior (RRB) Total"], errors="coerce"
    )
    clean["overall_total"] = pd.to_numeric(df["Overall Total "], errors="coerce")

    score_col = [c for c in df.columns if "Comparison Score" in c][0]
    clean["comparison_score"] = pd.to_numeric(df[score_col], errors="coerce")

    severity_col = [c for c in df.columns if "Severity" in c][0]
    clean["severity"] = df[severity_col]

    # 解析年龄为月数
    clean["age_months"] = clean["age_raw"].apply(_parse_age)

    clean = clean.dropna(subset=["child_id"])
    clean = clean[clean["child_id"] != "nan"]
    return clean


def _parse_age(raw) -> Optional[float]:
    """解析 '6Y,9M' / '9y' 格式为月数。"""
    if pd.isna(raw):
        return None
    s = str(raw).upper().replace(" ", "")
    years, months = 0, 0
    if "Y" in s:
        parts = s.split("Y")
        try:
            years = int(parts[0])
        except ValueError:
            return None
        rest = parts[1].replace(",", "").replace("M", "")
        if rest:
            try:
                months = int(rest)
            except ValueError:
                pass
    return years * 12 + months


def compute_child_features(
    child_id: str,
    sequences: List[np.ndarray],
    fs: float = 30.0,
) -> Dict[str, float]:
    """从一个儿童的所有骨架序列中提取聚合运动特征。

    Args:
        child_id: 儿童 ID
        sequences: list of (T, J, C) 预处理后的骨架序列
        fs: 采样频率

    Returns:
        dict of aggregated features
    """
    all_speeds = []
    all_accels = []
    all_dom_freqs = []
    clip_mean_speeds = []
    all_hf_powers = []

    for seq in sequences:
        if seq.shape[0] < 10:
            continue
        coords = seq[:, :, :2]  # (T, J, 2)

        # 速度: 帧间位移
        vel = np.diff(coords, axis=0) * fs  # (T-1, J, 2)
        speed = np.linalg.norm(vel, axis=2)  # (T-1, J)
        mean_speed_per_joint = speed.mean(axis=0)  # (J,)
        all_speeds.append(mean_speed_per_joint.mean())
        clip_mean_speeds.append(mean_speed_per_joint.mean())

        # 加速度
        accel = np.diff(vel, axis=0) * fs
        accel_mag = np.linalg.norm(accel, axis=2)
        all_accels.append(accel_mag.mean())

        # FFT 主频 (手腕关节, 刻板行为相关)
        for ji in [4, 7]:  # R/L Wrist
            y = coords[:, ji, 1]
            if np.isnan(y).sum() > len(y) * 0.5:
                continue
            y_clean = y.copy()
            nans = np.isnan(y_clean)
            if nans.any():
                y_clean[nans] = np.nanmean(y_clean)
            y_detrend = y_clean - np.convolve(
                y_clean, np.ones(min(30, len(y_clean))) / min(30, len(y_clean)),
                mode="same",
            )
            spec = np.abs(np.fft.rfft(y_detrend))
            freqs = np.fft.rfftfreq(len(y_detrend), 1 / fs)

            if len(spec) > 1:
                dom_idx = np.argmax(spec[1:]) + 1
                all_dom_freqs.append(freqs[dom_idx])

                # 2-6 Hz 功率比 (刻板行为频段)
                hf_mask = (freqs >= 2.0) & (freqs <= 6.0)
                total_power = (spec**2).sum()
                if total_power > 0:
                    hf_power = (spec[hf_mask] ** 2).sum() / total_power
                    all_hf_powers.append(hf_power)

    if not all_speeds:
        return {}

    return {
        "child_id": child_id,
        "n_samples": len(sequences),
        "mean_speed": float(np.mean(all_speeds)),
        "std_speed": float(np.std(all_speeds)) if len(all_speeds) > 1 else 0.0,
        "mean_accel": float(np.mean(all_accels)),
        "dominant_freq": float(np.median(all_dom_freqs)) if all_dom_freqs else 0.0,
        "movement_variability": float(np.std(clip_mean_speeds))
        if len(clip_mean_speeds) > 1
        else 0.0,
        "high_freq_power": float(np.mean(all_hf_powers)) if all_hf_powers else 0.0,
    }


def correlate_features_with_ados(
    profiles: List[ChildProfile],
) -> pd.DataFrame:
    """计算运动特征与 ADOS 子项的 Spearman 相关。"""
    feature_names = [
        "mean_speed", "std_speed", "mean_accel",
        "dominant_freq", "movement_variability", "high_freq_power",
    ]
    ados_names = ["sa_total", "rrb_total", "overall_total", "comparison_score"]

    rows = []
    for feat in feature_names:
        for ados in ados_names:
            x = [getattr(p, feat) for p in profiles if getattr(p, ados) is not None]
            y = [getattr(p, ados) for p in profiles if getattr(p, ados) is not None]

            if len(x) < 5:
                continue

            x, y = np.array(x), np.array(y)
            valid = ~(np.isnan(x) | np.isnan(y))
            if valid.sum() < 5:
                continue

            rho, pval = stats.spearmanr(x[valid], y[valid])
            rows.append({
                "feature": feat,
                "ados_subscale": ados,
                "spearman_rho": rho,
                "p_value": pval,
                "n": int(valid.sum()),
                "significant": pval < 0.05,
            })

    return pd.DataFrame(rows)


def regression_analysis(
    profiles: List[ChildProfile],
    target: str = "comparison_score",
) -> Dict:
    """多元回归：运动特征 → ADOS 评分。"""
    feature_names = [
        "mean_speed", "std_speed", "mean_accel",
        "dominant_freq", "movement_variability", "high_freq_power",
    ]

    X, y = [], []
    for p in profiles:
        t = getattr(p, target)
        if t is None or np.isnan(t):
            continue
        feats = [getattr(p, f) for f in feature_names]
        if any(np.isnan(f) for f in feats):
            continue
        X.append(feats)
        y.append(t)

    if len(X) < 8:
        return {"error": "Too few samples", "n": len(X)}

    X = np.array(X)
    y = np.array(y)

    # 标准化
    X_mean, X_std = X.mean(0), X.std(0) + 1e-10
    X_norm = (X - X_mean) / X_std

    # 添加截距
    X_aug = np.column_stack([np.ones(len(X_norm)), X_norm])

    # OLS
    beta, residuals, rank, sv = np.linalg.lstsq(X_aug, y, rcond=None)

    y_pred = X_aug @ beta
    ss_res = ((y - y_pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # 调整 R²
    n, p = len(y), len(feature_names)
    adj_r2 = 1 - (1 - r_squared) * (n - 1) / (n - p - 1) if n > p + 1 else r_squared

    # 留一法交叉验证 R²
    loo_errors = []
    for i in range(len(X_norm)):
        X_loo = np.delete(X_aug, i, axis=0)
        y_loo = np.delete(y, i)
        b_loo, _, _, _ = np.linalg.lstsq(X_loo, y_loo, rcond=None)
        pred_i = X_aug[i] @ b_loo
        loo_errors.append((y[i] - pred_i) ** 2)
    loo_r2 = 1 - np.mean(loo_errors) / y.var()

    return {
        "target": target,
        "n_children": n,
        "r_squared": r_squared,
        "adj_r_squared": adj_r2,
        "loo_r_squared": loo_r2,
        "coefficients": dict(zip(["intercept"] + feature_names, beta.tolist())),
        "y_true": y.tolist(),
        "y_pred": y_pred.tolist(),
    }
