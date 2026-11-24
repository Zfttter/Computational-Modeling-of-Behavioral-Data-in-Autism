"""Conformal Prediction 校准异常检测。

提供有限样本统计保证的异常检测:
  - 给定显著性水平 α，保证假阳性率 ≤ α
  - 输出 p-value 而非二分类，支持临床决策中的不确定性量化
  - 基于 nonconformity scores 的 split conformal 方法

参考: Angelopoulos & Bates, "Conformal Prediction: A Gentle Introduction" (2023)
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


@dataclass
class ConformalResult:
    """Conformal prediction 结果。"""
    p_values: np.ndarray        # 每个测试点的 p-value
    predictions: np.ndarray     # 给定 alpha 下的预测 (1=normal, -1=anomaly)
    alpha: float                # 显著性水平
    threshold: float            # nonconformity score 阈值
    calibration_scores: np.ndarray  # 校准集的 nonconformity scores
    coverage_guarantee: float   # 理论覆盖保证 1-alpha


def split_conformal_anomaly(
    X_train: np.ndarray,
    X_cal: np.ndarray,
    X_test: np.ndarray,
    alpha: float = 0.1,
    nu: float = 0.1,
    gamma: str = "scale",
) -> ConformalResult:
    """Split conformal prediction 用于异常检测。

    Args:
        X_train: 训练集（仅正常样本），用于拟合 OC-SVM
        X_cal: 校准集（仅正常样本），用于计算 nonconformity 分位数
        X_test: 测试集
        alpha: 显著性水平（如 0.1 = 90% 覆盖保证）
        nu, gamma: OC-SVM 参数

    Returns:
        ConformalResult
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_cal_s = scaler.transform(X_cal)
    X_test_s = scaler.transform(X_test)

    model = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
    model.fit(X_train_s)

    # Nonconformity score = -decision_function (越大越异常)
    cal_scores = -model.decision_function(X_cal_s)
    test_scores = -model.decision_function(X_test_s)

    # Conformal quantile (finite-sample correction)
    n_cal = len(cal_scores)
    q_level = np.ceil((1 - alpha) * (n_cal + 1)) / n_cal
    q_level = min(q_level, 1.0)
    threshold = np.quantile(cal_scores, q_level)

    # P-values for test points
    p_values = np.array([
        (np.sum(cal_scores >= s) + 1) / (n_cal + 1)
        for s in test_scores
    ])

    predictions = np.where(p_values >= alpha, 1, -1)

    return ConformalResult(
        p_values=p_values,
        predictions=predictions,
        alpha=alpha,
        threshold=threshold,
        calibration_scores=cal_scores,
        coverage_guarantee=1 - alpha,
    )


def conformal_with_multiple_alphas(
    X_train: np.ndarray,
    X_cal: np.ndarray,
    X_test: np.ndarray,
    y_test: Optional[np.ndarray] = None,
    alphas: Tuple[float, ...] = (0.01, 0.05, 0.1, 0.2),
    nu: float = 0.1,
    gamma: str = "scale",
) -> dict:
    """在多个显著性水平下运行 conformal prediction。

    返回每个 alpha 的结果 + 如果有 y_test 则计算实际覆盖率。
    """
    results = {}
    for a in alphas:
        cr = split_conformal_anomaly(X_train, X_cal, X_test, alpha=a, nu=nu, gamma=gamma)
        entry = {
            "alpha": a,
            "guarantee": 1 - a,
            "result": cr,
            "n_anomaly": int((cr.predictions == -1).sum()),
            "n_total": len(X_test),
            "anomaly_rate": float((cr.predictions == -1).mean()),
        }

        if y_test is not None:
            normal_mask = y_test == 1
            if normal_mask.any():
                entry["empirical_coverage"] = float(
                    (cr.predictions[normal_mask] == 1).mean()
                )
            anomaly_mask = y_test == -1
            if anomaly_mask.any():
                entry["detection_rate"] = float(
                    (cr.predictions[anomaly_mask] == -1).mean()
                )

        results[a] = entry

    return results
