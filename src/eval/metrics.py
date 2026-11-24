"""评估指标:分段质量 + 检测性能。"""
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_recall_curve, average_precision_score,
)


def segment_f1(pred_boundaries, true_boundaries, tolerance=5):
    """分段边界 F1(帧级),允许 ±tolerance 帧误差。

    pred_boundaries, true_boundaries: 1D arrays of frame indices.
    """
    if len(true_boundaries) == 0:
        return 1.0 if len(pred_boundaries) == 0 else 0.0
    if len(pred_boundaries) == 0:
        return 0.0

    matched_true = set()
    matched_pred = set()
    for pi, p in enumerate(pred_boundaries):
        for ti, t in enumerate(true_boundaries):
            if abs(p - t) <= tolerance and ti not in matched_true:
                matched_pred.add(pi)
                matched_true.add(ti)
                break

    tp = len(matched_true)
    prec = tp / len(pred_boundaries) if pred_boundaries.size else 0
    rec = tp / len(true_boundaries)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def boundary_error(pred_boundaries, true_boundaries):
    """平均边界误差(帧)。"""
    if len(pred_boundaries) == 0 or len(true_boundaries) == 0:
        return float("inf")
    errors = []
    for t in true_boundaries:
        dists = np.abs(pred_boundaries - t)
        errors.append(dists.min())
    return np.mean(errors)


def oneclass_report(y_true, scores, threshold=0.0):
    """One-class SVM 结果报告。
    y_true: 0=normal, 1=anomaly
    scores: decision function (>0 = normal)
    """
    preds = (scores < threshold).astype(int)
    report = classification_report(y_true, preds,
                                   target_names=["Normal", "Anomaly"],
                                   output_dict=True)
    return {
        "classification_report": report,
        "f1_anomaly": report["Anomaly"]["f1-score"],
        "confusion_matrix": confusion_matrix(y_true, preds).tolist(),
    }


def calibration_check(posterior_samples, true_value):
    """检查后验校准:true_value 落在哪些 credible interval 内。"""
    quantiles = [50, 80, 90, 95]
    results = {}
    for q in quantiles:
        lo = np.percentile(posterior_samples, (100 - q) / 2)
        hi = np.percentile(posterior_samples, 100 - (100 - q) / 2)
        results[f"{q}%_CI"] = {"lo": lo, "hi": hi,
                                "contains": bool(lo <= true_value <= hi)}
    return results
