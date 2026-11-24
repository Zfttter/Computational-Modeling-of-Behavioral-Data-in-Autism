"""One-class SVM:只用"目标/正常"片段训练,检测偏离片段。"""
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


def fit_oneclass(X_train, nu=0.05, gamma="scale"):
    return make_pipeline(StandardScaler(), OneClassSVM(nu=nu, gamma=gamma)).fit(X_train)


def score(model, X):
    """越大越"正常";<0 判为异常/不同类。"""
    return model.decision_function(X)
