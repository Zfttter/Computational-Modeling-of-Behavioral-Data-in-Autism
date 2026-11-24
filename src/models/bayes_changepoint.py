"""贝叶斯时序建模起点:用 PyMC 对关节速度序列做"单变点"模型,
估计运动状态切换时刻(活动分段)并给出后验不确定性。
后续可扩展为:隐马尔可夫 / 层级模型(个体 → 组)。
"""
import numpy as np
import pymc as pm


def single_changepoint(y, draws=1000, tune=1000, seed=0):
    T = len(y)
    with pm.Model() as m:
        tau = pm.DiscreteUniform("tau", 0, T - 1)
        mu1 = pm.Normal("mu1", y.mean(), y.std() * 2)
        mu2 = pm.Normal("mu2", y.mean(), y.std() * 2)
        sigma = pm.HalfNormal("sigma", y.std())
        mu = pm.math.switch(tau >= np.arange(T), mu1, mu2)
        pm.Normal("obs", mu, sigma, observed=y)
        idata = pm.sample(draws, tune=tune, random_seed=seed, progressbar=True)
    return idata
