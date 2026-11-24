"""层级贝叶斯模型：个体 → 组。

模型结构:
  - 组级 (population): 所有儿童共享的运动模式先验
  - 个体级 (individual): 每个儿童偏离组均值的程度
  - 观测级: 窗口特征 ~ Normal(个体均值, 个体方差)

用途:
  1. 借力 (partial pooling): 数据少的儿童从组级"借信息"
  2. 量化个体差异: 个体随机效应的后验分布
  3. ASD 严重度关联: 个体效应 vs 临床评分的相关性
"""
import numpy as np
import pymc as pm
import arviz as az


def hierarchical_movement_model(features_by_child: dict,
                                 draws: int = 1000,
                                 tune: int = 1000,
                                 seed: int = 42):
    """层级模型：关节速度的个体-组结构。

    Args:
        features_by_child: {child_id: np.ndarray of shape (n_windows,)}
            每个儿童的平均运动速度（标量特征/窗口）

    Returns:
        idata: ArviZ InferenceData
        model: PyMC model
    """
    child_ids = sorted(features_by_child.keys())
    n_children = len(child_ids)

    # 准备数据: 展平 + child index
    obs = []
    child_idx = []
    for i, cid in enumerate(child_ids):
        vals = features_by_child[cid]
        obs.extend(vals)
        child_idx.extend([i] * len(vals))

    obs = np.array(obs, dtype=np.float64)
    child_idx = np.array(child_idx, dtype=int)
    N = len(obs)

    with pm.Model() as model:
        # 组级超先验
        mu_pop = pm.Normal("mu_pop", mu=obs.mean(), sigma=obs.std())
        sigma_pop = pm.HalfNormal("sigma_pop", sigma=obs.std())

        # 个体级随机效应
        mu_child = pm.Normal("mu_child", mu=mu_pop, sigma=sigma_pop,
                             shape=n_children)

        # 观测噪声
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=obs.std() / 2)

        # 似然
        pm.Normal("y_obs", mu=mu_child[child_idx], sigma=sigma_obs,
                  observed=obs)

        # 采样
        idata = pm.sample(draws, tune=tune, random_seed=seed,
                          target_accept=0.9, progressbar=True)

    # 附加元数据
    idata.attrs["child_ids"] = child_ids

    return idata, model


def hierarchical_two_group(features_by_child: dict,
                            group_labels: dict,
                            draws: int = 1000,
                            tune: int = 1000,
                            seed: int = 42):
    """两组层级模型：比较不同标签组（如 pass vs fail）的运动模式。

    Args:
        features_by_child: {child_id: np.ndarray}
        group_labels: {child_id: 0 or 1}  (0=pass组, 1=fail组)
    """
    child_ids = sorted(features_by_child.keys())
    n_children = len(child_ids)
    groups = np.array([group_labels[c] for c in child_ids])

    obs = []
    child_idx = []
    for i, cid in enumerate(child_ids):
        vals = features_by_child[cid]
        obs.extend(vals)
        child_idx.extend([i] * len(vals))

    obs = np.array(obs, dtype=np.float64)
    child_idx = np.array(child_idx, dtype=int)

    with pm.Model() as model:
        # 两组的组级均值
        mu_group = pm.Normal("mu_group", mu=obs.mean(), sigma=obs.std(), shape=2)
        sigma_group = pm.HalfNormal("sigma_group", sigma=obs.std(), shape=2)

        # 组间差异（直接建模，方便看后验）
        delta = pm.Deterministic("delta", mu_group[1] - mu_group[0])

        # 个体级
        mu_child = pm.Normal("mu_child",
                             mu=mu_group[groups],
                             sigma=sigma_group[groups],
                             shape=n_children)

        sigma_obs = pm.HalfNormal("sigma_obs", sigma=obs.std() / 2)
        pm.Normal("y_obs", mu=mu_child[child_idx], sigma=sigma_obs,
                  observed=obs)

        idata = pm.sample(draws, tune=tune, random_seed=seed,
                          target_accept=0.9, progressbar=True)

    return idata, model


def shrinkage_plot_data(idata, features_by_child: dict):
    """提取画 shrinkage plot 需要的数据。

    返回 dict: {child_id: {raw_mean, posterior_mean, posterior_std}}
    """
    child_ids = sorted(features_by_child.keys())
    mu_child_samples = idata.posterior["mu_child"].values  # (chain, draw, n_children)
    mu_child_mean = mu_child_samples.mean(axis=(0, 1))
    mu_child_std = mu_child_samples.std(axis=(0, 1))
    mu_pop_mean = idata.posterior["mu_pop"].values.mean()

    result = {}
    for i, cid in enumerate(child_ids):
        result[cid] = {
            "raw_mean": features_by_child[cid].mean(),
            "n_obs": len(features_by_child[cid]),
            "posterior_mean": mu_child_mean[i],
            "posterior_std": mu_child_std[i],
            "pop_mean": mu_pop_mean,
        }
    return result
