"""隐马尔可夫模型 (HMM) 用于多状态活动分段。

用 Gaussian HMM 对关节运动特征序列做无监督分段:
  - 每个隐状态对应一种运动模式(如：静止、低活动、高活动/刻板行为)
  - 转移矩阵揭示行为切换动态
  - Viterbi 解码给出最优状态序列

不依赖 hmmlearn（避免安装问题），手写前向-后向 + Viterbi + EM。
"""
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class GaussianHMMResult:
    """HMM 拟合结果。"""
    n_states: int
    means: np.ndarray          # (K,) 每状态均值
    stds: np.ndarray           # (K,) 每状态标准差
    trans_mat: np.ndarray      # (K, K) 转移概率
    init_probs: np.ndarray     # (K,) 初始概率
    state_seq: np.ndarray      # (T,) Viterbi 解码的最优状态序列
    log_likelihood: float
    posteriors: np.ndarray     # (T, K) 每帧的状态后验概率


def _log_gaussian(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """对数高斯密度。"""
    return -0.5 * np.log(2 * np.pi) - np.log(sigma + 1e-10) \
           - 0.5 * ((x - mu) / (sigma + 1e-10)) ** 2


def _forward(log_obs: np.ndarray, log_trans: np.ndarray,
             log_init: np.ndarray) -> Tuple[np.ndarray, float]:
    """前向算法 (log domain)。
    log_obs: (T, K), log_trans: (K, K), log_init: (K,)
    返回 log_alpha: (T, K), log_likelihood: float
    """
    T, K = log_obs.shape
    log_alpha = np.full((T, K), -np.inf)
    log_alpha[0] = log_init + log_obs[0]

    for t in range(1, T):
        for j in range(K):
            log_alpha[t, j] = _logsumexp(log_alpha[t-1] + log_trans[:, j]) + log_obs[t, j]

    ll = _logsumexp(log_alpha[-1])
    return log_alpha, ll


def _backward(log_obs: np.ndarray, log_trans: np.ndarray) -> np.ndarray:
    """后向算法 (log domain)。"""
    T, K = log_obs.shape
    log_beta = np.zeros((T, K))  # log(1) = 0

    for t in range(T - 2, -1, -1):
        for j in range(K):
            log_beta[t, j] = _logsumexp(
                log_trans[j, :] + log_obs[t+1] + log_beta[t+1]
            )
    return log_beta


def _logsumexp(x: np.ndarray) -> float:
    """数值稳定的 log-sum-exp。"""
    m = x.max()
    if m == -np.inf:
        return -np.inf
    return m + np.log(np.sum(np.exp(x - m)))


def _viterbi(log_obs: np.ndarray, log_trans: np.ndarray,
             log_init: np.ndarray) -> np.ndarray:
    """Viterbi 解码。"""
    T, K = log_obs.shape
    delta = np.full((T, K), -np.inf)
    psi = np.zeros((T, K), dtype=int)

    delta[0] = log_init + log_obs[0]

    for t in range(1, T):
        for j in range(K):
            scores = delta[t-1] + log_trans[:, j]
            psi[t, j] = np.argmax(scores)
            delta[t, j] = scores[psi[t, j]] + log_obs[t, j]

    # 回溯
    states = np.zeros(T, dtype=int)
    states[-1] = np.argmax(delta[-1])
    for t in range(T - 2, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]

    return states


def fit_hmm(y: np.ndarray, n_states: int = 3, n_iter: int = 50,
            seed: int = 42) -> GaussianHMMResult:
    """用 EM 算法拟合 1D Gaussian HMM。

    Args:
        y: (T,) 观测序列（如关节速度）
        n_states: 隐状态数
        n_iter: EM 迭代数
        seed: 随机种子（用于初始化）

    Returns:
        GaussianHMMResult
    """
    rng = np.random.default_rng(seed)
    T = len(y)
    K = n_states

    # 初始化：K-means 风格
    percentiles = np.linspace(0, 100, K + 2)[1:-1]
    means = np.percentile(y, percentiles).copy()
    stds = np.full(K, y.std() / K)
    trans_mat = np.full((K, K), 0.1 / (K - 1))
    np.fill_diagonal(trans_mat, 0.9)  # 高自转移
    init_probs = np.ones(K) / K

    log_trans = np.log(trans_mat + 1e-10)
    log_init = np.log(init_probs + 1e-10)

    ll_prev = -np.inf

    for iteration in range(n_iter):
        # E-step: 前向-后向
        log_obs = np.array([_log_gaussian(y, means[k], stds[k]) for k in range(K)]).T  # (T, K)
        log_alpha, ll = _forward(log_obs, log_trans, log_init)
        log_beta = _backward(log_obs, log_trans)

        # 后验 gamma(t, k) = P(z_t = k | y)
        log_gamma = log_alpha + log_beta
        log_gamma -= np.array([_logsumexp(log_gamma[t]) for t in range(T)])[:, None]
        gamma = np.exp(log_gamma)

        # 转移后验 xi(t, i, j) = P(z_t=i, z_{t+1}=j | y)
        xi = np.zeros((T - 1, K, K))
        for t in range(T - 1):
            for i in range(K):
                for j in range(K):
                    xi[t, i, j] = log_alpha[t, i] + log_trans[i, j] + \
                                  log_obs[t+1, j] + log_beta[t+1, j]
            xi[t] -= _logsumexp(xi[t].ravel())
        xi = np.exp(xi)

        # M-step
        for k in range(K):
            w = gamma[:, k]
            w_sum = w.sum() + 1e-10
            means[k] = (w * y).sum() / w_sum
            stds[k] = np.sqrt((w * (y - means[k])**2).sum() / w_sum) + 1e-4

        init_probs = gamma[0] + 1e-10
        init_probs /= init_probs.sum()

        trans_mat = xi.sum(0) + 1e-10
        trans_mat /= trans_mat.sum(1, keepdims=True)

        log_trans = np.log(trans_mat)
        log_init = np.log(init_probs)

        # 收敛检查
        if abs(ll - ll_prev) < 1e-4:
            break
        ll_prev = ll

    # Viterbi 解码
    log_obs = np.array([_log_gaussian(y, means[k], stds[k]) for k in range(K)]).T
    states = _viterbi(log_obs, log_trans, log_init)

    # 按均值排序状态（0=最低活动）
    order = np.argsort(means)
    means = means[order]
    stds = stds[order]
    trans_mat = trans_mat[order][:, order]
    init_probs = init_probs[order]
    remap = np.argsort(order)
    states = remap[states]
    gamma = gamma[:, order]

    return GaussianHMMResult(
        n_states=K,
        means=means,
        stds=stds,
        trans_mat=trans_mat,
        init_probs=init_probs,
        state_seq=states,
        log_likelihood=ll,
        posteriors=gamma,
    )


def find_state_boundaries(states: np.ndarray) -> list:
    """从状态序列中提取状态变化的边界帧。"""
    boundaries = []
    for t in range(1, len(states)):
        if states[t] != states[t - 1]:
            boundaries.append(t)
    return boundaries


def compute_dwell_times(states: np.ndarray, fs: float = 30.0) -> dict:
    """计算每个状态的停留时间统计。"""
    K = states.max() + 1
    dwells = {k: [] for k in range(K)}
    current_state = states[0]
    current_len = 1

    for t in range(1, len(states)):
        if states[t] == current_state:
            current_len += 1
        else:
            dwells[current_state].append(current_len / fs)
            current_state = states[t]
            current_len = 1
    dwells[current_state].append(current_len / fs)

    stats = {}
    for k in range(K):
        d = np.array(dwells[k])
        if len(d) > 0:
            stats[k] = {
                "count": len(d),
                "mean_duration": d.mean(),
                "std_duration": d.std(),
                "total_time": d.sum(),
            }
        else:
            stats[k] = {"count": 0, "mean_duration": 0, "std_duration": 0, "total_time": 0}
    return stats
