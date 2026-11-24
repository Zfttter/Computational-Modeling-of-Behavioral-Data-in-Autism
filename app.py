"""Behavioral Data Analysis — Interactive Demo
==============================================

启动: streamlit run app.py

功能:
1. 合成数据生成 & 可视化（骨架动画 / 轨迹 / 频谱）
2. 预处理前后对比
3. HMM 多状态分段（交互调参）
4. 贝叶斯变点检测 + 后验可视化
5. One-class SVM 异常检测
6. 姿态几何刻板行为检测
7. 真实数据（MMASD）加载（如有）
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
import pandas as pd
from scipy.ndimage import uniform_filter1d

# ── 项目模块 ──
from src.data.synthetic import generate_sequence
from src.data.mmasd_loader import BODY_25_JOINTS, BODY_25_SKELETON, ACTIVITY_MAP
from src.preprocessing.signal import pipeline as preprocess_pipeline
from src.preprocessing.windows import sliding_windows, window_features
from src.models.hmm import fit_hmm, find_state_boundaries, compute_dwell_times
from src.models.oneclass import fit_oneclass, score
from src.models.vlm_analyzer import (
    detect_arm_flapping, detect_rocking, detect_head_banging, pose_geometry_analysis
)

# ══════════════════════════════════════════════════════════════
# 页面配置
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Behavioral Data in Autism — Demo",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Computational Modeling of Behavioral Data in Autism")
st.caption("Interactive demo — skeleton preprocessing, Bayesian inference, HMM segmentation, anomaly detection")

# ══════════════════════════════════════════════════════════════
# 侧边栏：数据源 & 参数
# ══════════════════════════════════════════════════════════════
st.sidebar.header("⚙️ Parameters")

data_source = st.sidebar.radio(
    "Data source",
    ["Synthetic (built-in)", "Upload skeleton JSON"],
    index=0,
)

if data_source == "Synthetic (built-in)":
    mode = st.sidebar.selectbox(
        "Movement pattern",
        ["normal", "repetitive", "changepoint"],
        index=1,
        help="normal=typical movement, repetitive=arm flapping, changepoint=calm→active transition",
    )
    duration = st.sidebar.slider("Duration (seconds)", 3, 30, 10)
    T = duration * 30
    seed = st.sidebar.number_input("Random seed", 0, 9999, 42)

st.sidebar.markdown("---")
st.sidebar.subheader("HMM")
n_states = st.sidebar.slider("Number of states", 2, 5, 3)
hmm_iter = st.sidebar.slider("EM iterations", 10, 200, 80)

st.sidebar.markdown("---")
st.sidebar.subheader("One-class SVM")
nu = st.sidebar.slider("ν (outlier fraction)", 0.01, 0.30, 0.10, 0.01)
win_size = st.sidebar.slider("Window size (frames)", 30, 120, 60, 10)
win_stride = st.sidebar.slider("Window stride (frames)", 10, 60, 30, 5)

# ══════════════════════════════════════════════════════════════
# 生成 / 加载数据
# ══════════════════════════════════════════════════════════════

@st.cache_data
def get_synthetic(T, mode, seed):
    return generate_sequence(T, mode=mode, seed=seed)

if data_source == "Synthetic (built-in)":
    data = get_synthetic(T, mode, seed)
    kp_raw = data["kp"]          # (T, 25, 2)
    conf = data["conf"]          # (T, 25)
    fs = 30.0
else:
    uploaded = st.file_uploader("Upload OpenPose keypoints JSON", type=["json"])
    if uploaded is None:
        st.info("👆 Upload an OpenPose `*_keypoints.json` file, or switch to Synthetic data.")
        st.stop()
    import json
    kp_data = json.load(uploaded)
    people = kp_data.get("people", [])
    if not people:
        st.error("No people found in JSON.")
        st.stop()
    arr = np.array(people[0]["pose_keypoints_2d"]).reshape(25, 3)
    kp_raw = arr[None, :, :2]
    conf = arr[None, :, 2]
    fs = 30.0
    st.warning("Single-frame upload — most analyses need a sequence. Use synthetic for full demo.")

# 预处理
kp3 = np.concatenate([kp_raw, np.zeros((*kp_raw.shape[:2], 1))], axis=-1)
kp_cleaned = preprocess_pipeline(kp3, conf, fs=fs)

T_actual = kp_raw.shape[0]
t_sec = np.arange(T_actual) / fs

# ══════════════════════════════════════════════════════════════
# Tab 布局
# ══════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🦴 Skeleton", "🔧 Preprocessing", "📊 HMM Segmentation",
    "📐 Bayesian Changepoint", "🎯 Anomaly Detection", "🔍 Behavior Detection",
])

# ── Tab 1: 骨架可视化 ────────────────────────────────────────
with tab1:
    st.subheader("Skeleton Visualization")

    col1, col2 = st.columns([1, 2])
    with col1:
        frame_idx = st.slider("Frame", 0, T_actual - 1, T_actual // 2, key="skel_frame")

    with col2:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        for ax, (kp, title) in zip(axes, [(kp_raw, "Raw"), (kp_cleaned[:, :, :2], "Cleaned")]):
            frame = kp[frame_idx]
            c = conf[frame_idx] if title == "Raw" else np.ones(25)
            for i, j in BODY_25_SKELETON:
                if c[i] > 0.3 and c[j] > 0.3:
                    ax.plot([frame[i, 0], frame[j, 0]], [frame[i, 1], frame[j, 1]],
                            "b-", lw=1.5, alpha=0.7)
            sc = ax.scatter(frame[:, 0], frame[:, 1], c=c, cmap="RdYlGn",
                           vmin=0, vmax=1, s=40, zorder=5, edgecolors="k", linewidths=0.5)
            ax.set_title(title)
            ax.set_aspect("equal")
            ax.invert_yaxis()
        fig.colorbar(sc, ax=axes[1], label="Confidence", shrink=0.6)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    # 关节轨迹
    st.markdown("#### Joint Trajectories")
    joint_name = st.selectbox("Joint", BODY_25_JOINTS, index=4)  # default RWrist
    ji = BODY_25_JOINTS.index(joint_name)

    fig, axes = plt.subplots(2, 1, figsize=(12, 4), sharex=True)
    axes[0].plot(t_sec, kp_raw[:, ji, 0], alpha=0.5, label="Raw X")
    axes[0].plot(t_sec, kp_cleaned[:, ji, 0], label="Cleaned X")
    axes[0].set_ylabel("X")
    axes[0].legend(fontsize=8)
    axes[1].plot(t_sec, kp_raw[:, ji, 1], alpha=0.5, label="Raw Y")
    axes[1].plot(t_sec, kp_cleaned[:, ji, 1], label="Cleaned Y")
    axes[1].set_ylabel("Y")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(fontsize=8)
    fig.suptitle(f"{joint_name} Trajectory", y=1.01)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── Tab 2: 预处理 ─────────────────────────────────────────
with tab2:
    st.subheader("Preprocessing Pipeline")
    st.markdown("""
    **Pipeline stages:** Confidence masking → Interpolation → Median despike → Butterworth lowpass → Root normalization
    """)

    ji_pre = BODY_25_JOINTS.index("RWrist")

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(t_sec, kp_raw[:, ji_pre, 1], color="gray", alpha=0.6, label="Raw")
    axes[0].plot(t_sec, kp_cleaned[:, ji_pre, 1], color="steelblue", label="Cleaned")
    axes[0].set_ylabel("RWrist Y")
    axes[0].legend()
    axes[0].set_title("Signal: Raw vs Cleaned")

    axes[1].plot(t_sec, conf[:, ji_pre], color="orange")
    axes[1].set_ylabel("Confidence")
    axes[1].axhline(0.3, color="red", ls="--", alpha=0.5, label="Threshold")
    axes[1].legend()

    # 频谱
    y_raw = kp_raw[:, ji_pre, 1]
    y_clean = kp_cleaned[:, ji_pre, 1]
    freqs = np.fft.rfftfreq(len(y_raw), 1 / fs)
    axes[2].plot(freqs, np.abs(np.fft.rfft(y_raw - y_raw.mean())), alpha=0.5, label="Raw")
    axes[2].plot(freqs, np.abs(np.fft.rfft(y_clean - y_clean.mean())), label="Cleaned")
    axes[2].set_xlabel("Frequency (Hz)")
    axes[2].set_ylabel("Amplitude")
    axes[2].set_xlim(0, 15)
    axes[2].legend()
    axes[2].set_title("Frequency Spectrum")

    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    # 统计
    low_conf = (conf < 0.3).mean()
    st.metric("Low confidence rate", f"{low_conf:.1%}")

# ── Tab 3: HMM 分段 ──────────────────────────────────────
with tab3:
    st.subheader("HMM Multi-State Segmentation")

    # 速度信号
    vel = np.linalg.norm(np.diff(kp_cleaned[:, :, :2], axis=0), axis=-1)  # (T-1, J)
    mean_speed = vel.mean(axis=1) * fs
    speed_smooth = uniform_filter1d(mean_speed, 10)

    @st.cache_data
    def run_hmm(speed, _n_states, _n_iter):
        return fit_hmm(speed, n_states=_n_states, n_iter=_n_iter, seed=42)

    hmm_res = run_hmm(speed_smooth, n_states, hmm_iter)

    state_colors = ["#4CAF50", "#FF9800", "#F44336", "#9C27B0", "#2196F3"][:n_states]
    state_names = [f"State {k}" for k in range(n_states)]

    t_vel = np.arange(len(speed_smooth)) / fs

    fig, axes = plt.subplots(3, 1, figsize=(12, 8),
                             gridspec_kw={"height_ratios": [2, 1, 1.2]})

    # 信号 + 状态
    axes[0].plot(t_vel, speed_smooth, color="gray", lw=0.5, alpha=0.5)
    for k in range(n_states):
        mask = hmm_res.state_seq == k
        axes[0].fill_between(t_vel, 0, speed_smooth, where=mask,
                             alpha=0.4, color=state_colors[k], label=state_names[k])
    axes[0].set_ylabel("Mean Joint Speed")
    axes[0].set_title("HMM State Segmentation")
    axes[0].legend(loc="upper right", fontsize=8)

    # 后验
    for k in range(n_states):
        axes[1].plot(t_vel, hmm_res.posteriors[:, k], color=state_colors[k], alpha=0.8)
    axes[1].set_ylabel("P(state)")

    # 转移矩阵
    im = axes[2].imshow(hmm_res.trans_mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(n_states):
        for j in range(n_states):
            axes[2].text(j, i, f"{hmm_res.trans_mat[i,j]:.2f}",
                         ha="center", va="center", fontsize=9,
                         color="white" if hmm_res.trans_mat[i, j] > 0.5 else "black")
    axes[2].set_xticks(range(n_states))
    axes[2].set_yticks(range(n_states))
    axes[2].set_xticklabels(state_names, fontsize=8)
    axes[2].set_yticklabels(state_names, fontsize=8)
    axes[2].set_title("Transition Matrix")
    plt.colorbar(im, ax=axes[2], fraction=0.03)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    # 停留时间
    dwells = compute_dwell_times(hmm_res.state_seq, fs)
    st.markdown("#### Dwell Time Statistics")
    dwell_df = pd.DataFrame([
        {"State": state_names[k], "Segments": v["count"],
         "Mean (s)": f"{v['mean_duration']:.2f}", "Total (s)": f"{v['total_time']:.2f}"}
        for k, v in dwells.items()
    ])
    st.dataframe(dwell_df, use_container_width=True, hide_index=True)

    boundaries = find_state_boundaries(hmm_res.state_seq)
    st.info(f"Detected **{len(boundaries)}** state boundaries")

# ── Tab 4: 贝叶斯变点 ────────────────────────────────────
with tab4:
    st.subheader("Bayesian Changepoint Detection")
    st.markdown("Estimates the moment of movement regime change with **posterior uncertainty**.")

    if T_actual < 60:
        st.warning("Sequence too short for changepoint detection (need ≥60 frames)")
    else:
        ji_cp = BODY_25_JOINTS.index("RWrist")
        wrist_speed = np.linalg.norm(np.diff(kp_cleaned[:, ji_cp, :2], axis=0), axis=-1) * fs
        wrist_smooth = uniform_filter1d(wrist_speed, 10)

        with st.spinner("Running MCMC sampling..."):
            try:
                from src.models.bayes_changepoint import single_changepoint
                idata = single_changepoint(wrist_smooth, draws=500, tune=500, seed=42)
                tau_samples = idata.posterior["tau"].values.flatten()
                tau_med = np.median(tau_samples)

                t_cp = np.arange(len(wrist_smooth)) / fs

                fig, axes = plt.subplots(2, 1, figsize=(12, 6))

                axes[0].plot(t_cp, wrist_smooth, color="steelblue", lw=0.8)
                axes[0].axvline(tau_med / fs, color="red", ls="--",
                                label=f"τ median: {tau_med/fs:.2f}s")
                lo = np.percentile(tau_samples, 5) / fs
                hi = np.percentile(tau_samples, 95) / fs
                axes[0].axvspan(lo, hi, alpha=0.15, color="red", label="90% CI")
                axes[0].set_ylabel("RWrist Speed")
                axes[0].set_title("Bayesian Changepoint Detection")
                axes[0].legend()

                axes[1].hist(tau_samples / fs, bins=50, density=True, color="coral", alpha=0.7)
                axes[1].set_xlabel("Time (s)")
                axes[1].set_ylabel("Posterior density")
                axes[1].set_title("Changepoint τ Posterior Distribution")

                fig.tight_layout()
                st.pyplot(fig)
                plt.close()

                col1, col2, col3 = st.columns(3)
                col1.metric("τ median", f"{tau_med/fs:.2f}s")
                col2.metric("90% CI", f"[{lo:.2f}, {hi:.2f}]s")
                col3.metric("Posterior std", f"{tau_samples.std()/fs:.3f}s")

            except Exception as e:
                st.error(f"Bayesian sampling failed: {e}")
                st.info("Try installing PyMC: `pip install pymc`")

# ── Tab 5: 异常检测 ──────────────────────────────────────
with tab5:
    st.subheader("One-class SVM Anomaly Detection")
    st.markdown("""
    Trains on the **first half** of the sequence (assumed normal),
    then scores **all windows** — anomalies (score < 0) are flagged.
    """)

    wins = sliding_windows(kp_cleaned, win=win_size, stride=win_stride)
    if len(wins) < 4:
        st.warning("Sequence too short for windowed analysis. Increase duration or decrease window size.")
    else:
        feats = []
        valid_idx = []
        for i, w in enumerate(wins):
            try:
                f = window_features(w)
                if np.isfinite(f).all():
                    feats.append(f)
                    valid_idx.append(i)
            except:
                pass

        X = np.array(feats)
        n_train = max(3, len(X) // 2)

        model = fit_oneclass(X[:n_train], nu=nu)
        scores_all = score(model, X)
        anomalies = scores_all < 0

        win_centers = np.array(valid_idx) * win_stride + win_size // 2
        win_times = win_centers / fs

        fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

        # 分数
        colors = ["salmon" if a else "steelblue" for a in anomalies]
        axes[0].bar(win_times, scores_all, width=win_stride / fs * 0.8, color=colors, alpha=0.7)
        axes[0].axhline(0, color="k", ls="--", lw=0.8)
        axes[0].set_ylabel("Decision score")
        axes[0].set_title("One-class SVM Scores (red = anomaly)")

        # 对应的信号
        mean_speed = np.linalg.norm(np.diff(kp_cleaned[:, :, :2], axis=0), axis=-1).mean(1) * fs
        t_speed = np.arange(len(mean_speed)) / fs
        axes[1].plot(t_speed, mean_speed, color="gray", lw=0.5)
        for i, (wt, anom) in enumerate(zip(win_times, anomalies)):
            if anom:
                axes[1].axvspan(wt - win_size / fs / 2, wt + win_size / fs / 2,
                               alpha=0.2, color="salmon")
        axes[1].set_ylabel("Mean speed")
        axes[1].set_xlabel("Time (s)")

        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

        n_anom = anomalies.sum()
        st.metric("Anomalous windows", f"{n_anom} / {len(anomalies)} ({n_anom/len(anomalies):.0%})")

# ── Tab 6: 行为检测 ──────────────────────────────────────
with tab6:
    st.subheader("Pose-based Behavior Detection")
    st.markdown("Rule-based detection of repetitive/self-stimulatory behaviors using skeleton geometry and frequency analysis.")

    result = pose_geometry_analysis(kp_cleaned, fs)

    if result.is_repetitive:
        st.error(f"⚠️ Repetitive behavior detected: **{result.activity_label}**")
    else:
        st.success("✅ No repetitive behaviors detected")

    st.markdown(f"**Description:** {result.description}")
    st.markdown(f"**Confidence:** {result.confidence:.0%}")
    st.markdown(f"**Method:** {result.method}")

    st.markdown("---")
    st.markdown("#### Detailed Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🫲 Arm Flapping**")
        flap = detect_arm_flapping(kp_cleaned, fs)
        if flap["is_detected"]:
            st.error("Detected")
            for d in flap["details"]:
                jname = BODY_25_JOINTS[d["joint"]]
                st.write(f"  {jname}: {d['dominant_freq']:.1f} Hz, amp={d['amplitude']:.4f}")
        else:
            st.success("Not detected")

    with col2:
        st.markdown("**🔄 Body Rocking**")
        rock = detect_rocking(kp_cleaned, fs)
        if rock["is_detected"]:
            st.error(f"Detected — {rock['dominant_freq']:.1f} Hz (power: {rock['power_ratio']:.1%})")
        else:
            st.success(f"Not detected (power: {rock['power_ratio']:.1%})")

    with col3:
        st.markdown("**💥 Head Banging**")
        hb = detect_head_banging(kp_cleaned, fs)
        if hb["is_detected"]:
            st.error(f"Detected — {hb['n_spikes']} spikes (max accel: {hb['max_accel']:.2f})")
        else:
            st.success(f"Not detected ({hb['n_spikes']} spike events)")

    # 频谱图
    st.markdown("#### Wrist Frequency Spectrum")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (ji, name) in zip(axes, [(4, "RWrist"), (7, "LWrist")]):
        y = kp_cleaned[:, ji, 1]
        y_det = y - np.convolve(y, np.ones(30) / 30, mode="same")
        spec = np.abs(np.fft.rfft(y_det))
        freqs = np.fft.rfftfreq(len(y_det), 1 / fs)
        ax.plot(freqs, spec, color="steelblue")
        ax.axvspan(2, 6, alpha=0.1, color="red", label="Flapping range (2-6 Hz)")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"{name} Spectrum")
        ax.set_xlim(0, 10)
        ax.legend(fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

# ══════════════════════════════════════════════════════════════
# Footer
# ══════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    "**Computational Modeling of Behavioral Data in Autism** · "
    "Independent Research Project · 2026"
)
