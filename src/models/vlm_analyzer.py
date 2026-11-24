"""VLM-based 上下文行为理解。

利用视觉语言模型(VLM)对骨架叠加视频帧做"上下文理解":
  - 输入: 视频帧 + 骨架叠加
  - 输出: 行为描述、活动类型判断、异常检测

支持两种后端:
  1. OpenAI GPT-4V / GPT-4o (需要 API key)
  2. 本地 BLIP-2 / LLaVA (需要 GPU + transformers)

这里实现一个**模块化框架**，真正调用 VLM 需要用户配置 API key。
同时提供基于骨架几何的规则基线，不依赖任何 VLM。
"""
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple
import json
import base64


@dataclass
class BehaviorAnalysis:
    """VLM 或规则方法的行为分析结果。"""
    frame_idx: int
    activity_label: str         # 推断的活动类型
    confidence: float           # 置信度 0-1
    description: str            # 自然语言描述
    is_repetitive: bool         # 是否检测到刻板行为
    method: str                 # "vlm" / "rule_based" / "pose_geometry"


# ═══════════════════════════════════════════════════════════════
# 1. 骨架几何规则方法（不需要 VLM，直接从姿态推断）
# ═══════════════════════════════════════════════════════════════

def detect_arm_flapping(kp_seq: np.ndarray, fs: float = 30.0,
                        freq_range: Tuple[float, float] = (2.0, 6.0),
                        amp_threshold: float = 0.02) -> dict:
    """基于频率分析检测手臂拍打。

    Args:
        kp_seq: (T, J, C) 预处理后的骨架序列
        freq_range: 拍打频率范围 (Hz)
        amp_threshold: 振幅阈值

    Returns:
        dict with is_detected, dominant_freq, amplitude, segments
    """
    wrist_joints = [4, 7]  # R/L Wrist (BODY_25)
    results = {"is_detected": False, "details": []}

    for ji in wrist_joints:
        y = kp_seq[:, ji, 1]  # Y 坐标
        y_detrend = y - np.convolve(y, np.ones(30) / 30, mode="same")  # 去趋势

        # FFT
        spec = np.abs(np.fft.rfft(y_detrend))
        freqs = np.fft.rfftfreq(len(y_detrend), 1 / fs)

        # 在目标频率范围内找峰值
        mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        if mask.any():
            peak_idx = np.argmax(spec[mask])
            peak_freq = freqs[mask][peak_idx]
            peak_amp = spec[mask][peak_idx] / len(y)

            if peak_amp > amp_threshold:
                results["is_detected"] = True
                results["details"].append({
                    "joint": ji,
                    "dominant_freq": float(peak_freq),
                    "amplitude": float(peak_amp),
                })

    return results


def detect_rocking(kp_seq: np.ndarray, fs: float = 30.0,
                   freq_range: Tuple[float, float] = (0.5, 3.0)) -> dict:
    """检测身体摇摆（全身周期性运动）。"""
    # 用躯干中心点（颈 + 髋的平均）
    torso = (kp_seq[:, 1, :2] + kp_seq[:, 8, :2]) / 2  # (T, 2)
    displacement = np.linalg.norm(np.diff(torso, axis=0), axis=1)

    spec = np.abs(np.fft.rfft(displacement - displacement.mean()))
    freqs = np.fft.rfftfreq(len(displacement), 1 / fs)

    mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    if mask.any():
        peak_idx = np.argmax(spec[mask])
        peak_freq = freqs[mask][peak_idx]
        peak_power = spec[mask][peak_idx] / spec.sum()

        return {
            "is_detected": peak_power > 0.15,
            "dominant_freq": float(peak_freq),
            "power_ratio": float(peak_power),
        }

    return {"is_detected": False, "dominant_freq": 0, "power_ratio": 0}


def detect_head_banging(kp_seq: np.ndarray, fs: float = 30.0,
                        accel_threshold: float = 0.5) -> dict:
    """检测撞头行为（头部高加速度事件）。"""
    nose = kp_seq[:, 0, :2]  # (T, 2)
    vel = np.diff(nose, axis=0) * fs
    accel = np.diff(vel, axis=0) * fs
    accel_mag = np.linalg.norm(accel, axis=1)

    spikes = accel_mag > accel_threshold
    n_spikes = spikes.sum()

    return {
        "is_detected": n_spikes > 3,
        "n_spikes": int(n_spikes),
        "max_accel": float(accel_mag.max()),
        "spike_frames": np.where(spikes)[0].tolist()[:20],
    }


def pose_geometry_analysis(kp_seq: np.ndarray, fs: float = 30.0) -> BehaviorAnalysis:
    """综合姿态几何分析:检测所有刻板行为类型。"""
    flapping = detect_arm_flapping(kp_seq, fs)
    rocking = detect_rocking(kp_seq, fs)
    head_bang = detect_head_banging(kp_seq, fs)

    behaviors = []
    if flapping["is_detected"]:
        behaviors.append("arm_flapping")
    if rocking["is_detected"]:
        behaviors.append("body_rocking")
    if head_bang["is_detected"]:
        behaviors.append("head_banging")

    is_repetitive = len(behaviors) > 0

    if not behaviors:
        desc = "No repetitive behaviors detected. Movement patterns appear typical."
        label = "typical_movement"
    else:
        desc = f"Detected repetitive behaviors: {', '.join(behaviors)}."
        if flapping["is_detected"]:
            freq = flapping["details"][0]["dominant_freq"]
            desc += f" Arm flapping at ~{freq:.1f} Hz."
        if rocking["is_detected"]:
            desc += f" Body rocking at ~{rocking['dominant_freq']:.1f} Hz."
        label = behaviors[0]  # 主要行为

    return BehaviorAnalysis(
        frame_idx=0,
        activity_label=label,
        confidence=0.8 if is_repetitive else 0.6,
        description=desc,
        is_repetitive=is_repetitive,
        method="pose_geometry",
    )


# ═══════════════════════════════════════════════════════════════
# 2. VLM 接口（框架，需要 API key 才能实际调用）
# ═══════════════════════════════════════════════════════════════

def render_skeleton_on_frame(frame: np.ndarray, kp_frame: np.ndarray,
                              skeleton_edges: list,
                              conf_threshold: float = 0.3) -> np.ndarray:
    """在视频帧上绘制骨架。

    Args:
        frame: (H, W, 3) BGR
        kp_frame: (25, 3) x,y,conf
        skeleton_edges: list of (i, j)
    """
    import cv2

    overlay = frame.copy()
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
    ]

    for idx, (i, j) in enumerate(skeleton_edges):
        if kp_frame[i, 2] > conf_threshold and kp_frame[j, 2] > conf_threshold:
            pt1 = tuple(kp_frame[i, :2].astype(int))
            pt2 = tuple(kp_frame[j, :2].astype(int))
            color = colors[idx % len(colors)]
            cv2.line(overlay, pt1, pt2, color, 2)

    for j in range(kp_frame.shape[0]):
        if kp_frame[j, 2] > conf_threshold:
            pt = tuple(kp_frame[j, :2].astype(int))
            cv2.circle(overlay, pt, 3, (255, 255, 255), -1)

    return overlay


VLM_PROMPT_TEMPLATE = """You are analyzing a video frame from a behavioral therapy session
involving a child with Autism Spectrum Disorder (ASD). The frame shows skeleton keypoints
overlaid on the scene.

Please analyze:
1. What activity is the child performing?
2. Are there any signs of repetitive/self-stimulatory behaviors (stimming)?
   Examples: arm flapping, body rocking, spinning, head banging
3. What is the child's engagement level with the therapist/activity?
4. Rate the movement quality on a scale of 1-5.

Respond in JSON format:
{
    "activity": "description of activity",
    "repetitive_behavior": true/false,
    "behavior_type": "none/arm_flapping/rocking/spinning/head_banging/other",
    "engagement_level": "high/medium/low",
    "movement_quality": 3,
    "reasoning": "brief explanation"
}"""


def vlm_analyze_frame(frame_path: str, api_key: Optional[str] = None,
                       model: str = "gpt-4o") -> Optional[BehaviorAnalysis]:
    """用 VLM 分析单帧。需要 OpenAI API key。

    如果没有 API key，返回 None。实际项目中替换为真实调用。
    """
    if api_key is None:
        return None

    try:
        import openai
        client = openai.OpenAI(api_key=api_key)

        with open(frame_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": VLM_PROMPT_TEMPLATE},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                ],
            }],
            max_tokens=500,
        )

        text = response.choices[0].message.content
        # 尝试解析 JSON
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(text[start:end])
            return BehaviorAnalysis(
                frame_idx=0,
                activity_label=data.get("behavior_type", "unknown"),
                confidence=data.get("movement_quality", 3) / 5.0,
                description=data.get("reasoning", text),
                is_repetitive=data.get("repetitive_behavior", False),
                method="vlm",
            )
    except Exception as e:
        print(f"VLM analysis failed: {e}")

    return None


# ═══════════════════════════════════════════════════════════════
# 3. 对比框架：规则 vs VLM
# ═══════════════════════════════════════════════════════════════

def compare_methods(kp_seq: np.ndarray,
                    frame_paths: Optional[List[str]] = None,
                    api_key: Optional[str] = None,
                    fs: float = 30.0) -> dict:
    """对比姿态几何方法与 VLM 方法。

    Args:
        kp_seq: (T, J, C) 骨架序列
        frame_paths: 对应的视频帧路径（VLM 需要）
        api_key: OpenAI API key（可选）

    Returns:
        dict with results from each method
    """
    results = {}

    # 方法 A: 姿态几何
    results["pose_geometry"] = pose_geometry_analysis(kp_seq, fs)

    # 方法 B: VLM（如果有 API key 和帧图像）
    if api_key and frame_paths:
        # 采样几帧进行分析
        sample_indices = np.linspace(0, len(frame_paths) - 1, 5, dtype=int)
        vlm_results = []
        for idx in sample_indices:
            r = vlm_analyze_frame(frame_paths[idx], api_key)
            if r:
                r.frame_idx = idx
                vlm_results.append(r)
        results["vlm"] = vlm_results
    else:
        results["vlm"] = None
        results["vlm_note"] = "VLM analysis skipped (no API key or frame images)"

    return results
