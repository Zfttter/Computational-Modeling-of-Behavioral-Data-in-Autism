"""MMASD 数据加载器。

数据集结构(Google Drive 下载后):
  data/raw/MMASD/
    skeleton_2d/           # OpenPose JSON: {activity}_{childID}_{day}_{clip}_{label}/
      {frame}_keypoints.json
    skeleton_3d/           # ROMP NPZ
      {activity}_{childID}_{day}_{clip}_{label}.npz
    optical_flow/          # jpg
    scores/                # 临床评分

文件名编码: {activity}_{childID}_{day}_{clipIdx}_{label}
  activity: as(assembling), bs(building_steps), ce(cause_effect), dr(drawing),
            fg(finger_game), mfs(moving_fast_slow), ms(making_sounds),
            sac(social_approach_child), sq(squeezing), tr(tracing), tw(tower)
  label: y(通过), n(未通过), i(中间)

2D 骨架: OpenPose BODY_25, 25 joints × (x, y, confidence)
3D 骨架: ROMP, 71 joints × (x, y, z)
"""

import json
import pathlib
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# OpenPose BODY_25 关节名
BODY_25_JOINTS = [
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist",
    "LShoulder", "LElbow", "LWrist", "MidHip", "RHip",
    "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
    "REye", "LEye", "REar", "LEar",
    "LBigToe", "LSmallToe", "LHeel",
    "RBigToe", "RSmallToe", "RHeel",
]

# BODY_25 骨骼连接
BODY_25_SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # 右臂
    (1, 5), (5, 6), (6, 7),                 # 左臂
    (1, 8), (8, 9), (9, 10), (10, 11),      # 右腿
    (8, 12), (12, 13), (13, 14),            # 左腿
    (0, 15), (0, 16), (15, 17), (16, 18),   # 头/耳
    (14, 19), (19, 20), (14, 21),           # 左脚
    (11, 22), (22, 23), (11, 24),           # 右脚
]

# 活动名 → 中英文
ACTIVITY_MAP = {
    "as": "assembling",
    "bs": "building_steps",
    "ce": "cause_effect",
    "dr": "drawing",
    "fg": "finger_game",
    "mfs": "moving_fast_slow",
    "ms": "making_sounds",
    "sac": "social_approach_child",
    "sq": "squeezing",
    "tr": "tracing",
    "tw": "tower",
}

LABEL_MAP = {"y": "pass", "n": "fail", "i": "intermediate"}

# 文件名解析
_FNAME_RE = re.compile(
    r"^(?P<activity>[a-z]+)_(?P<child>\d+)_D(?P<day>\d+)_(?P<clip>\d+)_(?P<label>[yni])"
)


@dataclass
class Sample:
    """一个 MMASD 片段。"""
    activity: str
    child_id: str
    day: int
    clip_idx: int
    label: str          # y / n / i
    path: pathlib.Path
    kp_2d: Optional[np.ndarray] = None   # (T, J=25, 3) x,y,conf
    kp_3d: Optional[np.ndarray] = None   # (P, J=71, 3)


def parse_sample_id(name: str) -> Optional[dict]:
    m = _FNAME_RE.match(name)
    if not m:
        return None
    return {
        "activity": m.group("activity"),
        "child_id": m.group("child"),
        "day": int(m.group("day")),
        "clip_idx": int(m.group("clip")),
        "label": m.group("label"),
    }


# ── 2D skeleton ──────────────────────────────────────────────

def load_openpose_frame(json_path: pathlib.Path, person_idx: int = 0) -> np.ndarray:
    """加载单帧 OpenPose JSON → (25, 3)。"""
    with open(json_path) as f:
        data = json.load(f)
    people = data.get("people", [])
    if person_idx >= len(people):
        return np.full((25, 3), np.nan)
    kp = np.array(people[person_idx]["pose_keypoints_2d"]).reshape(25, 3)
    return kp


def load_2d_sequence(folder: pathlib.Path, person_idx: int = 0) -> np.ndarray:
    """加载一个片段文件夹中所有帧 → (T, 25, 3)。

    folder 下应有 000000000000_keypoints.json, 000000000001_keypoints.json, ...
    """
    jsons = sorted(folder.glob("*_keypoints.json"))
    if not jsons:
        raise FileNotFoundError(f"No keypoint JSON in {folder}")
    frames = [load_openpose_frame(j, person_idx) for j in jsons]
    return np.stack(frames)


def select_target_person(folder: pathlib.Path, n_sample: int = 5) -> int:
    """启发式选择"目标人物"(通常是儿童,体型较小)。
    取前 n_sample 帧,选平均躯干长度最短的 person。
    """
    jsons = sorted(folder.glob("*_keypoints.json"))[:n_sample]
    if not jsons:
        return 0
    person_heights = {}
    for j in jsons:
        with open(j) as f:
            people = json.load(f).get("people", [])
        for pi, p in enumerate(people):
            kp = np.array(p["pose_keypoints_2d"]).reshape(25, 3)
            neck, hip = kp[1, :2], kp[8, :2]
            if kp[1, 2] > 0.3 and kp[8, 2] > 0.3:
                h = np.linalg.norm(neck - hip)
                person_heights.setdefault(pi, []).append(h)
    if not person_heights:
        return 0
    avg = {pi: np.mean(hs) for pi, hs in person_heights.items()}
    return min(avg, key=avg.get)


# ── 3D skeleton ──────────────────────────────────────────────

def load_3d_skeleton(npz_path: pathlib.Path) -> np.ndarray:
    """加载 ROMP 3D 骨架 NPZ → (num_people, 71, 3)。"""
    data = np.load(npz_path, allow_pickle=True)
    return data["coordinates"]  # (P, 71, 3)


# ── 数据集级 ─────────────────────────────────────────────────

def discover_samples(root: pathlib.Path, modality: str = "2d") -> List[Sample]:
    """扫描 data/raw/MMASD/ 下所有样本。

    modality: "2d" 扫描 skeleton_2d/, "3d" 扫描 skeleton_3d/
    """
    samples = []
    if modality == "2d":
        base = root / "skeleton_2d"
        if not base.exists():
            return samples
        for d in sorted(base.iterdir()):
            if d.is_dir():
                info = parse_sample_id(d.name)
                if info:
                    samples.append(Sample(
                        activity=info["activity"],
                        child_id=info["child_id"],
                        day=info["day"],
                        clip_idx=info["clip_idx"],
                        label=info["label"],
                        path=d,
                    ))
    elif modality == "3d":
        base = root / "skeleton_3d"
        if not base.exists():
            return samples
        for f in sorted(base.glob("*.npz")):
            info = parse_sample_id(f.stem)
            if info:
                samples.append(Sample(
                    activity=info["activity"],
                    child_id=info["child_id"],
                    day=info["day"],
                    clip_idx=info["clip_idx"],
                    label=info["label"],
                    path=f,
                ))
    return samples


def load_dataset(root: pathlib.Path, modality: str = "2d",
                 max_samples: int = 0) -> List[Sample]:
    """发现并加载所有样本。"""
    samples = discover_samples(root, modality)
    if max_samples > 0:
        samples = samples[:max_samples]
    for s in samples:
        if modality == "2d":
            pid = select_target_person(s.path)
            s.kp_2d = load_2d_sequence(s.path, pid)
        elif modality == "3d":
            s.kp_3d = load_3d_skeleton(s.path)
    return samples
