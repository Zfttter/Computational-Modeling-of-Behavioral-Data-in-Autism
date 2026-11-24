#!/usr/bin/env bash
# 数据下载指南（非自动——需要手动在浏览器完成）
#
# ═══════════════════════════════════════════════════════════════
# 1. MMASD (主数据集)
# ═══════════════════════════════════════════════════════════════
#
# 打开浏览器访问:
#   https://tinyurl.com/mr3rd2dp
# (这会跳转到 Google Drive 文件夹)
#
# 下载后解压到:
#   data/raw/MMASD/
# 预期结构:
#   data/raw/MMASD/
#     skeleton_2d/           # 每个片段一个文件夹
#       {activity}_{childID}_{day}_{clip}_{label}/
#         000000000000_keypoints.json
#         000000000001_keypoints.json
#         ...
#     skeleton_3d/           # 每个片段一个 .npz
#     optical_flow/          # .jpg
#     scores/                # 临床评分
#
# ═══════════════════════════════════════════════════════════════
# 2. SSBD+ (补充数据集)
# ═══════════════════════════════════════════════════════════════
#
# 方法 A: pip install ssbdplus  (代码/模型,不含视频)
# 方法 B: 访问 https://doi.org/10.57702/k3wlsa1z (TIB 数据中心)
#         或联系原作者 (Rajagopalan et al., ICCV 2013)
#
# 下载后放到:
#   data/raw/SSBD/
#
# ═══════════════════════════════════════════════════════════════
# 3. 验证
# ═══════════════════════════════════════════════════════════════

set -e
cd "$(dirname "$0")/.."

echo "检查数据目录..."

if [ -d "data/raw/MMASD/skeleton_2d" ]; then
    n2d=$(find data/raw/MMASD/skeleton_2d -mindepth 1 -maxdepth 1 -type d | wc -l)
    echo "✅ MMASD 2D skeleton: $n2d 个片段"
else
    echo "❌ MMASD 2D skeleton 未找到 — 请下载到 data/raw/MMASD/"
fi

if [ -d "data/raw/MMASD/skeleton_3d" ]; then
    n3d=$(find data/raw/MMASD/skeleton_3d -name "*.npz" | wc -l)
    echo "✅ MMASD 3D skeleton: $n3d 个文件"
else
    echo "❌ MMASD 3D skeleton 未找到"
fi

if [ -d "data/raw/SSBD" ]; then
    echo "✅ SSBD 目录存在"
else
    echo "⚠️  SSBD 未下载（可选）"
fi

echo ""
echo "下一步: python -I scripts/00_inspect_dataset.py data/raw/MMASD"
