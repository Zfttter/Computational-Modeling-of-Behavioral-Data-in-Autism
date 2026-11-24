"""数据集初检:统计样本数、帧数、缺失率,并画几条轨迹。用法:
python scripts/00_inspect_dataset.py data/raw/MMASD
(加载逻辑需按数据集实际文件格式在 src/data/ 中补全)"""
import sys, pathlib
root = pathlib.Path(sys.argv[1])
files = sorted(root.rglob("*"))
print(f"{root}: {len(files)} files")
for p in files[:20]:
    print(" ", p.relative_to(root))
