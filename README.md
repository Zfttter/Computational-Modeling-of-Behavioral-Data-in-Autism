# Computational Modeling of Behavioral Data in Autism

独立研究项目(2026.09 –):用贝叶斯推断、骨架预处理、姿态/VLM/one-class SVM 对自闭症相关行为数据做建模与活动分析。

## 研究问题
> 如何从 2D/3D 骨架时间序列中,稳健地检测并刻画活动片段(尤其是刻板/重复行为),并量化不确定性?

## 路线图(12 周)
| 阶段 | 周 | 内容 | 产出 |
|---|---|---|---|
| 0 数据 | 1 | 选数据集、申请、初检 (`docs/01_dataset_selection.md`) | 数据清单、EDA 报告 |
| 1 预处理 | 2–3 | 置信度掩码→插值→去脉冲→低通→归一化 (`src/preprocessing`) | 预处理前后对比图、质量指标 |
| 2 贝叶斯建模 | 4–6 | 变点 → HMM → 层级模型 (`src/models/bayes_*`) | 分段后验 + 不确定性可视化 |
| 3 检测方法 | 7–9 | A: 姿态特征+one-class SVM;B: VLM 上下文理解 | 方法对比表 |
| 4 评估 | 10 | 分段 F1 / 边界误差 / 校准度 | 结果表 + 消融 |
| 5 展示 | 11–12 | 报告、海报、Demo、GitHub | 见下 |

## 最终成果展出
1. **技术报告/短论文(6–8 页)**:问题、方法、实验、局限(可投 workshop 或挂 arXiv)
2. **方法对比表**:预处理消融;one-class SVM vs. VLM vs. 贝叶斯分段
3. **关键图**:原始 vs 清洗后的骨架轨迹;活动分段时间轴(含后验区间);片段特征 2D 嵌入
4. **交互 Demo**:上传/选择一段视频 → 骨架叠加 + 分段时间轴(Streamlit 或 Artifact)
5. **开源仓库**:可复现脚本 + README + 一页海报(简历直接放链接)

## 伦理
涉及自闭症儿童数据:仅使用公开/已授权且已匿名化数据,遵守 data use agreement,不重新分发,不做诊断性声明(定位为研究方法)。

## 结构
```
data/{raw,interim,processed}  数据(不入库)
docs/                          选型与笔记
src/{data,preprocessing,models,eval}
scripts/                       可执行流程
notebooks/                     EDA
results/{figures,tables}
```
# Computational-Modeling-of-Behavioral-Data-in-Autism
