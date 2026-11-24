# 自闭症行为数据的计算建模

> Computational Modeling of Behavioral Data in Autism

独立研究项目（2026.09 –）

---

## 1. 研究背景与动机

自闭症谱系障碍（ASD）的行为评估长期依赖临床医师的主观评分。骨架姿态估计技术（OpenPose、ROMP 等）使得从视频中提取客观、可量化的运动信号成为可能。本项目探索如何对这些**2D/3D 骨架时间序列**进行系统建模，实现：

- 自动化的**活动片段分割**（从连续录像中切分不同行为阶段）
- **刻板/重复行为**的检测与刻画（如手臂拍打、身体摇摆、撞头）
- 行为模式的**不确定性量化**（通过贝叶斯后验分布）
- **个体差异**的建模（层级模型：个体 → 群体）

**核心研究问题：** 如何从骨架时间序列中，稳健地检测活动片段并刻画刻板行为，同时量化估计的不确定性？

---

## 2. 数据集

| 数据集 | 内容 | 骨架格式 | ASD 相关 | 角色 |
|---|---|---|---|---|
| **MMASD** | 32 名 ASD 儿童，1,315 个干预片段，>100 小时 | OpenPose 2D (25 关节) + ROMP 3D (71 关节) | ✅ | 主数据集 |
| **SSBD** | 刻板行为视频（拍手、撞头、旋转） | 需自行提取 | ✅ | 补充验证 |

**MMASD 数据结构：**
- `skeleton_2d/`：每帧一个 OpenPose JSON，含 25 个关节的 (x, y, confidence)
- `skeleton_3d/`：每片段一个 NPZ，含 71 个关节的 (x, y, z)
- 文件名编码：`{活动}_{儿童ID}_{天数}_{片段}_{标签(y/n/i)}`
- 11 类干预活动：拼搭、搭积木、因果游戏、画画、手指游戏等

---

## 3. 方法

### 3.1 骨架数据预处理

原始骨架数据含噪声、遮挡导致的缺失和检测跳变。预处理流水线包含五个阶段：

```
原始骨架 → 置信度掩码 → 线性插值 → 中值去脉冲 → Butterworth 低通 → 根关节归一化
```

- **置信度掩码**：OpenPose 置信度 < 0.3 的关节置为 NaN
- **线性插值**：对缺失帧逐关节/逐坐标线性插值（缺口 > 15 帧保留 NaN）
- **中值去脉冲**：5 帧中值滤波消除脉冲噪声
- **Butterworth 低通**：4 阶零相位低通，截止 6 Hz（人体运动能量主要在此以下）
- **根关节归一化**：以髋部为原点，按颈-髋骨长缩放，消除相机位置/距离差异

**预处理消融实验结果：**

| 配置 | F1 | Precision | Recall |
|---|---|---|---|
| 无预处理 | 0.816 | 0.739 | 0.911 |
| 仅插值 | 0.845 | 0.732 | 1.000 |
| 插值 + 去脉冲 | 0.865 | 0.763 | 1.000 |
| 插值 + 去脉冲 + 低通 | 0.870 | 0.769 | 1.000 |
| 完整 pipeline | 0.861 | 0.756 | 1.000 |

> 预处理显著提升 Recall（从 91.1% 到 100%），F1 提升约 5 个百分点。完整 pipeline 包含归一化，对下游跨个体分析更重要。

### 3.2 特征提取

对预处理后的骨架序列使用滑动窗口（60 帧 / 2 秒，步长 30 帧），每个窗口提取 100 维特征向量：

- **速度统计**：每关节速度的均值和标准差（50 维）
- **加速度统计**：每关节加速度幅度均值（25 维）
- **主频**：每关节速度信号 FFT 的峰值频率（25 维），用于捕捉刻板行为的周期性

### 3.3 贝叶斯推断

#### 单变点模型

对关节速度时间序列建模运动状态切换：

$$y_t \sim \mathcal{N}(\mu_k, \sigma), \quad k = \begin{cases} 1 & t < \tau \\ 2 & t \geq \tau \end{cases}$$

- $\tau \sim \text{DiscreteUniform}(0, T-1)$：变点位置
- $\mu_1, \mu_2 \sim \mathcal{N}(\bar{y}, 2s_y)$：两个阶段的均值
- $\sigma \sim \text{HalfNormal}(s_y)$：观测噪声

使用 PyMC 进行 MCMC 采样（NUTS + Metropolis），获得变点 $\tau$ 的**完整后验分布**，从而量化分段位置的不确定性。

#### 层级模型（个体 → 群体）

```
组级:    μ_pop ~ Normal,  σ_pop ~ HalfNormal
个体级:  μ_child_i ~ Normal(μ_pop, σ_pop)
观测级:  y_ij ~ Normal(μ_child_i, σ_obs)
```

- **部分汇聚（partial pooling）**：数据量少的儿童从群体先验"借信息"，避免过拟合
- **收缩效应**：个体后验均值向群体均值收缩，收缩幅度与数据量成反比
- **两组对比变体**：对 pass/fail 两组建立不同的组级先验，直接建模组间差异 $\delta = \mu_{\text{fail}} - \mu_{\text{pass}}$

**实验结果：** 群体均值 $\mu_{\text{pop}} = 1.090 \pm 0.055$，群体标准差 $\sigma_{\text{pop}} = 0.128 \pm 0.054$，收缩效应在小样本个体上最为显著。

### 3.4 隐马尔可夫模型（HMM）

手写 Gaussian HMM（无外部依赖），包含前向-后向算法、Viterbi 解码和 EM 参数估计：

- **K 个隐状态**：对应不同活动强度（如低活动、中等活动、高活动）
- **转移矩阵**：揭示行为切换动态，高自转移概率（>0.91）表示状态稳定
- **Viterbi 解码**：给出最优状态序列
- **停留时间统计**：每状态的出现次数、平均持续时长

**HMM vs 贝叶斯变点分段精度对比（5 次实验）：**

| 方法 | 平均边界误差 |
|---|---|
| HMM（2 状态） | **0.10 秒** |
| 贝叶斯变点 | 1.24 秒 |

> HMM 在分段精度上优于单变点模型，但单变点模型提供了完整的后验不确定性。两者互补。

### 3.5 异常检测：One-class SVM

仅使用"正常"（通过标签）样本训练 One-class SVM，检测偏离正常运动模式的片段：

- 输入：100 维滑窗特征
- 预处理：StandardScaler 标准化
- 核函数：RBF，$\gamma$ 自适应，$\nu = 0.1$（预期异常比例）
- 决策函数 > 0 判为正常，< 0 判为异常

**结果：** F1 = 0.891，Precision = 0.804，Recall = 1.000

### 3.6 姿态几何行为检测

基于领域知识的规则方法，利用骨架几何和频率分析检测三类刻板行为：

| 行为 | 检测方法 | 关键参数 |
|---|---|---|
| 手臂拍打 (arm flapping) | 手腕 Y 坐标 FFT，2-6 Hz 频段峰值 | 振幅阈值 0.02 |
| 身体摇摆 (body rocking) | 躯干中心位移 FFT，0.5-3 Hz | 功率比 > 15% |
| 撞头 (head banging) | 鼻子关节加速度峰值计数 | 加速度 > 0.5，次数 > 3 |

**结果：** 样本级 F1 = 1.000（合成数据），与 One-class SVM 的窗口级分析互补。

### 3.7 VLM 上下文理解

提供了视觉语言模型（GPT-4o）的集成框架：

- 在视频帧上叠加骨架渲染
- 结构化 prompt 引导 VLM 分析活动类型、刻板行为、参与度
- 返回 JSON 格式结果，可与姿态方法对比

> VLM 模块为框架性实现，实际调用需配置 API key。定位为姿态方法的**上下文补充**（理解场景语义），而非替代。

---

## 4. 综合方法对比

| 方法 | 任务 | 关键指标 | 不确定性 | 监督方式 |
|---|---|---|---|---|
| 贝叶斯变点 | 活动分段 | 边界误差 1.24s | ✅ 后验 CI | 无监督 |
| HMM（3 状态） | 多状态分段 | 边界误差 0.10s | ✅ 状态后验 | 无监督 |
| One-class SVM | 异常检测 | F1 = 0.891 | ❌ | 半监督 |
| 姿态几何规则 | 行为分类 | F1 = 1.000 | ❌ | 无监督（领域知识）|
| 层级贝叶斯 | 个体差异 | σ_pop = 0.128 | ✅ 收缩 + CI | 无监督 |
| VLM 上下文 | 场景理解 | 定性 | ❌ | 零样本 |

---

## 5. 交互 Demo

```bash
source .venv/bin/activate
streamlit run app.py
```

浏览器打开 `http://localhost:8501`，包含 6 个 Tab：

| Tab | 功能 |
|---|---|
| 🦴 Skeleton | 逐帧骨架可视化，原始 vs 清洗对比 |
| 🔧 Preprocessing | 信号对比、置信度叠加、频谱分析 |
| 📊 HMM Segmentation | 可调状态数/迭代数，分段 + 转移矩阵 + 停留时间 |
| 📐 Bayesian Changepoint | MCMC 变点检测 + 后验分布 + 90% 置信区间 |
| 🎯 Anomaly Detection | One-class SVM 窗口级异常检测与可视化 |
| 🔍 Behavior Detection | 拍手/摇摆/撞头的频率分析检测 + 频谱图 |

支持三种合成数据模式（normal / repetitive / changepoint）和 OpenPose JSON 上传。

---

## 6. 项目结构

```
├── app.py                          # Streamlit 交互 Demo
├── requirements.txt
├── data/
│   ├── raw/MMASD/                  # 原始数据（不入库）
│   ├── interim/                    # 中间数据
│   └── processed/                  # 处理后数据
├── docs/
│   └── 01_dataset_selection.md     # 数据集选型文档
├── notebooks/
│   ├── 01_EDA_synthetic.py         # 合成数据 EDA（7 张图）
│   ├── 02_pipeline_real_data.py    # 真实数据全流程
│   └── 03_full_evaluation.py       # 全方法对比评估
├── src/
│   ├── data/
│   │   ├── mmasd_loader.py         # MMASD 数据加载器
│   │   └── synthetic.py            # 合成数据生成器
│   ├── preprocessing/
│   │   ├── signal.py               # 预处理流水线
│   │   └── windows.py              # 滑窗 + 特征提取
│   ├── models/
│   │   ├── bayes_changepoint.py    # 贝叶斯变点模型
│   │   ├── bayes_hierarchical.py   # 层级贝叶斯模型
│   │   ├── hmm.py                  # 隐马尔可夫模型
│   │   ├── oneclass.py             # One-class SVM
│   │   └── vlm_analyzer.py         # VLM 分析 + 姿态几何检测
│   └── eval/
│       └── metrics.py              # 评估指标
├── scripts/
│   ├── 00_inspect_dataset.py       # 数据集初检
│   └── 01_download_data.sh         # 数据下载说明
└── results/
    ├── figures/                    # 11 张分析图
    └── tables/                     # 4 张 CSV 结果表
```

---

## 7. 快速开始

```bash
# 1. 创建环境
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. 在合成数据上验证 pipeline
python -I notebooks/01_EDA_synthetic.py

# 3. 全方法评估
python -I notebooks/03_full_evaluation.py

# 4. 启动交互 Demo
streamlit run app.py

# 5.（可选）下载 MMASD 后跑真实数据
# 访问 https://tinyurl.com/mr3rd2dp 下载到 data/raw/MMASD/
python -I notebooks/02_pipeline_real_data.py
```

---

## 8. 局限与未来工作

- **合成数据验证**：当前结果基于合成数据，需在 MMASD 真实数据上验证泛化性
- **单变点 → 多变点**：当前贝叶斯模型假设单个变点，实际行为可能有多次切换
- **VLM 量化评估**：VLM 模块为框架性实现，需要系统的定量对比
- **跨数据集域适应**：不同数据集的骨架格式、相机参数、被试群体存在差异
- **临床验证**：方法需与临床评分（ADOS-2 等）做相关性分析

---

## 9. 参考文献

- Li, J. et al. "MMASD: A Multimodal Dataset for Autism Intervention Analysis." *ICMI '23*. DOI: 10.1145/3577190.3614117
- Rajagopalan, S.S. et al. "Self-Stimulatory Behaviours in the Wild for Autism Diagnosis." *ICCV Workshops, 2013*.
- Salvatier, J., Wiecki, T.V., & Fonnesbeck, C. "Probabilistic programming in Python using PyMC3." *PeerJ Computer Science, 2016*.

---

## 伦理声明

本项目仅使用公开发布且已匿名化的数据集，遵守相应的数据使用协议。项目定位为**研究方法探索**，不做诊断性声明，不重新分发原始数据。
