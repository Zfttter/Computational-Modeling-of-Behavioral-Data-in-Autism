# 数据集选择

## 选择标准
1. 含 **2D/3D 骨架**(简历第二条)
2. 含 **时间序列**(才能做时间依赖的贝叶斯建模)
3. 最好有 **ASD 相关**行为/标签(自闭症语境)
4. 许可允许学术使用,体量适合个人项目
5. 有视频(才能做 VLM 上下文理解)

## 候选

| 数据集 | 内容 | 骨架 | ASD 相关 | 角色 |
|---|---|---|---|---|
| **MMASD** (Multimodal ASD) | ASD 儿童治疗干预视频,含 2D/3D 骨架、光流、人体分割(已匿名化) | 2D+3D | ✅ | **主数据集(推荐)** |
| **SSBD** (Self-Stimulatory Behavior Dataset) | 网络视频:拍手(arm flapping)、摇头撞头(head banging)、旋转(spinning) | 需自行提取(MediaPipe/OpenPose) | ✅ | **次数据集 / VLM 与 one-class 验证** |
| **NTU RGB+D (60/120)** | 通用动作,3D Kinect 骨架 | 3D | ❌ | 典型发育(TD)参考 / 预训练基线 |
| **PKU-MMD** | 长时未裁剪动作序列 | 3D | ❌ | 动作**分段**方法调试 |
| AMASS / Human3.6M | 通用动作捕捉 | 3D | ❌ | 可选:噪声模型/平滑方法验证 |

## 推荐组合
- **主线:MMASD**(骨架现成 → 直接做预处理 + 贝叶斯建模 + one-class SVM)
- **补充:SSBD**(带刻板行为标签的原始视频 → 自己跑姿态估计 + VLM 对比)
- **对照/调试:NTU RGB+D 子集**(健康人动作,用于 one-class SVM 的"正常"分布;注意跨数据集域差异)

## 行动清单(第 1 周)
- [ ] 申请/下载 MMASD(见论文 Li et al., "MMASD: A Multimodal Dataset for Autism Intervention Analysis", ICMI 2023;下载方式以其官方页面为准)
- [ ] 获取 SSBD(需向作者申请,以官方说明为准)
- [ ] 确认各数据集 license / 数据使用协议,**不要重新分发**原始数据
- [ ] 把数据放入 `data/raw/<dataset>/`,运行 `scripts/00_inspect_dataset.py`

> ⚠️ 以上数据集信息来自我的已有知识,下载链接、规模、许可请你以官方页面为准再核实。
