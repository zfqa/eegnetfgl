# 面向短窗运动想象脑电解码的神经-黎曼双教师验证集控制全局路由

**作者：** [作者姓名]
**单位：** [院系、机构、城市、国家]
**通讯作者：** [姓名和邮箱]

## 摘要

缩短运动想象脑电（MI-EEG）的分析窗口有助于减少分类前所需的信号长度，但固定时间窗和单一监督方式无法稳定适用于所有被试。本文提出验证集控制全局路由框架，在四个 2 s Compact EEGNet 候选策略中进行被试级选择：固定窗交叉熵、检测器选择的自适应窗交叉熵、固定窗双教师知识蒸馏，以及自适应窗双教师知识蒸馏。双教师分别为冻结的 4 s 神经 EEGNet 和基于 4 s 协方差切空间逻辑回归的黎曼教师。候选策略仅依据训练会话中的验证集准确率选择，评估会话始终隔离。本文在 BCI Competition IV-2a 和 2b 的 9 名被试上采用 5 个随机种子开展跨会话实验。2a 上，全局路由将固定 2 s 交叉熵的准确率由 56.72% 提升至59.31%（配对 t 检验 p=0.0055），但跨 seed 共识选择仅达到 57.11%，相对固定策略不显著，说明单次验证划分存在选择不稳定性。2b 上，路由由 76.34% 提升至 77.37%，但差异不显著。等总权重消融显示，2a 上双教师 0.2+0.2 达到 58.52%，显著高于黎曼单教师 0.4（p=0.0192），相对神经单教师 0.4 接近显著（p=0.0567）；2b 上各蒸馏策略差异均不显著。推理阶段不调用教师，在 RTX 3070 Ti 上，固定策略每 trial 约需 1.01-1.42 ms，自适应策略约需 2.36-3.21 ms。结果支持双教师对四分类短窗解码的互补作用，但验证集路由仍需更严格的折外选择协议。本文方法应被解释为一种短窗策略选择与训练期知识迁移框架，而非对完整时间窗解码的普遍替代。

**关键词：** 脑机接口；脑电；运动想象；时间路由；知识蒸馏；黎曼几何；EEGNet

## 1 引言

运动想象脑机接口通过脑电图推断想象运动，可用于交流、康复和设备控制。许多离线解码研究使用提示后数秒的 EEG。较长时间窗提供更多时序证据，但也会延长系统在作出决策前必须等待的信号时长。因此，短窗解码对响应式系统具有吸引力，不过早期片段可能仅包含不完整的事件相关去同步或同步模式。

紧凑卷积网络与基于协方差的黎曼分类器为 MI-EEG 提供了互补视角。EEGNet 从波形中直接学习时间滤波和深度可分离空间滤波；黎曼方法则以对称正定通道协方差矩阵表示每个 trial，并使用流形距离或切空间特征进行分类。部署时希望模型足够紧凑，而训练阶段的完整时间窗协方差模型可提供较稳健的几何信息。

知识蒸馏可用具有额外信息的教师监督受限学生。本研究关注“时间特权”场景：教师在训练时观察 4 s 信号，学生在训练和推理时仅观察 2 s。初步实验显示，神经教师、黎曼教师与检测器选择的时间窗均具有明显的被试依赖性；对部分被试有益的策略也会对另一些被试造成负迁移。

因此，本文将短窗解码表述为验证集控制的策略路由问题。对每个被试和随机种子分别训练固定窗 CE、自适应窗 CE、固定窗双教师 KD、自适应窗双教师 KD 四个完整候选模型，并只依据训练会话验证准确率选择一个候选。这里的“全局”表示每个被试和训练种子只选择一条路线，而不是对每个测试 trial 使用混合专家门控。测试时不调用教师，仅当选择自适应路线时使用时间检测器。

本文主要贡献如下：

- 提出同时适配时间窗与长窗教师知识使用方式的验证集控制全局路由框架。
- 构建由 4 s 神经 EEGNet 与 4 s 黎曼切空间逻辑回归共同监督 2 s Compact EEGNet 的双教师训练方案。
- 在 BCI Competition IV-2a/2b 上完成 9 被试、5 随机种子、跨会话且严格隔离测试集的实验矩阵。
- 补充等总 KD 权重消融、跨 seed 路由稳健性、推理复杂度和逐类别误差分析，区分教师互补、路由误选与短窗信息上限。

## 2 材料与方法

### 2.1 数据集与实验协议

BCI Competition IV-2a 通过 MOABB 的 BNCI2014_001 获取，包含 9 名被试、22 个 EEG 通道和左手、右手、双脚、舌部 4 类运动想象。2b 通过 BNCI2014_004 获取，包含 9 名被试、3 个双极EEG 通道和左手/右手二分类。两套数据均以 250 Hz 采样。

实验采用被试内、跨会话协议。指定训练会话按类别分层划分为 80% 训练集和 20% 验证集，评估会话保持为完全独立的测试集。划分使用随机种子 42、2024、3407、7 和 123。模型训练、教师拟合、检测器训练、超参数选择、checkpoint 选择和路由选择均只使用训练会话。

**表 1 数据集特征与跨会话协议**

| 数据集 | 类别 | EEG 通道 | 采样率 | 协议 |
| --- | --- | --- | --- | --- |
| BCI IV-2a / BNCI2014_001 | 4 | 22 | 250 Hz | 训练会话→训练/验证；评估会话→测试 |
| BCI IV-2b / BNCI2014_004 | 2 | 3 | 250 Hz | 训练会话→训练/验证；评估会话→测试 |

### 2.2 预处理与数据泄漏控制

EEG trial 采用 8-32 Hz 带通滤波，并截取提示后的 0-4 s。逐通道标准化参数仅在训练子集上拟合，再原样应用于验证集与测试集。固定短窗使用 0-2 s；自适应窗长度始终为 2 s，但起点由检测器选择。

双教师训练中，学生的 2 s 输入和教师的 4 s 输入均从同一个 trial 裁剪并在批处理前保持对应关系。标签和样本顺序不被打乱。教师只产生训练集和验证集软标签，测试阶段仅调用 2 s 学生模型。测试会话不参与时间窗、阈值、超参数、模型 checkpoint 或路线选择。

### 2.3 Compact 学生与长窗教师

所有路由候选分类器均采用 PyTorch Compact EEGNet，输入形状为 [batch, 1, channels, samples]。模型使用 F1=8、深度倍增因子 D=2、F2=16、64 样本时间卷积核、ELU、平均池化和 dropout=0.5。2a 的 4 s 神经教师为 Compact EEGNet；2b 的神经教师为 EEGNetV4 风格模型，采用精确 same padding、Keras 风格 BatchNorm、125 样本时间核、16 样本可分离卷积核以及 max-norm 约束。教师均在同一被试的训练子集上训练并冻结。

黎曼教师使用 Oracle Approximating Shrinkage（OAS）估计 4 s trial 协方差矩阵，按照仿射不变黎曼度量投影到黎曼均值处的切空间，再使用多项逻辑回归分类。正则参数 C 从 {0.01, 0.03, 0.1, 0.3, 1, 3, 10} 中依据验证集选择。最终配置不进行全局黎曼对齐。

双教师目标函数为 L=CE+alpha_E*KL_E+alpha_R*KL_R，其中 CE 为真实标签交叉熵，KL_E 和 KL_R分别传递 4 s 神经教师和黎曼教师的类别分布。主实验设置 alpha_E=0.4、alpha_R=0.4、T=1.0。

### 2.4 自适应时间窗候选

2a 使用类别感知 MI-active 检测器，以 0.5 s 窗口和 0.25 s 步长扫描。早期窗口标记为非 MI，稳定 MI 窗口使用 trial 类别，过渡窗口不参与训练。检测阈值和连续阳性窗口数量由训练会话验证数据确定，并在五随机种子路由实验前固定。

2b 使用 EEGNetV4 检测器，以 1 s 窗口和 0.25 s 步长扫描提示后 0-2 s。置信度在相邻 3 个窗口上平滑，选择峰值置信度对应的起点并向后截取 2 s。两套检测器反映数据集在通道数和任务结构上的差异，本文不将其宣称为统一的起始点检测算法。

### 2.5 验证集控制全局路由

每个被试和随机种子均训练固定 CE、自适应 CE、固定双教师 KD、自适应双教师 KD 四个候选。所有候选在同一验证集上评估，选择准确率最高者。完全相同时采用保守优先级：固定 CE、固定双教师、自适应 CE、自适应双教师。选择完成后，才在评估会话上测试所选路线。

固定路线在部署时只运行 Compact EEGNet；自适应路线运行时间检测器后再运行 Compact EEGNet。无论选择哪条路线，两个教师均仅在训练期使用。

### 2.6 优化、指标与统计检验

神经候选最多训练 300 epoch，batch size=32，学习率 0.001，权重衰减 0.0001，早停 patience=100。报告准确率、macro-F1 和 Cohen's kappa。先对每个被试的 5 个随机种子取均值，再在 9 个被试均值上进行配对 t 检验和 Wilcoxon 符号秩检验。涉及多条外部基线时，在每个数据集内进行 Holm 校正。

对比方法包括 2 s/4 s EEGNetV4、OAS 黎曼切空间逻辑回归和 CSP+收缩 LDA，均采用相同的被试内跨会话五随机种子协议。ShallowConvNet 和 DeepConvNet 尚未纳入完成的实验矩阵。

### 2.7 等总 KD 权重消融

为区分双教师互补与总蒸馏权重增大造成的影响，新增两组等权对照。总 KD 权重为 0.4 时，比较神经单教师 0.4、黎曼单教师 0.4 和双教师 0.2+0.2；总 KD 权重为 0.8 时，比较神经单教师 0.8、黎曼单教师 0.8 和双教师 0.4+0.4。每个配置均在两个数据集的 9 名被试和 5 个随机种子上运行。

### 2.8 路由稳健性、复杂度与类别分析

为评估路线选择对单次验证划分的敏感性，进行了留一 seed 的跨 seed 共识分析：对每个被试和待评估seed，仅使用其余 4 个 seed 的验证集平均准确率选择路线，再读取被留出 seed 的测试结果。该分析不使用测试集选路，但由于不同 seed 的训练子集存在重叠，应视为稳健性诊断而非严格的 trial 级嵌套交叉验证。

推理复杂度统计学生和检测器的可训练参数、卷积/全连接 MACs，并在 CPU 与 RTX 3070 Ti 上测量batch=1 的每 trial 前向延迟。类别分析使用 45 次运行汇总的混淆计数计算 precision、recall 和 F1。

## 3 结果

### 3.1 BCI Competition IV-2a 主结果

全局路由达到 59.31% 准确率、58.42% macro-F1 和 0.4575 kappa；固定 2 s CE 分别为 56.72%、55.50% 和 0.4229。准确率提高 2.59 个百分点，9 名被试中 8 名提高、1 名下降。配对 t 检验p=0.0055，Wilcoxon p=0.0078；若仅对两个数据集的主比较进行 Holm 校正，2a 仍显著。

固定双教师候选达到 58.47%。路由进一步提高 0.84 个百分点，但路由与固定双教师的差异不显著（p=0.1741）。因此，路由收益主要来自利用被试差异并避免统一采用较差候选，而不是相对最强单一候选产生大幅提升。

**表 2 五随机种子候选策略与路由表现**

| 数据集 | 方法 | 时间窗 | 准确率 | Macro-F1 | Kappa |
| --- | --- | --- | --- | --- | --- |
| 2a | 固定 CE | 0-2 s | 0.5672 | 0.5550 | 0.4229 |
| 2a | 自适应 CE | 选择的 2 s | 0.5481 | 0.5350 | 0.3974 |
| 2a | 固定双教师 KD | 0-2 s | 0.5847 | 0.5724 | 0.4463 |
| 2a | 自适应双教师 KD | 选择的 2 s | 0.5353 | 0.5224 | 0.3805 |
| 2a | 全局路由 | 选择的路线 | 0.5931 | 0.5842 | 0.4575 |
| 2b | 固定 CE | 0-2 s | 0.7634 | 0.7566 | 0.5269 |
| 2b | 自适应 CE | 选择的 2 s | 0.7722 | 0.7696 | 0.5445 |
| 2b | 固定双教师 KD | 0-2 s | 0.7677 | 0.7610 | 0.5354 |
| 2b | 自适应双教师 KD | 选择的 2 s | 0.7744 | 0.7703 | 0.5488 |
| 2b | 全局路由 | 选择的路线 | 0.7737 | 0.7694 | 0.5473 |

### 3.2 BCI Competition IV-2b 跨数据集结果

2b 上，全局路由达到 77.37% 准确率、76.94% macro-F1 和 0.5473 kappa；固定 2 s CE 为76.34%、75.66% 和 0.5269。准确率提高 1.02 个百分点，9 名被试中 7 名提高，但未达到显著性（配对 t 检验 p=0.0720，Wilcoxon p=0.0703）。

自适应双教师在不经过路线选择时的平均准确率为 77.44%，略高于路由后的 77.37%。这说明有限验证数据上的最佳路线并不一定最大化测试均值，2b 仅支持方向一致的外部验证，而不能证明稳定增益。

### 3.3 神经教师与黎曼教师基础消融

2a 上，CE、神经单教师 0.4、黎曼单教师 0.4、双教师 0.4+0.4 的准确率分别为 56.72%、57.42%、56.55% 和 58.47%。双教师相对 CE 提高 1.75 个百分点（t 检验 p=0.0122；Wilcoxon p=0.0156），而两个单教师相对 CE 均不显著。

2b 上，四种配置分别达到 76.35%、76.67%、76.33% 和 76.77%。双教师相对 CE 仅提高 0.42 个百分点且不显著（p=0.4574）。较少类别和仅 3 个通道可能降低了附加教师知识的边际价值。

**表 3a 基础教师消融**

| 数据集 | 监督配置 | alpha_E | alpha_R | 准确率 | 相对 CE |
| --- | --- | --- | --- | --- | --- |
| 2a | CE | 0 | 0 | 0.5672 | -- |
| 2a | 神经单教师 | 0.4 | 0 | 0.5742 | +0.0069 |
| 2a | 黎曼单教师 | 0 | 0.4 | 0.5655 | -0.0017 |
| 2a | 双教师 | 0.4 | 0.4 | 0.5847 | +0.0175 |
| 2b | CE | 0 | 0 | 0.7635 | -- |
| 2b | 神经单教师 | 0.4 | 0 | 0.7667 | +0.0032 |
| 2b | 黎曼单教师 | 0 | 0.4 | 0.7633 | -0.0002 |
| 2b | 双教师 | 0.4 | 0.4 | 0.7677 | +0.0042 |

**表 3b 基础教师消融配对检验**

| 数据集 | 比较 | 差值 | 胜/平/负 | dz | t 检验 p | Wilcoxon p |
| --- | --- | --- | --- | --- | --- | --- |
| 2a | 双教师 vs CE | +0.0175 | 7/1/1 | 1.07 | 0.0122 | 0.0156 |
| 2a | 双教师 vs 神经 | +0.0106 | 7/1/1 | 1.01 | 0.0163 | 0.0391 |
| 2a | 双教师 vs 黎曼 | +0.0192 | 7/1/1 | 0.76 | 0.0510 | 0.0547 |
| 2a | 神经 vs CE | +0.0069 | 5/0/4 | 0.32 | 0.3603 | 0.5703 |
| 2a | 黎曼 vs CE | -0.0017 | 6/1/2 | -0.08 | 0.8057 | 0.7422 |
| 2b | 双教师 vs CE | +0.0042 | 6/0/3 | 0.26 | 0.4574 | 0.4258 |
| 2b | 双教师 vs 神经 | +0.0010 | 7/0/2 | 0.07 | 0.8472 | 0.1523 |
| 2b | 双教师 vs 黎曼 | +0.0044 | 7/0/2 | 0.29 | 0.4173 | 0.2031 |
| 2b | 神经 vs CE | +0.0032 | 6/0/3 | 0.29 | 0.4060 | 0.3789 |
| 2b | 黎曼 vs CE | -0.0002 | 5/2/2 | -0.03 | 0.9338 | 0.9375 |

### 3.4 等总 KD 权重消融

等权对照进一步检验了教师互补性。2a 上，总权重 0.4 时，双教师 0.2+0.2 达到 58.52%，高于神经单教师 0.4 的 57.42% 和黎曼单教师 0.4 的 56.55%。双教师相对黎曼单教师差异显著（p=0.0192，Wilcoxon p=0.0273），相对神经单教师接近显著（p=0.0567）。

总权重 0.8 时，2a 的双教师 0.4+0.4 为 58.47%，神经单教师 0.8 为 58.03%，黎曼单教师 0.8为 56.89%，但两项配对差异均不显著。双教师 0.2+0.2 与 0.4+0.4 几乎相同（58.52% vs 58.47%，p=0.8919），表明简单增大总 KD 权重没有额外收益。

2b 上，双教师 0.2+0.2 为 76.81%，双教师 0.4+0.4 为 76.77%，神经单教师 0.4/0.8 为76.67%/76.69%，黎曼单教师 0.4/0.8 为 76.33%/75.52%。所有等权双教师与单教师比较均不显著。

**表 4 等总 KD 权重性能比较**

| 数据集 | 配置 | 总 KD 权重 | 准确率 | Macro-F1 | Kappa |
| --- | --- | --- | --- | --- | --- |
| 2a | 双教师 0.2+0.2 | 0.4 | 0.5852 | 0.5732 | 0.4469 |
| 2a | 神经单教师 0.4 | 0.4 | 0.5742 | 0.5620 | 0.4322 |
| 2a | 黎曼单教师 0.4 | 0.4 | 0.5655 | 0.5536 | 0.4207 |
| 2a | 双教师 0.4+0.4 | 0.8 | 0.5847 | 0.5724 | 0.4463 |
| 2a | 神经单教师 0.8 | 0.8 | 0.5803 | 0.5682 | 0.4404 |
| 2a | 黎曼单教师 0.8 | 0.8 | 0.5689 | 0.5555 | 0.4252 |
| 2b | 双教师 0.2+0.2 | 0.4 | 0.7681 | 0.7643 | 0.5363 |
| 2b | 神经单教师 0.4 | 0.4 | 0.7667 | 0.7629 | 0.5334 |
| 2b | 黎曼单教师 0.4 | 0.4 | 0.7633 | 0.7564 | 0.5266 |
| 2b | 双教师 0.4+0.4 | 0.8 | 0.7677 | 0.7610 | 0.5354 |
| 2b | 神经单教师 0.8 | 0.8 | 0.7669 | 0.7623 | 0.5338 |
| 2b | 黎曼单教师 0.8 | 0.8 | 0.7552 | 0.7474 | 0.5104 |

**表 5 等总 KD 权重配对检验**

| 数据集 | 比较 | 差值 | 胜/平/负 | t 检验 p | Wilcoxon p |
| --- | --- | --- | --- | --- | --- |
| 2a | 双 0.2+0.2 vs 神经 0.4 | +0.0110 | 7/0/2 | 0.0567 | 0.0547 |
| 2a | 双 0.2+0.2 vs 黎曼 0.4 | +0.0197 | 7/0/2 | 0.0192 | 0.0273 |
| 2a | 双 0.4+0.4 vs 神经 0.8 | +0.0044 | 7/0/2 | 0.3091 | 0.2852 |
| 2a | 双 0.4+0.4 vs 黎曼 0.8 | +0.0158 | 5/0/4 | 0.2067 | 0.4961 |
| 2b | 双 0.2+0.2 vs 神经 0.4 | +0.0014 | 5/1/3 | 0.3581 | 0.3828 |
| 2b | 双 0.2+0.2 vs 黎曼 0.4 | +0.0048 | 7/0/2 | 0.0724 | 0.1289 |
| 2b | 双 0.4+0.4 vs 神经 0.8 | +0.0008 | 5/1/3 | 0.8574 | 0.7422 |
| 2b | 双 0.4+0.4 vs 黎曼 0.8 | +0.0125 | 7/0/2 | 0.1934 | 0.1367 |

### 3.5 与神经及经典基线的比较

2a 上，全局路由高于 EEGNetV4 2 s（59.31% vs 58.09%）、黎曼 TS-LogReg 2 s（56.94%）和CSP-LDA 2 s（55.30%）。其中，仅相对 CSP-LDA 2 s 的优势在 Holm 校正后仍显著。完整时间窗中，黎曼 TS-LogReg 4 s 以 65.25% 最强，路由并未超过该方法。

2b 上，全局路由略高于 EEGNetV4 2 s（77.37% vs 77.14%），但差异不显著。EEGNetV4 4 s 达到78.95%，是该数据集最强的已评估方法。路由的贡献因此应定位为短窗策略选择，而非不限制时间窗时的最高分类准确率。

路由策略的平均有效信号终点为 2a 的 2.11 s 和 2b 的 2.20 s，明显短于 4 s 参考，但该指标只描述所需 EEG 样本终点，不等同于在线系统的完整决策延迟。

**表 6 路由主比较的配对检验**

| 数据集 | 比较 | 准确率差值 | 胜/平/负 | t 检验 p | Wilcoxon p |
| --- | --- | --- | --- | --- | --- |
| 2a | 路由 vs 固定 2 s CE | +0.0259 | 8/0/1 | 0.0055 | 0.0078 |
| 2a | 路由 vs 固定双教师 | +0.0084 | 6/0/3 | 0.1741 | 0.2031 |
| 2a | 路由 vs Compact 4 s CE | +0.0096 | 6/0/3 | 0.6466 | 0.6523 |
| 2b | 路由 vs 固定 2 s CE | +0.0102 | 7/0/2 | 0.0720 | 0.0703 |
| 2b | 路由 vs 固定双教师 | +0.0059 | 6/0/3 | 0.4175 | 0.3594 |
| 2b | 路由 vs Compact 4 s CE | -0.0069 | 3/0/6 | 0.4160 | 0.4961 |

**表 7 五随机种子神经与经典基线**

| 数据集 | 方法 | 窗口 | 准确率 | Macro-F1 | Kappa |
| --- | --- | --- | --- | --- | --- |
| 2a | 全局路由 | 选择的 2 s | 0.5931 | 0.5842 | 0.4575 |
| 2a | EEGNetV4 CE | 2 s | 0.5809 | 0.5666 | 0.4412 |
| 2a | 黎曼 TS-LogReg | 2 s | 0.5694 | 0.5570 | 0.4259 |
| 2a | CSP-LDA | 2 s | 0.5530 | 0.5270 | 0.4040 |
| 2a | Compact EEGNet CE | 4 s | 0.5836 | 0.5749 | 0.4448 |
| 2a | EEGNetV4 CE | 4 s | 0.6025 | 0.5894 | 0.4700 |
| 2a | 黎曼 TS-LogReg | 4 s | 0.6525 | 0.6420 | 0.5367 |
| 2a | CSP-LDA | 4 s | 0.6029 | 0.5726 | 0.4705 |
| 2b | 全局路由 | 选择的 2 s | 0.7737 | 0.7694 | 0.5473 |
| 2b | EEGNetV4 CE | 2 s | 0.7714 | 0.7676 | 0.5427 |
| 2b | 黎曼 TS-LogReg | 2 s | 0.7159 | 0.7080 | 0.4317 |
| 2b | CSP-LDA | 2 s | 0.7136 | 0.7076 | 0.4272 |
| 2b | Compact EEGNet CE | 4 s | 0.7806 | 0.7763 | 0.5612 |
| 2b | EEGNetV4 CE | 4 s | 0.7895 | 0.7852 | 0.5790 |
| 2b | 黎曼 TS-LogReg | 4 s | 0.7297 | 0.7189 | 0.4593 |
| 2b | CSP-LDA | 4 s | 0.7303 | 0.7197 | 0.4606 |

**表 8a 2a 外部基线配对比较**

| 对比方法 | 差值 | 胜/平/负 | t 检验 p（Holm） | Wilcoxon p（Holm） | dz |
| --- | --- | --- | --- | --- | --- |
| EEGNetV4 2 s CE | +0.0123 | 7/0/2 | 0.1981 (0.5943) | 0.2031 (0.8125) | 0.47 |
| 黎曼 TS-LogReg 2 s | +0.0237 | 6/0/3 | 0.1443 (0.5772) | 0.2383 (0.8125) | 0.54 |
| CSP-LDA 2 s | +0.0401 | 9/0/0 | 0.0069 (0.0411) | 0.0039 (0.0234) | 1.20 |
| CSP-LDA 4 s | -0.0097 | 5/0/4 | 0.7148 (1.0000) | 1.0000 (1.0000) | -0.13 |
| EEGNetV4 4 s CE | -0.0093 | 5/0/4 | 0.5901 (1.0000) | 1.0000 (1.0000) | -0.19 |
| 黎曼 TS-LogReg 4 s | -0.0594 | 1/0/8 | 0.0108 (0.0542) | 0.0195 (0.0977) | -1.10 |

**表 8b 2b 外部基线配对比较**

| 对比方法 | 差值 | 胜/平/负 | t 检验 p（Holm） | Wilcoxon p（Holm） | dz |
| --- | --- | --- | --- | --- | --- |
| EEGNetV4 2 s CE | +0.0023 | 4/1/4 | 0.4885 (0.4885) | 0.5781 (0.5781) | 0.24 |
| 黎曼 TS-LogReg 2 s | +0.0578 | 8/0/1 | 0.0228 (0.0756) | 0.0195 (0.0781) | 0.94 |
| CSP-LDA 2 s | +0.0601 | 7/0/2 | 0.0189 (0.0756) | 0.0195 (0.0781) | 0.98 |
| CSP-LDA 4 s | +0.0433 | 8/0/1 | 0.0120 (0.0600) | 0.0078 (0.0469) | 1.08 |
| EEGNetV4 4 s CE | -0.0158 | 2/0/7 | 0.1324 (0.2648) | 0.2031 (0.4062) | -0.56 |
| 黎曼 TS-LogReg 4 s | +0.0440 | 8/0/1 | 0.0058 (0.0345) | 0.0117 (0.0586) | 1.24 |

### 3.6 路线使用与被试异质性

每个数据集共 45 次路由运行。2a 中固定路线占 64.4%，2b 中占 62.2%；双教师路线分别占 48.9%和 46.7%。没有一个候选策略占据绝对优势，说明时间窗与教师效用均依赖被试和随机划分。

仅用于诊断的测试集 oracle 分析显示：即使事后选择最佳 2 s 候选，2a 的 S1、S2、S3 和 2b 的S9 仍低于 4 s CE，提示短窗信息上限；2b 的 S1、S7、S8 存在可匹配或超过 4 s 的 2 s 候选，但验证集未稳定选中，提示路线选择误差。oracle 结果未用于正式选模或主结果。

**表 9 验证集路线选择次数**

| 数据集 | 固定 CE | 固定双教师 | 自适应 CE | 自适应双教师 | 固定/自适应 |
| --- | --- | --- | --- | --- | --- |
| BCI IV-2a | 14 | 15 | 9 | 7 | 29/16 |
| BCI IV-2b | 17 | 11 | 7 | 10 | 28/17 |

### 3.7 跨 seed 路由稳健性

留一 seed 共识路由在 2a 上达到 57.11%，固定 CE 为 56.72%，差值仅 0.39 个百分点且不显著（p=0.7127；Wilcoxon p=0.7422）。共识路由还显著低于原单次验证划分路由的 59.31%（p=0.0107）。

2b 上，共识路由为 77.04%，固定 CE 为 76.34%，提高 0.69 个百分点但不显著（p=0.1787）；与原路由 77.37% 的差异也不显著。该结果表明当前路由选择在 2a 尤其依赖单次验证划分，正式结论应强调验证控制的探索性收益，并将 trial 级嵌套交叉验证作为后续必要验证。

**表 10 跨 seed 共识路由稳健性**

| 数据集 | 跨 seed 共识 | 固定 CE | 原全局路由 | 共识 vs CE p | Wilcoxon p |
| --- | --- | --- | --- | --- | --- |
| 2a | 0.5711 | 0.5672 | 0.5931 | 0.7127 | 0.7422 |
| 2b | 0.7704 | 0.7634 | 0.7737 | 0.1787 | 0.4375 |

### 3.8 推理复杂度与逐类别表现

固定 2 s Compact EEGNet 在 2a/2b 上分别包含 2,420/1,634 个执行参数和 5.89/0.86 MMAC。加入检测器后分别为 3,526/3,500 个参数和 11.25/4.83 MMAC。RTX 3070 Ti 上固定路线延迟约为1.01/1.42 ms，自适应路线约为 2.36/3.21 ms；教师不参与推理。

2a 中双脚类别最难：CE 的 F1 为 0.468，双教师提高至 0.492。左手、右手和舌部 F1 也分别由0.593、0.598、0.595 提高至 0.610、0.626、0.599。2b 中双教师主要改善右手 recall（0.697 提高至 0.719），左右手 F1 分别为 0.782 和 0.758。

**表 11 推理复杂度与延迟**

| 数据集 | 部署路线 | 参数量 | MACs | CPU ms/trial | GPU ms/trial |
| --- | --- | --- | --- | --- | --- |
| 2a | 固定 2 s Compact | 2,420 | 5.89 M | 1.12 | 1.01 |
| 2a | 检测器+Compact | 3,526 | 11.25 M | 3.88 | 2.36 |
| 2b | 固定 2 s Compact | 1,634 | 0.86 M | 1.48 | 1.42 |
| 2b | 检测器+Compact | 3,500 | 4.83 M | 5.31 | 3.21 |

**表 12 逐类别 F1**

| 数据集 | 类别 | CE F1 | 双教师 F1 | 变化 |
| --- | --- | --- | --- | --- |
| 2a | 左手 | 0.593 | 0.610 | +0.018 |
| 2a | 右手 | 0.598 | 0.626 | +0.027 |
| 2a | 双脚 | 0.468 | 0.492 | +0.024 |
| 2a | 舌部 | 0.595 | 0.599 | +0.004 |
| 2b | 左手 | 0.782 | 0.782 | +0.000 |
| 2b | 右手 | 0.749 | 0.758 | +0.009 |

## 4 讨论

本文的主要结果是：验证集控制路由在四分类 2a 上显著提高了 Compact EEGNet 的短窗解码性能，但在二分类 2b 上仅获得较小且不显著的增益。该结果并非由单个普遍最优的专家产生，而是来自候选策略的被试依赖性以及验证集选择。

等总权重消融使双教师机制的解释更严格。2a 上，双教师 0.2+0.2 在总 KD 权重与单教师 0.4 完全一致时仍达到最高均值，并显著高于黎曼单教师、接近显著高于神经单教师。这排除了“仅因双教师总权重更大”这一简单解释，并支持神经决策边界与协方差几何信息具有一定互补性。不过，相对神经单教师的证据尚未越过 0.05 阈值，因此不宜将教师互补表述为已被完全证实。

2b 的等权结果显示，固定 2 s 基线已较高，神经单教师接近双教师，而提高黎曼教师权重反而下降。这说明三通道二分类任务中协方差教师的附加信息有限，并且教师权重需要根据数据集与被试校准。

跨 seed 共识分析揭示了比平均准确率更关键的问题：原路由在 2a 上的显著增益未能在更保守的路线选择方式下保持。当前路由比较四个候选，却只依赖一个 20% 验证划分，容易放大验证噪声。因此，论文可以报告原路由的显著结果，但必须同时报告稳健性诊断，并避免将其解释为已稳定泛化的最终路由器。

与 4 s 模型的比较支持谨慎的实际意义。2a 路由数值上略高于 4 s Compact EEGNet，2b 略低，均未显著。证据更适合支持“使用更短信号片段保持接近性能”，而不是“2 s 普遍优于 4 s”。若要强化该主张，应预先设定非劣效界值并开展正式非劣效检验。

复杂度结果表明训练期教师不会增加部署负担。固定路线只有约 1 ms 的 GPU 前向延迟，自适应路线也低于4 ms。实际在线延迟仍由 2 s EEG 采集时间主导，但模型计算不是主要瓶颈。

类别分析显示，2a 中双教师对最困难的双脚类别和右手类别改善较明显，这与黎曼协方差对传感运动空间关系的敏感性相符。由于这里只分析汇总混淆计数，未来仍需对每个被试的类别改变量与教师置信度进行配对统计。

下一步最重要的改进不是继续增加候选，而是在训练会话内部使用严格的 K 折折外预测估计候选效用，并设置安全阈值：只有当自适应或教师路线在多个内部折上稳定优于固定 CE 时才启用，否则回退固定路线。

### 4.1 局限性

- 路由器在分别训练的候选之间进行被试级选择，不是端到端或 trial 级混合专家模型。
- 主路由结果使用单个 20% 验证划分；跨 seed 共识分析显示其稳定性有限。
- 2a 与 2b 使用不同的检测器目标和结构，降低了跨数据集方法的一致性。
- 跨 seed 共识仍不是严格的 trial 级嵌套交叉验证，因为不同 seed 的训练子集存在重叠。
- 显著提升主要出现在 2a；2b 的提升较小且不显著。
- 方法未显著超过固定双教师或完整时间窗强基线，不能宣称不受时间窗限制的 SOTA。
- 当前实验为被试内训练，不能直接推断跨被试迁移或无校准部署能力。
- 尚未完成 ShallowConvNet、DeepConvNet 等卷积基线以及真实异步在线 BCI 测试。

## 5 结论

本文将短窗 MI-EEG 解码表述为验证集控制的时间窗与监督策略路由问题。9 被试、5 随机种子实验表明，原全局路由在 BCI IV-2a 上显著优于固定 2 s Compact EEGNet，在 2b 上获得方向一致但不显著的提高。等总权重消融进一步表明，2a 的双教师优势不能仅由总 KD 权重增大解释，并支持神经教师与黎曼教师存在一定互补性。与此同时，跨 seed 共识路由未显著超过固定 CE，说明当前路线选择仍受验证划分影响。推理阶段仅保留学生和可选检测器，计算延迟较低。综合而言，该方法为短窗训练期知识迁移提供了有价值的证据，但在形成稳定的通用路由框架之前，还需要严格的折外选择、统一检测器和前瞻性在线验证。

## 声明

**基金资助：** [填写基金声明。]

**利益冲突：** 作者声明不存在利益冲突。

**伦理审批：** 本研究使用公开、去标识化数据集。

**代码可用性：** https://github.com/zfqa/eegnetfgl

## 参考文献

参考文献保留原始英文书目信息。

1. Altaheri H, Muhammad G, Alsulaiman M et al (2023) Deep learning techniques for classification of electroencephalogram (EEG) motor imagery (MI) signals: a review. Neural Comput Appl 35:14681-14722. https://doi.org/10.1007/s00521-023-08542-7
2. Ang KK, Guan C (2018) EEG-based strategies for stroke rehabilitation. In: Nam CS, Nijholt A, Lotte F (eds) Brain-computer interfaces handbook. CRC Press, pp 553-574
3. Barachant A, Bonnet S, Congedo M, Jutten C (2012) Multiclass brain-computer interface classification by Riemannian geometry. IEEE Trans Biomed Eng 59:920-928. https://doi.org/10.1109/TBME.2011.2172210
4. Barachant A, Bonnet S, Congedo M, Jutten C (2013) Classification of covariance matrices using a Riemannian-based kernel for BCI applications. Neurocomputing 112:172-178. https://doi.org/10.1016/j.neucom.2012.12.039
5. Blankertz B, Tomioka R, Lemm S, Kawanabe M, Muller KR (2008) Optimizing spatial filters for robust EEG single-trial analysis. IEEE Signal Process Mag 25:41-56. https://doi.org/10.1109/MSP.2008.4408441
6. Brunner C, Leeb R, Muller-Putz G, Schlogl A, Pfurtscheller G (2010) BCI Competition 2008: Graz data set A. Institute for Knowledge Discovery, Graz University of Technology
7. Chen Y, Wiesel A, Eldar YC, Hero AO (2010) Shrinkage algorithms for MMSE covariance estimation. IEEE Trans Signal Process 58:5016-5029. https://doi.org/10.1109/TSP.2010.2053029
8. Congedo M, Barachant A, Bhatia R (2017) Riemannian geometry for EEG-based brain-computer interfaces; a primer and a review. BCI 4:1-39. https://doi.org/10.1080/2326263X.2017.1297192
9. Cui J, Lan Z, Liu Y et al (2022) A compact and interpretable convolutional neural network for cross-subject driver drowsiness recognition from single-channel EEG. IEEE Trans Neural Syst Rehabil Eng 30:286-296. https://doi.org/10.1109/TNSRE.2022.3147419
10. Ding Y, Robinson N, Zeng Q et al (2022) TSception: a deep learning framework for emotion detection using EEG. In: Proc Int Joint Conf Neural Networks (IJCNN), pp 1-8. https://doi.org/10.1109/IJCNN55064.2022.9967692
11. Hinton G, Vinyals O, Dean J (2015) Distilling the knowledge in a neural network. arXiv:1503.02531. https://doi.org/10.48550/arXiv.1503.02531
12. Jayaram V, Barachant A (2018) MOABB: trustworthy algorithm benchmarking for BCIs. J Neural Eng 15:066011. https://doi.org/10.1088/1741-2552/aadea0
13. Kalunga EK, Chevallier S, Barre Q, Djouani K, Monacelli E, Hamam Y (2016) Online SSVEP-based BCI using Riemannian geometry. Neurocomputing 191:55-68. https://doi.org/10.1016/j.neucom.2016.01.007
14. Lawhern VJ, Solon AJ, Waytowich NR, Gordon SM, Hung CP, Lance BJ (2018) EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces. J Neural Eng 15:056013. https://doi.org/10.1088/1741-2552/aace8c
15. Lotte F, Bougrain L, Cichocki A et al (2018) A review of classification algorithms for EEG-based brain-computer interfaces: a 10 year update. J Neural Eng 15:031005. https://doi.org/10.1088/1741-2552/aab2f2
16. Lotte F, Jeunet C (2015) Towards improved BCI spellers using a co-adaptive calibration of the P300 speller with RSVP and visual feedback. J Neural Eng 12:066007
17. Mane R, Chew E, Chua K et al (2021) FBCNet: a multi-view convolutional neural network for brain-computer interface. arXiv:2104.01233
18. Meisheri H, Ramrao N, Mitra S, Saha S (2022) Common spatial pattern for EEG based brain-computer interface: an implementation review. In: Proc Int Conf Comput Intell, pp 1-6
19. Pfurtscheller G, Neuper C (2001) Motor imagery and direct brain-computer communication. Proc IEEE 89:1123-1134. https://doi.org/10.1109/5.939829
20. Roy Y, Banville H, Albuquerque I, Gramfort A, Falk TH, Faubert J (2019) Deep learning-based electroencephalography analysis: a systematic review. J Neural Eng 16:051001. https://doi.org/10.1088/1741-2552/ab260c
21. Schirrmeister RT, Springenberg JT, Fiederer LDJ et al (2017) Deep learning with convolutional neural networks for EEG decoding and visualization. Hum Brain Mapp 38:5391-5420. https://doi.org/10.1002/hbm.23730
22. Song Y, Zheng Q, Liu B, Gao X (2023) EEG conformer: convolutional transformer for EEG decoding and visualization. IEEE Trans Neural Syst Rehabil Eng 31:710-719. https://doi.org/10.1109/TNSRE.2022.3230250
23. Tangermann M, Muller KR, Aertsen A et al (2012) Review of the BCI Competition IV. Front Neurosci 6:55. https://doi.org/10.3389/fnins.2012.00055
24. Tang J, Liu C, Li Q (2020) EEG feature distillation for motor imagery classification. In: Proc IEEE Int Conf Bioinformatics and Biomedicine (BIBM), pp 1680-1683
25. Wang J, Li S, Li R, Shi Y (2022) EEG motor imagery classification with multi-view convolutional neural network. Biomed Signal Process Control 77:103778
26. Wei Q, Wang Y, Gao X, Gao S (2022) Amplitude and phase coupling in steady-state motion visual evoked potentials for BCI. J Neural Eng 19:026002
27. Yger F, Berar M, Lotte F (2016) Riemannian approaches in brain-computer interfaces: a review. IEEE Trans Neural Syst Rehabil Eng 25:1753-1762. https://doi.org/10.1109/TNSRE.2016.2627016
28. Zhang L, Wang J, Chen B et al (2023) Temporal convolutional network for motor imagery EEG decoding with attention mechanism. Biomed Signal Process Control 81:104432
29. Zhang X, Yao L, Wang X, Monaghan J, McAlpine D (2022) A survey on deep learning-based non-invasive brain signals: recent advances and new frontiers. J Neural Eng 18:031002
30. Zuo C, Jin J, Yin Y et al (2023) Novel hybrid deep learning model for motor imagery EEG classification based on capsule network. Cogn Neurodyn 17:107-121
