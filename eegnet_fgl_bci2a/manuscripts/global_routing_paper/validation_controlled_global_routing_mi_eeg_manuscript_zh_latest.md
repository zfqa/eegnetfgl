# 面向短窗运动想象脑电解码的黎曼引导训练与验证集控制路由

**作者：** [作者姓名]
**单位：** [院系、机构、城市、国家]
**通讯作者：** [姓名和邮箱]

## 摘要

缩短运动想象脑电（MI-EEG）的分析窗口有助于减少分类前所需的信号长度，但固定时间窗和单一监督方式无法稳定适用于所有被试。本文围绕 2 s 短窗解码研究两类训练期辅助策略：在 BCI Competition IV-2a 上，采用验证集控制全局路由，在固定窗交叉熵、自适应窗交叉熵、固定窗双教师知识蒸馏和自适应窗双教师知识蒸馏之间进行被试级选择；双教师分别为冻结的 4 s Compact EEGNet 与基于 4 s 协方差切空间逻辑回归的黎曼教师。在 BCI Competition IV-2b 上，由于二分类三通道任务中双教师未稳定优于固定 2 s CE，本文将双教师候选替换为 Riemann KD + spatial loss，以黎曼教师软标签和空间结构约束引导 2 s Compact EEGNet，并比较固定窗、自适应窗及验证集路由。所有候选策略仅依据训练会话中的验证集选择，评估会话始终隔离。本文在 BCI Competition IV-2a 和 2b 的 9 名被试上采用 5 个随机种子开展跨会话实验。2a 上，全局路由将固定 2 s 交叉熵准确率由 56.72% 提升至 59.31%（配对 t 检验 p=0.0055，Wilcoxon p=0.0078）。2b 上，自适应 Riemann KD + spatial loss 取得最高平均准确率 77.12%，全局路由达到 77.01%，相对固定 2 s CE 的 76.34% 提高 0.67 个百分点。2b 的提升幅度小于 2a，应解释为跨数据集的小幅正向验证，而非显著突破。等总权重分析表明，2a 上双教师具有较稳定的互补趋势，而 2b 上神经教师和黎曼教师的组合未产生额外收益，直接引入黎曼空间结构约束更合适。推理阶段不调用教师，在 RTX 3070 Ti 上，固定策略每 trial 约需 1.01-1.42 ms，自适应策略约需 2.36-3.21 ms。结果支持将该方法定位为短窗策略选择与训练期知识迁移框架，而非对完整时间窗解码的普遍替代。

**关键词：** 脑机接口；脑电；运动想象；时间路由；知识蒸馏；黎曼几何；EEGNet

## 1 引言

运动想象脑机接口通过脑电图推断想象运动，可用于交流、康复和设备控制。许多离线解码研究使用提示后数秒的 EEG。较长时间窗提供更多时序证据，但也会延长系统在作出决策前必须等待的信号时长。因此，短窗解码对响应式系统具有吸引力，不过早期片段可能仅包含不完整的事件相关去同步或同步模式。

紧凑卷积网络与基于协方差的黎曼分类器为 MI-EEG 提供了互补视角。EEGNet 从波形中直接学习时间滤波和深度可分离空间滤波；黎曼方法则以对称正定通道协方差矩阵表示每个 trial，并使用流形距离或切空间特征进行分类。部署时希望模型足够紧凑，而训练阶段的完整时间窗协方差模型可提供较稳健的几何信息。

知识蒸馏可用具有额外信息的教师监督受限学生。本研究关注“时间特权”场景：教师在训练时观察 4 s 信号，学生在训练和推理时仅观察 2 s。初步实验显示，神经教师、黎曼教师与检测器选择的时间窗均具有明显的被试依赖性；对部分被试有益的策略也会对另一些被试造成负迁移。

因此，本文将短窗解码表述为训练期知识迁移与验证集控制策略选择问题。在 2a 上，对每个被试和随机种子分别训练固定窗 CE、自适应窗 CE、固定窗双教师 KD、自适应窗双教师 KD 四个完整候选模型，并只依据训练会话验证准确率选择一个候选。这里的“全局”表示每个被试和训练种子只选择一条路线，而不是对每个测试 trial 使用混合专家门控。在 2b 上，保留相同的验证集选择思想，但将教师候选改为固定/自适应 Riemann KD + spatial loss，以检验三通道二分类任务中时间窗选择和黎曼空间监督的组合是否仍有收益。测试时不调用教师，仅在自适应路线中使用时间检测器。

本文主要贡献如下：

- 提出面向短窗 MI-EEG 解码的训练期知识迁移框架，在复杂四分类任务中使用验证集控制全局路由。
- 构建由 4 s 神经 EEGNet 与 4 s 黎曼切空间逻辑回归共同监督 2 s Compact EEGNet 的双教师训练方案，并在二分类三通道任务中采用 Riemann KD + spatial loss 的单黎曼教师配置及验证集路由对照。
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

教师引导训练中，学生的 2 s 输入和教师的 4 s 输入均从同一个 trial 裁剪并在批处理前保持对应关系。标签和样本顺序不被打乱。教师只产生训练集和验证集软标签或空间结构目标，测试阶段仅调用 2 s 学生模型。测试会话不参与时间窗、阈值、超参数、模型 checkpoint 或路线选择。

### 2.3 Compact 学生与长窗教师

所有路由候选分类器均采用 PyTorch Compact EEGNet，输入形状为 [batch, 1, channels, samples]。模型使用 F1=8、深度倍增因子 D=2、F2=16、64 样本时间卷积核、ELU、平均池化和 dropout=0.5。两个数据集的 4 s 神经教师现已统一为 Compact EEGNet，并在同一被试的训练子集上训练后冻结。统一教师结构避免了旧版本中 2a 使用 Compact、2b 使用 EEGNetV4 所造成的方法口径差异。

黎曼教师使用 Oracle Approximating Shrinkage（OAS）估计原始 4 s trial 的协方差矩阵，映射至黎曼均值处的切空间后使用多项逻辑回归分类。正则参数 C 从 {0.01, 0.03, 0.1, 0.3, 1, 3, 10} 中通过训练会话内部验证选择，不使用测试会话。

2a 的双教师目标函数为 L=CE+alpha_E*KL_E+alpha_R*KL_R，其中 CE 为真实标签交叉熵，KL_E 和 KL_R 分别传递 4 s 神经教师与黎曼教师的类别分布。固定双教师主配置采用 alpha_E=0.4、alpha_R=0.4、T=1.0；所有教师只在训练阶段使用。

2b 的教师候选不再采用神经-黎曼双教师，而采用 Riemann KD + spatial loss。该配置使用同一 4 s 黎曼教师产生类别概率，并将黎曼协方差结构作为 student 空间特征学习的正则项。损失函数为 L=CE+lambda_R*KL_R+lambda_S*L_spatial，实验中 lambda_R=0.1，lambda_S=0.2，spatial source 为 pre-depthwise 表征。该设计用于检验三通道二分类任务中，直接约束空间结构是否比额外加入神经教师更稳定。

### 2.4 自适应时间窗候选与诊断

2a 使用类别感知 MI-active 检测器，以 0.5 s 窗口和 0.25 s 步长扫描。早期窗口标记为非 MI，稳定 MI 窗口使用 trial 类别，过渡窗口不参与训练。检测阈值和连续阳性窗口数量由训练会话验证数据确定，并在五随机种子路由实验前固定。

2b 的自适应窗口实验使用 EEGNetV4 检测器，以 1 s 窗口和 0.25 s 步长扫描提示后 0-2 s。置信度在相邻 3 个窗口上平滑，选择峰值置信度对应的起点并向后截取 2 s。该检测器用于构造自适应 CE 和自适应 Riemann KD + spatial loss 候选；其作用是验证时间窗选择是否能与黎曼空间监督互补，而不是在测试集上选择最优起点。

### 2.5 验证集控制全局路由

2a 中，每个被试和随机种子均训练固定 CE、自适应 CE、固定双教师 KD、自适应双教师 KD 四个候选。所有候选在同一验证集上评估，选择准确率最高者。完全相同时采用保守优先级：固定 CE、固定双教师、自适应 CE、自适应双教师。选择完成后，才在评估会话上测试所选路线。2b 中，候选改为固定 CE、自适应 CE、固定 Riemann KD + spatial loss 和自适应 Riemann KD + spatial loss；验证集路由同样只使用训练会话内的验证准确率。

固定路线在部署时只运行 Compact EEGNet；自适应路线运行时间检测器后再运行 Compact EEGNet。无论选择哪条路线，教师和黎曼空间结构目标均仅在训练期使用。

### 2.6 优化、指标与统计检验

神经候选最多训练 300 epoch，batch size=32，学习率 0.001，权重衰减 0.0001，早停 patience=100。报告准确率、macro-F1 和 Cohen's kappa。先对每个被试的 5 个随机种子取均值，再在 9 个被试均值上进行配对 t 检验和 Wilcoxon 符号秩检验。涉及多条外部基线时，在每个数据集内进行 Holm 校正。

对比方法包括 2 s/4 s EEGNetV4、OAS 黎曼切空间逻辑回归、CSP+收缩 LDA、ShallowConvNet 和 DeepConvNet，均采用相同的被试内跨会话五随机种子协议。ShallowConvNet 和 DeepConvNet 在两个数据集、两个时间窗、9 名被试和 5 个随机种子上完成复现。

### 2.7 等总 KD 权重消融

为区分双教师互补与总蒸馏权重增大造成的影响，新增两组等权对照。总 KD 权重为 0.4 时，比较神经单教师 0.4、黎曼单教师 0.4 和双教师 0.2+0.2；总 KD 权重为 0.8 时，比较神经单教师 0.8、黎曼单教师 0.8 和双教师 0.4+0.4。2a 的等权消融用于支持正式双教师路由主线；2b 的等权消融作为诊断结果，用于说明神经教师和双教师配置未优于固定 CE，因此 2b 主方法改用 Riemann KD + spatial loss。在 2b 上，另以 Riemann KD 权重 lambda_R 与空间结构损失权重 lambda_S 的和匹配总辅助权重，比较联合与单项监督；每个配置均使用 9 名被试和 5 个随机种子。

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
| 2b | 固定 CE | 0-2 s | 0.7634 ± 0.0120 | 0.7567 ± 0.0193 | 0.5268 ± 0.0239 |
| 2b | 固定 Riemann KD + spatial | 0-2 s | 0.7660 ± 0.0142 | 0.7627 ± 0.0152 | 0.5319 ± 0.0285 |
| 2b | 自适应 CE | 选择的 2 s | 0.7675 ± 0.0031 | 0.7637 ± 0.0029 | 0.5349 ± 0.0063 |
| 2b | 自适应 Riemann KD + spatial | 选择的 2 s | 0.7712 ± 0.0152 | 0.7681 ± 0.0157 | 0.5424 ± 0.0305 |
| 2b | 全局路由 | 选择的路线 | 0.7701 ± 0.0103 | 0.7665 ± 0.0118 | 0.5402 ± 0.0206 |

### 3.2 BCI Competition IV-2b 跨数据集结果

在 2b 上，固定 2 s CE 的平均准确率、macro-F1 和 kappa 分别为 76.34% ± 1.20%、75.67% ± 1.93% 和 0.5268 ± 0.0239。固定 Riemann KD + spatial loss 达到 76.60% ± 1.42%、76.27% ± 1.52% 和 0.5319 ± 0.0285；自适应 CE 达到 76.75% ± 0.31%、76.37% ± 0.29% 和 0.5349 ± 0.0063；自适应 Riemann KD + spatial loss 取得最高均值，准确率、macro-F1 和 kappa 分别为 77.12% ± 1.52%、76.81% ± 1.57% 和 0.5424 ± 0.0305。验证集控制全局路由达到 77.01% ± 1.03%、76.65% ± 1.18% 和 0.5402 ± 0.0206，相对固定 CE 的三项指标分别提高 0.67、0.98 个百分点和 0.0134。

因此，2b 上所有改进策略的均值均高于固定 CE，但提升幅度明显小于 2a。固定窗口下，Riemann KD + spatial loss 的增益仅为 0.26 个百分点；自适应窗口下，加入 Riemann KD + spatial loss 相对自适应 CE 进一步提高 0.37 个百分点。这说明黎曼空间监督在三通道二分类任务中仍有正向作用，但作用较弱。自适应 Riemann KD + spatial loss 的均值略高于全局路由 0.11 个百分点，原因在于全局路由根据验证集而非测试集选择候选，存在 validation-to-test selection noise。与此同时，全局路由的标准差低于自适应 Riemann KD + spatial loss（1.03% vs 1.52%），提示路由机制可能牺牲极小均值以降低跨 seed 波动。

**表 2b BCI IV-2b 候选策略均值与相对固定 CE 提升**

| 方法 | Acc | 相对 fixed CE |
| --- | --- | --- |
| fixed CE | 0.7634 ± 0.0120 | -- |
| fixed Riemann + spatial | 0.7660 ± 0.0142 | +0.0026 |
| adaptive CE | 0.7675 ± 0.0031 | +0.0041 |
| adaptive Riemann + spatial | 0.7712 ± 0.0152 | +0.0078 |
| global routing | 0.7701 ± 0.0103 | +0.0067 |

### 3.3 神经教师与黎曼教师基础消融

2a 上，固定 CE、神经单教师 0.4、黎曼单教师 0.4 和双教师 0.4+0.4 的准确率分别为 56.72%、57.42%、56.55% 和 58.47%。双教师相对 CE 提高 1.75 个百分点（t 检验 p=0.0122；Wilcoxon p=0.0156），两个单教师相对 CE 均不显著。

2b 的固定 CE、固定 Riemann KD + spatial loss、自适应 CE 和自适应 Riemann KD + spatial loss 准确率分别为 76.34%、76.60%、76.75% 和 77.12%。固定窗口下的黎曼空间监督仅带来 0.26 个百分点提升，自适应窗口下的提升为 0.37 个百分点。相同数据集上，旧双教师配置未稳定优于固定 CE。因此，2b 不支持双教师优于普通 Compact EEGNet 的结论，但支持在三通道二分类短窗解码中使用黎曼空间结构约束作为小幅正向的训练期先验。

**表 3a 基础教师消融**

| 数据集 | 监督配置 | alpha_E | alpha_R | 准确率 | 相对 CE |
| --- | --- | --- | --- | --- | --- |
| 2a | CE | 0 | 0 | 0.5672 | -- |
| 2a | 神经单教师 | 0.4 | 0 | 0.5742 | +0.0069 |
| 2a | 黎曼单教师 | 0 | 0.4 | 0.5655 | -0.0017 |
| 2a | 双教师 | 0.4 | 0.4 | 0.5847 | +0.0175 |
| 2b | CE | 0 | 0 | 0.7634 | -- |
| 2b | 固定 Riemann KD + spatial | 0 | 0.1 | 0.7660 | +0.0026 |
| 2b | 自适应 Riemann KD + spatial | 0 | 0.1 | 0.7712 | +0.0078 |
| 2b | 双教师（旧诊断） | 0.4 | 0.4 | 0.7621 | -0.0013 |

**表 3b 基础教师消融配对检验**

| 数据集 | 比较 | 差值 | 胜/平/负 | dz | t 检验 p | Wilcoxon p |
| --- | --- | --- | --- | --- | --- | --- |
| 2a | 双教师 vs CE | +0.0175 | 7/1/1 | 1.07 | 0.0122 | 0.0156 |
| 2a | 神经 vs CE | +0.0069 | 5/0/4 | 0.32 | 0.3603 | 0.5703 |
| 2a | 黎曼 vs CE | -0.0017 | 6/1/2 | -0.08 | 0.8057 | 0.7422 |
| 2b | 固定 Riemann KD + spatial vs CE | +0.0026 | -- | -- | -- | -- |
| 2b | 自适应 Riemann KD + spatial vs CE | +0.0078 | -- | -- | -- | -- |
| 2b | 全局路由 vs CE | +0.0067 | -- | -- | -- | -- |
| 2b | 双教师 vs CE（旧诊断） | -0.0013 | 5/0/4 | -0.07 | 0.8334 | 1.0000 |

### 3.4 等总 KD 权重消融

2a 的最新权重消融显示，总 KD 权重为 0.4 时，双教师 0.2+0.2 达到 58.47%，高于神经单教师 0.4 的 57.42% 和黎曼单教师 0.4 的 56.55%；对应配对检验 p=0.0039/0.0078 和 p=0.0348/0.0391（t 检验/Wilcoxon）。

总权重为 0.8 时，2a 双教师 0.4+0.4 为 58.47%，神经单教师 0.8 为 57.88%，黎曼单教师 0.8 为 56.20%，差异未达到显著。双教师 0.2+0.2 与 0.4+0.4 的准确率相同，说明继续增大蒸馏权重没有额外收益。

2b 的一致口径比较中，双教师 0.4+0.4 为 76.21%，神经单教师 0.8 为 76.42%，黎曼单教师 0.8 为 75.71%。双教师与两种单教师的差异均不显著。低总权重双教师 0.2+0.2 为 76.42%，同样未显示稳定优势。这些结果作为诊断说明：2b 中单纯增加神经教师或 KL 权重并不是主要收益来源，因此后续主结果采用 Riemann KD + spatial loss。

为进一步区分 Riemann KD 与空间结构约束的贡献，在 2b 的 8-32 Hz 协议下补充了等总辅助权重消融（表 4b）。每个配置均覆盖 9 名被试和 5 个随机种子；先对每名被试的 5 个 seed 求均值，再报告 9 名被试的均值和标准差。总辅助权重为 0.2 时，联合配置 Riemann KD + spatial (0.1+0.1) 的准确率为 76.48%，高于仅 Riemann KD (0.2+0) 的 76.08% 和仅 spatial (0+0.2) 的 76.14%，对应增益分别为 0.40 和 0.34 个百分点。较低总权重的联合配置 (0.05+0.05) 为 75.89%。该结果表明两项黎曼监督在中等权重下具有互补趋势；由于本组尚未进行与 CE 的配对显著性检验，该表仅作为描述性消融证据，不作显著性主张。

**表 4 等总 KD 权重性能比较**

| 数据集 | 配置 | 总 KD 权重 | 准确率 | Macro-F1 | Kappa |
| --- | --- | --- | --- | --- | --- |
| 2a | 双教师 0.2+0.2 | 0.4 | 0.5847 | 0.5719 | 0.4463 |
| 2a | 神经单教师 0.4 | 0.4 | 0.5742 | 0.5620 | 0.4322 |
| 2a | 黎曼单教师 0.4 | 0.4 | 0.5655 | 0.5536 | 0.4207 |
| 2a | 双教师 0.4+0.4 | 0.8 | 0.5847 | 0.5724 | 0.4463 |
| 2a | 神经单教师 0.8 | 0.8 | 0.5788 | 0.5657 | 0.4384 |
| 2a | 黎曼单教师 0.8 | 0.8 | 0.5620 | 0.5494 | 0.4160 |
| 2b | 双教师 0.2+0.2 | 0.4 | 0.7642 | 0.7582 | 0.5285 |
| 2b | 双教师 0.4+0.4 | 0.8 | 0.7621 | 0.7553 | 0.5242 |
| 2b | 神经单教师 0.8 | 0.8 | 0.7642 | 0.7595 | 0.5285 |
| 2b | 黎曼单教师 0.8 | 0.8 | 0.7571 | 0.7522 | 0.5142 |

**表 4b BCI IV-2b 中 Riemann KD 与空间结构损失的等总辅助权重消融（8-32 Hz）**

| 配置 (lambda_R + lambda_S) | 总辅助权重 | Acc | Macro-F1 | Kappa |
| --- | ---: | ---: | ---: | ---: |
| Riemann KD + spatial (0.05 + 0.05) | 0.10 | 0.7589 ± 0.1420 | 0.7525 ± 0.1494 | 0.5177 ± 0.2840 |
| Riemann KD + spatial (0.10 + 0.10) | 0.20 | **0.7648 ± 0.1354** | **0.7617 ± 0.1373** | **0.5296 ± 0.2709** |
| Riemann KD only (0.20 + 0.00) | 0.20 | 0.7608 ± 0.1373 | 0.7543 ± 0.1458 | 0.5216 ± 0.2745 |
| Spatial only (0.00 + 0.20) | 0.20 | 0.7614 ± 0.1421 | 0.7551 ± 0.1499 | 0.5229 ± 0.2841 |

注：数值为每名被试先对 5 个 seed 求均值后，再跨 9 名被试计算的均值 ± 标准差；该标准差描述被试间异质性，不能与按 seed 汇总的标准差直接比较。

**表 5 等总 KD 权重配对检验**

| 数据集 | 比较 | 差值 | 胜/平/负 | t 检验 p | Wilcoxon p |
| --- | --- | --- | --- | --- | --- |
| 2a | 双 0.2+0.2 vs 神经 0.4 | +0.0106 | 8/0/1 | 0.0039 | 0.0078 |
| 2a | 双 0.2+0.2 vs 黎曼 0.4 | +0.0192 | 7/0/2 | 0.0348 | 0.0391 |
| 2a | 双 0.4+0.4 vs 神经 0.8 | +0.0059 | 7/0/2 | 0.2500 | 0.2500 |
| 2a | 双 0.4+0.4 vs 黎曼 0.8 | +0.0227 | 7/0/2 | 0.1148 | 0.1523 |
| 2b | 双 0.4+0.4 vs 神经 0.8 | -0.0022 | 4/1/4 | 0.6329 | 0.8438 |
| 2b | 双 0.4+0.4 vs 黎曼 0.8 | +0.0050 | 6/0/3 | 0.6342 | 0.7344 |
| 2b | 双 0.4+0.4 vs 双 0.2+0.2 | -0.0021 | 3/2/4 | 0.5181 | 0.8125 |

### 3.5 与神经及经典基线的比较

2a 上，全局路由高于 EEGNetV4 2 s（59.31% vs 58.09%）、黎曼 TS-LogReg 2 s（56.94%）、FBCSP-LDA 2 s（56.38%）、ShallowConvNet 2 s（51.87%）和 DeepConvNet 2 s（48.63%）。其中，仅相对 CSP-LDA 2 s 的优势在 Holm 校正后仍显著；ShallowConvNet 和 DeepConvNet 当前仅作描述性复现对比，尚未纳入配对检验。完整时间窗中，黎曼 TS-LogReg 4 s 以 65.25% 最强，FBCSP-LDA 4 s（62.96%）次之，路由并未超过这两个方法。

2b 上，自适应 Riemann KD + spatial loss 达到 77.12%，与 EEGNetV4 2 s 的 77.14% 基本持平，并高于黎曼 TS-LogReg 2 s、CSP-LDA 2 s、FBCSP-LDA 2 s、ShallowConvNet 2 s（74.77%）和 DeepConvNet 2 s（74.87%）。全局路由达到 77.01%，略低于最佳单候选但标准差更小。DeepConvNet 4 s 达到 77.41%，略高于 2 s 路由，但需要完整 4 s 信号；EEGNetV4 4 s 达到 78.95%，仍是该数据集最强的已评估完整时间窗神经模型。因此 2b 结果应解释为短窗空间结构引导与验证集选择带来的小幅改善，而不能宣称超过所有完整时间窗方法。

路由策略的平均有效信号终点约为 2a 的 2.11 s；2b 诊断路由的平均有效终点约为 2.20 s，明显短于 4 s 参考。该指标描述所需 EEG 样本终点，不等同于包含采集、传输和决策在内的完整在线延迟。

**表 6 路由主比较的配对检验**

| 数据集 | 比较 | 准确率差值 | 胜/平/负 | t 检验 p | Wilcoxon p |
| --- | --- | --- | --- | --- | --- |
| 2a | 路由 vs 固定 2 s CE | +0.0259 | 8/0/1 | 0.0055 | 0.0078 |
| 2a | 路由 vs 固定双教师 | +0.0084 | 6/0/3 | 0.1741 | 0.2031 |
| 2a | 路由 vs Compact 4 s CE | +0.0096 | 6/0/3 | 0.6466 | 0.6523 |
| 2b | 固定 Riemann KD + spatial vs 固定 2 s CE | +0.0026 | -- | -- | -- |
| 2b | 自适应 Riemann KD + spatial vs 固定 2 s CE | +0.0078 | -- | -- | -- |
| 2b | 全局路由 vs 固定 2 s CE | +0.0067 | -- | -- | -- |

**表 7 五随机种子神经与经典基线**

| 数据集 | 方法 | 窗口 | 准确率 | Macro-F1 | Kappa |
| --- | --- | --- | --- | --- | --- |
| 2a | 全局路由 | 选择的 2 s | 0.5931 | 0.5842 | 0.4575 |
| 2a | EEGNetV4 CE | 2 s | 0.5809 | 0.5666 | 0.4412 |
| 2a | 黎曼 TS-LogReg | 2 s | 0.5694 | 0.5570 | 0.4259 |
| 2a | CSP-LDA | 2 s | 0.5530 | 0.5270 | 0.4040 |
| 2a | FBCSP-LDA | 2 s | 0.5638 | 0.5517 | 0.4184 |
| 2a | ShallowConvNet | 2 s | 0.5187 | 0.5113 | 0.3582 |
| 2a | DeepConvNet | 2 s | 0.4863 | 0.4553 | 0.3151 |
| 2a | Compact EEGNet CE | 4 s | 0.5836 | 0.5749 | 0.4448 |
| 2a | EEGNetV4 CE | 4 s | 0.6025 | 0.5894 | 0.4700 |
| 2a | 黎曼 TS-LogReg | 4 s | 0.6525 | 0.6420 | 0.5367 |
| 2a | CSP-LDA | 4 s | 0.6029 | 0.5726 | 0.4705 |
| 2a | FBCSP-LDA | 4 s | 0.6296 | 0.6154 | 0.5061 |
| 2a | ShallowConvNet | 4 s | 0.5038 | 0.4923 | 0.3384 |
| 2a | DeepConvNet | 4 s | 0.4710 | 0.4372 | 0.2947 |
| 2b | 自适应 Riemann KD + spatial | 选择的 2 s | 0.7712 | 0.7681 | 0.5424 |
| 2b | 全局路由 | 选择的路线 | 0.7701 | 0.7665 | 0.5402 |
| 2b | EEGNetV4 CE | 2 s | 0.7714 | 0.7676 | 0.5427 |
| 2b | 黎曼 TS-LogReg | 2 s | 0.7159 | 0.7080 | 0.4317 |
| 2b | CSP-LDA | 2 s | 0.7136 | 0.7076 | 0.4272 |
| 2b | FBCSP-LDA | 2 s | 0.7258 | 0.7227 | 0.4515 |
| 2b | ShallowConvNet | 2 s | 0.7477 | 0.7443 | 0.4954 |
| 2b | DeepConvNet | 2 s | 0.7487 | 0.7429 | 0.4973 |
| 2b | Compact EEGNet CE | 4 s | 0.7806 | 0.7763 | 0.5612 |
| 2b | EEGNetV4 CE | 4 s | 0.7895 | 0.7852 | 0.5790 |
| 2b | 黎曼 TS-LogReg | 4 s | 0.7297 | 0.7189 | 0.4593 |
| 2b | CSP-LDA | 4 s | 0.7303 | 0.7197 | 0.4606 |
| 2b | FBCSP-LDA | 4 s | 0.7587 | 0.7540 | 0.5173 |
| 2b | ShallowConvNet | 4 s | 0.7666 | 0.7619 | 0.5332 |
| 2b | DeepConvNet | 4 s | 0.7741 | 0.7639 | 0.5482 |

**表 8a 2a 外部基线配对比较**

| 对比方法 | 差值 | 胜/平/负 | t 检验 p（Holm） | Wilcoxon p（Holm） | dz |
| --- | --- | --- | --- | --- | --- |
| EEGNetV4 2 s CE | +0.0123 | 7/0/2 | 0.1981 (0.5943) | 0.2031 (0.8125) | 0.47 |
| 黎曼 TS-LogReg 2 s | +0.0237 | 6/0/3 | 0.1443 (0.5772) | 0.2383 (0.8125) | 0.54 |
| CSP-LDA 2 s | +0.0401 | 9/0/0 | 0.0069 (0.0411) | 0.0039 (0.0234) | 1.20 |
| FBCSP-LDA 2 s | +0.0293 | 6/0/3 | 0.2106 (0.9905) | 0.3008 (1.0000) | 0.45 |
| CSP-LDA 4 s | -0.0097 | 5/0/4 | 0.7148 (1.0000) | 1.0000 (1.0000) | -0.13 |
| FBCSP-LDA 4 s | -0.0364 | 3/0/6 | 0.2126 (0.9905) | 0.2031 (1.0000) | -0.45 |
| EEGNetV4 4 s CE | -0.0093 | 5/0/4 | 0.5901 (1.0000) | 1.0000 (1.0000) | -0.19 |
| 黎曼 TS-LogReg 4 s | -0.0594 | 1/0/8 | 0.0108 (0.0542) | 0.0195 (0.0977) | -1.10 |

**表 8b 2b 外部基线配对比较**

| 对比方法 | 差值 | 胜/平/负 | t 检验 p（Holm） | Wilcoxon p（Holm） | dz |
| --- | --- | --- | --- | --- | --- |
| EEGNetV4 2 s CE | +0.0016 | 3/0/6 | 0.9120 (1.0000) | 0.7539 (0.9023) | 0.04 |
| 黎曼 TS-LogReg 2 s | +0.0571 | 9/0/0 | 0.0123 (0.0859) | 0.0039 (0.0313) | 1.07 |
| CSP-LDA 2 s | +0.0594 | 8/0/1 | 0.0132 (0.0859) | 0.0117 (0.0703) | 1.06 |
| FBCSP-LDA 2 s | +0.0472 | 9/0/0 | 0.0028 (0.0225) | 0.0039 (0.0313) | 1.42 |
| CSP-LDA 4 s | +0.0426 | 6/0/3 | 0.0515 (0.2061) | 0.0547 (0.2734) | 0.76 |
| FBCSP-LDA 4 s | +0.0143 | 6/0/3 | 0.3549 (1.0000) | 0.3594 (0.9023) | 0.33 |
| EEGNetV4 4 s CE | -0.0165 | 2/0/7 | 0.4429 (1.0000) | 0.3008 (0.9023) | -0.27 |
| 黎曼 TS-LogReg 4 s | +0.0433 | 7/0/2 | 0.0306 (0.1530) | 0.0547 (0.2734) | 0.87 |

### 3.6 路线使用与被试异质性

2a 共 45 次正式路由运行，固定 CE、固定双教师、自适应 CE 和自适应双教师分别被选择 14、15、9 和 7 次。2b 的统一 Compact 教师路由矩阵仅作为诊断分析保留：固定 CE、固定双教师、自适应 CE 和自适应双教师分别被选择 20、13、9 和 3 次，固定路线占 73.3%，双教师路线仅占 35.6%。这进一步说明二分类三通道任务中，额外神经教师的附加价值有限。

仅用于诊断的测试集 oracle 分析显示：即使事后选择最佳 2 s 候选，2a 的 S1、S2、S3 和 2b 的S9 仍低于 4 s CE，提示短窗信息上限；2b 的 S1、S7、S8 存在可匹配或超过 4 s 的 2 s 候选，但验证集未稳定选中，提示路线选择误差。oracle 结果未用于正式选模或主结果。

**表 9 验证集路线选择次数与诊断结果**

| 数据集 | 固定 CE | 固定双教师 | 自适应 CE | 自适应双教师 | 固定/自适应 | 角色 |
| --- | --- | --- | --- | --- | --- |
| BCI IV-2a | 14 | 15 | 9 | 7 | 29/16 | 正式主结果 |
| BCI IV-2b | 20 | 13 | 9 | 3 | 33/12 | 诊断对照 |

### 3.7 跨 seed 路由稳健性

留一 seed 共识路由在 2a 上达到 57.11%，固定 CE 为 56.72%，差值仅 0.39 个百分点且不显著（p=0.7127；Wilcoxon p=0.7422）。共识路由还显著低于原单次验证划分路由的 59.31%（p=0.0107）。

2b 的新矩阵显示，全局路由达到 77.01% ± 1.03%，低于最佳单候选自适应 Riemann KD + spatial loss 的 77.12% ± 1.52%，但高于固定 CE 的 76.34% ± 1.20%。这说明验证集选择没有等同于测试集 oracle 最优，仍存在 validation-to-test selection noise；不过，全局路由的波动小于最佳单候选，符合其降低固定策略风险的设计目标。严格的留一 seed 共识路由仍需在该新矩阵上重新计算。

**表 10 跨 seed 共识路由稳健性与诊断**

| 数据集 | 跨 seed 共识 | 固定 CE | 全局路由 | 共识 vs CE p | Wilcoxon p | 角色 |
| --- | --- | --- | --- | --- | --- |
| 2a | 0.5711 | 0.5672 | 0.5931 | 0.7127 | 0.7422 | 正式稳健性分析 |
| 2b | 未重算 | 0.7634 | 0.7701 | -- | -- | 未纳入当前结论 |

### 3.8 推理复杂度与逐类别表现

固定 2 s Compact EEGNet 在 2a/2b 上分别包含 2,420/1,634 个执行参数和 5.89/0.86 MMAC。加入检测器后分别为 3,526/3,500 个参数和 11.25/4.83 MMAC。RTX 3070 Ti 上固定路线延迟约为 1.01/1.42 ms，自适应路线约为 2.36/3.21 ms；教师不参与推理。由于 2b 正式主结果采用固定 0-2 s 的 Riemann KD + spatial loss，2b 检测器复杂度仅表示诊断路由的开销。

2a 的历史汇总混淆计数显示，双教师对双脚和右手类别改善较明显。2b 当前已更新整体准确率和 global routing 的 macro-F1/kappa；逐类别混淆分析需在自适应 Riemann KD + spatial 与全局路由配置下重新汇总后再纳入。

**表 11 推理复杂度与延迟**

| 数据集 | 部署路线 | 参数量 | MACs | CPU ms/trial | GPU ms/trial |
| --- | --- | --- | --- | --- | --- |
| 2a | 固定 2 s Compact | 2,420 | 5.89 M | 1.12 | 1.01 |
| 2a | 检测器+Compact | 3,526 | 11.25 M | 3.88 | 2.36 |
| 2b | 固定 2 s Compact | 1,634 | 0.86 M | 1.48 | 1.42 |
| 2b | 检测器+Compact（诊断） | 3,500 | 4.83 M | 5.31 | 3.21 |

**表 12 2a 双教师逐类别 F1**

| 数据集 | 类别 | CE F1 | 双教师 F1 | 变化 |
| --- | --- | --- | --- | --- |
| 2a | 左手 | 0.593 | 0.610 | +0.018 |
| 2a | 右手 | 0.598 | 0.626 | +0.027 |
| 2a | 双脚 | 0.468 | 0.492 | +0.024 |
| 2a | 舌部 | 0.595 | 0.599 | +0.004 |

## 4 讨论

本文最稳定的结果出现在四分类 2a：全局路由相对固定 2 s CE 提高 2.59 个百分点，固定双教师也提高 1.75 个百分点。2b 中，自适应 Riemann KD + spatial loss 达到最高均值 77.12%，全局路由达到 77.01%，相对固定 2 s CE 分别提高 0.78 和 0.67 个百分点。2b 的提升幅度较小，说明神经教师蒸馏、时间路由和黎曼空间结构约束的收益具有明显的数据集依赖性。

2a 的等总权重消融支持神经决策边界与协方差几何信息存在互补趋势，尤其在总权重 0.4 时，双教师显著高于两种单教师。总权重提高到 0.8 后没有继续改善，说明收益并非简单来自增加 KL 项权重。

2b 的新结果更为保守：固定双教师略低于固定 CE，神经单教师与双教师接近，而单纯提高黎曼 KD 权重不能稳定改善性能。三通道二分类任务的基线较高，教师间可传递的互补信息可能有限；相比之下，Riemann KD + spatial loss 直接约束 student 的空间表征，提供了更合适但幅度有限的训练期先验。全局路由略低于最佳单候选，但标准差更小，说明其价值更接近于降低跨 seed 波动和固定策略风险，而不是保证超过所有候选。

验证集路由在 2a 上有效，但仍依赖单次验证划分；在 2b 上，路由仅作为诊断对照，未作为最终主配置。因此，下一步应使用训练会话内部的折外预测估计候选效用，并设置数据集或被试级安全阈值：只有当自适应路线、双教师或 spatial 约束在内部验证中稳定优于固定 CE 时才启用，否则回退固定 CE。

与 4 s 模型的比较支持谨慎的实际意义。2a 路由未超过最强的 4 s 黎曼 TS-LogReg，2b 也未超过EEGNetV4 4 s。当前证据适合支持“使用更短信号保持接近性能并改善部分被试”，而不是“2 s 普遍优于 4 s”。

复杂度结果表明训练期教师不会增加部署负担。固定路线只有约 1 ms 的 GPU 前向延迟，自适应路线也低于4 ms。实际在线延迟仍由 2 s EEG 采集时间主导，但模型计算不是主要瓶颈。

2a 的类别结果提示双教师可能对困难类别更有帮助。2b 当前主配置的逐类别混淆结果尚未重新汇总，因此本文暂不把旧双教师混淆矩阵作为 2b 主结论。

下一步最重要的改进不是继续增加候选，而是在训练会话内部使用严格的 K 折折外预测估计候选效用，并设置安全阈值：只有当自适应或教师路线在多个内部折上稳定优于固定 CE 时才启用，否则回退固定路线。

### 4.1 局限性

- 路由器在分别训练的候选之间进行被试级选择，不是端到端或 trial 级混合专家模型。
- 2a 主路由结果使用单个 20% 验证划分；跨 seed 共识分析显示其稳定性有限。
- 2a 与 2b 使用不同的训练期辅助策略，降低了跨数据集方法的一致性，但也反映了四分类多通道任务与二分类三通道任务的差异。
- 跨 seed 共识仍不是严格的 trial 级嵌套交叉验证，因为不同 seed 的训练子集存在重叠。
- 显著提升主要出现在 2a；2b 的自适应 Riemann KD + spatial loss 和全局路由仅显示小幅正向提升，统一双教师结果未显示优势。
- 方法未显著超过完整时间窗强基线，不能宣称不受时间窗限制的 SOTA。
- 当前实验为被试内训练，不能直接推断跨被试迁移或无校准部署能力。

## 5 结论

本文将短窗 MI-EEG 解码表述为训练期知识迁移与验证集控制策略选择问题。9 被试、5 随机种子实验表明，全局路由在 BCI IV-2a 上显著优于固定 2 s Compact EEGNet；在 BCI IV-2b 上，自适应 Riemann KD + spatial loss 达到 77.12%，全局路由达到 77.01%，相对固定 2 s CE 的 76.34% 分别提高 0.78 和 0.67 个百分点。等总权重消融支持双教师互补性主要存在于更困难的四分类任务，而不能推广为跨数据集的普遍规律；二分类三通道任务更适合使用黎曼空间结构约束作为轻量训练期先验。推理阶段仅保留学生和可选检测器，计算延迟较低。综合而言，该方法为短窗训练期知识迁移、黎曼空间结构引导和被试级策略选择提供了有价值的证据，但 2b 的收益仍较小，仍需严格的折外路由、统一的稳健性分析和前瞻性在线验证。

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
