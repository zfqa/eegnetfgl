from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.shared import Pt


SRC = Path(r"C:\Users\10906\Desktop\中文论文初稿.docx")
OUT_DIR = Path(r"C:\Users\10906\Documents\eegfgl 2\eegnet_fgl_bci2a\manuscripts\referenced_chinese")
OUT = OUT_DIR / "中文论文初稿_参考文献标注版.docx"


OLD_REFS = {
    1: "Altaheri H, Muhammad G, Alsulaiman M et al (2023) Deep learning techniques for classification of electroencephalogram (EEG) motor imagery (MI) signals: a review. Neural Comput Appl 35:14681-14722. https://doi.org/10.1007/s00521-023-08542-7",
    2: "Ang KK, Guan C (2018) EEG-based strategies for stroke rehabilitation. In: Nam CS, Nijholt A, Lotte F (eds) Brain-computer interfaces handbook. CRC Press, pp 553-574",
    3: "Barachant A, Bonnet S, Congedo M, Jutten C (2012) Multiclass brain-computer interface classification by Riemannian geometry. IEEE Trans Biomed Eng 59:920-928. https://doi.org/10.1109/TBME.2011.2172210",
    4: "Barachant A, Bonnet S, Congedo M, Jutten C (2013) Classification of covariance matrices using a Riemannian-based kernel for BCI applications. Neurocomputing 112:172-178. https://doi.org/10.1016/j.neucom.2012.12.039",
    5: "Blankertz B, Tomioka R, Lemm S, Kawanabe M, Muller KR (2008) Optimizing spatial filters for robust EEG single-trial analysis. IEEE Signal Process Mag 25:41-56. https://doi.org/10.1109/MSP.2008.4408441",
    6: "Brunner C, Leeb R, Muller-Putz G, Schlogl A, Pfurtscheller G (2010) BCI Competition 2008: Graz data set A. Institute for Knowledge Discovery, Graz University of Technology",
    7: "Chen Y, Wiesel A, Eldar YC, Hero AO (2010) Shrinkage algorithms for MMSE covariance estimation. IEEE Trans Signal Process 58:5016-5029. https://doi.org/10.1109/TSP.2010.2053029",
    8: "Congedo M, Barachant A, Bhatia R (2017) Riemannian geometry for EEG-based brain-computer interfaces; a primer and a review. BCI 4:1-39. https://doi.org/10.1080/2326263X.2017.1297192",
    9: "Cui J, Lan Z, Liu Y et al (2022) A compact and interpretable convolutional neural network for cross-subject driver drowsiness recognition from single-channel EEG. IEEE Trans Neural Syst Rehabil Eng 30:286-296. https://doi.org/10.1109/TNSRE.2022.3147419",
    10: "Ding Y, Robinson N, Zeng Q et al (2022) TSception: a deep learning framework for emotion detection using EEG. In: Proc Int Joint Conf Neural Networks (IJCNN), pp 1-8. https://doi.org/10.1109/IJCNN55064.2022.9967692",
    11: "Hinton G, Vinyals O, Dean J (2015) Distilling the knowledge in a neural network. arXiv:1503.02531. https://doi.org/10.48550/arXiv.1503.02531",
    12: "Jayaram V, Barachant A (2018) MOABB: trustworthy algorithm benchmarking for BCIs. J Neural Eng 15:066011. https://doi.org/10.1088/1741-2552/aadea0",
    13: "Kalunga EK, Chevallier S, Barre Q, Djouani K, Monacelli E, Hamam Y (2016) Online SSVEP-based BCI using Riemannian geometry. Neurocomputing 191:55-68. https://doi.org/10.1016/j.neucom.2016.01.007",
    14: "Lawhern VJ, Solon AJ, Waytowich NR, Gordon SM, Hung CP, Lance BJ (2018) EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces. J Neural Eng 15:056013. https://doi.org/10.1088/1741-2552/aace8c",
    15: "Lotte F, Bougrain L, Cichocki A et al (2018) A review of classification algorithms for EEG-based brain-computer interfaces: a 10 year update. J Neural Eng 15:031005. https://doi.org/10.1088/1741-2552/aab2f2",
    16: "Lotte F, Jeunet C (2015) Towards improved BCI spellers using a co-adaptive calibration of the P300 speller with RSVP and visual feedback. J Neural Eng 12:066007",
    17: "Mane R, Chew E, Chua K et al (2021) FBCNet: a multi-view convolutional neural network for brain-computer interface. arXiv:2104.01233",
    18: "Meisheri H, Ramrao N, Mitra S, Saha S (2022) Common spatial pattern for EEG based brain-computer interface: an implementation review. In: Proc Int Conf Comput Intell, pp 1-6",
    19: "Pfurtscheller G, Neuper C (2001) Motor imagery and direct brain-computer communication. Proc IEEE 89:1123-1134. https://doi.org/10.1109/5.939829",
    20: "Roy Y, Banville H, Albuquerque I, Gramfort A, Falk TH, Faubert J (2019) Deep learning-based electroencephalography analysis: a systematic review. J Neural Eng 16:051001. https://doi.org/10.1088/1741-2552/ab260c",
    21: "Schirrmeister RT, Springenberg JT, Fiederer LDJ et al (2017) Deep learning with convolutional neural networks for EEG decoding and visualization. Hum Brain Mapp 38:5391-5420. https://doi.org/10.1002/hbm.23730",
    22: "Song Y, Zheng Q, Liu B, Gao X (2023) EEG conformer: convolutional transformer for EEG decoding and visualization. IEEE Trans Neural Syst Rehabil Eng 31:710-719. https://doi.org/10.1109/TNSRE.2022.3230250",
    23: "Tangermann M, Muller KR, Aertsen A et al (2012) Review of the BCI Competition IV. Front Neurosci 6:55. https://doi.org/10.3389/fnins.2012.00055",
    24: "Tang J, Liu C, Li Q (2020) EEG feature distillation for motor imagery classification. In: Proc IEEE Int Conf Bioinformatics and Biomedicine (BIBM), pp 1680-1683",
    25: "Wang J, Li S, Li R, Shi Y (2022) EEG motor imagery classification with multi-view convolutional neural network. Biomed Signal Process Control 77:103778",
    26: "Wei Q, Wang Y, Gao X, Gao S (2022) Amplitude and phase coupling in steady-state motion visual evoked potentials for BCI. J Neural Eng 19:026002",
    27: "Yger F, Berar M, Lotte F (2016) Riemannian approaches in brain-computer interfaces: a review. IEEE Trans Neural Syst Rehabil Eng 25:1753-1762. https://doi.org/10.1109/TNSRE.2016.2627016",
    28: "Zhang L, Wang J, Chen B et al (2023) Temporal convolutional network for motor imagery EEG decoding with attention mechanism. Biomed Signal Process Control 81:104432",
    29: "Zhang X, Yao L, Wang X, Monaghan J, McAlpine D (2022) A survey on deep learning-based non-invasive brain signals: recent advances and new frontiers. J Neural Eng 18:031002",
    30: "Zuo C, Jin J, Yin Y et al (2023) Novel hybrid deep learning model for motor imagery EEG classification based on capsule network. Cogn Neurodyn 17:107-121",
}


# New order follows first appearance in the body text.
NEW_ORDER = [
    19, 2, 15, 1, 20, 29, 13, 26, 23, 6,
    3, 4, 8, 27, 7, 16, 14, 21, 5, 18,
    12, 11, 24, 9, 17, 22, 25, 28, 30, 10,
]
NEW_NUM = {old: new for new, old in enumerate(NEW_ORDER, start=1)}


def cite(*old_nums: int) -> str:
    nums = sorted(NEW_NUM[n] for n in old_nums)
    ranges: list[str] = []
    start = prev = nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
        else:
            ranges.append(f"{start}-{prev}" if start != prev else str(start))
            start = prev = n
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return "[" + ", ".join(ranges) + "]"


REPLACEMENTS = {
    12: (
        "运动想象脑电（motor imagery electroencephalography, MI-EEG）脑机接口通过识别被试想象肢体运动时产生的脑电模式，实现外部设备控制、康复训练和人机交互等功能"
        + cite(19, 2)
        + "。现有 MI-EEG 解码方法通常使用数秒级提示后 EEG 信号进行分类，相关分类算法和深度学习方法已在综述研究中得到系统总结"
        + cite(15, 1, 20, 29)
        + "。较长的分析窗口能够积累更充分的事件相关去同步化/同步化（event-related desynchronization/synchronization, ERD/ERS）信息，有利于提高分类稳定性。然而，较长窗口也意味着系统必须等待更多 EEG 信号后才能给出决策，这会增加响应延迟，限制脑机接口在早期决策和实时交互场景中的应用；类似的在线 BCI 和诱发电位研究也强调了实时反馈与稳定特征估计的重要性"
        + cite(13, 26)
        + "。因此，在较短 EEG 分析窗口下保持可靠解码性能，是 MI-EEG 脑机接口中的一个重要工程问题。"
    ),
    13: (
        "2 s 短窗解码是一种直接的早期决策方案。与完整 4 s 或更长分析窗口相比，2 s 输入可以减少分类前所需的信号长度，使系统更早输出预测结果。然而，短窗 EEG 也面临明显的信息不足问题。BCI Competition IV 的公开数据和综述工作表明，运动想象相关判别信息在单次 trial 内并非均匀分布，不同被试的运动想象起始时间、稳定程度和时空模式也存在差异"
        + cite(23, 6)
        + "。因此，固定使用经验性的 0-2 s 窗口可能无法适用于所有被试：部分被试的有效判别信息可能出现在更晚或更稳定的时间片段中，另一些被试则可能更适合固定早期窗口。这种被试依赖性使得单一固定时间窗难以稳定获得最优短窗性能。"
    ),
    14: (
        "除时间窗选择外，短窗学生模型还难以直接观察完整时间窗中的后续空间几何信息。黎曼几何方法将 EEG trial 表示为空间协方差矩阵，并在对称正定矩阵流形或切空间中进行分类，已被证明能够有效刻画 MI-EEG 的空间判别结构"
        + cite(3, 4, 8, 27)
        + "。协方差估计的稳定性也会影响这类空间统计表征，因此本文采用收缩协方差估计以提高小样本条件下的稳健性"
        + cite(7)
        + "。与 2 s 短窗输入相比，4 s 全窗协方差可以包含更完整的空间统计信息。因此，本文将 4 s 黎曼模型作为训练期特权教师，利用其类别软标签和空间协方差结构为 2 s Compact EEGNet 学生模型提供监督。该设计的目标不是在推理阶段部署黎曼分类器，而是将完整时间窗中的空间几何知识迁移到短窗神经网络中，使最终模型在测试时仍仅依赖 2 s EEG 输入。这样就可以实现更短的延迟和更好的性能保持。"
    ),
    15: (
        "但是同时在研究中我发现，长窗口有时也意味着包含更多的噪声，所包含的信息并不一定对所有被试均有益。由于 MI-EEG 具有显著的被试差异，某些被试可能受益于黎曼知识蒸馏或空间结构约束，而另一些被试可能更适合普通交叉熵训练。已有共适应校准研究也提示，BCI 系统性能与被试状态和校准策略之间存在相互作用"
        + cite(16)
        + "。同样，自适应 2 s 窗口也可能改善部分被试的时间对齐，但并非在所有被试上都优于固定窗口。为避免强制所有被试使用同一种监督方式或时间窗策略，本文进一步引入验证集控制的全局路由机制。该机制在固定 2 s 交叉熵、自适应 2 s 交叉熵、固定 2 s 黎曼空间蒸馏和自适应 2 s 黎曼空间蒸馏等候选策略之间进行被试级选择。候选策略仅依据训练会话内部验证集表现确定，评估会话始终隔离，仅用于最终测试。"
    ),
    21: (
        "（5）在 BCI Competition IV-2a 和 IV-2b 两个公开数据集上，采用 9 名被试、5 个随机种子和跨会话评估协议，对所提方法进行验证，并与 Compact EEGNet、EEGNetV4、ShallowConvNet、DeepConvNet、CSP-LDA、FBCSP-LDA 和黎曼切空间分类器等基线进行比较"
        + cite(14, 21, 5, 18)
        + "。"
    ),
    24: (
        "信号通过 MOABB 获取"
        + cite(12)
        + "。本文使用的 BCI Competition IV-2a 与 IV-2b 是运动想象 EEG 解码中常用的公开评测数据集"
        + cite(23, 6)
        + "。训练会话被划分为分层训练子集和验证子集（80/20）；评估会话作为唯一测试集保留。信号采用 8-32 Hz 滤波，所有 0-4 s 信号采样率为 250 Hz。5 个随机种子用于控制内部划分和优化过程。"
    ),
    28: (
        "部署阶段的学生模型为输入长度为 2 s 的 Compact EEGNet。EEGNet 及其紧凑卷积结构已被广泛用于 EEG 解码"
        + cite(14, 9)
        + "。在 IV-2a 上，冻结的 4 s Compact EEGNet 教师和黎曼教师提供软标签；该训练方式与知识蒸馏思想一致，即利用教师模型的软输出约束学生模型"
        + cite(11, 24)
        + "。在 IV-2b 上，4 s 黎曼教师提供软标签，并使用由协方差导出的空间目标对学生模型的 depthwise 空间卷积前表示进行正则化。候选路由仅依据验证集表现进行选择；评估会话不用于拟合教师、选择检查点或选择路由。"
    ),
    41: (
        "本文将所提出 2 s 策略与复现的 ShallowConvNet、DeepConvNet、经典方法、黎曼方法和 EEGNetV4 基线进行比较。除 EEGNet 与 Shallow/DeepConvNet 外，近年来还出现了 FBCNet、EEG Conformer、多视图 CNN、注意力时序卷积网络、胶囊网络和 TSception 等 EEG 深度解码模型"
        + cite(17, 22, 25, 28, 30, 10)
        + "。在 IV-2a 上，路由方法超过复现的 ShallowConvNet 和 DeepConvNet 基线。在 IV-2b 上，自适应 Riemann KD 加空间监督接近 EEGNetV4，并超过 Shallow/Deep 2 s 基线。"
    ),
    57: (
        "本文主要结果是，在推理阶段不携带 4 s 教师的条件下，IV-2a 的短窗解码性能获得了显著提升。IV-2a 与 IV-2b 的差异表明，特权长窗信息的效用具有任务依赖性：当类别结构和空间协方差信息更丰富时，该信息更有帮助。本文不声称 2 s 能替代 4 s，而是在可用证据被限制时，为更早决策建立一种受控路径。该结论与 MI-EEG 解码中算法、被试差异和校准条件共同影响性能的认识一致"
        + cite(15, 20, 27)
        + "。"
    ),
}


def set_paragraph_text(paragraph, text: str) -> None:
    style = paragraph.style
    paragraph.clear()
    paragraph.style = style
    run = paragraph.add_run(text)
    run.font.name = "宋体"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    run.font.size = Pt(10.5)


def rebuild_references(doc: Document) -> None:
    ref_start = None
    for idx, paragraph in enumerate(doc.paragraphs):
        if paragraph.text.strip() == "参考文献":
            ref_start = idx
            break
    if ref_start is None:
        raise RuntimeError("Cannot find reference heading.")

    # Remove old reference paragraphs after the heading.
    for paragraph in list(doc.paragraphs[ref_start + 1 :]):
        p = paragraph._element
        p.getparent().remove(p)

    heading = doc.paragraphs[ref_start]
    heading.style = "Heading 1" if "Heading 1" in [s.name for s in doc.styles] else heading.style

    for new_idx, old_idx in enumerate(NEW_ORDER, start=1):
        p = doc.add_paragraph(f"{new_idx}. {OLD_REFS[old_idx]}")
        p.paragraph_format.space_after = Pt(3)
        for run in p.runs:
            run.font.name = "Times New Roman"
            run._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
            run.font.size = Pt(9)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    doc = Document(SRC)
    for idx, text in REPLACEMENTS.items():
        set_paragraph_text(doc.paragraphs[idx], text)
    rebuild_references(doc)
    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
