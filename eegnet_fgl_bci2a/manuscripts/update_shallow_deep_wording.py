"""Replace the provisional Shallow/Deep statistics wording in the manuscript."""

from pathlib import Path

from docx import Document


path = (
    Path(__file__).resolve().parent
    / "global_routing_paper"
    / "validation_controlled_global_routing_mi_eeg_manuscript_zh_latest.docx"
)
doc = Document(path)
for paragraph in doc.paragraphs:
    if paragraph.text.startswith("2a 上，全局路由高于 EEGNetV4"):
        paragraph.text = (
            "2a 上，全局路由高于 EEGNetV4 2 s（59.31% vs 58.09%）、黎曼 TS-LogReg 2 s（56.94%）、"
            "FBCSP-LDA 2 s（56.38%）、ShallowConvNet 2 s（51.87%）和 DeepConvNet 2 s（48.63%）。"
            "其中，仅相对 CSP-LDA 2 s 的优势在 Holm 校正后仍显著；ShallowConvNet 和 DeepConvNet 当前仅作描述性复现对比，"
            "尚未纳入配对检验。完整时间窗中，黎曼 TS-LogReg 4 s 以 65.25% 最强，FBCSP-LDA 4 s（62.96%）次之，"
            "路由并未超过这两个方法。"
        )
        doc.save(path)
        print(path)
        break
else:
    raise RuntimeError("Target paragraph not found")
