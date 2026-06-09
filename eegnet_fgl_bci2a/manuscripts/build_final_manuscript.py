from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "manuscripts" / "final_paper_draft"
DOCX_PATH = OUT_DIR / "dual_teacher_short_window_mi_eeg_manuscript.docx"
MD_PATH = OUT_DIR / "dual_teacher_short_window_mi_eeg_manuscript.md"

FIGURES = {
    "compact": ROOT / "results" / "publication_complete_analysis" / "bci2a_compact_ablation_summary.png",
    "confusion": ROOT / "results" / "publication_complete_analysis" / "bci2a_main_confusion_matrices.png",
    "bci2b": ROOT / "results" / "publication_complete_analysis" / "bci2b_method_accuracy_summary.png",
}

BLUE = "2E5E86"
DARK_BLUE = "173A5E"
LIGHT_BLUE = "EAF1F7"
LIGHT_GRAY = "F2F4F7"
MID_GRAY = "667085"
BLACK = "111827"


TITLE = (
    "Dual-Teacher Knowledge Distillation for Compact Short-Window "
    "Motor Imagery EEG Decoding"
)

ABSTRACT = (
    "Short analysis windows can reduce the response latency of motor imagery (MI) "
    "brain-computer interfaces, but they provide less temporal evidence than full-trial "
    "electroencephalography (EEG). This study investigated whether knowledge extracted "
    "from 4-s EEG can improve a compact classifier that receives only the first 2 s at "
    "inference. A 2-s Compact EEGNet student was jointly supervised by the ground-truth "
    "labels, a frozen 4-s EEGNet teacher, and a 4-s Riemannian tangent-space logistic-"
    "regression teacher. Subject-dependent cross-session experiments were conducted on "
    "BCI Competition IV datasets 2a and 2b using nine subjects and five random seeds. "
    "Training-session data were separated into training and validation subsets, whereas "
    "the evaluation sessions were retained for final testing. On dataset 2a, dual-teacher "
    "distillation increased mean accuracy from 56.72% to 58.47% (+1.75 percentage points), "
    "with improvements in seven of nine subjects (paired t-test p=0.0122; Wilcoxon "
    "p=0.0156; Cohen dz=1.07). The advantage did not remain significant after Holm "
    "correction across all compact-model comparisons. The same method produced a smaller, "
    "non-significant gain on dataset 2b (76.35% to 76.77%, p=0.4574). Moreover, applying "
    "dual and spatial Riemannian guidance to EEGNetV4 yielded only a 0.24-point gain on "
    "dataset 2a. These findings show that long-window neural and covariance-geometric "
    "teachers can improve a capacity-limited short-window student, but the effect is "
    "backbone- and dataset-dependent. The proposed method is therefore best interpreted "
    "as a compact low-latency decoding strategy rather than a universal accuracy "
    "improvement or a state-of-the-art full-window classifier."
)


INTRODUCTION = [
    (
        "Motor imagery brain-computer interfaces (BCIs) infer imagined movements from "
        "electroencephalography and can provide communication, rehabilitation, and control "
        "channels without overt muscular activity. In offline studies, classification is "
        "often performed on several seconds of post-cue EEG. In an interactive system, "
        "however, a long observation interval delays feedback. Reducing the analysis window "
        "is therefore attractive, but the shorter segment may contain only a partial "
        "event-related desynchronization or synchronization response and is more vulnerable "
        "to non-stationarity and noise."
    ),
    (
        "EEGNet is a compact convolutional architecture that combines temporal filtering, "
        "depthwise spatial filtering, and separable convolutions. Its low parameter count "
        "makes it appealing for subject-dependent EEG datasets, where the number of trials "
        "is limited. Riemannian methods offer a complementary view: each trial is represented "
        "by a symmetric positive-definite channel covariance matrix, and discrimination is "
        "performed using manifold-aware distances or tangent-space features. These methods "
        "are particularly relevant to MI because covariance captures cross-channel spatial "
        "organization in sensorimotor rhythms."
    ),
    (
        "Knowledge distillation provides a mechanism for transferring information from a "
        "teacher to a smaller or more constrained student. Most EEG distillation studies "
        "focus on model compression or teacher-student architectures using the same input "
        "duration. A less explored setting is temporal privilege: a teacher receives a "
        "longer segment during training, while the student is restricted to an earlier "
        "segment during both training and testing. This setting is clinically and "
        "computationally meaningful because privileged full-window information is available "
        "offline but not required during deployment."
    ),
    (
        "We study whether two heterogeneous 4-s teachers can improve a 2-s Compact EEGNet "
        "student. The first teacher is an EEGNet classifier that provides neural soft labels. "
        "The second teacher is a Riemannian tangent-space logistic-regression classifier that "
        "provides covariance-geometric soft labels. The student is trained with cross-entropy "
        "and two Kullback-Leibler divergence terms, but only the student is retained at "
        "inference. We further test Riemannian spatial-filter regularization on EEGNetV4, "
        "compare with classical covariance and CSP baselines, and repeat the evaluation on "
        "BCI Competition IV dataset 2b."
    ),
]

CONTRIBUTIONS = [
    "A dual-teacher training framework that transfers same-trial 4-s neural and Riemannian soft labels to a Compact EEGNet using only the first 2 s at inference.",
    "A leakage-controlled, subject-dependent cross-session evaluation over nine subjects and five random seeds on BCI Competition IV-2a and IV-2b.",
    "Ablations covering single teachers, dual teachers, Riemannian spatial regularization, a stronger EEGNetV4 backbone, and classical CSP/Riemannian baselines.",
    "An explicit analysis of where the method succeeds and fails, including multiple-comparison correction, class-wise confusion, teacher-quality correlations, and cross-dataset generalization.",
]


METHODS = {
    "2.1 Datasets": [
        (
            "BCI Competition IV dataset 2a was accessed through MOABB as BNCI2014_001. "
            "It contains EEG from nine subjects performing four MI classes: left hand, "
            "right hand, both feet, and tongue. The experiments used the 22 EEG channels "
            "sampled at 250 Hz. Dataset 2b was accessed as BNCI2014_004 and contains "
            "left- versus right-hand MI from nine subjects using three bipolar EEG channels "
            "at 250 Hz. Both datasets provide temporally separated training and evaluation "
            "recordings, enabling cross-session assessment."
        )
    ],
    "2.2 Preprocessing and leakage control": [
        (
            "MOABB MotorImagery was used to load 0-4 s after-cue epochs with an 8-32 Hz "
            "band-pass filter. The analysis was subject dependent. For each subject and "
            "random seed, the designated training sessions were split into 80% training and "
            "20% validation data using stratification. The designated evaluation sessions "
            "were never used for fitting, hyperparameter selection, teacher training, or "
            "normalization."
        ),
        (
            "Per-channel standardization parameters were estimated only from the training "
            "subset and then applied to validation and test trials. The 2-s input contained "
            "the first 500 samples of each trial, whereas the 4-s input contained 1000 "
            "samples. The paired 2-s and 4-s tensors were cropped from the same trial before "
            "batching, preserving labels and sample order. Consequently, every teacher "
            "prediction used for distillation corresponded to the same trial as the student "
            "input."
        ),
    ],
    "2.3 Compact EEGNet student and neural teacher": [
        (
            "The Compact EEGNet used F1=8 temporal filters, depth multiplier D=2, F2=16 "
            "pointwise filters, a 64-sample temporal kernel, ELU activations, average pooling, "
            "and dropout of 0.5. The 2-s student and 4-s EEGNet teacher shared the same compact "
            "architecture but had input-dependent classifier dimensions. The 4-s teacher was "
            "trained with cross-entropy on the subject's training data, selected by validation "
            "performance, frozen, and evaluated only to produce training/validation soft "
            "targets. It was not called during student test inference."
        )
    ],
    "2.4 Riemannian teacher": [
        (
            "For the Riemannian teacher, covariance matrices were estimated from raw 4-s "
            "training trials using Oracle Approximating Shrinkage. Symmetric positive-definite "
            "matrices were projected to the tangent space with the Riemannian metric and "
            "classified using multinomial logistic regression. The regularization parameter "
            "C was chosen from {0.01, 0.03, 0.1, 0.3, 1, 3, 10} using only the validation "
            "subset. No alignment was used in the final paper configuration. Class "
            "probabilities from the selected model served as geometry-derived soft labels."
        )
    ],
    "2.5 Dual-teacher objective": [
        (
            "Let z_s denote the 2-s student logits, z_e the frozen 4-s EEGNet teacher logits, "
            "and z_r the logits corresponding to the Riemannian teacher probabilities. The "
            "training objective was"
        ),
        (
            "L = CE(z_s, y) + alpha_E T^2 KL[p_E^T || p_S^T] + alpha_R T^2 KL[p_R^T || p_S^T],"
        ),
        (
            "where p^T=softmax(z/T). The main Compact EEGNet experiment used alpha_E=0.4, "
            "alpha_R=0.4, and T=1.0. Single-teacher ablations set one of the two teacher "
            "weights to zero. The test-time model consisted only of the 2-s student."
        ),
    ],
    "2.6 Strong-backbone and spatial-structure analysis": [
        (
            "To determine whether the benefit persisted with a stronger implementation, "
            "we repeated the experiment using EEGNetV4 with exact same-padding, Keras-like "
            "batch normalization, and max-norm constraints. The main strong-backbone "
            "configuration used F1=8, D=2, F2=16, a 125-sample temporal kernel, and dropout "
            "of 0.5. Riemannian spatial regularization matched the off-diagonal correlation "
            "structure of the depthwise spatial-filter weights to a confidence-weighted "
            "average covariance-correlation target derived from 4-s trials. The dual KD plus "
            "spatial setting used neural KD weight 0.1, Riemannian KD weight 0.1, and spatial "
            "weight 0.2."
        )
    ],
    "2.7 Training and model selection": [
        (
            "Neural models were trained for up to 300 epochs with batch size 32, learning "
            "rate 0.001, weight decay 0.0001, and early stopping patience of 100 epochs. "
            "Compact dual-teacher experiments used AdamW. Each experiment was repeated with "
            "seeds 42, 2024, 3407, 7, and 123. Checkpoints were selected using validation "
            "accuracy; the evaluation session was scored once after model selection."
        )
    ],
    "2.8 Baselines, metrics, and statistical analysis": [
        (
            "Baselines included Compact EEGNet and EEGNetV4 at 2 s and 4 s, single-teacher "
            "KD, Riemannian spatial-only and combined losses, minimum-distance-to-mean (MDM), "
            "Riemannian tangent-space logistic regression, and common spatial patterns with "
            "linear discriminant analysis (CSP-LDA). Accuracy, macro-F1, and Cohen's kappa "
            "were reported."
        ),
        (
            "For inference-oriented performance estimates, results were first averaged over "
            "the five seeds within each subject. Paired t-tests and Wilcoxon signed-rank tests "
            "were then performed over the nine subject means. Cohen's dz quantified the paired "
            "effect size. Holm correction was applied within each family of comparisons. "
            "Teacher-quality associations were assessed with Pearson and Spearman correlations."
        ),
    ],
}


RESULTS = {
    "3.1 Main BCI Competition IV-2a result": [
        (
            "Across 45 runs, Compact EEGNet 2s achieved 56.72% accuracy, 55.50% macro-F1, "
            "and kappa 0.4229. Dual-teacher KD increased these values to 58.47%, 57.24%, and "
            "0.4463, respectively. On the nine seed-averaged subject scores, the accuracy "
            "difference was +1.75 percentage points; seven subjects improved, one was tied, "
            "and one declined. The unadjusted paired t-test (p=0.0122) and Wilcoxon test "
            "(p=0.0156) were significant, with Cohen's dz=1.07. After Holm correction across "
            "the six compact-model comparisons, the p-values were 0.0734 and 0.0938."
        ),
        (
            "The dual-teacher 2-s mean was close to the Compact EEGNet 4-s reference "
            "(58.47% versus 58.36%). However, the 4-s compact model itself did not "
            "significantly outperform its 2-s counterpart across subjects. This comparison "
            "therefore suggests that the student recovered useful full-window supervisory "
            "information, but it does not establish equivalence to all long-window decoders."
        ),
    ],
    "3.2 Ablation and strong-backbone results": [
        (
            "The neural teacher alone increased Compact EEGNet accuracy to 57.42%, whereas "
            "the Riemannian teacher alone reached 56.55%. Spatial-only and Riemannian KD plus "
            "spatial variants did not improve the compact baseline. The best result therefore "
            "required the combination of neural and Riemannian soft-label supervision rather "
            "than either teacher in isolation."
        ),
        (
            "With EEGNetV4, the 2-s cross-entropy baseline reached 58.09% and the 4-s model "
            "reached 60.25%. Dual KD plus spatial regularization achieved 58.33%, an increase "
            "of only 0.24 percentage points over EEGNetV4 2s (five wins and four losses; "
            "paired t-test p=0.7570). Thus, the main improvement was specific to the smaller "
            "Compact EEGNet student and was largely absorbed by the stronger backbone."
        ),
    ],
    "3.3 Classical baselines": [
        (
            "Riemannian tangent-space logistic regression with 4-s input was the strongest "
            "BCI IV-2a method in the evaluated matrix, obtaining 65.25% mean accuracy. "
            "CSP-LDA 4s and Riemannian MDM 4s achieved 60.29% and 59.48%, respectively. "
            "At 2 s, tangent-space logistic regression reached 56.94%, close to the Compact "
            "EEGNet baseline but below the dual-teacher student. The proposed method should "
            "therefore be interpreted as improving a compact short-window neural decoder, "
            "not as outperforming the strongest full-window classical pipeline."
        )
    ],
    "3.4 Cross-dataset evaluation on BCI Competition IV-2b": [
        (
            "On the binary BCI IV-2b dataset, Compact EEGNet 2s already achieved 76.35% "
            "accuracy. Dual-teacher KD increased accuracy to 76.77% (+0.42 percentage "
            "points), with six subject-level wins and three losses, but the difference was "
            "not significant (paired t-test p=0.4574; Wilcoxon p=0.4258). EEGNetV4 2s and "
            "4s achieved 77.14% and 78.95%, respectively. The weaker relative benefit on "
            "dataset 2b suggests that the teacher signal is less useful when the short-window "
            "binary task is already comparatively easy."
        )
    ],
    "3.5 Class-wise errors and teacher quality": [
        (
            "Checkpoint-level re-inference reproduced the aggregate BCI IV-2a result. "
            "Dual-teacher training increased right-hand recall from 62.4% to 65.9% and feet "
            "recall from 41.3% to 44.4%; left-hand recall changed from 62.1% to 62.5%, while "
            "tongue recall remained 61.1%. The persistent difficulty of the feet class "
            "indicates that the improvement was not merely a uniform confidence shift."
        ),
        (
            "Teacher validation accuracy was not a reliable predictor of Compact EEGNet "
            "improvement. At the subject-mean level, Pearson correlations were -0.119 for the "
            "EEGNet teacher (p=0.760) and -0.300 for the Riemannian teacher (p=0.433). "
            "This finding argues against a simple global trust rule based only on teacher "
            "validation accuracy and motivates future sample- or subject-adaptive gating."
        ),
    ],
}


DISCUSSION = [
    (
        "The central result is that two heterogeneous long-window teachers can improve a "
        "capacity-limited short-window neural decoder. The 4-s EEGNet teacher supplies a "
        "smooth discriminative class distribution learned directly from the waveform, while "
        "the 4-s Riemannian teacher encodes covariance geometry across channels. Neither "
        "teacher was sufficient alone in the compact ablation, but their combination yielded "
        "the highest 2-s compact-model accuracy. This pattern is consistent with complementary "
        "supervision rather than simple duplication of the ground-truth label."
    ),
    (
        "The gain is practically relevant but modest in absolute terms. A 1.75-point mean "
        "increase can matter in a compact low-latency decoder, particularly because inference "
        "does not require either teacher or the second half of the trial. At the same time, "
        "the result must be interpreted with statistical care. The primary unadjusted paired "
        "comparison was significant and showed a large standardized paired effect, but it did "
        "not survive Holm correction across all compact ablations. The study therefore "
        "provides encouraging evidence rather than definitive proof of a universal benefit."
    ),
    (
        "Backbone dependence is one of the most informative findings. EEGNetV4 improved the "
        "2-s cross-entropy baseline relative to Compact EEGNet, yet Riemannian and dual "
        "guidance added almost nothing. A stronger model may already learn much of the "
        "available short-window spatial structure, reducing the marginal information supplied "
        "by the teachers. Alternatively, fixed KD and spatial weights may over-constrain a "
        "higher-capacity network. This result supports adaptive transfer rather than applying "
        "the same loss to every backbone."
    ),
    (
        "Dataset dependence was similarly clear. BCI IV-2b is a two-class task with only "
        "three EEG channels, and its 2-s baseline was already high. The small, non-significant "
        "gain suggests a ceiling effect and reduced opportunity for a 22-channel covariance "
        "teacher to contribute. In contrast, the four-class IV-2a task contains a more complex "
        "decision boundary and richer spatial covariance structure. The method's most plausible "
        "use case is therefore a constrained short-window decoder on a difficult multi-class "
        "task rather than all MI-EEG settings."
    ),
    (
        "The strong 4-s tangent-space logistic-regression result also deserves emphasis. "
        "Classical Riemannian approaches remain competitive and should not be treated only as "
        "weak baselines. Here, their value is twofold: they provide an interpretable benchmark "
        "and a source of privileged geometric supervision. Future work could improve transfer "
        "by learning a validation-controlled trust gate, calibrating teacher probabilities, "
        "or activating Riemannian losses only when teacher uncertainty and student uncertainty "
        "indicate useful complementary information."
    ),
]

LIMITATIONS = [
    "The evaluation is offline and subject dependent; no asynchronous or online BCI experiment was performed.",
    "The main effect is modest and does not remain significant after Holm correction over all compact-model comparisons.",
    "The method improves Compact EEGNet on IV-2a but not EEGNetV4, indicating backbone dependence.",
    "Cross-dataset improvement on IV-2b is small and non-significant.",
    "Fixed teacher weights were used; teacher validation accuracy alone did not identify subjects that would benefit.",
    "The strongest full-window Riemannian baseline remained more accurate than the proposed 2-s neural student.",
]

CONCLUSION = (
    "This study evaluated dual-teacher knowledge distillation from 4-s neural and "
    "Riemannian teachers to a 2-s Compact EEGNet student. Across nine subjects and five "
    "seeds on BCI Competition IV-2a, the method improved short-window accuracy by 1.75 "
    "percentage points and benefited seven subjects, while requiring only the 2-s student "
    "at inference. The benefit was not reproduced to the same extent with EEGNetV4 or on "
    "BCI IV-2b. The evidence therefore supports dual-teacher transfer as a targeted strategy "
    "for compact, difficult short-window MI decoding, while highlighting the need for "
    "adaptive teacher trust and broader validation."
)


REFERENCES = [
    "Barachant A, Bonnet S, Congedo M, Jutten C (2012) Multiclass brain-computer interface classification by Riemannian geometry. IEEE Trans Biomed Eng 59:920-928. https://doi.org/10.1109/TBME.2011.2172210",
    "Barachant A, Bonnet S, Congedo M, Jutten C (2013) Classification of covariance matrices using a Riemannian-based kernel for BCI applications. Neurocomputing 112:172-178. https://doi.org/10.1016/j.neucom.2012.12.039",
    "Blankertz B, Tomioka R, Lemm S, Kawanabe M, Muller KR (2008) Optimizing spatial filters for robust EEG single-trial analysis. IEEE Signal Process Mag 25:41-56. https://doi.org/10.1109/MSP.2008.4408441",
    "Chen Y, Wiesel A, Eldar YC, Hero AO (2010) Shrinkage algorithms for MMSE covariance estimation. IEEE Trans Signal Process 58:5016-5029. https://doi.org/10.1109/TSP.2010.2053029",
    "Hinton G, Vinyals O, Dean J (2015) Distilling the knowledge in a neural network. arXiv:1503.02531. https://doi.org/10.48550/arXiv.1503.02531",
    "Jayaram V, Barachant A (2018) MOABB: trustworthy algorithm benchmarking for BCIs. J Neural Eng 15:066011. https://doi.org/10.1088/1741-2552/aadea0",
    "Lawhern VJ, Solon AJ, Waytowich NR, Gordon SM, Hung CP, Lance BJ (2018) EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces. J Neural Eng 15:056013. https://doi.org/10.1088/1741-2552/aace8c",
    "Lotte F, Bougrain L, Cichocki A et al (2018) A review of classification algorithms for EEG-based brain-computer interfaces: a 10 year update. J Neural Eng 15:031005. https://doi.org/10.1088/1741-2552/aab2f2",
    "Schirrmeister RT, Springenberg JT, Fiederer LDJ et al (2017) Deep learning with convolutional neural networks for EEG decoding and visualization. Hum Brain Mapp 38:5391-5420. https://doi.org/10.1002/hbm.23730",
    "Tangermann M, Muller KR, Aertsen A et al (2012) Review of the BCI Competition IV. Front Neurosci 6:55. https://doi.org/10.3389/fnins.2012.00055",
]


TABLE_1 = [
    ["Dataset", "Task", "EEG channels", "Sampling", "Train/test protocol"],
    ["BCI IV-2a / BNCI2014_001", "4-class MI", "22", "250 Hz", "Training session -> train/validation; evaluation session -> test"],
    ["BCI IV-2b / BNCI2014_004", "2-class MI", "3", "250 Hz", "Training sessions -> train/validation; evaluation sessions -> test"],
]

TABLE_2 = [
    ["Method", "Window", "Accuracy", "Macro-F1", "Kappa"],
    ["Compact EEGNet CE", "2 s", "0.5672", "0.5550", "0.4229"],
    ["Compact + EEGNet teacher KD", "2 s", "0.5742", "0.5620", "0.4322"],
    ["Compact + Riemann teacher KD", "2 s", "0.5655", "0.5536", "0.4207"],
    ["Compact + dual teacher KD", "2 s", "0.5847", "0.5724", "0.4463"],
    ["Compact EEGNet CE", "4 s", "0.5836", "0.5749", "0.4448"],
    ["EEGNetV4 CE", "2 s", "0.5809", "0.5666", "0.4412"],
    ["EEGNetV4 + dual KD + spatial", "2 s", "0.5833", "0.5727", "0.4443"],
    ["EEGNetV4 CE", "4 s", "0.6025", "0.5894", "0.4700"],
    ["Riemann TS-LogReg", "2 s", "0.5694", "0.5570", "0.4259"],
    ["Riemann TS-LogReg", "4 s", "0.6525", "0.6420", "0.5367"],
    ["CSP-LDA", "4 s", "0.6029", "0.5726", "0.4705"],
]

TABLE_3 = [
    ["Comparison", "Delta acc.", "Wins/ties/losses", "Paired t p", "Wilcoxon p", "Cohen dz"],
    ["Compact dual KD vs Compact CE 2 s", "+0.0175", "7/1/1", "0.0122", "0.0156", "1.073"],
    ["Compact EEG teacher vs Compact CE 2 s", "+0.0069", "5/0/4", "0.3603", "0.5703", "0.323"],
    ["Compact Riemann teacher vs Compact CE 2 s", "-0.0017", "6/1/2", "0.8057", "0.7422", "-0.085"],
    ["EEGNetV4 dual+spatial vs EEGNetV4 CE 2 s", "+0.0024", "5/0/4", "0.7570", "0.5703", "0.107"],
]

TABLE_4 = [
    ["Method", "Window", "Accuracy", "Macro-F1", "Kappa"],
    ["Compact EEGNet CE", "2 s", "0.7635", "0.7568", "0.5270"],
    ["Dual teacher KD", "2 s", "0.7677", "0.7610", "0.5354"],
    ["EEGNetV4 CE", "2 s", "0.7714", "0.7676", "0.5427"],
    ["Compact EEGNet CE", "4 s", "0.7806", "0.7763", "0.5612"],
    ["EEGNetV4 CE", "4 s", "0.7895", "0.7852", "0.5790"],
    ["Riemann TS-LogReg", "4 s", "0.7297", "0.7189", "0.4593"],
]


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=80, start=120, bottom=80, end=120) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{margin}"))
        if node is None:
            node = OxmlElement(f"w:{margin}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def set_table_widths(table, widths_inches: list[float]) -> None:
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for row in table.rows:
        for idx, width in enumerate(widths_inches):
            row.cells[idx].width = Inches(width)
            tc_pr = row.cells[idx]._tc.get_or_add_tcPr()
            tc_w = tc_pr.find(qn("w:tcW"))
            if tc_w is None:
                tc_w = OxmlElement("w:tcW")
                tc_pr.append(tc_w)
            tc_w.set(qn("w:w"), str(int(width * 1440)))
            tc_w.set(qn("w:type"), "dxa")

    tbl_pr = table._tbl.tblPr
    tbl_w = tbl_pr.find(qn("w:tblW"))
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(int(sum(widths_inches) * 1440)))
    tbl_w.set(qn("w:type"), "dxa")

    tbl_ind = tbl_pr.find(qn("w:tblInd"))
    if tbl_ind is None:
        tbl_ind = OxmlElement("w:tblInd")
        tbl_pr.append(tbl_ind)
    tbl_ind.set(qn("w:w"), "120")
    tbl_ind.set(qn("w:type"), "dxa")


def set_run_font(run, size=11, bold=False, italic=False, color=BLACK, name="Calibri") -> None:
    run.font.name = name
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), name)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = RGBColor.from_string(color)


def add_page_number(paragraph) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = paragraph.add_run()
    fld_char1 = OxmlElement("w:fldChar")
    fld_char1.set(qn("w:fldCharType"), "begin")
    instr_text = OxmlElement("w:instrText")
    instr_text.set(qn("xml:space"), "preserve")
    instr_text.text = "PAGE"
    fld_char2 = OxmlElement("w:fldChar")
    fld_char2.set(qn("w:fldCharType"), "end")
    run._r.extend([fld_char1, instr_text, fld_char2])
    set_run_font(run, size=9, color=MID_GRAY)


def configure_document(doc: Document) -> None:
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)
    section.header_distance = Inches(0.492)
    section.footer_distance = Inches(0.492)

    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Calibri"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
    normal.font.size = Pt(11)
    normal.font.color.rgb = RGBColor.from_string(BLACK)
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.10

    for style_name, size, color, before, after in [
        ("Heading 1", 16, BLUE, 16, 8),
        ("Heading 2", 13, BLUE, 12, 6),
        ("Heading 3", 12, DARK_BLUE, 8, 4),
    ]:
        style = styles[style_name]
        style.font.name = "Calibri"
        style._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
        style._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(color)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True

    caption = styles["Caption"]
    caption.font.name = "Calibri"
    caption._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
    caption._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
    caption.font.size = Pt(9)
    caption.font.italic = True
    caption.font.color.rgb = RGBColor.from_string(MID_GRAY)
    caption.paragraph_format.space_before = Pt(4)
    caption.paragraph_format.space_after = Pt(6)

    header = section.header.paragraphs[0]
    header.text = "Short-window MI-EEG decoding | Manuscript draft"
    header.alignment = WD_ALIGN_PARAGRAPH.LEFT
    for run in header.runs:
        set_run_font(run, size=9, color=MID_GRAY)

    footer = section.footer.paragraphs[0]
    add_page_number(footer)


def add_title_page(doc: Document) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(44)
    p.paragraph_format.space_after = Pt(14)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = p.add_run(TITLE)
    set_run_font(run, size=24, bold=True, color=DARK_BLUE)

    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(24)
    run = p.add_run(
        "A multi-seed, cross-session evaluation on BCI Competition IV datasets 2a and 2b"
    )
    set_run_font(run, size=13, color=MID_GRAY)

    for label, value in [
        ("Authors", "[Author name(s)]"),
        ("Affiliations", "[Department, Institution, City, Country]"),
        ("Corresponding author", "[Name and email]"),
        ("Target journal", "Medical & Biological Engineering & Computing"),
        ("Document status", "Complete initial manuscript based on final experiment matrix"),
    ]:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(3)
        r1 = p.add_run(f"{label}: ")
        set_run_font(r1, bold=True, color=DARK_BLUE)
        r2 = p.add_run(value)
        set_run_font(r2)

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(24)
    p.paragraph_format.space_after = Pt(8)
    set_cell = None
    r = p.add_run("Scope note")
    set_run_font(r, size=11, bold=True, color=DARK_BLUE)
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.18)
    p.paragraph_format.right_indent = Inches(0.18)
    p.paragraph_format.space_after = Pt(0)
    r = p.add_run(
        "This draft deliberately avoids a state-of-the-art claim. Its central claim is that "
        "dual long-window supervision improves a compact 2-s student on BCI IV-2a, while the "
        "benefit is limited for a stronger backbone and for BCI IV-2b."
    )
    set_run_font(r, size=10.5, italic=True, color=MID_GRAY)

    doc.add_page_break()


def add_heading(doc: Document, text: str, level: int = 1) -> None:
    doc.add_heading(text, level=level)


def add_body(doc: Document, text: str, italic=False, centered=False) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER if centered else WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.widow_control = True
    r = p.add_run(text)
    set_run_font(r, italic=italic)


def add_bullets(doc: Document, items: list[str]) -> None:
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.left_indent = Inches(0.5)
        p.paragraph_format.first_line_indent = Inches(-0.25)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.10
        r = p.add_run(item)
        set_run_font(r)


def add_table(doc: Document, title: str, rows: list[list[str]], widths: list[float]) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(title)
    set_run_font(r, size=10, bold=True, color=DARK_BLUE)

    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_widths(table, widths)
    set_repeat_table_header(table.rows[0])
    for row_idx, row_data in enumerate(rows):
        for col_idx, value in enumerate(row_data):
            cell = table.cell(row_idx, col_idx)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_margins(cell)
            if row_idx == 0:
                set_cell_shading(cell, LIGHT_BLUE)
            p = cell.paragraphs[0]
            p.alignment = (
                WD_ALIGN_PARAGRAPH.LEFT
                if col_idx == 0
                else WD_ALIGN_PARAGRAPH.CENTER
            )
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            r = p.add_run(str(value))
            set_run_font(r, size=8.5, bold=(row_idx == 0), color=(DARK_BLUE if row_idx == 0 else BLACK))
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def add_figure(
    doc: Document,
    image_path: Path,
    caption: str,
    alt_text: str,
    width=6.25,
) -> None:
    if not image_path.exists():
        add_body(doc, f"[Figure unavailable: {image_path.name}]", italic=True)
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    run = p.add_run()
    shape = run.add_picture(str(image_path), width=Inches(width))
    doc_pr = shape._inline.docPr
    doc_pr.set("descr", alt_text)
    doc_pr.set("title", image_path.stem)
    cp = doc.add_paragraph(style="Caption")
    cp.alignment = WD_ALIGN_PARAGRAPH.LEFT
    cp.paragraph_format.keep_with_next = False
    cp.add_run(caption)


def build_docx() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    doc = Document()
    configure_document(doc)
    add_title_page(doc)

    add_heading(doc, "Abstract", 1)
    add_body(doc, ABSTRACT)
    p = doc.add_paragraph()
    r = p.add_run("Keywords: ")
    set_run_font(r, bold=True, color=DARK_BLUE)
    r = p.add_run(
        "brain-computer interface; electroencephalography; motor imagery; "
        "knowledge distillation; Riemannian geometry; short-window decoding"
    )
    set_run_font(r)

    add_heading(doc, "1 Introduction", 1)
    for paragraph in INTRODUCTION:
        add_body(doc, paragraph)
    add_body(doc, "The contributions of this work are:", italic=True)
    add_bullets(doc, CONTRIBUTIONS)

    add_heading(doc, "2 Materials and methods", 1)
    for heading, paragraphs in METHODS.items():
        add_heading(doc, heading, 2)
        for paragraph in paragraphs:
            add_body(doc, paragraph)
        if heading == "2.1 Datasets":
            add_table(
                doc,
                "Table 1. Dataset and cross-session evaluation protocol.",
                TABLE_1,
                [1.65, 0.8, 0.75, 0.75, 2.55],
            )

    add_heading(doc, "3 Results", 1)
    for heading, paragraphs in RESULTS.items():
        add_heading(doc, heading, 2)
        for paragraph in paragraphs:
            add_body(doc, paragraph)
        if heading == "3.1 Main BCI Competition IV-2a result":
            add_table(
                doc,
                "Table 2. BCI IV-2a results averaged over nine subjects and five seeds (45 runs per method).",
                TABLE_2,
                [2.75, 0.65, 1.0, 1.0, 1.0],
            )
            add_figure(
                doc,
                FIGURES["compact"],
                "Fig. 1 Compact EEGNet ablation results on BCI IV-2a. Bars summarize the final multi-seed experiment matrix.",
                "Bar chart comparing Compact EEGNet baseline and teacher-distillation ablations on BCI Competition IV-2a.",
            )
        elif heading == "3.2 Ablation and strong-backbone results":
            add_table(
                doc,
                "Table 3. Subject-mean paired comparisons on BCI IV-2a.",
                TABLE_3,
                [2.55, 0.85, 1.05, 0.75, 0.75, 0.7],
            )
        elif heading == "3.4 Cross-dataset evaluation on BCI Competition IV-2b":
            add_table(
                doc,
                "Table 4. BCI IV-2b results averaged over nine subjects and five seeds.",
                TABLE_4,
                [2.75, 0.65, 1.0, 1.0, 1.0],
            )
            add_figure(
                doc,
                FIGURES["bci2b"],
                "Fig. 2 Method accuracy on BCI IV-2b. The dual-teacher gain over Compact EEGNet 2s was small and non-significant.",
                "Bar chart comparing neural and classical method accuracy on BCI Competition IV-2b.",
            )
        elif heading == "3.5 Class-wise errors and teacher quality":
            add_figure(
                doc,
                FIGURES["confusion"],
                "Fig. 3 Aggregated BCI IV-2a confusion matrices from checkpoint re-inference across five seeds and nine subjects.",
                "Two confusion matrices comparing Compact EEGNet 2-second cross-entropy training with dual-teacher distillation on BCI IV-2a.",
            )

    add_heading(doc, "4 Discussion", 1)
    for paragraph in DISCUSSION:
        add_body(doc, paragraph)

    add_heading(doc, "4.1 Limitations", 2)
    add_bullets(doc, LIMITATIONS)

    add_heading(doc, "5 Conclusion", 1)
    add_body(doc, CONCLUSION)

    add_heading(doc, "Declarations", 1)
    declaration_rows = [
        ("Funding", "[Insert funding statement or: No funding was received for this work.]"),
        ("Competing interests", "The authors declare no competing interests."),
        ("Ethics approval", "This study used publicly available, de-identified datasets and collected no new participant data. The original dataset publications should be cited for ethical approval and consent."),
        ("Consent to participate", "Not applicable to this secondary analysis."),
        ("Data availability", "BCI Competition IV datasets 2a and 2b are publicly available through BNCI Horizon and MOABB."),
        ("Code availability", "Code is available at https://github.com/zfqa/eegnetfgl after repository cleanup and release tagging."),
        ("Author contributions", "[Insert CRediT contribution statement.]"),
        ("Acknowledgements", "[Insert acknowledgements, if applicable.]"),
    ]
    for label, value in declaration_rows:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(5)
        r = p.add_run(f"{label}: ")
        set_run_font(r, bold=True, color=DARK_BLUE)
        r = p.add_run(value)
        set_run_font(r)

    add_heading(doc, "References", 1)
    for idx, reference in enumerate(REFERENCES, 1):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.first_line_indent = Inches(-0.25)
        p.paragraph_format.space_after = Pt(5)
        r = p.add_run(f"{idx}. {reference}")
        set_run_font(r, size=9.5)

    doc.save(DOCX_PATH)


def build_markdown() -> None:
    lines: list[str] = []
    lines.extend(
        [
            f"# {TITLE}",
            "",
            "**Authors:** [Author name(s)]",
            "**Affiliations:** [Department, Institution, City, Country]",
            "**Corresponding author:** [Name and email]",
            "",
            "## Abstract",
            "",
            ABSTRACT,
            "",
            "**Keywords:** brain-computer interface; electroencephalography; motor imagery; knowledge distillation; Riemannian geometry; short-window decoding",
            "",
            "## 1 Introduction",
            "",
        ]
    )
    for paragraph in INTRODUCTION:
        lines.extend([paragraph, ""])
    lines.append("The contributions of this work are:")
    lines.append("")
    for item in CONTRIBUTIONS:
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## 2 Materials and methods")
    lines.append("")
    for heading, paragraphs in METHODS.items():
        lines.extend([f"### {heading}", ""])
        for paragraph in paragraphs:
            lines.extend([paragraph, ""])

    lines.append("## 3 Results")
    lines.append("")
    for heading, paragraphs in RESULTS.items():
        lines.extend([f"### {heading}", ""])
        for paragraph in paragraphs:
            lines.extend([paragraph, ""])

    lines.extend(["## 4 Discussion", ""])
    for paragraph in DISCUSSION:
        lines.extend([paragraph, ""])
    lines.extend(["### 4.1 Limitations", ""])
    for item in LIMITATIONS:
        lines.append(f"- {item}")
    lines.extend(["", "## 5 Conclusion", "", CONCLUSION, "", "## Declarations", ""])
    lines.extend(
        [
            "**Funding:** [Insert funding statement.]",
            "",
            "**Competing interests:** The authors declare no competing interests.",
            "",
            "**Ethics approval:** This study used publicly available, de-identified datasets and collected no new participant data.",
            "",
            "**Code availability:** https://github.com/zfqa/eegnetfgl",
            "",
            "## References",
            "",
        ]
    )
    for idx, reference in enumerate(REFERENCES, 1):
        lines.append(f"{idx}. {reference}")
    lines.append("")
    MD_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    build_docx()
    build_markdown()
    print(DOCX_PATH)
    print(MD_PATH)
