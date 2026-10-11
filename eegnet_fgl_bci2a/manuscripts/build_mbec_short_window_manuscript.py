"""Build an MBE&C-oriented manuscript and reproducible figures for short-window MI-EEG decoding.

All numeric values are transcribed from completed experiment summaries in the
project manuscript and user-verified five-seed summaries.  This script keeps
the figure source data beside the manuscript so visual claims remain auditable.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "manuscripts" / "mbec_short_window_paper"
FIG_DIR = OUT / "figures"
DATA_DIR = OUT / "source_data"
MD_PATH = OUT / "short_window_mi_eeg_mbec_manuscript_en.md"
DOCX_PATH = OUT / "short_window_mi_eeg_mbec_manuscript_en.docx"


COLORS = {
    "baseline": "#737373",
    "fixed": "#7E9CC7",
    "ours": "#0F4D92",
    "spatial": "#42949E",
    "reference": "#C9C9C9",
    "positive": "#2E9E44",
    "accent": "#B64342",
    "ink": "#272727",
}


def set_plot_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = 8
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["legend.frameon"] = False


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in {
        ".svg": {},
        ".pdf": {},
        ".png": {"dpi": 300},
        ".tiff": {"dpi": 600},
    }.items():
        fig.savefig(FIG_DIR / f"{stem}{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def label_panel(ax: plt.Axes, letter: str) -> None:
    ax.text(-0.11, 1.04, letter, transform=ax.transAxes, fontweight="bold", fontsize=10)


def write_source_data() -> dict[str, pd.DataFrame]:
    main = pd.DataFrame(
        [
            ["BCI IV-2a", "Fixed 2-s CE", 2.0, 0.5672, 0.0093, 0.5550, 0.4229, "candidate"],
            ["BCI IV-2a", "Fixed 2-s dual teacher", 2.0, 0.5847, 0.0125, 0.5724, 0.4463, "candidate"],
            ["BCI IV-2a", "Validation-controlled routing", 2.0, 0.5931, 0.0114, 0.5842, 0.4575, "primary"],
            ["BCI IV-2a", "Compact EEGNet CE", 4.0, 0.5836, np.nan, 0.5749, 0.4448, "reference"],
            ["BCI IV-2b", "Fixed 2-s CE", 2.0, 0.7634, 0.0120, 0.7567, 0.5268, "candidate"],
            ["BCI IV-2b", "Fixed Riemann KD + spatial", 2.0, 0.7660, 0.0142, 0.7627, 0.5319, "candidate"],
            ["BCI IV-2b", "Adaptive CE", 2.0, 0.7675, 0.0031, 0.7637, 0.5349, "candidate"],
            ["BCI IV-2b", "Adaptive Riemann KD + spatial", 2.0, 0.7712, 0.0152, 0.7681, 0.5424, "primary"],
            ["BCI IV-2b", "Validation-controlled routing", 2.0, 0.7701, 0.0103, 0.7665, 0.5402, "diagnostic"],
            ["BCI IV-2b", "Compact EEGNet CE", 4.0, 0.7806, np.nan, 0.7763, 0.5612, "reference"],
        ],
        columns=["dataset", "method", "window_s", "accuracy", "seed_sd", "macro_f1", "kappa", "role"],
    )
    baselines = pd.DataFrame(
        [
            ["BCI IV-2a", "ShallowConvNet", 2.0, 0.5187, 0.0026, 0.5113, 0.3582],
            ["BCI IV-2a", "DeepConvNet", 2.0, 0.4863, 0.0131, 0.4553, 0.3151],
            ["BCI IV-2a", "CSP-LDA", 2.0, 0.5530, np.nan, 0.5270, 0.4040],
            ["BCI IV-2a", "FBCSP-LDA", 2.0, 0.5638, np.nan, 0.5517, 0.4184],
            ["BCI IV-2a", "Riemann TS-LogReg", 2.0, 0.5694, np.nan, 0.5570, 0.4259],
            ["BCI IV-2a", "EEGNetV4 CE", 2.0, 0.5809, np.nan, 0.5666, 0.4412],
            ["BCI IV-2a", "Validation-controlled routing", 2.0, 0.5931, 0.0114, 0.5842, 0.4575],
            ["BCI IV-2b", "ShallowConvNet", 2.0, 0.7477, 0.0073, 0.7443, 0.4954],
            ["BCI IV-2b", "DeepConvNet", 2.0, 0.7487, 0.0118, 0.7429, 0.4973],
            ["BCI IV-2b", "CSP-LDA", 2.0, 0.7136, np.nan, 0.7076, 0.4272],
            ["BCI IV-2b", "FBCSP-LDA", 2.0, 0.7258, np.nan, 0.7227, 0.4515],
            ["BCI IV-2b", "Riemann TS-LogReg", 2.0, 0.7159, np.nan, 0.7080, 0.4317],
            ["BCI IV-2b", "EEGNetV4 CE", 2.0, 0.7714, np.nan, 0.7676, 0.5427],
            ["BCI IV-2b", "Adaptive Riemann KD + spatial", 2.0, 0.7712, 0.0152, 0.7681, 0.5424],
        ],
        columns=["dataset", "method", "window_s", "accuracy", "seed_sd", "macro_f1", "kappa"],
    )
    long_window = pd.DataFrame(
        [
            ["BCI IV-2a", "ShallowConvNet", 0.5038, 0.4923, 0.3384],
            ["BCI IV-2a", "DeepConvNet", 0.4710, 0.4372, 0.2947],
            ["BCI IV-2a", "Compact EEGNet CE", 0.5836, 0.5749, 0.4448],
            ["BCI IV-2a", "EEGNetV4 CE", 0.6025, 0.5894, 0.4700],
            ["BCI IV-2a", "Riemann TS-LogReg", 0.6525, 0.6420, 0.5367],
            ["BCI IV-2a", "CSP-LDA", 0.6029, 0.5726, 0.4705],
            ["BCI IV-2a", "FBCSP-LDA", 0.6296, 0.6154, 0.5061],
            ["BCI IV-2b", "ShallowConvNet", 0.7666, 0.7619, 0.5332],
            ["BCI IV-2b", "DeepConvNet", 0.7741, 0.7639, 0.5482],
            ["BCI IV-2b", "Compact EEGNet CE", 0.7806, 0.7763, 0.5612],
            ["BCI IV-2b", "EEGNetV4 CE", 0.7895, 0.7852, 0.5790],
            ["BCI IV-2b", "Riemann TS-LogReg", 0.7297, 0.7189, 0.4593],
            ["BCI IV-2b", "CSP-LDA", 0.7303, 0.7197, 0.4606],
            ["BCI IV-2b", "FBCSP-LDA", 0.7587, 0.7540, 0.5173],
        ],
        columns=["dataset", "method", "accuracy", "macro_f1", "kappa"],
    )
    ablation = pd.DataFrame(
        [
            ["Riemann KD + spatial", "0.05 + 0.05", 0.10, 0.7589, 0.1420, 0.7525, 0.1494, 0.5177, 0.2840],
            ["Riemann KD + spatial", "0.10 + 0.10", 0.20, 0.7648, 0.1354, 0.7617, 0.1373, 0.5296, 0.2709],
            ["Riemann KD only", "0.20 + 0.00", 0.20, 0.7608, 0.1373, 0.7543, 0.1458, 0.5216, 0.2745],
            ["Spatial only", "0.00 + 0.20", 0.20, 0.7614, 0.1421, 0.7551, 0.1499, 0.5229, 0.2841],
        ],
        columns=["method", "lambda_r_plus_lambda_s", "total_aux_weight", "accuracy", "subject_sd_accuracy", "macro_f1", "subject_sd_macro_f1", "kappa", "subject_sd_kappa"],
    )
    tables = {"main_results": main, "short_window_baselines": baselines, "long_window_context": long_window, "spatial_ablation": ablation}
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(DATA_DIR / f"{name}.csv", index=False)
    return tables


def fig1_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.text(0.2, 4.58, "Training-time privileged information", fontsize=9, fontweight="bold", color=COLORS["ink"])
    ax.text(9.2, 4.58, "Deployment", fontsize=9, fontweight="bold", color=COLORS["ink"])
    y = 3.45
    ax.add_patch(FancyBboxPatch((0.35, y), 3.55, 0.65, boxstyle="round,pad=0.03", facecolor="#E8EEF5", edgecolor=COLORS["ink"], linewidth=0.8))
    ax.add_patch(FancyBboxPatch((0.35, y), 1.78, 0.65, boxstyle="round,pad=0.03", facecolor="#F2C9A5", edgecolor="none"))
    ax.text(1.24, y + 0.33, "real 0-2 s", ha="center", va="center", fontsize=8, fontweight="bold")
    ax.text(3.0, y + 0.33, "privileged 2-4 s", ha="center", va="center", fontsize=8)
    ax.text(0.35, 2.95, "4-s training trial", fontsize=7.5, color=COLORS["neutral"] if "neutral" in COLORS else COLORS["ink"])
    teacher_boxes = [
        (4.65, 3.2, "4-s Compact EEGNet\nteacher (2a)", COLORS["fixed"]),
        (4.65, 1.65, "4-s Riemann teacher\nOAS -> tangent space -> LR", COLORS["spatial"]),
    ]
    for x, yy, label, color in teacher_boxes:
        ax.add_patch(FancyBboxPatch((x, yy), 2.7, 0.82, boxstyle="round,pad=0.06", facecolor=color, alpha=0.22, edgecolor=color, linewidth=1.0))
        ax.text(x + 1.35, yy + 0.41, label, ha="center", va="center", fontsize=7.5)
        ax.add_patch(FancyArrowPatch((3.95, y + 0.31), (x, yy + 0.41), arrowstyle="-|>", mutation_scale=10, linewidth=0.9, color=COLORS["ink"]))
    ax.add_patch(FancyBboxPatch((8.0, 2.25), 2.25, 1.2, boxstyle="round,pad=0.06", facecolor="#DDEAF8", edgecolor=COLORS["ours"], linewidth=1.2))
    ax.text(9.12, 2.85, "2-s Compact EEGNet\nstudent", ha="center", va="center", fontsize=8, fontweight="bold")
    ax.text(9.12, 2.43, "CE + training-only guidance", ha="center", va="center", fontsize=6.8)
    for yy in (3.61, 2.06):
        ax.add_patch(FancyArrowPatch((7.38, yy), (7.98, 2.95), arrowstyle="-|>", mutation_scale=10, linewidth=0.9, color=COLORS["ink"]))
    ax.add_patch(FancyArrowPatch((10.32, 2.85), (11.12, 2.85), arrowstyle="-|>", mutation_scale=10, linewidth=1.1, color=COLORS["ours"]))
    ax.add_patch(FancyBboxPatch((11.18, 2.25), 2.35, 1.2, boxstyle="round,pad=0.06", facecolor="#F2C9A5", edgecolor="#C37A32", linewidth=1.1))
    ax.text(12.35, 2.85, "2-s EEG only\nclassification", ha="center", va="center", fontsize=8, fontweight="bold")
    ax.text(12.35, 2.43, "no teacher at inference", ha="center", va="center", fontsize=6.8)
    ax.text(0.35, 0.65, "2a: validation-controlled selection among fixed/adaptive CE and dual-teacher candidates", fontsize=7.1)
    ax.text(0.35, 0.25, "2b: Riemann KD + spatial supervision; adaptive routing retained as a diagnostic analysis", fontsize=7.1)
    save_figure(fig, "figure_1_pipeline")


def fig2_primary(main: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1, 1]})
    subset = main[main.dataset.eq("BCI IV-2a")].set_index("method")
    order = ["Fixed 2-s CE", "Fixed 2-s dual teacher", "Validation-controlled routing", "Compact EEGNet CE"]
    labels = ["Fixed\n2-s CE", "Fixed\ndual", "2-s\nrouting", "4-s\nCompact"]
    vals = [subset.loc[x, "accuracy"] * 100 for x in order]
    colors = [COLORS["baseline"], COLORS["fixed"], COLORS["ours"], COLORS["reference"]]
    bars = axes[0].bar(range(len(vals)), vals, color=colors, edgecolor="white", linewidth=0.7)
    axes[0].set_xticks(range(len(vals)), labels)
    axes[0].set_ylim(50, 62)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("BCI IV-2a: four-class, 22 channels", loc="left", fontsize=8.5, fontweight="bold")
    for bar, val in zip(bars, vals):
        axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.35, f"{val:.2f}", ha="center", fontsize=7)
    axes[0].annotate("+2.59 pp\np=0.0055", xy=(2, vals[2]), xytext=(0.62, 61.1), textcoords="data", arrowprops={"arrowstyle": "-", "lw": 0.8}, fontsize=7, ha="center")
    label_panel(axes[0], "a")
    subset = main[main.dataset.eq("BCI IV-2b")].set_index("method")
    order = ["Fixed 2-s CE", "Fixed Riemann KD + spatial", "Adaptive Riemann KD + spatial", "Validation-controlled routing", "Compact EEGNet CE"]
    labels = ["Fixed\nCE", "Fixed\nR+S", "Adaptive\nR+S", "2-s\nrouting", "4-s\nCompact"]
    vals = [subset.loc[x, "accuracy"] * 100 for x in order]
    colors = [COLORS["baseline"], COLORS["spatial"], COLORS["ours"], COLORS["fixed"], COLORS["reference"]]
    bars = axes[1].bar(range(len(vals)), vals, color=colors, edgecolor="white", linewidth=0.7)
    axes[1].set_xticks(range(len(vals)), labels)
    axes[1].set_ylim(72, 80)
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("BCI IV-2b: binary, 3 channels", loc="left", fontsize=8.5, fontweight="bold")
    for bar, val in zip(bars, vals):
        axes[1].text(bar.get_x() + bar.get_width()/2, val + 0.22, f"{val:.2f}", ha="center", fontsize=6.6)
    axes[1].annotate("small positive\nshort-window gain", xy=(2, vals[2]), xytext=(2.25, 79.0), arrowprops={"arrowstyle": "-", "lw": 0.8}, fontsize=7, ha="center")
    label_panel(axes[1], "b")
    save_figure(fig, "figure_2_primary_results")


def fig3_ablation(ablation: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.15, 0.85]})
    labels = ["R+S\n0.05+0.05", "R+S\n0.10+0.10", "R only\n0.20", "S only\n0.20"]
    vals = ablation.accuracy.to_numpy() * 100
    colors = [COLORS["fixed"], COLORS["ours"], COLORS["spatial"], "#A8C9C6"]
    bars = axes[0].bar(range(4), vals, color=colors, edgecolor="white")
    axes[0].set_xticks(range(4), labels)
    axes[0].set_ylim(74, 78)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("BCI IV-2b auxiliary-loss ablation", loc="left", fontsize=8.5, fontweight="bold")
    for bar, val in zip(bars, vals):
        axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.14, f"{val:.2f}", ha="center", fontsize=7)
    label_panel(axes[0], "a")
    components = ["Riemann KD", "Spatial loss", "Joint"]
    gains = [0.7608 - 0.7589, 0.7614 - 0.7589, 0.7648 - 0.7589]
    bars = axes[1].barh(components, np.array(gains) * 100, color=[COLORS["spatial"], "#A8C9C6", COLORS["ours"]])
    axes[1].axvline(0, color=COLORS["ink"], linewidth=0.7)
    axes[1].set_xlabel("Gain over 0.05+0.05 (pp)")
    axes[1].set_xlim(-0.1, 0.7)
    axes[1].set_title("Moderate joint weighting was highest", loc="left", fontsize=8.5, fontweight="bold")
    for bar, gain in zip(bars, gains):
        axes[1].text(gain * 100 + 0.02, bar.get_y() + bar.get_height()/2, f"+{gain*100:.2f}", va="center", fontsize=7)
    label_panel(axes[1], "b")
    save_figure(fig, "figure_4_spatial_ablation")


def fig4_baselines(baselines: pd.DataFrame, long_window: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.15))
    for ax, dataset, panel in zip(axes, ["BCI IV-2a", "BCI IV-2b"], ["a", "b"]):
        data = baselines[baselines.dataset.eq(dataset)].sort_values("accuracy")
        y = np.arange(len(data))
        colors = [COLORS["ours"] if ("routing" in m.lower() or "Adaptive Riemann" in m) else COLORS["baseline"] for m in data.method]
        bars = ax.barh(y, data.accuracy * 100, color=colors, edgecolor="white")
        labels = data.method.tolist()
        labels = [
            "Validation-controlled\nrouting" if label == "Validation-controlled routing" else
            "Adaptive Riemann KD\n+ spatial" if label == "Adaptive Riemann KD + spatial" else
            label
            for label in labels
        ]
        ax.set_yticks(y, labels, fontsize=6.4)
        ax.set_xlim((45, 62) if dataset.endswith("2a") else (68, 79))
        ax.set_xlabel("2-s accuracy (%)")
        ax.set_title(dataset, loc="left", fontsize=8.5, fontweight="bold")
        for bar, val in zip(bars, data.accuracy * 100):
            ax.text(val + 0.18, bar.get_y() + bar.get_height()/2, f"{val:.2f}", va="center", fontsize=6.5)
        label_panel(ax, panel)
    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.17, wspace=0.72)
    fig.text(0.5, 0.035, "Blue bars: proposed short-window strategy. Gray bars: reproduced external baselines under the same cross-session protocol.", ha="center", fontsize=7)
    save_figure(fig, "figure_3_short_window_baselines")


def fmt_metric(value: float, sd: float | None = None) -> str:
    if sd is None or pd.isna(sd):
        return f"{value:.4f}"
    return f"{value:.4f} +/- {sd:.4f}"


def markdown(tables: dict[str, pd.DataFrame]) -> str:
    main, baselines, long_window, ablation = (tables[k] for k in ("main_results", "short_window_baselines", "long_window_context", "spatial_ablation"))
    table1 = """| Dataset | Classes | EEG channels | Sampling rate | Outer protocol |
| --- | ---: | ---: | ---: | --- |
| BCI Competition IV-2a / BNCI2014_001 | 4 | 22 | 250 Hz | Training session -> train/validation; evaluation session -> test |
| BCI Competition IV-2b / BNCI2014_004 | 2 | 3 | 250 Hz | Training session -> train/validation; evaluation session -> test |"""
    main_rows = []
    for row in main.itertuples(index=False):
        main_rows.append(f"| {row.dataset} | {row.method} | {row.window_s:g} s | {fmt_metric(row.accuracy, row.seed_sd)} | {row.macro_f1:.4f} | {row.kappa:.4f} |")
    table2 = "| Dataset | Method | Input | Accuracy | Macro-F1 | Kappa |\n| --- | --- | ---: | ---: | ---: | ---: |\n" + "\n".join(main_rows)
    base_rows = []
    for row in baselines.itertuples(index=False):
        base_rows.append(f"| {row.dataset} | {row.method} | {fmt_metric(row.accuracy, row.seed_sd)} | {row.macro_f1 if not pd.isna(row.macro_f1) else '--'} | {row.kappa if not pd.isna(row.kappa) else '--'} |")
    table3 = "| Dataset | 2-s comparator | Accuracy | Macro-F1 | Kappa |\n| --- | --- | ---: | ---: | ---: |\n" + "\n".join(base_rows)
    long_rows = []
    for row in long_window.itertuples(index=False):
        long_rows.append(f"| {row.dataset} | {row.method} | {row.accuracy:.4f} | {row.macro_f1:.4f} | {row.kappa:.4f} |")
    table4 = "| Dataset | 4-s reference | Accuracy | Macro-F1 | Kappa |\n| --- | --- | ---: | ---: | ---: |\n" + "\n".join(long_rows)
    abl_rows = []
    for row in ablation.itertuples(index=False):
        abl_rows.append(f"| {row.method} ({row.lambda_r_plus_lambda_s}) | {row.total_aux_weight:.2f} | {row.accuracy:.4f} +/- {row.subject_sd_accuracy:.4f} | {row.macro_f1:.4f} +/- {row.subject_sd_macro_f1:.4f} | {row.kappa:.4f} +/- {row.subject_sd_kappa:.4f} |")
    table5 = "| Configuration | Total auxiliary weight | Accuracy | Macro-F1 | Kappa |\n| --- | ---: | ---: | ---: | ---: |\n" + "\n".join(abl_rows)
    refs = extract_references()
    return dedent(f"""# Short-window motor-imagery EEG decoding with training-time privileged information and validation-controlled routing

## Abstract

Motor-imagery EEG (MI-EEG) decoders often use several seconds of post-cue data, which improves evidence accumulation but delays a decision. We investigated whether a student decoder restricted to the first 2 s can benefit from information available only during training. The framework used 4-s neural and Riemannian teachers in the four-class BCI Competition IV-2a setting and Riemannian soft-label and spatial supervision in the binary BCI Competition IV-2b setting. Candidate short-window strategies were selected using only a validation partition of the training session; the independent evaluation session was never used for teacher fitting, model selection, or routing. Across nine participants and five random seeds, validation-controlled routing on IV-2a achieved 59.31% accuracy, compared with 56.72% for fixed 2-s cross-entropy training (mean paired difference 2.59 percentage points; paired t-test p=0.0055; Wilcoxon p=0.0078). On IV-2b, adaptive Riemannian knowledge distillation with spatial supervision achieved the highest short-window mean accuracy (77.12%), whereas routing reached 77.01% and fixed 2-s cross-entropy reached 76.34%. The smaller IV-2b gain indicates that the benefit depends on task complexity and available spatial information. Teachers and Riemannian targets were removed at inference, leaving only a 2-s Compact EEGNet and, when selected, a temporal detector. These results support training-time privileged information and validation-controlled strategy selection as a route to earlier MI-EEG decisions, but not as a universal replacement for complete-window decoding.

**Keywords:** motor imagery; electroencephalography; brain-computer interface; short-window decoding; knowledge distillation; Riemannian geometry; EEGNet

## 1 Introduction

Motor-imagery brain-computer interfaces (BCIs) infer an intended movement from EEG and can support communication, rehabilitation, and device control. A practical decoder must balance accuracy against the amount of signal that must be acquired before a decision. Longer post-cue windows provide more temporal evidence, but they also impose an unavoidable waiting period. This makes short-window decoding a biomedical engineering problem rather than merely a smaller-input classification problem.

Compact convolutional networks and Riemannian covariance methods provide complementary views of MI-EEG. EEGNet learns temporal and spatial filters directly from waveform segments, whereas Riemannian approaches represent each trial by a channel covariance matrix and exploit its geometry. Knowledge distillation further provides a way to expose a short-window model to training-time information that will be unavailable at deployment. However, a single teacher or a single temporal policy can improve some participants while degrading others.

We therefore formulate early MI-EEG decoding as a training-time privileged-information problem followed by validation-controlled strategy selection. A 2-s student is the only classifier used during deployment. During training, 4-s neural and Riemannian teachers can supply class-distribution targets; for IV-2b, the Riemannian covariance structure additionally constrains the student spatial representation. On IV-2a, four candidate strategies are compared per participant and seed, and a validation set from the training session selects the candidate tested on the held-out evaluation session.

The study makes three bounded contributions. First, it demonstrates that validation-controlled routing significantly improves 2-s decoding on the more difficult four-class, 22-channel IV-2a task. Second, it shows that Riemannian spatial supervision is a more suitable short-window training prior than neural dual-teacher distillation on the three-channel IV-2b task, although the gain is small. Third, it quantifies the position of the proposed short-window methods relative to reproduced classical, compact, shallow, and deep neural baselines. The results are intended to support earlier decisions under a 2-s signal budget, not to claim that 2 s universally outperforms 4-s decoding.

## 2 Materials and methods

### 2.1 Datasets and cross-session protocol

{table1}

Both datasets were obtained through MOABB. For every participant, the training session was stratified into a training subset and a validation subset (80/20), while the evaluation session remained an independent test set. Five seeds (42, 2024, 3407, 7, and 123) controlled the training/validation split and neural optimization. All preprocessing parameters, teacher fitting, checkpoint selection, and route selection used only the training session.

### 2.2 Signal preprocessing and short-window task

Signals were band-pass filtered from 8 to 32 Hz and cropped to 0-4 s after the cue. Per-channel standardization was fitted on the training subset and applied unchanged to validation and test data. The fixed student input was the real 0-2 s segment. The 4-s signal was available only to training-time teachers and reference models. Student and teacher inputs for a batch always originated from the same trial, preserving labels and sample order.

### 2.3 Training-time privileged information

All student classifiers used Compact EEGNet. On IV-2a, a frozen 4-s Compact EEGNet teacher and a 4-s Riemannian teacher produced soft class targets. The latter used OAS covariance estimation, tangent-space mapping at the Riemannian mean, and logistic regression with the regularization constant selected within the training session. The dual-teacher objective was the sum of cross-entropy and neural/Riemannian Kullback-Leibler terms.

On IV-2b, the neural-Riemannian dual-teacher formulation was not retained as the main configuration because it did not consistently improve fixed 2-s cross-entropy. Instead, a 4-s Riemannian teacher supplied a soft-label term and its covariance-derived structure regularized the pre-depthwise spatial representation of the 2-s student. The loss was L = CE + lambda_R KL_R + lambda_S L_spatial, with lambda_R=0.1 and lambda_S=0.2 in the fixed main configuration. Figure 1 summarizes the separation between training-time privileged information and deployment-time computation.

### 2.4 Validation-controlled routing

For IV-2a, the candidate set comprised fixed 2-s cross-entropy, adaptive-window cross-entropy, fixed dual-teacher distillation, and adaptive dual-teacher distillation. Each candidate was evaluated on the same validation subset. The candidate with the highest validation accuracy was selected, with a conservative tie order of fixed CE, fixed dual teacher, adaptive CE, and adaptive dual teacher. The selected candidate was then evaluated once on the independent evaluation session. This is participant-level routing, not trial-level mixture-of-experts inference.

For IV-2b, fixed CE, adaptive CE, fixed Riemann KD plus spatial supervision, and adaptive Riemann KD plus spatial supervision were compared diagnostically. The detector used a 1-s sliding window with a 0.25-s step within the first 2 s. It selected a 2-s segment after confidence smoothing. Teachers and covariance targets were never called at inference.

### 2.5 Metrics and statistics

Accuracy, macro-F1, and Cohen's kappa were reported. For primary seed-level summaries, each seed was first averaged across nine participants, followed by the mean and standard deviation over five seeds. For the IV-2a primary comparison, per-participant values were first averaged over seeds and then compared with paired t-tests and Wilcoxon signed-rank tests. The fixed-versus-adaptive IV-2b Riemannian spatial comparisons are interpreted descriptively because corresponding paired tests have not yet been recomputed under the final spatial-supervision matrix.

## 3 Results

### 3.1 Validation-controlled routing improved short-window decoding on IV-2a

The principal result was observed on IV-2a (Figure 2a and Table 2). Fixed 2-s CE achieved 56.72% accuracy, whereas validation-controlled routing achieved 59.31%, an absolute gain of 2.59 percentage points. Routing improved eight of nine participants after seed averaging (8/0/1 wins/ties/losses). The difference from fixed CE was significant (paired t-test p=0.0055; Wilcoxon p=0.0078). Fixed dual-teacher training achieved 58.47%, but routing was not significantly higher than this best fixed candidate (p=0.1741). Thus, the evidence supports the value of selecting among short-window strategies across participants, rather than a claim that one candidate dominates every participant.

**Table 2 | Main short-window results.** Values with +/- report mean +/- standard deviation across five seeds after averaging each seed across nine participants. Entries without variability were retained from previously completed reference summaries.

{table2}

### 3.2 IV-2b showed a smaller, task-dependent short-window benefit

The pattern changed on IV-2b (Figure 2b). Fixed 2-s CE achieved 76.34% accuracy. Fixed Riemann KD plus spatial supervision increased the mean to 76.60%, while adaptive Riemann KD plus spatial supervision reached 77.12%. Validation-controlled routing reached 77.01%, slightly below the best individual candidate but with lower reported seed-level variability. These differences are modest compared with IV-2a. Because IV-2b is a binary task with three channels, the room for additional covariance-based spatial information and teacher complementarity is more limited. The results therefore support a small cross-dataset benefit rather than a universal superiority claim.

### 3.3 The proposed short-window methods were competitive with reproduced baselines

Figure 3 and Table 3 place the 2-s methods against reproduced classical and neural baselines. On IV-2a, routing exceeded ShallowConvNet (51.87%), DeepConvNet (48.63%), CSP-LDA (55.30%), FBCSP-LDA (56.38%), Riemannian tangent-space logistic regression (56.94%), and EEGNetV4 CE (58.09%). On IV-2b, adaptive Riemann KD plus spatial supervision (77.12%) was close to EEGNetV4 CE (77.14%) and exceeded the shallow/deep 2-s baselines (74.77% and 74.87%). The ShallowConvNet and DeepConvNet comparisons are descriptive because paired tests against the proposed methods have not yet been added.

**Table 3 | Reproduced 2-s external baselines.**

{table3}

### 3.4 Complete-window results remain important context

Four-second reference models generally retained an advantage on the strongest complete-window comparisons (Table 4). For example, EEGNetV4 reached 60.25% on IV-2a and 78.95% on IV-2b, while Riemannian TS-LogReg reached 65.25% on IV-2a. The short-window goal is therefore not to claim a universal replacement of 4-s decoding; it is to recover useful performance under a fixed 2-s evidence budget.

**Table 4 | Complete-window reference context. These models use 4 s of EEG and are not direct latency-matched comparators.**

{table4}

### 3.5 Riemannian soft-label and spatial supervision were complementary at moderate auxiliary weight

The IV-2b auxiliary-loss ablation is shown in Figure 4 and Table 5. At a total auxiliary weight of 0.20, the joint Riemann KD plus spatial configuration (0.10+0.10) reached 76.48%, exceeding Riemann KD alone (76.08%) and spatial supervision alone (76.14%). A lower-weight joint configuration (0.05+0.05) reached 75.89%. These values are participant-level summaries after averaging five seeds; their standard deviations reflect participant heterogeneity and should not be compared directly with the seed-level standard deviations in Table 2. The ablation suggests complementarity at moderate weights, but it is descriptive without a matched paired test against CE.

**Table 5 | IV-2b Riemannian KD and spatial-loss ablation at 8-32 Hz.**

{table5}

## 4 Discussion

The central engineering result is that a 2-s MI-EEG decoder can be improved without carrying a 4-s teacher into deployment. On IV-2a, training-time long-window information and validation-controlled routing increased short-window accuracy relative to a fixed 2-s CE student. This is relevant when an interface is constrained by the duration of EEG evidence available before issuing a decision.

The IV-2a and IV-2b contrast is informative. IV-2a has four classes and 22 channels, providing a richer spatial covariance structure and a more difficult class boundary. In that setting, combining neural and Riemannian information and selecting a participant-specific route was beneficial. IV-2b is a binary three-channel task with a higher short-window baseline. Its small gain from Riemannian spatial supervision is consistent with a reduced opportunity for additional spatial structure to change the decision boundary. This difference should be interpreted as task dependence, not as a failure to reproduce the IV-2a result.

The method has important boundaries. First, the routing result uses a single internal validation split per seed. The independent evaluation session was protected from selection, but candidate selection may still be sensitive to how the training session is partitioned. An out-of-fold or inner K-fold routing study is the most important next validation. Second, short-window methods did not exceed the strongest 4-s references. Third, the study is participant-dependent and does not establish cross-participant transfer or asynchronous online performance. These constraints motivate cautious use of the term earlier decision rather than universal early decoding.

## 5 Conclusion

This study frames short-window MI-EEG decoding as an early-decision problem with training-time privileged information. A 2-s student benefited from validation-controlled strategy selection on the four-class IV-2a dataset, where routing significantly improved accuracy over fixed CE. On IV-2b, Riemannian spatial supervision produced only a small positive effect, demonstrating that the usefulness of long-window guidance depends on task complexity and channel availability. The inference pipeline retains only the short-window student and optional detector. The evidence supports a practical route toward earlier MI-EEG decisions while preserving the distinction between latency-matched short-window decoding and complete-window reference performance.

## Declarations

**Data availability.** BCI Competition IV-2a and IV-2b are publicly accessible through MOABB as BNCI2014_001 and BNCI2014_004.

**Code availability.** The training and analysis code will be released in a public repository upon acceptance. The release should include exact environment versions, result CSV files, figure source data, and commands used for every reported matrix.

**Competing interests.** The authors declare no competing interests.

## References

{refs}
""").strip() + "\n"


def extract_references() -> str:
    source = ROOT / "manuscripts" / "global_routing_paper" / "validation_controlled_global_routing_mi_eeg_manuscript_zh_latest.md"
    text = source.read_text(encoding="utf-8")
    marker = "## 参考文献"
    if marker not in text:
        return "[Reference list to be imported from the verified working manuscript.]"
    refs = text.split(marker, 1)[1].strip()
    return refs


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_cell_margins(cell, top: int = 80, start: int = 80, bottom: int = 80, end: int = 80) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for m, v in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{m}"))
        if node is None:
            node = OxmlElement(f"w:{m}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(v))
        node.set(qn("w:type"), "dxa")


def add_table(doc: Document, headers: list[str], rows: list[list[str]], widths: list[float]) -> None:
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    for i, text in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.width = Inches(widths[i])
        set_cell_shading(cell, "1F4E79")
        set_cell_margins(cell)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(text)
        run.font.name = "Arial"
        run.font.size = Pt(7.5)
        run.font.color.rgb = RGBColor(255, 255, 255)
        run.bold = True
    for row in rows:
        cells = table.add_row().cells
        for i, text in enumerate(row):
            cell = cells[i]
            cell.width = Inches(widths[i])
            set_cell_margins(cell)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if i > 1 else WD_ALIGN_PARAGRAPH.LEFT
            run = p.add_run(str(text))
            run.font.name = "Arial"
            run.font.size = Pt(7.2)
    doc.add_paragraph()


def add_caption(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    r = p.add_run(text)
    r.bold = True
    r.font.name = "Arial"
    r.font.size = Pt(8.5)


def add_body(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(5)
    p.paragraph_format.line_spacing = 1.15
    r = p.add_run(text)
    r.font.name = "Arial"
    r.font.size = Pt(9.5)


def add_heading(doc: Document, text: str, level: int) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10 if level == 1 else 7)
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run(text)
    r.font.name = "Arial"
    r.font.size = Pt(13 if level == 1 else 11)
    r.bold = True
    r.font.color.rgb = RGBColor(31, 78, 121)


def build_docx(tables: dict[str, pd.DataFrame]) -> None:
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(0.72)
    section.bottom_margin = Inches(0.72)
    section.left_margin = Inches(0.72)
    section.right_margin = Inches(0.72)
    styles = doc.styles
    styles["Normal"].font.name = "Arial"
    styles["Normal"]._element.rPr.rFonts.set(qn("w:eastAsia"), "Arial")
    styles["Normal"].font.size = Pt(9.5)
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = title.add_run("Short-window motor-imagery EEG decoding with training-time privileged information and validation-controlled routing")
    r.bold = True
    r.font.name = "Arial"
    r.font.size = Pt(16)
    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.add_run("MBEC-oriented manuscript draft | Authors and affiliations to be inserted").italic = True
    add_heading(doc, "Abstract", 1)
    add_body(doc, "Motor-imagery EEG (MI-EEG) decoders often use several seconds of post-cue data, which improves evidence accumulation but delays a decision. We investigated whether a student decoder restricted to the first 2 s can benefit from information available only during training. Across nine participants and five random seeds, validation-controlled routing on BCI Competition IV-2a achieved 59.31% accuracy versus 56.72% for fixed 2-s cross-entropy training (paired t-test p=0.0055; Wilcoxon p=0.0078). On BCI Competition IV-2b, adaptive Riemannian knowledge distillation with spatial supervision achieved 77.12%, compared with 76.34% for fixed 2-s cross-entropy. Teachers were removed at inference. These findings support training-time privileged information for earlier MI-EEG decisions, while showing that gains depend on task complexity and spatial information.")
    add_body(doc, "Keywords: motor imagery; electroencephalography; brain-computer interface; short-window decoding; knowledge distillation; Riemannian geometry; EEGNet")
    add_heading(doc, "1 Introduction", 1)
    add_body(doc, "Longer MI-EEG windows accumulate more evidence but delay a BCI decision. We address the engineering question of how much decoding performance can be retained when the deployed classifier is restricted to the first 2 s after the cue. The proposed framework uses 4-s information only during training and removes teachers from inference.")
    add_body(doc, "The central hypothesis is bounded: training-time long-window information and validation-controlled strategy selection can improve a latency-matched 2-s student for difficult participant-dependent MI decoding, but need not replace complete-window decoding.")
    add_heading(doc, "2 Materials and methods", 1)
    add_heading(doc, "2.1 Datasets and protocol", 2)
    add_body(doc, "Signals were retrieved through MOABB. The training session was split into stratified training and validation subsets (80/20); the evaluation session was held out as the only test set. Filtering used 8-32 Hz, and all 0-4 s signals were sampled at 250 Hz. Five seeds controlled the internal split and optimization.")
    add_caption(doc, "Table 1 | Datasets and cross-session evaluation protocol.")
    add_table(doc, ["Dataset", "Classes", "Channels", "Rate", "Protocol"], [["BCI IV-2a", "4", "22", "250 Hz", "Train session -> train/validation; eval session -> test"], ["BCI IV-2b", "2", "3", "250 Hz", "Train session -> train/validation; eval session -> test"]], [1.15, 0.55, 0.65, 0.65, 3.75])
    add_heading(doc, "2.2 Training-time privileged information and routing", 2)
    add_body(doc, "The deployed student was Compact EEGNet with a 2-s input. In IV-2a, frozen 4-s Compact EEGNet and Riemannian teachers supplied soft labels. In IV-2b, a 4-s Riemannian teacher supplied soft labels and a covariance-derived spatial target regularized the pre-depthwise student representation. Candidate routes were selected only from validation performance; the evaluation session was not used to fit teachers, select checkpoints, or select a route.")
    doc.add_picture(str(FIG_DIR / "figure_1_pipeline.png"), width=Inches(6.7))
    add_caption(doc, "Figure 1 | Training-time privileged information and deployment-time short-window inference. The 4-s teachers and covariance targets are available only during training. Deployment uses the 2-s student and, for an adaptive candidate, an optional detector.")
    add_heading(doc, "3 Results", 1)
    add_heading(doc, "3.1 Routing improved short-window decoding on IV-2a", 2)
    add_body(doc, "Validation-controlled routing achieved 59.31% accuracy on IV-2a, compared with 56.72% for fixed 2-s CE (+2.59 percentage points). The gain was observed in eight of nine participants after seed averaging and was significant (paired t-test p=0.0055; Wilcoxon p=0.0078). The routing-versus-fixed-dual difference was not significant, indicating that the main advantage is participant-specific selection among candidates rather than universal dominance of one route.")
    doc.add_picture(str(FIG_DIR / "figure_2_primary_results.png"), width=Inches(6.7))
    add_caption(doc, "Figure 2 | Main short-window evidence. (a) IV-2a routing significantly improved 2-s accuracy over fixed CE. (b) IV-2b showed smaller gains and retained a complete-window advantage. Values are five-seed means across nine participants; the figure does not compare seed standard deviations across different statistical summaries.")
    add_caption(doc, "Table 2 | Main short-window results.")
    rows = [[r.dataset.replace("BCI IV-", ""), r.method, f"{r.window_s:g} s", fmt_metric(r.accuracy, r.seed_sd), f"{r.macro_f1:.4f}", f"{r.kappa:.4f}"] for r in tables["main_results"].itertuples(index=False)]
    add_table(doc, ["Dataset", "Method", "Input", "Accuracy", "Macro-F1", "Kappa"], rows, [0.55, 2.3, 0.55, 1.0, 0.9, 0.7])
    add_heading(doc, "3.2 Cross-dataset boundary and Riemannian spatial ablation", 2)
    add_body(doc, "On IV-2b, adaptive Riemannian KD plus spatial supervision achieved 77.12%, while global routing reached 77.01% and fixed CE reached 76.34%. The gain was smaller than on IV-2a, consistent with the lower channel count and binary task. A matched auxiliary-loss ablation showed the highest mean at moderate joint weighting (0.10+0.10), but this ablation is descriptive because paired tests against CE have not yet been recomputed for the final spatial-supervision matrix.")
    add_heading(doc, "3.3 Reproduced 2-s baseline comparison", 2)
    add_body(doc, "The proposed 2-s strategies were compared with reproduced shallow, deep, classical, Riemannian, and EEGNetV4 baselines. On IV-2a, routing exceeded the reproduced ShallowConvNet and DeepConvNet baselines. On IV-2b, adaptive Riemann KD plus spatial supervision was close to EEGNetV4 and exceeded the shallow/deep 2-s baselines. These neural baseline comparisons are descriptive pending paired statistical tests.")
    doc.add_picture(str(FIG_DIR / "figure_3_short_window_baselines.png"), width=Inches(6.7))
    add_caption(doc, "Figure 3 | Reproduced latency-matched 2-s baselines. Blue bars indicate the proposed short-window strategy for each dataset; gray bars are external baselines under the same cross-session setting.")
    add_caption(doc, "Table 3 | Reproduced 2-s external baselines.")
    rows = [[r.dataset.replace("BCI IV-", ""), r.method, fmt_metric(r.accuracy, r.seed_sd), "--" if pd.isna(r.macro_f1) else f"{r.macro_f1:.4f}", "--" if pd.isna(r.kappa) else f"{r.kappa:.4f}"] for r in tables["short_window_baselines"].itertuples(index=False)]
    add_table(doc, ["Dataset", "2-s comparator", "Accuracy", "Macro-F1", "Kappa"], rows, [0.65, 2.4, 1.15, 1.05, 0.85])
    add_heading(doc, "3.4 Complete-window context", 2)
    add_body(doc, "Four-second models remained strong references. The short-window contribution should therefore be read as preserving useful decoding performance under an earlier-decision constraint, rather than as surpassing all complete-window methods.")
    add_caption(doc, "Table 4 | Four-second reference context.")
    rows = [[r.dataset.replace("BCI IV-", ""), r.method, f"{r.accuracy:.4f}", f"{r.macro_f1:.4f}", f"{r.kappa:.4f}"] for r in tables["long_window_context"].itertuples(index=False)]
    add_table(doc, ["Dataset", "4-s reference", "Accuracy", "Macro-F1", "Kappa"], rows, [0.65, 2.4, 1.15, 1.05, 0.85])
    add_heading(doc, "3.5 Riemannian soft-label and spatial supervision were complementary", 2)
    add_body(doc, "The IV-2b auxiliary-loss ablation is descriptive: moderate joint weighting (0.10+0.10) reached the highest participant-level mean. These values summarize participants after five-seed averaging, so their standard deviations quantify participant heterogeneity and are not directly comparable with the seed-level standard deviations in Table 2. A paired comparison against CE remains necessary before drawing a confirmatory ablation claim.")
    doc.add_picture(str(FIG_DIR / "figure_4_spatial_ablation.png"), width=Inches(6.7))
    add_caption(doc, "Figure 4 | IV-2b Riemannian KD and spatial-supervision ablation. Values are participant-level means after five-seed averaging; error values are reported in Table 5.")
    add_caption(doc, "Table 5 | IV-2b Riemannian KD and spatial-loss ablation at 8-32 Hz.")
    rows = [[f"{r.method} ({r.lambda_r_plus_lambda_s})", f"{r.total_aux_weight:.2f}", f"{r.accuracy:.4f} +/- {r.subject_sd_accuracy:.4f}", f"{r.macro_f1:.4f} +/- {r.subject_sd_macro_f1:.4f}", f"{r.kappa:.4f} +/- {r.subject_sd_kappa:.4f}"] for r in tables["spatial_ablation"].itertuples(index=False)]
    add_table(doc, ["Configuration", "Total weight", "Accuracy", "Macro-F1", "Kappa"], rows, [2.15, 0.75, 1.25, 1.25, 1.25])
    add_heading(doc, "4 Discussion", 1)
    add_body(doc, "The main result is a significant improvement in IV-2a short-window decoding without carrying the 4-s teachers into inference. The IV-2a/IV-2b contrast suggests that the utility of privileged long-window information is task-dependent: it is more useful when class structure and spatial covariance are richer. The study does not claim that 2 s replaces 4 s. It instead establishes a controlled route toward earlier decisions when the available evidence is restricted.")
    add_body(doc, "The principal limitation is route-selection stability. Although the evaluation session was isolated, routing used one internal validation partition per seed. A future out-of-fold or inner K-fold routing analysis should test whether the selected route remains stable across partitions. The work is also participant-dependent and does not establish cross-participant transfer or asynchronous online performance.")
    add_heading(doc, "5 Conclusion", 1)
    add_body(doc, "Training-time privileged information and validation-controlled strategy selection improved 2-s MI-EEG decoding on the four-class IV-2a dataset, while Riemannian spatial supervision produced a smaller positive effect on IV-2b. The results support short-window decoding as an earlier-decision engineering strategy with explicit dataset-dependent limits.")
    add_heading(doc, "Declarations", 1)
    add_body(doc, "Data availability: BCI Competition IV-2a and IV-2b are available through MOABB as BNCI2014_001 and BNCI2014_004. Code availability: the release should include environment versions, result CSV files, source-data tables, and all commands used for the reported matrices. Competing interests: none declared.")
    add_heading(doc, "References", 1)
    for line in extract_references().splitlines():
        if line.strip():
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(2)
            r = p.add_run(line)
            r.font.name = "Arial"
            r.font.size = Pt(7.5)
    doc.save(DOCX_PATH)


def main() -> None:
    set_plot_style()
    tables = write_source_data()
    fig1_pipeline()
    fig2_primary(tables["main_results"])
    fig3_ablation(tables["spatial_ablation"])
    fig4_baselines(tables["short_window_baselines"], tables["long_window_context"])
    OUT.mkdir(parents=True, exist_ok=True)
    MD_PATH.write_text(markdown(tables), encoding="utf-8")
    build_docx(tables)
    print(MD_PATH)
    print(DOCX_PATH)
    print(FIG_DIR)
    print(DATA_DIR)


if __name__ == "__main__":
    main()
