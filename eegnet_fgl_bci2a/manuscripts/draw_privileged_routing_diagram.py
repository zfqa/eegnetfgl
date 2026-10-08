from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


OUT_DIR = Path("manuscripts/referenced_chinese/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def add_round_box(ax, xy, wh, text, fc, ec, fs=8.6, weight="normal", lw=1.15, radius=0.025):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        weight=weight,
        color="#152230",
        linespacing=1.22,
        zorder=4,
    )
    return patch


def add_arrow(ax, p1, p2, color="#405062", lw=1.2, dashed=False, rad=0.0, ms=12):
    patch = FancyArrowPatch(
        p1,
        p2,
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=lw,
        color=color,
        linestyle=(0, (4, 3)) if dashed else "solid",
        connectionstyle=f"arc3,rad={rad}",
        zorder=2,
    )
    ax.add_patch(patch)
    return patch


def label(ax, x, y, text, fs=7.8, color="#566575", ha="center", weight="normal"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fs, color=color, weight=weight, linespacing=1.18, zorder=5)


def panel(ax, x, y, w, h, tag, title, fc):
    rect = Rectangle((x, y), w, h, facecolor=fc, edgecolor="none", alpha=0.78, zorder=0)
    ax.add_patch(rect)
    ax.text(x + 0.018, y + h - 0.035, tag, fontsize=10.5, weight="bold", color="#1a2a3a", va="top")
    ax.text(x + 0.052, y + h - 0.035, title, fontsize=10.5, weight="bold", color="#1a2a3a", va="top")


def main() -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(14.2, 7.0), dpi=180)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    navy = "#1d2b3a"
    blue_fc, blue_ec = "#dceafe", "#54779c"
    green_fc, green_ec = "#ddf1eb", "#4f8c7f"
    orange_fc, orange_ec = "#fff0d9", "#b48144"
    gray_fc, gray_ec = "#f7f9fb", "#c6d0db"

    panel(ax, 0.045, 0.20, 0.30, 0.72, "A", "训练会话：特权教师监督", "#fbf5ea")
    panel(ax, 0.365, 0.20, 0.31, 0.72, "B", "验证集控制：候选策略路由", "#eef7f1")
    panel(ax, 0.695, 0.20, 0.26, 0.72, "C", "测试/部署：2 s 推理", "#edf4fc")

    # Panel A: data and teachers.
    add_round_box(ax, (0.075, 0.665), (0.105, 0.075), "前 2 s\n学生输入", blue_fc, blue_ec, fs=8.2, weight="bold")
    add_round_box(ax, (0.205, 0.665), (0.105, 0.075), "完整 4 s\n教师输入", orange_fc, orange_ec, fs=8.2, weight="bold")
    add_round_box(ax, (0.075, 0.780), (0.235, 0.055), "同一 trial，先划分 train/val/test 再截窗", gray_fc, gray_ec, fs=7.9)
    add_arrow(ax, (0.127, 0.780), (0.127, 0.742), color="#788799", ms=9)
    add_arrow(ax, (0.257, 0.780), (0.257, 0.742), color="#b48144", ms=9)

    add_round_box(ax, (0.075, 0.505), (0.235, 0.075), "4 s Compact EEGNet 教师\nsoft label", blue_fc, blue_ec, fs=8.15)
    add_round_box(ax, (0.075, 0.380), (0.235, 0.090), "4 s 黎曼教师\nOAS 协方差 → Tangent Space → LogReg", green_fc, green_ec, fs=7.75)
    add_arrow(ax, (0.257, 0.665), (0.257, 0.582), color="#b48144")
    add_arrow(ax, (0.257, 0.665), (0.257, 0.472), color="#b48144", rad=0.04)
    label(ax, 0.193, 0.335, "输出：类别概率 + 空间协方差目标", fs=7.4, color="#4d6f69")

    # Panel B: student and routing.
    add_round_box(ax, (0.405, 0.610), (0.225, 0.095), "2 s Compact EEGNet 学生\nCE + KD / spatial loss", blue_fc, blue_ec, fs=8.8, weight="bold")
    add_arrow(ax, (0.180, 0.703), (0.405, 0.650), color="#657789", dashed=True, rad=-0.08)
    add_arrow(ax, (0.310, 0.542), (0.405, 0.665), color=blue_ec, rad=0.12)
    add_arrow(ax, (0.310, 0.425), (0.405, 0.635), color=green_ec, rad=-0.08)
    label(ax, 0.365, 0.735, "训练期监督", fs=7.4, color="#526375")

    add_round_box(ax, (0.405, 0.480), (0.225, 0.055), "训练会话内部验证集打分", "#ffffff", "#8fa6bc", fs=8.1, weight="bold")
    add_arrow(ax, (0.518, 0.610), (0.518, 0.536), color="#8fa6bc")

    route_nodes = [
        ("固定 CE", 0.390, 0.365),
        ("固定蒸馏", 0.510, 0.365),
        ("自适应 CE", 0.390, 0.280),
        ("自适应\nRiemann+spatial", 0.510, 0.280),
    ]
    for text, x, y in route_nodes:
        add_round_box(ax, (x, y), (0.095, 0.052), text, "#ffffff", "#a0adbb", fs=7.2, radius=0.02)
        add_arrow(ax, (0.518, 0.480), (x + 0.048, y + 0.054), color="#9aabba", dashed=True, lw=0.95, ms=8)

    add_round_box(ax, (0.430, 0.205), (0.175, 0.055), "选择每个被试最终路线", "#e3f4e8", "#5d9a69", fs=8.1, weight="bold")
    for _, x, y in route_nodes:
        add_arrow(ax, (x + 0.048, y), (0.518, 0.260), color="#5d9a69", lw=0.95, ms=8)

    # Panel C: deployment.
    add_round_box(ax, (0.735, 0.610), (0.180, 0.095), "仅输入 2 s EEG\n选中路线的学生模型", blue_fc, blue_ec, fs=8.9, weight="bold")
    add_round_box(ax, (0.735, 0.430), (0.180, 0.080), "运动想象类别预测\n2a: 4 类 / 2b: 2 类", orange_fc, orange_ec, fs=8.6)
    add_arrow(ax, (0.605, 0.232), (0.735, 0.648), color="#5d9a69", rad=-0.20)
    add_arrow(ax, (0.825, 0.610), (0.825, 0.512), color=blue_ec)
    label(ax, 0.825, 0.355, "不调用 4 s 教师\n不计算黎曼分类器", fs=8.1, color="#7a5b4b")
    add_arrow(ax, (0.630, 0.660), (0.735, 0.660), color="#b45a5a", dashed=True, lw=1.0)
    label(ax, 0.682, 0.690, "教师不进入部署", fs=7.4, color="#a04c4c")

    # Bottom constraint statement.
    add_round_box(
        ax,
        (0.065, 0.085),
        (0.870, 0.070),
        "无泄漏约束：教师训练、温度校准、协方差目标与路由选择均只使用训练会话内部数据；评估会话仅用于最终测试。",
        gray_fc,
        gray_ec,
        fs=8.3,
        weight="bold",
        radius=0.018,
    )

    for ext in ("png", "svg", "pdf", "tiff"):
        fig.savefig(
            OUT_DIR / f"figure_1_privileged_routing_diagram.{ext}",
            bbox_inches="tight",
            pad_inches=0.10,
            dpi=600 if ext in {"png", "tiff"} else 300,
        )
    plt.close(fig)
    print(OUT_DIR.resolve())


if __name__ == "__main__":
    main()
