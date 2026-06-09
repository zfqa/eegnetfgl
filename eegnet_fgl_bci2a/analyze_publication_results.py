from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, ttest_rel, wilcoxon


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT = RESULTS / "publication_complete_analysis"
SEEDS = [42, 2024, 3407, 7, 123]
SUBJECTS = list(range(1, 10))
METRICS = ["accuracy", "macro_f1", "kappa"]


def read_csvs(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for priority, path in enumerate(paths):
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame["_source"] = str(path.relative_to(ROOT))
        frame["_priority"] = priority
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def label_deep_config(frame: pd.DataFrame) -> pd.Series:
    model = frame["model_variant"].astype(str)
    window = numeric(frame, "window")
    eeg = numeric(frame, "eegnet_kd_weight")
    rie = numeric(frame, "kl_weight")
    geo = numeric(frame, "geometry_weight")
    spatial = numeric(frame, "spatial_structure_weight")
    fixed = frame.get("teacher_weight_mode", pd.Series("fixed", index=frame.index)).fillna("fixed").astype(str) == "fixed"

    label = pd.Series("", index=frame.index, dtype=object)
    zero_common = np.isclose(geo, 0.0)
    label[fixed & (model == "compact") & np.isclose(window, 2.0) & np.isclose(eeg, 0) & np.isclose(rie, 0) & np.isclose(spatial, 0) & zero_common] = "Compact EEGNet 2s CE"
    label[fixed & (model == "compact") & np.isclose(window, 4.0) & np.isclose(eeg, 0) & np.isclose(rie, 0) & np.isclose(spatial, 0) & zero_common] = "Compact EEGNet 4s CE"
    label[fixed & (model == "eegnetv4") & np.isclose(window, 2.0) & np.isclose(eeg, 0) & np.isclose(rie, 0) & np.isclose(spatial, 0) & zero_common] = "EEGNetV4 2s CE"
    label[fixed & (model == "eegnetv4") & np.isclose(window, 4.0) & np.isclose(eeg, 0) & np.isclose(rie, 0) & np.isclose(spatial, 0) & zero_common] = "EEGNetV4 4s CE"
    label[fixed & (model == "compact") & np.isclose(window, 2.0) & np.isclose(eeg, 0.4) & np.isclose(rie, 0) & np.isclose(spatial, 0) & zero_common] = "EEGNetV4 teacher only"
    label[fixed & (model == "compact") & np.isclose(window, 2.0) & np.isclose(eeg, 0) & np.isclose(rie, 0.4) & np.isclose(spatial, 0) & zero_common] = "Riemann teacher only"
    label[fixed & (model == "compact") & np.isclose(window, 2.0) & np.isclose(eeg, 0.4) & np.isclose(rie, 0.4) & np.isclose(spatial, 0) & zero_common] = "Dual teacher KD"
    label[fixed & (model == "compact") & np.isclose(window, 2.0) & np.isclose(eeg, 0) & np.isclose(rie, 0) & np.isclose(spatial, 0.1) & zero_common] = "Riemann spatial only"
    label[fixed & (model == "compact") & np.isclose(window, 2.0) & np.isclose(eeg, 0) & np.isclose(rie, 0.2) & np.isclose(spatial, 0.1) & zero_common] = "Riemann KD + spatial"
    return label


def bci2b_deep_rows() -> pd.DataFrame:
    paths = [
        RESULTS / "bci2b_eegnetv4_4s_teachers_seed42" / "riemann_guided_results.csv",
        RESULTS / "bci2b_compact_2s_dual_teacher_seed42" / "riemann_guided_results.csv",
        RESULTS / "bci2b_compact_2s_single_teacher_ablation_seed42" / "riemann_guided_results.csv",
        RESULTS / "bci2b_eegnetv4_2s_matched_ce_vs_riemann_spatial_seed42" / "riemann_guided_results.csv",
        RESULTS / "bci2b_compact_2s_teacher_vs_4s_seed42" / "riemann_guided_results.csv",
    ]
    paths.extend(sorted((RESULTS / "bci2b_paper_multiseed").glob("seed_*/*/riemann_guided_results.csv")))
    frame = read_csvs(paths)
    frame = frame[frame.get("dataset", "bci2b").fillna("bci2b").astype(str) == "bci2b"].copy()
    frame["label"] = label_deep_config(frame)
    frame = frame[frame["label"] != ""].copy()
    frame["seed"] = numeric(frame, "seed").astype(int)
    frame["subject"] = numeric(frame, "subject").astype(int)
    frame = frame[frame["seed"].isin(SEEDS) & frame["subject"].isin(SUBJECTS)]
    frame = frame.sort_values("_priority").drop_duplicates(["label", "seed", "subject"], keep="first")
    return frame


def bci2b_classical_rows() -> pd.DataFrame:
    frames = []
    for path in sorted((RESULTS / "bci2b_paper_multiseed").glob("seed_*/riemann/riemann_paper_results.csv")):
        frame = pd.read_csv(path)
        frame["label"] = frame["method"].astype(str) + " " + frame["window"].astype(float).map(lambda value: f"{value:g}s")
        frames.append(frame)
    for path in sorted((RESULTS / "bci2b_paper_multiseed").glob("seed_*/csp/csp_baseline_results.csv")):
        frame = pd.read_csv(path)
        frame["label"] = frame["method"].astype(str) + " " + frame["window"].astype(float).map(lambda value: f"{value:g}s")
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True, sort=False)
    frame["seed"] = numeric(frame, "seed").astype(int)
    frame["subject"] = numeric(frame, "subject").astype(int)
    return frame.drop_duplicates(["label", "seed", "subject"], keep="last")


def summary_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, group in frame.groupby("label", sort=False):
        subject_means = group.groupby("subject")[METRICS].mean()
        row: dict[str, float | int | str] = {
            "method": label,
            "n_rows": len(group),
            "n_subjects": group["subject"].nunique(),
            "n_seeds": group["seed"].nunique(),
        }
        for metric in METRICS:
            row[f"mean_{metric}"] = group[metric].mean()
            row[f"std_all_{metric}"] = group[metric].std(ddof=1)
            row[f"std_subject_mean_{metric}"] = subject_means[metric].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mean_accuracy", ascending=False)


def coverage_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    expected = {(seed, subject) for seed in SEEDS for subject in SUBJECTS}
    for label, group in frame.groupby("label", sort=False):
        actual = set(zip(group["seed"].astype(int), group["subject"].astype(int)))
        missing = sorted(expected - actual)
        rows.append(
            {
                "method": label,
                "rows": len(group),
                "expected_rows": len(expected),
                "complete": not missing,
                "missing": ";".join(f"seed{seed}-s{subject}" for seed, subject in missing),
            }
        )
    return pd.DataFrame(rows).sort_values("method")


def paired_test_rows(frame: pd.DataFrame, reference: str) -> pd.DataFrame:
    methods = [method for method in frame["label"].unique() if method != reference]
    rows = []
    subject_frame = frame.groupby(["label", "subject"], as_index=False)[METRICS].mean()
    for method in methods:
        left = subject_frame[subject_frame["label"] == reference].set_index("subject")
        right = subject_frame[subject_frame["label"] == method].set_index("subject")
        paired = left[["accuracy"]].join(right[["accuracy"]], lsuffix="_ref", rsuffix="_method", how="inner").dropna()
        if len(paired) < 2:
            continue
        differences = paired["accuracy_method"] - paired["accuracy_ref"]
        t_result = ttest_rel(paired["accuracy_method"], paired["accuracy_ref"])
        try:
            w_result = wilcoxon(differences, zero_method="wilcox", alternative="two-sided")
            wilcoxon_p = float(w_result.pvalue)
        except ValueError:
            wilcoxon_p = 1.0
        rows.append(
            {
                "reference": reference,
                "method": method,
                "n_subjects": len(paired),
                "reference_mean_acc": paired["accuracy_ref"].mean(),
                "method_mean_acc": paired["accuracy_method"].mean(),
                "mean_difference": differences.mean(),
                "median_difference": differences.median(),
                "wins": int((differences > 0).sum()),
                "ties": int(np.isclose(differences, 0).sum()),
                "losses": int((differences < 0).sum()),
                "paired_t_p": float(t_result.pvalue),
                "wilcoxon_p": wilcoxon_p,
                "cohen_dz": differences.mean() / differences.std(ddof=1) if differences.std(ddof=1) > 0 else np.nan,
            }
        )
    result = pd.DataFrame(rows).sort_values("mean_difference", ascending=False)
    if not result.empty:
        result["paired_t_p_holm"] = holm_adjust(result["paired_t_p"].to_numpy())
        result["wilcoxon_p_holm"] = holm_adjust(result["wilcoxon_p"].to_numpy())
    return result


def holm_adjust(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    adjusted = np.empty_like(values, dtype=float)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        current = min(1.0, (count - rank) * float(values[index]))
        running = max(running, current)
        adjusted[index] = running
    return adjusted


def bci2a_main_rows() -> pd.DataFrame:
    frames = []
    for path in sorted((RESULTS / "paper_moabb_multiseed" / "ce_compact").glob("seed_*/student_ce_results.csv")):
        frame = pd.read_csv(path)
        frame = frame[np.isclose(pd.to_numeric(frame["window"]), 2.0)].copy()
        frame["label"] = "Compact EEGNet 2s CE"
        frames.append(frame)
    for path in sorted((RESULTS / "paper_moabb_multiseed" / "dual_ablation").glob("seed_*/dual_kd_results.csv")):
        frame = pd.read_csv(path)
        frame = frame[frame["method"].astype(str) == "Riemann-guided dual KD"].copy()
        frame["label"] = "Dual teacher KD"
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True, sort=False)
    result["seed"] = numeric(result, "seed").astype(int)
    result["subject"] = numeric(result, "subject").astype(int)
    return result.drop_duplicates(["label", "seed", "subject"], keep="last")


def save_plots(deep: pd.DataFrame, combined: pd.DataFrame) -> None:
    subject = deep.groupby(["label", "subject"], as_index=False)["accuracy"].mean()
    pivot = subject.pivot(index="subject", columns="label", values="accuracy")
    difference = pivot["Dual teacher KD"] - pivot["Compact EEGNet 2s CE"]

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    colors = np.where(difference.to_numpy() >= 0, "#167D67", "#C64B40")
    ax.bar([f"S{index}" for index in difference.index], difference.to_numpy() * 100, color=colors)
    ax.axhline(0, color="#222222", linewidth=0.9)
    ax.set_ylabel("Accuracy change (percentage points)")
    ax.set_title("BCI IV-2b: Dual-teacher KD vs 2s compact EEGNet")
    fig.tight_layout()
    fig.savefig(OUT / "bci2b_subject_improvement_dual_vs_ce.png", dpi=220)
    plt.close(fig)

    selected = combined[
        combined["method"].isin(
            [
                "Compact EEGNet 2s CE",
                "EEGNetV4 teacher only",
                "Riemann teacher only",
                "Dual teacher KD",
                "Compact EEGNet 4s CE",
                "EEGNetV4 4s CE",
            ]
        )
    ].copy()
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    x = np.arange(len(selected))
    ax.bar(x, selected["mean_accuracy"] * 100, color=["#4B6A88", "#7C8F52", "#B06C49", "#167D67", "#6D5A7E", "#8A7355"][: len(selected)])
    ax.errorbar(
        x,
        selected["mean_accuracy"] * 100,
        yerr=selected["std_subject_mean_accuracy"] * 100,
        fmt="none",
        ecolor="#222222",
        capsize=3,
        linewidth=1,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(selected["method"], rotation=25, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("BCI IV-2b five-seed method comparison")
    fig.tight_layout()
    fig.savefig(OUT / "bci2b_method_accuracy_summary.png", dpi=220)
    plt.close(fig)


def teacher_quality_analysis(deep: pd.DataFrame) -> None:
    ce = deep[deep["label"] == "Compact EEGNet 2s CE"][
        ["seed", "subject", "accuracy"]
    ].rename(columns={"accuracy": "ce_accuracy"})
    dual_columns = [
        "seed",
        "subject",
        "accuracy",
        "eegnet_teacher_val_accuracy",
        "riemann_val_accuracy",
        "confidence_mean",
    ]
    dual = deep[deep["label"] == "Dual teacher KD"][
        [column for column in dual_columns if column in deep.columns]
    ].rename(columns={"accuracy": "dual_accuracy"})
    merged = dual.merge(ce, on=["seed", "subject"], how="inner")
    merged["accuracy_improvement"] = merged["dual_accuracy"] - merged["ce_accuracy"]
    merged.to_csv(OUT / "bci2b_teacher_quality_vs_improvement.csv", index=False)

    predictors = [
        ("eegnet_teacher_val_accuracy", "EEGNet teacher validation accuracy"),
        ("riemann_val_accuracy", "Riemann teacher validation accuracy"),
        ("confidence_mean", "Mean Riemann confidence"),
    ]
    rows = []
    fig, axes = plt.subplots(1, len(predictors), figsize=(13.0, 4.0))
    for axis, (column, display) in zip(axes, predictors):
        if column not in merged:
            axis.set_visible(False)
            continue
        for unit, data in [
            ("subject_seed", merged),
            ("subject_mean", merged.groupby("subject", as_index=False)[[column, "accuracy_improvement"]].mean()),
        ]:
            clean = data[[column, "accuracy_improvement"]].dropna()
            if len(clean) < 3 or clean[column].nunique() < 2:
                continue
            pearson = pearsonr(clean[column], clean["accuracy_improvement"])
            spearman = spearmanr(clean[column], clean["accuracy_improvement"])
            rows.append(
                {
                    "predictor": column,
                    "unit": unit,
                    "n": len(clean),
                    "pearson_r": float(pearson.statistic),
                    "pearson_p": float(pearson.pvalue),
                    "spearman_rho": float(spearman.statistic),
                    "spearman_p": float(spearman.pvalue),
                }
            )
        subject_mean = merged.groupby("subject", as_index=False)[[column, "accuracy_improvement"]].mean()
        axis.scatter(subject_mean[column], subject_mean["accuracy_improvement"] * 100, color="#167D67", s=42)
        if subject_mean[column].nunique() > 1:
            slope, intercept = np.polyfit(subject_mean[column], subject_mean["accuracy_improvement"] * 100, 1)
            x_line = np.linspace(subject_mean[column].min(), subject_mean[column].max(), 100)
            axis.plot(x_line, slope * x_line + intercept, color="#333333", linewidth=1)
        axis.axhline(0, color="#777777", linewidth=0.8)
        axis.set_xlabel(display)
        axis.set_ylabel("Dual KD improvement (pp)")
    fig.suptitle("BCI IV-2b teacher quality and student improvement")
    fig.tight_layout()
    fig.savefig(OUT / "bci2b_teacher_quality_analysis.png", dpi=220)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT / "bci2b_teacher_quality_correlations.csv", index=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    deep = bci2b_deep_rows()
    classical = bci2b_classical_rows()
    all_bci2b = pd.concat([deep, classical], ignore_index=True, sort=False)

    deep.to_csv(OUT / "bci2b_deep_rows_deduplicated.csv", index=False)
    classical.to_csv(OUT / "bci2b_classical_rows_deduplicated.csv", index=False)
    coverage = coverage_table(all_bci2b)
    coverage.to_csv(OUT / "bci2b_coverage.csv", index=False)
    summary = summary_table(all_bci2b)
    summary.to_csv(OUT / "bci2b_method_summary.csv", index=False)
    paired = paired_test_rows(deep, "Compact EEGNet 2s CE")
    paired.to_csv(OUT / "bci2b_subject_mean_paired_tests_vs_compact2s.csv", index=False)
    subject_means = deep.groupby(["subject", "label"], as_index=False)[METRICS].mean()
    subject_means.to_csv(OUT / "bci2b_subject_mean_results.csv", index=False)

    bci2a = bci2a_main_rows()
    cross_rows = []
    for dataset, frame in [("BCI IV-2a", bci2a), ("BCI IV-2b", deep)]:
        subset = frame[frame["label"].isin(["Compact EEGNet 2s CE", "Dual teacher KD"])]
        table = summary_table(subset)
        table.insert(0, "dataset", dataset)
        cross_rows.append(table)
    cross = pd.concat(cross_rows, ignore_index=True)
    cross.to_csv(OUT / "cross_dataset_main_summary.csv", index=False)
    save_plots(deep, summary)
    teacher_quality_analysis(deep)

    incomplete = coverage[~coverage["complete"]]
    print(f"BCI2b deep rows: {len(deep)}")
    print(f"BCI2b classical rows: {len(classical)}")
    print(f"Coverage complete: {incomplete.empty}")
    if not incomplete.empty:
        print(incomplete.to_string(index=False))
    print("\nBCI2b summary:")
    print(
        summary[
            ["method", "n_rows", "mean_accuracy", "std_subject_mean_accuracy", "mean_macro_f1", "mean_kappa"]
        ].to_string(index=False)
    )
    print("\nPrimary subject-mean paired tests versus Compact EEGNet 2s CE:")
    print(
        paired[
            ["method", "mean_difference", "wins", "ties", "losses", "paired_t_p", "wilcoxon_p", "cohen_dz"]
        ].to_string(index=False)
    )
    print("\nCross-dataset main summary:")
    print(cross[["dataset", "method", "mean_accuracy", "mean_macro_f1", "mean_kappa"]].to_string(index=False))


if __name__ == "__main__":
    main()
