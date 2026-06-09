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


def read_many(paths: list[Path]) -> pd.DataFrame:
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


def number(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def normalize_ids(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["subject"] = number(frame, "subject").astype(int)
    frame["seed"] = number(frame, "seed", 42).astype(int)
    return frame[frame["subject"].isin(SUBJECTS) & frame["seed"].isin(SEEDS)]


def compact_rows() -> pd.DataFrame:
    frames = []
    for path in sorted((RESULTS / "paper_moabb_multiseed" / "ce_compact").glob("seed_*/student_ce_results.csv")):
        frame = pd.read_csv(path)
        frame["label"] = frame["window"].astype(float).map(
            {2.0: "Compact EEGNet 2s CE", 4.0: "Compact EEGNet 4s CE"}
        )
        frames.append(frame)
    for path in sorted((RESULTS / "paper_moabb_multiseed" / "dual_ablation").glob("seed_*/dual_kd_results.csv")):
        frame = pd.read_csv(path)
        frame["label"] = frame["method"].astype(str).map(
            {
                "EEGNet teacher only": "Compact + EEGNet teacher KD",
                "Riemann teacher only": "Compact + Riemann teacher KD",
                "Riemann-guided dual KD": "Compact + dual teacher KD",
            }
        )
        frames.append(frame)
    for path in sorted((RESULTS / "paper_moabb_multiseed" / "riemann_guided").glob("seed_*/riemann_guided_results.csv")):
        frame = pd.read_csv(path)
        kl = number(frame, "kl_weight")
        geo = number(frame, "geometry_weight")
        spatial = number(frame, "spatial_structure_weight")
        label = pd.Series("", index=frame.index, dtype=object)
        label[np.isclose(kl, 0) & np.isclose(geo, 0) & np.isclose(spatial, 0.1)] = "Compact + Riemann spatial"
        label[np.isclose(kl, 0.2) & np.isclose(geo, 0) & np.isclose(spatial, 0.1)] = "Compact + Riemann KD + spatial"
        frame["label"] = label
        frames.append(frame)
    result = normalize_ids(pd.concat(frames, ignore_index=True, sort=False))
    result = result[result["label"].notna() & (result["label"] != "")]
    return result.sort_values("_priority" if "_priority" in result else "seed").drop_duplicates(
        ["label", "seed", "subject"], keep="last"
    )


def strong_rows() -> pd.DataFrame:
    paths = sorted((RESULTS / "bci2a_eegnetv4_strong_multiseed").glob("seed_*/*/riemann_guided_results.csv"))
    paths.extend(
        [
            RESULTS / "official_eegnetv4_2s_matched_ce_baseline_seed42" / "riemann_guided_results.csv",
            RESULTS / "official_eegnetv4_2s_ablation_s1_spatial_filter" / "riemann_guided_results.csv",
            RESULTS / "official_eegnetv4_2s_riemannkd_spatial_filter_subjects1_9" / "riemann_guided_results.csv",
        ]
    )
    frame = normalize_ids(read_many(paths))
    frame = frame[frame["model_variant"].astype(str) == "eegnetv4"].copy()
    window = number(frame, "window")
    eeg = number(frame, "eegnet_kd_weight")
    rie = number(frame, "kl_weight")
    geo = number(frame, "geometry_weight")
    spatial = number(frame, "spatial_structure_weight")
    label = pd.Series("", index=frame.index, dtype=object)
    common = np.isclose(geo, 0)
    label[np.isclose(window, 4) & np.isclose(eeg, 0) & np.isclose(rie, 0) & np.isclose(spatial, 0) & common] = "EEGNetV4 4s CE"
    label[np.isclose(window, 2) & np.isclose(eeg, 0) & np.isclose(rie, 0) & np.isclose(spatial, 0) & common] = "EEGNetV4 2s CE"
    label[np.isclose(window, 2) & np.isclose(eeg, 0.1) & np.isclose(rie, 0) & np.isclose(spatial, 0) & common] = "EEGNetV4 + EEGNet teacher KD"
    label[np.isclose(window, 2) & np.isclose(eeg, 0) & np.isclose(rie, 0.1) & np.isclose(spatial, 0) & common] = "EEGNetV4 + Riemann KD"
    label[np.isclose(window, 2) & np.isclose(eeg, 0) & np.isclose(rie, 0) & np.isclose(spatial, 0.2) & common] = "EEGNetV4 + Riemann spatial"
    label[np.isclose(window, 2) & np.isclose(eeg, 0) & np.isclose(rie, 0.1) & np.isclose(spatial, 0.2) & common] = "EEGNetV4 + Riemann KD + spatial"
    label[np.isclose(window, 2) & np.isclose(eeg, 0.1) & np.isclose(rie, 0.1) & np.isclose(spatial, 0.2) & common] = "EEGNetV4 + dual KD + spatial"
    frame["label"] = label
    frame = frame[frame["label"] != ""]
    return frame.sort_values("_priority").drop_duplicates(["label", "seed", "subject"], keep="first")


def classical_rows() -> pd.DataFrame:
    frames = []
    for path in sorted((RESULTS / "bci2a_classical_multiseed").glob("seed_*/riemann/riemann_paper_results.csv")):
        frame = pd.read_csv(path)
        frame["label"] = frame["method"].astype(str) + " " + frame["window"].astype(float).map(lambda x: f"{x:g}s")
        frames.append(frame)
    for path in sorted((RESULTS / "bci2a_classical_multiseed").glob("seed_*/csp/csp_baseline_results.csv")):
        frame = pd.read_csv(path)
        frame["label"] = frame["method"].astype(str) + " " + frame["window"].astype(float).map(lambda x: f"{x:g}s")
        frames.append(frame)
    return normalize_ids(pd.concat(frames, ignore_index=True, sort=False)).drop_duplicates(
        ["label", "seed", "subject"], keep="last"
    )


def coverage(frame: pd.DataFrame) -> pd.DataFrame:
    expected = {(seed, subject) for seed in SEEDS for subject in SUBJECTS}
    rows = []
    for label, group in frame.groupby("label", sort=False):
        actual = set(zip(group["seed"], group["subject"]))
        missing = sorted(expected - actual)
        rows.append(
            {
                "method": label,
                "rows": len(group),
                "expected_rows": 45,
                "complete": not missing,
                "missing": ";".join(f"seed{seed}-s{subject}" for seed, subject in missing),
            }
        )
    return pd.DataFrame(rows).sort_values("method")


def summary(frame: pd.DataFrame, family: str) -> pd.DataFrame:
    rows = []
    for label, group in frame.groupby("label", sort=False):
        subject_means = group.groupby("subject")[METRICS].mean()
        row: dict[str, float | int | str] = {
            "family": family,
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


def holm(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    adjusted = np.empty_like(values, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        current = min(1.0, (len(values) - rank) * float(values[index]))
        running = max(running, current)
        adjusted[index] = running
    return adjusted


def paired_tests(frame: pd.DataFrame, reference: str, family: str) -> pd.DataFrame:
    subject = frame.groupby(["label", "subject"], as_index=False)[METRICS].mean()
    reference_frame = subject[subject["label"] == reference].set_index("subject")
    rows = []
    for method in subject["label"].unique():
        if method == reference:
            continue
        current = subject[subject["label"] == method].set_index("subject")
        paired = reference_frame[["accuracy"]].join(
            current[["accuracy"]], lsuffix="_reference", rsuffix="_method", how="inner"
        ).dropna()
        difference = paired["accuracy_method"] - paired["accuracy_reference"]
        if len(paired) < 2:
            continue
        t_result = ttest_rel(paired["accuracy_method"], paired["accuracy_reference"])
        try:
            w_result = wilcoxon(difference, zero_method="wilcox")
            wilcoxon_p = float(w_result.pvalue)
        except ValueError:
            wilcoxon_p = 1.0
        rows.append(
            {
                "family": family,
                "reference": reference,
                "method": method,
                "n_subjects": len(paired),
                "reference_mean_accuracy": paired["accuracy_reference"].mean(),
                "method_mean_accuracy": paired["accuracy_method"].mean(),
                "mean_difference": difference.mean(),
                "median_difference": difference.median(),
                "wins": int((difference > 0).sum()),
                "ties": int(np.isclose(difference, 0).sum()),
                "losses": int((difference < 0).sum()),
                "paired_t_p": float(t_result.pvalue),
                "wilcoxon_p": wilcoxon_p,
                "cohen_dz": difference.mean() / difference.std(ddof=1) if difference.std(ddof=1) > 0 else np.nan,
            }
        )
    result = pd.DataFrame(rows).sort_values("mean_difference", ascending=False)
    if not result.empty:
        result["paired_t_p_holm"] = holm(result["paired_t_p"].to_numpy())
        result["wilcoxon_p_holm"] = holm(result["wilcoxon_p"].to_numpy())
    return result


def teacher_quality(frame: pd.DataFrame, baseline: str, method: str, prefix: str) -> None:
    base = frame[frame["label"] == baseline][["seed", "subject", "accuracy"]].rename(
        columns={"accuracy": "baseline_accuracy"}
    )
    candidates = ["eegnet_teacher_val_accuracy", "riemann_val_accuracy", "confidence_mean"]
    available = [column for column in candidates if column in frame.columns]
    guided = frame[frame["label"] == method][
        ["seed", "subject", "accuracy", *available]
    ].rename(columns={"accuracy": "guided_accuracy"})
    merged = guided.merge(base, on=["seed", "subject"], how="inner")
    merged["accuracy_improvement"] = merged["guided_accuracy"] - merged["baseline_accuracy"]
    merged.to_csv(OUT / f"{prefix}_teacher_quality_vs_improvement.csv", index=False)

    rows = []
    plot_columns = [column for column in available if merged[column].notna().sum() >= 3]
    fig, axes = plt.subplots(1, max(len(plot_columns), 1), figsize=(4.5 * max(len(plot_columns), 1), 4.0))
    axes = np.atleast_1d(axes)
    for axis, column in zip(axes, plot_columns):
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
            x_values = np.linspace(subject_mean[column].min(), subject_mean[column].max(), 100)
            axis.plot(x_values, slope * x_values + intercept, color="#333333", linewidth=1)
        axis.axhline(0, color="#777777", linewidth=0.8)
        axis.set_xlabel(column.replace("_", " "))
        axis.set_ylabel("Accuracy improvement (pp)")
    fig.suptitle(f"{prefix.upper()} teacher quality and student improvement")
    fig.tight_layout()
    fig.savefig(OUT / f"{prefix}_teacher_quality_analysis.png", dpi=220)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT / f"{prefix}_teacher_quality_correlations.csv", index=False)


def bar_plot(table: pd.DataFrame, filename: str, title: str) -> None:
    table = table.sort_values("mean_accuracy", ascending=False)
    fig, axis = plt.subplots(figsize=(10.0, 5.2))
    x = np.arange(len(table))
    colors = ["#4B6A88", "#167D67", "#B06C49", "#7C8F52", "#6D5A7E", "#8A7355", "#557A72"]
    axis.bar(x, table["mean_accuracy"] * 100, color=[colors[i % len(colors)] for i in range(len(table))])
    axis.errorbar(
        x,
        table["mean_accuracy"] * 100,
        yerr=table["std_subject_mean_accuracy"] * 100,
        fmt="none",
        ecolor="#222222",
        capsize=3,
    )
    axis.set_xticks(x)
    axis.set_xticklabels(table["method"], rotation=28, ha="right")
    axis.set_ylabel("Accuracy (%)")
    axis.set_title(title)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=220)
    plt.close(fig)


def subject_improvement_plot(frame: pd.DataFrame, baseline: str, method: str, filename: str, title: str) -> None:
    subject = frame.groupby(["label", "subject"], as_index=False)["accuracy"].mean()
    pivot = subject.pivot(index="subject", columns="label", values="accuracy")
    difference = pivot[method] - pivot[baseline]
    fig, axis = plt.subplots(figsize=(8.0, 4.4))
    colors = np.where(difference.to_numpy() >= 0, "#167D67", "#C64B40")
    axis.bar([f"S{subject}" for subject in difference.index], difference * 100, color=colors)
    axis.axhline(0, color="#222222", linewidth=0.9)
    axis.set_ylabel("Accuracy change (percentage points)")
    axis.set_title(title)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=220)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    compact = compact_rows()
    strong = strong_rows()
    classical = classical_rows()

    compact.to_csv(OUT / "bci2a_compact_rows_deduplicated.csv", index=False)
    strong.to_csv(OUT / "bci2a_eegnetv4_strong_rows_deduplicated.csv", index=False)
    classical.to_csv(OUT / "bci2a_classical_rows_deduplicated.csv", index=False)

    coverage_table = pd.concat(
        [
            coverage(compact).assign(family="compact"),
            coverage(strong).assign(family="eegnetv4"),
            coverage(classical).assign(family="classical"),
        ],
        ignore_index=True,
    )
    coverage_table.to_csv(OUT / "bci2a_final_coverage.csv", index=False)

    compact_summary = summary(compact, "compact")
    strong_summary = summary(strong, "eegnetv4")
    classical_summary = summary(classical, "classical")
    final_summary = pd.concat([compact_summary, strong_summary, classical_summary], ignore_index=True)
    final_summary.to_csv(OUT / "bci2a_final_method_summary.csv", index=False)

    compact_tests = paired_tests(compact, "Compact EEGNet 2s CE", "compact")
    strong_tests = paired_tests(strong, "EEGNetV4 2s CE", "eegnetv4")
    pd.concat([compact_tests, strong_tests], ignore_index=True).to_csv(
        OUT / "bci2a_final_subject_mean_paired_tests.csv", index=False
    )

    teacher_quality(compact, "Compact EEGNet 2s CE", "Compact + dual teacher KD", "bci2a_compact")
    teacher_quality(strong, "EEGNetV4 2s CE", "EEGNetV4 + dual KD + spatial", "bci2a_eegnetv4")
    bar_plot(compact_summary, "bci2a_compact_ablation_summary.png", "BCI IV-2a compact EEGNet ablation (five seeds)")
    bar_plot(strong_summary, "bci2a_eegnetv4_strong_summary.png", "BCI IV-2a EEGNetV4 strong-backbone ablation")
    subject_improvement_plot(
        compact,
        "Compact EEGNet 2s CE",
        "Compact + dual teacher KD",
        "bci2a_compact_subject_improvement.png",
        "BCI IV-2a: dual-teacher KD vs compact 2s CE",
    )
    subject_improvement_plot(
        strong,
        "EEGNetV4 2s CE",
        "EEGNetV4 + dual KD + spatial",
        "bci2a_eegnetv4_subject_improvement.png",
        "BCI IV-2a: dual KD + spatial vs EEGNetV4 2s CE",
    )

    bci2b_path = OUT / "bci2b_method_summary.csv"
    if bci2b_path.exists():
        bci2b = pd.read_csv(bci2b_path)
        bci2b.insert(0, "dataset", "BCI IV-2b")
        bci2a = final_summary.copy()
        bci2a.insert(0, "dataset", "BCI IV-2a")
        pd.concat([bci2a, bci2b], ignore_index=True, sort=False).to_csv(
            OUT / "final_cross_dataset_all_methods.csv", index=False
        )

    print(f"Compact rows: {len(compact)}")
    print(f"EEGNetV4 rows: {len(strong)}")
    print(f"Classical rows: {len(classical)}")
    print(f"Coverage complete: {bool(coverage_table['complete'].all())}")
    print("\nCompact summary:")
    print(compact_summary[["method", "mean_accuracy", "mean_macro_f1", "mean_kappa"]].to_string(index=False))
    print("\nEEGNetV4 strong-backbone summary:")
    print(strong_summary[["method", "mean_accuracy", "mean_macro_f1", "mean_kappa"]].to_string(index=False))
    print("\nStrong-backbone paired tests:")
    print(
        strong_tests[
            ["method", "mean_difference", "wins", "ties", "losses", "paired_t_p", "wilcoxon_p"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
