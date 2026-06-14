from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


SEEDS = [42, 2024, 3407, 7, 123]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze total-weight-matched KD ablations.")
    parser.add_argument("--matched_root", default="results/weight_matched_kd")
    parser.add_argument("--output_dir", default="results/weight_matched_kd/analysis")
    return parser.parse_args()


def safe_wilcoxon(x: np.ndarray, y: np.ndarray) -> float:
    if np.allclose(x, y):
        return 1.0
    return float(wilcoxon(x, y).pvalue)


def collect_csvs(patterns: list[str]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for pattern in patterns for path in sorted(Path(".").glob(pattern))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_bci2a(matched_root: Path) -> pd.DataFrame:
    existing = collect_csvs(
        ["results/paper_moabb_multiseed/dual_ablation/seed_*/dual_kd_results.csv"]
    )
    matched = collect_csvs([str(matched_root / "bci2a" / "seed_*" / "dual_kd_results.csv")])
    frame = pd.concat([existing, matched], ignore_index=True)
    frame["dataset"] = "bci2a"
    frame["eegnet_weight"] = frame["alpha1"].astype(float)
    frame["riemann_weight"] = frame["alpha2"].astype(float)
    return frame


def load_bci2b(matched_root: Path) -> pd.DataFrame:
    existing = collect_csvs(
        [
            "results/bci2b_paper_multiseed/seed_*/*/riemann_guided_results.csv",
            "results/bci2b_eegnetv4_4s_teachers_seed42/riemann_guided_results.csv",
            "results/bci2b_compact_2s_dual_teacher_seed42/riemann_guided_results.csv",
            "results/bci2b_compact_2s_single_teacher_ablation_seed42/riemann_guided_results.csv",
            "results/bci2b_eegnetv4_2s_matched_ce_vs_riemann_spatial_seed42/riemann_guided_results.csv",
            "results/bci2b_compact_2s_teacher_vs_4s_seed42/riemann_guided_results.csv",
        ]
    )
    matched = collect_csvs([str(matched_root / "bci2b" / "seed_*" / "riemann_guided_results.csv")])
    frame = pd.concat([existing, matched], ignore_index=True)
    frame = frame[
        (frame["dataset"] == "bci2b")
        & (frame["model_variant"] == "compact")
        & np.isclose(frame["window"].astype(float), 2.0)
        & np.isclose(frame["geometry_weight"].astype(float), 0.0)
        & np.isclose(frame["spatial_structure_weight"].astype(float), 0.0)
        & np.isclose(frame["supcon_weight"].astype(float), 0.0)
    ].copy()
    frame["eegnet_weight"] = frame["eegnet_kd_weight"].astype(float)
    frame["riemann_weight"] = frame["kl_weight"].astype(float)
    return frame


def label_config(row: pd.Series) -> str:
    eegnet = float(row["eegnet_weight"])
    riemann = float(row["riemann_weight"])
    if np.isclose(eegnet, 0.0) and np.isclose(riemann, 0.0):
        return "CE"
    if np.isclose(riemann, 0.0):
        return f"EEGNet-only ({eegnet:g})"
    if np.isclose(eegnet, 0.0):
        return f"Riemann-only ({riemann:g})"
    return f"Dual ({eegnet:g}+{riemann:g})"


def paired_test(subject_means: pd.DataFrame, dataset: str, first: str, second: str) -> dict[str, object] | None:
    subset = subject_means[subject_means["dataset"] == dataset]
    pivot = subset.pivot(index="subject", columns="config", values="accuracy")
    if first not in pivot or second not in pivot:
        return None
    pair = pivot[[first, second]].dropna()
    if len(pair) < 2:
        return None
    x = pair[first].to_numpy()
    y = pair[second].to_numpy()
    difference = x - y
    return {
        "dataset": dataset,
        "comparison": f"{first} vs {second}",
        "n_subjects": len(pair),
        "first_mean": x.mean(),
        "second_mean": y.mean(),
        "mean_difference": difference.mean(),
        "wins": int((difference > 0).sum()),
        "ties": int(np.isclose(difference, 0).sum()),
        "losses": int((difference < 0).sum()),
        "paired_t_p": float(ttest_rel(x, y).pvalue),
        "wilcoxon_p": safe_wilcoxon(x, y),
    }


def main() -> None:
    args = parse_args()
    matched_root = Path(args.matched_root)
    frames = [load_bci2a(matched_root), load_bci2b(matched_root)]
    data = pd.concat(frames, ignore_index=True)
    data = data[data["seed"].astype(int).isin(SEEDS)].copy()
    data["config"] = data.apply(label_config, axis=1)
    data = data.drop_duplicates(["dataset", "subject", "seed", "config"], keep="last")

    subject_means = (
        data.groupby(["dataset", "subject", "config"], as_index=False)
        .agg(
            accuracy=("accuracy", "mean"),
            macro_f1=("macro_f1", "mean"),
            kappa=("kappa", "mean"),
            completed_seeds=("seed", "nunique"),
        )
    )
    summary = (
        subject_means.groupby(["dataset", "config"], as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            mean_kappa=("kappa", "mean"),
            subjects=("subject", "nunique"),
            minimum_completed_seeds=("completed_seeds", "min"),
        )
    )

    comparisons = [
        ("Dual (0.4+0.4)", "EEGNet-only (0.8)"),
        ("Dual (0.4+0.4)", "Riemann-only (0.8)"),
        ("Dual (0.2+0.2)", "EEGNet-only (0.4)"),
        ("Dual (0.2+0.2)", "Riemann-only (0.4)"),
        ("Dual (0.4+0.4)", "Dual (0.2+0.2)"),
    ]
    tests = [
        result
        for dataset in ("bci2a", "bci2b")
        for first, second in comparisons
        if (result := paired_test(subject_means, dataset, first, second)) is not None
    ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_dir / "weight_matched_all_rows.csv", index=False)
    subject_means.to_csv(output_dir / "weight_matched_subject_means.csv", index=False)
    summary.to_csv(output_dir / "weight_matched_summary.csv", index=False)
    pd.DataFrame(tests).to_csv(output_dir / "weight_matched_paired_tests.csv", index=False)
    print(summary.to_string(index=False))
    if tests:
        print("\nPaired tests")
        print(pd.DataFrame(tests).to_string(index=False))


if __name__ == "__main__":
    main()
