from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METHOD_ORDER = {
    "EEGNet": 0,
    "Riemann-LogReg": 1,
    "Riemann-MDM": 2,
    "EEGNet+Riemann-Ensemble": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate multi-seed experiment matrix results.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 2024, 3407])
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 10)))
    parser.add_argument("--windows", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--cov_shrinkage", type=float, default=0.001)
    parser.add_argument("--riemann_cov_shrinkage", type=float, default=0.001)
    return parser.parse_args()


def seed_dirs(seed: int) -> tuple[Path, Path, Path]:
    if seed == 42:
        return (
            Path("results/student_ce"),
            Path("results/riemann_baseline"),
            Path("results/eegnet_riemann_fusion"),
        )
    base = Path("results") / f"seed_{seed}"
    return base / "student_ce", base / "riemann_baseline", base / "eegnet_riemann_fusion"


def window_tag(window: float) -> str:
    return f"{window:g}"


def expected_checkpoint(seed: int, subject: int, window: float) -> str:
    ce_dir, _, _ = seed_dirs(seed)
    return str(ce_dir / f"subject_{subject}_window_{window_tag(window)}_ce_best.pth")


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def normalize_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["subject"] = out["subject"].astype(int)
    out["window"] = out["window"].astype(float)
    out["seed"] = out["seed"].astype(int)
    return out


def collect_results(args: argparse.Namespace) -> pd.DataFrame:
    frames = []
    for seed in args.seeds:
        ce_dir, riemann_dir, ensemble_dir = seed_dirs(seed)

        ce = normalize_key_columns(load_csv(ce_dir / "student_ce_results.csv"))
        if not ce.empty:
            ce = ce[
                (ce["method"] == "EEGNet-CE")
                & (ce["seed"].isin([seed]))
                & (ce["subject"].isin(args.subjects))
                & (ce["window"].isin(args.windows))
            ].copy()
            ce["method"] = "EEGNet"
            frames.append(ce)

        riemann = normalize_key_columns(load_csv(riemann_dir / "riemann_baseline_results.csv"))
        if not riemann.empty:
            riemann = riemann[
                (riemann["method"].isin(["Riemann-LogReg", "Riemann-MDM"]))
                & (riemann["seed"].isin([seed]))
                & (riemann["subject"].isin(args.subjects))
                & (riemann["window"].isin(args.windows))
                & (np.isclose(riemann["cov_shrinkage"].astype(float), args.riemann_cov_shrinkage))
            ].copy()
            frames.append(riemann)

        ensemble = normalize_key_columns(load_csv(ensemble_dir / "eegnet_riemann_ensemble_results.csv"))
        if not ensemble.empty:
            ensemble = ensemble[
                (ensemble["method"] == "EEGNet-Riemann-Ensemble")
                & (ensemble["seed"].isin([seed]))
                & (ensemble["subject"].isin(args.subjects))
                & (ensemble["window"].isin(args.windows))
                & (np.isclose(ensemble["cov_shrinkage"].astype(float), args.cov_shrinkage))
            ].copy()
            if "checkpoint" in ensemble.columns:
                expected = {
                    (subject, window): expected_checkpoint(seed, subject, window)
                    for subject in args.subjects
                    for window in args.windows
                }
                ensemble = ensemble[
                    ensemble.apply(
                        lambda row: str(row["checkpoint"]) == expected[(int(row["subject"]), float(row["window"]))],
                        axis=1,
                    )
                ].copy()
            ensemble["method"] = "EEGNet+Riemann-Ensemble"
            frames.append(ensemble)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True, sort=False)
    df = df.drop_duplicates(["method", "subject", "window", "seed"], keep="last")
    df = df.sort_values(
        ["window", "method", "seed", "subject"],
        key=lambda col: col.map(METHOD_ORDER) if col.name == "method" else col,
    ).reset_index(drop=True)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["window", "method"], as_index=False)
        .agg(
            n=("accuracy", "size"),
            subjects=("subject", "nunique"),
            seeds=("seed", "nunique"),
            mean_acc=("accuracy", "mean"),
            std_acc=("accuracy", "std"),
            mean_f1=("macro_f1", "mean"),
            std_f1=("macro_f1", "std"),
            mean_kappa=("kappa", "mean"),
            std_kappa=("kappa", "std"),
        )
        .sort_values(["window", "method"], key=lambda col: col.map(METHOD_ORDER) if col.name == "method" else col)
    )
    metric_cols = ["mean_acc", "std_acc", "mean_f1", "std_f1", "mean_kappa", "std_kappa"]
    summary[metric_cols] = summary[metric_cols].round(4)
    summary["acc_mean_std"] = summary.apply(lambda row: f"{row['mean_acc']:.4f} +/- {row['std_acc']:.4f}", axis=1)
    summary["f1_mean_std"] = summary.apply(lambda row: f"{row['mean_f1']:.4f} +/- {row['std_f1']:.4f}", axis=1)
    summary["kappa_mean_std"] = summary.apply(lambda row: f"{row['mean_kappa']:.4f} +/- {row['std_kappa']:.4f}", axis=1)
    return summary


def summarize_subjects(df: pd.DataFrame) -> pd.DataFrame:
    subject_summary = (
        df.groupby(["window", "method", "subject"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            mean_acc=("accuracy", "mean"),
            std_acc=("accuracy", "std"),
            mean_f1=("macro_f1", "mean"),
            mean_kappa=("kappa", "mean"),
        )
        .sort_values(
            ["window", "method", "subject"],
            key=lambda col: col.map(METHOD_ORDER) if col.name == "method" else col,
        )
    )
    return subject_summary.round(4)


def wilcoxon_table(df: pd.DataFrame) -> pd.DataFrame:
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return pd.DataFrame(
            [{"note": "scipy is not installed; Wilcoxon signed-rank tests were not computed."}]
        )

    rows = []
    comparisons = [
        ("EEGNet+Riemann-Ensemble", "EEGNet"),
        ("EEGNet+Riemann-Ensemble", "Riemann-LogReg"),
    ]
    for window in sorted(df["window"].unique()):
        window_df = df[df["window"] == window]
        for left, right in comparisons:
            left_df = window_df[window_df["method"] == left][["subject", "seed", "accuracy"]]
            right_df = window_df[window_df["method"] == right][["subject", "seed", "accuracy"]]
            paired = left_df.merge(right_df, on=["subject", "seed"], suffixes=("_left", "_right"))
            if len(paired) == 0:
                continue
            diff = paired["accuracy_left"] - paired["accuracy_right"]
            try:
                stat, p_value = wilcoxon(diff)
            except ValueError:
                stat, p_value = np.nan, np.nan
            rows.append(
                {
                    "window": window,
                    "comparison": f"{left} vs {right}",
                    "n_pairs": int(len(paired)),
                    "mean_delta_acc": round(float(diff.mean()), 4),
                    "median_delta_acc": round(float(diff.median()), 4),
                    "wilcoxon_stat": stat,
                    "p_value": p_value,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = collect_results(args)
    if df.empty:
        raise RuntimeError("No results were found for the requested seeds/subjects/windows.")

    full_grid = len(args.seeds) * len(args.subjects) * len(args.windows)
    expected_per_method = full_grid
    coverage = (
        df.groupby("method")
        .size()
        .rename("n_rows")
        .reset_index()
        .assign(expected_rows=expected_per_method)
    )

    subject_seed_path = output_dir / "multiseed_matrix_subject_seed.csv"
    subject_mean_path = output_dir / "multiseed_matrix_subject_mean.csv"
    summary_path = output_dir / "multiseed_matrix_summary.csv"
    wilcoxon_path = output_dir / "multiseed_matrix_wilcoxon.csv"
    coverage_path = output_dir / "multiseed_matrix_coverage.csv"

    df.to_csv(subject_seed_path, index=False)
    summarize_subjects(df).to_csv(subject_mean_path, index=False)
    summary = summarize(df)
    summary.to_csv(summary_path, index=False)
    wilcoxon_table(df).to_csv(wilcoxon_path, index=False)
    coverage.to_csv(coverage_path, index=False)

    print(f"Saved subject-seed table: {subject_seed_path}")
    print(f"Saved subject mean table: {subject_mean_path}")
    print(f"Saved summary table: {summary_path}")
    print(f"Saved Wilcoxon table: {wilcoxon_path}")
    print(f"Saved coverage table: {coverage_path}")
    print("\nCoverage:")
    print(coverage.to_string(index=False))
    print("\nSummary:")
    print(summary[["window", "method", "n", "subjects", "seeds", "acc_mean_std", "f1_mean_std", "kappa_mean_std"]].to_string(index=False))


if __name__ == "__main__":
    main()
