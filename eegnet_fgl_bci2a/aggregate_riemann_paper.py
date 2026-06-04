from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate paper-grade Riemannian baseline results.")
    parser.add_argument("--results_path", type=str, default="results/riemann_paper/riemann_paper_results.csv")
    parser.add_argument("--output_dir", type=str, default="results/riemann_paper")
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 10)))
    parser.add_argument("--windows", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    return parser.parse_args()


def round_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].round(4)
    return out


def main() -> None:
    args = parse_args()
    results_path = Path(args.results_path)
    if not results_path.exists():
        raise FileNotFoundError(results_path)

    df = pd.read_csv(results_path)
    df = df[
        df["subject"].astype(int).isin(args.subjects)
        & df["seed"].astype(int).isin(args.seeds)
        & df["window"].astype(float).isin(args.windows)
    ].copy()
    if df.empty:
        raise RuntimeError("No Riemann paper rows matched the requested filters.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filtered_path = output_dir / "riemann_paper_filtered_rows.csv"
    summary_path = output_dir / "riemann_paper_summary.csv"
    subject_summary_path = output_dir / "riemann_paper_subject_summary.csv"
    best_per_subject_path = output_dir / "riemann_paper_best_per_subject.csv"
    oracle_summary_path = output_dir / "riemann_paper_oracle_summary.csv"

    group_cols = ["window", "method", "classifier", "cov_estimator", "alignment", "metric"]
    summary = (
        df.groupby(group_cols, as_index=False)
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
            mean_val_acc=("val_accuracy", "mean"),
        )
        .sort_values(["window", "mean_acc", "mean_f1"], ascending=[True, False, False])
    )

    subject_summary = (
        df.groupby(group_cols + ["subject"], as_index=False)
        .agg(
            n=("accuracy", "size"),
            seeds=("seed", "nunique"),
            mean_acc=("accuracy", "mean"),
            mean_f1=("macro_f1", "mean"),
            mean_kappa=("kappa", "mean"),
            mean_val_acc=("val_accuracy", "mean"),
        )
        .sort_values(["window", "method", "cov_estimator", "alignment", "subject"])
    )

    best_per_subject = (
        df.sort_values(["val_accuracy", "val_macro_f1", "val_kappa"], ascending=False)
        .groupby(["window", "seed", "subject"], as_index=False)
        .head(1)
        .sort_values(["window", "seed", "subject"])
        .reset_index(drop=True)
    )
    oracle_summary = (
        best_per_subject.groupby(["window"], as_index=False)
        .agg(
            n=("accuracy", "size"),
            subjects=("subject", "nunique"),
            seeds=("seed", "nunique"),
            mean_acc=("accuracy", "mean"),
            std_acc=("accuracy", "std"),
            mean_f1=("macro_f1", "mean"),
            mean_kappa=("kappa", "mean"),
        )
        .sort_values("window")
    )

    df.to_csv(filtered_path, index=False)
    round_numeric(summary).to_csv(summary_path, index=False)
    round_numeric(subject_summary).to_csv(subject_summary_path, index=False)
    best_per_subject.to_csv(best_per_subject_path, index=False)
    round_numeric(oracle_summary).to_csv(oracle_summary_path, index=False)

    print(f"Saved filtered rows: {filtered_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved subject summary: {subject_summary_path}")
    print(f"Saved val-selected best per subject: {best_per_subject_path}")
    print(f"Saved val-selected oracle summary: {oracle_summary_path}")
    print()
    print("Top fixed pipelines:")
    print(round_numeric(summary).head(20).to_string(index=False))
    print()
    print("Validation-selected best per subject/window:")
    print(round_numeric(oracle_summary).to_string(index=False))


if __name__ == "__main__":
    main()
