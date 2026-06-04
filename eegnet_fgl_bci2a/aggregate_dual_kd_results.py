from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate diffusion and dual-teacher KD results.")
    parser.add_argument("--results_path", type=str, default="results/dual_kd/dual_kd_results.csv")
    parser.add_argument("--output_dir", type=str, default="results/dual_kd")
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 10)))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--windows", type=float, nargs="+", default=[2.0])
    return parser.parse_args()


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
        raise RuntimeError("No dual-KD rows matched the requested filters.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filtered_path = output_dir / "dual_kd_filtered_rows.csv"
    summary_path = output_dir / "dual_kd_summary.csv"
    subject_path = output_dir / "dual_kd_subject_summary.csv"

    group_cols = ["method", "window", "teacher_mode", "diffusion_multiplier", "student_model_variant"]
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
            mean_best_epoch=("best_epoch", "mean"),
            mean_val_acc=("val_accuracy", "mean"),
            mean_synthetic=("synthetic_samples", "mean"),
        )
        .sort_values(["window", "mean_acc", "mean_f1"], ascending=[True, False, False])
    )
    subject_summary = (
        df.groupby(["method", "window", "subject"], as_index=False)
        .agg(
            n=("accuracy", "size"),
            seeds=("seed", "nunique"),
            mean_acc=("accuracy", "mean"),
            std_acc=("accuracy", "std"),
            mean_f1=("macro_f1", "mean"),
            mean_kappa=("kappa", "mean"),
        )
        .sort_values(["window", "method", "subject"])
    )
    for table in (summary, subject_summary):
        numeric_cols = table.select_dtypes(include=[np.number]).columns
        table[numeric_cols] = table[numeric_cols].round(4)

    df.to_csv(filtered_path, index=False)
    summary.to_csv(summary_path, index=False)
    subject_summary.to_csv(subject_path, index=False)
    print(f"Saved filtered rows: {filtered_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved subject summary: {subject_path}")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
