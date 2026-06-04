from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


EXPECTED_COLUMNS = ["method", "subject", "window", "accuracy", "macro_f1", "kappa"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate EEGNet-FGL result CSV files.")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--output_csv", type=str, default="results/summary_results.csv")
    parser.add_argument("--figure_path", type=str, default="results/figures/window_vs_accuracy.png")
    return parser.parse_args()


def load_results(results_dir: Path) -> pd.DataFrame:
    candidates = [
        results_dir / "teacher" / "teacher_results.csv",
        results_dir / "student_ce" / "student_ce_results.csv",
        results_dir / "student_fgl" / "student_fgl_results.csv",
    ]
    frames = []
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            missing = [column for column in EXPECTED_COLUMNS if column not in df.columns]
            if missing:
                raise ValueError(f"{path} is missing columns: {missing}")
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No result CSV files found under {results_dir}")
    return pd.concat(frames, ignore_index=True)


def plot_window_accuracy(df: pd.DataFrame, figure_path: Path) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = df.copy()
    plt.figure(figsize=(7, 4.5))
    for method, group in plot_df.groupby("method"):
        summary = group.groupby("window")["accuracy"].agg(["mean", "std"]).reset_index()
        plt.errorbar(
            summary["window"],
            summary["mean"],
            yerr=summary["std"].fillna(0.0),
            marker="o",
            capsize=3,
            label=method,
        )
    plt.xlabel("Input window (s)")
    plt.ylabel("Accuracy")
    plt.title("Input window vs accuracy")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_csv = Path(args.output_csv)
    figure_path = Path(args.figure_path)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df = load_results(results_dir)
    main_table = df[EXPECTED_COLUMNS].sort_values(["method", "subject", "window"]).reset_index(drop=True)
    summary = main_table.groupby(["method", "window"], as_index=False, dropna=False).agg(
        n_subjects=("subject", "nunique"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        kappa_mean=("kappa", "mean"),
        kappa_std=("kappa", "std"),
    )
    for column in ["accuracy_std", "macro_f1_std", "kappa_std"]:
        summary[column] = summary[column].fillna(0.0)
    summary = summary.sort_values(["method", "window"])

    main_path = output_csv.parent / "all_subject_results.csv"
    main_table.to_csv(main_path, index=False)
    summary.to_csv(output_csv, index=False)
    plot_window_accuracy(main_table, figure_path)

    print(f"Saved per-subject table: {main_path}")
    print(f"Saved summary table:     {output_csv}")
    print(f"Saved figure:            {figure_path}")
    print("\nMean +/- std:")
    for _, row in summary.iterrows():
        print(
            f"{row['method']} window={row['window']}: "
            f"acc={row['accuracy_mean']:.4f}+/-{row['accuracy_std']:.4f}, "
            f"f1={row['macro_f1_mean']:.4f}+/-{row['macro_f1_std']:.4f}, "
            f"kappa={row['kappa_mean']:.4f}+/-{row['kappa_std']:.4f}"
        )


if __name__ == "__main__":
    main()
