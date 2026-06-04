from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate EEGNet official-protocol ablation results.")
    parser.add_argument("--results_csv", type=str, default="results/eegnet_official/eegnet_official_results.csv")
    parser.add_argument("--output_dir", type=str, default="results/eegnet_official")
    parser.add_argument("--windows", type=float, nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--subjects", type=int, nargs="+", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.results_csv)
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    defaults = {
        "refit_epochs": 0,
        "dropout_rate": 0.5,
        "optimizer": "adam",
        "final_train": "train_val",
        "model_variant": "eegnetv4",
        "fmin": 4.0,
        "fmax": 38.0,
        "tmin": 0.0,
        "tmax": 4.0,
        "kernel_length": 64,
    }
    for column, value in defaults.items():
        if column not in df.columns:
            df[column] = value
    df["subject"] = df["subject"].astype(int)
    df["window"] = df["window"].astype(float)
    df["seed"] = df["seed"].astype(int)
    if args.windows is not None:
        df = df[df["window"].isin(args.windows)].copy()
    if args.seeds is not None:
        df = df[df["seed"].isin(args.seeds)].copy()
    if args.subjects is not None:
        df = df[df["subject"].isin(args.subjects)].copy()
    if df.empty:
        raise RuntimeError("No EEGNet official rows matched the requested filters.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    subject_summary = (
        df.groupby(
            [
                "method",
                "window",
                "subject",
                "model_variant",
                "final_train",
                "fmin",
                "fmax",
                "tmin",
                "tmax",
                "kernel_length",
                "dropout_rate",
                "optimizer",
                "refit_epochs",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            seeds=("seed", "nunique"),
            n=("accuracy", "size"),
            mean_acc=("accuracy", "mean"),
            std_acc=("accuracy", "std"),
            mean_f1=("macro_f1", "mean"),
            mean_kappa=("kappa", "mean"),
            mean_best_epoch=("best_epoch", "mean"),
            mean_val_acc=("val_accuracy", "mean"),
        )
        .sort_values(["method", "window", "subject"])
    )
    summary = (
        df.groupby(
            [
                "method",
                "window",
                "model_variant",
                "final_train",
                "fmin",
                "fmax",
                "tmin",
                "tmax",
                "kernel_length",
                "dropout_rate",
                "optimizer",
                "refit_epochs",
            ],
            dropna=False,
            as_index=False,
        )
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
        )
        .sort_values(["mean_acc", "mean_f1", "mean_kappa"], ascending=False)
    )

    round_cols = [
        "mean_acc",
        "std_acc",
        "mean_f1",
        "std_f1",
        "mean_kappa",
        "std_kappa",
        "mean_best_epoch",
        "mean_val_acc",
    ]
    for frame in (subject_summary, summary):
        for column in round_cols:
            if column in frame.columns:
                frame[column] = frame[column].round(4)

    subject_path = output_dir / "eegnet_official_subject_summary.csv"
    summary_path = output_dir / "eegnet_official_protocol_summary.csv"
    df.to_csv(output_dir / "eegnet_official_filtered_rows.csv", index=False)
    subject_summary.to_csv(subject_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"Saved subject summary: {subject_path}")
    print(f"Saved protocol summary: {summary_path}")
    print()
    print(
        summary[
            [
                "method",
                "window",
                "n",
                "subjects",
                "seeds",
                "mean_acc",
                "std_acc",
                "mean_f1",
                "mean_kappa",
                "mean_best_epoch",
                "mean_val_acc",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
