"""Summarize BCI Riemann-KD and spatial-loss ablations across seeds.

The script first averages each configuration across seeds within each subject,
then reports the mean and standard deviation across subjects.  This avoids
mistaking seed-level variation for independent subject-level evidence.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRICS = ["accuracy", "macro_f1", "kappa"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Result directories or riemann_guided_results.csv files to include.",
    )
    parser.add_argument(
        "--output_dir",
        default="results/riemann_spatial_ablation_summary",
        help="Directory for the raw, subject-level, and summary CSV files.",
    )
    parser.add_argument("--dataset", default="bci2b", help="Keep only this dataset.")
    return parser.parse_args()


def discover_csvs(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for raw_path in inputs:
        path = Path(raw_path)
        if path.is_file():
            paths.append(path)
        elif path.is_dir():
            paths.extend(sorted(path.rglob("riemann_guided_results.csv")))
        else:
            raise FileNotFoundError(f"Input does not exist: {path}")
    return list(dict.fromkeys(paths))


def config_label(row: pd.Series) -> str:
    kd = float(row["kl_weight"])
    spatial = float(row["spatial_structure_weight"])
    if kd == 0.0 and spatial == 0.0:
        return "CE (0+0)"
    if spatial == 0.0:
        return f"Riemann KD only ({kd:g}+0)"
    if kd == 0.0:
        return f"Spatial only (0+{spatial:g})"
    return f"Riemann KD + spatial ({kd:g}+{spatial:g})"


def main() -> None:
    args = parse_args()
    csv_paths = discover_csvs(args.inputs)
    if not csv_paths:
        raise FileNotFoundError("No riemann_guided_results.csv files were found.")

    data = pd.concat([pd.read_csv(path) for path in csv_paths], ignore_index=True)
    required = {"dataset", "subject", "seed", "kl_weight", "spatial_structure_weight", *METRICS}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Required CSV columns are missing: {sorted(missing)}")

    data = data[data["dataset"].eq(args.dataset)].copy()
    if data.empty:
        raise ValueError(f"No rows for dataset={args.dataset!r}.")

    data["config"] = data.apply(config_label, axis=1)
    data = data.drop_duplicates(["dataset", "subject", "seed", "config"], keep="last")
    data = data.sort_values(["config", "seed", "subject"])

    subject_means = (
        data.groupby(["dataset", "config", "subject"], as_index=False)
        .agg(**{metric: (metric, "mean") for metric in METRICS}, n_seeds=("seed", "nunique"))
        .sort_values(["config", "subject"])
    )
    summary = (
        subject_means.groupby(["dataset", "config"], as_index=False)
        .agg(
            n_subjects=("subject", "nunique"),
            min_completed_seeds=("n_seeds", "min"),
            mean_acc=("accuracy", "mean"),
            std_acc=("accuracy", "std"),
            mean_f1=("macro_f1", "mean"),
            std_f1=("macro_f1", "std"),
            mean_kappa=("kappa", "mean"),
            std_kappa=("kappa", "std"),
        )
        .sort_values("config")
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_dir / "all_rows.csv", index=False)
    subject_means.to_csv(output_dir / "subject_seed_means.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)

    print("=== Riemann KD + spatial ablation ===")
    for row in summary.itertuples(index=False):
        print(
            f"{row.config:34s} | subjects={row.n_subjects} | "
            f"acc={row.mean_acc:.4f} +/- {row.std_acc:.4f} | "
            f"f1={row.mean_f1:.4f} +/- {row.std_f1:.4f} | "
            f"kappa={row.mean_kappa:.4f} +/- {row.std_kappa:.4f} | "
            f"min_seeds={row.min_completed_seeds}"
        )
    print(f"\nSaved: {output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
