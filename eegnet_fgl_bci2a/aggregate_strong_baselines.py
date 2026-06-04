from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METHOD_ORDER = {
    "EEGNet-CE": 0,
    "EEGNetv4-CE": 1,
    "EEGNetv4-CE-k64-f38": 2,
    "EEGNetv4-k64-f38-fixed500": 3,
    "EEGNetv4-k64-f38-do25": 4,
    "EEGNetv4-k64-f38-t05-4": 5,
    "EEGNet-Compact-CE": 6,
    "Riemann-LogReg": 20,
    "Riemann-MDM": 21,
    "EEGNet-Riemann-Ensemble": 22,
    "CSP-LDA": 30,
    "CSP-LogReg": 31,
    "FBCSP-LDA": 32,
    "FBCSP-LogReg": 33,
    "FBCSP-LDA-ANOVA": 34,
    "FBCSP-LogReg-ANOVA": 35,
    "FBCSP-LDA-MI": 36,
    "FBCSP-LogReg-MI": 37,
}


def method_order(col: pd.Series) -> pd.Series:
    return col.map(METHOD_ORDER).fillna(100)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate EEGNet/Riemann/CSP strong baseline results.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 10)))
    parser.add_argument("--windows", type=float, nargs="+", default=[4.0])
    parser.add_argument("--output_dir", type=str, default="results")
    return parser.parse_args()


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def seed_dirs(seed: int) -> tuple[Path, Path, Path]:
    if seed == 42:
        return (
            Path("results/student_ce"),
            Path("results/riemann_baseline"),
            Path("results/eegnet_riemann_fusion"),
        )
    base = Path("results") / f"seed_{seed}"
    return base / "student_ce", base / "riemann_baseline", base / "eegnet_riemann_fusion"


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["subject"] = out["subject"].astype(int)
    out["window"] = out["window"].astype(float)
    out["seed"] = out["seed"].astype(int)
    return out


def filter_common(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if df.empty:
        return df
    return df[
        df["seed"].isin(args.seeds)
        & df["subject"].isin(args.subjects)
        & df["window"].astype(float).isin(args.windows)
    ].copy()


def filter_seed_common(df: pd.DataFrame, args: argparse.Namespace, seed: int) -> pd.DataFrame:
    if df.empty:
        return df
    return df[
        df["seed"].eq(seed)
        & df["subject"].isin(args.subjects)
        & df["window"].astype(float).isin(args.windows)
    ].copy()


def collect(args: argparse.Namespace) -> pd.DataFrame:
    frames = []

    for seed in args.seeds:
        ce_dir, riemann_dir, ensemble_dir = seed_dirs(seed)

        ce = filter_seed_common(normalize(load_csv(ce_dir / "student_ce_results.csv")), args, seed)
        if not ce.empty:
            ce = ce[ce["method"] == "EEGNet-CE"].copy()
            frames.append(ce)

        riemann = filter_seed_common(normalize(load_csv(riemann_dir / "riemann_baseline_results.csv")), args, seed)
        if not riemann.empty:
            riemann = riemann[riemann["method"].isin(["Riemann-LogReg", "Riemann-MDM"])].copy()
            frames.append(riemann)

        ensemble = filter_seed_common(
            normalize(load_csv(ensemble_dir / "eegnet_riemann_ensemble_results.csv")),
            args,
            seed,
        )
        if not ensemble.empty:
            ensemble = ensemble[ensemble["method"] == "EEGNet-Riemann-Ensemble"].copy()
            frames.append(ensemble)

    official = filter_common(normalize(load_csv(Path("results/eegnet_official/eegnet_official_results.csv"))), args)
    if not official.empty:
        official = official[
            official["method"].str.startswith("EEGNetv4")
            | official["method"].str.startswith("EEGNet-Compact")
        ].copy()
        frames.append(official)

    csp = filter_common(normalize(load_csv(Path("results/csp_baseline/csp_baseline_results.csv"))), args)
    if not csp.empty:
        csp = csp[
            csp["method"].isin(
                [
                    "CSP-LDA",
                    "CSP-LogReg",
                    "FBCSP-LDA",
                    "FBCSP-LogReg",
                    "FBCSP-LDA-ANOVA",
                    "FBCSP-LogReg-ANOVA",
                    "FBCSP-LDA-MI",
                    "FBCSP-LogReg-MI",
                ]
            )
        ].copy()
        frames.append(csp)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True, sort=False)
    df = df.drop_duplicates(["method", "subject", "window", "seed"], keep="last")
    df = df.sort_values(
        ["window", "method", "seed", "subject"],
        key=lambda col: method_order(col) if col.name == "method" else col,
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
        .sort_values(
            ["window", "method"],
            key=lambda col: method_order(col) if col.name == "method" else col,
        )
    )
    for column in ["mean_acc", "std_acc", "mean_f1", "std_f1", "mean_kappa", "std_kappa"]:
        summary[column] = summary[column].round(4)
    return summary


def diagnostic_best_csp(df: pd.DataFrame) -> pd.DataFrame:
    csp_methods = [
        "CSP-LDA",
        "CSP-LogReg",
        "FBCSP-LDA",
        "FBCSP-LogReg",
        "FBCSP-LDA-ANOVA",
        "FBCSP-LogReg-ANOVA",
        "FBCSP-LDA-MI",
        "FBCSP-LogReg-MI",
    ]
    csp = df[df["method"].isin(csp_methods)].copy()
    if csp.empty:
        return csp
    return (
        csp.sort_values("accuracy", ascending=False)
        .groupby(["window", "seed", "subject"], as_index=False)
        .head(1)
        .sort_values(["window", "seed", "subject"])
        .reset_index(drop=True)
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = collect(args)
    if df.empty:
        raise RuntimeError("No baseline results found for the requested filters.")

    subject_path = output_dir / "strong_baseline_subject_seed.csv"
    summary_path = output_dir / "strong_baseline_summary.csv"
    best_csp_path = output_dir / "strong_baseline_best_csp_diagnostic.csv"

    df.to_csv(subject_path, index=False)
    summary = summarize(df)
    summary.to_csv(summary_path, index=False)
    diagnostic_best_csp(df).to_csv(best_csp_path, index=False)

    print(f"Saved subject-seed table: {subject_path}")
    print(f"Saved summary table: {summary_path}")
    print(f"Saved diagnostic best-CSP table: {best_csp_path}")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
