from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate paper experiments, run paired tests, and generate figures.")
    parser.add_argument("--results_roots", type=str, nargs="+", default=["results/paper_moabb_multiseed", "results"])
    parser.add_argument("--output_dir", type=str, default="results/paper_analysis")
    parser.add_argument("--baseline_label", type=str, default="Compact EEGNet 2s")
    parser.add_argument("--mainline_label", type=str, default="Riemann-guided dual KD 2s")
    parser.add_argument(
        "--exclude_patterns",
        type=str,
        default="tmp,smoke,diffusion,desam,completion,adaptive_window,paper_analysis",
        help="Comma-separated path substrings excluded from recursive result scanning.",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


def parse_excludes(exclude_patterns: str) -> list[str]:
    return [part.strip().lower() for part in exclude_patterns.split(",") if part.strip()]


def read_csvs(roots: list[str], filename: str, excludes: list[str]) -> list[pd.DataFrame]:
    frames = []
    seen: set[Path] = set()
    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for path in root_path.rglob(filename):
            path = path.resolve()
            path_text = str(path).lower()
            if any(pattern in path_text for pattern in excludes):
                continue
            if path in seen:
                continue
            seen.add(path)
            try:
                df = pd.read_csv(path)
            except Exception as exc:
                print(f"Skip unreadable CSV {path}: {exc}")
                continue
            df["_source_csv"] = str(path)
            frames.append(df)
    return frames


def normalize_metric_frame(
    df: pd.DataFrame,
    label: str,
    result_family: str,
    window_from_col: bool = True,
) -> pd.DataFrame:
    required = {"subject", "accuracy", "macro_f1", "kappa"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    out = pd.DataFrame(
        {
            "label": label,
            "result_family": result_family,
            "subject": df["subject"].astype(int),
            "seed": pd.to_numeric(df.get("seed", 42), errors="coerce").fillna(42).astype(int),
            "window": pd.to_numeric(df["window"], errors="coerce") if window_from_col and "window" in df.columns else np.nan,
            "accuracy": pd.to_numeric(df["accuracy"], errors="coerce"),
            "macro_f1": pd.to_numeric(df["macro_f1"], errors="coerce"),
            "kappa": pd.to_numeric(df["kappa"], errors="coerce"),
            "_source_csv": df.get("_source_csv", ""),
        }
    )
    return out.dropna(subset=["accuracy"])


def label_student_ce(df: pd.DataFrame) -> pd.DataFrame:
    if "window" not in df.columns:
        return pd.DataFrame()
    frames = []
    for window, label in [(2.0, "Compact EEGNet 2s"), (4.0, "Compact EEGNet 4s")]:
        sub = df[np.isclose(pd.to_numeric(df["window"], errors="coerce"), window)].copy()
        if not sub.empty:
            frames.append(normalize_metric_frame(sub, label, "compact_ce"))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def label_eegnetv4(df: pd.DataFrame) -> pd.DataFrame:
    if "model_variant" not in df.columns or "window" not in df.columns:
        return pd.DataFrame()
    sub = df[df["model_variant"].astype(str) == "eegnetv4"].copy()
    frames = []
    for window, label in [(2.0, "EEGNetV4 2s"), (4.0, "EEGNetV4 4s")]:
        win = sub[np.isclose(pd.to_numeric(sub["window"], errors="coerce"), window)].copy()
        if not win.empty:
            frames.append(normalize_metric_frame(win, label, "eegnetv4"))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def dual_label(row: pd.Series) -> str:
    mode = str(row.get("teacher_mode", ""))
    method = str(row.get("method", ""))
    if mode == "eegnet":
        return "EEGNet teacher KD 2s"
    if mode == "riemann":
        return "Riemann teacher KD 2s"
    if mode == "dual":
        return "Riemann-guided dual KD 2s"
    if "PaperDualKD" in method or "DualKD" in method:
        return "Riemann-guided dual KD 2s"
    return f"DualKD/{mode}"


def label_dual_kd(df: pd.DataFrame) -> pd.DataFrame:
    if "teacher_mode" not in df.columns:
        return pd.DataFrame()
    sub = df.copy()
    if "window" in sub.columns:
        sub = sub[np.isclose(pd.to_numeric(sub["window"], errors="coerce"), 2.0)].copy()
    out = normalize_metric_frame(sub, "", "dual_kd")
    if out.empty:
        return out
    out["label"] = sub.apply(dual_label, axis=1).to_numpy()
    return out


def riemann_guided_label(row: pd.Series) -> str:
    kl = float(row.get("kl_weight", 0.0))
    geo = float(row.get("geometry_weight", 0.0))
    spatial = float(row.get("spatial_structure_weight", 0.0))
    eeg = float(row.get("eegnet_kd_weight", 0.0))
    parts = []
    if eeg > 0:
        parts.append("EEGNetKD")
    if kl > 0:
        parts.append("RiemannKD")
    if geo > 0:
        parts.append("Geometry")
    if spatial > 0:
        parts.append("Spatial")
    if not parts:
        return "Riemann guided CE-only"
    return " + ".join(parts) + " 2s"


def label_riemann_guided(df: pd.DataFrame) -> pd.DataFrame:
    if "kl_weight" not in df.columns:
        return pd.DataFrame()
    sub = df.copy()
    if "window" in sub.columns:
        sub = sub[np.isclose(pd.to_numeric(sub["window"], errors="coerce"), 2.0)].copy()
    out = normalize_metric_frame(sub, "", "riemann_guided")
    if out.empty:
        return out
    out["label"] = sub.apply(riemann_guided_label, axis=1).to_numpy()
    return out


def label_riemann_paper(df: pd.DataFrame) -> pd.DataFrame:
    if "method" not in df.columns or "estimator" not in df.columns:
        if "method" not in df.columns or "cov_estimator" not in df.columns:
            return pd.DataFrame()
    frames = []
    method_map = {
        "mdm": "Riemann MDM",
        "riemannpaper-mdm": "Riemann MDM",
        "ts_logreg": "Riemann LogReg",
        "riemannpaper-ts-logreg": "Riemann LogReg",
    }
    method_col = df["method"].astype(str).str.lower()
    for method, base_label in method_map.items():
        for window in [2.0, 4.0]:
            sub = df[
                (method_col == method)
                & np.isclose(pd.to_numeric(df["window"], errors="coerce"), window)
            ].copy()
            if sub.empty:
                continue
            frames.append(normalize_metric_frame(sub, f"{base_label} {window:g}s", "riemann_baseline"))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def label_csp(df: pd.DataFrame) -> pd.DataFrame:
    if "method" not in df.columns:
        return pd.DataFrame()
    frames = []
    for window in [2.0, 4.0]:
        sub = df[
            df["method"].astype(str).str.contains("CSP", case=False, na=False)
            & np.isclose(pd.to_numeric(df["window"], errors="coerce"), window)
        ].copy()
        if sub.empty:
            continue
        frames.append(normalize_metric_frame(sub, f"CSP + LDA {window:g}s", "csp"))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def collect_results(roots: list[str], excludes: list[str]) -> pd.DataFrame:
    frames = []
    for df in read_csvs(roots, "student_ce_results.csv", excludes):
        frames.append(label_student_ce(df))
    for df in read_csvs(roots, "eegnet_official_results.csv", excludes):
        frames.append(label_eegnetv4(df))
    for df in read_csvs(roots, "dual_kd_results.csv", excludes):
        frames.append(label_dual_kd(df))
    for df in read_csvs(roots, "riemann_guided_results.csv", excludes):
        frames.append(label_riemann_guided(df))
    for df in read_csvs(roots, "riemann_paper_results.csv", excludes):
        frames.append(label_riemann_paper(df))
    for df in read_csvs(roots, "csp_baseline_results.csv", excludes):
        frames.append(label_csp(df))
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        raise RuntimeError("No supported result CSVs found.")
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(["label", "subject", "seed", "window", "accuracy", "_source_csv"])
    return out


def summarize(long_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        long_df.groupby("label", as_index=False)
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
        .sort_values("mean_acc", ascending=False)
    )
    summary["sem_acc"] = summary["std_acc"] / np.sqrt(summary["n"].clip(lower=1))
    return summary


def paired_tests(long_df: pd.DataFrame, baseline_label: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    baseline = long_df[long_df["label"] == baseline_label][["subject", "seed", "accuracy"]].rename(
        columns={"accuracy": "baseline_acc"}
    )
    if baseline.empty:
        print(f"Baseline label not found for paired tests: {baseline_label}")
        return pd.DataFrame()
    for label in sorted(long_df["label"].unique()):
        if label == baseline_label:
            continue
        current = long_df[long_df["label"] == label][["subject", "seed", "accuracy"]].rename(
            columns={"accuracy": "method_acc"}
        )
        paired = baseline.merge(current, on=["subject", "seed"], how="inner")
        if len(paired) < 2:
            continue
        diff = paired["method_acc"] - paired["baseline_acc"]
        try:
            t_stat, t_p = stats.ttest_rel(paired["method_acc"], paired["baseline_acc"])
        except Exception:
            t_stat, t_p = np.nan, np.nan
        try:
            w_stat, w_p = stats.wilcoxon(diff)
        except Exception:
            w_stat, w_p = np.nan, np.nan
        rows.append(
            {
                "baseline": baseline_label,
                "method": label,
                "n_pairs": len(paired),
                "mean_baseline_acc": paired["baseline_acc"].mean(),
                "mean_method_acc": paired["method_acc"].mean(),
                "mean_diff": diff.mean(),
                "median_diff": diff.median(),
                "wins": int((diff > 0).sum()),
                "losses": int((diff < 0).sum()),
                "ties": int((diff == 0).sum()),
                "paired_t_stat": t_stat,
                "paired_t_p": t_p,
                "wilcoxon_stat": w_stat,
                "wilcoxon_p": w_p,
            }
        )
    return pd.DataFrame(rows).sort_values("mean_diff", ascending=False) if rows else pd.DataFrame()


def plot_bar(summary: pd.DataFrame, out_dir: Path) -> None:
    top = summary.sort_values("mean_acc", ascending=True)
    fig_height = max(5, 0.35 * len(top))
    plt.figure(figsize=(10, fig_height))
    plt.barh(top["label"], top["mean_acc"], xerr=top["sem_acc"].fillna(0.0), color="#4c78a8")
    plt.xlabel("Accuracy")
    plt.xlim(0, min(1.0, max(0.8, float(top["mean_acc"].max()) + 0.08)))
    plt.title("Mean Accuracy by Method")
    plt.tight_layout()
    plt.savefig(out_dir / "method_mean_accuracy_bar.png", dpi=200)
    plt.close()


def plot_subject_improvements(long_df: pd.DataFrame, baseline_label: str, mainline_label: str, out_dir: Path) -> None:
    base = (
        long_df[long_df["label"] == baseline_label]
        .groupby("subject", as_index=False)["accuracy"]
        .mean()
        .rename(columns={"accuracy": "baseline"})
    )
    main = (
        long_df[long_df["label"] == mainline_label]
        .groupby("subject", as_index=False)["accuracy"]
        .mean()
        .rename(columns={"accuracy": "mainline"})
    )
    paired = base.merge(main, on="subject", how="inner")
    if paired.empty:
        return
    paired["delta"] = paired["mainline"] - paired["baseline"]
    colors = ["#59a14f" if value >= 0 else "#e15759" for value in paired["delta"]]
    plt.figure(figsize=(9, 4))
    plt.bar(paired["subject"].astype(str), paired["delta"], color=colors)
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Subject")
    plt.ylabel("Accuracy improvement")
    plt.title(f"{mainline_label} vs {baseline_label}")
    plt.tight_layout()
    plt.savefig(out_dir / "subject_improvement_mainline_vs_baseline.png", dpi=200)
    plt.close()


def plot_confidence_diagnostics(roots: list[str], excludes: list[str], out_dir: Path) -> None:
    frames = read_csvs(roots, "dual_kd_results.csv", excludes)
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True)
    needed = {"accuracy", "riemann_val_accuracy", "teacher_mode"}
    if not needed.issubset(df.columns):
        return
    df = df[df["teacher_mode"].astype(str).isin(["riemann", "dual"])].copy()
    if df.empty:
        return
    df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")
    df["riemann_val_accuracy"] = pd.to_numeric(df["riemann_val_accuracy"], errors="coerce")
    df = df.dropna(subset=["accuracy", "riemann_val_accuracy"])
    if df.empty:
        return
    plt.figure(figsize=(6, 5))
    for mode, sub in df.groupby("teacher_mode"):
        plt.scatter(sub["riemann_val_accuracy"], sub["accuracy"], label=mode, alpha=0.75)
    plt.xlabel("Riemann teacher validation accuracy")
    plt.ylabel("Student test accuracy")
    plt.title("Riemann Confidence Diagnostic")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "riemann_val_vs_student_test_scatter.png", dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    excludes = parse_excludes(args.exclude_patterns)
    long_df = collect_results(args.results_roots, excludes)
    summary = summarize(long_df)
    tests = paired_tests(long_df, args.baseline_label)

    long_path = out_dir / "paper_long_results.csv"
    summary_path = out_dir / "paper_summary_by_method.csv"
    tests_path = out_dir / "paper_paired_significance_tests.csv"
    pivot_path = out_dir / "paper_subject_seed_accuracy_pivot.csv"

    long_df.to_csv(long_path, index=False)
    summary.to_csv(summary_path, index=False)
    if not tests.empty:
        tests.to_csv(tests_path, index=False)
    pivot = long_df.pivot_table(index=["subject", "seed"], columns="label", values="accuracy", aggfunc="mean")
    pivot.to_csv(pivot_path)

    plot_bar(summary, out_dir)
    plot_subject_improvements(long_df, args.baseline_label, args.mainline_label, out_dir)
    plot_confidence_diagnostics(args.results_roots, excludes, out_dir)

    print(f"Saved long results: {long_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved paired tests: {tests_path if not tests.empty else 'not available'}")
    print(f"Saved pivot: {pivot_path}")
    print("\nTop methods:")
    print(summary.head(12).to_string(index=False))
    if not tests.empty:
        print("\nPaired tests vs baseline:")
        print(tests.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
