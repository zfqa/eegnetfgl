from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


ROUTES = ("fixed_ce", "adaptive_ce", "fixed_dual", "adaptive_dual")
METRICS = ("acc", "f1", "kappa")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate leave-one-seed-out validation-consensus routing without test-set selection."
    )
    parser.add_argument(
        "--input_csv",
        default="results/global_routing_multiseed/all_subject_seed_results.csv",
    )
    parser.add_argument(
        "--output_dir",
        default="results/global_routing_multiseed/cross_seed_consensus",
    )
    return parser.parse_args()


def safe_wilcoxon(x: np.ndarray, y: np.ndarray) -> float:
    if np.allclose(x, y):
        return 1.0
    try:
        return float(wilcoxon(x, y).pvalue)
    except ValueError:
        return np.nan


def select_route(training_rows: pd.DataFrame) -> tuple[str, dict[str, float]]:
    scores = {
        route: float(training_rows[f"{route}_val_acc"].mean())
        for route in ROUTES
    }
    # np.argmax preserves ROUTES order, making ties deterministic and conservative.
    selected = ROUTES[int(np.argmax([scores[route] for route in ROUTES]))]
    return selected, scores


def build_consensus_rows(source: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (dataset, subject), group in source.groupby(["dataset", "subject"], sort=True):
        seeds = sorted(group["seed"].unique())
        for held_out_seed in seeds:
            train_rows = group[group["seed"] != held_out_seed]
            test_row = group[group["seed"] == held_out_seed]
            if train_rows.empty or len(test_row) != 1:
                continue
            selected, validation_scores = select_route(train_rows)
            held_out = test_row.iloc[0]
            row: dict[str, object] = {
                "dataset": dataset,
                "subject": int(subject),
                "held_out_seed": int(held_out_seed),
                "selected_route": selected,
                "current_global_selected": held_out["global_selected"],
            }
            for route in ROUTES:
                row[f"other_seed_{route}_mean_val_acc"] = validation_scores[route]
            for metric in METRICS:
                row[f"consensus_test_{metric}"] = held_out[f"{selected}_test_{metric}"]
                row[f"fixed_ce_test_{metric}"] = held_out[f"fixed_ce_test_{metric}"]
                row[f"current_global_test_{metric}"] = held_out[f"global_test_{metric}"]
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed_summary = (
        rows.groupby(["dataset", "held_out_seed"], as_index=False)
        .agg(
            consensus_acc=("consensus_test_acc", "mean"),
            consensus_f1=("consensus_test_f1", "mean"),
            consensus_kappa=("consensus_test_kappa", "mean"),
            fixed_ce_acc=("fixed_ce_test_acc", "mean"),
            current_global_acc=("current_global_test_acc", "mean"),
        )
    )
    subject_summary = (
        rows.groupby(["dataset", "subject"], as_index=False)
        .agg(
            consensus_acc=("consensus_test_acc", "mean"),
            consensus_f1=("consensus_test_f1", "mean"),
            consensus_kappa=("consensus_test_kappa", "mean"),
            fixed_ce_acc=("fixed_ce_test_acc", "mean"),
            fixed_ce_f1=("fixed_ce_test_f1", "mean"),
            fixed_ce_kappa=("fixed_ce_test_kappa", "mean"),
            current_global_acc=("current_global_test_acc", "mean"),
            current_global_f1=("current_global_test_f1", "mean"),
            current_global_kappa=("current_global_test_kappa", "mean"),
        )
    )

    tests: list[dict[str, object]] = []
    for dataset, group in subject_summary.groupby("dataset", sort=True):
        for baseline in ("fixed_ce", "current_global"):
            for metric in METRICS:
                x = group[f"consensus_{metric}"].to_numpy(dtype=float)
                y = group[f"{baseline}_{metric}"].to_numpy(dtype=float)
                tests.append(
                    {
                        "dataset": dataset,
                        "comparison": f"consensus_vs_{baseline}",
                        "metric": metric,
                        "n_subjects": len(group),
                        "consensus_mean": x.mean(),
                        "baseline_mean": y.mean(),
                        "mean_difference": (x - y).mean(),
                        "paired_t_p": float(ttest_rel(x, y).pvalue),
                        "wilcoxon_p": safe_wilcoxon(x, y),
                    }
                )
    return seed_summary, subject_summary, pd.DataFrame(tests)


def main() -> None:
    args = parse_args()
    source = pd.read_csv(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = build_consensus_rows(source)
    seed_summary, subject_summary, tests = summarize(rows)
    selection_counts = (
        rows.groupby(["dataset", "selected_route"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )

    rows.to_csv(output_dir / "cross_seed_consensus_rows.csv", index=False)
    seed_summary.to_csv(output_dir / "cross_seed_consensus_seed_summary.csv", index=False)
    subject_summary.to_csv(output_dir / "cross_seed_consensus_subject_summary.csv", index=False)
    tests.to_csv(output_dir / "cross_seed_consensus_paired_tests.csv", index=False)
    selection_counts.to_csv(output_dir / "cross_seed_consensus_selection_counts.csv", index=False)

    print("\nCross-seed consensus routing summary")
    print(subject_summary.groupby("dataset")[["consensus_acc", "fixed_ce_acc", "current_global_acc"]].mean())
    print("\nPaired tests on subject means")
    print(tests[tests["metric"] == "acc"].to_string(index=False))


if __name__ == "__main__":
    main()
