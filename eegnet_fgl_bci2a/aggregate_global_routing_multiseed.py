from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


ROOT = Path("results/global_routing_multiseed")
SEEDS = [42, 2024, 3407, 7, 123]
DATASETS = ["bci2a", "bci2b"]
METHOD_COLUMNS = {
    "fixed_ce": "fixed_ce_test_acc",
    "adaptive_ce": "adaptive_ce_test_acc",
    "fixed_dual": "fixed_dual_test_acc",
    "adaptive_dual": "adaptive_dual_test_acc",
    "global_routing": "global_test_acc",
}


def paired_row(dataset: str, subject_means: pd.DataFrame, reference: str) -> dict[str, float | int | str]:
    selected = subject_means["global_test_acc"].to_numpy(dtype=float)
    baseline = subject_means[reference].to_numpy(dtype=float)
    differences = selected - baseline
    return {
        "dataset": dataset,
        "selected": "global_test_acc",
        "reference": reference,
        "n_subjects": len(subject_means),
        "selected_mean": float(selected.mean()),
        "reference_mean": float(baseline.mean()),
        "mean_difference": float(differences.mean()),
        "wins": int((differences > 0).sum()),
        "ties": int(np.isclose(differences, 0.0).sum()),
        "losses": int((differences < 0).sum()),
        "paired_t_p": float(ttest_rel(selected, baseline).pvalue),
        "wilcoxon_p": float(wilcoxon(selected, baseline).pvalue),
    }


def main() -> None:
    all_rows = []
    seed_rows = []
    for seed in SEEDS:
        for dataset in DATASETS:
            subject_path = ROOT / f"seed_{seed}" / f"{dataset}_routing" / "subject_results.csv"
            frame = pd.read_csv(subject_path)
            all_rows.append(frame)
            row = {"dataset": dataset, "seed": seed}
            for method, column in METHOD_COLUMNS.items():
                row[f"{method}_mean_acc"] = float(frame[column].mean())
            row["global_mean_f1"] = float(frame["global_test_f1"].mean())
            row["global_mean_kappa"] = float(frame["global_test_kappa"].mean())
            seed_rows.append(row)

    subjects = pd.concat(all_rows, ignore_index=True)
    seed_summary = pd.DataFrame(seed_rows)
    seed_summary.to_csv(ROOT / "all_seed_dataset_summary.csv", index=False)
    subjects.to_csv(ROOT / "all_subject_seed_results.csv", index=False)

    aggregate_rows = []
    for dataset, group in seed_summary.groupby("dataset"):
        for method in METHOD_COLUMNS:
            values = group[f"{method}_mean_acc"].to_numpy(dtype=float)
            aggregate_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "mean_acc": float(values.mean()),
                    "std_across_seeds": float(values.std(ddof=1)),
                    "min_seed_acc": float(values.min()),
                    "max_seed_acc": float(values.max()),
                }
            )
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(ROOT / "aggregate_method_summary.csv", index=False)

    subject_means = (
        subjects.groupby(["dataset", "subject"], as_index=False)[list(METHOD_COLUMNS.values())]
        .mean()
        .sort_values(["dataset", "subject"])
    )
    subject_means.to_csv(ROOT / "subject_mean_across_seeds.csv", index=False)

    tests = []
    for dataset, group in subject_means.groupby("dataset"):
        for reference in [
            "fixed_ce_test_acc",
            "adaptive_ce_test_acc",
            "fixed_dual_test_acc",
            "adaptive_dual_test_acc",
        ]:
            tests.append(paired_row(dataset, group, reference))
    pd.DataFrame(tests).to_csv(ROOT / "paired_tests_subject_means.csv", index=False)

    selection_counts = (
        subjects.groupby(["dataset", "global_selected"])
        .size()
        .rename("count")
        .reset_index()
    )
    selection_counts["ratio"] = selection_counts.groupby("dataset")["count"].transform(
        lambda values: values / values.sum()
    )
    selection_counts.to_csv(ROOT / "selection_counts.csv", index=False)

    combined = (
        seed_summary.groupby("seed", as_index=False)["global_routing_mean_acc"]
        .mean()
        .rename(columns={"global_routing_mean_acc": "combined_dataset_mean_acc"})
    )
    combined.to_csv(ROOT / "combined_seed_summary.csv", index=False)

    print("\nPer-seed results")
    print(seed_summary.to_string(index=False))
    print("\nAcross-seed method summary")
    print(aggregate.to_string(index=False))
    print("\nSelection counts")
    print(selection_counts.to_string(index=False))
    print("\nCombined datasets")
    print(combined.to_string(index=False))


if __name__ == "__main__":
    main()
