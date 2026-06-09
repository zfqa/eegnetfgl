from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT_DIR = RESULTS / "global_routing_multiseed"

METHODS = [
    "EEGNetV4 2s CE",
    "RiemannPaper-TS-LogReg 2s",
    "CSP-LDA 2s",
    "CSP-LDA 4s",
    "EEGNetV4 4s CE",
    "RiemannPaper-TS-LogReg 4s",
]


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values, dtype=float)
    running_max = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        value = min(1.0, (total - rank) * float(p_values[index]))
        running_max = max(running_max, value)
        adjusted[index] = running_max
    return adjusted


def load_baselines(dataset: str) -> pd.DataFrame:
    if dataset == "bci2a":
        neural = pd.read_csv(
            RESULTS
            / "publication_complete_analysis"
            / "bci2a_eegnetv4_strong_rows_deduplicated.csv"
        )
        classical = pd.read_csv(
            RESULTS
            / "publication_complete_analysis"
            / "bci2a_classical_rows_deduplicated.csv"
        )
    else:
        neural = pd.read_csv(
            RESULTS
            / "publication_complete_analysis"
            / "bci2b_deep_rows_deduplicated.csv"
        )
        classical = pd.read_csv(
            RESULTS
            / "publication_complete_analysis"
            / "bci2b_classical_rows_deduplicated.csv"
        )
    frame = pd.concat([neural, classical], ignore_index=True, sort=False)
    return frame[frame["label"].isin(METHODS)].copy()


def paired_rows(dataset: str, routing: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = load_baselines(dataset)
    baseline_means = (
        baseline.groupby(["subject", "label"], as_index=False)["accuracy"]
        .mean()
        .rename(columns={"label": "method", "accuracy": "baseline_accuracy"})
    )
    routing_means = (
        routing[routing["dataset"] == dataset][["subject", "global_test_acc"]]
        .rename(columns={"global_test_acc": "routing_accuracy"})
        .copy()
    )
    long_rows = baseline_means.merge(routing_means, on="subject", how="inner")
    long_rows["difference"] = long_rows["routing_accuracy"] - long_rows["baseline_accuracy"]

    rows = []
    for method in METHODS:
        paired = long_rows[long_rows["method"] == method].sort_values("subject")
        differences = paired["difference"].to_numpy(dtype=float)
        t_p = float(
            ttest_rel(paired["routing_accuracy"], paired["baseline_accuracy"]).pvalue
        )
        try:
            wilcoxon_p = float(
                wilcoxon(differences, zero_method="wilcox", alternative="two-sided").pvalue
            )
        except ValueError:
            wilcoxon_p = 1.0
        std = differences.std(ddof=1)
        rows.append(
            {
                "dataset": dataset,
                "comparison": f"Global routing vs {method}",
                "method": method,
                "n_subjects": len(paired),
                "routing_mean_accuracy": paired["routing_accuracy"].mean(),
                "baseline_mean_accuracy": paired["baseline_accuracy"].mean(),
                "mean_difference": differences.mean(),
                "wins": int((differences > 1e-12).sum()),
                "ties": int((np.abs(differences) <= 1e-12).sum()),
                "losses": int((differences < -1e-12).sum()),
                "paired_t_p": t_p,
                "wilcoxon_p": wilcoxon_p,
                "cohen_dz": differences.mean() / std if std > 0 else np.nan,
            }
        )

    tests = pd.DataFrame(rows)
    tests["paired_t_p_holm"] = holm_adjust(tests["paired_t_p"].to_numpy())
    tests["wilcoxon_p_holm"] = holm_adjust(tests["wilcoxon_p"].to_numpy())
    return tests, long_rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    routing = pd.read_csv(OUT_DIR / "subject_mean_across_seeds.csv")
    test_frames = []
    subject_frames = []
    for dataset in ["bci2a", "bci2b"]:
        tests, subject_rows = paired_rows(dataset, routing)
        test_frames.append(tests)
        subject_frames.append(subject_rows.assign(dataset=dataset))

    tests = pd.concat(test_frames, ignore_index=True)
    subjects = pd.concat(subject_frames, ignore_index=True)
    tests.to_csv(OUT_DIR / "routing_vs_external_baselines_paired_tests.csv", index=False)
    subjects.to_csv(
        OUT_DIR / "routing_vs_external_baselines_subject_means.csv", index=False
    )
    print(
        tests[
            [
                "dataset",
                "method",
                "mean_difference",
                "wins",
                "ties",
                "losses",
                "paired_t_p",
                "wilcoxon_p",
                "cohen_dz",
                "paired_t_p_holm",
                "wilcoxon_p_holm",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
