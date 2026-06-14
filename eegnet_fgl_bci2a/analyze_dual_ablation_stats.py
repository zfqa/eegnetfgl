"""Compute paired statistical tests for the dual-teacher ablation on BCI IV-2a.

Comparisons:
  1. Dual KD vs CE only
  2. Dual KD vs EEGNet teacher only
  3. Dual KD vs Riemann teacher only
  4. EEGNet teacher only vs CE only
  5. Riemann teacher only vs CE only

Each comparison first averages accuracy over five seeds within each subject,
then applies paired t-test and Wilcoxon signed-rank test across nine subjects.
Cohen's dz and W/T/L counts are also reported.

Input:
  - results/paper_moabb_multiseed/ce_compact/seed_*/student_ce_results.csv
  - results/paper_moabb_multiseed/dual_ablation/seed_*/dual_kd_results.csv

Output:
  - results/dual_ablation_paired_tests.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
CE_ROOT = ROOT / "results" / "paper_moabb_multiseed" / "ce_compact"
ABL_ROOT = ROOT / "results" / "paper_moabb_multiseed" / "dual_ablation"
OUT_PATH = ROOT / "results" / "dual_ablation_paired_tests.csv"

SEEDS = [42, 2024, 3407, 7, 123]


def load_ce_subject_means() -> pd.DataFrame:
    dfs = []
    for seed in SEEDS:
        f = CE_ROOT / f"seed_{seed}" / "student_ce_results.csv"
        df = pd.read_csv(f)
        df = df[df["window"] == 2.0][["subject", "accuracy", "seed"]]
        dfs.append(df)
    all_ce = pd.concat(dfs, ignore_index=True)
    return all_ce.groupby("subject")["accuracy"].mean().reset_index().rename(columns={"accuracy": "ce_acc"})


def load_ablation_subject_means() -> pd.DataFrame:
    dfs = []
    for seed in SEEDS:
        f = ABL_ROOT / f"seed_{seed}" / "dual_kd_results.csv"
        df = pd.read_csv(f)
        df = df[["method", "subject", "accuracy", "seed"]]
        dfs.append(df)
    all_abl = pd.concat(dfs, ignore_index=True)
    return all_abl.groupby(["method", "subject"])["accuracy"].mean().reset_index()


def cohens_dz(diff: np.ndarray) -> float:
    return float(np.mean(diff) / np.std(diff, ddof=1))


def paired_stats(a: np.ndarray, b: np.ndarray, label_a: str, label_b: str) -> dict:
    diff = a - b
    n = len(diff)
    wins = int(np.sum(diff > 0))
    losses = int(np.sum(diff < 0))
    ties = int(np.sum(diff == 0))
    t_stat, t_p = stats.ttest_rel(a, b)
    w_stat, w_p = stats.wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
    return {
        "comparison": f"{label_a} vs {label_b}",
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "mean_diff": float(np.mean(diff)),
        "n_subjects": n,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "cohens_dz": cohens_dz(diff),
        "paired_t_p": float(t_p),
        "wilcoxon_p": float(w_p),
    }


def main() -> None:
    ce = load_ce_subject_means()
    abl = load_ablation_subject_means()

    eeg_only = abl[abl["method"] == "EEGNet teacher only"].set_index("subject")["accuracy"]
    rie_only = abl[abl["method"] == "Riemann teacher only"].set_index("subject")["accuracy"]
    dual = abl[abl["method"] == "Riemann-guided dual KD"].set_index("subject")["accuracy"]
    ce_acc = ce.set_index("subject")["ce_acc"]

    subjects = sorted(ce_acc.index)
    ce_arr = ce_acc.loc[subjects].values
    eeg_arr = eeg_only.loc[subjects].values
    rie_arr = rie_only.loc[subjects].values
    dual_arr = dual.loc[subjects].values

    rows = [
        paired_stats(dual_arr, ce_arr, "Dual KD", "CE only"),
        paired_stats(dual_arr, eeg_arr, "Dual KD", "EEGNet teacher only"),
        paired_stats(dual_arr, rie_arr, "Dual KD", "Riemann teacher only"),
        paired_stats(eeg_arr, ce_arr, "EEGNet teacher only", "CE only"),
        paired_stats(rie_arr, ce_arr, "Riemann teacher only", "CE only"),
    ]

    out_df = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False)

    print("=== Dual-teacher ablation paired tests (BCI IV-2a, 9 subjects, 5 seeds) ===\n")
    for r in rows:
        sig = ""
        if r["paired_t_p"] < 0.05:
            sig += " *t"
        if r["wilcoxon_p"] < 0.05:
            sig += " *W"
        print(f"  {r['comparison']}")
        print(f"    mean: {r['mean_a']:.4f} vs {r['mean_b']:.4f} (diff={r['mean_diff']:+.4f})")
        print(f"    W/T/L: {r['wins']}/{r['ties']}/{r['losses']}  dz={r['cohens_dz']:.2f}")
        print(f"    t-test p={r['paired_t_p']:.4f}  Wilcoxon p={r['wilcoxon_p']:.4f}{sig}")
        print()

    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
