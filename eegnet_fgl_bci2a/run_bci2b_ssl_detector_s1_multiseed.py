from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


SEEDS = [42, 2024, 3407, 7, 123]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BCI IV-2b subject 1 with SimCLR-pretrained detector."
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs_ssl", type=int, default=100)
    parser.add_argument("--epochs_detector", type=int, default=300)
    parser.add_argument("--epochs_classifier", type=int, default=300)
    parser.add_argument("--patience_detector", type=int, default=80)
    parser.add_argument("--patience_classifier", type=int, default=100)
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/bci2b_ssl_detector_subject1_multiseed",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.save_dir)
    results_csv = root / "ssl_detector_results.csv"
    root.mkdir(parents=True, exist_ok=True)

    for seed in SEEDS:
        if results_csv.exists():
            existing = pd.read_csv(results_csv)
            matched = (
                (existing["subject"].astype(int) == 1)
                & (existing["seed"].astype(int) == seed)
                & (existing["detector_pretrain"].astype(str) == "simclr")
            )
            if matched.any():
                print(f"Skip existing SSL detector result: seed={seed}", flush=True)
                continue

        command = [
            sys.executable,
            "run_bci2b_routing_candidates.py",
            "--subject",
            "1",
            "--seed",
            str(seed),
            "--detector_pretrain",
            "simclr",
            "--epochs_ssl",
            str(args.epochs_ssl),
            "--epochs_detector",
            str(args.epochs_detector),
            "--epochs_classifier",
            str(args.epochs_classifier),
            "--patience_detector",
            str(args.patience_detector),
            "--patience_classifier",
            str(args.patience_classifier),
            "--device",
            args.device,
            "--save_dir",
            str(root / f"seed_{seed}"),
            "--results_csv",
            str(results_csv),
        ]
        print("\n" + " ".join(command), flush=True)
        subprocess.run(command, check=True)

    ssl_rows = pd.read_csv(results_csv).sort_values("seed")
    baseline_path = Path(
        "results/global_routing_multiseed/all_subject_seed_results.csv"
    )
    baseline = pd.read_csv(baseline_path)
    baseline = baseline[
        (baseline["dataset"] == "bci2b") & (baseline["subject"].astype(int) == 1)
    ][["seed", "adaptive_ce_test_acc"]].rename(
        columns={"adaptive_ce_test_acc": "supervised_detector_acc"}
    )
    comparison = ssl_rows[
        [
            "seed",
            "detector_val_acc",
            "stable_peak_confidence_test_avg_start",
            "stable_peak_confidence_acc",
            "stable_peak_confidence_f1",
            "stable_peak_confidence_kappa",
            "ssl_final_loss",
        ]
    ].rename(columns={"stable_peak_confidence_acc": "ssl_detector_acc"})
    comparison = baseline.merge(comparison, on="seed", how="inner")
    comparison["improvement"] = (
        comparison["ssl_detector_acc"] - comparison["supervised_detector_acc"]
    )
    comparison.to_csv(root / "comparison_vs_supervised_detector.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "subject": 1,
                "n_seeds": len(comparison),
                "supervised_detector_mean_acc": comparison[
                    "supervised_detector_acc"
                ].mean(),
                "ssl_detector_mean_acc": comparison["ssl_detector_acc"].mean(),
                "mean_improvement": comparison["improvement"].mean(),
                "ssl_detector_std_acc": comparison["ssl_detector_acc"].std(ddof=1),
                "wins": int((comparison["improvement"] > 0).sum()),
                "ties": int((comparison["improvement"] == 0).sum()),
                "losses": int((comparison["improvement"] < 0).sum()),
            }
        ]
    )
    summary.to_csv(root / "summary.csv", index=False)
    print("\nSSL DETECTOR COMPARISON", flush=True)
    print(comparison.to_string(index=False), flush=True)
    print("\nSUMMARY", flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
