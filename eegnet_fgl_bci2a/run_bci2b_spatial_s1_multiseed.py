from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


SEEDS = [42, 2024, 3407, 7, 123]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BCI IV-2b subject 1 adaptive dual KD with spatial loss."
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--spatial_weight", type=float, default=0.1)
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/bci2b_subject1_adaptive_dual_spatial_multiseed",
    )
    return parser.parse_args()


def find_teacher(subject: int, seed: int) -> Path:
    root = Path("results/bci2b_paper_multiseed") / f"seed_{seed}" / "eegnetv4_4s_teacher"
    matches = list(root.glob(f"bci2b_subject_{subject}_eegnetv4_window_4_*_best.pth"))
    if seed == 42 and not matches:
        root = Path("results/bci2b_eegnetv4_4s_teachers_seed42")
        matches = list(root.glob(f"bci2b_subject_{subject}_eegnetv4_window_4_*_best.pth"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one teacher for seed={seed}, found {len(matches)}.")
    return matches[0]


def main() -> None:
    args = parse_args()
    root = Path(args.save_dir)
    root.mkdir(parents=True, exist_ok=True)
    subject = 1

    for seed in SEEDS:
        seed_dir = root / f"seed_{seed}"
        results_path = seed_dir / "detector_dual_teacher_results.csv"
        if results_path.exists():
            rows = pd.read_csv(results_path)
            if (
                (rows["subject"].astype(int) == subject)
                & (rows["seed"].astype(int) == seed)
                & (rows["spatial_structure_weight"].astype(float) == args.spatial_weight)
            ).any():
                print(f"Skip existing spatial result: seed={seed}", flush=True)
                continue

        detector = (
            Path("results/global_routing_multiseed")
            / f"seed_{seed}"
            / "bci2b_candidates"
            / f"subject_{subject}"
            / "eegnetv4_05s_sliding_detector_best.pth"
        )
        command = [
            sys.executable,
            "run_bci2b_detector_dual_teacher_subject1.py",
            "--subject",
            str(subject),
            "--seed",
            str(seed),
            "--detector_path",
            str(detector),
            "--eegnet_teacher_path",
            str(find_teacher(subject, seed)),
            "--spatial_structure_weight",
            str(args.spatial_weight),
            "--spatial_structure_source",
            "filter",
            "--spatial_structure_mode",
            "corr",
            "--spatial_structure_trust_mode",
            "sample",
            "--epochs",
            str(args.epochs),
            "--patience",
            str(args.patience),
            "--device",
            args.device,
            "--save_dir",
            str(seed_dir),
        ]
        print("\n" + " ".join(command), flush=True)
        subprocess.run(command, check=True)

    new_rows = []
    for seed in SEEDS:
        path = root / f"seed_{seed}" / "detector_dual_teacher_results.csv"
        frame = pd.read_csv(path)
        row = frame[
            (frame["subject"].astype(int) == subject)
            & (frame["seed"].astype(int) == seed)
            & (frame["spatial_structure_weight"].astype(float) == args.spatial_weight)
        ].iloc[-1]
        new_rows.append(row)
    spatial = pd.DataFrame(new_rows)

    old = pd.read_csv("results/global_routing_multiseed/all_subject_seed_results.csv")
    old = old[(old["dataset"] == "bci2b") & (old["subject"].astype(int) == subject)]
    comparison = old[
        ["seed", "adaptive_ce_test_acc", "adaptive_dual_test_acc"]
    ].merge(
        spatial[["seed", "accuracy", "macro_f1", "kappa", "best_epoch"]],
        on="seed",
        how="inner",
    )
    comparison = comparison.rename(columns={"accuracy": "adaptive_dual_spatial_acc"})
    comparison["spatial_vs_dual"] = (
        comparison["adaptive_dual_spatial_acc"] - comparison["adaptive_dual_test_acc"]
    )
    comparison["spatial_vs_ce"] = (
        comparison["adaptive_dual_spatial_acc"] - comparison["adaptive_ce_test_acc"]
    )
    comparison.to_csv(root / "comparison.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "subject": subject,
                "n_seeds": len(comparison),
                "adaptive_ce_mean_acc": comparison["adaptive_ce_test_acc"].mean(),
                "adaptive_dual_mean_acc": comparison["adaptive_dual_test_acc"].mean(),
                "adaptive_dual_spatial_mean_acc": comparison[
                    "adaptive_dual_spatial_acc"
                ].mean(),
                "spatial_vs_dual_mean": comparison["spatial_vs_dual"].mean(),
                "spatial_vs_ce_mean": comparison["spatial_vs_ce"].mean(),
                "spatial_vs_dual_wins": int((comparison["spatial_vs_dual"] > 0).sum()),
                "spatial_vs_ce_wins": int((comparison["spatial_vs_ce"] > 0).sum()),
            }
        ]
    )
    summary.to_csv(root / "summary.csv", index=False)
    print("\nCOMPARISON", flush=True)
    print(comparison.to_string(index=False), flush=True)
    print("\nSUMMARY", flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
