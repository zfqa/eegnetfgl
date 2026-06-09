from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all missing global-routing candidates for one seed.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 10)))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=100)
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("\n" + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def csv_has(path: Path, subject: int, seed: int) -> bool:
    if not path.exists():
        return False
    rows = pd.read_csv(path)
    return bool(
        ((rows["subject"].astype(int) == subject) & (rows["seed"].astype(int) == seed)).any()
    )


def bci2b_teacher_path(subject: int, seed: int) -> Path:
    root = Path("results/bci2b_paper_multiseed") / f"seed_{seed}" / "eegnetv4_4s_teacher"
    matches = list(root.glob(f"bci2b_subject_{subject}_eegnetv4_window_4_*_best.pth"))
    if not matches and seed == 42:
        root = Path("results/bci2b_eegnetv4_4s_teachers_seed42")
        matches = list(root.glob(f"bci2b_subject_{subject}_eegnetv4_window_4_*_best.pth"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one BCI2b teacher, found {len(matches)} in {root}.")
    return matches[0]


def main() -> None:
    args = parse_args()
    python = sys.executable
    seed_root = Path("results/global_routing_multiseed") / f"seed_{args.seed}"
    bci2a_candidates = seed_root / "bci2a_candidates"
    bci2a_dual = seed_root / "bci2a_adaptive_dual"
    bci2b_candidates = seed_root / "bci2b_candidates"
    bci2b_dual = seed_root / "bci2b_adaptive_dual"
    for path in [bci2a_candidates, bci2a_dual, bci2b_candidates, bci2b_dual]:
        path.mkdir(parents=True, exist_ok=True)

    for subject in args.subjects:
        bci2a_ce_csv = bci2a_candidates / "routing_ce_results.csv"
        if not csv_has(bci2a_ce_csv, subject, args.seed):
            run(
                [
                    python,
                    "run_bci2a_routing_candidates.py",
                    "--subject",
                    str(subject),
                    "--seed",
                    str(args.seed),
                    "--epochs_eegnet",
                    str(args.epochs),
                    "--patience_eegnet",
                    str(args.patience),
                    "--device",
                    args.device,
                    "--save_dir",
                    str(bci2a_candidates),
                    "--results_csv",
                    str(bci2a_ce_csv),
                ]
            )
        bci2a_dual_csv = bci2a_dual / "adaptive_dual_results.csv"
        if not csv_has(bci2a_dual_csv, subject, args.seed):
            teacher = (
                Path("results/paper_moabb_multiseed/teachers")
                / f"seed_{args.seed}"
                / f"subject_{subject}_teacher_best.pth"
            )
            run(
                [
                    python,
                    "run_bci2a_adaptive_dual_teacher.py",
                    "--subject",
                    str(subject),
                    "--seed",
                    str(args.seed),
                    "--epochs",
                    str(args.epochs),
                    "--patience",
                    str(args.patience),
                    "--device",
                    args.device,
                    "--save_dir",
                    str(bci2a_dual),
                    "--adaptive_results",
                    str(bci2a_ce_csv),
                    "--adaptive_root",
                    str(bci2a_candidates),
                    "--eegnet_teacher_path",
                    str(teacher),
                ]
            )

        bci2b_ce_csv = bci2b_candidates / "routing_ce_results.csv"
        if not csv_has(bci2b_ce_csv, subject, args.seed):
            run(
                [
                    python,
                    "run_bci2b_routing_candidates.py",
                    "--subject",
                    str(subject),
                    "--seed",
                    str(args.seed),
                    "--epochs_classifier",
                    str(args.epochs),
                    "--patience_classifier",
                    str(args.patience),
                    "--device",
                    args.device,
                    "--save_dir",
                    str(bci2b_candidates),
                    "--results_csv",
                    str(bci2b_ce_csv),
                ]
            )
        bci2b_dual_csv = bci2b_dual / "detector_dual_teacher_results.csv"
        if not csv_has(bci2b_dual_csv, subject, args.seed):
            run(
                [
                    python,
                    "run_bci2b_detector_dual_teacher_subject1.py",
                    "--subject",
                    str(subject),
                    "--seed",
                    str(args.seed),
                    "--detector_path",
                    str(
                        bci2b_candidates
                        / f"subject_{subject}"
                        / "eegnetv4_05s_sliding_detector_best.pth"
                    ),
                    "--eegnet_teacher_path",
                    str(bci2b_teacher_path(subject, args.seed)),
                    "--detector_win",
                    "1.0",
                    "--detector_step",
                    "0.25",
                    "--class_window",
                    "2.0",
                    "--smooth_k",
                    "3",
                    "--epochs",
                    str(args.epochs),
                    "--patience",
                    str(args.patience),
                    "--device",
                    args.device,
                    "--save_dir",
                    str(bci2b_dual),
                ]
            )

    for dataset, candidates, dual in [
        ("bci2a", bci2a_candidates, bci2a_dual),
        ("bci2b", bci2b_candidates, bci2b_dual),
    ]:
        output_dir = seed_root / f"{dataset}_routing"
        run(
            [
                python,
                "evaluate_global_routing_seed.py",
                "--dataset",
                dataset,
                "--seed",
                str(args.seed),
                "--subjects",
                *[str(subject) for subject in args.subjects],
                "--candidate_root",
                str(candidates),
                "--adaptive_dual_root",
                str(dual),
                "--device",
                args.device,
                "--output_dir",
                str(output_dir),
            ]
        )

    summaries = []
    for dataset in ["bci2a", "bci2b"]:
        summaries.append(pd.read_csv(seed_root / f"{dataset}_routing" / "summary.csv").iloc[0])
    combined = pd.DataFrame(summaries)
    combined["combined_dataset_mean_acc"] = combined["global_mean_acc"].mean()
    combined.to_csv(seed_root / "seed_summary.csv", index=False)
    print("\nSEED COMPLETE", flush=True)
    print(combined.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
