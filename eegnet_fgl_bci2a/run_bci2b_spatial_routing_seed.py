from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BCI2b routing where the previous dual-teacher branch is replaced by Riemann KD + spatial loss."
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 10)))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--epochs_detector", type=int, default=300)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--patience_detector", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--output_root", type=str, default="results/bci2b_spatial_routing")
    parser.add_argument(
        "--candidate_source_root",
        type=str,
        default="",
        help="Optional completed BCI2b routing CE/detector root to reuse.",
    )
    parser.add_argument("--rerun_existing", action="store_true")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("\n" + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def csv_has(path: Path, subject: int, seed: int, **conditions: object) -> bool:
    if not path.exists():
        return False
    rows = pd.read_csv(path)
    mask = (rows["subject"].astype(int) == subject) & (rows["seed"].astype(int) == seed)
    for column, expected in conditions.items():
        if column not in rows.columns:
            return False
        if isinstance(expected, float):
            values = pd.to_numeric(rows[column], errors="coerce")
            mask &= np.isclose(values, expected)
        else:
            mask &= rows[column].astype(str) == str(expected)
    return bool(mask.any())


def train_fixed_riemann_guided(
    python: str,
    args: argparse.Namespace,
    subject: int,
    save_dir: Path,
    kl_weight: float,
    spatial_weight: float,
) -> None:
    command = [
        python,
        "train_student_riemann_guided.py",
        "--dataset",
        "bci2b",
        "--subject",
        str(subject),
        "--window",
        "2.0",
        "--riemann_window",
        "4.0",
        "--model_variant",
        "compact",
        "--fmin",
        "8",
        "--fmax",
        "32",
        "--tmin",
        "0",
        "--tmax",
        "4",
        "--eegnet_kd_weight",
        "0.0",
        "--kl_weight",
        f"{kl_weight:g}",
        "--spatial_structure_weight",
        f"{spatial_weight:g}",
        "--spatial_structure_source",
        "pre_depthwise",
        "--spatial_structure_mode",
        "corr",
        "--spatial_structure_trust_mode",
        "sample",
        "--riemann_cov_estimator",
        "oas",
        "--riemann_alignment",
        "none",
        "--riemann_metric",
        "riemann",
        "--riemann_method",
        "ts_logreg",
        "--riemann_selection",
        "inner_cv",
        "--riemann_selection_folds",
        "5",
        "--c_grid",
        "0.01,0.03,0.1,0.3,1,3,10",
        "--teacher_calibration",
        "temperature",
        "--confidence_mode",
        "none",
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--batch_size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--save_dir",
        str(save_dir),
    ]
    run(command)


def main() -> None:
    args = parse_args()
    python = sys.executable
    seed_root = Path(args.output_root) / f"seed_{args.seed}"
    candidate_root = (
        Path(args.candidate_source_root) / f"seed_{args.seed}" / "bci2b_candidates"
        if args.candidate_source_root
        else seed_root / "bci2b_candidates"
    )
    fixed_ce_root = seed_root / "bci2b_fixed_ce"
    fixed_spatial_root = seed_root / "bci2b_fixed_riemann_spatial"
    adaptive_spatial_root = seed_root / "bci2b_adaptive_riemann_spatial"
    for path in [candidate_root, fixed_ce_root, fixed_spatial_root, adaptive_spatial_root]:
        path.mkdir(parents=True, exist_ok=True)

    for subject in args.subjects:
        ce_csv = candidate_root / "routing_ce_results.csv"
        if args.rerun_existing or not csv_has(ce_csv, subject, args.seed):
            if args.candidate_source_root:
                raise FileNotFoundError(
                    f"Reusable BCI2b routing candidate is missing for S{subject}, "
                    f"seed={args.seed}: {ce_csv}"
                )
            run(
                [
                    python,
                    "run_bci2b_routing_candidates.py",
                    "--subject",
                    str(subject),
                    "--seed",
                    str(args.seed),
                    "--epochs_detector",
                    str(args.epochs_detector),
                    "--epochs_classifier",
                    str(args.epochs),
                    "--patience_detector",
                    str(args.patience_detector),
                    "--patience_classifier",
                    str(args.patience),
                    "--batch_size",
                    str(args.batch_size),
                    "--lr",
                    str(args.lr),
                    "--weight_decay",
                    str(args.weight_decay),
                    "--device",
                    args.device,
                    "--save_dir",
                    str(candidate_root),
                    "--results_csv",
                    str(ce_csv),
                ]
            )

        fixed_ce_csv = fixed_ce_root / "riemann_guided_results.csv"
        if args.rerun_existing or not csv_has(
            fixed_ce_csv,
            subject,
            args.seed,
            dataset="bci2b",
            model_variant="compact",
            window=2.0,
            riemann_window=4.0,
            eegnet_kd_weight=0.0,
            kl_weight=0.0,
            spatial_structure_weight=0.0,
            teacher_calibration="temperature",
            confidence_mode="none",
            riemann_selection="inner_cv",
            riemann_selection_folds=5,
        ):
            train_fixed_riemann_guided(
                python, args, subject, fixed_ce_root, kl_weight=0.0, spatial_weight=0.0
            )

        fixed_spatial_csv = fixed_spatial_root / "riemann_guided_results.csv"
        if args.rerun_existing or not csv_has(
            fixed_spatial_csv,
            subject,
            args.seed,
            dataset="bci2b",
            model_variant="compact",
            window=2.0,
            riemann_window=4.0,
            eegnet_kd_weight=0.0,
            kl_weight=0.1,
            spatial_structure_weight=0.2,
            spatial_structure_source="pre_depthwise",
            spatial_structure_mode="corr",
            spatial_structure_trust_mode="sample",
            teacher_calibration="temperature",
            confidence_mode="none",
            riemann_selection="inner_cv",
            riemann_selection_folds=5,
        ):
            train_fixed_riemann_guided(
                python, args, subject, fixed_spatial_root, kl_weight=0.1, spatial_weight=0.2
            )

        adaptive_csv = adaptive_spatial_root / "detector_dual_teacher_results.csv"
        if args.rerun_existing or not csv_has(
            adaptive_csv,
            subject,
            args.seed,
            teacher_calibration="temperature",
            confidence_mode="none",
            alpha_eegnet=0.0,
            alpha_riemann=0.1,
            spatial_structure_weight=0.2,
            spatial_structure_source="pre_depthwise",
            spatial_structure_mode="corr",
            spatial_structure_trust_mode="sample",
            riemann_selection="inner_cv",
            riemann_selection_folds=5,
        ):
            run(
                [
                    python,
                    "run_bci2b_detector_dual_teacher_subject1.py",
                    "--subject",
                    str(subject),
                    "--seed",
                    str(args.seed),
                    "--detector_path",
                    str(candidate_root / f"subject_{subject}" / "eegnetv4_05s_sliding_detector_best.pth"),
                    "--alpha_eegnet",
                    "0.0",
                    "--alpha_riemann",
                    "0.1",
                    "--spatial_structure_weight",
                    "0.2",
                    "--spatial_structure_source",
                    "pre_depthwise",
                    "--spatial_structure_mode",
                    "corr",
                    "--spatial_structure_trust_mode",
                    "sample",
                    "--detector_win",
                    "1.0",
                    "--detector_step",
                    "0.25",
                    "--class_window",
                    "2.0",
                    "--smooth_k",
                    "3",
                    "--teacher_calibration",
                    "temperature",
                    "--confidence_mode",
                    "none",
                    "--riemann_selection",
                    "inner_cv",
                    "--riemann_selection_folds",
                    "5",
                    "--epochs",
                    str(args.epochs),
                    "--patience",
                    str(args.patience),
                    "--batch_size",
                    str(args.batch_size),
                    "--lr",
                    str(args.lr),
                    "--weight_decay",
                    str(args.weight_decay),
                    "--device",
                    args.device,
                    "--save_dir",
                    str(adaptive_spatial_root),
                ]
            )

    output_dir = seed_root / "bci2b_spatial_routing"
    run(
        [
            python,
            "evaluate_bci2b_spatial_routing_seed.py",
            "--seed",
            str(args.seed),
            "--subjects",
            *[str(subject) for subject in args.subjects],
            "--candidate_root",
            str(candidate_root),
            "--fixed_ce_root",
            str(fixed_ce_root),
            "--fixed_spatial_root",
            str(fixed_spatial_root),
            "--adaptive_spatial_root",
            str(adaptive_spatial_root),
            "--device",
            args.device,
            "--batch_size",
            str(args.batch_size),
            "--output_dir",
            str(output_dir),
        ]
    )
    summary = pd.read_csv(output_dir / "summary.csv")
    print("\nBCI2b SPATIAL ROUTING SEED COMPLETE", flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
