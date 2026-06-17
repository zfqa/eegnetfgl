from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd


DEFAULT_SEEDS = [42, 2024, 3407, 7, 123]
DEFAULT_SUBJECTS = list(range(1, 10))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume only the BCI IV-2b experiments affected by the compact-teacher change."
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=["weight_matched", "fixed", "adaptive", "routing", "all"],
        default=["all"],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--subjects", type=int, nargs="+", default=DEFAULT_SUBJECTS)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--teacher_root", default="results/unified_teachers")
    parser.add_argument("--weight_root", default="results/weight_matched_kd_rebuild")
    parser.add_argument("--fixed_root", default="results/bci2b_paper_multiseed_unified")
    parser.add_argument(
        "--candidate_root",
        default="results/global_routing_multiseed",
        help="Completed detector/CE candidate root. These models are teacher-independent.",
    )
    parser.add_argument("--routing_root", default="results/global_routing_multiseed_unified")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def selected_phases(raw: list[str]) -> set[str]:
    if "all" in raw:
        return {"weight_matched", "fixed", "adaptive", "routing"}
    return set(raw)


def set_data_environment() -> None:
    os.environ.setdefault("MNE_DATA", "data/mne_data")
    os.environ.setdefault("MNE_DATASETS_BNCI_PATH", "data/mne_data")
    os.environ.setdefault("MNE_DONTWRITE_HOME", "true")


def run(command: list[str], dry_run: bool) -> None:
    print("\n" + subprocess.list2cmdline(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def float_equal(actual: object, expected: float) -> bool:
    try:
        return abs(float(actual) - expected) < 1e-8
    except (TypeError, ValueError):
        return False


def csv_has_config(
    path: Path,
    subject: int,
    seed: int,
    eegnet_weight: float,
    riemann_weight: float,
) -> bool:
    if not path.exists():
        return False
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                int(row.get("subject", -1)) == subject
                and int(row.get("seed", -1)) == seed
                and row.get("dataset") == "bci2b"
                and row.get("model_variant") == "compact"
                and float_equal(row.get("window"), 2.0)
                and float_equal(row.get("eegnet_kd_weight"), eegnet_weight)
                and float_equal(row.get("kl_weight"), riemann_weight)
                and row.get("teacher_calibration") == "temperature"
                and row.get("confidence_mode") == "none"
                and row.get("riemann_cov_estimator") == "oas"
                and row.get("riemann_selection") == "inner_cv"
                and int(row.get("riemann_selection_folds", 0)) == 5
            ):
                return True
    return False


def csv_has_adaptive(path: Path, subject: int, seed: int) -> bool:
    if not path.exists():
        return False
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                int(row.get("subject", -1)) == subject
                and int(row.get("seed", -1)) == seed
                and row.get("dataset") == "bci2b"
                and row.get("teacher_calibration") == "temperature"
                and row.get("confidence_mode") == "none"
                and row.get("riemann_input_preprocessing") == "raw"
                and row.get("riemann_selection") == "inner_cv"
                and int(row.get("riemann_selection_folds", 0)) == 5
                and float_equal(row.get("alpha_eegnet"), 0.4)
                and float_equal(row.get("alpha_riemann"), 0.4)
            ):
                return True
    return False


def teacher_path(root: Path, seed: int, subject: int) -> Path:
    path = root / "bci2b" / f"seed_{seed}" / f"bci2b_subject_{subject}_teacher_best.pth"
    if not path.exists():
        raise FileNotFoundError(f"Validated compact teacher not found: {path}")
    return path


def common_guided_command(
    args: argparse.Namespace,
    seed: int,
    subject: int,
    save_dir: Path,
    eegnet_weight: float,
    riemann_weight: float,
) -> list[str]:
    command = [
        sys.executable,
        "train_student_riemann_guided.py",
        "--dataset",
        "bci2b",
        "--subject",
        str(subject),
        "--window",
        "2.0",
        "--riemann_window",
        "4.0",
        "--fmin",
        "8",
        "--fmax",
        "32",
        "--tmin",
        "0",
        "--tmax",
        "4",
        "--model_variant",
        "compact",
        "--eegnet_kd_weight",
        str(eegnet_weight),
        "--kl_weight",
        str(riemann_weight),
        "--teacher_weight_mode",
        "fixed",
        "--temperature",
        "1.0",
        "--teacher_calibration",
        "temperature",
        "--geometry_weight",
        "0",
        "--spatial_structure_weight",
        "0",
        "--confidence_mode",
        "none",
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
        "--trust_gate_mode",
        "confidence",
        "--optimizer",
        "adamw",
        "--seed",
        str(seed),
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--device",
        args.device,
        "--save_dir",
        str(save_dir),
        "--patience",
        str(args.patience),
    ]
    if eegnet_weight > 0:
        command.extend(
            [
                "--eegnet_teacher_path",
                str(teacher_path(Path(args.teacher_root), seed, subject)),
            ]
        )
    return command


def run_weight_matched(args: argparse.Namespace) -> None:
    run(
        [
            sys.executable,
            "run_weight_matched_kd_ablation.py",
            "--datasets",
            "bci2b",
            "--subjects",
            *[str(subject) for subject in args.subjects],
            "--seeds",
            *[str(seed) for seed in args.seeds],
            "--configs",
            "eegnet_08",
            "riemann_08",
            "dual_02",
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--weight_decay",
            str(args.weight_decay),
            "--patience",
            str(args.patience),
            "--device",
            args.device,
            "--teacher_root",
            args.teacher_root,
            "--root_dir",
            args.weight_root,
        ],
        args.dry_run,
    )


def run_fixed(args: argparse.Namespace) -> None:
    root = Path(args.fixed_root)
    for seed in args.seeds:
        save_dir = root / f"seed_{seed}" / "deep_models"
        save_dir.mkdir(parents=True, exist_ok=True)
        csv_path = save_dir / "riemann_guided_results.csv"
        for subject in args.subjects:
            for eegnet_weight, riemann_weight, label in [
                (0.0, 0.0, "fixed CE"),
                (0.4, 0.4, "fixed dual"),
            ]:
                if csv_has_config(csv_path, subject, seed, eegnet_weight, riemann_weight):
                    print(f"Skip existing BCI2b S{subject} seed={seed}: {label}", flush=True)
                    continue
                run(
                    common_guided_command(
                        args,
                        seed,
                        subject,
                        save_dir,
                        eegnet_weight,
                        riemann_weight,
                    ),
                    args.dry_run,
                )


def verify_candidate(root: Path, seed: int, subject: int) -> tuple[Path, Path]:
    subject_dir = root / f"seed_{seed}" / "bci2b_candidates" / f"subject_{subject}"
    detector = subject_dir / "eegnetv4_05s_sliding_detector_best.pth"
    adaptive_ce = subject_dir / "final_stable_peak_confidence_best.pth"
    for path in [detector, adaptive_ce]:
        if not path.exists():
            raise FileNotFoundError(f"Reusable routing candidate not found: {path}")
    return detector, adaptive_ce


def run_adaptive(args: argparse.Namespace) -> None:
    candidate_root = Path(args.candidate_root)
    routing_root = Path(args.routing_root)
    for seed in args.seeds:
        save_dir = routing_root / f"seed_{seed}" / "bci2b_adaptive_dual"
        save_dir.mkdir(parents=True, exist_ok=True)
        csv_path = save_dir / "detector_dual_teacher_results.csv"
        for subject in args.subjects:
            if csv_has_adaptive(csv_path, subject, seed):
                print(f"Skip existing adaptive dual BCI2b S{subject} seed={seed}", flush=True)
                continue
            detector, _ = verify_candidate(candidate_root, seed, subject)
            run(
                [
                    sys.executable,
                    "run_bci2b_detector_dual_teacher_subject1.py",
                    "--subject",
                    str(subject),
                    "--seed",
                    str(seed),
                    "--detector_path",
                    str(detector),
                    "--eegnet_teacher_path",
                    str(teacher_path(Path(args.teacher_root), seed, subject)),
                    "--detector_win",
                    "1.0",
                    "--detector_step",
                    "0.25",
                    "--class_window",
                    "2.0",
                    "--smooth_k",
                    "3",
                    "--alpha_eegnet",
                    "0.4",
                    "--alpha_riemann",
                    "0.4",
                    "--temperature",
                    "1.0",
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
                    str(save_dir),
                ],
                args.dry_run,
            )


def run_routing(args: argparse.Namespace) -> None:
    routing_root = Path(args.routing_root)
    candidate_root = Path(args.candidate_root)
    for seed in args.seeds:
        for subject in args.subjects:
            verify_candidate(candidate_root, seed, subject)
        run(
            [
                sys.executable,
                "evaluate_global_routing_seed.py",
                "--dataset",
                "bci2b",
                "--seed",
                str(seed),
                "--subjects",
                *[str(subject) for subject in args.subjects],
                "--candidate_root",
                str(candidate_root / f"seed_{seed}" / "bci2b_candidates"),
                "--adaptive_dual_root",
                str(routing_root / f"seed_{seed}" / "bci2b_adaptive_dual"),
                "--paper2b_root",
                args.fixed_root,
                "--device",
                args.device,
                "--output_dir",
                str(routing_root / f"seed_{seed}" / "bci2b_routing"),
            ],
            args.dry_run,
        )
    if args.dry_run:
        return

    subject_frames = []
    summary_frames = []
    for seed in args.seeds:
        seed_root = routing_root / f"seed_{seed}" / "bci2b_routing"
        subject_frames.append(pd.read_csv(seed_root / "subject_results.csv"))
        summary_frames.append(pd.read_csv(seed_root / "summary.csv"))
    subjects = pd.concat(subject_frames, ignore_index=True)
    summaries = pd.concat(summary_frames, ignore_index=True)
    subjects.to_csv(routing_root / "bci2b_subject_seed_results.csv", index=False)
    summaries.to_csv(routing_root / "bci2b_seed_summary.csv", index=False)
    method_columns = [
        "fixed_ce_test_acc",
        "adaptive_ce_test_acc",
        "fixed_dual_test_acc",
        "adaptive_dual_test_acc",
        "global_test_acc",
    ]
    aggregate = pd.DataFrame(
        [
            {
                "dataset": "bci2b",
                "n_subject_seed": len(subjects),
                **{f"{column}_mean": float(subjects[column].mean()) for column in method_columns},
                **{f"{column}_std": float(subjects[column].std(ddof=1)) for column in method_columns},
            }
        ]
    )
    aggregate.to_csv(routing_root / "bci2b_aggregate_summary.csv", index=False)
    print("\nBCI2b routing aggregate", flush=True)
    print(aggregate.to_string(index=False), flush=True)


def main() -> None:
    args = parse_args()
    set_data_environment()
    phases = selected_phases(args.phases)
    if "weight_matched" in phases:
        run_weight_matched(args)
    if "fixed" in phases:
        run_fixed(args)
    if "adaptive" in phases:
        run_adaptive(args)
    if "routing" in phases:
        run_routing(args)


if __name__ == "__main__":
    main()
