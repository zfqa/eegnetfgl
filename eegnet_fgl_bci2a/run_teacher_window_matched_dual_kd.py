from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

from run_teacher_window_matched_kd import (
    _paired_test,
    common_args,
    run,
    train_teacher,
)


METHODS = {
    "ce": "Matched CE 2s",
    "teacher_2s": "Matched dual KD 2s-to-2s",
    "teacher_4s": "Matched dual KD 4s-to-2s",
}


def method_names(reset_student_seed_before_fit: bool) -> dict[str, str]:
    if reset_student_seed_before_fit:
        return METHODS
    return {
        "ce": "Legacy-control CE 2s",
        "teacher_2s": "Legacy dual KD 2s-to-2s",
        "teacher_4s": "Legacy dual KD 4s-to-2s",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare matched 2s EEGNet+Riemann teachers and 4s EEGNet+Riemann "
            "teachers for a fixed 2s Compact EEGNet student."
        )
    )
    parser.add_argument("--dataset", choices=["bci2a", "bci2b"], default="bci2a")
    parser.add_argument("--subjects", nargs="+", type=int, default=list(range(1, 10)))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 2024, 3407, 7, 123])
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=["ce", "teacher_2s", "teacher_4s"],
        default=["ce", "teacher_2s", "teacher_4s"],
        help="Experiment arms to run. Use --arms teacher_2s for the legacy 2s-teacher add-on only.",
    )
    parser.add_argument(
        "--preprocessing_mode",
        choices=["causal_mat", "moabb_zero_phase"],
        default="moabb_zero_phase",
    )
    parser.add_argument("--causal_filter_order", type=int, default=4)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--epochs_teacher", type=int, default=300)
    parser.add_argument("--epochs_student", type=int, default=300)
    parser.add_argument("--patience_teacher", type=int, default=100)
    parser.add_argument("--patience_student", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--alpha_eegnet", type=float, default=0.4)
    parser.add_argument("--alpha_riemann", type=float, default=0.4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--teacher_selection_metric",
        choices=["nll", "accuracy"],
        default="accuracy",
    )
    student_rng_group = parser.add_mutually_exclusive_group()
    student_rng_group.add_argument(
        "--reset_student_seed_before_fit",
        dest="reset_student_seed_before_fit",
        action="store_true",
        help="Reset RNG after preparing teacher logits so all student arms share matched initialization.",
    )
    student_rng_group.add_argument(
        "--legacy_continuous_student_rng",
        dest="reset_student_seed_before_fit",
        action="store_false",
        help=(
            "Reproduce the historical paper-matrix behavior: do not reset RNG after "
            "teacher-logit preparation. Use only for a clearly labelled legacy reproduction."
        ),
    )
    parser.set_defaults(reset_student_seed_before_fit=True)
    parser.add_argument(
        "--pretrained_teacher_root",
        type=str,
        default="",
        help="Optional root containing tw2(.0)/seed_X and tw4(.0)/seed_X EEGNet teachers.",
    )
    parser.add_argument("--riemann_paper_method", default="ts_logreg")
    parser.add_argument("--riemann_cov_estimator", default="oas")
    parser.add_argument("--riemann_alignment", default="none")
    parser.add_argument("--riemann_metric", default="riemann")
    parser.add_argument("--riemann_selection", choices=["validation", "inner_cv"], default="inner_cv")
    parser.add_argument("--riemann_selection_folds", type=int, default=5)
    parser.add_argument("--riemann_c_grid", default="0.01,0.03,0.1,0.3,1,3,10")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--auto_select_gpu", action="store_true")
    parser.add_argument("--gpu_candidates", nargs="+", type=int, default=None)
    parser.add_argument("--gpu_min_free_mb", type=int, default=2048)
    parser.add_argument("--gpu_max_utilization", type=int, default=30)
    parser.add_argument("--gpu_poll_seconds", type=float, default=60.0)
    parser.add_argument("--gpu_wait_timeout_minutes", type=float, default=0.0)
    parser.add_argument(
        "--output_dir",
        default="results/teacher_window_matched_dual_kd_offline_bci2a",
    )
    parser.add_argument("--rerun_existing", action="store_true")
    return parser.parse_args()


def build_student_command(
    args: argparse.Namespace,
    subject: int,
    seed: int,
    arm: str,
    teacher_path: Path | None,
    teacher_window: float,
    save_dir: Path,
) -> list[str]:
    use_teachers = arm != "ce"
    names = method_names(args.reset_student_seed_before_fit)
    command = [
        sys.executable,
        "train_student_dual_kd.py",
        *common_args(args, subject, seed),
        "--student_window", "2.0",
        "--teacher_window", str(teacher_window),
        "--teacher_mode", "dual" if use_teachers else "none",
        "--method_name", names[arm],
        "--alpha1", str(args.alpha_eegnet if use_teachers else 0.0),
        "--alpha2", str(args.alpha_riemann if use_teachers else 0.0),
        "--temperature", str(args.temperature),
        "--teacher_calibration", "temperature" if use_teachers else "none",
        "--loss_type", "raw_dual_kd",
        "--diffusion_ratio", "0",
        "--student_model_variant", "compact",
        "--riemann_teacher_style", "paper",
        "--riemann_paper_method", args.riemann_paper_method,
        "--riemann_cov_estimator", args.riemann_cov_estimator,
        "--riemann_alignment", args.riemann_alignment,
        "--riemann_metric", args.riemann_metric,
        "--riemann_selection", args.riemann_selection,
        "--riemann_selection_folds", str(args.riemann_selection_folds),
        "--riemann_c_grid", args.riemann_c_grid,
        "--epochs", str(args.epochs_student),
        "--patience", str(args.patience_student),
        "--save_dir", str(save_dir),
    ]
    if args.reset_student_seed_before_fit:
        command.append("--reset_student_seed_before_fit")
    if teacher_path is not None:
        command.extend(
            [
                "--eegnet_teacher_path", str(teacher_path),
                "--require_teacher_preprocess_metadata",
            ]
        )
    if args.rerun_existing:
        command.append("--rerun_existing")
    return command


def summarize(
    output_dir: Path,
    dataset: str,
    expected_subjects: int,
    reset_student_seed_before_fit: bool,
) -> None:
    names = method_names(reset_student_seed_before_fit)
    paths = sorted(output_dir.glob("students/*/seed_*/dual_kd_results.csv"))
    if not paths:
        return
    rows = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    rows = rows[
        rows["method"].isin(names.values())
        & (rows["dataset"].astype(str) == dataset)
    ].copy()
    rows = rows.drop_duplicates(["dataset", "subject", "seed", "method"], keep="last")
    rows = rows.sort_values(["seed", "subject", "method"])
    rows.to_csv(output_dir / "matched_dual_subject_seed_results.csv", index=False)

    seed_summary = (
        rows.groupby(["dataset", "seed", "method"], as_index=False)
        .agg(
            n_subjects=("subject", "nunique"),
            mean_accuracy=("accuracy", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            mean_kappa=("kappa", "mean"),
        )
    )
    seed_summary.to_csv(output_dir / "matched_dual_seed_summary.csv", index=False)
    complete = seed_summary[seed_summary["n_subjects"] == expected_subjects]
    summary = (
        complete.groupby(["dataset", "method"], as_index=False)
        .agg(
            n_complete_seeds=("seed", "nunique"),
            mean_accuracy=("mean_accuracy", "mean"),
            std_accuracy=("mean_accuracy", "std"),
            mean_macro_f1=("mean_macro_f1", "mean"),
            std_macro_f1=("mean_macro_f1", "std"),
            mean_kappa=("mean_kappa", "mean"),
            std_kappa=("mean_kappa", "std"),
        )
    )
    summary.to_csv(output_dir / "matched_dual_summary.csv", index=False)

    subject_means = (
        rows.groupby(["dataset", "subject", "method"], as_index=False)
        .agg(accuracy=("accuracy", "mean"))
    )
    available = set(subject_means["method"])
    comparisons = [
        (names["ce"], names["teacher_2s"]),
        (names["ce"], names["teacher_4s"]),
        (names["teacher_2s"], names["teacher_4s"]),
    ]
    tests = pd.DataFrame(
        [_paired_test(subject_means, left, right) for left, right in comparisons if {left, right} <= available]
    )
    if not tests.empty:
        tests.insert(0, "dataset", dataset)
    tests.to_csv(output_dir / "matched_dual_paired_tests.csv", index=False)
    print("\nMATCHED DUAL-TEACHER SUMMARY", flush=True)
    print(summary.to_string(index=False), flush=True)
    print("\nSUBJECT-LEVEL PAIRED TESTS", flush=True)
    print(tests.to_string(index=False) if not tests.empty else "Not enough arms for paired tests.", flush=True)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pretrained_teacher_root = (
        Path(args.pretrained_teacher_root).resolve()
        if args.pretrained_teacher_root
        else None
    )
    config = {
        "dataset": args.dataset,
        "subjects": sorted(set(args.subjects)),
        "seeds": sorted(set(args.seeds)),
        "arms": args.arms,
        "preprocessing_mode": args.preprocessing_mode,
        "causal_filter_order": args.causal_filter_order,
        "data_dir": str(Path(args.data_dir).resolve()),
        "epochs_teacher": args.epochs_teacher,
        "epochs_student": args.epochs_student,
        "patience_teacher": args.patience_teacher,
        "patience_student": args.patience_student,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "alpha_eegnet": args.alpha_eegnet,
        "alpha_riemann": args.alpha_riemann,
        "temperature": args.temperature,
        "teacher_selection_metric": args.teacher_selection_metric,
        "reset_student_seed_before_fit": args.reset_student_seed_before_fit,
        "pretrained_teacher_root": str(pretrained_teacher_root or ""),
        "riemann_paper_method": args.riemann_paper_method,
        "riemann_cov_estimator": args.riemann_cov_estimator,
        "riemann_alignment": args.riemann_alignment,
        "riemann_metric": args.riemann_metric,
        "riemann_selection": args.riemann_selection,
        "riemann_selection_folds": args.riemann_selection_folds,
        "riemann_c_grid": args.riemann_c_grid,
    }
    config_path = output_dir / "run_config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != config:
            raise ValueError(
                f"Output directory contains a different configuration: {config_path}."
            )
    else:
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    for seed in args.seeds:
        teacher_paths: dict[tuple[float, int], Path] = {}
        teacher_windows = sorted(
            {2.0 if arm == "teacher_2s" else 4.0 for arm in args.arms if arm != "ce"}
        )
        for window in teacher_windows:
            if pretrained_teacher_root is None:
                teacher_dir = output_dir / "teachers" / f"tw{window:g}" / f"seed_{seed}"
                teacher_dir.mkdir(parents=True, exist_ok=True)
            else:
                candidates = [
                    pretrained_teacher_root / f"tw{window:.1f}" / f"seed_{seed}",
                    pretrained_teacher_root / f"tw{window:g}" / f"seed_{seed}",
                ]
                teacher_dir = next((path for path in candidates if path.exists()), candidates[0])
            for subject in args.subjects:
                teacher_paths[(window, subject)] = train_teacher(
                    args,
                    subject,
                    seed,
                    window,
                    teacher_dir,
                    allow_training=pretrained_teacher_root is None,
                )

        teacher_window_by_arm = {"ce": 2.0, "teacher_2s": 2.0, "teacher_4s": 4.0}
        for arm in args.arms:
            teacher_window = teacher_window_by_arm[arm]
            student_dir = output_dir / "students" / arm / f"seed_{seed}"
            student_dir.mkdir(parents=True, exist_ok=True)
            for subject in args.subjects:
                teacher_path = None if arm == "ce" else teacher_paths[(teacher_window, subject)]
                command = build_student_command(
                    args,
                    subject,
                    seed,
                    arm,
                    teacher_path,
                    teacher_window,
                    student_dir,
                )
                run(command, args)
        summarize(
            output_dir,
            args.dataset,
            len(set(args.subjects)),
            args.reset_student_seed_before_fit,
        )


if __name__ == "__main__":
    main()
