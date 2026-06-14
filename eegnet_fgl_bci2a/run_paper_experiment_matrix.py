from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


SEEDS = [42, 2024, 3407, 7, 123]
SUBJECTS = list(range(1, 10))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the paper-grade MOABB experiment matrix.")
    parser.add_argument("--subjects", type=int, nargs="+", default=SUBJECTS)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument(
        "--groups",
        type=str,
        nargs="+",
        default=["teachers", "ce", "dual_ablation", "riemann_guided", "baselines"],
        choices=["teachers", "ce", "eegnetv4", "dual_ablation", "riemann_guided", "baselines", "all"],
    )
    parser.add_argument("--root_dir", type=str, default="results/paper_moabb_multiseed_unified")
    parser.add_argument("--teacher_root", type=str, default="results/unified_teachers/bci2a")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--teacher_epochs", type=int, default=300)
    parser.add_argument("--eegnetv4_epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--rerun_existing", action="store_true")
    return parser.parse_args()


def expand_groups(groups: list[str]) -> set[str]:
    if "all" in groups:
        return {"teachers", "ce", "eegnetv4", "dual_ablation", "riemann_guided", "baselines"}
    return set(groups)


def run_command(command: list[str], dry_run: bool) -> None:
    print("\n" + " ".join(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, check=True)


def _csv_value_matches(actual: str, expected: object) -> bool:
    if isinstance(expected, float):
        try:
            return abs(float(actual) - expected) < 1e-6
        except (TypeError, ValueError):
            return False
    return str(actual) == str(expected)


def csv_has_result(csv_path: Path, **conditions: object) -> bool:
    if not csv_path.exists():
        return False
    try:
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if all(_csv_value_matches(row.get(key, ""), value) for key, value in conditions.items()):
                    return True
    except Exception as exc:
        print(f"Could not inspect existing result CSV {csv_path}: {exc}", flush=True)
    return False


def base_env() -> None:
    os.environ.setdefault("MNE_DATA", "data/mne_data")
    os.environ.setdefault("MNE_DATASETS_BNCI_PATH", "data/mne_data")
    os.environ.setdefault("MNE_DONTWRITE_HOME", "true")


def seed_dir(root: Path, group: str, seed: int) -> Path:
    path = root / group / f"seed_{seed}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def teacher_checkpoint(teacher_root: Path, seed: int, subject: int) -> Path:
    return teacher_root / f"seed_{seed}" / f"subject_{subject}_teacher_best.pth"


def common_train_args(args: argparse.Namespace, seed: int, save_dir: Path) -> list[str]:
    return [
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


def run_teachers(args: argparse.Namespace, teacher_root: Path) -> None:
    for seed in args.seeds:
        out_dir = teacher_root / f"seed_{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for subject in args.subjects:
            command = [
                sys.executable,
                "train_teacher.py",
                "--subject",
                str(subject),
                "--seed",
                str(seed),
                "--epochs",
                str(args.teacher_epochs),
                "--batch_size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
                "--weight_decay",
                str(args.weight_decay),
                "--device",
                args.device,
                "--save_dir",
                str(out_dir),
                "--patience",
                str(args.patience),
                "--protocol",
                "moabb_standard",
                "--model_variant",
                "compact",
                "--selection_metric",
                "nll",
            ]
            run_command(command, args.dry_run)


def run_ce(args: argparse.Namespace, root: Path) -> None:
    for seed in args.seeds:
        out_dir = seed_dir(root, "ce_compact", seed)
        for subject in args.subjects:
            for window in [2.0, 4.0]:
                command = [
                    sys.executable,
                    "train_student_ce.py",
                    "--subject",
                    str(subject),
                    "--window",
                    str(window),
                    "--protocol",
                    "moabb_standard",
                    *common_train_args(args, seed, out_dir),
                ]
                run_command(command, args.dry_run)


def run_eegnetv4(args: argparse.Namespace, root: Path) -> None:
    for seed in args.seeds:
        out_dir = seed_dir(root, "eegnetv4", seed)
        result_csv = out_dir / "eegnet_official_results.csv"
        for subject in args.subjects:
            for window in [2.0, 4.0]:
                if not args.rerun_existing and csv_has_result(
                    result_csv,
                    subject=subject,
                    seed=seed,
                    window=window,
                    model_variant="eegnetv4",
                ):
                    print(f"Skip existing EEGNetV4 row: subject={subject}, window={window:g}, seed={seed}", flush=True)
                    continue
                command = [
                    sys.executable,
                    "train_eegnet_official.py",
                    "--subject",
                    str(subject),
                    "--window",
                    str(window),
                    "--seed",
                    str(seed),
                    "--epochs",
                    str(args.eegnetv4_epochs),
                    "--batch_size",
                    str(args.batch_size),
                    "--lr",
                    str(args.lr),
                    "--weight_decay",
                    "0.0001",
                    "--device",
                    args.device,
                    "--save_dir",
                    str(out_dir),
                    "--patience",
                    str(args.patience),
                    "--protocol",
                    "moabb_standard",
                    "--model_variant",
                    "eegnetv4",
                    "--kernel_length",
                    "125",
                    "--optimizer",
                    "adamw",
                    "--final_train",
                    "train",
                ]
                if args.rerun_existing:
                    command.append("--rerun_existing")
                run_command(command, args.dry_run)


def dual_command(
    args: argparse.Namespace,
    teacher_root: Path,
    seed: int,
    subject: int,
    out_dir: Path,
    name: str,
    teacher_mode: str,
    alpha1: float,
    alpha2: float,
) -> list[str]:
    command = [
        sys.executable,
        "train_student_dual_kd.py",
        "--subject",
        str(subject),
        "--student_window",
        "2.0",
        "--teacher_window",
        "4.0",
        "--protocol",
        "moabb_standard",
        "--teacher_mode",
        teacher_mode,
        "--method_name",
        name,
        "--alpha1",
        str(alpha1),
        "--alpha2",
        str(alpha2),
        "--temperature",
        "1.0",
        "--teacher_calibration",
        "temperature",
        "--loss_type",
        "raw_dual_kd",
        "--diffusion_ratio",
        "0",
        "--student_model_variant",
        "compact",
        "--riemann_teacher_style",
        "paper",
        "--riemann_paper_method",
        "ts_logreg",
        "--riemann_cov_estimator",
        "oas",
        "--riemann_alignment",
        "none",
        "--riemann_metric",
        "riemann",
        "--riemann_selection",
        "inner_cv",
        "--riemann_selection_folds",
        "5",
        "--riemann_c_grid",
        "0.01,0.03,0.1,0.3,1,3,10",
        *common_train_args(args, seed, out_dir),
    ]
    if teacher_mode in {"eegnet", "dual"}:
        checkpoint = teacher_checkpoint(teacher_root, seed, subject)
        if not args.dry_run and not checkpoint.exists():
            raise FileNotFoundError(
                f"Validated BCI2a teacher not found: {checkpoint}. "
                "Run this matrix with --groups teachers first."
            )
        command.extend(["--eegnet_teacher_path", str(checkpoint)])
    return command


def run_dual_ablation(args: argparse.Namespace, root: Path, teacher_root: Path) -> None:
    experiments = [
        ("EEGNet teacher only", "eegnet", 0.4, 0.0),
        ("Riemann teacher only", "riemann", 0.0, 0.4),
        ("Riemann-guided dual KD", "dual", 0.4, 0.4),
    ]
    for seed in args.seeds:
        out_dir = seed_dir(root, "dual_ablation", seed)
        for subject in args.subjects:
            for name, teacher_mode, alpha1, alpha2 in experiments:
                command = dual_command(
                    args,
                    teacher_root,
                    seed,
                    subject,
                    out_dir,
                    name,
                    teacher_mode,
                    alpha1,
                    alpha2,
                )
                run_command(command, args.dry_run)


def run_riemann_guided(args: argparse.Namespace, root: Path) -> None:
    experiments = [
        ("Riemann spatial only", 0.0, 0.0, 0.1),
        ("Riemann KD + spatial", 0.2, 0.0, 0.1),
        ("Riemann KD + geometry", 0.2, 0.05, 0.0),
        ("Riemann KD + geometry + spatial", 0.2, 0.05, 0.1),
    ]
    for seed in args.seeds:
        out_dir = seed_dir(root, "riemann_guided", seed)
        for subject in args.subjects:
            for name, kl_weight, geometry_weight, spatial_weight in experiments:
                command = [
                    sys.executable,
                    "train_student_riemann_guided.py",
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
                    "--fs",
                    "250",
                    "--model_variant",
                    "compact",
                    "--kl_weight",
                    str(kl_weight),
                    "--geometry_weight",
                    str(geometry_weight),
                    "--spatial_structure_weight",
                    str(spatial_weight),
                    "--spatial_structure_source",
                    "filter",
                    "--spatial_structure_mode",
                    "corr",
                    "--spatial_structure_trust_mode",
                    "sample",
                    "--geometry_mode",
                    "cosine",
                    "--geometry_trust_mode",
                    "sample",
                    "--confidence_mode",
                    "none",
                    "--teacher_calibration",
                    "temperature",
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
                    *common_train_args(args, seed, out_dir),
                ]
                print(f"\n# {name}", flush=True)
                run_command(command, args.dry_run)


def run_baselines(args: argparse.Namespace, root: Path) -> None:
    for seed in args.seeds:
        riem_dir = seed_dir(root, "riemann_paper", seed)
        csp_dir = seed_dir(root, "csp_lda", seed)
        riem_csv = riem_dir / "riemann_paper_results.csv"
        csp_csv = csp_dir / "csp_baseline_results.csv"
        for subject in args.subjects:
            for window in [2.0, 4.0]:
                riem_done = all(
                    csv_has_result(
                        riem_csv,
                        subject=subject,
                        seed=seed,
                        window=window,
                        method=method,
                        input_preprocessing="raw",
                        calibration="temperature",
                        selection="inner_cv",
                        selection_folds=5,
                    )
                    for method in ["mdm", "ts_logreg"]
                )
                if riem_done and not args.rerun_existing:
                    print(f"Skip existing Riemann baseline rows: subject={subject}, window={window:g}, seed={seed}", flush=True)
                else:
                    run_command(
                        [
                            sys.executable,
                            "train_riemann_paper.py",
                            "--subject",
                            str(subject),
                            "--window",
                            str(window),
                            "--seed",
                            str(seed),
                            "--save_dir",
                            str(riem_dir),
                            "--methods",
                            "mdm",
                            "ts_logreg",
                            "--estimators",
                            "oas",
                            "--alignments",
                            "none",
                            "--selection",
                            "inner_cv",
                            "--selection_folds",
                            "5",
                            "--metric",
                            "riemann",
                            "--fmin",
                            "8",
                            "--fmax",
                            "32",
                            "--tmin",
                            "0",
                            "--tmax",
                            "4",
                            "--fs",
                            "250",
                        ],
                        args.dry_run,
                    )

                if not args.rerun_existing and csv_has_result(
                    csp_csv,
                    subject=subject,
                    seed=seed,
                    window=window,
                    method="csp_lda",
                ):
                    print(f"Skip existing CSP+LDA row: subject={subject}, window={window:g}, seed={seed}", flush=True)
                    continue
                run_command(
                    [
                        sys.executable,
                        "train_csp_baseline.py",
                        "--subject",
                        str(subject),
                        "--window",
                        str(window),
                        "--seed",
                        str(seed),
                        "--save_dir",
                        str(csp_dir),
                        "--method",
                        "csp_lda",
                        "--fmin",
                        "8",
                        "--fmax",
                        "32",
                        "--tmin",
                        "0",
                        "--tmax",
                        "4",
                        "--fs",
                        "250",
                    ],
                    args.dry_run,
                )


def main() -> None:
    args = parse_args()
    base_env()
    groups = expand_groups(args.groups)
    root = Path(args.root_dir)
    teacher_root = Path(args.teacher_root)
    root.mkdir(parents=True, exist_ok=True)

    if "teachers" in groups:
        run_teachers(args, teacher_root)
    if "ce" in groups:
        run_ce(args, root)
    if "eegnetv4" in groups:
        run_eegnetv4(args, root)
    if "dual_ablation" in groups:
        run_dual_ablation(args, root, teacher_root)
    if "riemann_guided" in groups:
        run_riemann_guided(args, root)
    if "baselines" in groups:
        run_baselines(args, root)


if __name__ == "__main__":
    main()
