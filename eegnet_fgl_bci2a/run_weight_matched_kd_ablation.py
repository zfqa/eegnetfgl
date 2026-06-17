from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_SEEDS = [42, 2024, 3407, 7, 123]
DEFAULT_SUBJECTS = list(range(1, 10))


@dataclass(frozen=True)
class KDConfig:
    name: str
    eegnet_weight: float
    riemann_weight: float

    @property
    def teacher_mode(self) -> str:
        if self.eegnet_weight > 0 and self.riemann_weight > 0:
            return "dual"
        if self.eegnet_weight > 0:
            return "eegnet"
        return "riemann"


CONFIGS = [
    KDConfig("EEGNet teacher only alpha0.8", 0.8, 0.0),
    KDConfig("Riemann teacher only alpha0.8", 0.0, 0.8),
    KDConfig("Dual KD alpha0.2+0.2", 0.2, 0.2),
]
CONFIG_ALIASES = {
    "eegnet_08": CONFIGS[0],
    "riemann_08": CONFIGS[1],
    "dual_02": CONFIGS[2],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run total-KD-weight-matched single- versus dual-teacher ablations."
    )
    parser.add_argument("--datasets", nargs="+", choices=["bci2a", "bci2b"], default=["bci2a", "bci2b"])
    parser.add_argument("--subjects", type=int, nargs="+", default=DEFAULT_SUBJECTS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=sorted(CONFIG_ALIASES),
        default=list(CONFIG_ALIASES),
    )
    parser.add_argument("--root_dir", default="results/weight_matched_kd_unified")
    parser.add_argument("--teacher_root", default="results/unified_teachers")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--rerun_existing", action="store_true")
    return parser.parse_args()


def set_data_environment() -> None:
    os.environ.setdefault("MNE_DATA", "data/mne_data")
    os.environ.setdefault("MNE_DATASETS_BNCI_PATH", "data/mne_data")
    os.environ.setdefault("MNE_DONTWRITE_HOME", "true")


def float_equal(actual: object, expected: float) -> bool:
    try:
        return abs(float(actual) - expected) < 1e-8
    except (TypeError, ValueError):
        return False


def csv_has_result(
    csv_path: Path,
    dataset: str,
    subject: int,
    seed: int,
    eegnet_weight: float,
    riemann_weight: float,
) -> bool:
    if not csv_path.exists():
        return False
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                row.get("dataset", dataset) == dataset
                and int(row["subject"]) == subject
                and int(row["seed"]) == seed
                and float_equal(row.get("window", row.get("student_window")), 2.0)
                and float_equal(row.get("eegnet_kd_weight", row.get("alpha1")), eegnet_weight)
                and float_equal(row.get("kl_weight", row.get("alpha2")), riemann_weight)
                and row.get("teacher_calibration") == "temperature"
                and (dataset != "bci2b" or row.get("riemann_input_preprocessing") == "raw")
            ):
                return True
    return False


def bci2a_teacher(root: str | Path, seed: int, subject: int) -> Path:
    path = Path(root) / "bci2a" / f"seed_{seed}" / f"subject_{subject}_teacher_best.pth"
    if not path.exists():
        raise FileNotFoundError(f"Validated BCI IV-2a teacher checkpoint not found: {path}")
    return path


def bci2b_teacher(root: str | Path, seed: int, subject: int) -> Path:
    path = (
        Path(root)
        / "bci2b"
        / f"seed_{seed}"
        / f"bci2b_subject_{subject}_teacher_best.pth"
    )
    if not path.exists():
        raise FileNotFoundError(f"Validated compact BCI IV-2b teacher checkpoint not found: {path}")
    return path


def run_command(command: list[str], dry_run: bool) -> None:
    print("\n" + subprocess.list2cmdline(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def common_args(args: argparse.Namespace, seed: int, save_dir: Path) -> list[str]:
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


def selected_configs(args: argparse.Namespace) -> list[KDConfig]:
    return [CONFIG_ALIASES[name] for name in args.configs]


def run_bci2a(args: argparse.Namespace, root: Path) -> None:
    for seed in args.seeds:
        save_dir = root / "bci2a" / f"seed_{seed}"
        save_dir.mkdir(parents=True, exist_ok=True)
        result_csv = save_dir / "dual_kd_results.csv"
        for subject in args.subjects:
            for config in selected_configs(args):
                if not args.rerun_existing and csv_has_result(
                    result_csv,
                    "bci2a",
                    subject,
                    seed,
                    config.eegnet_weight,
                    config.riemann_weight,
                ):
                    print(f"Skip existing BCI2a S{subject} seed={seed}: {config.name}", flush=True)
                    continue
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
                    config.teacher_mode,
                    "--method_name",
                    config.name,
                    "--alpha1",
                    str(config.eegnet_weight),
                    "--alpha2",
                    str(config.riemann_weight),
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
                    *common_args(args, seed, save_dir),
                ]
                if config.eegnet_weight > 0:
                    command.extend(
                        [
                            "--eegnet_teacher_path",
                            str(bci2a_teacher(args.teacher_root, seed, subject)),
                        ]
                    )
                run_command(command, args.dry_run)


def run_bci2b(args: argparse.Namespace, root: Path) -> None:
    for seed in args.seeds:
        save_dir = root / "bci2b" / f"seed_{seed}"
        save_dir.mkdir(parents=True, exist_ok=True)
        result_csv = save_dir / "riemann_guided_results.csv"
        for subject in args.subjects:
            for config in selected_configs(args):
                if not args.rerun_existing and csv_has_result(
                    result_csv,
                    "bci2b",
                    subject,
                    seed,
                    config.eegnet_weight,
                    config.riemann_weight,
                ):
                    print(f"Skip existing BCI2b S{subject} seed={seed}: {config.name}", flush=True)
                    continue
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
                    str(config.eegnet_weight),
                    "--kl_weight",
                    str(config.riemann_weight),
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
                    *common_args(args, seed, save_dir),
                ]
                if config.eegnet_weight > 0:
                    command.extend(
                        [
                            "--eegnet_teacher_path",
                            str(bci2b_teacher(args.teacher_root, seed, subject)),
                        ]
                    )
                run_command(command, args.dry_run)


def main() -> None:
    args = parse_args()
    set_data_environment()
    root = Path(args.root_dir)
    if "bci2a" in args.datasets:
        run_bci2a(args, root)
    if "bci2b" in args.datasets:
        run_bci2b(args, root)


if __name__ == "__main__":
    main()
