from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the multi-seed EEGNet/Riemann/ensemble experiment matrix.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 2024, 3407])
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 10)))
    parser.add_argument("--windows", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cov_shrinkage", type=float, default=0.001)
    parser.add_argument("--riemann_cov_shrinkage", type=float, default=0.001)
    parser.add_argument("--skip_ce", action="store_true")
    parser.add_argument("--skip_riemann", action="store_true")
    parser.add_argument("--skip_ensemble", action="store_true")
    parser.add_argument("--rerun_existing", action="store_true")
    return parser.parse_args()


def window_tag(window: float) -> str:
    return f"{window:g}"


def seed_dirs(seed: int) -> tuple[Path, Path, Path]:
    if seed == 42:
        return (
            Path("results/student_ce"),
            Path("results/riemann_baseline"),
            Path("results/eegnet_riemann_fusion"),
        )
    base = Path("results") / f"seed_{seed}"
    return base / "student_ce", base / "riemann_baseline", base / "eegnet_riemann_fusion"


def ce_checkpoint(seed: int, subject: int, window: float) -> Path:
    ce_dir, _, _ = seed_dirs(seed)
    return ce_dir / f"subject_{subject}_window_{window_tag(window)}_ce_best.pth"


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def row_exists(csv_path: Path, **criteria: object) -> bool:
    for row in read_rows(csv_path):
        matched = True
        for key, expected in criteria.items():
            actual = row.get(key)
            if isinstance(expected, float):
                try:
                    matched = abs(float(actual) - expected) < 1e-9
                except (TypeError, ValueError):
                    matched = False
            else:
                matched = str(actual) == str(expected)
            if not matched:
                break
        if matched:
            return True
    return False


def both_riemann_rows_exist(save_dir: Path, subject: int, window: float, seed: int, shrinkage: float) -> bool:
    csv_path = save_dir / "riemann_baseline_results.csv"
    return row_exists(
        csv_path,
        method="Riemann-LogReg",
        subject=subject,
        window=window,
        seed=seed,
        cov_shrinkage=shrinkage,
    ) and row_exists(
        csv_path,
        method="Riemann-MDM",
        subject=subject,
        window=window,
        seed=seed,
        cov_shrinkage=shrinkage,
    )


def ensemble_row_exists(save_dir: Path, subject: int, window: float, seed: int, shrinkage: float, checkpoint: Path) -> bool:
    csv_path = save_dir / "eegnet_riemann_ensemble_results.csv"
    return row_exists(
        csv_path,
        method="EEGNet-Riemann-Ensemble",
        subject=subject,
        window=window,
        seed=seed,
        cov_shrinkage=shrinkage,
        checkpoint=str(checkpoint),
    )


def run_command(command: list[str], cwd: Path) -> None:
    print("\n" + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent

    for seed in args.seeds:
        ce_dir, riemann_dir, ensemble_dir = seed_dirs(seed)
        ce_dir.mkdir(parents=True, exist_ok=True)
        riemann_dir.mkdir(parents=True, exist_ok=True)
        ensemble_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Seed {seed} ===", flush=True)
        for subject in args.subjects:
            for window in args.windows:
                print(f"\n--- subject={subject} window={window_tag(window)}s seed={seed} ---", flush=True)
                checkpoint = ce_checkpoint(seed, subject, window)

                if not args.skip_ce:
                    ce_csv = ce_dir / "student_ce_results.csv"
                    if (
                        not args.rerun_existing
                        and checkpoint.exists()
                        and row_exists(ce_csv, method="EEGNet-CE", subject=subject, window=window, seed=seed)
                    ):
                        print(f"Skip existing EEGNet CE: {checkpoint}", flush=True)
                    else:
                        run_command(
                            [
                                sys.executable,
                                "train_student_ce.py",
                                "--subject",
                                str(subject),
                                "--window",
                                str(window),
                                "--epochs",
                                str(args.epochs),
                                "--batch_size",
                                str(args.batch_size),
                                "--lr",
                                str(args.lr),
                                "--weight_decay",
                                str(args.weight_decay),
                                "--seed",
                                str(seed),
                                "--device",
                                args.device,
                                "--save_dir",
                                str(ce_dir),
                            ],
                            cwd=root,
                        )

                if not args.skip_riemann:
                    if (
                        not args.rerun_existing
                        and both_riemann_rows_exist(riemann_dir, subject, window, seed, args.riemann_cov_shrinkage)
                    ):
                        print("Skip existing Riemann LogReg/MDM rows.", flush=True)
                    else:
                        run_command(
                            [
                                sys.executable,
                                "train_riemann_baseline.py",
                                "--subject",
                                str(subject),
                                "--window",
                                str(window),
                                "--seed",
                                str(seed),
                                "--save_dir",
                                str(riemann_dir),
                                "--classifier",
                                "all",
                                "--cov_shrinkage",
                                str(args.riemann_cov_shrinkage),
                            ],
                            cwd=root,
                        )

                if not args.skip_ensemble:
                    if not checkpoint.exists():
                        raise FileNotFoundError(f"Missing EEGNet CE checkpoint for ensemble: {checkpoint}")
                    if (
                        not args.rerun_existing
                        and ensemble_row_exists(ensemble_dir, subject, window, seed, args.cov_shrinkage, checkpoint)
                    ):
                        print("Skip existing EEGNet+Riemann ensemble row.", flush=True)
                    else:
                        run_command(
                            [
                                sys.executable,
                                "evaluate_eegnet_riemann_ensemble.py",
                                "--subject",
                                str(subject),
                                "--window",
                                str(window),
                                "--seed",
                                str(seed),
                                "--device",
                                args.device,
                                "--save_dir",
                                str(ensemble_dir),
                                "--checkpoint",
                                str(checkpoint),
                                "--cov_shrinkage",
                                str(args.cov_shrinkage),
                            ],
                            cwd=root,
                        )


if __name__ == "__main__":
    main()
