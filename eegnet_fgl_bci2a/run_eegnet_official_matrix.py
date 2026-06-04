from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EEGNetv4 stronger-protocol matrix.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 10)))
    parser.add_argument("--windows", type=float, nargs="+", default=[4.0])
    parser.add_argument("--save_dir", type=str, default="results/eegnet_official")
    parser.add_argument("--model_variant", type=str, default="eegnetv4", choices=["compact", "eegnetv4"])
    parser.add_argument("--method_name", type=str, default="")
    parser.add_argument("--final_train", type=str, default="train_val", choices=["train", "train_val"])
    parser.add_argument("--refit_epochs", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fmin", type=float, default=4.0)
    parser.add_argument("--fmax", type=float, default=40.0)
    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=4.0)
    parser.add_argument("--fs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=150)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--dropout_rate", type=float, default=0.5)
    parser.add_argument("--kernel_length", type=int, default=125)
    parser.add_argument("--rerun_existing", action="store_true")
    return parser.parse_args()


def run_command(command: list[str], cwd: Path) -> None:
    print("\n" + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    for seed in args.seeds:
        print(f"\n=== Seed {seed} ===", flush=True)
        for subject in args.subjects:
            for window in args.windows:
                print(f"\n--- subject={subject} window={window:g}s seed={seed} ---", flush=True)
                command = [
                    sys.executable,
                    "train_eegnet_official.py",
                    "--subject",
                    str(subject),
                    "--window",
                    str(window),
                    "--seed",
                    str(seed),
                    "--save_dir",
                    args.save_dir,
                    "--model_variant",
                    args.model_variant,
                ]
                if args.method_name:
                    command.extend(["--method_name", args.method_name])
                command.extend(
                    [
                        "--final_train",
                        args.final_train,
                        "--refit_epochs",
                        str(args.refit_epochs),
                        "--epochs",
                        str(args.epochs),
                        "--batch_size",
                        str(args.batch_size),
                        "--lr",
                        str(args.lr),
                        "--weight_decay",
                        str(args.weight_decay),
                        "--optimizer",
                        args.optimizer,
                        "--device",
                        args.device,
                        "--fmin",
                        str(args.fmin),
                        "--fmax",
                        str(args.fmax),
                        "--tmin",
                        str(args.tmin),
                        "--tmax",
                        str(args.tmax),
                        "--fs",
                        str(args.fs),
                        "--patience",
                        str(args.patience),
                        "--log_interval",
                        str(args.log_interval),
                        "--dropout_rate",
                        str(args.dropout_rate),
                        "--kernel_length",
                        str(args.kernel_length),
                    ]
                )
                if args.rerun_existing:
                    command.append("--rerun_existing")
                run_command(command, cwd=root)


if __name__ == "__main__":
    main()
