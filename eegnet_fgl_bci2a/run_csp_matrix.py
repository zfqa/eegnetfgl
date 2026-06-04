from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CSP/FBCSP baseline matrix.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 10)))
    parser.add_argument("--windows", type=float, nargs="+", default=[4.0])
    parser.add_argument(
        "--method",
        type=str,
        default="all",
        choices=[
            "all",
            "csp_lda",
            "csp_logreg",
            "fbcsp_lda",
            "fbcsp_logreg",
            "fbcsp_lda_fs",
            "fbcsp_logreg_fs",
        ],
    )
    parser.add_argument("--save_dir", type=str, default="results/csp_baseline")
    parser.add_argument("--fmin", type=float, default=4.0)
    parser.add_argument("--fmax", type=float, default=38.0)
    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=4.0)
    parser.add_argument("--fs", type=int, default=250)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--n_components_grid", type=str, default="4,6,8,10,12")
    parser.add_argument("--c_grid", type=str, default="0.01,0.1,1,10")
    parser.add_argument("--reg", type=str, default="ledoit_wolf")
    parser.add_argument("--filter_bank", type=str, default="4-8,8-12,12-16,16-20,20-24,24-28,28-32,32-38")
    parser.add_argument("--feature_selector", type=str, default="anova", choices=["anova", "mi"])
    parser.add_argument("--feature_k_grid", type=str, default="8,12,16,24,32,48,64,96,all")
    parser.add_argument("--use_train_val", action="store_true")
    parser.add_argument("--rerun_existing", action="store_true")
    return parser.parse_args()


def run_command(command: list[str], cwd: Path) -> None:
    print("\n" + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        print(f"\n=== Seed {seed} ===", flush=True)
        for subject in args.subjects:
            for window in args.windows:
                print(f"\n--- subject={subject} window={window:g}s seed={seed} ---", flush=True)
                command = [
                    sys.executable,
                    "train_csp_baseline.py",
                    "--subject",
                    str(subject),
                    "--window",
                    str(window),
                    "--seed",
                    str(seed),
                    "--save_dir",
                    str(save_dir),
                    "--method",
                    args.method,
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
                    "--val_ratio",
                    str(args.val_ratio),
                    "--n_components_grid",
                    args.n_components_grid,
                    "--c_grid",
                    args.c_grid,
                    "--reg",
                    args.reg,
                    "--filter_bank",
                    args.filter_bank,
                    "--feature_selector",
                    args.feature_selector,
                    "--feature_k_grid",
                    args.feature_k_grid,
                ]
                if args.use_train_val:
                    command.append("--use_train_val")
                if args.rerun_existing:
                    command.append("--rerun_existing")
                run_command(command, cwd=root)


if __name__ == "__main__":
    main()
