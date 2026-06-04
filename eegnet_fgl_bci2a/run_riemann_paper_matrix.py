from __future__ import annotations

import argparse
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper-grade Riemannian baselines over subjects/windows/seeds.")
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 10)))
    parser.add_argument("--windows", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--save_dir", type=str, default="results/riemann_paper")
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["mdm", "fgmdm", "ts_logreg", "ts_svm", "ts_lda"],
        choices=["mdm", "fgmdm", "ts_logreg", "ts_svm", "ts_lda"],
    )
    parser.add_argument("--estimators", type=str, nargs="+", default=["oas", "lwf", "scm"])
    parser.add_argument("--alignments", type=str, nargs="+", default=["none", "euclid", "riemann"])
    parser.add_argument("--metric", type=str, default="riemann")
    parser.add_argument("--fmin", type=float, default=4)
    parser.add_argument("--fmax", type=float, default=38)
    parser.add_argument("--tmin", type=float, default=0)
    parser.add_argument("--tmax", type=float, default=4)
    parser.add_argument("--fs", type=int, default=250)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--c_grid", type=str, default="0.01,0.03,0.1,0.3,1,3,10")
    parser.add_argument("--cov_shrinkage", type=float, default=0.001)
    parser.add_argument("--rerun_existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for seed in args.seeds:
        for subject in args.subjects:
            for window in args.windows:
                print(f"\n--- Riemann paper subject={subject} window={window:g}s seed={seed} ---", flush=True)
                command = [
                    sys.executable,
                    "train_riemann_paper.py",
                    "--subject",
                    str(subject),
                    "--window",
                    str(window),
                    "--seed",
                    str(seed),
                    "--save_dir",
                    args.save_dir,
                    "--methods",
                    *args.methods,
                    "--estimators",
                    *args.estimators,
                    "--alignments",
                    *args.alignments,
                    "--metric",
                    args.metric,
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
                    "--c_grid",
                    args.c_grid,
                    "--cov_shrinkage",
                    str(args.cov_shrinkage),
                ]
                if args.rerun_existing:
                    command.append("--rerun_existing")
                print(" ".join(command), flush=True)
                subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
