from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_SEEDS = [42, 2024, 3407, 7, 123]
DEFAULT_SUBJECTS = list(range(1, 10))
DEFAULT_WINDOWS = [2.0, 4.0]
DEFAULT_DATASETS = ["bci2a", "bci2b"]
DEFAULT_BANDS = "8-12,12-16,16-20,20-24,24-28,28-32"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 8-32 Hz FBCSP-LDA baselines.")
    parser.add_argument("--datasets", nargs="+", choices=["bci2a", "bci2b"], default=DEFAULT_DATASETS)
    parser.add_argument("--subjects", type=int, nargs="+", default=DEFAULT_SUBJECTS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--windows", type=float, nargs="+", default=DEFAULT_WINDOWS)
    parser.add_argument("--save_root", type=str, default="results/fbcsp_lda_8_32")
    parser.add_argument("--filter_bank", type=str, default=DEFAULT_BANDS)
    parser.add_argument("--n_components_grid", type=str, default="4,6,8,10,12")
    parser.add_argument("--reg", type=str, default="ledoit_wolf")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--rerun_existing", action="store_true")
    return parser.parse_args()


def run(command: list[str], cwd: Path) -> None:
    print("\n" + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    save_root = Path(args.save_root)
    total = len(args.datasets) * len(args.seeds) * len(args.subjects) * len(args.windows)
    done = 0

    for dataset in args.datasets:
        for seed in args.seeds:
            save_dir = save_root / dataset / f"seed_{seed}" / "csp"
            save_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n=== dataset={dataset} seed={seed} save_dir={save_dir} ===", flush=True)
            for subject in args.subjects:
                for window in args.windows:
                    done += 1
                    print(
                        f"\n[{done}/{total}] FBCSP-LDA 8-32Hz | "
                        f"dataset={dataset} subject={subject} window={window:g}s seed={seed}",
                        flush=True,
                    )
                    command = [
                        sys.executable,
                        "train_csp_baseline.py",
                        "--dataset",
                        dataset,
                        "--subject",
                        str(subject),
                        "--window",
                        str(window),
                        "--seed",
                        str(seed),
                        "--save_dir",
                        str(save_dir),
                        "--method",
                        "fbcsp_lda",
                        "--fmin",
                        "8",
                        "--fmax",
                        "32",
                        "--filter_bank",
                        args.filter_bank,
                        "--n_components_grid",
                        args.n_components_grid,
                        "--reg",
                        args.reg,
                        "--val_ratio",
                        str(args.val_ratio),
                    ]
                    if args.rerun_existing:
                        command.append("--rerun_existing")
                    run(command, root)


if __name__ == "__main__":
    main()
