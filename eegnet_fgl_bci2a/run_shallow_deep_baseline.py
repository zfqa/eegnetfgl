from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ShallowConvNet and DeepConvNet baselines for all subjects.")
    parser.add_argument("--model", type=str, default="both", choices=["shallow", "deep", "both"])
    parser.add_argument("--dataset", type=str, default="bci2a", choices=["bci2a", "bci2b"])
    parser.add_argument("--subjects", type=str, default="1-9")
    parser.add_argument("--seeds", type=str, default="42,7,123,2024,3407")
    parser.add_argument("--window", type=float, default=2.0)
    parser.add_argument("--protocol", type=str, default="moabb_standard", choices=["project", "moabb_standard"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="results/shallow_deep_baseline")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    return parser.parse_args()


def parse_range(range_str: str) -> list[int]:
    if "-" in range_str:
        start, end = range_str.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(x) for x in range_str.split(",")]


def main() -> None:
    args = parse_args()
    subjects = parse_range(args.subjects)
    seeds = [int(x) for x in args.seeds.split(",")]
    models = ["shallow", "deep"] if args.model == "both" else [args.model]

    script_path = Path(__file__).parent / "train_shallow_deep_baseline.py"
    if not script_path.exists():
        print(f"Error: {script_path} not found")
        sys.exit(1)

    for model in models:
        for subject in subjects:
            for seed in seeds:
                cmd = [
                    sys.executable,
                    str(script_path),
                    "--model", model,
                    "--dataset", args.dataset,
                    "--subject", str(subject),
                    "--seed", str(seed),
                    "--window", str(args.window),
                    "--protocol", args.protocol,
                    "--device", args.device,
                    "--save_dir", args.save_dir,
                    "--epochs", str(args.epochs),
                    "--patience", str(args.patience),
                    "--batch_size", str(args.batch_size),
                    "--lr", str(args.lr),
                    "--weight_decay", str(args.weight_decay),
                ]
                print(f"Running: {model} subject {subject} seed {seed}")
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"Error: {result.stderr}")
                else:
                    print(result.stdout.strip())


if __name__ == "__main__":
    main()
