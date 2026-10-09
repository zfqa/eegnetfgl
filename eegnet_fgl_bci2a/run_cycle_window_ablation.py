"""Compare MI-EEG decoding windows indexed by nominal mu-rhythm cycles.

The cycle count is an experimental index, not a physiological optimum. Every
candidate is trained independently and evaluated on its held-out session.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
BANDS = {"full": (8.0, 32.0), "mu": (8.0, 13.0), "beta": (13.0, 30.0)}
METHODS = ("ce", "riemann_spatial")
METRICS = ("accuracy", "macro_f1", "kappa")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=("bci2a", "bci2b"), default=["bci2a", "bci2b"])
    parser.add_argument("--subjects", nargs="+", type=int, default=list(range(1, 10)))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 2024, 3407, 7, 123])
    parser.add_argument("--bands", nargs="+", choices=tuple(BANDS), default=["full"])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=["ce"])
    parser.add_argument("--mu_cycles", nargs="+", type=float, default=[5, 10, 15, 20, 25, 30, 40])
    parser.add_argument("--mu_reference_hz", type=float, default=10.0)
    parser.add_argument("--beta_reference_hz", type=float, default=20.0)
    parser.add_argument("--teacher_window", type=float, default=4.0)
    parser.add_argument("--riemann_kd_weight", type=float, default=0.1)
    parser.add_argument("--spatial_weight", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data_dir", type=Path, help="MNE data root (the parent of MNE-bnci-data).")
    parser.add_argument("--require_cached_data", action="store_true", help="Fail before training if a BNCI file is absent.")
    parser.add_argument("--output_dir", type=Path, default=Path("results/cycle_window_ablation"))
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--summarize_only", action="store_true")
    return parser.parse_args()


def windows_from_cycles(cycles: list[float], reference_hz: float, teacher_window: float) -> list[float]:
    if not math.isfinite(reference_hz) or reference_hz <= 0:
        raise ValueError("mu_reference_hz must be positive and finite")
    if not math.isfinite(teacher_window) or teacher_window <= 0:
        raise ValueError("teacher_window must be positive and finite")
    if not cycles or any(not math.isfinite(n) or n <= 0 for n in cycles):
        raise ValueError("mu_cycles must contain positive, finite numbers")
    windows = sorted({round(n / reference_hz, 6) for n in cycles})
    if windows[0] <= 0 or windows[-1] > teacher_window:
        raise ValueError("Every resulting window must be in (0, teacher_window]")
    if any(not math.isclose(window * 250, round(window * 250), abs_tol=1e-6) for window in windows):
        raise ValueError("Every window must map to an integer sample count at 250 Hz")
    return windows


def result_path(root: Path, dataset: str, band: str, method: str) -> Path:
    return root / dataset / band / method / "riemann_guided_results.csv"


def training_config(args: argparse.Namespace, dataset: str, band: str, method: str) -> dict[str, object]:
    return {
        "dataset": dataset, "band": band, "method": method,
        "fmin": BANDS[band][0], "fmax": BANDS[band][1],
        "teacher_window": args.teacher_window,
        "riemann_kd_weight": args.riemann_kd_weight if method == "riemann_spatial" else 0.0,
        "spatial_weight": args.spatial_weight if method == "riemann_spatial" else 0.0,
        "epochs": args.epochs, "patience": args.patience,
        "batch_size": args.batch_size, "lr": args.lr,
        "weight_decay": args.weight_decay,
    }


def required_cache_files(data_dir: Path, dataset: str, subject: int) -> list[Path]:
    if dataset == "bci2a":
        base = data_dir / "MNE-bnci-data" / "~bci" / "database" / "001-2014"
        return [base / f"A{subject:02d}{suffix}.mat" for suffix in ("T", "E")]
    base = data_dir / "MNE-bnci-data" / "~bci" / "database" / "004-2014"
    return [base / f"B{subject:02d}{suffix}.mat" for suffix in ("T", "E")]


def mne_data_path_for_subprocess(data_dir: Path) -> str:
    # MNE's BNCI path resolver treats a Windows drive-letter absolute path as a URI.
    return os.path.relpath(data_dir, PROJECT_ROOT) if os.name == "nt" else str(data_dir)


def check_training_config(path: Path, config: dict[str, object]) -> None:
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            if json.load(stream) != config:
                raise ValueError(f"Training settings differ from previous runs in {path.parent}; use a new output_dir")
        return
    if (path.parent / "riemann_guided_results.csv").exists():
        raise ValueError(f"Existing results have no run_config.json in {path.parent}; use a new output_dir")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def matching_row(rows: list[dict[str, str]], subject: int, seed: int, window: float) -> dict[str, str] | None:
    for row in reversed(rows):
        try:
            if (int(row["subject"]) == subject and int(row["seed"]) == seed
                    and math.isclose(float(row["window"]), window, abs_tol=1e-6)):
                return row
        except (KeyError, TypeError, ValueError):
            continue
    return None


def build_command(
    args: argparse.Namespace, dataset: str, band: str, method: str,
    subject: int, seed: int, window: float, save_dir: Path,
) -> list[str]:
    fmin, fmax = BANDS[band]
    kd_weight = args.riemann_kd_weight if method == "riemann_spatial" else 0.0
    spatial_weight = args.spatial_weight if method == "riemann_spatial" else 0.0
    return [
        sys.executable, str(PROJECT_ROOT / "train_student_riemann_guided.py"),
        "--dataset", dataset, "--subject", str(subject), "--seed", str(seed),
        "--window", f"{window:g}", "--riemann_window", f"{args.teacher_window:g}",
        "--model_variant", "compact", "--fmin", f"{fmin:g}", "--fmax", f"{fmax:g}",
        "--tmin", "0", "--tmax", f"{args.teacher_window:g}",
        "--eegnet_kd_weight", "0", "--kl_weight", f"{kd_weight:g}",
        "--geometry_weight", "0", "--spatial_structure_weight", f"{spatial_weight:g}",
        "--spatial_structure_source", "pre_depthwise", "--spatial_structure_mode", "corr",
        "--spatial_structure_trust_mode", "sample", "--riemann_cov_estimator", "oas",
        "--riemann_alignment", "none", "--riemann_metric", "riemann",
        "--riemann_method", "ts_logreg", "--riemann_selection", "inner_cv",
        "--riemann_selection_folds", "5", "--c_grid", "0.01,0.03,0.1,0.3,1,3,10",
        "--skip_riemann_teacher_test",
        "--teacher_calibration", "temperature", "--confidence_mode", "none",
        "--epochs", str(args.epochs), "--patience", str(args.patience),
        "--batch_size", str(args.batch_size), "--lr", str(args.lr),
        "--weight_decay", str(args.weight_decay), "--device", args.device,
        "--save_dir", str(save_dir),
    ]


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(args: argparse.Namespace, windows: list[float], root: Path) -> tuple[int, int]:
    subject_rows: list[dict[str, object]] = []
    seed_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    expected_subjects = set(args.subjects)
    for dataset in args.datasets:
        for band in args.bands:
            for method in args.methods:
                source = read_rows(result_path(root, dataset, band, method))
                for window in windows:
                    per_seed: list[dict[str, object]] = []
                    for seed in args.seeds:
                        values: list[dict[str, object]] = []
                        for subject in sorted(expected_subjects):
                            row = matching_row(source, subject, seed, window)
                            if row is None:
                                continue
                            try:
                                metrics = {metric: float(row[metric]) for metric in METRICS}
                            except (KeyError, TypeError, ValueError):
                                continue
                            item = {
                                "dataset": dataset, "band": band, "method": method,
                                "subject": subject, "seed": seed, "window_s": window,
                                "mu_cycles": round(window * args.mu_reference_hz, 6),
                                "beta_cycles": round(window * args.beta_reference_hz, 6),
                                "low_band_cycles": round(window * BANDS[band][0], 6),
                                **metrics,
                            }
                            subject_rows.append(item)
                            values.append(item)
                        if values:
                            seed_item = {
                                "dataset": dataset, "band": band, "method": method,
                                "seed": seed, "window_s": window,
                                "mu_cycles": round(window * args.mu_reference_hz, 6),
                                "beta_cycles": round(window * args.beta_reference_hz, 6),
                                "n_subjects": len(values),
                                "complete": int(len(values) == len(expected_subjects)),
                            }
                            for metric in METRICS:
                                seed_item[metric] = sum(float(row[metric]) for row in values) / len(values)
                            seed_rows.append(seed_item)
                            if seed_item["complete"]:
                                per_seed.append(seed_item)
                    if per_seed:
                        summary_item = {
                            "dataset": dataset, "band": band, "method": method,
                            "window_s": window,
                            "mu_cycles": round(window * args.mu_reference_hz, 6),
                            "beta_cycles": round(window * args.beta_reference_hz, 6),
                            "n_complete_seeds": len(per_seed),
                            "n_subjects_per_seed": len(expected_subjects),
                        }
                        for metric in METRICS:
                            numbers = [float(row[metric]) for row in per_seed]
                            mean = sum(numbers) / len(numbers)
                            std = (sum((x - mean) ** 2 for x in numbers) / (len(numbers) - 1)) ** 0.5 if len(numbers) > 1 else ""
                            summary_item[f"mean_{metric}"] = mean
                            summary_item[f"std_{metric}"] = std
                        summary_rows.append(summary_item)
    common = ["dataset", "band", "method", "window_s", "mu_cycles", "beta_cycles"]
    write_csv(root / "subject_results.csv", subject_rows,
              common + ["low_band_cycles", "subject", "seed", *METRICS])
    write_csv(root / "seed_summary.csv", seed_rows,
              common + ["seed", "n_subjects", "complete", *METRICS])
    write_csv(root / "summary.csv", summary_rows,
              common + ["n_complete_seeds", "n_subjects_per_seed"]
              + [field for metric in METRICS for field in (f"mean_{metric}", f"std_{metric}")])
    return len(subject_rows), len(summary_rows)


def main() -> None:
    args = parse_args()
    if any(subject not in range(1, 10) for subject in args.subjects):
        raise ValueError("subjects must be in 1..9")
    if not args.subjects or not args.seeds:
        raise ValueError("subjects and seeds must be nonempty")
    if args.epochs <= 0 or args.batch_size <= 0 or args.patience < 0:
        raise ValueError("epochs/batch_size must be positive; patience must be nonnegative")
    if args.riemann_kd_weight < 0 or args.spatial_weight < 0:
        raise ValueError("loss weights must be nonnegative")
    if not math.isfinite(args.beta_reference_hz) or args.beta_reference_hz <= 0:
        raise ValueError("beta_reference_hz must be positive and finite")
    windows = windows_from_cycles(args.mu_cycles, args.mu_reference_hz, args.teacher_window)
    root = args.output_dir.resolve()
    jobs = [(dataset, band, method, subject, seed, window)
            for dataset in dict.fromkeys(args.datasets)
            for band in dict.fromkeys(args.bands)
            for method in dict.fromkeys(args.methods)
            for seed in dict.fromkeys(args.seeds)
            for subject in dict.fromkeys(args.subjects)
            for window in windows]
    print(f"Jobs={len(jobs)} | windows={windows} | output={root}", flush=True)
    if args.dry_run:
        for dataset, band, method, subject, seed, window in jobs:
            print(f"{dataset} {band} {method} s{subject} seed{seed} {window:g}s "
                  f"mu={window * args.mu_reference_hz:g} beta={window * args.beta_reference_hz:g}")
        return
    if args.data_dir:
        data_dir = args.data_dir.resolve()
        if not data_dir.is_dir():
            raise FileNotFoundError(f"Data root does not exist: {data_dir}")
    else:
        data_dir = None
    if args.require_cached_data:
        if data_dir is None:
            raise ValueError("--require_cached_data requires --data_dir")
        missing = [path for dataset in args.datasets for subject in set(args.subjects)
                   for path in required_cache_files(data_dir, dataset, subject) if not path.is_file()]
        if missing:
            raise FileNotFoundError("BNCI cache is incomplete; first missing file: " + str(missing[0]))
    if not args.summarize_only:
        for index, (dataset, band, method, subject, seed, window) in enumerate(jobs, 1):
            save_dir = root / dataset / band / method
            source = result_path(root, dataset, band, method)
            check_training_config(save_dir / "run_config.json", training_config(args, dataset, band, method))
            if matching_row(read_rows(source), subject, seed, window):
                print(f"[{index}/{len(jobs)}] SKIP {dataset} {band} {method} s{subject} seed{seed} w{window:g}", flush=True)
                continue
            save_dir.mkdir(parents=True, exist_ok=True)
            command = build_command(args, dataset, band, method, subject, seed, window, save_dir)
            log_path = save_dir / "logs" / f"s{subject}_seed{seed}_w{window:g}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            if data_dir:
                cache_path = mne_data_path_for_subprocess(data_dir)
                env.update(MNE_DATA=cache_path, MNE_DATASETS_BNCI_PATH=cache_path, MNE_DONTWRITE_HOME="true")
            print(f"[{index}/{len(jobs)}] RUN {dataset} {band} {method} s{subject} seed{seed} w{window:g} | {log_path}", flush=True)
            with log_path.open("w", encoding="utf-8") as stream:
                result = subprocess.run(command, cwd=PROJECT_ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=False)
            if result.returncode:
                raise RuntimeError(f"Training failed (exit {result.returncode}); see {log_path}")
            if not matching_row(read_rows(source), subject, seed, window):
                raise RuntimeError(f"Training exited successfully without a result row; see {log_path}")
            count, completed = summarize(args, windows, root)
            print(f"[{index}/{len(jobs)}] DONE | subject_rows={count} complete_groups={completed}", flush=True)
    count, completed = summarize(args, windows, root)
    print(f"Saved summaries in {root} | subject_rows={count} complete_groups={completed}", flush=True)


if __name__ == "__main__":
    main()
