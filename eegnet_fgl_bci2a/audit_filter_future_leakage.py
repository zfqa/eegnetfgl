from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import sosfiltfilt

from utils.data_loader import (
    causal_bandpass_continuous,
    design_causal_bandpass_sos,
    resolve_bnci_mat_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit whether samples after the student deadline affect its filtered input."
    )
    parser.add_argument("--datasets", nargs="+", choices=["bci2a", "bci2b"], default=["bci2a", "bci2b"])
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--student_window", type=float, default=2.0)
    parser.add_argument("--fmin", type=float, default=8.0)
    parser.add_argument("--fmax", type=float, default=32.0)
    parser.add_argument("--filter_order", type=int, default=4)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_csv", type=str, default="results/filter_future_leakage_audit.csv")
    return parser.parse_args()


def _first_trial_run(mat_path: Path):
    runs = np.atleast_1d(loadmat(mat_path, struct_as_record=False, squeeze_me=True)["data"])
    for run_index, run in enumerate(runs):
        trials = np.atleast_1d(run.trial).astype(np.int64)
        if trials.size:
            return run_index, run, trials
    raise ValueError(f"No trial-containing run found in {mat_path}.")


def _difference_metrics(reference: np.ndarray, perturbed: np.ndarray) -> dict[str, float]:
    difference = np.asarray(perturbed, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    rmse = float(np.sqrt(np.mean(np.square(difference))))
    denominator = float(np.sqrt(np.mean(np.square(reference))))
    return {
        "max_abs_change": float(np.max(np.abs(difference))),
        "rmse_change": rmse,
        "relative_rmse": rmse / max(denominator, 1e-12),
    }


def audit_dataset(args: argparse.Namespace, dataset: str) -> list[dict[str, float | int | str]]:
    specification = {
        "bci2a": ("BNCI2014_001", 22, 2.0),
        "bci2b": ("BNCI2014_004", 3, 3.0),
    }
    dataset_name, n_channels, cue_offset = specification[dataset]
    mat_path = resolve_bnci_mat_path(dataset_name, args.subject, "T", args.data_dir)
    run_index, run, trials = _first_trial_run(mat_path)
    fs = int(run.fs)
    continuous = np.asarray(run.X[:, :n_channels], dtype=np.float64)
    trial_start = int(trials[0]) - 1
    epoch_start = trial_start + int(round(cue_offset * fs))
    student_stop = epoch_start + int(round(args.student_window * fs))
    perturbation_stop = min(
        len(continuous),
        student_stop + int(round(max(4.0 - args.student_window, 0.5) * fs)),
    )
    perturbed = continuous.copy()
    perturbed[student_stop:perturbation_stop] = 0.0

    sos = design_causal_bandpass_sos(args.fmin, args.fmax, fs, args.filter_order)
    zero_reference = sosfiltfilt(sos, continuous, axis=0)
    zero_perturbed = sosfiltfilt(sos, perturbed, axis=0)
    causal_reference = causal_bandpass_continuous(
        continuous, args.fmin, args.fmax, fs, args.filter_order
    )
    causal_perturbed = causal_bandpass_continuous(
        perturbed, args.fmin, args.fmax, fs, args.filter_order
    )
    student_slice = slice(epoch_start, student_stop)

    rows = []
    for filter_mode, reference, changed in [
        ("zero_phase_forward_backward", zero_reference, zero_perturbed),
        ("causal_one_pass", causal_reference, causal_perturbed),
    ]:
        rows.append(
            {
                "dataset": dataset,
                "subject": args.subject,
                "source_file": str(mat_path),
                "run": run_index,
                "fs": fs,
                "fmin": args.fmin,
                "fmax": args.fmax,
                "filter_order": args.filter_order,
                "student_window": args.student_window,
                "future_perturbation_start_s": args.student_window,
                "future_perturbation_duration_s": (perturbation_stop - student_stop) / fs,
                "filter_mode": filter_mode,
                **_difference_metrics(reference[student_slice], changed[student_slice]),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    rows = []
    for dataset in args.datasets:
        rows.extend(audit_dataset(args, dataset))
    frame = pd.DataFrame(rows)
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(frame.to_string(index=False))
    print(f"\nSaved: {output.resolve()}")
    causal = frame[frame["filter_mode"] == "causal_one_pass"]
    if not np.allclose(causal["max_abs_change"], 0.0, atol=1e-12):
        raise RuntimeError("Causal filter audit failed: future perturbation changed the student prefix.")


if __name__ == "__main__":
    main()
