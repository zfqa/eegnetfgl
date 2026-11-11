"""Independent, paper-guided dynamic-window PVN-LBRL experiment."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.pvn_lbrl import PVNLBRL
from pvn_lbrl_training import evaluate, pretrain_belief, pretrain_encoder, train_policy
from utils.data_loader import (
    configure_mne_data_path,
    load_bci_subject_preprocessed,
    split_session_train_val_test,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="+", type=int, default=list(range(1, 10)))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 2024, 3407, 7, 123])
    parser.add_argument("--dataset", choices=["bci2a"], default="bci2a")
    parser.add_argument("--slice_samples", type=int, choices=[40, 60, 80, 100], default=40)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--preprocessing_mode", choices=["moabb_zero_phase", "causal_mat"], default="moabb_zero_phase")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--fmin", type=float, default=1.0)
    parser.add_argument("--fmax", type=float, default=48.0)
    parser.add_argument("--scale", type=float, default=1e6)
    parser.add_argument("--train_zscore", action="store_true")
    parser.add_argument("--epochs_pvn", type=int, default=300)
    parser.add_argument("--epochs_belief", type=int, default=300)
    parser.add_argument("--rl_steps", type=int, default=50_000)
    parser.add_argument("--actors", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--replay_batch_size", type=int, default=16)
    parser.add_argument("--validate_every", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="results/pvn_lbrl_paper_guided")
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def complete_online_samples(trial_samples: int, slice_samples: int) -> int:
    """Only full incoming slices are observable, matching floor(T/L)."""
    if slice_samples <= 0 or trial_samples < slice_samples:
        raise ValueError("A positive slice_samples must fit within the trial")
    return (trial_samples // slice_samples) * slice_samples


def load_split(args: argparse.Namespace, subject: int, seed: int) -> tuple[torch.Tensor, ...]:
    configure_mne_data_path(args.data_dir)
    X, y, metadata = load_bci_subject_preprocessed(
        args.dataset, subject, preprocessing_mode=args.preprocessing_mode,
        fmin=args.fmin, fmax=args.fmax, tmin=0.0, tmax=4.0,
        data_dir=args.data_dir,
    )
    X_train, y_train, X_val, y_val, X_test, y_test = split_session_train_val_test(
        X, y, metadata, seed=seed,
    )
    X_train = X_train[..., :1000].astype(np.float32) * args.scale
    X_val = X_val[..., :1000].astype(np.float32) * args.scale
    X_test = X_test[..., :1000].astype(np.float32) * args.scale
    if args.train_zscore:
        mean = X_train.mean(axis=(0, 2), keepdims=True)
        std = X_train.std(axis=(0, 2), keepdims=True).clip(min=1e-6)
        X_train, X_val, X_test = (
            (X_train - mean) / std, (X_val - mean) / std, (X_test - mean) / std,
        )
    print(
        f"subject={subject} seed={seed} train={len(X_train)} val={len(X_val)} "
        f"test={len(X_test)} shape={X_train.shape[1:]}", flush=True,
    )
    return tuple(
        torch.from_numpy(array.copy())
        for array in (X_train, y_train, X_val, y_val, X_test, y_test)
    )


def run_subject(args: argparse.Namespace, subject: int, seed: int, output_dir: Path) -> dict[str, object]:
    seed_all(seed)
    X_train, y_train, X_val, y_val, X_test, y_test = load_split(args, subject, seed)
    device = torch.device(args.device)
    model = PVNLBRL(
        channels=X_train.shape[1], classes=4, slice_samples=args.slice_samples,
    ).to(device)
    pvn_accuracy = pretrain_encoder(
        model, X_train, y_train, X_val, y_val,
        epochs=args.epochs_pvn, batch_size=args.batch_size,
    )
    online_samples = complete_online_samples(X_train.shape[-1], args.slice_samples)
    train_online = X_train[..., :online_samples]
    val_online = X_val[..., :online_samples]
    test_online = X_test[..., :online_samples]
    belief_accuracy = pretrain_belief(
        model, train_online, y_train, val_online, y_val,
        epochs=args.epochs_belief, batch_size=args.batch_size,
    )
    val_metrics = train_policy(
        model, train_online, y_train, val_online, y_val, steps=args.rl_steps,
        actors=args.actors, gamma=args.gamma, batch_size=args.replay_batch_size,
        validate_every=args.validate_every,
    )
    # No evaluation-session labels enter fitting, stopping, or checkpoint choice.
    test_metrics = evaluate(model, test_online, y_test)
    checkpoint = output_dir / f"subject_{subject}_seed_{seed}_best.pth"
    torch.save({"model_state": model.state_dict(), "config": vars(args),
                "subject": subject, "seed": seed, "validation": val_metrics}, checkpoint)
    result = {
        "dataset": args.dataset, "subject": subject, "seed": seed,
        "slice_samples": args.slice_samples, "gamma": args.gamma,
        "preprocessing_mode": args.preprocessing_mode, "fmin": args.fmin,
        "fmax": args.fmax, "scale": args.scale, "train_zscore": args.train_zscore,
        "train_trials": len(X_train), "val_trials": len(X_val), "test_trials": len(X_test),
        "online_samples": online_samples,
        "pvn_val_accuracy": pvn_accuracy, "belief_val_accuracy": belief_accuracy,
        **{f"val_{key}": value for key, value in val_metrics.items()},
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }
    print(f"FINAL subject={subject} seed={seed} {test_metrics}", flush=True)
    return result


def write_summary(rows: list[dict[str, object]], root: Path, expected_subjects: set[int]) -> None:
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "subject_results.csv", index=False)
    if frame.empty:
        return
    metrics = ["test_accuracy", "test_macro_f1", "test_kappa", "test_mean_prediction_s", "test_itr_bits_min"]
    seed_summary = frame.groupby("seed", as_index=False)[metrics].mean()
    seed_summary["n_subjects"] = frame.groupby("seed")["subject"].nunique().to_numpy()
    seed_summary.to_csv(root / "seed_summary.csv", index=False)
    complete_seeds = [
        seed for seed, group in frame.groupby("seed")
        if set(group.subject.astype(int)) == expected_subjects
    ]
    complete = seed_summary[seed_summary.seed.isin(complete_seeds)]
    summary = complete[metrics].agg(["mean", "std"]).T.reset_index(names="metric")
    summary.to_csv(root / "summary.csv", index=False)


def main() -> None:
    args = parse_args()
    if args.slice_samples <= 0:
        raise ValueError("slice_samples must be positive")
    if not 0 <= args.gamma < 1:
        raise ValueError("gamma must be in [0, 1)")
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "run_config.json"
    config = {key: value for key, value in vars(args).items() if key != "output_dir"}
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("Existing output directory has different experiment settings")
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    results_path = root / "subject_results.csv"
    rows = pd.read_csv(results_path).to_dict("records") if results_path.exists() else []
    for seed in args.seeds:
        for subject in args.subjects:
            if any(int(row["seed"]) == seed and int(row["subject"]) == subject for row in rows):
                print(f"SKIP subject={subject} seed={seed}", flush=True)
                continue
            result = run_subject(args, subject, seed, root)
            rows.append(result)
            write_summary(rows, root, set(args.subjects))
    print(f"DONE rows={len(rows)} output={root}", flush=True)


if __name__ == "__main__":
    main()
