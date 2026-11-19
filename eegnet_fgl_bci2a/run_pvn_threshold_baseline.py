"""Cumulative-PVN fixed-window versus validation-selected threshold stopping."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

from models.pvn_lbrl import PVNLBRL
from pvn_lbrl_training import pretrain_encoder
from run_pvn_lbrl_ablation import load_split, seed_all

VERSION = "cumulative_pvn_vote_threshold_v1"


@torch.no_grad()
def vote(model: PVNLBRL, prefix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    classes = model.encoder.forward_logits(prefix).argmax(1)
    counts = torch.nn.functional.one_hot(classes, model.classes).sum(1)
    maximum, prediction = counts.max(1)
    # Vote share is a confidence heuristic, not a calibrated correctness probability.
    return prediction, maximum.float() / prefix.shape[-1]


@torch.no_grad()
def predict(
    model: PVNLBRL, x: torch.Tensor, samples: list[int], threshold: float,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Only unresolved trials are evaluated at each currently available prefix."""
    model.eval()
    device = next(model.parameters()).device
    if not samples or samples != sorted(set(samples)) or samples[0] < 1 or samples[-1] > x.shape[-1]:
        raise ValueError("Invalid observation endpoints")
    if math.isnan(threshold):
        raise ValueError("Threshold cannot be NaN")
    predictions, endpoints = [], []
    for batch in x.split(batch_size):
        active = torch.arange(len(batch))
        chosen = torch.empty(len(batch), dtype=torch.long)
        times = torch.empty(len(batch), dtype=torch.long)
        for length in samples:
            label, confidence = vote(model, batch[active, :, :length].to(device))
            label, confidence = label.cpu(), confidence.cpu()
            halt = confidence >= threshold
            if length == samples[-1]:
                halt[:] = True
            chosen[active[halt]] = label[halt]
            times[active[halt]] = length
            active = active[~halt]
            if not len(active):
                break
        predictions.append(chosen.numpy())
        endpoints.append(times.numpy())
    return np.concatenate(predictions), np.concatenate(endpoints)


def select_threshold(
    model: PVNLBRL, x_val: torch.Tensor, y_val: torch.Tensor,
    samples: list[int], thresholds: list[float], batch_size: int,
) -> tuple[float, pd.DataFrame]:
    """Minimize validation observation time with no accuracy loss versus fixed."""
    rows = []
    truth = y_val.cpu().numpy()
    for threshold in [math.inf, *sorted(set(thresholds))]:
        prediction, times = predict(model, x_val, samples, threshold, batch_size)
        rows.append({
            "threshold": threshold, "accuracy": accuracy_score(truth, prediction),
            "mean_samples": float(times.mean()),
        })
    baseline = rows[0]["accuracy"]
    feasible = [r for r in rows if r["accuracy"] >= baseline]
    # Prefer fixed fallback on exact ties; no reward tuning or test labels.
    selected = min(feasible, key=lambda r: (r["mean_samples"], -r["accuracy"], -r["threshold"]))
    frame = pd.DataFrame(rows)
    frame["selected"] = frame.threshold == selected["threshold"]
    return float(selected["threshold"]), frame


def metrics(y: np.ndarray, prediction: np.ndarray, times: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro", labels=list(range(4)), zero_division=0)),
        "kappa": float(cohen_kappa_score(y, prediction)),
        "mean_observation_s": float(times.mean() / 250),
        "confusion_matrix": json.dumps(confusion_matrix(y, prediction, labels=list(range(4))).tolist()),
    }


def run_subject(args: argparse.Namespace, subject: int, seed: int, root: Path) -> list[dict]:
    seed_all(seed)
    x_train, y_train, x_val, y_val, x_test, y_test = load_split(args, subject, seed)
    if not len(y_val):
        raise ValueError("A training-session validation split is required")
    model = PVNLBRL(channels=x_train.shape[1], classes=4, slice_samples=40).to(args.device)
    pretrain_encoder(model, x_train, y_train, x_val, y_val,
                     epochs=args.epochs_pvn, batch_size=args.batch_size)
    folder = root / f"s{subject}_seed{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    samples = [280, 320, 360, 400, 440, 480]  # 1.12 ... 1.92 s, fixed before testing.
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    threshold, selection = select_threshold(model, x_val, y_val, samples, thresholds, args.batch_size)
    selection.to_csv(folder / "validation_threshold_selection.csv", index=False)
    policy = {
        "version": VERSION, "subject": subject, "seed": seed, "samples": samples,
        "threshold": None if math.isinf(threshold) else threshold,
        "fallback_fixed": math.isinf(threshold),
        "selection_rule": "min mean time; validation accuracy >= fixed; deterministic tie break",
        "confidence": "hard point-vote share, uncalibrated",
    }
    # Persist the chosen rule and frozen weights before any test labels are scored.
    (folder / "locked_policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
    torch.save({"encoder_state": model.encoder.state_dict(), "policy": policy,
                "config": vars(args)}, folder / "pvn.pth")
    rows = []
    for name, stop_threshold in (("fixed_1p92s", math.inf), ("threshold", threshold)):
        prediction, times = predict(model, x_test, samples, stop_threshold, args.batch_size)
        row = {"subject": subject, "seed": seed, "method": name,
               "threshold": None if math.isinf(stop_threshold) else stop_threshold,
               **metrics(y_test.numpy(), prediction, times)}
        rows.append(row)
        pd.DataFrame({"trial_index": np.arange(len(y_test)), "y_true": y_test.numpy(),
                      "y_pred": prediction, "observation_s": times / 250}).to_csv(
                          folder / f"{name}_test_predictions.csv", index=False)
        print(f"FINAL subject={subject} seed={seed} {name} {row}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="+", type=int, default=[1])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 2024, 3407, 7, 123])
    parser.add_argument("--epochs_pvn", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--cpu_threads", type=int, default=1)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    if min(args.epochs_pvn, args.batch_size, args.cpu_threads) < 1:
        parser.error("Training settings must be positive")
    # Match the audited offline pipeline; do not silently introduce new preprocessing.
    args.dataset, args.preprocessing_mode = "bci2a", "moabb_zero_phase"
    args.full_training_session, args.train_zscore = False, False
    args.fmin, args.fmax, args.scale = 1.0, 48.0, 1.0
    torch.set_num_threads(args.cpu_threads)
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    config = {k: v for k, v in vars(args).items() if k != "output_dir"}
    config["version"] = VERSION
    manifest = root / "run_config.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != config:
        raise ValueError("Use a new directory for changed settings")
    manifest.write_text(json.dumps(config, indent=2), encoding="utf-8")
    path = root / "subject_results.csv"
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    for seed in args.seeds:
        for subject in args.subjects:
            existing = [r for r in rows if r["subject"] == subject and r["seed"] == seed]
            if existing:
                if len(existing) != 2 or {r["method"] for r in existing} != {"fixed_1p92s", "threshold"}:
                    raise ValueError("Incomplete paired result in CSV")
                print(f"SKIP subject={subject} seed={seed}", flush=True)
                continue
            rows.extend(run_subject(args, subject, seed, root))
            frame = pd.DataFrame(rows)
            frame.to_csv(path, index=False)
            columns = ["accuracy", "macro_f1", "kappa", "mean_observation_s"]
            by_seed = frame.groupby(["seed", "method"])[columns].mean().reset_index()
            by_seed.to_csv(root / "seed_summary.csv", index=False)
            counts = frame.groupby(["seed", "method"]).subject.nunique()
            complete = counts[counts == len(set(args.subjects))].index
            complete_rows = by_seed.set_index(["seed", "method"]).loc[complete].reset_index()
            complete_rows.groupby("method")[columns].agg(["mean", "std", "count"]).to_csv(
                root / "summary.csv")
    print(f"DONE rows={len(rows)} output={root.resolve()}", flush=True)


if __name__ == "__main__":
    main()
