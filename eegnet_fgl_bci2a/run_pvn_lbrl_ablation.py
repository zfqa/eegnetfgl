"""Independent, paper-guided dynamic-window PVN-LBRL experiment."""

from __future__ import annotations

import argparse
import copy
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

PROTOCOL_VERSION = "pvn_fig2_algorithm1_balanced_warmup_v3"
SAMPLE_RATE = 250


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
    parser.add_argument(
        "--scale", type=float, default=1.0,
        help="Optional multiplier on loader output, which is already in microvolts.",
    )
    parser.add_argument("--train_zscore", action="store_true")
    parser.add_argument("--full_training_session", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--epochs_pvn", type=int, default=300)
    parser.add_argument("--epochs_belief", type=int, default=300)
    parser.add_argument("--rl_steps", type=int, default=50_000)
    parser.add_argument(
        "--audit_prefixes", action="store_true",
        help="Write train/validation prefix accuracy before and after RL without evaluating the test session.",
    )
    parser.add_argument(
        "--compare_prefix_training", action="store_true",
        help="Compare random-HALT and budget-matched balanced GRU supervision in a pretraining-only audit.",
    )
    parser.add_argument("--actors", type=int, default=8)
    parser.add_argument(
        "--epsilon_schedule", choices=["paper_literal", "r2d2_standard"],
        default="r2d2_standard",
        help="Printed PVN-LBRL formula or the standard schedule from its cited R2D2 method.",
    )
    parser.add_argument(
        "--replay_warmup_episodes", type=int, default=1024,
        help="Collect this many complete episodes before the first Q update (not reported by PVN-LBRL).",
    )
    parser.add_argument(
        "--replay_warmup_epsilon", type=float, default=1.0,
        help="Exploration rate used only during replay warm-up; 1.0 gives balanced random actions.",
    )
    parser.add_argument("--belief_updates", type=int, default=25)
    parser.add_argument("--online_belief_epochs", type=int, default=20)
    parser.add_argument("--actor_sync_every", type=int, default=64)
    parser.add_argument("--target_sync_every", type=int, default=500)
    parser.add_argument("--cpu_threads", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--replay_batch_size", type=int, default=16)
    parser.add_argument("--validate_every", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="results/pvn_lbrl_algorithm1")
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
    if getattr(args, "full_training_session", False):
        X_train, y_train = np.concatenate((X_train, X_val)), np.concatenate((y_train, y_val))
        X_val, y_val = X_val[:0], y_val[:0]
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
        f"test={len(X_test)} shape={X_train.shape[1:]} input_scale={args.scale:g}", flush=True,
    )
    return tuple(
        torch.from_numpy(array.copy())
        for array in (X_train, y_train, X_val, y_val, X_test, y_test)
    )


@torch.no_grad()
def prefix_audit(
    model: PVNLBRL, X: torch.Tensor, y: torch.Tensor, batch_size: int,
    stage: str, split: str, *, pvn_match_slices: int | None = None,
) -> list[dict[str, object]]:
    model.eval()
    device = next(model.parameters()).device
    correct = torch.zeros(X.shape[-1] // model.slice_samples, device=device)
    pvn_correct = torch.zeros((), device=device)
    pvn_match_correct = torch.zeros((), device=device)
    pvn_segmented_correct = torch.zeros((), device=device)
    if pvn_match_slices is not None and not 1 <= pvn_match_slices <= len(correct):
        raise ValueError("Matched PVN length must be a valid prefix")
    for x_batch, y_batch in zip(X.split(batch_size), y.split(batch_size)):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        encoded = model.encode_sequence(x_batch)
        beliefs, _ = model.forward_encoded_sequence(encoded)
        correct += (beliefs.argmax(-1) == y_batch[:, None]).sum(0)
        pvn_correct += (model.encoder.predict_vote(x_batch) == y_batch).sum()
        if pvn_match_slices is not None and pvn_match_slices < len(correct):
            prefix = x_batch[..., :pvn_match_slices * model.slice_samples]
            pvn_match_correct += (model.encoder.predict_vote(prefix) == y_batch).sum()
        if pvn_match_slices is not None:
            # Match PVN's point-level vote, but use exactly the GRU's slice logits.
            logits = encoded[:, :pvn_match_slices].reshape(
                len(x_batch), pvn_match_slices, model.classes, model.slice_samples,
            )
            point_classes = logits.argmax(dim=2)
            votes = torch.nn.functional.one_hot(point_classes, model.classes).sum(dim=(1, 2))
            pvn_segmented_correct += (votes.argmax(-1) == y_batch).sum()
    pvn_accuracy = pvn_correct.item() / len(y)
    rows = [
        {
            "stage": stage, "split": split, "n_trials": len(y), "slices": step,
            "observation_s": step * model.slice_samples / SAMPLE_RATE,
            "belief_accuracy": hits / len(y), "pvn_full_prefix_accuracy": pvn_accuracy,
        }
        for step, hits in enumerate(correct.tolist(), 1)
    ]
    if pvn_match_slices is not None:
        rows[pvn_match_slices - 1]["pvn_matched_prefix_accuracy"] = (
            pvn_accuracy if pvn_match_slices == len(rows) else pvn_match_correct.item() / len(y)
        )
        rows[pvn_match_slices - 1]["pvn_segmented_prefix_accuracy"] = (
            pvn_segmented_correct.item() / len(y)
        )
    return rows


def run_subject(args: argparse.Namespace, subject: int, seed: int, output_dir: Path) -> dict[str, object]:
    seed_all(seed)
    X_train, y_train, X_val, y_val, X_test, y_test = load_split(args, subject, seed)
    if args.audit_prefixes and not len(y_val):
        raise ValueError("The prefix audit requires a nonempty validation split")
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
    audit_rows: list[dict[str, object]] = []

    def record_prefixes(candidate: PVNLBRL, stage: str) -> None:
        match_slices = min(2 * SAMPLE_RATE, online_samples) // args.slice_samples
        for split, x, y in (("train", train_online, y_train), ("val", val_online, y_val)):
            rows = prefix_audit(
                candidate, x, y, args.batch_size, stage, split,
                pvn_match_slices=match_slices if args.compare_prefix_training else None,
            )
            audit_rows.extend(rows)
            print(
                f"PREFIX {stage} {split} | first={rows[0]['belief_accuracy']:.4f} "
                f"last={rows[-1]['belief_accuracy']:.4f} "
                f"pvn_full={rows[-1]['pvn_full_prefix_accuracy']:.4f}",
                flush=True,
            )
            if args.compare_prefix_training:
                matched = rows[match_slices - 1]
                print(
                    f"MATCHED {stage} {split} | seconds={matched['observation_s']:.2f} "
                    f"gru={matched['belief_accuracy']:.4f} "
                    f"pvn={matched['pvn_matched_prefix_accuracy']:.4f} "
                    f"pvn_segmented={matched['pvn_segmented_prefix_accuracy']:.4f}",
                    flush=True,
                )

    def save_audit() -> dict[str, object]:
        path = output_dir / f"subject_{subject}_seed_{seed}_prefix_audit.csv"
        pd.DataFrame(audit_rows).to_csv(path, index=False)
        print(f"Saved train/validation prefix audit: {path}", flush=True)
        return {"dataset": args.dataset, "subject": subject, "seed": seed}

    if args.compare_prefix_training:
        balanced_model = copy.deepcopy(model)
        balanced_generator = torch.Generator(device=device).manual_seed(seed)
        cuda_devices = [device] if device.type == "cuda" else []
        for stage, candidate, generator in (
            ("random_halt", model, None),
            ("balanced_prefix", balanced_model, balanced_generator),
        ):
            print(f"GRU PRETRAIN {stage} | epochs={args.epochs_belief}", flush=True)
            # Restore shuffling and rollout RNG for each arm. B has its own sampling RNG.
            with torch.random.fork_rng(devices=cuda_devices):
                pretrain_belief(
                    candidate, train_online, y_train, val_online, y_val,
                    epochs=args.epochs_belief, batch_size=args.batch_size,
                    balanced_prefix_generator=generator,
                )
            record_prefixes(candidate, stage)
        return save_audit()

    belief_accuracy = pretrain_belief(
        model, train_online, y_train, val_online, y_val,
        epochs=args.epochs_belief, batch_size=args.batch_size,
    )
    print(
        f"PRETRAIN subject={subject} seed={seed} "
        f"pvn_val_accuracy={pvn_accuracy} "
        f"belief_val_accuracy={belief_accuracy}",
        flush=True,
    )
    if args.audit_prefixes:
        record_prefixes(model, "pre_rl")
    val_metrics = train_policy(
        model, train_online, y_train, val_online, y_val, steps=args.rl_steps,
        actors=args.actors, gamma=args.gamma, batch_size=args.replay_batch_size,
        belief_updates=args.belief_updates, belief_epochs=args.online_belief_epochs,
        actor_sync_every=args.actor_sync_every, target_every=args.target_sync_every,
        validate_every=args.validate_every, epsilon_schedule=args.epsilon_schedule,
        replay_warmup_episodes=args.replay_warmup_episodes,
        replay_warmup_epsilon=args.replay_warmup_epsilon,
    ) if args.rl_steps else {}
    if args.audit_prefixes:
        if args.rl_steps:
            record_prefixes(model, "post_rl")
        return save_audit()

    # No evaluation-session labels enter fitting, stopping, or checkpoint choice.
    test_metrics = evaluate(model, X_test[..., :online_samples], y_test)
    checkpoint = output_dir / f"subject_{subject}_seed_{seed}_final.pth"
    torch.save({"model_state": model.state_dict(), "config": vars(args),
                "protocol_version": PROTOCOL_VERSION,
                "subject": subject, "seed": seed, "validation": val_metrics}, checkpoint)
    result = {
        "dataset": args.dataset, "subject": subject, "seed": seed,
        "protocol_version": PROTOCOL_VERSION, "q_updates": args.rl_steps,
        "online_belief_rounds": args.belief_updates, "online_belief_epochs": args.online_belief_epochs,
        "epsilon_schedule": args.epsilon_schedule,
        "replay_warmup_episodes": args.replay_warmup_episodes,
        "replay_warmup_epsilon": args.replay_warmup_epsilon,
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
    if not 0 <= args.belief_updates <= args.rl_steps:
        raise ValueError("Use --belief_updates <= --rl_steps (reduce both for smoke tests)")
    if args.audit_prefixes and args.full_training_session:
        raise ValueError("Use --no-full_training_session for a held-out prefix audit")
    if not args.audit_prefixes and args.rl_steps == 0:
        raise ValueError("Zero RL steps are only supported with --audit_prefixes")
    if args.compare_prefix_training and (not args.audit_prefixes or args.rl_steps != 0):
        raise ValueError("Use --audit_prefixes --rl_steps 0 for the prefix-training comparison")
    if not 0 <= args.replay_warmup_epsilon <= 1:
        raise ValueError("replay_warmup_epsilon must be in [0, 1]")
    torch.set_num_threads(args.cpu_threads)
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "run_config.json"
    config = {key: value for key, value in vars(args).items() if key != "output_dir"}
    if not args.audit_prefixes:
        config.pop("audit_prefixes")
    if not args.compare_prefix_training:
        config.pop("compare_prefix_training")
    else:
        config["prefix_comparison_version"] = 2
    config["protocol_version"] = PROTOCOL_VERSION
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
            if args.audit_prefixes:
                pd.DataFrame(rows).to_csv(results_path, index=False)
            else:
                write_summary(rows, root, set(args.subjects))
    print(f"DONE rows={len(rows)} output={root}", flush=True)


if __name__ == "__main__":
    main()
