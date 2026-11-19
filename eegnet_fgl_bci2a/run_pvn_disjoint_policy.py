"""Disjoint classifier/policy training diagnostic; not exact PVN-LBRL reproduction."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from models.pvn_lbrl import PVNLBRL
from pvn_lbrl_training import (
    Episode, PrioritizedEpisodeReplay, actor_epsilon, encode_trials,
    pretrain_belief, pretrain_encoder, q_update,
)
from run_pvn_lbrl_ablation import load_split, seed_all
from run_pvn_threshold_baseline import metrics

VERSION = "pvn_disjoint_frozen_classifier_v1"
SAMPLES, STEPS = 40, 12


def split_training(x, y, seed):
    """Partition the old training subset (230) into classifier (172)/policy (58)."""
    fit, policy = train_test_split(np.arange(len(y)), test_size=0.25,
                                   stratify=y.numpy(), random_state=seed)
    assert not set(fit) & set(policy)
    return x[fit], y[fit], x[policy], y[policy], fit, policy


def freeze_classifier(model):
    model.requires_grad_(False)
    model.q_network.requires_grad_(True)
    model.eval()


@torch.no_grad()
def prefix_outputs(model, x, batch_size):
    encoded = encode_trials(model, x[..., :STEPS * SAMPLES], batch_size)
    beliefs, qs = [], []
    device = next(model.parameters()).device
    for batch in encoded.split(batch_size):
        belief, q = model.forward_encoded_sequence(batch.to(device))
        beliefs.append(belief.cpu())
        qs.append(q.cpu())
    return encoded, torch.cat(beliefs), torch.cat(qs)


def choose(beliefs, q, mode, threshold=math.inf):
    if mode == "fixed":
        halt = torch.zeros(beliefs.shape[:2], dtype=torch.bool)
    elif mode == "threshold":
        halt = beliefs.max(-1).values >= threshold
    elif mode == "q":
        halt = q.argmax(-1) == PVNLBRL.HALT
    else:
        raise ValueError(mode)
    halt[:, -1] = True
    index = halt.long().argmax(1)
    prediction = beliefs[torch.arange(len(index)), index].argmax(-1)
    return prediction.numpy(), (index.numpy() + 1) * SAMPLES


@torch.no_grad()
def predict_stream(model, x, mode, threshold=math.inf):
    """Actual inference: encode only the currently available slice of active trials."""
    model.eval()
    device = next(model.parameters()).device
    predictions, endpoints = [], []
    for trial in x:
        hidden = None
        for step in range(STEPS):
            piece = trial[:, step * SAMPLES:(step + 1) * SAMPLES][None].to(device)
            previous = piece.new_full((1,), PVNLBRL.WAIT if step else PVNLBRL.HALT)
            hidden, belief, q = model.step(piece, hidden, previous, piece.new_zeros(1))
            halt = (mode == "threshold" and float(belief.max()) >= threshold) or (
                mode == "q" and int(q.argmax(-1)) == PVNLBRL.HALT
            )
            if halt or step == STEPS - 1:
                predictions.append(int(belief.argmax(-1)))
                endpoints.append((step + 1) * SAMPLES)
                break
    return np.asarray(predictions), np.asarray(endpoints)


def score(beliefs, q, y, mode, threshold=math.inf):
    prediction, times = choose(beliefs, q, mode, threshold)
    return metrics(y.numpy(), prediction, times)


def select_threshold(beliefs, y):
    rows = []
    for tau in [math.inf, .5, .6, .7, .8, .9, .95, .99]:
        row = score(beliefs, None, y, "threshold", tau)
        rows.append({"threshold": tau, **row})
    feasible = [r for r in rows if r["accuracy"] >= rows[0]["accuracy"]]
    best = min(feasible, key=lambda r: (r["mean_observation_s"], -r["accuracy"], -r["threshold"]))
    return best["threshold"], pd.DataFrame(rows)


def checkpoint_rank(row, baseline):
    """Feasible: minimize time; otherwise retain highest accuracy. No test selection."""
    feasible = row["accuracy"] >= baseline
    return (feasible, -row["mean_observation_s"], row["accuracy"]) if feasible else (
        feasible, row["accuracy"], -row["mean_observation_s"])


@torch.no_grad()
def collect_frozen(model, encoded, labels, count, generator, epsilon=None, actor_offset=0):
    """Classification reward is stationary; the last slice is always HALT."""
    device = next(model.parameters()).device
    episodes = []
    for actor in range(count):
        index = int(torch.randint(len(labels), (1,), generator=generator))
        sequence = encoded[index]
        hidden, actions, rewards = None, [], []
        eps = actor_epsilon((actor + actor_offset) % 8, 8, "r2d2_standard") if epsilon is None else epsilon
        for step in range(len(sequence)):
            previous = sequence.new_full((1,), PVNLBRL.WAIT if step else PVNLBRL.HALT).to(device)
            hidden, belief, q = model.step_encoded(sequence[step:step + 1].to(device),
                                                   hidden, previous)
            action = int(q.argmax(-1))
            if float(torch.rand((), generator=generator)) < eps:
                action = int(torch.randint(2, (), generator=generator))
            if step == len(sequence) - 1:
                action = PVNLBRL.HALT
            reward = (1. if int(belief.argmax(-1)) == int(labels[index]) else -1.) if action == 0 else 0.
            actions.append(action)
            rewards.append(reward)
            if action == PVNLBRL.HALT:
                break
        episodes.append(Episode(sequence[:len(actions)], int(labels[index]), actions, rewards))
    return episodes


def fit_q(model, x_policy, y_policy, x_val, y_val, args, folder):
    freeze_classifier(model)
    snapshot = {n: p.detach().clone() for n, p in model.named_parameters()
                if not n.startswith("q_network.")}
    encoded, _, _ = prefix_outputs(model, x_policy, args.batch_size)
    _, val_beliefs, val_q = prefix_outputs(model, x_val, args.batch_size)
    baseline = score(val_beliefs, val_q, y_val, "fixed")["accuracy"]
    target = copy.deepcopy(model).eval()
    optimizer = torch.optim.AdamW(model.q_network.parameters(), lr=1e-4)
    replay = PrioritizedEpisodeReplay(capacity=100_000)
    generator = torch.Generator().manual_seed(args.seed)
    for episode in collect_frozen(model, encoded, y_policy, args.warmup, generator, epsilon=1.):
        replay.add(episode)
    best_state, best_rank, best_step = None, None, None
    log = []
    for step in range(args.q_steps + 1):
        if step:
            for ep in collect_frozen(model, encoded, y_policy, 1, generator, actor_offset=step - 1):
                replay.add(ep)
            q_update(model, target, replay, optimizer, args.gamma, args.replay_batch_size)
            if step % 100 == 0:
                target.load_state_dict(model.state_dict())
        if step % args.validate_every == 0 or step == args.q_steps:
            _, _, q = prefix_outputs(model, x_val, args.batch_size)
            row = score(val_beliefs, q, y_val, "q")
            rank = checkpoint_rank(row, baseline)
            if best_rank is None or rank > best_rank:
                best_state = copy.deepcopy(model.q_network.state_dict())
                best_rank, best_step = rank, step
            log.append({"q_step": step, "val_fixed_accuracy": baseline, **row})
            pd.DataFrame(log).to_csv(folder / "q_validation_history.csv", index=False)
            print(f"Q step={step}/{args.q_steps} validation={row}", flush=True)
    model.q_network.load_state_dict(best_state)
    for name, p in model.named_parameters():
        if name in snapshot and not torch.equal(p.detach(), snapshot[name]):
            raise RuntimeError(f"Classifier changed during Q fitting: {name}")
    return best_step


def load_parts(args):
    x, y, xv, yv, xt, yt = load_split(args, args.subject, args.seed)
    xc, yc, xp, yp, fit_indices, policy_indices = split_training(x, y, args.seed)
    return xc, yc, xp, yp, xv, yv, xt, yt, fit_indices, policy_indices


def run(args, root):
    parts = load_parts(args)
    xc, yc, xp, yp, xv, yv, xt, yt, fit_indices, policy_indices = parts
    model = PVNLBRL(channels=xc.shape[1], classes=4, slice_samples=SAMPLES).to(args.device)
    checkpoint_path = root / "locked_model.pth"
    if args.stage == "test":
        checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=True)
        if checkpoint["version"] != VERSION:
            raise ValueError("Checkpoint version mismatch")
        model.load_state_dict(checkpoint["state"])
        tau = math.inf if checkpoint["threshold"] is None else checkpoint["threshold"]
        rows = []
        for mode in ("fixed", "threshold", "q"):
            prediction, times = predict_stream(model, xt, mode, tau)
            row = {"subject": args.subject, "seed": args.seed, "method": mode,
                   **metrics(yt.numpy(), prediction, times)}
            rows.append(row)
            pd.DataFrame({"trial_index": np.arange(len(yt)), "y_true": yt.numpy(),
                          "y_pred": prediction, "observation_s": times / 250}).to_csv(
                              root / f"{mode}_test_predictions.csv", index=False)
            print(f"TEST {row}", flush=True)
        pd.DataFrame(rows).to_csv(root / "test_results.csv", index=False)
        return
    if checkpoint_path.exists():
        raise ValueError("Pilot already locked; use --stage test or a new output directory")
    np.savez(root / "split_indices.npz", classifier=fit_indices, policy=policy_indices)
    print(f"SPLIT classifier={len(yc)} policy={len(yp)} validation={len(yv)} test={len(yt)}",
          flush=True)
    pretrained = root / "pretrained_classifier.pth"
    if pretrained.exists():
        model.load_state_dict(torch.load(pretrained, map_location=args.device, weights_only=True))
    else:
        # Fixed epochs, no checkpoint choice using the policy-training split.
        pretrain_encoder(model, xc, yc, xv, yv, epochs=args.epochs_pvn, batch_size=args.batch_size)
        pretrain_belief(model, xc[..., :480], yc, xv[..., :480], yv,
                        epochs=args.epochs_belief, batch_size=args.batch_size,
                        balanced_prefix_generator=torch.Generator(device=args.device).manual_seed(args.seed))
        torch.save(model.state_dict(), pretrained)
    prefix_rows = []
    for name, x, y in (("classifier", xc, yc), ("policy", xp, yp), ("validation", xv, yv)):
        _, b, q = prefix_outputs(model, x, args.batch_size)
        for step in range(STEPS):
            prefix_rows.append({"split": name, "observation_s": (step + 1) * .16,
                                "accuracy": float((b[:, step].argmax(-1) == y).float().mean())})
    pd.DataFrame(prefix_rows).to_csv(root / "pre_rl_prefix_accuracy.csv", index=False)
    for name in ("classifier", "policy", "validation"):
        records = [r for r in prefix_rows if r["split"] == name]
        print(f"PREFIX {name} first={records[0]['accuracy']:.4f} "
              f"last={records[-1]['accuracy']:.4f}", flush=True)
    if args.stage == "pretrain":
        print("PRETRAIN DONE; review pre_rl_prefix_accuracy.csv before --stage pilot; no Q or test",
              flush=True)
        return
    _, b, q = prefix_outputs(model, xv, args.batch_size)
    tau, candidates = select_threshold(b, yv)
    candidates.to_csv(root / "threshold_validation.csv", index=False)
    best_step = fit_q(model, xp, yp, xv, yv, args, root)
    _, b_after, q = prefix_outputs(model, xv, args.batch_size)
    torch.testing.assert_close(b, b_after, atol=0, rtol=0)
    validation = [{"method": mode, **score(b_after, q, yv, mode, tau)}
                  for mode in ("fixed", "threshold", "q")]
    pd.DataFrame(validation).to_csv(root / "validation_results.csv", index=False)
    for row in validation:
        print(f"VALIDATION {row}", flush=True)
    torch.save({"version": VERSION, "state": model.state_dict(),
                "threshold": None if math.isinf(tau) else tau,
                "selected_q_step": best_step}, checkpoint_path)
    print(f"LOCKED threshold={tau} selected_q_step={best_step}; test NOT scored", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage", choices=["pretrain", "pilot", "test"], default="pretrain")
    parser.add_argument("--epochs_pvn", type=int, default=300)
    parser.add_argument("--epochs_belief", type=int, default=300)
    parser.add_argument("--q_steps", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=256)
    parser.add_argument("--validate_every", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=.1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--replay_batch_size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    if min(args.epochs_pvn, args.epochs_belief, args.q_steps, args.warmup,
           args.validate_every, args.batch_size, args.replay_batch_size) < 1:
        parser.error("Training counts must be positive")
    if not 0 <= args.gamma < 1 or args.warmup > 8000:
        parser.error("Invalid gamma or warmup exceeds replay-safe capacity")
    args.dataset, args.preprocessing_mode = "bci2a", "moabb_zero_phase"
    args.full_training_session, args.train_zscore = False, False
    args.fmin, args.fmax, args.scale = 1., 48., 1.
    torch.set_num_threads(1)
    seed_all(args.seed)
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    config = {k: v for k, v in vars(args).items() if k not in {"stage", "output_dir", "device", "data_dir"}}
    config["version"] = VERSION
    manifest = root / "run_config.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != config:
            raise ValueError("Changed settings require a new directory")
    elif args.stage == "test":
        raise FileNotFoundError("Run pilot and lock the model first")
    else:
        manifest.write_text(json.dumps(config, indent=2), encoding="utf-8")
    run(args, root)


if __name__ == "__main__":
    main()
