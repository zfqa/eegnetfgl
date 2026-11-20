"""Staged paper-guided PVN/LDM/Q experiment; not author-code equivalence."""

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
from run_pvn_lbrl_ablation import complete_online_samples, load_split, prefix_audit, seed_all

VERSION = "pvn_joint_staged_v1"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["pretrain", "pilot", "test"], required=True)
    p.add_argument("--subject", type=int, default=1, choices=range(1, 10))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--slice_samples", type=int, choices=[40, 60, 80, 100], default=40)
    p.add_argument("--gamma", type=float, default=.1)
    p.add_argument("--epochs_pvn", type=int, default=300)
    p.add_argument("--epochs_belief", type=int, default=300)
    p.add_argument("--rl_steps", type=int, default=50_000)
    p.add_argument("--belief_updates", type=int, default=25)
    p.add_argument("--online_belief_epochs", type=int, default=20)
    p.add_argument("--actors", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--replay_batch_size", type=int, default=16)
    p.add_argument("--actor_sync_every", type=int, default=64)
    p.add_argument("--target_sync_every", type=int, default=500)
    p.add_argument("--validate_every", type=int, default=500)
    p.add_argument("--epsilon_schedule", choices=["paper_literal", "r2d2_standard"],
                   default="paper_literal")
    p.add_argument("--replay_warmup_episodes", type=int, default=0)
    p.add_argument("--replay_warmup_epsilon", type=float, default=1.)
    p.add_argument("--full_training_session", action="store_true")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()
    counts = [args.epochs_pvn, args.epochs_belief, args.rl_steps, args.belief_updates,
              args.online_belief_epochs, args.actors, args.batch_size,
              args.replay_batch_size, args.actor_sync_every, args.target_sync_every,
              args.validate_every]
    if min(counts) < 1 or args.belief_updates > args.rl_steps:
        p.error("Positive counts and belief_updates <= rl_steps are required")
    if not 0 <= args.gamma < 1 or not 0 <= args.replay_warmup_epsilon <= 1:
        p.error("Invalid discount or warm-up epsilon")
    if not 0 <= args.replay_warmup_episodes <= 1_000_000:
        p.error("Warm-up must fit replay capacity")
    args.dataset, args.preprocessing_mode = "bci2a", "moabb_zero_phase"
    args.fmin, args.fmax, args.scale, args.train_zscore = 1., 48., 1., False
    return args


def manifest(args, root):
    excluded = {"stage", "device", "data_dir", "output_dir"}
    config = {k: v for k, v in vars(args).items() if k not in excluded}
    config["version"] = VERSION
    path = root / "run_config.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != config:
            raise ValueError("Changed settings require a new output directory")
    elif args.stage != "pretrain":
        raise FileNotFoundError("Run pretrain in this directory first")
    else:
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def rng_state(device):
    n = np.random.get_state()
    return {"torch": torch.get_rng_state(), "python": random.getstate(),
            "numpy": [n[0], n[1].tolist(), n[2], n[3], n[4]],
            "cuda": torch.cuda.get_rng_state_all() if torch.device(device).type == "cuda" else []}


def restore_rng(state):
    torch.set_rng_state(state["torch"].cpu())
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    if state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def save_prefixes(model, train, y, val, yv, args, root, stage):
    rows = []
    for name, x, labels in (("train", train, y), ("validation", val, yv)):
        if not len(labels):
            continue
        curve = prefix_audit(model, x, labels, args.batch_size, stage, name)
        rows.extend(curve)
        print(f"PREFIX {stage} {name} first={curve[0]['belief_accuracy']:.4f} "
              f"last={curve[-1]['belief_accuracy']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(root / f"{stage}_prefix_accuracy.csv", index=False)


def run(args, root):
    seed_all(args.seed)
    train, y, val, yv, test, yt = load_split(args, args.subject, args.seed)
    count = complete_online_samples(train.shape[-1], args.slice_samples)
    online, online_val = train[..., :count], val[..., :count]
    model = PVNLBRL(channels=train.shape[1], classes=4,
                    slice_samples=args.slice_samples).to(args.device)
    pretrained, locked = root / "pretrained.pth", root / "locked_model.pth"
    if args.stage == "pretrain":
        if pretrained.exists() or locked.exists():
            raise FileExistsError("Pretraining already saved; use pilot or a new directory")
        pretrain_encoder(model, train, y, val, yv,
                         epochs=args.epochs_pvn, batch_size=args.batch_size)
        # No balanced-prefix replacement: use the paper-guided random-HALT objective.
        pretrain_belief(model, online, y, online_val, yv,
                        epochs=args.epochs_belief, batch_size=args.batch_size)
        save_prefixes(model, online, y, online_val, yv, args, root, "pre_rl")
        torch.save({"version": VERSION, "state": model.state_dict(),
                    "rng": rng_state(args.device)}, pretrained)
        print("PRETRAIN SAVED. Inspect pre_rl_prefix_accuracy.csv; no Q or test scoring.", flush=True)
        return
    if args.stage == "test":
        checkpoint = torch.load(locked, map_location="cpu", weights_only=True)
        if checkpoint["version"] != VERSION:
            raise ValueError("Checkpoint version mismatch")
        model.load_state_dict(checkpoint["state"])
        # Same locked classifier, with either learned stopping or the last prefix.
        from run_pvn_threshold_baseline import metrics
        from pvn_lbrl_training import encode_trials
        with torch.no_grad():
            model.eval()
            encoded = encode_trials(model, test[..., :count], args.batch_size)
            predictions = torch.cat([
                model.forward_encoded_sequence(b.to(args.device))[0][:, -1].argmax(-1).cpu()
                for b in encoded.split(args.batch_size)
            ]).numpy()
        fixed = metrics(yt.numpy(), predictions, np.full(len(yt), count))
        q = evaluate(model, test[..., :count], yt)
        rows = [{"subject": args.subject, "seed": args.seed, "method": "fixed",
                 "accuracy": fixed["accuracy"], "macro_f1": fixed["macro_f1"],
                 "kappa": fixed["kappa"], "mean_observation_s": fixed["mean_observation_s"]},
                {"subject": args.subject, "seed": args.seed, "method": "q",
                 "accuracy": q["accuracy"], "macro_f1": q["macro_f1"],
                 "kappa": q["kappa"], "mean_observation_s": q["mean_prediction_s"]}]
        pd.DataFrame(rows).to_csv(root / "test_results.csv", index=False)
        for row in rows:
            print(f"TEST {row}", flush=True)
        return
    if locked.exists() or (root / "pilot_started.json").exists():
        raise FileExistsError("Pilot already started; no mid-Q resume. Use a new directory for a new run")
    checkpoint = torch.load(pretrained, map_location="cpu", weights_only=True)
    if checkpoint["version"] != VERSION:
        raise ValueError("Checkpoint version mismatch")
    model.load_state_dict(checkpoint["state"])
    model.requires_grad_(True)
    model.encoder.requires_grad_(False)
    restore_rng(checkpoint["rng"])
    (root / "pilot_started.json").write_text(json.dumps({"version": VERSION}), encoding="utf-8")
    before = {k: v.detach().cpu().clone() for k, v in model.gru.state_dict().items()}
    validation = train_policy(
        model, online, y, online_val, yv, steps=args.rl_steps,
        actors=args.actors, gamma=args.gamma, batch_size=args.replay_batch_size,
        belief_updates=args.belief_updates, belief_epochs=args.online_belief_epochs,
        actor_sync_every=args.actor_sync_every, target_every=args.target_sync_every,
        validate_every=args.validate_every, epsilon_schedule=args.epsilon_schedule,
        replay_warmup_episodes=args.replay_warmup_episodes,
        replay_warmup_epsilon=args.replay_warmup_epsilon)
    changed = any(not torch.equal(before[k], v.detach().cpu()) for k, v in model.gru.state_dict().items())
    if not changed:
        raise RuntimeError("GRU weights did not change during joint learning; inspect belief updates")
    save_prefixes(model, online, y, online_val, yv, args, root, "post_rl")
    torch.save({"version": VERSION, "state": model.state_dict(), "validation": validation,
                "gru_weights_changed": changed}, locked)
    print("LOCKED final iterate; GRU weights changed; test NOT scored.", flush=True)


def main():
    args = parse_args()
    torch.set_num_threads(1)
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest(args, root)
    run(args, root)


if __name__ == "__main__":
    main()
