"""Budget-matched GRU prefix sampling audit. No Q training or test scoring."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch
from torch.nn import functional as F

from models.pvn_lbrl import PVNLBRL
from pvn_lbrl_training import pretrain_belief
from run_pvn_joint_learning import VERSION as SOURCE_VERSION
from run_pvn_joint_learning import restore_rng, rng_state
from run_pvn_lbrl_ablation import load_split, seed_all

VERSION = "pvn_prefix_sampling_ab_v1"


def load_source(source):
    config = json.loads((source / "run_config.json").read_text(encoding="utf-8"))
    expected = {"version": SOURCE_VERSION, "dataset": "bci2a", "slice_samples": 40,
                "preprocessing_mode": "moabb_zero_phase", "fmin": 1., "fmax": 48.,
                "scale": 1., "train_zscore": False, "full_training_session": False}
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError("Source must be the joint-learning 40-point 1-48Hz holdout pretraining")
    path = source / "pretrained.pth"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("version") != SOURCE_VERSION:
        raise ValueError("Source checkpoint version mismatch")
    encoder = {k.removeprefix("encoder."): v for k, v in checkpoint["state"].items()
               if k.startswith("encoder.")}
    if not encoder:
        raise ValueError("Source checkpoint has no PVN")
    return config, encoder, hashlib.sha256(path.read_bytes()).hexdigest()


@torch.no_grad()
def compare_curve(model, x, y, batch_size, split, arm):
    """Vote on identical slice features, or independently encode contiguous prefixes."""
    model.eval()
    steps = x.shape[-1] // model.slice_samples
    device = next(model.parameters()).device
    hits = torch.zeros(3, steps, dtype=torch.long, device=device)
    for batch, labels in zip(x.split(batch_size), y.split(batch_size)):
        batch, labels = batch.to(device), labels.to(device)
        encoded = model.encode_sequence(batch)
        beliefs, _ = model.forward_encoded_sequence(encoded)
        hits[0] += (beliefs.argmax(-1) == labels[:, None]).sum(0)
        point_classes = encoded.reshape(len(batch), steps, model.classes,
                                        model.slice_samples).argmax(2)
        votes = F.one_hot(point_classes, model.classes).sum(2).cumsum(1)
        hits[1] += (votes.argmax(-1) == labels[:, None]).sum(0)
        for step in range(steps):
            prediction = model.encoder.predict_vote(batch[..., :(step + 1) * model.slice_samples])
            hits[2, step] += (prediction == labels).sum()
    values = (hits.float() / len(y)).cpu().tolist()
    return [{"arm": arm, "split": split, "n_trials": len(y), "slices": step + 1,
             "observation_s": (step + 1) * model.slice_samples / 250,
             "gru_accuracy": values[0][step], "segmented_pvn_accuracy": values[1][step],
             "contiguous_pvn_accuracy": values[2][step]} for step in range(steps)]


def fit_arms(base, train, y, val, yv, *, epochs, batch_size, seed, root):
    """Restore RNG before B; its extra mask permutation uses an independent generator."""
    device = next(base.parameters()).device
    seed_all(seed)
    shared_rng = rng_state(device)
    results, logs, first_budgets = [], [], None
    encoder_before = {k: v.detach().clone() for k, v in base.encoder.state_dict().items()}
    for arm in ("random_halt", "balanced_prefix"):
        candidate = copy.deepcopy(base)
        restore_rng(shared_rng)
        generator = torch.Generator(device=device).manual_seed(seed) if arm == "balanced_prefix" else None
        counts = torch.zeros(train.shape[-1] // base.slice_samples, dtype=torch.long, device=device)
        budgets = []
        print(f"ARM {arm} | epochs={epochs} | train={len(y)} val={len(yv)}", flush=True)
        pretrain_belief(candidate, train, y, val, yv, epochs=epochs, batch_size=batch_size,
                        balanced_prefix_generator=generator, supervision_counts=counts,
                        supervision_budgets=budgets)
        if first_budgets is None:
            first_budgets = budgets
        elif budgets != first_budgets:
            raise RuntimeError("Per-batch, per-trial supervision budgets differ between arms")
        for name, value in candidate.encoder.state_dict().items():
            if not torch.equal(value, encoder_before[name]):
                raise RuntimeError("Frozen PVN changed")
        for split, x, labels in (("train", train, y), ("validation", val, yv)):
            rows = compare_curve(candidate, x, labels, batch_size, split, arm)
            results.extend(rows)
            print(f"CURVE {arm} {split} | GRU first={rows[0]['gru_accuracy']:.4f} "
                  f"last={rows[-1]['gru_accuracy']:.4f} | "
                  f"segmented_last={rows[-1]['segmented_pvn_accuracy']:.4f} | "
                  f"contiguous_last={rows[-1]['contiguous_pvn_accuracy']:.4f}", flush=True)
        logs.extend({"arm": arm, "slices": i + 1, "observation_s": (i + 1) * base.slice_samples / 250,
                     "supervised_targets": n} for i, n in enumerate(counts.cpu().tolist()))
        torch.save({"version": VERSION, "state": candidate.state_dict()}, root / f"{arm}.pth")
        pd.DataFrame(results).to_csv(root / "prefix_curves.csv", index=False)
        pd.DataFrame(logs).to_csv(root / "supervision_counts.csv", index=False)
    return pd.DataFrame(results), pd.DataFrame(logs)


def run(args):
    source, root = Path(args.source_dir).resolve(), Path(args.output_dir).resolve()
    if root == source or (root.exists() and any(root.iterdir())):
        raise FileExistsError("Use a new empty output directory, separate from source")
    config, encoder, digest = load_source(source)
    data_args = argparse.Namespace(**{k: v for k, v in config.items() if k != "version"})
    data_args.data_dir, data_args.device = args.data_dir, args.device
    seed_all(config["seed"])
    # Loader returns the separate test session, but neither test inputs nor labels are scored.
    train, y, val, yv, _, _ = load_split(data_args, config["subject"], config["seed"])
    if not len(yv) or train.shape[-1] != 1000 or val.shape[-1] != 1000:
        raise ValueError("Expected 4s/1000-point trials and a nonempty holdout")
    seed_all(args.init_seed)
    base = PVNLBRL(channels=train.shape[1], classes=4, slice_samples=40).to(args.device)
    # Discard the source's already-trained GRU. Only transfer its frozen PVN.
    base.encoder.load_state_dict(encoder, strict=True)
    base.encoder.requires_grad_(False)
    base.eval()
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"version": VERSION, **vars(args), "source_checkpoint_sha256": digest,
                "source_config": config, "train_trials": len(y), "val_trials": len(yv),
                "note": "Balanced arm is a local control, not strict paper reproduction; no Q/KD/test"}
    (root / "run_config.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    torch.save({"version": VERSION, "state": base.state_dict()}, root / "shared_initial.pth")
    curves, counts = fit_arms(base, train, y, val, yv, epochs=args.epochs,
                              batch_size=args.batch_size, seed=args.fit_seed, root=root)
    totals = counts.groupby("arm").supervised_targets.sum()
    if totals.nunique() != 1:
        raise RuntimeError("Supervision totals differ")
    # Prespecified 1.92s and 4s endpoints, not best validation-prefix selection.
    summary = curves[(curves.split == "validation") & curves.slices.isin([12, 25])]
    summary.to_csv(root / "validation_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    print(f"DONE | matched supervision totals={totals.to_dict()} | no Q or test scoring", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--data_dir", default="data")
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--init_seed", type=int, default=42)
    p.add_argument("--fit_seed", type=int, default=42)
    args = p.parse_args()
    if min(args.epochs, args.batch_size) < 1:
        p.error("epochs and batch_size must be positive")
    torch.set_num_threads(1)
    run(args)


if __name__ == "__main__":
    main()
