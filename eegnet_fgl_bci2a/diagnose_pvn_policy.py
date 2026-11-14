"""Validation-only prefix curve and stopping audit for a PVN-LBRL checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

from models.pvn_lbrl import PVNLBRL
from run_pvn_lbrl_ablation import PROTOCOL_VERSION, complete_online_samples, load_split


@torch.no_grad()
def validation_diagnostics(
    model: PVNLBRL, X_val: torch.Tensor, y_val: torch.Tensor,
    sample_rate: int = 250,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    model.eval()
    device = next(model.parameters()).device
    x = X_val.to(device)
    beliefs, _ = model.forward_sequence(x)
    labels = y_val.cpu().numpy()
    rows = []
    for step in range(beliefs.shape[1]):
        predictions = beliefs[:, step].argmax(dim=-1).cpu().numpy()
        rows.append({
            "slices": step + 1,
            "observation_s": (step + 1) * model.slice_samples / sample_rate,
            "accuracy": accuracy_score(labels, predictions),
            "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
            "kappa": cohen_kappa_score(labels, predictions),
        })
    policy_predictions, lengths = model.predict_until_halt(x)
    stopped = policy_predictions.cpu().numpy()
    lengths = lengths.cpu().numpy()
    halt_distribution = pd.Series(lengths).value_counts().sort_index().rename_axis("slices").reset_index(name="trials")
    halt_distribution["observation_s"] = halt_distribution.slices * model.slice_samples / sample_rate
    policy = {
        "accuracy": float(accuracy_score(labels, stopped)),
        "macro_f1": float(f1_score(labels, stopped, average="macro", zero_division=0)),
        "kappa": float(cohen_kappa_score(labels, stopped)),
        "mean_observation_s": float(lengths.mean() * model.slice_samples / sample_rate),
    }
    return pd.DataFrame(rows), halt_distribution, policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("protocol_version") != PROTOCOL_VERSION:
        parser.error("Checkpoint uses a different PVN architecture; use its original code version")
    config = argparse.Namespace(**checkpoint["config"])
    if config.full_training_session:
        parser.error("This checkpoint used the entire training session and has no held-out validation set")
    config.data_dir = args.data_dir
    subject = int(checkpoint["subject"])
    seed = int(checkpoint["seed"])
    X_train, _, X_val, y_val, _, _ = load_split(config, subject, seed)
    online_samples = complete_online_samples(X_train.shape[-1], config.slice_samples)
    model = PVNLBRL(
        channels=X_train.shape[1], classes=4, slice_samples=config.slice_samples,
    ).to(args.device)
    model.load_state_dict(checkpoint["model_state"])
    prefix, halts, policy = validation_diagnostics(model, X_val[..., :online_samples], y_val)
    expected = checkpoint.get("validation", {})
    if "accuracy" in expected and abs(policy["accuracy"] - expected["accuracy"]) > 1e-6:
        raise ValueError("Validation accuracy disagrees with checkpoint metadata; verify data and code versions")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prefix.to_csv(output / "validation_prefix_curve.csv", index=False)
    halts.to_csv(output / "validation_halt_distribution.csv", index=False)
    print("Validation policy:", policy)
    print("Fixed-prefix validation curve:")
    print(prefix.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("Validation halt counts:")
    print(halts.to_string(index=False))
    print(f"Saved validation-only diagnostics to {output.resolve()}")


if __name__ == "__main__":
    main()
