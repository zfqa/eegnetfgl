"""BCI2a baseline preserving supplied mmc2 core, with explicit local input adapter.

Not a verified reproduction of90.28%: preprocessing helpers, epoch selection,
seeds and full launch config were not supplied. No KD, students, or teachers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from models.edgcn import _region_ids
from models.edgcn_supplement import EDGCNModule, objective
from utils.data_loader import configure_mne_data_path, load_bci2a_subject, split_session_train_val_test
from utils.metrics import evaluate_metrics
from utils.seed import set_seed

VERSION = "edgcn_supplement_core_local_inputs_v1"
REFERENCE = Path(__file__).resolve().parent / "references" / "edgcn_supplement"


class LocalAuxiliaryInputs(nn.Module):
    """Missing author helpers: chosen PSD bands/grid and broadband PLV/regional mask.

    Author model receives externally supplied features; these are adaptations,
    not recovered author preprocessing. No learned parameters or label access.
    """

    def __init__(self, bands, window_s=.5, mask_probability=.1, threshold=.5, log_psd=True):
        super().__init__()
        self.bands, self.window_s = bands, window_s
        self.mask_probability, self.threshold, self.log_psd = mask_probability, threshold, log_psd
        regions = _region_ids(22)
        self.register_buffer("same_region", regions[:, None] == regions[None, :])

    def psd(self, x, fs):
        width = min(x.shape[-1], max(8, round(fs * self.window_s)))
        frames = x.unfold(-1, width, max(1, width // 2))
        window = torch.hann_window(width, device=x.device, dtype=x.dtype)
        spectrum = torch.fft.rfft(frames * window, dim=-1).abs().square()
        spectrum = spectrum / (fs * window.square().sum()).clamp_min(1e-8)
        spectrum[..., 1:-1 if width % 2 == 0 else None] *= 2
        hz = torch.fft.rfftfreq(width, 1 / fs).to(x.device)
        bands = []
        for lo, hi in self.bands:
            selected = (hz >= lo) & (hz < hi)
            if not selected.any():
                raise ValueError("A PSD band has no FFT bins; change documented window/band settings")
            bands.append(spectrum[..., selected].mean(-1))
        values = torch.stack(bands, -1).permute(0, 2, 1, 3).flatten(2)
        return torch.log1p(values) if self.log_psd else values

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (22, 1125):
            raise ValueError("Source baseline expects [B,22,1125]")
        up = x.new_empty(*x.shape[:-1], 2 * x.shape[-1] - 1)
        up[..., ::2] = x
        up[..., 1::2] = .5 * (x[..., :-1] + x[..., 1:])
        if self.training:
            up[..., 1::2] *= torch.rand_like(up[..., 1::2]) >= self.mask_probability
        psd_up, psd_down = self.psd(up, 500), self.psd(x[..., ::2], 125)
        n = x.shape[-1]
        multiplier = x.new_zeros(n)
        multiplier[0], multiplier[1:(n + 1) // 2] = 1, 2
        if n % 2 == 0:
            multiplier[n // 2] = 1
        analytic = torch.fft.ifft(torch.fft.fft(x, dim=-1) * multiplier, dim=-1)
        phase = analytic / analytic.abs().clamp_min(1e-6)
        plv = (phase @ phase.conj().transpose(-1, -2) / n).abs().clamp(0, 1)
        return psd_up, psd_down, plv * self.same_region, (plv > self.threshold).to(x.dtype)


def source_hashes():
    return {name: hashlib.sha256((REFERENCE / name).read_bytes()).hexdigest()
            for name in ("mmc1.py", "mmc2.py")}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", required=True, choices=["fit", "test"])
    p.add_argument("--subject", type=int, choices=range(1, 10), default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_dir", default="data")
    p.add_argument("--output_dir", default="results/edgcn_supplement_s1_local_adapter")
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--val_ratio", type=float, default=.2)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=.0009)
    p.add_argument("--weight_decay", type=float, default=0.)
    p.add_argument("--standardize", action="store_true")
    p.add_argument("--psd_bands", type=float, nargs=8, default=[.5, 4, 4, 8, 8, 13, 13, 38])
    p.add_argument("--psd_window_s", type=float, default=.5)
    p.add_argument("--mask_probability", type=float, default=.1)
    p.add_argument("--graph_threshold", type=float, default=.5)
    p.add_argument("--psd_transform", choices=["log1p", "none"], default="log1p")
    a = p.parse_args()
    if min(a.epochs, a.patience, a.batch_size) < 1 or a.lr <= 0 or a.weight_decay < 0:
        p.error("Invalid training counts or optimizer settings")
    if not 0 < a.val_ratio < 1 or not 0 <= a.mask_probability <= 1 or not 0 <= a.graph_threshold <= 1:
        p.error("Invalid split/mask/threshold")
    if a.psd_window_s <= 0 or any(not 0 <= lo < hi <= 62.5 for lo, hi in zip(a.psd_bands[::2], a.psd_bands[1::2])):
        p.error("Invalid PSD bands/window")
    return a


def config(a):
    excluded = {"stage", "device", "data_dir", "output_dir"}
    return {"version": VERSION, "source_sha256": source_hashes(),
            "protocol": "2a_session_split_0.5-38Hz_minus0.5_to4s_1125samples",
            "auxiliary_inputs": "local_adapter_not_author_helpers",
            **{k: v for k, v in vars(a).items() if k not in excluded}}


def prepare_data(a):
    cache = Path(a.data_dir).resolve()
    for split in ("T", "E"):
        path = cache / "MNE-bnci-data" / "~bci" / "database" / "001-2014" / f"A{a.subject:02d}{split}.mat"
        if not path.is_file():
            raise FileNotFoundError(f"Missing cache {path}; refusing automatic download")
    os.environ["MNE_DATA"] = os.environ["MNE_DATASETS_BNCI_PATH"] = str(cache)
    configure_mne_data_path(cache)
    x, y, metadata = load_bci2a_subject(a.subject, fmin=.5, fmax=38., tmin=-.5, tmax=4.)
    if x.shape[1] != 22 or x.shape[-1] not in (1125, 1126):
        raise ValueError(f"Expected22 channels and1125/1126 points, got {x.shape}")
    # Half-open interval: discard only the inclusive right endpoint if present.
    x = x[..., :1125].astype(np.float32)
    train, labels, val, yv, test, yt = split_session_train_val_test(x, y, metadata,
                                                                 val_ratio=a.val_ratio, seed=a.seed)
    signature = hashlib.sha256()
    for values in (train, labels, val, yv):
        signature.update(np.ascontiguousarray(values).tobytes())
    mean = train.mean((0, 2), keepdims=True) if a.standardize else np.zeros((1, 22, 1), np.float32)
    std = train.std((0, 2), keepdims=True) + 1e-6 if a.standardize else np.ones((1, 22, 1), np.float32)
    print(f"SPLIT train={len(labels)} validation={len(yv)} test={len(yt)} "
          "shape=(22,1125) band=0.5-38Hz window=-0.5..4s; local input adapter", flush=True)
    return ((train - mean) / std, labels, (val - mean) / std, yv,
            test, yt, mean, std, signature.hexdigest())


def batches(x, y, a, shuffle=False):
    return DataLoader(TensorDataset(torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.long)),
                      batch_size=a.batch_size, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(a.seed))


@torch.no_grad()
def predict(model, adapter, x, y, a):
    model.eval(); adapter.eval()
    logits = []
    for xb, _ in batches(x, y, a):
        xb = xb.to(a.device)
        up, down, local, glob = adapter(xb)
        logits.append(model(xb, up, down, None, local, glob).cpu())
    return torch.cat(logits)


def atomic_save(value, path):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def run(a):
    root = Path(a.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "run_config.json"
    settings = config(a)
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")) != settings:
            raise ValueError("Changed settings require a NEW output directory")
    elif a.stage == "test":
        raise FileNotFoundError("Fit and lock a model first")
    else:
        manifest.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    path = root / "locked_model.pth"
    if a.stage == "fit" and path.exists():
        raise FileExistsError("Already completed; use test stage, not refitting over locked results")
    set_seed(a.seed)
    train, y, val, yv, test, yt, mean, std, signature = prepare_data(a)
    model = EDGCNModule().to(a.device)
    adapter = LocalAuxiliaryInputs(list(zip(a.psd_bands[::2], a.psd_bands[1::2])),
        a.psd_window_s, a.mask_probability, a.graph_threshold, a.psd_transform == "log1p").to(a.device)
    if a.stage == "test":
        cp = torch.load(path, map_location="cpu", weights_only=True)
        if cp["config"] != settings or cp["data_signature"] != signature:
            raise ValueError("Checkpoint protocol/data mismatch")
        model.load_state_dict(cp["state"])
        test = (test - cp["mean"].numpy()) / cp["std"].numpy()
        logits = predict(model, adapter, test, yt, a)
        pred = logits.argmax(-1).numpy()
        result = {"subject": a.subject, "seed": a.seed, **evaluate_metrics(yt, pred),
                  "reference_s1_accuracy": .9028 if a.subject == 1 else None, "protocol": VERSION}
        pd.DataFrame([result]).to_csv(root / "test_results.csv", index=False)
        pd.DataFrame({"trial_index": np.arange(len(yt)), "y_true": yt, "y_pred": pred}).to_csv(root / "test_predictions.csv", index=False)
        print(f"TEST {result}; not verified author preprocessing", flush=True)
        return
    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    data = batches(train, y, a, shuffle=True)
    best, rank, stale, history = None, None, 0, []
    for epoch in range(1, a.epochs + 1):
        model.train(); adapter.train()
        total, count = 0., 0
        for xb, labels in data:
            xb, labels = xb.to(a.device), labels.to(a.device)
            up, down, local, glob = adapter(xb)
            optimizer.zero_grad(set_to_none=True)
            loss = objective(model, model(xb, up, down, None, local, glob), labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss")
            loss.backward(); optimizer.step()
            total += float(loss.detach()) * len(labels); count += len(labels)
        logits = predict(model, adapter, val, yv, a)
        accuracy = float((logits.argmax(-1) == torch.as_tensor(yv)).float().mean())
        nll = float(F.cross_entropy(logits, torch.as_tensor(yv)))
        if not np.isfinite(nll):
            raise FloatingPointError("Nonfinite validation loss")
        current = (accuracy, -nll)
        if rank is None or current > rank:
            rank, stale = current, 0
            best = {"config": settings, "data_signature": signature, "selected_epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    "mean": torch.from_numpy(mean), "std": torch.from_numpy(std),
                    "val_accuracy": accuracy, "val_nll": nll}
        else:
            stale += 1
        history.append({"epoch": epoch, "train_loss": total / count, "val_accuracy": accuracy, "val_nll": nll})
        pd.DataFrame(history).to_csv(root / "history.csv", index=False)
        if epoch == 1 or epoch % 10 == 0 or stale >= a.patience or epoch == a.epochs:
            print(f"epoch={epoch}/{a.epochs} loss={total/count:.4f} val_acc={accuracy:.4f} val_nll={nll:.4f}", flush=True)
        if stale >= a.patience:
            break
    atomic_save(best, path)
    pd.DataFrame([{k: best[k] for k in ("selected_epoch", "val_accuracy", "val_nll")}]).to_csv(root / "validation_results.csv", index=False)
    print(f"LOCKED selected_epoch={best['selected_epoch']} val_accuracy={best['val_accuracy']:.4f}; test NOT scored", flush=True)


def main():
    a = parse_args()
    torch.set_num_threads(1)
    run(a)


if __name__ == "__main__":
    main()
