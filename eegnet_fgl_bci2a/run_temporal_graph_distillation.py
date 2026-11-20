"""4s temporal-probability + graph-relation guidance for a fixed2s Compact.

New research experiment, not EDGCN author-code/result equivalence.
Stages: teachers -> students (no test) -> test (locked models only).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from models.temporal_graph_distillation import (
    CompactRelationStudent, EDGCNGraphTeacher, MultiscaleTemporalTeacher,
    node_relations, relation_loss, temporal_kd,
)
from utils.data_loader import (
    EEGDataset, configure_mne_data_path, load_bci_subject_preprocessed,
    split_session_train_val_test,
)
from utils.metrics import evaluate_metrics
from utils.seed import set_seed
from utils.teacher_utils import fit_temperature_scaling

VERSION = "temporal_probability_graph_relation_v1"
METHODS = ("ce", "temporal", "spatial", "combined")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["teachers", "students", "test"], required=True)
    p.add_argument("--dataset", choices=["bci2a", "bci2b"], default="bci2a")
    p.add_argument("--subjects", type=int, nargs="+", default=[1])
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument("--data_dir", default="data")
    p.add_argument("--output_dir", default="results/temporal_graph_distillation_s1")
    p.add_argument("--device", default="cuda")
    p.add_argument("--fmin", type=float, default=8.)
    p.add_argument("--fmax", type=float, default=32.)
    p.add_argument("--epochs_teacher", type=int, default=300)
    p.add_argument("--epochs_student", type=int, default=300)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=.001)
    p.add_argument("--weight_decay", type=float, default=.0001)
    p.add_argument("--temporal_weight", type=float, default=.4)
    p.add_argument("--relation_weight", type=float, default=.2)
    p.add_argument("--temperature", type=float, default=1.)
    p.add_argument("--calibration", choices=["none", "temperature"], default="temperature")
    p.add_argument("--embedding_dim", type=int, default=16)
    p.add_argument("--hidden_dim", type=int, default=16)
    p.add_argument("--cheb_order", type=int, default=3)
    p.add_argument("--plv_threshold", type=float, default=.5)
    p.add_argument("--orth_weight", type=float, default=.01)
    p.add_argument("--require_cached_data", action=argparse.BooleanOptionalAction, default=True)
    a = p.parse_args()
    counts = [a.epochs_teacher, a.epochs_student, a.patience, a.batch_size,
              a.embedding_dim, a.hidden_dim, a.cheb_order]
    if min(counts) < 1 or not 0 < a.fmin < a.fmax < 125:
        p.error("Positive counts and 0<fmin<fmax<125 are required")
    if min(a.lr, a.temperature, a.orth_weight) <= 0 or min(a.weight_decay, a.temporal_weight, a.relation_weight) < 0:
        p.error("Invalid learning rate, temperature, regularizer or loss weight")
    if not 0 <= a.plv_threshold <= 1 or any(s not in range(1, 10) for s in a.subjects):
        p.error("Invalid PLV threshold or subjects")
    if len(set(a.subjects)) != len(a.subjects) or len(set(a.seeds)) != len(a.seeds) or len(set(a.methods)) != len(a.methods):
        p.error("Duplicate subjects, seeds or methods")
    return a


def atomic_save(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def config_for(a):
    omitted = {"stage", "device", "data_dir", "output_dir"}
    return {"version": VERSION, "preprocessing": "moabb_zero_phase",
            "teacher_samples": 1000, "student_samples": 500,
            **{k: v for k, v in vars(a).items() if k not in omitted}}


def guard_config(a, root):
    path, config = root / "run_config.json", config_for(a)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != config:
            raise ValueError("Changed experiment settings require a new output directory")
    elif a.stage != "teachers":
        raise FileNotFoundError("Run the teachers stage first")
    else:
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def fit_norm(train):
    return train.mean((0, 2), keepdims=True), train.std((0, 2), keepdims=True) + 1e-6


def apply_norm(x, norm):
    return ((x - norm[0]) / norm[1]).astype(np.float32)


def prepare_data(a, subject, seed):
    cache = Path(a.data_dir).resolve()
    if a.require_cached_data:
        folder, prefix = ("001-2014", "A") if a.dataset == "bci2a" else ("004-2014", "B")
        for split in ("T", "E"):
            path = cache / "MNE-bnci-data" / "~bci" / "database" / folder / f"{prefix}{subject:02d}{split}.mat"
            if not path.is_file():
                raise FileNotFoundError(f"Cached MAT missing: {path}; refusing automatic download")
    os.environ["MNE_DATA"] = os.environ["MNE_DATASETS_BNCI_PATH"] = str(cache)
    configure_mne_data_path(cache)
    x, y, metadata = load_bci_subject_preprocessed(a.dataset, subject,
        preprocessing_mode="moabb_zero_phase", fmin=a.fmin, fmax=a.fmax,
        tmin=0., tmax=4., data_dir=cache)
    xtr, ytr, xv, yv, xt, yt = split_session_train_val_test(x, y, metadata, seed=seed)
    if xtr.shape[-1] < 1000:
        raise ValueError("Expected at least4s at250Hz")
    xtr, xv, xt = (v[..., :1000].astype(np.float32) for v in (xtr, xv, xt))
    expected_channels, classes = (22, 4) if a.dataset == "bci2a" else (3, 2)
    if xtr.shape[1] != expected_channels:
        raise ValueError("Expected standard BNCI EEG channel order/count")
    tn, sn = fit_norm(xtr), fit_norm(xtr[..., :500])
    signature = hashlib.sha256()
    for values in (xtr, ytr, xv, yv):
        signature.update(np.ascontiguousarray(values).tobytes())
    print(f"SPLIT s{subject} seed{seed} train={len(ytr)} val={len(yv)} test={len(yt)} "
          f"teacher=4s student=0-2s band={a.fmin:g}-{a.fmax:g}Hz", flush=True)
    return {"teacher_train": apply_norm(xtr, tn), "teacher_val": apply_norm(xv, tn),
            "student_train": apply_norm(xtr[..., :500], sn), "student_val": apply_norm(xv[..., :500], sn),
            "raw_test": xt, "test_labels": yt, "y_train": ytr, "y_val": yv,
            "teacher_norm": tn, "student_norm": sn, "signature": signature.hexdigest(),
            "channels": expected_channels, "classes": classes, "subject": subject, "seed": seed}


def loader(x, y, batch_size, shuffle=False, seed=0):
    return DataLoader(EEGDataset(x, y, return_index=True), batch_size=batch_size,
                      shuffle=shuffle, generator=torch.Generator().manual_seed(seed))


@torch.no_grad()
def logits_for(model, x, y, a):
    model.eval()
    return torch.cat([model(xb.to(a.device)).cpu() for xb, _, _ in loader(x, y, a.batch_size)])


def fit_model(model, x, y, xv, yv, a, epochs, seed, objective, history_path):
    set_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    batches = loader(x, y, a.batch_size, True, seed)
    best, rank, stale, history = None, None, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        totals, seen = {}, 0
        for xb, labels, index in batches:
            optimizer.zero_grad(set_to_none=True)
            loss, parts = objective(model, xb.to(a.device), labels.to(a.device), index)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss in {history_path.stem}")
            loss.backward()
            optimizer.step()
            seen += len(labels)
            for name, value in parts.items():
                totals[name] = totals.get(name, 0.) + float(value.detach()) * len(labels)
        logits = logits_for(model, xv, yv, a)
        accuracy = float((logits.argmax(1) == torch.as_tensor(yv)).float().mean())
        nll = float(F.cross_entropy(logits, torch.as_tensor(yv)))
        if not np.isfinite(nll):
            raise FloatingPointError("Nonfinite validation loss")
        row = {"epoch": epoch, "val_accuracy": accuracy, "val_nll": nll,
               **{k: v / seen for k, v in totals.items()}}
        history.append(row)
        current = (accuracy, -nll)
        if rank is None or current > rank:
            rank, stale = current, 0
            best = {"state": copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()}),
                    "selected_epoch": epoch, "val_accuracy": accuracy, "val_nll": nll}
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs or stale >= a.patience:
            print(f"{history_path.stem} epoch={epoch}/{epochs} val_acc={accuracy:.4f} "
                  f"val_nll={nll:.4f} losses={row}", flush=True)
        pd.DataFrame(history).to_csv(history_path, index=False)
        if stale >= a.patience:
            break
    model.load_state_dict(best["state"])
    model.eval()
    return best


def make_graph(data, a):
    return EDGCNGraphTeacher(data["channels"], 1000, data["classes"],
        embedding_dim=a.embedding_dim, hidden_dim=a.hidden_dim, cheb_order=a.cheb_order,
        plv_threshold=a.plv_threshold, orth_weight=a.orth_weight).to(a.device)


def metadata(a, data):
    return {"version": VERSION, "config": config_for(a), "signature": data["signature"],
            "channels": data["channels"], "classes": data["classes"],
            "subject": data["subject"], "seed": data["seed"]}


def load_checkpoint(path, a, data):
    cp = torch.load(path, map_location="cpu", weights_only=True)
    for key, expected in metadata(a, data).items():
        if cp.get(key) != expected:
            raise ValueError(f"Checkpoint metadata mismatch: {path} field={key}")
    return cp


def train_teachers(a, data, folder, seed):
    rows = []
    for kind in ("temporal", "graph"):
        path = folder / f"{kind}_teacher.pth"
        set_seed(seed + (1000 if kind == "temporal" else 2000))
        model = (MultiscaleTemporalTeacher(data["channels"], data["classes"]).to(a.device)
                 if kind == "temporal" else make_graph(data, a))
        if path.exists():
            cp = load_checkpoint(path, a, data)
            print(f"SKIP completed {kind} teacher", flush=True)
        else:
            def objective(m, x, labels, index):
                ce = F.cross_entropy(m(x), labels)
                orth = m.orthogonality_loss() if kind == "graph" else ce.new_zeros(())
                return ce + orth, {"ce": ce, "orth": orth}
            cp = fit_model(model, data["teacher_train"], data["y_train"],
                           data["teacher_val"], data["y_val"], a, a.epochs_teacher,
                           seed + (1000 if kind == "temporal" else 2000), objective,
                           folder / f"{kind}_teacher_history.csv")
            model.load_state_dict(cp["state"])
            calibration = 1.
            if kind == "temporal" and a.calibration == "temperature":
                calibration, _ = fit_temperature_scaling(
                    logits_for(model, data["teacher_val"], data["y_val"], a).numpy(), data["y_val"])
            cp.update(metadata(a, data))
            cp.update({"calibration_temperature": calibration,
                       "normalization": [torch.from_numpy(v) for v in data["teacher_norm"]]})
            atomic_save(cp, path)
        rows.append({"teacher": kind, "parameters": sum(p.numel() for p in model.parameters()),
                     "selected_epoch": cp["selected_epoch"], "val_accuracy": cp["val_accuracy"],
                     "val_nll": cp["val_nll"]})
    pd.DataFrame(rows).to_csv(folder / "teacher_validation.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


@torch.no_grad()
def teacher_targets(a, data, folder):
    t = MultiscaleTemporalTeacher(data["channels"], data["classes"]).to(a.device)
    g = make_graph(data, a)
    tcp, gcp = (load_checkpoint(folder / f"{name}_teacher.pth", a, data) for name in ("temporal", "graph"))
    t.load_state_dict(tcp["state"])
    g.load_state_dict(gcp["state"])
    t.eval().requires_grad_(False)
    g.eval().requires_grad_(False)
    logits, relations = [], []
    for x, _, _ in loader(data["teacher_train"], data["y_train"], a.batch_size):
        logits.append(t(x.to(a.device)).cpu() / tcp["calibration_temperature"])
        _, nodes = g.forward_with_nodes(x.to(a.device))
        relations.append(node_relations(nodes).cpu())
    relation = torch.cat(relations)
    mask = ~torch.eye(data["channels"], dtype=torch.bool)
    values = relation[:, mask]
    print(f"GRAPH TARGET offdiag_mean={float(values.mean()):.5f} "
          f"offdiag_std={float(values.std()):.5f}; check for collapsed relations", flush=True)
    return torch.cat(logits), relation


def train_students(a, data, folder, seed):
    targets = teacher_targets(a, data, folder) if any(m != "ce" for m in a.methods) else None
    rows = []
    for method in a.methods:
        path = folder / f"student_{method}.pth"
        set_seed(seed + 3000)
        model = CompactRelationStudent(data["channels"], data["classes"]).to(a.device)
        if path.exists():
            cp = load_checkpoint(path, a, data)
            print(f"SKIP completed student {method}", flush=True)
        else:
            def objective(m, x, labels, index):
                logits, nodes = m.forward_with_nodes(x)
                ce = F.cross_entropy(logits, labels)
                kd, rel = ce.new_zeros(()), ce.new_zeros(())
                if method in ("temporal", "combined"):
                    kd = temporal_kd(logits, targets[0][index].to(a.device), a.temperature)
                if method in ("spatial", "combined"):
                    rel = relation_loss(nodes, targets[1][index].to(a.device))
                return ce + a.temporal_weight * kd + a.relation_weight * rel, {"ce": ce, "kd": kd, "relation": rel}
            cp = fit_model(model, data["student_train"], data["y_train"],
                           data["student_val"], data["y_val"], a, a.epochs_student,
                           seed + 3000, objective, folder / f"student_{method}_history.csv")
            cp.update(metadata(a, data))
            cp["normalization"] = [torch.from_numpy(v) for v in data["student_norm"]]
            atomic_save(cp, path)
        rows.append({"method": method, "val_accuracy": cp["val_accuracy"], "val_nll": cp["val_nll"],
                     "selected_epoch": cp["selected_epoch"], "parameters": sum(p.numel() for p in model.parameters())})
    pd.DataFrame(rows).to_csv(folder / "student_validation.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print("STUDENTS LOCKED; test NOT scored", flush=True)


def test_students(a, data, folder, subject, seed):
    rows = []
    for method in a.methods:
        cp = load_checkpoint(folder / f"student_{method}.pth", a, data)
        model = CompactRelationStudent(data["channels"], data["classes"]).to(a.device)
        model.load_state_dict(cp["state"])
        norm = tuple(t.numpy() for t in cp["normalization"])
        x = apply_norm(data["raw_test"][..., :500], norm)
        prediction = logits_for(model, x, data["test_labels"], a).argmax(1).numpy()
        values = evaluate_metrics(data["test_labels"], prediction)
        row = {"dataset": a.dataset, "subject": subject, "seed": seed, "method": method, **values}
        rows.append(row)
        pd.DataFrame({"trial_index": np.arange(len(prediction)), "y_true": data["test_labels"],
                      "y_pred": prediction}).to_csv(folder / f"{method}_test_predictions.csv", index=False)
        print(f"TEST {row}", flush=True)
    pd.DataFrame(rows).to_csv(folder / "test_results.csv", index=False)


def summarize(a, root):
    frames = []
    for seed in a.seeds:
        for subject in a.subjects:
            path = root / f"s{subject}_seed{seed}" / "test_results.csv"
            if path.exists():
                frames.append(pd.read_csv(path))
    if not frames:
        return
    frame = pd.concat(frames, ignore_index=True)
    if frame.duplicated(["dataset", "subject", "seed", "method"]).any():
        raise ValueError("Duplicate subject/seed/method test rows")
    frame.to_csv(root / "subject_results.csv", index=False)
    rows = []
    for (seed, method), group in frame.groupby(["seed", "method"]):
        rows.append({"seed": seed, "method": method, "n_subjects": len(group),
                     "complete": int(set(group.subject) == set(a.subjects)),
                     **{name: group[name].mean() for name in ("accuracy", "macro_f1", "kappa")}})
    seeds = pd.DataFrame(rows)
    seeds.to_csv(root / "seed_summary.csv", index=False)
    rows = []
    for method, group in seeds[seeds.complete == 1].groupby("method"):
        rows.append({"method": method, "n_complete_seeds": len(group),
                     **{f"{name}_{stat}": getattr(group[name], stat)()
                        for name in ("accuracy", "macro_f1", "kappa") for stat in ("mean", "std")}})
    pd.DataFrame(rows).to_csv(root / "summary.csv", index=False)


def main():
    a = parse_args()
    torch.set_num_threads(1)
    root = Path(a.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    guard_config(a, root)
    for seed in a.seeds:
        for subject in a.subjects:
            folder = root / f"s{subject}_seed{seed}"
            folder.mkdir(parents=True, exist_ok=True)
            data = prepare_data(a, subject, seed)
            if a.stage == "teachers":
                train_teachers(a, data, folder, seed)
            elif a.stage == "students":
                train_students(a, data, folder, seed)
            else:
                test_students(a, data, folder, subject, seed)
                summarize(a, root)
    print(f"DONE stage={a.stage} | {root.resolve()}", flush=True)


if __name__ == "__main__":
    main()
