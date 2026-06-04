from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Riemannian-guided EEGNet experiment matrix.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 10)))
    parser.add_argument("--windows", type=float, nargs="+", default=[1.0, 2.0])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="results/riemann_guided")
    parser.add_argument("--cov_shrinkage", type=float, default=0.001)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--geometry_mode", type=str, default="cosine", choices=["cosine", "soft_graph", "knn_graph"])
    parser.add_argument("--geometry_trust_mode", type=str, default="none", choices=["none", "sample"])
    parser.add_argument("--graph_k", type=int, default=8)
    parser.add_argument("--graph_temperature", type=float, default=1.0)
    parser.add_argument("--supcon_weight", type=float, default=0.0)
    parser.add_argument("--confidence_mode", type=str, default="normalized", choices=["none", "raw", "normalized"])
    parser.add_argument("--confidence_threshold", type=float, default=0.0)
    parser.add_argument("--trust_gate_mode", type=str, default="confidence", choices=["confidence", "val_logreg"])
    parser.add_argument("--trust_gate_c", type=float, default=1.0)
    parser.add_argument("--correct_only_kl", action="store_true")
    parser.add_argument(
        "--config",
        type=str,
        default="short_adaptive",
        choices=["short_adaptive", "weak_kl", "strong_kl_geo", "scaffold_graph"],
    )
    return parser.parse_args()


def loss_config(name: str, window: float) -> tuple[float, float]:
    if name == "weak_kl":
        return 0.05, 0.0
    if name == "strong_kl_geo":
        return 0.2, 0.05
    if name == "scaffold_graph":
        if window <= 1.0:
            return 0.05, 0.05
        return 0.1, 0.1
    if window <= 1.0:
        return 0.05, 0.0
    return 0.2, 0.05


def row_exists(
    save_dir: Path,
    subject: int,
    window: float,
    seed: int,
    kl_weight: float,
    geometry_weight: float,
    geometry_mode: str,
    geometry_trust_mode: str,
    graph_k: int,
    graph_temperature: float,
    supcon_weight: float,
    temperature: float,
    confidence_mode: str,
    confidence_threshold: float,
    trust_gate_mode: str,
    trust_gate_c: float,
    correct_only_kl: bool,
    cov_shrinkage: float,
) -> bool:
    path = save_dir / "riemann_guided_results.csv"
    if not path.exists():
        return False
    df = pd.read_csv(path)
    required = {
        "method",
        "subject",
        "window",
        "seed",
        "kl_weight",
        "geometry_weight",
        "geometry_mode",
        "geometry_trust_mode",
        "graph_k",
        "graph_temperature",
        "supcon_weight",
        "temperature",
        "confidence_mode",
        "confidence_threshold",
        "trust_gate_mode",
        "trust_gate_c",
        "correct_only_kl",
        "cov_shrinkage",
    }
    if not required.issubset(df.columns):
        return False
    mask = (
        (df["method"] == "EEGNet-RiemannGuided")
        & (df["subject"].astype(int) == int(subject))
        & np.isclose(df["window"].astype(float), float(window))
        & (df["seed"].astype(int) == int(seed))
        & np.isclose(df["kl_weight"].astype(float), float(kl_weight))
        & np.isclose(df["geometry_weight"].astype(float), float(geometry_weight))
        & (df["geometry_mode"].fillna("").astype(str) == geometry_mode)
        & (df["geometry_trust_mode"].fillna("none").astype(str) == geometry_trust_mode)
        & np.isclose(pd.to_numeric(df["graph_k"], errors="coerce"), float(graph_k))
        & np.isclose(pd.to_numeric(df["graph_temperature"], errors="coerce"), float(graph_temperature))
        & np.isclose(df["supcon_weight"].astype(float), float(supcon_weight))
        & np.isclose(df["temperature"].astype(float), float(temperature))
        & (df["confidence_mode"].astype(str) == confidence_mode)
        & np.isclose(df["confidence_threshold"].astype(float), float(confidence_threshold))
        & (df["trust_gate_mode"].astype(str) == trust_gate_mode)
        & np.isclose(df["trust_gate_c"].astype(float), float(trust_gate_c))
        & (df["correct_only_kl"].astype(str).str.lower() == str(bool(correct_only_kl)).lower())
        & np.isclose(df["cov_shrinkage"].astype(float), float(cov_shrinkage))
    )
    return bool(mask.any())


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        print(f"\n=== Seed {seed} ===", flush=True)
        for subject in args.subjects:
            for window in args.windows:
                kl_weight, geometry_weight = loss_config(args.config, window)
                print(
                    f"\n--- subject={subject} window={window:g}s seed={seed} "
                    f"kl={kl_weight:g} geo={geometry_weight:g} ---",
                    flush=True,
                )
                if row_exists(
                    save_dir,
                    subject,
                    window,
                    seed,
                    kl_weight,
                    geometry_weight,
                    args.geometry_mode,
                    args.geometry_trust_mode,
                    args.graph_k,
                    args.graph_temperature,
                    args.supcon_weight,
                    args.temperature,
                    args.confidence_mode,
                    args.confidence_threshold,
                    args.trust_gate_mode,
                    args.trust_gate_c,
                    args.correct_only_kl,
                    args.cov_shrinkage,
                ):
                    print("Skip existing Riemann-guided row.", flush=True)
                    continue

                cmd = [
                    sys.executable,
                    "train_student_riemann_guided.py",
                    "--subject",
                    str(subject),
                    "--window",
                    str(window),
                    "--epochs",
                    str(args.epochs),
                    "--batch_size",
                    str(args.batch_size),
                    "--lr",
                    str(args.lr),
                    "--weight_decay",
                    str(args.weight_decay),
                    "--seed",
                    str(seed),
                    "--device",
                    args.device,
                    "--save_dir",
                    str(save_dir),
                    "--cov_shrinkage",
                    str(args.cov_shrinkage),
                    "--kl_weight",
                    str(kl_weight),
                    "--geometry_weight",
                    str(geometry_weight),
                    "--geometry_mode",
                    args.geometry_mode,
                    "--geometry_trust_mode",
                    args.geometry_trust_mode,
                    "--graph_k",
                    str(args.graph_k),
                    "--graph_temperature",
                    str(args.graph_temperature),
                    "--supcon_weight",
                    str(args.supcon_weight),
                    "--temperature",
                    str(args.temperature),
                    "--confidence_mode",
                    args.confidence_mode,
                    "--confidence_threshold",
                    str(args.confidence_threshold),
                    "--trust_gate_mode",
                    args.trust_gate_mode,
                    "--trust_gate_c",
                    str(args.trust_gate_c),
                ]
                if args.correct_only_kl:
                    cmd.append("--correct_only_kl")
                print(" ".join(cmd), flush=True)
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
