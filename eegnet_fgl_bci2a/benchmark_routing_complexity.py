from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from models import EEGNet, EEGNetV4
from models.mi_active_detector import MIActiveDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark inference-only routing complexity.")
    parser.add_argument("--output_csv", default="results/publication_complete_analysis/inference_complexity.csv")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=500)
    return parser.parse_args()


def compact_eegnet(n_channels: int, n_classes: int) -> EEGNet:
    return EEGNet(
        n_channels=n_channels,
        n_samples=500,
        n_classes=n_classes,
        dropout_rate=0.5,
        F1=8,
        D=2,
        F2=16,
        kernel_length=64,
    )


def bci2b_detector() -> EEGNetV4:
    return EEGNetV4(
        n_channels=3,
        n_samples=250,
        n_classes=2,
        dropout_rate=0.5,
        F1=8,
        D=2,
        F2=16,
        kernel_length=125,
        separable_kernel_length=16,
        pool1=4,
        pool2=8,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def count_macs(model: nn.Module, sample: torch.Tensor) -> int:
    macs = 0
    hooks = []

    def conv_hook(module: nn.Conv2d, _inputs: tuple[torch.Tensor], output: torch.Tensor) -> None:
        nonlocal macs
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels // module.groups
        macs += int(output.numel() * kernel_ops)

    def linear_hook(module: nn.Linear, _inputs: tuple[torch.Tensor], output: torch.Tensor) -> None:
        nonlocal macs
        macs += int(output.numel() * module.in_features)

    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            hooks.append(layer.register_forward_hook(conv_hook))
        elif isinstance(layer, nn.Linear):
            hooks.append(layer.register_forward_hook(linear_hook))
    with torch.inference_mode():
        model(sample)
    for hook in hooks:
        hook.remove()
    return macs


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(callable_model, device: torch.device, warmup: int, repeats: int) -> float:
    with torch.inference_mode():
        for _ in range(warmup):
            callable_model()
        synchronize(device)
        started = time.perf_counter()
        for _ in range(repeats):
            callable_model()
        synchronize(device)
    return (time.perf_counter() - started) * 1000.0 / repeats


def benchmark_dataset(
    dataset: str,
    n_channels: int,
    n_classes: int,
    detector: nn.Module,
    detector_samples: int,
    detector_windows: int,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> list[dict[str, object]]:
    student = compact_eegnet(n_channels, n_classes)
    detector_batch = torch.randn(detector_windows, 1, n_channels, detector_samples)
    student_batch = torch.randn(1, 1, n_channels, 500)

    student_macs = count_macs(student.eval(), student_batch)
    detector_macs_per_window = count_macs(detector.eval(), detector_batch[:1])
    student_params = count_parameters(student)
    detector_params = count_parameters(detector)

    student = student.to(device).eval()
    detector = detector.to(device).eval()
    detector_batch = detector_batch.to(device)
    student_batch = student_batch.to(device)

    fixed_latency = benchmark(lambda: student(student_batch), device, warmup, repeats)

    def adaptive_forward() -> None:
        detector(detector_batch)
        student(student_batch)

    adaptive_latency = benchmark(adaptive_forward, device, warmup, repeats)
    return [
        {
            "dataset": dataset,
            "route": "fixed_2s_compact",
            "device": str(device),
            "train_time_teachers_used_at_inference": False,
            "detector_windows": 0,
            "parameters_executed": student_params,
            "macs": student_macs,
            "latency_ms_per_trial": fixed_latency,
        },
        {
            "dataset": dataset,
            "route": "adaptive_detector_plus_2s_compact",
            "device": str(device),
            "train_time_teachers_used_at_inference": False,
            "detector_windows": detector_windows,
            "parameters_executed": student_params + detector_params,
            "macs": student_macs + detector_windows * detector_macs_per_window,
            "latency_ms_per_trial": adaptive_latency,
        },
    ]


def main() -> None:
    args = parse_args()
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    rows: list[dict[str, object]] = []
    for device in devices:
        rows.extend(
            benchmark_dataset(
                dataset="bci2a",
                n_channels=22,
                n_classes=4,
                detector=MIActiveDetector(n_channels=22, n_samples=125, n_classes=2),
                detector_samples=125,
                detector_windows=7,
                device=device,
                warmup=args.warmup,
                repeats=args.repeats,
            )
        )
        rows.extend(
            benchmark_dataset(
                dataset="bci2b",
                n_channels=3,
                n_classes=2,
                detector=bci2b_detector(),
                detector_samples=250,
                detector_windows=5,
                device=device,
                warmup=args.warmup,
                repeats=args.repeats,
            )
        )

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
