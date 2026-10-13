from argparse import Namespace

import pytest

import run_teacher_window_matched_kd as runner


def test_parse_nvidia_smi_rows_ignores_malformed_lines():
    output = "0, 12000, 24576, 4\ninvalid\n1, 8000, 24576, 20\n"

    assert runner.parse_nvidia_smi_rows(output) == [
        {"index": 0, "free_mb": 12000, "total_mb": 24576, "utilization": 4},
        {"index": 1, "free_mb": 8000, "total_mb": 24576, "utilization": 20},
    ]


def test_wait_for_idle_gpu_retries_until_available(monkeypatch):
    attempts = iter([RuntimeError("busy"), {"index": 3, "free_mb": 18000}])
    sleeps = []

    def fake_select(_args):
        result = next(attempts)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(runner, "select_idle_gpu", fake_select)
    monkeypatch.setattr(runner.time, "sleep", sleeps.append)
    args = Namespace(gpu_poll_seconds=15.0, gpu_wait_timeout_minutes=0.0)

    assert runner.wait_for_idle_gpu(args)["index"] == 3
    assert sleeps == [15.0]


def test_wait_for_idle_gpu_validates_poll_interval():
    args = Namespace(gpu_poll_seconds=0.0, gpu_wait_timeout_minutes=0.0)

    with pytest.raises(ValueError, match="gpu_poll_seconds"):
        runner.wait_for_idle_gpu(args)
