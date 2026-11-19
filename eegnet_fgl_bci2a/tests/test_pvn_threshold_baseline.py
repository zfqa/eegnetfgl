from argparse import Namespace
import json
import math

import numpy as np
import pandas as pd
import pytest
import torch

import run_pvn_threshold_baseline as runner
from models.pvn_lbrl import PVNLBRL


def test_vote_matches_hard_point_voting():
    model = PVNLBRL(channels=3, hidden_size=8)
    x = torch.randn(2, 3, 80)
    prediction, share = runner.vote(model, x)
    torch.testing.assert_close(prediction, model.encoder.predict_vote(x))
    assert ((share >= 0.25) & (share <= 1)).all()


def test_predict_never_uses_later_data_for_stopped_trials(monkeypatch):
    model = PVNLBRL(channels=3, hidden_size=8)
    x = torch.ones(2, 3, 480)
    x[1] *= 2
    calls = []

    def vote(_, prefix):
        calls.append((len(prefix), prefix.shape[-1]))
        label = (prefix[:, 0, 0] - 1).long()
        share = torch.where(label == 0, 0.9, 0.4)
        return label, share

    monkeypatch.setattr(runner, "vote", vote)
    prediction, times = runner.predict(model, x, [280, 320, 480], 0.8)
    assert calls == [(2, 280), (1, 320), (1, 480)]
    np.testing.assert_array_equal(prediction, [0, 1])
    np.testing.assert_array_equal(times, [280, 480])
    changed = x.clone()
    changed[0, :, 280:] = -999
    second = runner.predict(model, changed, [280, 320, 480], 0.8)
    np.testing.assert_array_equal(second[0], prediction)
    np.testing.assert_array_equal(second[1], times)
    _, fixed_times = runner.predict(model, x, [280, 320, 480], math.inf)
    np.testing.assert_array_equal(fixed_times, [480, 480])


def test_threshold_selection_accuracy_guard_and_fixed_fallback(monkeypatch):
    model = PVNLBRL(channels=3, hidden_size=8)
    y = torch.tensor([0, 1])
    def fake_predict(_, x, samples, threshold, batch):
        if threshold == 0.5:
            return np.array([1, 1]), np.array([280, 280])
        if threshold == 0.8:
            return np.array([0, 1]), np.array([320, 480])
        return np.array([0, 1]), np.array([480, 480])
    monkeypatch.setattr(runner, "predict", fake_predict)
    tau, frame = runner.select_threshold(model, torch.zeros(2, 3, 480), y, [280, 320, 480],
                                         [0.5, 0.8, 0.9], 32)
    assert tau == 0.8 and frame.selected.sum() == 1
    tau, _ = runner.select_threshold(model, torch.zeros(2, 3, 480), y, [280, 320, 480],
                                    [0.5, 0.9], 32)
    assert math.isinf(tau)


def test_subject_locks_policy_before_test_scoring_and_fits_only_train(monkeypatch, tmp_path):
    model = PVNLBRL(channels=3, hidden_size=8)
    train, val, test = torch.randn(4, 3, 1000), torch.randn(2, 3, 1000), torch.randn(2, 3, 1000)
    y_train, y_val, y_test = torch.tensor([0, 1, 2, 3]), torch.tensor([0, 1]), torch.tensor([1, 0])
    monkeypatch.setattr(runner, "load_split", lambda *a: (train, y_train, val, y_val, test, y_test))
    monkeypatch.setattr(runner, "PVNLBRL", lambda **kw: model)
    def fit(m, x, y, xv, yv, **kwargs):
        assert x is train and y is y_train and xv is val and yv is y_val
    monkeypatch.setattr(runner, "pretrain_encoder", fit)
    def select(m, x, y, *a):
        assert x is val and y is y_val
        return math.inf, pd.DataFrame([{"threshold": math.inf, "selected": True}])
    monkeypatch.setattr(runner, "select_threshold", select)
    def predict(m, x, *a):
        assert x is test
        policy = json.loads((tmp_path / "s1_seed42/locked_policy.json").read_text())
        assert policy["fallback_fixed"]
        assert (tmp_path / "s1_seed42/pvn.pth").exists()
        return np.array([0, 1]), np.array([480, 480])
    monkeypatch.setattr(runner, "predict", predict)
    args = Namespace(device="cpu", epochs_pvn=1, batch_size=2)
    rows = runner.run_subject(args, 1, 42, tmp_path)
    assert len(rows) == 2 and rows[0]["accuracy"] == 0
    saved = pd.read_csv(tmp_path / "s1_seed42/threshold_test_predictions.csv")
    assert saved.y_true.tolist() == y_test.tolist()
    assert saved.observation_s.tolist() == [1.92, 1.92]


def test_main_resume_does_not_retrain_completed_pairs(monkeypatch, tmp_path):
    fits = []
    def run(args, subject, seed, root):
        fits.append((subject, seed))
        return [
            {"subject": subject, "seed": seed, "method": method,
             "accuracy": 0.5, "macro_f1": 0.5, "kappa": 0.3, "mean_observation_s": 1.92}
            for method in ("fixed_1p92s", "threshold")
        ]
    monkeypatch.setattr(runner, "run_subject", run)
    monkeypatch.setattr("sys.argv", ["run_pvn_threshold_baseline.py", "--subjects", "1",
                                     "--seeds", "42", "--device", "cpu",
                                     "--output_dir", str(tmp_path)])
    runner.main()
    runner.main()
    assert fits == [(1, 42)]
    assert len(pd.read_csv(tmp_path / "subject_results.csv")) == 2
    assert (tmp_path / "seed_summary.csv").exists()
    assert (tmp_path / "summary.csv").exists()
