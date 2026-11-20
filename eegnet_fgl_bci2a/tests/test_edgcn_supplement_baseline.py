"""Supplement core equivalence and local-adapter tests; no real EEG training."""

import ast
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from models.edgcn_supplement import EDGCNModule, LocalFeatureExtraction, objective, orthogonal_loss
import run_edgcn_supplement_baseline as runner


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def source_namespace(name, definitions):
    # Do not import the broken Lightning wrapper or any external attachment imports.
    tree = ast.parse((runner.REFERENCE / name).read_text(encoding="utf-8-sig"))
    nodes = [n for n in tree.body if isinstance(n, definitions)]
    namespace = {"torch": torch, "nn": nn, "F": F}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), name, "exec"), namespace)
    return namespace


def test_core_matches_source_parameters_and_predictions_in_train_and_eval():
    reference = source_namespace("mmc2.py", ast.ClassDef)["EDGCNModule"]
    torch.manual_seed(42); original = reference()
    torch.manual_seed(42); ported = EDGCNModule()
    assert original.state_dict().keys() == ported.state_dict().keys()
    for key, value in original.state_dict().items():
        torch.testing.assert_close(value, ported.state_dict()[key], atol=0, rtol=0)
    x = torch.randn(2, 22, 1125)
    up, down = torch.randn(2, 6, 88), torch.randn(2, 4, 88)
    local, glob = torch.rand(2, 22, 22), torch.rand(2, 22, 22)
    for training in (False, True):
        original.train(training); ported.train(training)
        torch.manual_seed(7); a = original(x, up, down, None, local, glob)
        torch.manual_seed(7); b = ported(x, up, down, None, local, glob)
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_whole_bank_regularizer_matches_mmc1_not_order_block_formula():
    source = source_namespace("mmc1.py", ast.FunctionDef)["orthogonal_loss"]
    pool = torch.randn(5, 12, 3)
    torch.testing.assert_close(source(pool), orthogonal_loss(pool), atol=0, rtol=0)
    model = EDGCNModule()
    logits, labels = torch.randn(2, 4), torch.tensor([0, 1])
    expected = F.cross_entropy(logits, labels) + .1 * sum(source(p) for p in model.theta_pool)
    torch.testing.assert_close(objective(model, logits, labels), expected, atol=0, rtol=0)


def test_frontend_time33_matches1125_not1000():
    model = LocalFeatureExtraction(22, 11).eval()
    with torch.no_grad():
        assert model(torch.randn(1, 22, 1125)).shape == (1, 22, 33, 30)
        assert model(torch.randn(1, 22, 1000)).shape == (1, 22, 29, 30)


def make_args(tmp_path):
    return Namespace(stage="fit", subject=1, seed=42, data_dir=str(tmp_path),
        output_dir=str(tmp_path / "experiment"), device="cpu", epochs=1, patience=1,
        val_ratio=.2, batch_size=2, lr=.0009, weight_decay=0., standardize=False,
        psd_bands=[.5, 4, 4, 8, 8, 13, 13, 38], psd_window_s=.5,
        mask_probability=.1, graph_threshold=.5, psd_transform="log1p")


def test_adapter_shapes_bands_and_deterministic_eval():
    adapter = runner.LocalAuxiliaryInputs([(.5, 4), (4, 8), (8, 13), (13, 38)]).eval()
    x = torch.randn(2, 22, 1125)
    output = adapter(x)
    assert output[0].shape[0] == 2 and output[0].shape[-1] == 88
    assert output[1].shape[0] == 2 and output[1].shape[-1] == 88
    for i in (2, 3):
        assert output[i].shape == (2, 22, 22)
        torch.testing.assert_close(output[i], output[i].transpose(-1, -2))
    for left, right in zip(output, adapter(x)):
        torch.testing.assert_close(left, right, atol=0, rtol=0)


def test_data_protocol_negative_tmin_and_train_only_norm(tmp_path, monkeypatch):
    args = make_args(tmp_path)
    folder = tmp_path / "MNE-bnci-data" / "~bci" / "database" / "001-2014"
    folder.mkdir(parents=True)
    for split in ("T", "E"):
        (folder / f"A01{split}.mat").touch()
    raw = np.random.default_rng(42).normal(size=(40, 22, 1126)).astype(np.float32)
    y = np.tile(np.arange(4), 10)
    meta = pd.DataFrame({"session": ["train"] * 20 + ["eval"] * 20})
    def load(subject, **kw):
        assert kw == {"fmin": .5, "fmax": 38., "tmin": -.5, "tmax": 4.}
        return raw, y, meta
    monkeypatch.setattr(runner, "load_bci2a_subject", load)
    monkeypatch.setattr(runner, "configure_mne_data_path", lambda *a: None)
    first = runner.prepare_data(args)
    assert first[0].shape == (16, 22, 1125) and first[2].shape == (4, 22, 1125)
    np.testing.assert_array_equal(first[6], 0)
    np.testing.assert_array_equal(first[7], 1)
    args.standardize = True
    before = runner.prepare_data(args)
    raw[20:] += 100000
    after = runner.prepare_data(args)
    for i in (0, 1, 2, 3, 6, 7):
        np.testing.assert_array_equal(before[i], after[i])
    assert before[-1] == after[-1]


def test_two_stages_do_not_score_test_until_explicit_request(tmp_path, monkeypatch):
    args = make_args(tmp_path)
    rng = np.random.default_rng(42)
    train, val = rng.normal(size=(4, 22, 1125)).astype(np.float32), rng.normal(size=(4, 22, 1125)).astype(np.float32)
    y = np.arange(4)
    class ForbiddenTest:
        def __sub__(self, value):
            pytest.fail("Fit must not normalize or score test")
    norm = (np.zeros((1, 22, 1), np.float32), np.ones((1, 22, 1), np.float32))
    monkeypatch.setattr(runner, "prepare_data", lambda a: (train, y, val, y, ForbiddenTest(), None, *norm, "datahash"))
    runner.run(args)
    root = Path(args.output_dir)
    assert (root / "locked_model.pth").exists()
    assert not (root / "test_results.csv").exists()
    with pytest.raises(FileExistsError):
        runner.run(args)
    args.stage = "test"
    monkeypatch.setattr(runner, "prepare_data", lambda a: (train, y, val, y, val, y, *norm, "datahash"))
    runner.run(args)
    assert len(pd.read_csv(root / "test_predictions.csv")) == 4
    args.lr = .01
    with pytest.raises(ValueError):
        runner.run(args)
