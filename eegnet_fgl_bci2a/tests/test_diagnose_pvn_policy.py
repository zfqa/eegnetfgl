from __future__ import annotations

import pytest
import torch

from diagnose_pvn_policy import validation_diagnostics
from models.pvn_lbrl import PVNLBRL


def test_validation_diagnostics_has_prefixes_and_halt_counts() -> None:
    torch.manual_seed(11)
    model = PVNLBRL(channels=3, classes=4, slice_samples=40, hidden_size=16)
    for parameter in model.q_network.parameters():
        parameter.data.zero_()
    model.q_network.advantage[-1].bias.data[PVNLBRL.HALT] = 10
    x = torch.randn(3, 3, 120)
    y = torch.tensor([0, 1, 2])
    prefix, halts, policy = validation_diagnostics(model, x, y)
    assert prefix.observation_s.tolist() == pytest.approx([0.16, 0.32, 0.48])
    assert halts.slices.tolist() == [1]
    assert halts.trials.tolist() == [3]
    assert policy["mean_observation_s"] == pytest.approx(0.16)
