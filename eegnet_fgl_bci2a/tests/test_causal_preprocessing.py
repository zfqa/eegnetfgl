from __future__ import annotations

import numpy as np
from scipy.signal import sosfiltfilt

from train_teacher import teacher_checkpoint_name
from utils.data_loader import causal_bandpass_continuous, design_causal_bandpass_sos


def test_causal_filter_prefix_is_invariant_to_future_changes() -> None:
    rng = np.random.default_rng(42)
    original = rng.normal(size=(2000, 3))
    changed = original.copy()
    changed[1000:] = rng.normal(size=changed[1000:].shape)

    causal_original = causal_bandpass_continuous(original, 8.0, 32.0, 250.0, order=4)
    causal_changed = causal_bandpass_continuous(changed, 8.0, 32.0, 250.0, order=4)
    assert np.array_equal(causal_original[:1000], causal_changed[:1000])

    sos = design_causal_bandpass_sos(8.0, 32.0, 250.0, order=4)
    zero_original = sosfiltfilt(sos, original, axis=0)
    zero_changed = sosfiltfilt(sos, changed, axis=0)
    assert not np.allclose(zero_original[:1000], zero_changed[:1000])


def test_teacher_checkpoint_name_keeps_legacy_4s_and_separates_2s() -> None:
    assert teacher_checkpoint_name(1, "compact", "bci2a", 4.0) == "subject_1_teacher_best.pth"
    assert teacher_checkpoint_name(1, "compact", "bci2a", 2.0) == "subject_1_teacher_2s_best.pth"
