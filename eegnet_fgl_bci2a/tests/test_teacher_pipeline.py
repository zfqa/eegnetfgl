from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from train_student_riemann_guided import RiemannGuidedDataset
from utils.riemann_teacher import PaperRiemannTeacher
from utils.teacher_utils import (
    apply_temperature_scaling,
    fit_temperature_scaling,
    multiclass_nll,
    validate_teacher_checkpoint,
)


class TeacherPipelineTests(unittest.TestCase):
    def test_temperature_scaling_does_not_increase_validation_nll(self) -> None:
        logits = np.asarray(
            [
                [8.0, -2.0],
                [7.0, -1.0],
                [-2.0, 8.0],
                [-1.0, 7.0],
                [5.0, -4.0],
                [-4.0, 5.0],
            ],
            dtype=np.float32,
        )
        labels = np.asarray([0, 1, 1, 0, 0, 1], dtype=np.int64)
        temperature, diagnostics = fit_temperature_scaling(logits, labels)
        calibrated = apply_temperature_scaling(logits, temperature)
        self.assertGreater(temperature, 0.0)
        self.assertLessEqual(multiclass_nll(calibrated, labels), multiclass_nll(logits, labels) + 1e-8)
        self.assertLessEqual(diagnostics["nll_after"], diagnostics["nll_before"] + 1e-8)

    def test_paper_riemann_teacher_outputs_are_aligned_and_finite(self) -> None:
        rng = np.random.default_rng(42)
        n_channels = 3
        n_samples = 128

        def make_trials(labels: np.ndarray) -> np.ndarray:
            trials = rng.normal(size=(len(labels), n_channels, n_samples)).astype(np.float32)
            for index, label in enumerate(labels):
                trials[index, label % n_channels] *= 1.0 + 0.8 * label
                trials[index, (label + 1) % n_channels] += 0.2 * trials[index, label % n_channels]
            return trials

        y_train = np.tile(np.arange(2, dtype=np.int64), 12)
        y_val = np.tile(np.arange(2, dtype=np.int64), 4)
        X_train = make_trials(y_train)
        X_val = make_trials(y_val)

        teacher = PaperRiemannTeacher(
            estimator="oas",
            alignment="none",
            metric="riemann",
            method="ts_logreg",
            c_grid=(0.1, 1.0),
            calibration="temperature",
            n_classes=2,
        ).fit(X_train, y_train, X_val, y_val)
        output = teacher.predict_output(X_val)

        self.assertEqual(output.logits.shape, (len(y_val), 2))
        self.assertEqual(output.probabilities.shape, (len(y_val), 2))
        self.assertEqual(output.covariances.shape, (len(y_val), n_channels, n_channels))
        self.assertEqual(output.features.shape[0], len(y_val))
        self.assertTrue(np.isfinite(output.logits).all())
        self.assertTrue(np.isfinite(output.features).all())
        np.testing.assert_allclose(output.probabilities.sum(axis=1), 1.0, atol=1e-6)
        self.assertEqual(teacher.diagnostics["riemann_input_preprocessing"], "raw")

    def test_checkpoint_validation_rejects_wrong_subject(self) -> None:
        checkpoint = {
            "subject": 1,
            "dataset": "bci2a",
            "split_seed": 42,
            "model_args": {
                "n_samples": 1000,
                "n_channels": 22,
                "n_classes": 4,
            },
            "preprocess": {
                "fmin": 8.0,
                "fmax": 32.0,
                "tmin": 0.0,
                "tmax": 4.0,
                "fs": 250,
                "val_ratio": 0.2,
                "protocol": "moabb_standard",
            },
        }
        with self.assertRaisesRegex(ValueError, "subject"):
            validate_teacher_checkpoint(
                checkpoint,
                Path("teacher.pth"),
                expected_samples=1000,
                expected_channels=22,
                expected_classes=4,
                expected_subject=2,
                expected_seed=42,
                expected_dataset="bci2a",
            )

    def test_dataset_checks_confidence_length(self) -> None:
        X = np.zeros((3, 3, 16), dtype=np.float32)
        y = np.zeros(3, dtype=np.int64)
        probs = np.full((3, 2), 0.5, dtype=np.float32)
        features = np.zeros((3, 6), dtype=np.float32)
        structure = np.zeros((3, 3, 3), dtype=np.float32)
        with self.assertRaises(ValueError):
            RiemannGuidedDataset(X, y, probs, probs, features, structure, np.ones(2, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
