import argparse
import csv
import os
from pathlib import Path
import tempfile
import unittest

from run_cycle_window_ablation import (
    build_command,
    check_training_config,
    matching_row,
    mne_data_path_for_subprocess,
    required_cache_files,
    result_path,
    summarize,
    windows_from_cycles,
)


class CycleWindowAblationTests(unittest.TestCase):
    def test_nominal_cycles_map_to_valid_windows(self):
        self.assertEqual(windows_from_cycles([20, 5, 10], 10, 4), [0.5, 1.0, 2.0])
        with self.assertRaises(ValueError):
            windows_from_cycles([2.01], 10, 4)
        with self.assertRaises(ValueError):
            windows_from_cycles([50], 10, 4)

    def test_cache_paths_match_bnci_filenames(self):
        root = Path("data")
        self.assertEqual([path.name for path in required_cache_files(root, "bci2b", 1)],
                         ["B01T.mat", "B01E.mat"])
        cache_path = mne_data_path_for_subprocess(root.resolve())
        if os.name == "nt":
            self.assertFalse(":" in cache_path)

    def test_commands_use_student_only_test_evaluation(self):
        args = argparse.Namespace(
            riemann_kd_weight=0.1, spatial_weight=0.2, teacher_window=4.0,
            epochs=300, patience=100, batch_size=32, lr=0.001,
            weight_decay=0.0001, device="cpu",
        )
        command = build_command(args, "bci2b", "full", "riemann_spatial", 1, 42, 2.0, Path("results/test"))
        self.assertIn("--skip_riemann_teacher_test", command)
        self.assertEqual(command[command.index("--kl_weight") + 1], "0.1")
        self.assertEqual(command[command.index("--spatial_structure_weight") + 1], "0.2")

    def test_resume_config_rejects_changed_training_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run_config.json"
            check_training_config(path, {"epochs": 300})
            check_training_config(path, {"epochs": 300})
            with self.assertRaises(ValueError):
                check_training_config(path, {"epochs": 100})

    def test_summary_uses_only_complete_subject_sets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = result_path(root, "bci2a", "full", "ce")
            source.parent.mkdir(parents=True)
            with source.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["subject", "seed", "window", "accuracy", "macro_f1", "kappa"])
                writer.writeheader()
                writer.writerows([
                    {"subject": 1, "seed": 42, "window": 2, "accuracy": 0.6, "macro_f1": 0.5, "kappa": 0.4},
                    {"subject": 2, "seed": 42, "window": 2, "accuracy": 0.8, "macro_f1": 0.7, "kappa": 0.6},
                    {"subject": 1, "seed": 7, "window": 2, "accuracy": 0.9, "macro_f1": 0.8, "kappa": 0.7},
                ])
            args = argparse.Namespace(
                datasets=["bci2a"], bands=["full"], methods=["ce"],
                subjects=[1, 2], seeds=[42, 7],
                mu_reference_hz=10.0, beta_reference_hz=20.0,
            )
            self.assertIsNotNone(matching_row(
                [{"subject": "1", "seed": "42", "window": "2.0"}], 1, 42, 2.0,
            ))
            self.assertEqual(summarize(args, [2.0], root), (3, 1))
            with (root / "summary.csv").open(newline="", encoding="utf-8") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(int(row["n_complete_seeds"]), 1)
            self.assertAlmostEqual(float(row["mean_accuracy"]), 0.7)


if __name__ == "__main__":
    unittest.main()
