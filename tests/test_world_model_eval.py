import unittest
import tempfile
from pathlib import Path

import numpy as np

import torch

from scripts.evaluate_world_model import binary_auroc, calibration_metrics
from openpi_libero_reproduction.world_model import LatentChangeCritic
from openpi_libero_reproduction.world_model_controller import WorldModelActionSelector, align_action_chunk


class WorldModelEvaluationMetricsTest(unittest.TestCase):
    def test_auroc_handles_perfect_scores_and_ties(self):
        targets = np.asarray([0, 1, 0, 1])
        self.assertAlmostEqual(binary_auroc(targets, np.asarray([0.1, 0.9, 0.2, 0.8])), 1.0)
        self.assertAlmostEqual(binary_auroc(targets, np.ones(4)), 0.5)

    def test_auroc_returns_none_for_single_class(self):
        self.assertIsNone(binary_auroc(np.ones(3), np.asarray([0.1, 0.5, 0.9])))

    def test_calibration_is_bounded(self):
        metrics = calibration_metrics(np.asarray([0, 1, 0, 1]), np.asarray([0.1, 0.9, 0.2, 0.8]))
        self.assertEqual(set(metrics), {"brier", "ece_10bin", "mean_probability", "positive_rate"})
        self.assertGreaterEqual(metrics["brier"], 0.0)
        self.assertLessEqual(metrics["ece_10bin"], 1.0)

    def test_squared_error_cannot_be_negative(self):
        errors = np.asarray([1.0, -2.0, 0.0])
        self.assertGreaterEqual(float(np.mean(errors**2)), 0.0)

    def test_selector_requires_a_real_checkpoint(self):
        with self.assertRaises(FileNotFoundError):
            WorldModelActionSelector("/cfsdata/does-not-exist/critic.pt")

    def test_selector_ignores_action_source_checkpoint_metadata(self):
        model = LatentChangeCritic(
            latent_dim=1024,
            state_dim=8,
            action_horizon=5,
            action_dim=7,
            text_dim=256,
            hidden_dim=8,
            dropout=0.0,
        )
        config = {
            "latent_dim": 1024,
            "state_dim": 8,
            "action_horizon": 5,
            "action_dim": 7,
            "text_dim": 256,
            "hidden_dim": 8,
            "dropout": 0.0,
            "action_source": "selected_actions",
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "critic.pt"
            torch.save({"model": model.state_dict(), "config": config}, checkpoint)
            selector = WorldModelActionSelector(checkpoint, device="cpu", encoder_weights="none")
            self.assertEqual(selector.action_horizon, 5)

    def test_align_action_chunk_uses_future_slice_and_tail_padding(self):
        chunk = np.arange(70, dtype=np.float32).reshape(10, 7)
        aligned = align_action_chunk(chunk, offset=7, horizon=5)
        np.testing.assert_allclose(aligned[:3], chunk[7:10])
        np.testing.assert_allclose(aligned[3:], np.repeat(chunk[9:10], 2, axis=0))


if __name__ == "__main__":
    unittest.main()
