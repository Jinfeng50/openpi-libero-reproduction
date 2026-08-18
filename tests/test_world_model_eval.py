import unittest

import numpy as np

from scripts.evaluate_world_model import binary_auroc, calibration_metrics


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


if __name__ == "__main__":
    unittest.main()
