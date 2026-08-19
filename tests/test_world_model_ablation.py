import unittest

from scripts.analyze_world_model_ablation import exact_mcnemar_p, paired_rows
from scripts.analyze_gate_metadata import sweep_gate_thresholds


class WorldModelAblationAnalysisTest(unittest.TestCase):
    def test_exact_mcnemar_known_values(self):
        self.assertEqual(exact_mcnemar_p(0, 0), 1.0)
        self.assertEqual(exact_mcnemar_p(1, 1), 1.0)
        self.assertAlmostEqual(exact_mcnemar_p(0, 5), 0.0625)

    def test_paired_rows_count_each_outcome(self):
        keys = [("suite", 0, episode, 7) for episode in range(4)]
        baseline = dict(zip(keys, [True, True, False, False]))
        world_model = dict(zip(keys, [True, False, True, False]))
        suite, aggregate = paired_rows(baseline, world_model)
        self.assertEqual(suite["both_success"], 1)
        self.assertEqual(suite["baseline_only"], 1)
        self.assertEqual(suite["world_model_only"], 1)
        self.assertEqual(suite["both_failure"], 1)
        self.assertEqual(suite["delta_pp"], 0.0)
        self.assertEqual(suite, {**aggregate, "suite": "suite"})

    def test_requires_exactly_paired_keys(self):
        with self.assertRaises(ValueError):
            paired_rows({("suite", 0, 0, 7): True}, {("suite", 0, 1, 7): True})

    def test_gate_threshold_sweep_is_observational(self):
        rows = [
            {"margin": 0.001, "uncertainty": 0.30, "episode_success": True},
            {"margin": 0.005, "uncertainty": 0.40, "episode_success": False},
        ]
        result = sweep_gate_thresholds(rows, [0.001], [0.35])[0]
        self.assertEqual(result["accepted_transitions"], 1)
        self.assertEqual(result["acceptance_rate"], 0.5)
        self.assertEqual(result["accepted_transition_success_rate"], 1.0)
        self.assertIn("observational", result["interpretation"])


if __name__ == "__main__":
    unittest.main()
