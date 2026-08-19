import unittest

from scripts.analyze_world_model_ablation import exact_mcnemar_p, paired_rows


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


if __name__ == "__main__":
    unittest.main()
