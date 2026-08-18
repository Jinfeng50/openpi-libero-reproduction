import unittest

import numpy as np

from openpi_libero_reproduction.temporal_ensemble import (
    DGTEConfig,
    DisagreementGatedTemporalEnsembler,
    freshness_weight,
)


class TemporalEnsembleTest(unittest.TestCase):
    def test_first_chunk_is_unchanged(self):
        chunk = np.arange(35, dtype=np.float32).reshape(5, 7)
        controller = DisagreementGatedTemporalEnsembler()
        controller.add_chunk(chunk, start_step=0)
        np.testing.assert_allclose(controller.next_actions(0, 5), chunk)

    def test_overlap_averages_pose_and_keeps_newest_gripper(self):
        config = DGTEConfig(decay=0.0, gate_strength=0.0, replan_steps=5)
        controller = DisagreementGatedTemporalEnsembler(config)
        old = np.zeros((10, 7), dtype=np.float32)
        old[:, -1] = -1
        new = np.ones((10, 7), dtype=np.float32)
        new[:, -1] = 1
        controller.add_chunk(old, 0)
        controller.add_chunk(new, 5)

        fused = controller.next_actions(5, 5)
        np.testing.assert_allclose(fused[:, :6], 0.5)
        np.testing.assert_allclose(fused[:, -1], 1.0)

    def test_disagreement_gate_prefers_newest_prediction(self):
        config = DGTEConfig(decay=0.0, disagreement_threshold=0.1, gate_strength=3.0)
        controller = DisagreementGatedTemporalEnsembler(config)
        old = np.zeros((10, 7), dtype=np.float32)
        new = np.ones((10, 7), dtype=np.float32)
        controller.add_chunk(old, 0)
        controller.add_chunk(new, 5)

        fused = controller.action_at(5)
        # Gate=1, so latest weight is 1+3=4 and old weight is 1.
        np.testing.assert_allclose(fused[:6], 0.8)
        self.assertEqual(controller.last_candidate_count, 2)
        # Mean over both candidates: old differs by 1, newest differs by 0.
        self.assertAlmostEqual(controller.last_disagreement, 0.5)

    def test_old_chunks_are_pruned_after_execution(self):
        controller = DisagreementGatedTemporalEnsembler()
        controller.add_chunk(np.zeros((10, 7), dtype=np.float32), 0)
        controller.add_chunk(np.ones((10, 7), dtype=np.float32), 5)
        controller.next_actions(5, 5)
        with self.assertRaises(KeyError):
            controller.action_at(0)
        np.testing.assert_allclose(controller.action_at(10), 1.0)

    def test_rejects_invalid_chunk(self):
        controller = DisagreementGatedTemporalEnsembler()
        with self.assertRaises(ValueError):
            controller.add_chunk(np.zeros((7,), dtype=np.float32), 0)
        with self.assertRaises(ValueError):
            controller.add_chunk(np.full((2, 7), np.nan), 0)
        controller.add_chunk(np.zeros((2, 7), dtype=np.float32), 0)
        with self.assertRaises(ValueError):
            controller.add_chunk(np.zeros((2, 6), dtype=np.float32), 2)

    def test_freshness_weight(self):
        self.assertAlmostEqual(freshness_weight(0), 1.0)
        self.assertGreater(freshness_weight(0), freshness_weight(5))


if __name__ == "__main__":
    unittest.main()
