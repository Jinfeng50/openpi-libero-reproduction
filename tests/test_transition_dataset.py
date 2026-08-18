import pathlib
import tempfile
import unittest

import numpy as np

from openpi_libero_reproduction.transition_dataset import EpisodeTransitionRecorder
from openpi_libero_reproduction.transition_dataset import load_episode_shard


class TransitionRecorderTest(unittest.TestCase):
    def _image(self, value):
        return np.full((8, 8, 3), value, dtype=np.uint8)

    def test_round_trip_episode_shard(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = EpisodeTransitionRecorder(
                directory,
                suite="libero_spatial",
                controller="baseline",
                task_id=2,
                episode_idx=3,
                prompt="pick up the mug",
                seed=7,
                replan_steps=5,
            )
            recorder.add(
                image=self._image(1),
                wrist_image=self._image(2),
                future_image=self._image(3),
                future_wrist_image=self._image(4),
                state=np.arange(8),
                future_state=np.arange(8) + 1,
                action_chunk=np.zeros((50, 7)),
                selected_actions=np.ones((5, 7)),
                executed_steps=5,
                start_step=10,
                future_step=15,
                terminal_within_horizon=False,
            )
            path = recorder.finish(episode_success=True)
            self.assertIsNotNone(path)
            shard = load_episode_shard(path)
            self.assertEqual(shard["image"].shape, (1, 8, 8, 3))
            self.assertEqual(shard["action_chunk"].shape, (1, 50, 7))
            self.assertTrue(bool(shard["episode_success"]))
            self.assertEqual(str(shard["prompt"]), "pick up the mug")

    def test_partial_terminal_transition(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = EpisodeTransitionRecorder(
                directory,
                suite="libero_10",
                controller="dgte",
                task_id=0,
                episode_idx=0,
                prompt="open drawer",
                seed=7,
                replan_steps=5,
            )
            recorder.add(
                image=self._image(0),
                wrist_image=self._image(0),
                future_image=self._image(1),
                future_wrist_image=self._image(1),
                state=np.zeros(8),
                future_state=np.ones(8),
                action_chunk=np.zeros((50, 7)),
                selected_actions=np.zeros((5, 7)),
                executed_steps=2,
                start_step=10,
                future_step=12,
                terminal_within_horizon=True,
            )
            shard = load_episode_shard(recorder.finish(episode_success=True))
            self.assertEqual(int(shard["executed_steps"][0]), 2)
            self.assertTrue(bool(shard["terminal_within_horizon"][0]))

    def test_rejects_inconsistent_step_alignment(self):
        recorder = EpisodeTransitionRecorder(
            pathlib.Path("/tmp/unused"),
            suite="suite",
            controller="baseline",
            task_id=0,
            episode_idx=0,
            prompt="task",
            seed=7,
            replan_steps=5,
        )
        with self.assertRaises(ValueError):
            recorder.add(
                image=self._image(0),
                wrist_image=self._image(0),
                future_image=self._image(1),
                future_wrist_image=self._image(1),
                state=np.zeros(8),
                future_state=np.ones(8),
                action_chunk=np.zeros((50, 7)),
                selected_actions=np.zeros((5, 7)),
                executed_steps=2,
                start_step=10,
                future_step=15,
                terminal_within_horizon=False,
            )


if __name__ == "__main__":
    unittest.main()

