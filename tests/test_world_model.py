import unittest
import pathlib

try:
    import torch
    from openpi_libero_reproduction.world_model import LatentChangeCritic
    from openpi_libero_reproduction.world_model import critic_loss
    from openpi_libero_reproduction.world_model import hashed_text_features
    from openpi_libero_reproduction.world_model_data import split_episode_shards
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch/torchvision are not installed")
class LatentChangeCriticTest(unittest.TestCase):
    def test_forward_loss_and_score(self):
        model = LatentChangeCritic(
            latent_dim=16,
            state_dim=8,
            action_horizon=5,
            action_dim=7,
            text_dim=32,
            hidden_dim=24,
            dropout=0.0,
        )
        prompts = hashed_text_features(["pick mug", "open drawer"], dimension=32)
        output = model(
            torch.zeros(2, 16),
            torch.zeros(2, 8),
            torch.zeros(2, 5, 7),
            prompts,
        )
        self.assertEqual(tuple(output.predicted_future_latent.shape), (2, 16))
        self.assertEqual(tuple(model.score(output).shape), (2,))
        loss, metrics = critic_loss(
            output,
            future_latent=torch.ones(2, 16),
            terminal_target=torch.tensor([0.0, 1.0]),
            success_target=torch.tensor([1.0, 0.0]),
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(metrics), {"latent_loss", "terminal_loss", "success_loss"})
        loss.backward()

    def test_hash_features_are_deterministic(self):
        first = hashed_text_features(["Pick the red mug"], dimension=16)
        second = hashed_text_features(["Pick the red mug"], dimension=16)
        torch.testing.assert_close(first, second)
        self.assertAlmostEqual(float(first.norm()), 1.0)

    def test_episode_split_is_deterministic_and_disjoint(self):
        paths = [pathlib.Path(f"episode_{index}.npz") for index in range(20)]
        first = split_episode_shards(paths)
        second = split_episode_shards(paths)
        self.assertEqual(first, second)
        self.assertEqual(set(first[0]) & set(first[1]), set())
        self.assertEqual(set(first[0]) & set(first[2]), set())
        self.assertEqual(set(first[1]) & set(first[2]), set())
        self.assertEqual(set().union(*map(set, first)), set(paths))

    def test_dataset_rejects_unknown_action_source(self):
        from openpi_libero_reproduction.world_model_data import LatentTransitionDataset

        with self.assertRaises(ValueError):
            LatentTransitionDataset([], action_key="unknown")


if __name__ == "__main__":
    unittest.main()
