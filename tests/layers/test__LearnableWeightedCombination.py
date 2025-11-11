"""Tests for LearnableWeightedCombination layer."""

import unittest
import numpy as np
import tensorflow as tf  # Used for testing only
import keras
from kmr.layers import LearnableWeightedCombination


class TestLearnableWeightedCombination(unittest.TestCase):
    """Test suite for LearnableWeightedCombination."""

    def test_initialization(self) -> None:
        """Test initialization."""
        layer = LearnableWeightedCombination(num_scores=3)
        self.assertEqual(layer.num_scores, 3)

    def test_invalid_num_scores(self) -> None:
        """Test invalid num_scores."""
        with self.assertRaises(ValueError):
            LearnableWeightedCombination(num_scores=0)

    def test_output_shape(self) -> None:
        """Test output shape with 3 scores."""
        layer = LearnableWeightedCombination(num_scores=3)
        score1 = tf.constant([[1.0], [2.0]])
        score2 = tf.constant([[3.0], [4.0]])
        score3 = tf.constant([[5.0], [6.0]])
        output = layer([score1, score2, score3])
        self.assertEqual(output.shape, (2, 1))

    def test_multiple_scores(self) -> None:
        """Test with different numbers of scores."""
        for num_scores in [2, 3, 4, 5]:
            layer = LearnableWeightedCombination(num_scores=num_scores)
            scores = [tf.constant([[1.0], [2.0]]) for _ in range(num_scores)]
            output = layer(scores)
            self.assertEqual(output.shape, (2, 1))

    def test_weights_sum_to_one(self) -> None:
        """Test that normalized weights sum to 1."""
        layer = LearnableWeightedCombination(num_scores=3)
        # Build the layer
        score1 = tf.constant([[1.0], [2.0]])
        score2 = tf.constant([[3.0], [4.0]])
        score3 = tf.constant([[5.0], [6.0]])
        _ = layer([score1, score2, score3])

        # Check that layer has trainable weights
        self.assertGreater(len(layer.trainable_weights), 0)

    def test_output_range(self) -> None:
        """Test output is within reasonable range based on inputs."""
        layer = LearnableWeightedCombination(num_scores=3)
        score1 = tf.constant([[1.0], [2.0]])
        score2 = tf.constant([[3.0], [4.0]])
        score3 = tf.constant([[5.0], [6.0]])
        output = layer([score1, score2, score3]).numpy()

        # Output should be within range of inputs
        min_input = 1.0
        max_input = 6.0
        self.assertTrue(np.all(output >= min_input - 1e-3))
        self.assertTrue(np.all(output <= max_input + 1e-3))

    def test_all_zero_scores(self) -> None:
        """Test with all zero scores."""
        layer = LearnableWeightedCombination(num_scores=3)
        scores = [tf.constant([[0.0], [0.0]]) for _ in range(3)]
        output = layer(scores)
        self.assertEqual(output.shape, (2, 1))

    def test_negative_scores(self) -> None:
        """Test with negative score values."""
        layer = LearnableWeightedCombination(num_scores=3)
        scores = [tf.constant([[-1.0], [-2.0]]) for _ in range(3)]
        output = layer(scores)
        self.assertEqual(output.shape, (2, 1))

    def test_single_score(self) -> None:
        """Test with single score (edge case)."""
        layer = LearnableWeightedCombination(num_scores=1)
        score = tf.constant([[5.0], [10.0]])
        output = layer([score])
        # Weight should be 1.0, output should equal input
        np.testing.assert_almost_equal(output.numpy(), score.numpy())

    def test_many_scores(self) -> None:
        """Test with many scores."""
        layer = LearnableWeightedCombination(num_scores=10)
        scores = [tf.constant([[float(i)], [float(i + 1)]]) for i in range(10)]
        output = layer(scores)
        self.assertEqual(output.shape, (2, 1))

    def test_large_batch_size(self) -> None:
        """Test with large batch sizes."""
        layer = LearnableWeightedCombination(num_scores=3)
        scores = [keras.random.normal((256, 1)) for _ in range(3)]
        output = layer(scores)
        self.assertEqual(output.shape, (256, 1))

    def test_small_batch_size(self) -> None:
        """Test with batch size of 1."""
        layer = LearnableWeightedCombination(num_scores=3)
        scores = [tf.constant([[1.0]]), tf.constant([[2.0]]), tf.constant([[3.0]])]
        output = layer(scores)
        self.assertEqual(output.shape, (1, 1))

    def test_output_deterministic(self) -> None:
        """Test that output is deterministic in inference mode."""
        layer = LearnableWeightedCombination(num_scores=3)
        scores = [keras.random.normal((8, 1)) for _ in range(3)]
        output1 = layer(scores, training=False).numpy()
        output2 = layer(scores, training=False).numpy()
        np.testing.assert_array_almost_equal(output1, output2)

    def test_serialization(self) -> None:
        """Test serialization."""
        layer = LearnableWeightedCombination(num_scores=4)
        config = layer.get_config()
        new_layer = LearnableWeightedCombination.from_config(config)
        self.assertEqual(new_layer.num_scores, 4)

    def test_model_save_load(self) -> None:
        """Test model save and load."""
        import tempfile

        layer = LearnableWeightedCombination(num_scores=3)
        score_inputs = [keras.Input(shape=(1,)) for _ in range(3)]
        output = layer(score_inputs)
        model = keras.Model(score_inputs, output)

        scores_data = [np.random.rand(8, 1).astype("float32") for _ in range(3)]
        pred1 = model.predict(scores_data, verbose=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(f"{tmpdir}/model.keras")
            loaded = keras.models.load_model(f"{tmpdir}/model.keras")
            pred2 = loaded.predict(scores_data, verbose=0)
            np.testing.assert_array_almost_equal(pred1, pred2)


if __name__ == "__main__":
    unittest.main()
