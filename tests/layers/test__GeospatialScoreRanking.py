"""Tests for GeospatialScoreRanking layer."""

import unittest
import numpy as np
import keras
from kmr.layers import GeospatialScoreRanking


class TestGeospatialScoreRanking(unittest.TestCase):
    """Test suite for GeospatialScoreRanking."""

    def test_initialization_default(self) -> None:
        """Test layer initialization with default parameters."""
        layer = GeospatialScoreRanking()
        self.assertEqual(layer.embedding_dim, 32)
        self.assertEqual(layer.input_dim, 5)

    def test_initialization_custom(self) -> None:
        """Test layer initialization with custom parameters."""
        layer = GeospatialScoreRanking(embedding_dim=64, input_dim=10)
        self.assertEqual(layer.embedding_dim, 64)
        self.assertEqual(layer.input_dim, 10)

    def test_invalid_embedding_dim(self) -> None:
        """Test that invalid embedding_dim raises error."""
        with self.assertRaises(ValueError):
            GeospatialScoreRanking(embedding_dim=0)

    def test_output_shape(self) -> None:
        """Test output shape is ranking score matrix."""
        clusters = keras.random.uniform((32, 5))
        layer = GeospatialScoreRanking(embedding_dim=32, input_dim=5)
        scores = layer(clusters)
        self.assertEqual(scores.shape, (32, 32))

    def test_score_range(self) -> None:
        """Test that scores are in [0, 1] due to sigmoid."""
        clusters = keras.random.uniform((16, 5))
        layer = GeospatialScoreRanking(embedding_dim=32, input_dim=5)
        scores = layer(clusters).numpy()
        self.assertTrue(np.all(scores >= 0))
        self.assertTrue(np.all(scores <= 1))

    def test_training_mode(self) -> None:
        """Test layer behavior in training vs inference mode."""
        clusters = keras.random.uniform((16, 5))
        layer = GeospatialScoreRanking(embedding_dim=32, input_dim=5)
        scores_train = layer(clusters, training=True)
        scores_infer = layer(clusters, training=False)
        self.assertEqual(scores_train.shape, scores_infer.shape)

    def test_serialization(self) -> None:
        """Test layer serialization."""
        layer = GeospatialScoreRanking(embedding_dim=64, input_dim=10)
        config = layer.get_config()
        new_layer = GeospatialScoreRanking.from_config(config)
        self.assertEqual(new_layer.embedding_dim, 64)
        self.assertEqual(new_layer.input_dim, 10)


if __name__ == "__main__":
    unittest.main()
