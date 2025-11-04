"""Tests for SpatialFeatureClustering layer."""

import unittest
import numpy as np
import keras
from kmr.layers import SpatialFeatureClustering


class TestSpatialFeatureClustering(unittest.TestCase):
    """Test suite for SpatialFeatureClustering."""

    def test_initialization_default(self) -> None:
        """Test layer initialization with default parameters."""
        layer = SpatialFeatureClustering()
        self.assertEqual(layer.n_clusters, 5)

    def test_initialization_custom_clusters(self) -> None:
        """Test layer initialization with custom cluster count."""
        layer = SpatialFeatureClustering(n_clusters=10)
        self.assertEqual(layer.n_clusters, 10)

    def test_invalid_clusters_zero(self) -> None:
        """Test that zero clusters raises error."""
        with self.assertRaises(ValueError):
            SpatialFeatureClustering(n_clusters=0)

    def test_output_shape(self) -> None:
        """Test output shape matches cluster count."""
        distances = keras.random.uniform((32, 32))
        layer = SpatialFeatureClustering(n_clusters=5)
        clusters = layer(distances)
        self.assertEqual(clusters.shape, (32, 5))

    def test_cluster_probabilities(self) -> None:
        """Test that outputs are valid probability distributions."""
        distances = keras.random.uniform((16, 16))
        layer = SpatialFeatureClustering(n_clusters=5)
        clusters = layer(distances).numpy()
        # Each row should sum to approximately 1 (probabilities)
        np.testing.assert_array_almost_equal(clusters.sum(axis=1), 1.0, decimal=5)

    def test_training_mode(self) -> None:
        """Test layer behavior in training mode."""
        distances = keras.random.uniform((16, 16))
        layer = SpatialFeatureClustering(n_clusters=5)
        clusters_train = layer(distances, training=True)
        clusters_infer = layer(distances, training=False)
        self.assertEqual(clusters_train.shape, clusters_infer.shape)

    def test_serialization(self) -> None:
        """Test layer serialization."""
        layer = SpatialFeatureClustering(n_clusters=8)
        config = layer.get_config()
        new_layer = SpatialFeatureClustering.from_config(config)
        self.assertEqual(new_layer.n_clusters, 8)


if __name__ == "__main__":
    unittest.main()
