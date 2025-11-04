"""Tests for HaversineGeospatialDistance layer."""

import unittest
import numpy as np
import keras
from kmr.layers import HaversineGeospatialDistance


class TestHaversineGeospatialDistance(unittest.TestCase):
    """Test suite for HaversineGeospatialDistance."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.layer = HaversineGeospatialDistance(earth_radius=6371.0)

    def test_initialization_default(self) -> None:
        """Test layer initialization with default parameters."""
        layer = HaversineGeospatialDistance()
        self.assertEqual(layer.earth_radius, 6371.0)

    def test_initialization_custom_radius(self) -> None:
        """Test layer initialization with custom earth radius."""
        layer = HaversineGeospatialDistance(earth_radius=6378.0)
        self.assertEqual(layer.earth_radius, 6378.0)

    def test_invalid_radius_zero(self) -> None:
        """Test that zero radius raises error."""
        with self.assertRaises(ValueError):
            HaversineGeospatialDistance(earth_radius=0)

    def test_invalid_radius_negative(self) -> None:
        """Test that negative radius raises error."""
        with self.assertRaises(ValueError):
            HaversineGeospatialDistance(earth_radius=-1)

    def test_output_shape(self) -> None:
        """Test output shape is distance matrix."""
        lat1 = keras.random.uniform((32,), minval=-np.pi / 2, maxval=np.pi / 2)
        lon1 = keras.random.uniform((32,), minval=-np.pi, maxval=np.pi)
        lat2 = keras.random.uniform((32,), minval=-np.pi / 2, maxval=np.pi / 2)
        lon2 = keras.random.uniform((32,), minval=-np.pi, maxval=np.pi)

        distances = self.layer([lat1, lon1, lat2, lon2])
        self.assertEqual(distances.shape, (32, 32))

    def test_normalized_distances(self) -> None:
        """Test that distances are normalized to [0, 1]."""
        lat1 = keras.random.uniform((16,), minval=-np.pi / 2, maxval=np.pi / 2)
        lon1 = keras.random.uniform((16,), minval=-np.pi, maxval=np.pi)
        lat2 = keras.random.uniform((16,), minval=-np.pi / 2, maxval=np.pi / 2)
        lon2 = keras.random.uniform((16,), minval=-np.pi, maxval=np.pi)

        distances = self.layer([lat1, lon1, lat2, lon2]).numpy()
        self.assertTrue(np.all(distances >= 0))
        self.assertTrue(np.all(distances <= 1))

    def test_distance_symmetry(self) -> None:
        """Test that distance matrix is approximately symmetric."""
        lat1 = keras.constant([0.0, 0.1, 0.2])
        lon1 = keras.constant([0.0, 0.1, 0.2])
        lat2 = keras.constant([0.0, 0.1, 0.2])
        lon2 = keras.constant([0.0, 0.1, 0.2])

        distances = self.layer([lat1, lon1, lat2, lon2]).numpy()
        # Distances should be approximately symmetric (D[i,j] ≈ D[j,i])
        np.testing.assert_array_almost_equal(distances, distances.T, decimal=5)

    def test_zero_distance_same_coordinates(self) -> None:
        """Test that distance between same coordinates is near zero."""
        lat = keras.constant([0.0, 0.5])
        lon = keras.constant([0.0, 0.5])

        distances = self.layer([lat, lon, lat, lon]).numpy()
        # Diagonal should be close to 0 or 1 after normalization
        np.testing.assert_almost_equal(distances[0, 0], 0.0, decimal=1)

    def test_serialization(self) -> None:
        """Test layer serialization."""
        layer = HaversineGeospatialDistance(earth_radius=6378.0)
        config = layer.get_config()
        new_layer = HaversineGeospatialDistance.from_config(config)
        self.assertEqual(new_layer.earth_radius, 6378.0)

    def test_model_save_load(self) -> None:
        """Test model save and load with layer."""
        import tempfile

        inputs = [
            keras.Input(shape=(32,)),
            keras.Input(shape=(32,)),
            keras.Input(shape=(32,)),
            keras.Input(shape=(32,)),
        ]
        outputs = self.layer(inputs)
        model = keras.Model(inputs, outputs)

        lat1 = keras.random.uniform((16, 32), minval=-np.pi / 2, maxval=np.pi / 2)
        lon1 = keras.random.uniform((16, 32), minval=-np.pi, maxval=np.pi)
        lat2 = keras.random.uniform((16, 32), minval=-np.pi / 2, maxval=np.pi / 2)
        lon2 = keras.random.uniform((16, 32), minval=-np.pi, maxval=np.pi)

        pred1 = model.predict([lat1, lon1, lat2, lon2], verbose=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = f"{tmpdir}/model.keras"
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)
            pred2 = loaded_model.predict([lat1, lon1, lat2, lon2], verbose=0)
            np.testing.assert_array_almost_equal(pred1, pred2)


if __name__ == "__main__":
    unittest.main()
