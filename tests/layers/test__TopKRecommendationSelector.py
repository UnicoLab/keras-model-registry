"""Tests for TopKRecommendationSelector layer."""

import unittest
import numpy as np
import keras
from kmr.layers import TopKRecommendationSelector


class TestTopKRecommendationSelector(unittest.TestCase):
    """Test suite for TopKRecommendationSelector."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.layer = TopKRecommendationSelector(k=10)

    def test_initialization_default(self) -> None:
        """Test layer initialization with default k."""
        layer = TopKRecommendationSelector()
        self.assertEqual(layer.k, 10)

    def test_initialization_custom_k(self) -> None:
        """Test layer initialization with custom k."""
        for k in [1, 5, 10, 20]:
            layer = TopKRecommendationSelector(k=k)
            self.assertEqual(layer.k, k)

    def test_initialization_with_name(self) -> None:
        """Test layer initialization with custom name."""
        layer = TopKRecommendationSelector(k=10, name="top_k")
        self.assertEqual(layer.name, "top_k")

    def test_invalid_k_zero(self) -> None:
        """Test that k=0 raises error."""
        with self.assertRaises(ValueError):
            TopKRecommendationSelector(k=0)

    def test_invalid_k_negative(self) -> None:
        """Test that negative k raises error."""
        with self.assertRaises(ValueError):
            TopKRecommendationSelector(k=-1)

    def test_invalid_k_type(self) -> None:
        """Test that non-integer k raises error."""
        with self.assertRaises(ValueError):
            TopKRecommendationSelector(k=10.5)

    def test_output_tuple_structure(self) -> None:
        """Test that output is tuple of indices and scores."""
        scores = keras.random.normal((32, 100))
        indices, top_scores = self.layer(scores)
        self.assertIsInstance(indices, keras.KerasTensor)
        self.assertIsInstance(top_scores, keras.KerasTensor)

    def test_output_shape_k_less_than_items(self) -> None:
        """Test output shape when k < number of items."""
        scores = keras.random.normal((32, 100))
        indices, top_scores = self.layer(scores)
        self.assertEqual(indices.shape, (32, 10))
        self.assertEqual(top_scores.shape, (32, 10))

    def test_output_shape_k_greater_than_items(self) -> None:
        """Test output shape when k > number of items (should adjust)."""
        layer = TopKRecommendationSelector(k=150)
        scores = keras.random.normal((32, 100))
        indices, top_scores = layer(scores)
        # Should return only 100 items
        self.assertEqual(indices.shape[1], 100)
        self.assertEqual(top_scores.shape[1], 100)

    def test_output_shape_k_equals_items(self) -> None:
        """Test output shape when k equals number of items."""
        layer = TopKRecommendationSelector(k=100)
        scores = keras.random.normal((32, 100))
        indices, top_scores = layer(scores)
        self.assertEqual(indices.shape, (32, 100))
        self.assertEqual(top_scores.shape, (32, 100))

    def test_top_scores_ordered_descending(self) -> None:
        """Test that returned scores are in descending order."""
        scores = keras.constant([[5.0, 1.0, 3.0, 4.0, 2.0]])
        layer = TopKRecommendationSelector(k=5)
        indices, top_scores = layer(scores)
        scores_array = top_scores.numpy()[0]
        # Check that scores are sorted in descending order
        self.assertTrue(np.all(scores_array[:-1] >= scores_array[1:]))

    def test_indices_correspond_to_scores(self) -> None:
        """Test that returned indices correspond to original top scores."""
        scores = keras.constant([[1.0, 5.0, 3.0, 2.0, 4.0]])
        layer = TopKRecommendationSelector(k=3)
        indices_out, scores_out = layer(scores)

        # Get original values at returned indices
        scores_array = scores.numpy()[0]
        indices_array = indices_out.numpy()[0]
        scores_out_array = scores_out.numpy()[0]

        # Verify that returned scores match the scores at returned indices
        for i, idx in enumerate(indices_array):
            np.testing.assert_allclose(scores_array[idx], scores_out_array[i])

    def test_output_dtype_preserved(self) -> None:
        """Test that output dtypes are correct."""
        scores = keras.random.normal((32, 100), dtype="float32")
        indices, top_scores = self.layer(scores)
        self.assertEqual(top_scores.dtype, scores.dtype)
        self.assertEqual(indices.dtype, keras.config.floatx())

    def test_single_batch(self) -> None:
        """Test with batch size of 1."""
        scores = keras.random.normal((1, 100))
        indices, top_scores = self.layer(scores)
        self.assertEqual(indices.shape, (1, 10))
        self.assertEqual(top_scores.shape, (1, 10))

    def test_large_batch(self) -> None:
        """Test with large batch size."""
        scores = keras.random.normal((256, 100))
        indices, top_scores = self.layer(scores)
        self.assertEqual(indices.shape, (256, 10))
        self.assertEqual(top_scores.shape, (256, 10))

    def test_k_one(self) -> None:
        """Test with k=1 (top 1 prediction)."""
        scores = keras.constant([[1.0, 5.0, 3.0, 2.0, 4.0]])
        layer = TopKRecommendationSelector(k=1)
        indices, top_scores = layer(scores)
        # Should return the highest score (5.0 at index 1)
        self.assertEqual(indices.numpy()[0, 0], 1)
        np.testing.assert_allclose(top_scores.numpy()[0, 0], 5.0)

    def test_serialization_get_config(self) -> None:
        """Test layer serialization via get_config."""
        layer = TopKRecommendationSelector(k=15)
        config = layer.get_config()
        self.assertEqual(config["k"], 15)

    def test_deserialization_from_config(self) -> None:
        """Test layer deserialization via from_config."""
        layer = TopKRecommendationSelector(k=15)
        config = layer.get_config()
        new_layer = TopKRecommendationSelector.from_config(config)
        self.assertEqual(new_layer.k, 15)

    def test_model_save_load(self) -> None:
        """Test that model with layer can be saved and loaded."""
        import tempfile

        layer = TopKRecommendationSelector(k=10)
        inputs = keras.Input(shape=(100,))
        indices, scores = layer(inputs)
        model = keras.Model(inputs, outputs=[indices, scores])

        x = keras.random.normal((32, 100))
        pred_indices1, pred_scores1 = model.predict(x, verbose=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = f"{tmpdir}/model.keras"
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)
            pred_indices2, pred_scores2 = loaded_model.predict(x, verbose=0)

            np.testing.assert_array_equal(pred_indices1, pred_indices2)
            np.testing.assert_array_almost_equal(pred_scores1, pred_scores2)


if __name__ == "__main__":
    unittest.main()
