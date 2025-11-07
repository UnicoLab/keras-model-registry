"""Unit tests for GeospatialMarginLoss.

Tests cover:
- Initialization with various parameters
- Loss computation with sample data
- Distance penalty calculation
- Edge cases (no positives, no negatives, equal distances)
- Serialization (save/load config)
- Zero and extreme distances
- Integration with ImprovedMarginRankingLoss
"""

import numpy as np
import pytest
import keras
from keras import ops

from kmr.losses import GeospatialMarginLoss


class TestGeospatialMarginLossInitialization:
    """Test GeospatialMarginLoss initialization."""

    def test_initialization_default_params(self):
        """Test initialization with default parameters."""
        loss = GeospatialMarginLoss()

        assert loss.margin == 1.0
        assert loss.distance_weight == 0.1
        assert loss.max_min_weight == 0.7
        assert loss.avg_weight == 0.3
        assert loss.name == "geospatial_margin_loss"

    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        loss = GeospatialMarginLoss(
            margin=2.0,
            distance_weight=0.5,
            max_min_weight=0.6,
            avg_weight=0.4,
            name="custom_geo_loss",
        )

        assert loss.margin == 2.0
        assert loss.distance_weight == 0.5
        assert loss.max_min_weight == 0.6
        assert loss.avg_weight == 0.4
        assert loss.name == "custom_geo_loss"

    def test_initialization_inherits_from_improved_margin_loss(self):
        """Test that GeospatialMarginLoss properly inherits from ImprovedMarginRankingLoss."""
        loss = GeospatialMarginLoss()

        # Should have parent class components
        assert hasattr(loss, "max_min_loss")
        assert hasattr(loss, "avg_loss")
        assert hasattr(loss, "distance_weight")

    def test_initialization_negative_distance_weight(self):
        """Test initialization with negative distance weight (edge case)."""
        # Should allow negative weight to penalize far items and reward close ones
        loss = GeospatialMarginLoss(distance_weight=-0.1)
        assert loss.distance_weight == -0.1

    def test_initialization_zero_distance_weight(self):
        """Test initialization with zero distance weight (degenerates to parent loss)."""
        loss = GeospatialMarginLoss(distance_weight=0.0)
        assert loss.distance_weight == 0.0


class TestGeospatialMarginLossComputation:
    """Test GeospatialMarginLoss computation."""

    def test_loss_computation_concatenated_format(self):
        """Test loss computation with concatenated [similarities, distances] format."""
        loss_fn = GeospatialMarginLoss(margin=1.0, distance_weight=0.1)

        # Create sample data: batch_size=2, num_items=5
        y_true = keras.ops.array([[1.0, 0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0, 0.0]])
        similarities = keras.ops.array(
            [[0.8, 0.2, 0.7, 0.1, 0.0], [0.1, 0.9, 0.2, 0.8, 0.0]],
        )
        distances = keras.ops.array(
            [[0.1, 0.5, 0.2, 0.8, 0.9], [0.5, 0.1, 0.7, 0.2, 0.9]],
        )

        # Concatenate: [similarities, distances]
        y_pred = keras.ops.concatenate([similarities, distances], axis=-1)

        loss_value = loss_fn(y_true, y_pred)

        # Loss should be a scalar positive value
        assert loss_value.shape == ()
        assert ops.convert_to_numpy(loss_value) > 0.0

    def test_loss_computation_single_distance_format(self):
        """Test loss computation with single distance format."""
        loss_fn = GeospatialMarginLoss(margin=1.0, distance_weight=0.1)

        # Create sample data
        y_true = keras.ops.array([[1.0, 0.0, 1.0, 0.0, 0.0]])
        similarities = keras.ops.array([[0.8, 0.2, 0.7, 0.1, 0.0]])
        mean_distance = keras.ops.array([[0.3]])

        # Format: [similarities, mean_distance] - last column is distance
        y_pred = keras.ops.concatenate([similarities, mean_distance], axis=-1)

        loss_value = loss_fn(y_true, y_pred)

        # Loss should be a scalar positive value
        assert loss_value.shape == ()
        assert ops.convert_to_numpy(loss_value) > 0.0

    def test_loss_decreases_with_small_distances(self):
        """Test that loss decreases when positive items are close (small distances)."""
        loss_fn = GeospatialMarginLoss(
            margin=1.0,
            distance_weight=0.1,
            max_min_weight=0.0,
            avg_weight=1.0,
        )

        y_true = keras.ops.array([[1.0, 0.0, 1.0, 0.0, 0.0]])
        similarities = keras.ops.array([[0.8, 0.2, 0.7, 0.1, 0.0]])

        # Case 1: Small distances for positive items
        small_distances = keras.ops.array([[0.01, 0.5, 0.02, 0.5, 0.5]])
        y_pred_small = keras.ops.concatenate([similarities, small_distances], axis=-1)
        loss_small = loss_fn(y_true, y_pred_small)

        # Case 2: Large distances for positive items
        large_distances = keras.ops.array([[0.99, 0.1, 0.98, 0.1, 0.1]])
        y_pred_large = keras.ops.concatenate([similarities, large_distances], axis=-1)
        loss_large = loss_fn(y_true, y_pred_large)

        # Loss should be lower with smaller distances
        assert ops.convert_to_numpy(loss_small) < ops.convert_to_numpy(loss_large)

    def test_loss_computation_all_positives(self):
        """Test loss computation when all items are positive."""
        loss_fn = GeospatialMarginLoss(margin=1.0, distance_weight=0.1)

        y_true = keras.ops.array([[1.0, 1.0, 1.0, 1.0, 1.0]])
        similarities = keras.ops.array([[0.8, 0.8, 0.8, 0.8, 0.8]])
        distances = keras.ops.array([[0.1, 0.2, 0.3, 0.4, 0.5]])

        y_pred = keras.ops.concatenate([similarities, distances], axis=-1)
        loss_value = loss_fn(y_true, y_pred)

        assert loss_value.shape == ()
        assert ops.convert_to_numpy(loss_value) >= 0.0

    def test_loss_computation_all_negatives(self):
        """Test loss computation when all items are negative."""
        loss_fn = GeospatialMarginLoss(margin=1.0, distance_weight=0.1)

        y_true = keras.ops.array([[0.0, 0.0, 0.0, 0.0, 0.0]])
        similarities = keras.ops.array([[0.8, 0.8, 0.8, 0.8, 0.8]])
        distances = keras.ops.array([[0.1, 0.2, 0.3, 0.4, 0.5]])

        y_pred = keras.ops.concatenate([similarities, distances], axis=-1)
        loss_value = loss_fn(y_true, y_pred)

        assert loss_value.shape == ()
        assert ops.convert_to_numpy(loss_value) >= 0.0

    def test_loss_with_batch_data(self):
        """Test loss computation with batch data."""
        import tensorflow as tf

        loss_fn = GeospatialMarginLoss(margin=1.0, distance_weight=0.1)

        batch_size = 32
        num_items = 100

        y_true = keras.ops.cast(
            tf.random.uniform((batch_size, num_items)) > 0.7,
            dtype="float32",
        )
        similarities = tf.random.uniform((batch_size, num_items))
        distances = tf.random.uniform((batch_size, num_items))

        y_pred = keras.ops.concatenate([similarities, distances], axis=-1)
        loss_value = loss_fn(y_true, y_pred)

        assert loss_value.shape == ()
        assert ops.convert_to_numpy(loss_value) >= 0.0


class TestGeospatialMarginLossDistancePenalty:
    """Test distance penalty calculation."""

    def test_distance_penalty_zero_distances(self):
        """Test distance penalty with zero distances."""
        loss_fn = GeospatialMarginLoss(distance_weight=1.0)

        y_true = keras.ops.array([[1.0, 0.0, 1.0, 0.0, 0.0]])
        distances = keras.ops.zeros((1, 5))

        penalty = loss_fn._compute_distance_penalty(y_true, distances)

        # Penalty should be zero when all distances are zero
        assert ops.convert_to_numpy(penalty) == pytest.approx(0.0, abs=1e-6)

    def test_distance_penalty_computation(self):
        """Test correct computation of distance penalty."""
        loss_fn = GeospatialMarginLoss(distance_weight=1.0)

        # Positive items: 0, 2; Negative items: 1, 3, 4
        y_true = keras.ops.array([[1.0, 0.0, 1.0, 0.0, 0.0]])
        distances = keras.ops.array([[0.1, 0.9, 0.2, 0.9, 0.9]])

        penalty = loss_fn._compute_distance_penalty(y_true, distances)

        # Expected: (1.0*0.1 + 1.0*0.2) / (1.0 + 1.0) = 0.3 / 2.0 = 0.15
        expected = 0.15
        assert ops.convert_to_numpy(penalty) == pytest.approx(expected, abs=1e-6)

    def test_distance_penalty_batch_computation(self):
        """Test distance penalty with batch data."""
        loss_fn = GeospatialMarginLoss(distance_weight=1.0)

        y_true = keras.ops.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        distances = keras.ops.array([[0.1, 0.5, 0.2], [0.8, 0.1, 0.9]])

        penalty = loss_fn._compute_distance_penalty(y_true, distances)

        # Batch penalties: [(0.1 + 0.2) / 2, 0.1 / 1] = [0.15, 0.1]
        # Mean: (0.15 + 0.1) / 2 = 0.125
        expected = 0.125
        assert ops.convert_to_numpy(penalty) == pytest.approx(expected, abs=1e-6)

    def test_distance_penalty_no_positives(self):
        """Test distance penalty when there are no positive items."""
        loss_fn = GeospatialMarginLoss(distance_weight=1.0)

        y_true = keras.ops.array([[0.0, 0.0, 0.0, 0.0, 0.0]])
        distances = keras.ops.array([[0.1, 0.2, 0.3, 0.4, 0.5]])

        penalty = loss_fn._compute_distance_penalty(y_true, distances)

        # When no positives, penalty should approach 0 (prevented by epsilon)
        assert ops.convert_to_numpy(penalty) >= 0.0
        assert ops.convert_to_numpy(penalty) < 1e-6  # Should be very small


class TestGeospatialMarginLossEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_y_pred_shape(self):
        """Test error when y_pred has invalid shape."""
        loss_fn = GeospatialMarginLoss()

        y_true = keras.ops.array([[1.0, 0.0, 1.0, 0.0, 0.0]])
        y_pred = keras.ops.array([[0.8, 0.2, 0.7, 0.1, 0.0]])  # Missing distances

        with pytest.raises(ValueError, match="Invalid y_pred shape"):
            loss_fn(y_true, y_pred)

    def test_nan_handling(self):
        """Test handling of NaN values in distances."""
        loss_fn = GeospatialMarginLoss(distance_weight=0.1)

        y_true = keras.ops.array([[1.0, 0.0, 1.0, 0.0, 0.0]])
        similarities = keras.ops.array([[0.8, 0.2, 0.7, 0.1, 0.0]])
        distances = keras.ops.array([[0.1, float("nan"), 0.2, 0.8, 0.9]])

        y_pred = keras.ops.concatenate([similarities, distances], axis=-1)

        # Should handle NaN gracefully (may produce NaN loss or handle it)
        loss_value = loss_fn(y_true, y_pred)
        assert loss_value.shape == ()

    def test_inf_handling(self):
        """Test handling of infinite values in distances."""
        loss_fn = GeospatialMarginLoss(distance_weight=0.1)

        y_true = keras.ops.array([[1.0, 0.0, 1.0, 0.0, 0.0]])
        similarities = keras.ops.array([[0.8, 0.2, 0.7, 0.1, 0.0]])
        distances = keras.ops.array([[0.1, float("inf"), 0.2, 0.8, 0.9]])

        y_pred = keras.ops.concatenate([similarities, distances], axis=-1)

        # Should handle infinity gracefully
        loss_value = loss_fn(y_true, y_pred)
        assert loss_value.shape == ()

    def test_very_small_distance_weight(self):
        """Test with very small distance weight."""
        loss_fn = GeospatialMarginLoss(distance_weight=1e-8)

        y_true = keras.ops.array([[1.0, 0.0, 1.0, 0.0, 0.0]])
        similarities = keras.ops.array([[0.8, 0.2, 0.7, 0.1, 0.0]])
        distances = keras.ops.array([[0.1, 0.5, 0.2, 0.8, 0.9]])

        y_pred = keras.ops.concatenate([similarities, distances], axis=-1)
        loss_value = loss_fn(y_true, y_pred)

        assert loss_value.shape == ()
        assert ops.convert_to_numpy(loss_value) > 0.0

    def test_very_large_distance_weight(self):
        """Test with very large distance weight."""
        loss_fn = GeospatialMarginLoss(distance_weight=100.0)

        y_true = keras.ops.array([[1.0, 0.0, 1.0, 0.0, 0.0]])
        similarities = keras.ops.array([[0.8, 0.2, 0.7, 0.1, 0.0]])
        distances = keras.ops.array([[0.1, 0.5, 0.2, 0.8, 0.9]])

        y_pred = keras.ops.concatenate([similarities, distances], axis=-1)
        loss_value = loss_fn(y_true, y_pred)

        assert loss_value.shape == ()
        assert ops.convert_to_numpy(loss_value) > 0.0


class TestGeospatialMarginLossSerialization:
    """Test serialization and deserialization."""

    def test_get_config(self):
        """Test get_config returns correct configuration."""
        loss = GeospatialMarginLoss(
            margin=1.5,
            distance_weight=0.2,
            max_min_weight=0.6,
            avg_weight=0.4,
        )

        config = loss.get_config()

        assert config["margin"] == 1.5
        assert config["distance_weight"] == 0.2
        assert config["max_min_weight"] == 0.6
        assert config["avg_weight"] == 0.4

    def test_from_config(self):
        """Test creating loss from config."""
        original_loss = GeospatialMarginLoss(
            margin=2.0,
            distance_weight=0.15,
            max_min_weight=0.65,
            avg_weight=0.35,
        )

        config = original_loss.get_config()
        restored_loss = GeospatialMarginLoss.from_config(config)

        assert restored_loss.margin == original_loss.margin
        assert restored_loss.distance_weight == original_loss.distance_weight
        assert restored_loss.max_min_weight == original_loss.max_min_weight
        assert restored_loss.avg_weight == original_loss.avg_weight

    def test_serialization_roundtrip(self):
        """Test full serialization and deserialization."""
        loss_fn = GeospatialMarginLoss(
            margin=1.2,
            distance_weight=0.25,
            max_min_weight=0.65,
            avg_weight=0.35,
        )

        y_true = keras.ops.array([[1.0, 0.0, 1.0, 0.0, 0.0]])
        similarities = keras.ops.array([[0.8, 0.2, 0.7, 0.1, 0.0]])
        distances = keras.ops.array([[0.1, 0.5, 0.2, 0.8, 0.9]])
        y_pred = keras.ops.concatenate([similarities, distances], axis=-1)

        # Compute loss with original
        original_loss = loss_fn(y_true, y_pred)

        # Serialize and deserialize
        config = loss_fn.get_config()
        restored_loss_fn = GeospatialMarginLoss.from_config(config)

        # Compute loss with restored
        restored_loss = restored_loss_fn(y_true, y_pred)

        # Should produce same loss value
        assert ops.convert_to_numpy(original_loss) == pytest.approx(
            ops.convert_to_numpy(restored_loss),
            rel=1e-5,
        )


class TestGeospatialMarginLossIntegration:
    """Test integration with Keras models."""

    def test_integration_with_model_compile(self):
        """Test that loss can be used in model compilation."""
        # Create simple model
        inputs = keras.Input(shape=(10,))
        outputs = keras.layers.Dense(5, activation="sigmoid")(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)

        # Compile with GeospatialMarginLoss
        loss_fn = GeospatialMarginLoss(distance_weight=0.1)
        model.compile(optimizer="adam", loss=loss_fn)

        # Should compile without error
        assert model.loss == loss_fn or isinstance(model.loss, GeospatialMarginLoss)

    def test_loss_preserves_parent_behavior_with_zero_distance_weight(self):
        """Test that loss matches parent class when distance_weight=0."""
        from kmr.losses import ImprovedMarginRankingLoss

        parent_loss_fn = ImprovedMarginRankingLoss(
            margin=1.0,
            max_min_weight=0.7,
            avg_weight=0.3,
        )
        geo_loss_fn = GeospatialMarginLoss(
            margin=1.0,
            distance_weight=0.0,
            max_min_weight=0.7,
            avg_weight=0.3,
        )

        y_true = keras.ops.array([[1.0, 0.0, 1.0, 0.0, 0.0]])
        similarities = keras.ops.array([[0.8, 0.2, 0.7, 0.1, 0.0]])

        # For geospatial loss with zero distance_weight, need to add dummy distances
        dummy_distances = keras.ops.zeros((1, 5))
        y_pred = keras.ops.concatenate([similarities, dummy_distances], axis=-1)

        parent_loss = parent_loss_fn(y_true, similarities)
        geo_loss = geo_loss_fn(y_true, y_pred)

        # Losses should be approximately equal
        assert ops.convert_to_numpy(parent_loss) == pytest.approx(
            ops.convert_to_numpy(geo_loss),
            rel=1e-5,
        )

    def test_gradient_computation(self):
        """Test that gradients can be computed through the loss."""
        import tensorflow as tf

        loss_fn = GeospatialMarginLoss(distance_weight=0.1)

        y_true = tf.constant([[1.0, 0.0, 1.0, 0.0, 0.0]], dtype="float32")
        y_pred = tf.Variable(
            [[0.8, 0.2, 0.7, 0.1, 0.0, 0.1, 0.5, 0.2, 0.8, 0.9]],
            trainable=True,
            dtype="float32",
        )

        with tf.GradientTape() as tape:
            loss_value = loss_fn(y_true, y_pred)

        # Gradients should be computable
        assert loss_value.shape == ()
        grad = tape.gradient(loss_value, y_pred)
        assert grad is not None
        assert grad.shape == y_pred.shape


class TestGeospatialMarginLossNumericalStability:
    """Test numerical stability of the loss."""

    def test_stability_with_large_batch(self):
        """Test numerical stability with large batch sizes."""
        import tensorflow as tf

        loss_fn = GeospatialMarginLoss(distance_weight=0.1)

        batch_size = 1000
        num_items = 500

        y_true = keras.ops.cast(
            tf.random.uniform((batch_size, num_items)) > 0.7,
            dtype="float32",
        )
        similarities = tf.random.uniform((batch_size, num_items))
        distances = tf.random.uniform((batch_size, num_items))

        y_pred = keras.ops.concatenate([similarities, distances], axis=-1)

        loss_value = loss_fn(y_true, y_pred)

        # Loss should be finite and valid
        assert not ops.convert_to_numpy(ops.isnan(loss_value))
        assert not ops.convert_to_numpy(ops.isinf(loss_value))

    def test_stability_with_extreme_values(self):
        """Test numerical stability with extreme similarity values."""
        loss_fn = GeospatialMarginLoss(distance_weight=0.1)

        y_true = keras.ops.array([[1.0, 0.0, 1.0, 0.0, 0.0]])
        similarities = keras.ops.array([[1e3, 1e-3, 1e3, 1e-3, 1e-3]])
        distances = keras.ops.array([[0.1, 0.5, 0.2, 0.8, 0.9]])

        y_pred = keras.ops.concatenate([similarities, distances], axis=-1)
        loss_value = loss_fn(y_true, y_pred)

        # Loss should still be valid
        assert loss_value.shape == ()
        assert not ops.convert_to_numpy(ops.isnan(loss_value))


class TestGeospatialMarginLossTupleInput:
    """Test GeospatialMarginLoss with unified tuple output format."""

    def test_tuple_input_format(self):
        """Test that loss handles tuple input from unified model output."""
        loss_fn = GeospatialMarginLoss(margin=1.0, distance_weight=0.1)

        y_true = keras.ops.array([[1.0, 0.0, 1.0, 0.0, 0.0]])
        concatenated = keras.ops.array(
            [[0.8, 0.2, 0.7, 0.1, 0.0, 0.1, 0.5, 0.2, 0.8, 0.9]],
        )
        indices = keras.ops.array([[0, 2]], dtype="int32")
        scores = keras.ops.array([[0.8, 0.7]])

        # Create tuple output format (masked_scores, indices, scores, masks)
        y_pred_tuple = (concatenated, indices, scores, None)

        # Loss should extract concatenated and compute correctly
        loss_value_tuple = loss_fn(y_true, y_pred_tuple)
        loss_value_direct = loss_fn(y_true, concatenated)

        # Both should be equivalent
        assert ops.convert_to_numpy(loss_value_tuple) == pytest.approx(
            ops.convert_to_numpy(loss_value_direct),
            rel=1e-5,
        )

    def test_backward_compatibility_concatenated(self):
        """Test backward compatibility with raw concatenated format."""
        loss_fn = GeospatialMarginLoss(distance_weight=0.1)

        y_true = keras.ops.array([[1.0, 0.0, 1.0, 0.0, 0.0]])
        similarities = keras.ops.array([[0.8, 0.2, 0.7, 0.1, 0.0]])
        distances = keras.ops.array([[0.1, 0.5, 0.2, 0.8, 0.9]])
        y_pred = keras.ops.concatenate([similarities, distances], axis=-1)

        # Should work without error
        loss_value = loss_fn(y_true, y_pred)
        assert loss_value.shape == ()
        assert not ops.convert_to_numpy(ops.isnan(loss_value))
