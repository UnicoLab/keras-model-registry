"""Unit tests for RecommendationMetricsLogger callback."""

import pytest
import keras
import numpy as np

from kmr.callbacks import RecommendationMetricsLogger


class TestRecommendationMetricsLogger:
    """Test suite for RecommendationMetricsLogger."""

    @pytest.fixture
    def callback(self):
        """Create a fresh callback instance."""
        return RecommendationMetricsLogger(verbose=1, log_frequency=1)

    def test_initialization(self):
        """Test callback initialization."""
        callback = RecommendationMetricsLogger(verbose=1, log_frequency=5)
        assert callback.verbose == 1
        assert callback.log_frequency == 5
        assert callback.name == "RecommendationMetricsLogger"

    def test_initialization_with_custom_name(self):
        """Test initialization with custom name."""
        callback = RecommendationMetricsLogger(name="CustomLogger")
        assert callback.name == "CustomLogger"

    def test_on_epoch_end_stores_metrics(self, callback):
        """Test that metrics are stored on epoch end."""
        logs = {"loss": 0.5, "acc@5": 0.7, "prec@5": 0.8}
        callback.on_epoch_end(epoch=0, logs=logs)

        assert "loss" in callback.epoch_metrics
        assert "acc@5" in callback.epoch_metrics
        assert "prec@5" in callback.epoch_metrics
        assert callback.epoch_metrics["loss"][0] == 0.5

    def test_on_epoch_end_accumulates_metrics(self, callback):
        """Test that metrics accumulate across epochs."""
        for epoch in range(3):
            logs = {"loss": 0.5 - epoch * 0.1, "acc@5": 0.7 + epoch * 0.05}
            callback.on_epoch_end(epoch=epoch, logs=logs)

        assert len(callback.epoch_metrics["loss"]) == 3
        assert len(callback.epoch_metrics["acc@5"]) == 3
        assert callback.epoch_metrics["loss"] == [0.5, 0.4, 0.3]

    def test_log_frequency_respected(self, callback):
        """Test that logging respects frequency setting."""
        callback.log_frequency = 2

        # Should not log at epoch 0 (frequency=2 means log at epochs 1, 3, 5...)
        logs = {"loss": 0.5}
        callback.on_epoch_end(epoch=0, logs=logs)

        # Should log at epoch 1
        callback.on_epoch_end(epoch=1, logs=logs)

    def test_get_config(self, callback):
        """Test get_config for serialization."""
        config = callback.get_config()
        assert config["verbose"] == 1
        assert config["log_frequency"] == 1
        assert config["name"] == "RecommendationMetricsLogger"

    def test_on_train_end_summary(self, callback):
        """Test training end summary generation."""
        # Add some epoch metrics
        for epoch in range(3):
            logs = {"loss": 0.5 - epoch * 0.1, "acc@5": 0.7 + epoch * 0.05}
            callback.on_epoch_end(epoch=epoch, logs=logs)

        # Train end should not raise error
        callback.on_train_end(logs=None)

    def test_handles_validation_metrics(self, callback):
        """Test handling of validation metrics."""
        logs = {
            "loss": 0.5,
            "acc@5": 0.7,
            "val_loss": 0.6,
            "val_acc@5": 0.65,
        }
        callback.on_epoch_end(epoch=0, logs=logs)

        # Should store all metrics
        assert "loss" in callback.epoch_metrics
        assert "val_loss" in callback.epoch_metrics

    def test_empty_logs(self, callback):
        """Test handling of empty logs."""
        callback.on_epoch_end(epoch=0, logs=None)
        assert len(callback.epoch_metrics) == 0

    def test_multiple_metrics(self, callback):
        """Test handling of multiple recommendation metrics."""
        logs = {
            "loss": 0.5,
            "acc@5": 0.7,
            "acc@10": 0.8,
            "prec@5": 0.6,
            "prec@10": 0.65,
            "recall@5": 0.75,
            "recall@10": 0.85,
        }
        callback.on_epoch_end(epoch=0, logs=logs)

        # All metrics should be stored
        expected_metrics = {
            "loss",
            "acc@5",
            "acc@10",
            "prec@5",
            "prec@10",
            "recall@5",
            "recall@10",
        }
        assert set(callback.epoch_metrics.keys()) == expected_metrics


class TestRecommendationMetricsLoggerIntegration:
    """Integration tests with actual Keras training."""

    def test_with_simple_model(self):
        """Test callback with a simple Keras model."""
        # Create a simple model
        model = keras.Sequential(
            [
                keras.layers.Dense(10, activation="relu", input_shape=(5,)),
                keras.layers.Dense(3),
            ],
        )
        model.compile(optimizer="adam", loss="mse")

        # Create callback
        callback = RecommendationMetricsLogger(verbose=0, log_frequency=1)

        # Create dummy data
        x_train = np.random.randn(32, 5).astype(np.float32)
        y_train = np.random.randn(32, 3).astype(np.float32)

        # Train with callback
        history = model.fit(
            x_train,
            y_train,
            epochs=2,
            batch_size=8,
            callbacks=[callback],
            verbose=0,
        )

        # Verify callback stored metrics
        assert len(callback.epoch_metrics["loss"]) == 2

    def test_with_validation_data(self):
        """Test callback with validation data."""
        model = keras.Sequential(
            [
                keras.layers.Dense(10, activation="relu", input_shape=(5,)),
                keras.layers.Dense(3),
            ],
        )
        model.compile(optimizer="adam", loss="mse")

        callback = RecommendationMetricsLogger(verbose=0)

        x_train = np.random.randn(32, 5).astype(np.float32)
        y_train = np.random.randn(32, 3).astype(np.float32)
        x_val = np.random.randn(16, 5).astype(np.float32)
        y_val = np.random.randn(16, 3).astype(np.float32)

        model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=2,
            batch_size=8,
            callbacks=[callback],
            verbose=0,
        )

        # Should have both training and validation metrics
        assert "loss" in callback.epoch_metrics
        assert "val_loss" in callback.epoch_metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
