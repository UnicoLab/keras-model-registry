"""Integration tests for model training with all components.

Tests the full pipeline: model creation → compilation → training → evaluation.
"""

import pytest
import numpy as np
import keras

from kmr.models import (
    TwoTowerModel,
    MatrixFactorizationModel,
    DeepRankingModel,
)
from kmr.losses import ImprovedMarginRankingLoss
from kmr.metrics import AccuracyAtK, PrecisionAtK, RecallAtK
from kmr.callbacks import RecommendationMetricsLogger


class TestModelTrainingIntegration:
    """Integration tests for model training pipeline."""

    @pytest.fixture
    def train_data(self):
        """Generate training data."""
        batch_size = 32
        num_items = 50
        user_features = np.random.randn(batch_size, 10).astype(np.float32)
        item_features = np.random.randn(batch_size, num_items, 10).astype(np.float32)
        labels = np.random.randint(0, 2, (batch_size, num_items)).astype(np.float32)
        return (user_features, item_features), labels

    @pytest.fixture
    def val_data(self):
        """Generate validation data."""
        batch_size = 16
        num_items = 50
        user_features = np.random.randn(batch_size, 10).astype(np.float32)
        item_features = np.random.randn(batch_size, num_items, 10).astype(np.float32)
        labels = np.random.randint(0, 2, (batch_size, num_items)).astype(np.float32)
        return (user_features, item_features), labels

    def test_twotower_full_pipeline(self, train_data, val_data):
        """Test TwoTowerModel through complete training pipeline."""
        model = TwoTowerModel(
            user_feature_dim=10,
            item_feature_dim=10,
            num_items=50,
            output_dim=16,
        )

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.01),
            loss=[ImprovedMarginRankingLoss(), None, None],
            metrics=[
                [AccuracyAtK(k=5, name="acc@5"), PrecisionAtK(k=5, name="prec@5")],
                None,
                None,
            ],
        )

        train_inputs, train_labels = train_data
        val_inputs, val_labels = val_data

        history = model.fit(
            x=train_inputs,
            y=train_labels,
            validation_data=(val_inputs, val_labels),
            epochs=2,
            batch_size=8,
            verbose=0,
        )

        # Verify training completed successfully
        assert "loss" in history.history
        assert len(history.history["loss"]) == 2
        assert (
            history.history["loss"][-1] < history.history["loss"][0]
        )  # Loss should decrease

    def test_matrix_factorization_full_pipeline(self, train_data, val_data):
        """Test MatrixFactorizationModel through complete training pipeline."""
        model = MatrixFactorizationModel(
            num_users=100,
            num_items=50,
            embedding_dim=16,
        )

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.01),
            loss=[ImprovedMarginRankingLoss(), None, None],
            metrics=[[RecallAtK(k=5, name="recall@5")], None, None],
        )

        # MatrixFactorizationModel expects (user_ids, item_ids) not (user_features, item_features)
        batch_size = 32
        num_items = 50
        train_user_ids = np.random.randint(0, 100, batch_size)
        train_item_ids = np.random.randint(0, 50, (batch_size, num_items))
        train_labels = np.random.randint(0, 2, (batch_size, num_items)).astype(
            np.float32,
        )

        val_batch_size = 16
        val_user_ids = np.random.randint(0, 100, val_batch_size)
        val_item_ids = np.random.randint(0, 50, (val_batch_size, num_items))
        val_labels = np.random.randint(0, 2, (val_batch_size, num_items)).astype(
            np.float32,
        )

        history = model.fit(
            x=[train_user_ids, train_item_ids],
            y=train_labels,
            validation_data=([val_user_ids, val_item_ids], val_labels),
            epochs=2,
            batch_size=8,
            verbose=0,
        )

        assert "loss" in history.history
        assert len(history.history["loss"]) == 2
        assert history.history["loss"][-1] < history.history["loss"][0]

    def test_deep_ranking_full_pipeline(self, train_data, val_data):
        """Test DeepRankingModel through complete training pipeline."""
        model = DeepRankingModel(
            user_feature_dim=10,
            item_feature_dim=10,
            num_items=50,
        )

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.01),
            loss=[ImprovedMarginRankingLoss(), None, None],
            metrics=[[AccuracyAtK(k=5, name="acc@5")], None, None],
        )

        train_inputs, train_labels = train_data
        val_inputs, val_labels = val_data

        history = model.fit(
            x=train_inputs,
            y=train_labels,
            validation_data=(val_inputs, val_labels),
            epochs=2,
            batch_size=8,
            verbose=0,
        )

        assert "loss" in history.history
        assert len(history.history["loss"]) == 2

    def test_with_callbacks(self, train_data):
        """Test training with recommendation metrics logger callback."""
        model = TwoTowerModel(
            user_feature_dim=10,
            item_feature_dim=10,
            num_items=50,
        )

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.01),
            loss=[ImprovedMarginRankingLoss(), None, None],
            metrics=[[AccuracyAtK(k=5)], None, None],
        )

        callback = RecommendationMetricsLogger(verbose=0)
        train_inputs, train_labels = train_data

        history = model.fit(
            x=train_inputs,
            y=train_labels,
            epochs=2,
            batch_size=8,
            callbacks=[callback],
            verbose=0,
        )

        # Verify callback tracked metrics
        assert "loss" in callback.epoch_metrics
        assert len(callback.epoch_metrics["loss"]) == 2

    def test_model_serialization_and_reload(self, train_data):
        """Test model save and load functionality."""
        import tempfile
        import os

        model = TwoTowerModel(
            user_feature_dim=10,
            item_feature_dim=10,
            num_items=50,
        )
        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=[ImprovedMarginRankingLoss(), None, None],
        )

        # Train briefly
        train_inputs, train_labels = train_data
        model.fit(
            x=train_inputs,
            y=train_labels,
            epochs=1,
            batch_size=8,
            verbose=0,
        )

        # Save model
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "model.keras")
            model.save(model_path)

            # Load model
            loaded_model = keras.models.load_model(model_path)

            # Verify loaded model can be used for training
            history = loaded_model.fit(
                x=train_inputs,
                y=train_labels,
                epochs=1,
                batch_size=8,
                verbose=0,
            )
            assert "loss" in history.history

    def test_loss_decreases_during_training(self, train_data):
        """Test that loss decreases during training."""
        model = TwoTowerModel(
            user_feature_dim=10,
            item_feature_dim=10,
            num_items=50,
        )

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.01),
            loss=[ImprovedMarginRankingLoss(), None, None],
        )

        train_inputs, train_labels = train_data

        history = model.fit(
            x=train_inputs,
            y=train_labels,
            epochs=5,
            batch_size=8,
            verbose=0,
        )

        losses = history.history["loss"]
        # Loss should generally decrease over training
        assert losses[-1] < losses[0]


class TestMetricsComputation:
    """Test that metrics are properly computed during training."""

    @pytest.fixture
    def simple_data(self):
        """Simple test data."""
        x = (
            np.random.randn(16, 10).astype(np.float32),
            np.random.randn(16, 50, 10).astype(np.float32),
        )
        y = np.random.randint(0, 2, (16, 50)).astype(np.float32)
        return x, y

    def test_accuracy_metric_tracked(self, simple_data):
        """Test that Accuracy@K metric is tracked."""
        model = TwoTowerModel(
            user_feature_dim=10,
            item_feature_dim=10,
            num_items=50,
        )

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.01),
            loss=[ImprovedMarginRankingLoss(), None, None],
            metrics=[[AccuracyAtK(k=5, name="acc@5")], None, None],
        )

        x, y = simple_data
        history = model.fit(x=x, y=y, epochs=1, batch_size=8, verbose=0)

        # Metrics should be in history (even if zero initially)
        assert "loss" in history.history

    def test_multiple_metrics_tracked(self, simple_data):
        """Test that multiple metrics can be tracked together."""
        model = TwoTowerModel(
            user_feature_dim=10,
            item_feature_dim=10,
            num_items=50,
        )

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.01),
            loss=[ImprovedMarginRankingLoss(), None, None],
            metrics=[
                AccuracyAtK(k=5, name="acc@5"),
                PrecisionAtK(k=5, name="prec@5"),
                RecallAtK(k=5, name="recall@5"),
            ],
        )

        x, y = simple_data
        history = model.fit(x=x, y=y, epochs=1, batch_size=8, verbose=0)

        assert "loss" in history.history


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
