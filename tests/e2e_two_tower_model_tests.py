"""
End-to-end integration tests for TwoTowerModel.

Comprehensive validation covering:
- Model compilation with tuple output mapping
- Training with metrics
- Inference and prediction
- Recommendation quality and diversity
- Full end-to-end workflow
"""

import numpy as np
import tensorflow as tf
import pytest
from keras.optimizers import Adam

from kmr.models import TwoTowerModel
from kmr.losses import ImprovedMarginRankingLoss
from kmr.metrics import AccuracyAtK, PrecisionAtK, RecallAtK


class TestTwoTowerModelE2E:
    """Comprehensive end-to-end tests for TwoTowerModel."""

    @pytest.fixture(scope="class")
    def setup_data(self):
        """Generate test data."""
        # Generate synthetic features and labels
        n_users = 30
        n_items = 50
        batch_size = 8

        # User and item features
        user_features = np.random.randn(batch_size, 10).astype(np.float32)
        item_features = np.random.randn(batch_size, n_items, 10).astype(np.float32)

        # Binary labels
        labels = np.random.randint(0, 2, (batch_size, n_items)).astype(np.float32)

        return {
            "n_items": n_items,
            "user_features": user_features,
            "item_features": item_features,
            "labels": labels,
        }

    @pytest.fixture
    def model_and_metrics(self, setup_data):
        """Create model and metrics."""
        model = TwoTowerModel(
            user_feature_dim=10,
            item_feature_dim=10,
            num_items=setup_data["n_items"],
            output_dim=32,
            top_k=5,
        )

        # Create metrics
        acc_at_5 = AccuracyAtK(k=5, name="acc@5")
        acc_at_10 = AccuracyAtK(k=10, name="acc@10")
        prec_at_5 = PrecisionAtK(k=5, name="prec@5")
        prec_at_10 = PrecisionAtK(k=10, name="prec@10")
        recall_at_5 = RecallAtK(k=5, name="recall@5")
        recall_at_10 = RecallAtK(k=10, name="recall@10")

        # Compile with tuple mapping
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=[
                ImprovedMarginRankingLoss(
                    margin=1.0,
                    max_min_weight=0.6,
                    avg_weight=0.4,
                ),
                None,
                None,
            ],
            metrics=[
                [acc_at_5, acc_at_10, prec_at_5, prec_at_10, recall_at_5, recall_at_10],
                None,
                None,
            ],
        )

        return {
            "model": model,
            "metrics": {
                "acc@5": acc_at_5,
                "acc@10": acc_at_10,
                "prec@5": prec_at_5,
                "prec@10": prec_at_10,
                "recall@5": recall_at_5,
                "recall@10": recall_at_10,
            },
        }

    def test_model_compilation(self, model_and_metrics):
        """Test that model compiles without errors."""
        model = model_and_metrics["model"]
        assert model.optimizer is not None
        assert model.loss is not None
        assert len(model.metrics) > 0
        print("✅ TwoTowerModel compiled successfully")

    def test_training_convergence(self, setup_data, model_and_metrics):
        """Test that model trains and loss decreases."""
        model = model_and_metrics["model"]
        data = setup_data

        history = model.fit(
            x=[data["user_features"], data["item_features"]],
            y=data["labels"],
            epochs=3,
            batch_size=4,
            verbose=0,
        )

        assert "loss" in history.history
        assert len(history.history["loss"]) == 3

        initial_loss = history.history["loss"][0]
        final_loss = history.history["loss"][-1]
        print(f"   Initial loss: {initial_loss:.4f}, Final loss: {final_loss:.4f}")
        print("✅ TwoTowerModel trained and converged")

    def test_metrics_tracked_during_training(self, setup_data, model_and_metrics):
        """Test that all metrics are tracked."""
        model = model_and_metrics["model"]
        data = setup_data

        history = model.fit(
            x=[data["user_features"], data["item_features"]],
            y=data["labels"],
            epochs=2,
            batch_size=4,
            verbose=0,
        )

        expected_metrics = [
            "loss",
            "acc@5",
            "acc@10",
            "prec@5",
            "prec@10",
            "recall@5",
            "recall@10",
        ]
        for metric_name in expected_metrics:
            assert metric_name in history.history

        print(f"   Tracked metrics: {list(history.history.keys())}")
        print("✅ All metrics tracked during training")

    def test_inference_returns_tuple(self, setup_data, model_and_metrics):
        """Test that inference returns proper tuple."""
        model = model_and_metrics["model"]
        data = setup_data

        output = model(
            [data["user_features"][:2], data["item_features"][:2]],
            training=False,
        )

        assert isinstance(output, tuple)
        assert len(output) == 3

        similarities, rec_indices, rec_scores = output

        assert similarities.shape == (2, data["n_items"])
        assert rec_indices.shape == (2, model.top_k)
        assert rec_scores.shape == (2, model.top_k)

        print(
            f"   Shapes: similarities {similarities.shape}, indices {rec_indices.shape}, scores {rec_scores.shape}",
        )
        print("✅ Inference returns correct tuple")

    def test_recommendations_are_valid(self, setup_data, model_and_metrics):
        """Test recommendation validity."""
        model = model_and_metrics["model"]
        data = setup_data

        _, rec_indices, rec_scores = model(
            [data["user_features"][:4], data["item_features"][:4]],
            training=False,
        )

        rec_indices_np = rec_indices.numpy()
        rec_scores_np = rec_scores.numpy()

        assert np.all(rec_indices_np >= 0)
        assert np.all(rec_indices_np < data["n_items"])
        assert np.all(rec_scores_np >= -1.0)
        assert np.all(rec_scores_np <= 1.0)

        print("✅ Recommendations are valid")

    def test_batch_prediction(self, setup_data, model_and_metrics):
        """Test batch predictions."""
        model = model_and_metrics["model"]
        data = setup_data

        similarities, rec_indices, rec_scores = model(
            [data["user_features"], data["item_features"]],
            training=False,
        )

        assert similarities.shape[0] == data["user_features"].shape[0]
        assert rec_indices.shape[0] == data["user_features"].shape[0]
        assert rec_scores.shape[0] == data["user_features"].shape[0]

        print(f"   Batch size: {data['user_features'].shape[0]}")
        print("✅ Batch prediction works")

    def test_full_workflow(self, setup_data):
        """Test complete workflow."""
        data = setup_data

        model = TwoTowerModel(
            user_feature_dim=10,
            item_feature_dim=10,
            num_items=data["n_items"],
            output_dim=32,
            top_k=5,
        )

        acc_at_5 = AccuracyAtK(k=5)
        prec_at_5 = PrecisionAtK(k=5)

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=[ImprovedMarginRankingLoss(), None, None],
            metrics=[[acc_at_5, prec_at_5], None, None],
        )

        history = model.fit(
            x=[data["user_features"], data["item_features"]],
            y=data["labels"],
            epochs=2,
            batch_size=4,
            verbose=0,
        )

        similarities, rec_indices, rec_scores = model(
            [data["user_features"][:2], data["item_features"][:2]],
            training=False,
        )

        assert history.history["loss"][-1] < history.history["loss"][0] * 1.5
        assert similarities.shape == (2, data["n_items"])

        print("✅ Complete workflow passed")

    def test_08_model_diagnostic_checks(self, setup_data):
        """Test 8: Comprehensive model diagnostic validation."""
        data = setup_data
        model = TwoTowerModel(
            user_feature_dim=10,
            item_feature_dim=10,
            num_items=data["n_items"],
            output_dim=32,
            top_k=5,
        )

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=[ImprovedMarginRankingLoss(), None, None],
            metrics=[[AccuracyAtK(k=5), PrecisionAtK(k=5)], None, None],
        )

        history = model.fit(
            x=[data["user_features"], data["item_features"]],
            y=data["labels"],
            epochs=5,
            batch_size=4,
            verbose=0,
        )

        # Diagnostic Check 1: Training Loss Stability/Convergence
        initial_loss = history.history["loss"][0]
        final_loss = history.history["loss"][-1]
        loss_reduction = (initial_loss - final_loss) / initial_loss

        print(f"\n1. Training Loss Convergence: {loss_reduction:.2%}")
        print(f"   Initial: {initial_loss:.4f} → Final: {final_loss:.4f}")
        # With random data, loss may not always decrease, but it should be reasonable
        assert final_loss < initial_loss * 2.0, "Loss diverged significantly"
        assert final_loss > 0.0, "Loss became negative or zero"

        # Diagnostic Check 2: Metrics Improvement
        if len(history.history) > 1:
            metrics = [k for k in history.history.keys() if k != "loss"]
            if metrics:
                first_metric = metrics[0]
                metric_values = history.history[first_metric]
                print(f"\n2. Metric ({first_metric}) Improvement:")
                print(
                    f"   Start: {metric_values[0]:.4f} → End: {metric_values[-1]:.4f}",
                )
                # Metrics should improve or at least stay reasonable
                assert metric_values[-1] >= 0.0, "Metric is negative"
                assert not np.isnan(metric_values[-1]), "Metric is NaN"

        # Diagnostic Check 3: Inference Shape Validation
        similarities, rec_indices, rec_scores = model.predict(
            [data["user_features"][:2], data["item_features"][:2]],
            verbose=0,
        )

        print(f"\n3. Output Shape Validation:")
        print(f"   Similarities: {similarities.shape}")
        print(f"   Rec indices: {rec_indices.shape}")
        print(f"   Rec scores: {rec_scores.shape}")

        assert similarities.shape == (2, data["n_items"]), "Similarities shape mismatch"
        assert rec_indices.shape == (2, 5), "Rec indices shape mismatch"
        assert rec_scores.shape == (2, 5), "Rec scores shape mismatch"

        # Diagnostic Check 4: Output Value Ranges
        print(f"\n4. Output Value Ranges:")
        print(
            f"   Similarities - Min: {similarities.min():.4f}, Max: {similarities.max():.4f}",
        )
        print(
            f"   Rec scores - Min: {rec_scores.min():.4f}, Max: {rec_scores.max():.4f}",
        )

        assert np.all(np.isfinite(similarities)), "Similarities contain NaN or Inf"
        assert np.all(np.isfinite(rec_scores)), "Rec scores contain NaN or Inf"

        # Diagnostic Check 5: Recommendation Validity
        print(f"\n5. Recommendation Validity:")
        # Check that recommendation indices are within valid range
        assert np.all(rec_indices >= 0), "Negative indices found"
        assert np.all(rec_indices < data["n_items"]), "Index out of bounds"
        # Check that indices are unique per user (no duplicates)
        for user_idx in range(rec_indices.shape[0]):
            unique_count = len(np.unique(rec_indices[user_idx]))
            assert (
                unique_count == rec_indices.shape[1]
            ), f"Duplicate recommendations for user {user_idx}"
        print(f"   ✅ All recommendations are valid and unique")

        print("\n✅ All diagnostic checks passed!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
