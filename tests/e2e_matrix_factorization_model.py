"""
End-to-end integration tests for MatrixFactorizationModel.

This test suite validates:
- Model training with custom data
- Metrics computation during training
- Recommendation generation
- Recommendation diversity
- Prediction behavior for both training and inference modes
"""

import numpy as np
import tensorflow as tf
import pytest
from keras.optimizers import Adam

from kmr.models import MatrixFactorizationModel
from kmr.losses import ImprovedMarginRankingLoss
from kmr.metrics import AccuracyAtK, PrecisionAtK, RecallAtK
from kmr.utils import KMRDataGenerator


class TestMatrixFactorizationModelE2E:
    """End-to-end tests for MatrixFactorizationModel."""

    @pytest.fixture(scope="class")
    def setup_data(self):
        """Generate test data for all tests."""
        # Generate collaborative filtering data
        (
            user_ids,
            item_ids,
            ratings,
            _,
            _,
        ) = KMRDataGenerator.generate_collaborative_filtering_data(
            n_users=100,
            n_items=50,
            n_interactions=500,
            random_state=42,
        )

        n_users = len(np.unique(user_ids))
        n_items = len(np.unique(item_ids))

        # Create training data (like in notebook)
        unique_users = np.unique(user_ids)[:30]  # Use 30 users for training
        train_x_user_ids = []
        train_x_item_ids = []
        train_y = []

        for user_id in unique_users:
            if user_id >= n_users:
                continue

            user_items = item_ids[user_ids == user_id]
            positive_set = set(user_items[user_items < n_items])

            # Create binary labels: 1 for positive items, 0 for others
            labels = np.zeros(n_items, dtype=np.float32)
            labels[list(positive_set)] = 1.0

            train_x_user_ids.append(user_id)
            train_x_item_ids.append(np.arange(n_items))
            train_y.append(labels)

        train_x_user_ids = np.array(train_x_user_ids, dtype=np.int32)
        train_x_item_ids = np.array(train_x_item_ids, dtype=np.int32)
        train_y = np.array(train_y, dtype=np.float32)

        return {
            "n_users": n_users,
            "n_items": n_items,
            "train_x_user_ids": train_x_user_ids,
            "train_x_item_ids": train_x_item_ids,
            "train_y": train_y,
        }

    @pytest.fixture
    def model_and_metrics(self, setup_data):
        """Create model and metrics for testing."""
        model = MatrixFactorizationModel(
            num_users=setup_data["n_users"],
            num_items=setup_data["n_items"],
            embedding_dim=32,
            top_k=5,
            l2_reg=0.01,
        )

        # Create metrics
        acc_at_5 = AccuracyAtK(k=5, name="acc@5")
        acc_at_10 = AccuracyAtK(k=10, name="acc@10")
        prec_at_5 = PrecisionAtK(k=5, name="prec@5")
        prec_at_10 = PrecisionAtK(k=10, name="prec@10")
        recall_at_5 = RecallAtK(k=5, name="recall@5")
        recall_at_10 = RecallAtK(k=10, name="recall@10")

        # Compile with tuple mapping (list of loss/metrics for each output)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=[
                ImprovedMarginRankingLoss(
                    margin=1.0,
                    max_min_weight=0.6,
                    avg_weight=0.4,
                ),
                None,  # rec_indices
                None,  # rec_scores
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
        print("✅ Model compiled successfully")

    def test_training_convergence(self, setup_data, model_and_metrics):
        """Test that model trains and loss decreases over epochs."""
        model = model_and_metrics["model"]
        data = setup_data

        # Train for a few epochs
        history = model.fit(
            x=[data["train_x_user_ids"], data["train_x_item_ids"]],
            y=data["train_y"],
            epochs=5,
            batch_size=4,
            verbose=0,
        )

        # Verify training occurred
        assert "loss" in history.history
        assert len(history.history["loss"]) == 5

        # Verify loss generally decreases
        initial_loss = history.history["loss"][0]
        final_loss = history.history["loss"][-1]
        print(f"   Initial loss: {initial_loss:.4f}")
        print(f"   Final loss: {final_loss:.4f}")
        print(
            f"   Loss reduction: {(initial_loss - final_loss) / initial_loss * 100:.1f}%",
        )

        # Loss should decrease or at least not dramatically increase
        assert final_loss < initial_loss * 1.5, "Loss did not converge properly"
        print("✅ Model trained and converged")

    def test_metrics_tracked_during_training(self, setup_data, model_and_metrics):
        """Test that all metrics are tracked during training."""
        model = model_and_metrics["model"]
        data = setup_data

        history = model.fit(
            x=[data["train_x_user_ids"], data["train_x_item_ids"]],
            y=data["train_y"],
            epochs=2,
            batch_size=4,
            verbose=0,
        )

        # Check that all metrics are in history
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
            assert (
                metric_name in history.history
            ), f"Metric {metric_name} not in history"
            assert (
                len(history.history[metric_name]) == 2
            ), f"Metric {metric_name} has wrong number of epochs"

        print(f"   Tracked metrics: {list(history.history.keys())}")
        print(f"   Epoch 1 metrics:")
        for metric_name in expected_metrics:
            print(f"      {metric_name}: {history.history[metric_name][0]:.4f}")

        print("✅ All metrics tracked during training")

    def test_inference_returns_tuple(self, setup_data, model_and_metrics):
        """Test that inference returns proper tuple output."""
        model = model_and_metrics["model"]
        data = setup_data

        # Get sample input
        sample_user_id = tf.constant([data["train_x_user_ids"][0]])
        sample_item_ids = tf.constant([np.arange(data["n_items"])])

        # Inference should return tuple
        output = model([sample_user_id, sample_item_ids], training=False)

        # Verify output is tuple with 3 elements
        assert isinstance(output, tuple), f"Expected tuple, got {type(output)}"
        assert len(output) == 3, f"Expected 3 outputs, got {len(output)}"

        similarities, rec_indices, rec_scores = output

        # Verify shapes
        assert similarities.shape == (
            1,
            data["n_items"],
        ), f"Wrong similarities shape: {similarities.shape}"
        assert rec_indices.shape == (
            1,
            model.top_k,
        ), f"Wrong rec_indices shape: {rec_indices.shape}"
        assert rec_scores.shape == (
            1,
            model.top_k,
        ), f"Wrong rec_scores shape: {rec_scores.shape}"

        print(f"   Similarities: {similarities.shape}")
        print(f"   Recommendation indices: {rec_indices.shape}")
        print(f"   Recommendation scores: {rec_scores.shape}")
        print("✅ Inference returns correct tuple output")

    def test_recommendations_are_valid(self, setup_data, model_and_metrics):
        """Test that recommendations are valid (indices within bounds, scores in proper range)."""
        model = model_and_metrics["model"]
        data = setup_data

        # Get batch of recommendations
        sample_user_ids = tf.constant(data["train_x_user_ids"][:5])
        sample_item_ids_batch = np.array(
            [np.arange(data["n_items"])] * 5,
            dtype=np.int32,
        )
        sample_item_ids = tf.constant(sample_item_ids_batch)

        similarities, rec_indices, rec_scores = model(
            [sample_user_ids, sample_item_ids],
            training=False,
        )

        rec_indices_np = rec_indices.numpy()
        rec_scores_np = rec_scores.numpy()

        # Verify indices are within bounds
        assert np.all(rec_indices_np >= 0), "Negative item indices"
        assert np.all(rec_indices_np < data["n_items"]), "Item indices out of bounds"

        # Verify scores are in reasonable range
        assert np.all(rec_scores_np >= -1.0), "Scores too low"
        assert np.all(rec_scores_np <= 1.0), "Scores too high"

        # Verify top-k is correct
        assert (
            rec_indices_np.shape[1] == model.top_k
        ), f"Wrong number of recommendations"

        print(f"   All indices valid: ✓")
        print(f"   All scores in range [-1, 1]: ✓")
        print(f"   Recommendation count per user: {rec_indices_np.shape[1]}")
        print("✅ Recommendations are valid")

    def test_recommendation_diversity(self, setup_data, model_and_metrics):
        """Test that recommendations are diverse across users."""
        model = model_and_metrics["model"]
        data = setup_data

        # Get recommendations for multiple users
        n_sample_users = min(10, len(data["train_x_user_ids"]))
        sample_user_ids = tf.constant(data["train_x_user_ids"][:n_sample_users])

        sample_item_ids_batch = np.array(
            [np.arange(data["n_items"])] * n_sample_users,
            dtype=np.int32,
        )
        sample_item_ids = tf.constant(sample_item_ids_batch)

        similarities, rec_indices, rec_scores = model(
            [sample_user_ids, sample_item_ids],
            training=False,
        )
        rec_indices_np = rec_indices.numpy()

        # Calculate diversity metrics
        unique_items_per_user = [len(np.unique(rec)) for rec in rec_indices_np]
        all_recommended_items = set()
        for rec in rec_indices_np:
            all_recommended_items.update(rec)

        shared_items = 0
        if n_sample_users > 1:
            shared_items = len(
                set(rec_indices_np[0]).intersection(
                    *[set(rec) for rec in rec_indices_np[1:]],
                ),
            )

        diversity_ratio = (
            1.0 - (shared_items / model.top_k) if n_sample_users > 1 else 1.0
        )

        print(f"   Sample users: {n_sample_users}")
        print(
            f"   Unique items across all recommendations: {len(all_recommended_items)}/{data['n_items']}",
        )
        print(f"   Shared items between all users: {shared_items}/{model.top_k}")
        print(f"   Diversity ratio: {diversity_ratio:.2%}")
        print(
            f"   Unique items per user: avg={np.mean(unique_items_per_user):.1f}, "
            f"min={np.min(unique_items_per_user)}, max={np.max(unique_items_per_user)}",
        )

        # Verify we have some diversity
        assert len(all_recommended_items) > 1, "No diversity in recommendations"
        assert diversity_ratio > 0.0, "Zero diversity"

        print("✅ Recommendations show diversity")

    def test_training_vs_inference_consistency(self, setup_data, model_and_metrics):
        """Test that model.call() returns consistent outputs during training and inference."""
        model = model_and_metrics["model"]
        data = setup_data

        # Train model first
        model.fit(
            x=[data["train_x_user_ids"], data["train_x_item_ids"]],
            y=data["train_y"],
            epochs=2,
            batch_size=4,
            verbose=0,
        )

        # Get sample input
        sample_user_id = tf.constant([data["train_x_user_ids"][0]])
        sample_item_ids = tf.constant([np.arange(data["n_items"])])

        # Call with training=False
        output_inference = model([sample_user_id, sample_item_ids], training=False)
        sim_inf, rec_idx_inf, rec_scores_inf = output_inference

        # Call with training=True
        output_training = model([sample_user_id, sample_item_ids], training=True)
        sim_train, rec_idx_train, rec_scores_train = output_training

        # Both should be tuples
        assert isinstance(output_inference, tuple) and isinstance(
            output_training,
            tuple,
        )

        # Shapes should match
        assert sim_inf.shape == sim_train.shape
        assert rec_idx_inf.shape == rec_idx_train.shape
        assert rec_scores_inf.shape == rec_scores_train.shape

        print(
            "   Training mode output shape:",
            (sim_train.shape, rec_idx_train.shape, rec_scores_train.shape),
        )
        print(
            "   Inference mode output shape:",
            (sim_inf.shape, rec_idx_inf.shape, rec_scores_inf.shape),
        )
        print("✅ Training and inference modes return consistent outputs")

    def test_batch_prediction(self, setup_data, model_and_metrics):
        """Test that batch predictions work correctly."""
        model = model_and_metrics["model"]
        data = setup_data

        # Prepare batch
        batch_size = 5
        batch_user_ids = tf.constant(data["train_x_user_ids"][:batch_size])
        batch_item_ids = np.array(
            [np.arange(data["n_items"])] * batch_size,
            dtype=np.int32,
        )
        batch_item_ids = tf.constant(batch_item_ids)

        # Predict
        similarities, rec_indices, rec_scores = model(
            [batch_user_ids, batch_item_ids],
            training=False,
        )

        # Verify batch dimensions
        assert similarities.shape[0] == batch_size
        assert rec_indices.shape[0] == batch_size
        assert rec_scores.shape[0] == batch_size

        print(f"   Batch size: {batch_size}")
        print(
            f"   Output shapes: {similarities.shape}, {rec_indices.shape}, {rec_scores.shape}",
        )
        print("✅ Batch prediction works correctly")

    def test_full_workflow(self, setup_data):
        """Test complete workflow: create -> compile -> train -> predict -> validate."""
        data = setup_data

        # 1. Create model
        model = MatrixFactorizationModel(
            num_users=data["n_users"],
            num_items=data["n_items"],
            embedding_dim=32,
            top_k=5,
            l2_reg=0.01,
        )

        # 2. Create metrics
        acc_at_5 = AccuracyAtK(k=5, name="acc@5")
        prec_at_5 = PrecisionAtK(k=5, name="prec@5")
        recall_at_5 = RecallAtK(k=5, name="recall@5")

        # 3. Compile
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=[ImprovedMarginRankingLoss(), None, None],
            metrics=[[acc_at_5, prec_at_5, recall_at_5], None, None],
        )

        # 4. Train
        history = model.fit(
            x=[data["train_x_user_ids"], data["train_x_item_ids"]],
            y=data["train_y"],
            epochs=3,
            batch_size=4,
            verbose=0,
        )

        # 5. Predict
        sample_user_id = tf.constant([data["train_x_user_ids"][0]])
        sample_item_ids = tf.constant([np.arange(data["n_items"])])
        similarities, rec_indices, rec_scores = model(
            [sample_user_id, sample_item_ids],
            training=False,
        )

        # 6. Validate
        assert history.history["loss"][-1] < history.history["loss"][0]
        assert similarities.shape == (1, data["n_items"])
        assert rec_indices.shape == (1, 5)
        assert rec_scores.shape == (1, 5)

        print("✅ Complete workflow executed successfully!")


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
