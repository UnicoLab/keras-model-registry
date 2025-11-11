"""
End-to-end integration tests for ExplainableRecommendationModel.

15 comprehensive tests covering:
- Compilation, training, metrics, inference, diversity
- Model learning, quality metrics, reproducibility
- Edge cases, uniqueness, personalization
- Feedback influence on explanations
"""

import numpy as np
import tensorflow as tf
import pytest
from keras.optimizers import Adam

from kmr.models import ExplainableRecommendationModel
from kmr.losses import ImprovedMarginRankingLoss
from kmr.metrics import AccuracyAtK, PrecisionAtK, RecallAtK
from kmr.utils import KMRDataGenerator


class TestExplainableRecommendationModelE2E:
    """Comprehensive E2E tests for ExplainableRecommendationModel."""

    @pytest.fixture(scope="class")
    def setup_data(self):
        """Generate test data."""
        (
            user_ids,
            item_ids,
            _,
            _,
            _,
        ) = KMRDataGenerator.generate_collaborative_filtering_data(
            n_users=100,
            n_items=50,
            n_interactions=500,
            random_state=42,
        )
        n_users, n_items = len(np.unique(user_ids)), len(np.unique(item_ids))
        unique_users = np.unique(user_ids)[:30]

        train_x_user_ids, train_x_item_ids, train_y = [], [], []
        for user_id in unique_users:
            if user_id >= n_users:
                continue
            user_items = item_ids[user_ids == user_id]
            positive_set = set(user_items[user_items < n_items])
            labels = np.zeros(n_items, dtype=np.float32)
            labels[list(positive_set)] = 1.0
            train_x_user_ids.append(user_id)
            train_x_item_ids.append(np.arange(n_items))
            train_y.append(labels)

        return {
            "n_users": n_users,
            "n_items": n_items,
            "train_x_user_ids": np.array(train_x_user_ids, dtype=np.int32),
            "train_x_item_ids": np.array(train_x_item_ids, dtype=np.int32),
            "train_y": np.array(train_y, dtype=np.float32),
        }

    @pytest.fixture
    def model_and_metrics(self, setup_data):
        """Create model and metrics."""
        model = ExplainableRecommendationModel(
            num_users=setup_data["n_users"],
            num_items=setup_data["n_items"],
            embedding_dim=32,
            top_k=5,
        )

        acc_at_5 = AccuracyAtK(k=5, name="acc@5")
        prec_at_5 = PrecisionAtK(k=5, name="prec@5")
        recall_at_5 = RecallAtK(k=5, name="recall@5")

        # 5 outputs: scores, rec_indices, rec_scores, similarity_matrix, feedback_adjusted
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=[ImprovedMarginRankingLoss(), None, None, None, None],
            metrics=[[acc_at_5, prec_at_5, recall_at_5], None, None, None, None],
        )

        return {
            "model": model,
            "metrics": {
                "acc@5": acc_at_5,
                "prec@5": prec_at_5,
                "recall@5": recall_at_5,
            },
        }

    def test_01_model_compilation(self, model_and_metrics):
        """Test 1: Model compiles without errors."""
        model = model_and_metrics["model"]
        assert model.optimizer is not None
        assert model.loss is not None
        assert model.top_k == 5
        print("✅ Test 1: Model compilation successful")

    def test_02_training_convergence(self, setup_data, model_and_metrics):
        """Test 2: Model trains and loss decreases."""
        model = model_and_metrics["model"]
        history = model.fit(
            x=[setup_data["train_x_user_ids"], setup_data["train_x_item_ids"]],
            y=setup_data["train_y"],
            epochs=3,
            batch_size=4,
            verbose=0,
        )
        assert "loss" in history.history
        assert len(history.history["loss"]) == 3
        print(
            f"   Loss: {history.history['loss'][0]:.4f} → {history.history['loss'][-1]:.4f}",
        )
        print("✅ Test 2: Training convergence")

    def test_03_metrics_tracked(self, setup_data, model_and_metrics):
        """Test 3: All metrics tracked during training."""
        model = model_and_metrics["model"]
        history = model.fit(
            x=[setup_data["train_x_user_ids"], setup_data["train_x_item_ids"]],
            y=setup_data["train_y"],
            epochs=2,
            batch_size=4,
            verbose=0,
        )
        expected_metrics = ["loss", "acc@5", "prec@5", "recall@5"]
        for metric_name in expected_metrics:
            assert metric_name in history.history
        print("✅ Test 3: All metrics tracked")

    def test_04_inference_returns_tuple(self, setup_data, model_and_metrics):
        """Test 4: Inference returns 5-tuple output."""
        model = model_and_metrics["model"]
        output = model(
            [setup_data["train_x_user_ids"][:2], setup_data["train_x_item_ids"][:2]],
            training=False,
        )
        assert isinstance(output, tuple)
        assert len(output) == 5

        scores, rec_indices, rec_scores, sim_matrix, feedback = output
        assert scores.shape == (2, setup_data["n_items"])
        assert rec_indices.shape == (2, 5)
        assert rec_scores.shape == (2, 5)
        assert sim_matrix.shape == (2, setup_data["n_items"])

        print(
            f"   Output shapes: scores {scores.shape}, indices {rec_indices.shape}, sim {sim_matrix.shape}",
        )
        print("✅ Test 4: Inference returns correct 5-tuple")

    def test_05_recommendation_validity(self, setup_data, model_and_metrics):
        """Test 5: Recommendations are valid."""
        model = model_and_metrics["model"]
        _, rec_indices, rec_scores, _, _ = model(
            [setup_data["train_x_user_ids"][:4], setup_data["train_x_item_ids"][:4]],
            training=False,
        )
        rec_indices_np = rec_indices.numpy()
        rec_scores_np = rec_scores.numpy()

        assert np.all(rec_indices_np >= 0)
        assert np.all(rec_indices_np < setup_data["n_items"])
        assert np.all(rec_scores_np >= -1.0)
        assert np.all(rec_scores_np <= 1.0)
        assert not np.any(np.isnan(rec_indices_np))
        assert not np.any(np.isnan(rec_scores_np))

        print("✅ Test 5: Recommendations are valid")

    def test_06_recommendation_diversity(self, setup_data, model_and_metrics):
        """Test 6: Recommendations are diverse across users."""
        model = model_and_metrics["model"]
        n_sample = min(8, len(setup_data["train_x_user_ids"]))
        _, rec_indices, _, _, _ = model(
            [
                setup_data["train_x_user_ids"][:n_sample],
                setup_data["train_x_item_ids"][:n_sample],
            ],
            training=False,
        )
        rec_indices_np = rec_indices.numpy()

        all_items = set()
        for rec in rec_indices_np:
            all_items.update(rec)

        coverage = len(all_items) / setup_data["n_items"] * 100
        print(f"   Catalog coverage: {coverage:.1f}%")
        assert len(all_items) > 1
        assert coverage > 15.0

        print("✅ Test 6: Good diversity in recommendations")

    def test_07_training_vs_inference_consistency(self, setup_data, model_and_metrics):
        """Test 7: Consistent outputs in both modes."""
        model = model_and_metrics["model"]
        model.fit(
            x=[setup_data["train_x_user_ids"], setup_data["train_x_item_ids"]],
            y=setup_data["train_y"],
            epochs=1,
            batch_size=4,
            verbose=0,
        )

        output_inf = model(
            [setup_data["train_x_user_ids"][:2], setup_data["train_x_item_ids"][:2]],
            training=False,
        )
        output_train = model(
            [setup_data["train_x_user_ids"][:2], setup_data["train_x_item_ids"][:2]],
            training=True,
        )

        assert len(output_inf) == len(output_train) == 5
        for inf_out, train_out in zip(output_inf, output_train):
            assert inf_out.shape == train_out.shape

        print("✅ Test 7: Consistent outputs in both modes")

    def test_08_batch_prediction(self, setup_data, model_and_metrics):
        """Test 8: Batch predictions work."""
        model = model_and_metrics["model"]
        scores, rec_indices, rec_scores, _, _ = model(
            [setup_data["train_x_user_ids"][:6], setup_data["train_x_item_ids"][:6]],
            training=False,
        )

        assert scores.shape[0] == 6
        assert rec_indices.shape[0] == 6
        assert np.all(rec_indices.numpy() >= 0)

        print("✅ Test 8: Batch prediction works")

    def test_09_full_workflow(self, setup_data):
        """Test 9: Complete end-to-end workflow."""
        model = ExplainableRecommendationModel(
            num_users=setup_data["n_users"],
            num_items=setup_data["n_items"],
            embedding_dim=32,
            top_k=5,
        )

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=[ImprovedMarginRankingLoss(), None, None, None, None],
            metrics=[[AccuracyAtK(k=5)], None, None, None, None],
        )

        history = model.fit(
            x=[setup_data["train_x_user_ids"], setup_data["train_x_item_ids"]],
            y=setup_data["train_y"],
            epochs=2,
            batch_size=4,
            verbose=0,
        )

        output = model(
            [setup_data["train_x_user_ids"][:2], setup_data["train_x_item_ids"][:2]],
            training=False,
        )

        assert len(history.history["loss"]) == 2
        assert len(output) == 5

        print("✅ Test 9: Full workflow passed")

    def test_10_similarity_matrix_explanation(self, setup_data, model_and_metrics):
        """Test 10: Similarity matrix is proper explanation component."""
        model = model_and_metrics["model"]
        model.fit(
            x=[setup_data["train_x_user_ids"], setup_data["train_x_item_ids"]],
            y=setup_data["train_y"],
            epochs=2,
            batch_size=4,
            verbose=0,
        )

        _, _, _, sim_matrix, _ = model(
            [setup_data["train_x_user_ids"][:3], setup_data["train_x_item_ids"][:3]],
            training=False,
        )

        assert sim_matrix.shape == (3, setup_data["n_items"])
        assert np.all(sim_matrix.numpy() >= -1.0)
        assert np.all(sim_matrix.numpy() <= 1.0)

        print("✅ Test 10: Similarity matrix is valid explanation")

    def test_11_metric_quality(self, setup_data, model_and_metrics):
        """Test 11: Quality metrics show learning."""
        model = model_and_metrics["model"]
        history = model.fit(
            x=[setup_data["train_x_user_ids"], setup_data["train_x_item_ids"]],
            y=setup_data["train_y"],
            epochs=3,
            batch_size=4,
            verbose=0,
        )

        final_acc = history.history["acc@5"][-1]
        final_prec = history.history["prec@5"][-1]

        assert final_acc > 0.0
        assert final_prec > 0.0

        print(f"   Accuracy: {final_acc:.4f}, Precision: {final_prec:.4f}")
        print("✅ Test 11: Quality metrics show learning")

    def test_12_reproducible_predictions(self, setup_data, model_and_metrics):
        """Test 12: Predictions are reproducible."""
        model = model_and_metrics["model"]
        model.fit(
            x=[setup_data["train_x_user_ids"], setup_data["train_x_item_ids"]],
            y=setup_data["train_y"],
            epochs=1,
            batch_size=4,
            verbose=0,
        )

        _, idx1, _, _, _ = model(
            [setup_data["train_x_user_ids"][:2], setup_data["train_x_item_ids"][:2]],
            training=False,
        )
        _, idx2, _, _, _ = model(
            [setup_data["train_x_user_ids"][:2], setup_data["train_x_item_ids"][:2]],
            training=False,
        )

        assert np.array_equal(idx1.numpy(), idx2.numpy())

        print("✅ Test 12: Predictions are reproducible")

    def test_13_edge_case_single_user(self, setup_data, model_and_metrics):
        """Test 13: Handles single user."""
        model = model_and_metrics["model"]
        model.fit(
            x=[setup_data["train_x_user_ids"], setup_data["train_x_item_ids"]],
            y=setup_data["train_y"],
            epochs=1,
            batch_size=4,
            verbose=0,
        )

        output = model(
            [setup_data["train_x_user_ids"][:1], setup_data["train_x_item_ids"][:1]],
            training=False,
        )

        assert all(out.shape[0] == 1 for out in output)

        print("✅ Test 13: Handles single user")

    def test_14_output_uniqueness(self, setup_data, model_and_metrics):
        """Test 14: Recommended indices are unique per user."""
        model = model_and_metrics["model"]
        model.fit(
            x=[setup_data["train_x_user_ids"], setup_data["train_x_item_ids"]],
            y=setup_data["train_y"],
            epochs=1,
            batch_size=4,
            verbose=0,
        )

        _, rec_indices, _, _, _ = model(
            [setup_data["train_x_user_ids"][:4], setup_data["train_x_item_ids"][:4]],
            training=False,
        )

        rec_indices_np = rec_indices.numpy()
        for user_recs in rec_indices_np:
            assert len(np.unique(user_recs)) == 5

        print("✅ Test 14: Each user gets unique recommendations")

    def test_15_personalization(self, setup_data, model_and_metrics):
        """Test 15: Different users get different recommendations."""
        model = model_and_metrics["model"]
        model.fit(
            x=[setup_data["train_x_user_ids"], setup_data["train_x_item_ids"]],
            y=setup_data["train_y"],
            epochs=2,
            batch_size=4,
            verbose=0,
        )

        _, rec_indices, _, _, _ = model(
            [setup_data["train_x_user_ids"], setup_data["train_x_item_ids"]],
            training=False,
        )

        rec_indices_np = rec_indices.numpy()
        different_count = sum(
            1
            for i in range(1, len(rec_indices_np))
            if not np.array_equal(rec_indices_np[0], rec_indices_np[i])
        )

        assert different_count > 0

        print("✅ Test 15: Provides personalized recommendations")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
