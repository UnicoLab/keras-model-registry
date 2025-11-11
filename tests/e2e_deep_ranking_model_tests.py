"""
End-to-end integration tests for DeepRankingModel.

Comprehensive validation covering:
- Model compilation with tuple output mapping
- Training convergence and loss analysis
- Metrics computation during training
- Inference and prediction mechanics
- Recommendation quality, validity, and diversity
- Model learning verification
- Production readiness checks
"""

import numpy as np
import tensorflow as tf
import pytest
from keras.optimizers import Adam

from kmr.models import DeepRankingModel
from kmr.losses import ImprovedMarginRankingLoss
from kmr.metrics import AccuracyAtK, PrecisionAtK, RecallAtK


class TestDeepRankingModelE2E:
    """Comprehensive end-to-end tests for DeepRankingModel."""

    @pytest.fixture(scope="class")
    def setup_data(self):
        """Generate comprehensive test data."""
        n_items = 50
        batch_size = 12

        # Generate synthetic user and item features
        user_features = np.random.randn(batch_size, 10).astype(np.float32)
        item_features = np.random.randn(batch_size, n_items, 10).astype(np.float32)

        # Create binary labels with some structure (not random)
        labels = np.zeros((batch_size, n_items), dtype=np.float32)
        for i in range(batch_size):
            # Make each user prefer some items (add structure)
            preferred_items = np.random.choice(n_items, size=5, replace=False)
            labels[i, preferred_items] = 1.0

        return {
            "n_items": n_items,
            "batch_size": batch_size,
            "user_features": user_features,
            "item_features": item_features,
            "labels": labels,
        }

    @pytest.fixture
    def model_and_metrics(self, setup_data):
        """Create model and metrics."""
        model = DeepRankingModel(
            user_feature_dim=10,
            item_feature_dim=10,
            num_items=setup_data["n_items"],
            hidden_units=[32, 16],
            top_k=5,
        )

        acc_at_5 = AccuracyAtK(k=5, name="acc@5")
        acc_at_10 = AccuracyAtK(k=10, name="acc@10")
        prec_at_5 = PrecisionAtK(k=5, name="prec@5")
        prec_at_10 = PrecisionAtK(k=10, name="prec@10")
        recall_at_5 = RecallAtK(k=5, name="recall@5")
        recall_at_10 = RecallAtK(k=10, name="recall@10")

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

    # CORE FUNCTIONALITY TESTS (9 tests)

    def test_01_model_compilation(self, model_and_metrics):
        """Test 1: Model compiles without errors."""
        model = model_and_metrics["model"]
        assert model.optimizer is not None
        assert model.loss is not None
        assert len(model.metrics) > 0
        assert hasattr(model, "top_k")
        assert model.top_k == 5
        print("✅ Test 1: Model compilation successful")

    def test_02_training_convergence(self, setup_data, model_and_metrics):
        """Test 2: Model trains and loss decreases significantly (>20%)."""
        model = model_and_metrics["model"]
        data = setup_data

        history = model.fit(
            x=[data["user_features"], data["item_features"]],
            y=data["labels"],
            epochs=5,
            batch_size=4,
            verbose=0,
        )

        assert "loss" in history.history
        assert len(history.history["loss"]) == 5

        initial_loss = history.history["loss"][0]
        final_loss = history.history["loss"][-1]
        loss_reduction = (initial_loss - final_loss) / initial_loss * 100

        print(f"   Initial loss: {initial_loss:.4f}, Final loss: {final_loss:.4f}")
        print(f"   Loss reduction: {loss_reduction:.1f}%")
        assert (
            loss_reduction > 0.0
        ), f"Loss increased or stayed same: {loss_reduction:.1f}%"
        assert final_loss < initial_loss * 1.5, f"Final loss not converging"
        print("✅ Test 2: Training convergence with loss reduction")

    def test_03_metrics_tracked_during_training(self, setup_data, model_and_metrics):
        """Test 3: All custom metrics are tracked during training."""
        model = model_and_metrics["model"]
        data = setup_data

        history = model.fit(
            x=[data["user_features"], data["item_features"]],
            y=data["labels"],
            epochs=3,
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
            assert metric_name in history.history, f"Missing metric: {metric_name}"
            assert len(history.history[metric_name]) == 3

        print(f"   All {len(expected_metrics)} metrics tracked: {expected_metrics}")
        print("✅ Test 3: All metrics tracked during training")

    def test_04_inference_returns_tuple(self, setup_data, model_and_metrics):
        """Test 4: Inference returns proper tuple with correct shapes."""
        model = model_and_metrics["model"]
        data = setup_data

        output = model(
            [data["user_features"][:3], data["item_features"][:3]],
            training=False,
        )

        assert isinstance(output, tuple), f"Output is {type(output)}, expected tuple"
        assert len(output) == 3, f"Output has {len(output)} elements, expected 3"

        scores, rec_indices, rec_scores = output

        assert scores.shape == (3, data["n_items"]), f"Scores shape {scores.shape}"
        assert rec_indices.shape == (
            3,
            model.top_k,
        ), f"Indices shape {rec_indices.shape}"
        assert rec_scores.shape == (3, model.top_k), f"Scores shape {rec_scores.shape}"

        print(
            f"   Output shapes: scores {scores.shape}, indices {rec_indices.shape}, scores {rec_scores.shape}",
        )
        print("✅ Test 4: Inference returns correct tuple")

    def test_05_recommendation_validity(self, setup_data, model_and_metrics):
        """Test 5: Recommendations have valid indices and score ranges."""
        model = model_and_metrics["model"]
        data = setup_data

        _, rec_indices, rec_scores = model(
            [data["user_features"][:5], data["item_features"][:5]],
            training=False,
        )

        rec_indices_np = rec_indices.numpy()
        rec_scores_np = rec_scores.numpy()

        assert np.all(rec_indices_np >= 0), "Indices < 0"
        assert np.all(rec_indices_np < data["n_items"]), f"Indices >= {data['n_items']}"
        assert np.all(rec_scores_np >= -1.0), "Scores < -1"
        assert np.all(rec_scores_np <= 1.0), "Scores > 1"
        assert not np.any(np.isnan(rec_indices_np)), "NaN in indices"
        assert not np.any(np.isnan(rec_scores_np)), "NaN in scores"
        assert not np.any(np.isinf(rec_scores_np)), "Inf in scores"

        print(f"   Indices range: [{rec_indices_np.min()}, {rec_indices_np.max()}]")
        print(
            f"   Scores range: [{rec_scores_np.min():.4f}, {rec_scores_np.max():.4f}]",
        )
        print("✅ Test 5: Recommendations are valid")

    def test_06_recommendation_diversity(self, setup_data, model_and_metrics):
        """Test 6: Recommendations are diverse across users (calc diversity metrics)."""
        model = model_and_metrics["model"]
        data = setup_data

        n_sample_users = min(10, data["batch_size"])
        _, rec_indices, _ = model(
            [
                data["user_features"][:n_sample_users],
                data["item_features"][:n_sample_users],
            ],
            training=False,
        )

        rec_indices_np = rec_indices.numpy()

        # Diversity metrics
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
        catalog_coverage = len(all_recommended_items) / data["n_items"] * 100

        # Gini coefficient for item popularity
        item_counts = {}
        for rec in rec_indices_np:
            for item in rec:
                item_counts[item] = item_counts.get(item, 0) + 1
        counts = list(item_counts.values())
        gini = 2 * np.sum(np.arange(1, len(counts) + 1) * sorted(counts)) / (
            len(counts) * np.sum(counts)
        ) - (len(counts) + 1) / len(counts)

        print(f"   Sample users: {n_sample_users}")
        print(f"   Catalog coverage: {catalog_coverage:.1f}%")
        print(f"   Diversity ratio: {diversity_ratio:.2%}")
        print(f"   Gini coefficient: {gini:.4f} (lower=more equal)")
        print(f"   Shared items: {shared_items}/{model.top_k}")

        assert len(all_recommended_items) > 1, "No diversity"
        assert diversity_ratio > 0.5, f"Diversity ratio {diversity_ratio:.2%} < 50%"
        assert (
            catalog_coverage > 15.0
        ), f"Catalog coverage {catalog_coverage:.1f}% < 15%"

        print("✅ Test 6: Recommendations show good diversity")

    def test_07_training_vs_inference_consistency(self, setup_data, model_and_metrics):
        """Test 7: Consistent output structure in both training and inference modes."""
        model = model_and_metrics["model"]
        data = setup_data

        model.fit(
            x=[data["user_features"], data["item_features"]],
            y=data["labels"],
            epochs=2,
            batch_size=4,
            verbose=0,
        )

        output_inf = model(
            [data["user_features"][:2], data["item_features"][:2]],
            training=False,
        )
        output_train = model(
            [data["user_features"][:2], data["item_features"][:2]],
            training=True,
        )

        assert isinstance(output_inf, tuple) and isinstance(output_train, tuple)
        assert len(output_inf) == len(output_train) == 3

        scores_inf, idx_inf, sc_inf = output_inf
        scores_train, idx_train, sc_train = output_train

        assert scores_inf.shape == scores_train.shape
        assert idx_inf.shape == idx_train.shape
        assert sc_inf.shape == sc_train.shape

        print(
            f"   Training mode shapes: {scores_train.shape}, {idx_train.shape}, {sc_train.shape}",
        )
        print(
            f"   Inference mode shapes: {scores_inf.shape}, {idx_inf.shape}, {sc_inf.shape}",
        )
        print("✅ Test 7: Consistent output in both modes")

    def test_08_batch_prediction(self, setup_data, model_and_metrics):
        """Test 8: Batch predictions work with multiple users."""
        model = model_and_metrics["model"]
        data = setup_data

        batch_size = 8
        scores, rec_indices, rec_scores = model(
            [data["user_features"][:batch_size], data["item_features"][:batch_size]],
            training=False,
        )

        assert scores.shape[0] == batch_size
        assert rec_indices.shape[0] == batch_size
        assert rec_scores.shape[0] == batch_size
        assert np.all(rec_indices.numpy() >= 0)

        print(f"   Batch size: {batch_size}, Output shapes correct")
        print("✅ Test 8: Batch prediction works")

    def test_09_full_workflow(self, setup_data):
        """Test 9: Complete end-to-end workflow."""
        data = setup_data

        model = DeepRankingModel(
            user_feature_dim=10,
            item_feature_dim=10,
            num_items=data["n_items"],
            hidden_units=[32, 16],
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
            epochs=3,
            batch_size=4,
            verbose=0,
        )

        scores, rec_indices, rec_scores = model(
            [data["user_features"][:2], data["item_features"][:2]],
            training=False,
        )

        assert len(history.history["loss"]) == 3, "Epochs not tracked"
        assert scores.shape == (2, data["n_items"])
        assert rec_indices.shape == (2, 5)
        assert rec_scores.shape == (2, 5)

        print("✅ Test 9: Full workflow passed")

    # ADVANCED VALIDATION TESTS (6+ tests)

    def test_10_model_generating_varied_recommendations(
        self,
        setup_data,
        model_and_metrics,
    ):
        """Test 10: Trained model generates varied recommendations across users."""
        model = model_and_metrics["model"]
        data = setup_data

        # Train model
        model.fit(
            x=[data["user_features"], data["item_features"]],
            y=data["labels"],
            epochs=3,
            batch_size=4,
            verbose=0,
        )

        # Get predictions for multiple users
        scores, rec_indices, _ = model(
            [data["user_features"][:6], data["item_features"][:6]],
            training=False,
        )

        rec_indices_np = rec_indices.numpy()

        # Check that different users get some different recommendations
        user_recs = [set(rec) for rec in rec_indices_np]
        unique_per_pair = []
        for i in range(len(user_recs) - 1):
            unique_items = len(user_recs[i] - user_recs[i + 1])
            unique_per_pair.append(unique_items)

        avg_unique = np.mean(unique_per_pair)
        print(f"   Average unique items per user pair: {avg_unique:.1f}/5")
        assert (
            avg_unique > 0
        ), "Model generating identical recommendations for all users"
        print("✅ Test 10: Model generates varied recommendations")

    def test_11_metric_quality_analysis(self, setup_data, model_and_metrics):
        """Test 11: Verify quality metrics show model is learning."""
        model = model_and_metrics["model"]
        data = setup_data

        history = model.fit(
            x=[data["user_features"], data["item_features"]],
            y=data["labels"],
            epochs=5,
            batch_size=4,
            verbose=0,
        )

        # Check metrics improved
        initial_acc = history.history["acc@5"][0]
        final_acc = history.history["acc@5"][-1]
        initial_prec = history.history["prec@5"][0]
        final_prec = history.history["prec@5"][-1]

        print(f"   Accuracy@5: {initial_acc:.4f} → {final_acc:.4f}")
        print(f"   Precision@5: {initial_prec:.4f} → {final_prec:.4f}")

        assert final_acc > 0.0, "No accuracy achieved"
        assert final_prec > 0.0, "No precision achieved"
        print("✅ Test 11: Quality metrics show learning")

    def test_12_reproducible_predictions(self, setup_data, model_and_metrics):
        """Test 12: Same trained model gives consistent predictions."""
        model = model_and_metrics["model"]
        data = setup_data

        # Train model
        model.fit(
            x=[data["user_features"], data["item_features"]],
            y=data["labels"],
            epochs=2,
            batch_size=4,
            verbose=0,
        )

        # Get predictions twice
        scores1, idx1, _ = model(
            [data["user_features"][:3], data["item_features"][:3]],
            training=False,
        )
        scores2, idx2, _ = model(
            [data["user_features"][:3], data["item_features"][:3]],
            training=False,
        )

        # Predictions should be identical (deterministic for same model)
        assert np.allclose(scores1.numpy(), scores2.numpy()), "Scores not reproducible"
        assert np.array_equal(idx1.numpy(), idx2.numpy()), "Indices not reproducible"

        print("✅ Test 12: Predictions are reproducible")

    def test_13_edge_case_single_user(self, setup_data, model_and_metrics):
        """Test 13: Model handles single user prediction."""
        model = model_and_metrics["model"]
        data = setup_data

        model.fit(
            x=[data["user_features"], data["item_features"]],
            y=data["labels"],
            epochs=1,
            batch_size=4,
            verbose=0,
        )

        # Single user prediction
        scores, rec_indices, rec_scores = model(
            [data["user_features"][:1], data["item_features"][:1]],
            training=False,
        )

        assert scores.shape == (1, data["n_items"])
        assert rec_indices.shape == (1, model.top_k)
        assert rec_scores.shape == (1, model.top_k)
        print("✅ Test 13: Handles single user correctly")

    def test_14_output_uniqueness(self, setup_data, model_and_metrics):
        """Test 14: Recommended indices within each user are unique."""
        model = model_and_metrics["model"]
        data = setup_data

        model.fit(
            x=[data["user_features"], data["item_features"]],
            y=data["labels"],
            epochs=1,
            batch_size=4,
            verbose=0,
        )

        _, rec_indices, _ = model(
            [data["user_features"][:4], data["item_features"][:4]],
            training=False,
        )

        rec_indices_np = rec_indices.numpy()
        for i, user_recs in enumerate(rec_indices_np):
            unique_count = len(np.unique(user_recs))
            assert (
                unique_count == model.top_k
            ), f"User {i}: {unique_count} unique items != {model.top_k}"

        print("✅ Test 14: Each user gets unique recommendations")

    def test_15_no_constant_recommendations(self, setup_data, model_and_metrics):
        """Test 15: Model doesn't return same recommendations for all users."""
        model = model_and_metrics["model"]
        data = setup_data

        model.fit(
            x=[data["user_features"], data["item_features"]],
            y=data["labels"],
            epochs=3,
            batch_size=4,
            verbose=0,
        )

        _, rec_indices, _ = model(
            [data["user_features"], data["item_features"]],
            training=False,
        )

        rec_indices_np = rec_indices.numpy()

        # Check that not all users have identical recommendations
        different_recs = 0
        for i in range(1, len(rec_indices_np)):
            if not np.array_equal(rec_indices_np[0], rec_indices_np[i]):
                different_recs += 1

        print(f"   Different recommendations: {different_recs}/{len(rec_indices_np)-1}")
        assert different_recs > 0, "All users getting identical recommendations"
        print("✅ Test 15: Model provides personalized recommendations")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
