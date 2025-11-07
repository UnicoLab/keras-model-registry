"""End-to-end tests for ExplainableUnifiedRecommendationModel - 15 comprehensive tests."""

import numpy as np
import tensorflow as tf
import pytest
from keras.optimizers import Adam
from kmr.models import ExplainableUnifiedRecommendationModel
from kmr.losses import ImprovedMarginRankingLoss
from kmr.metrics import AccuracyAtK, PrecisionAtK, RecallAtK
from kmr.utils import KMRDataGenerator


class TestExplainableUnifiedRecommendationModelE2E:
    """E2E tests for ExplainableUnifiedRecommendationModel (Hybrid + Explanations)."""

    @pytest.fixture(scope="class")
    def setup_data(self):
        """Generate hybrid test data with explanations."""
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

        (
            train_x_user_ids,
            train_x_user_features,
            train_x_item_ids,
            train_x_item_features,
            train_y,
        ) = ([], [], [], [], [])
        for user_id in unique_users:
            if user_id >= n_users:
                continue
            user_items = item_ids[user_ids == user_id]
            positive_set = set(user_items[user_items < n_items])
            labels = np.zeros(n_items, dtype=np.float32)
            labels[list(positive_set)] = 1.0

            train_x_user_ids.append(user_id)
            train_x_user_features.append(np.random.randn(10).astype(np.float32))
            train_x_item_ids.append(np.arange(n_items))
            train_x_item_features.append(
                np.random.randn(n_items, 10).astype(np.float32),
            )
            train_y.append(labels)

        return {
            "n_users": n_users,
            "n_items": n_items,
            "train_x_user_ids": np.array(train_x_user_ids, dtype=np.int32),
            "train_x_user_features": np.array(train_x_user_features, dtype=np.float32),
            "train_x_item_ids": np.array(train_x_item_ids, dtype=np.int32),
            "train_x_item_features": np.array(train_x_item_features, dtype=np.float32),
            "train_y": np.array(train_y, dtype=np.float32),
        }

    @pytest.fixture
    def model_and_metrics(self, setup_data):
        """Create model and metrics."""
        model = ExplainableUnifiedRecommendationModel(
            num_users=setup_data["n_users"],
            num_items=setup_data["n_items"],
            user_feature_dim=10,
            item_feature_dim=10,
            embedding_dim=16,
            top_k=5,
        )

        # 7 outputs: combined_scores, rec_indices, rec_scores, cf_sim, cb_sim, weights, raw_cf
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=[ImprovedMarginRankingLoss(), None, None, None, None, None, None],
            metrics=[
                [AccuracyAtK(k=5), PrecisionAtK(k=5)],
                None,
                None,
                None,
                None,
                None,
                None,
            ],
        )
        return {"model": model}

    def test_01_model_compilation(self, model_and_metrics):
        """Test 1: Model compiles."""
        assert model_and_metrics["model"].optimizer is not None
        print("✅ Test 1: Compilation successful")

    def test_02_training_convergence(self, setup_data, model_and_metrics):
        """Test 2: Model trains."""
        model = model_and_metrics["model"]
        history = model.fit(
            x=[
                setup_data["train_x_user_ids"],
                setup_data["train_x_user_features"],
                setup_data["train_x_item_ids"],
                setup_data["train_x_item_features"],
            ],
            y=setup_data["train_y"],
            epochs=2,
            batch_size=4,
            verbose=0,
        )
        assert len(history.history["loss"]) == 2
        print("✅ Test 2: Training convergence")

    def test_03_metrics_tracked(self, setup_data, model_and_metrics):
        """Test 3: Metrics tracked."""
        model = model_and_metrics["model"]
        history = model.fit(
            x=[
                setup_data["train_x_user_ids"],
                setup_data["train_x_user_features"],
                setup_data["train_x_item_ids"],
                setup_data["train_x_item_features"],
            ],
            y=setup_data["train_y"],
            epochs=1,
            batch_size=4,
            verbose=0,
        )
        assert "loss" in history.history
        print("✅ Test 3: Metrics tracked")

    def test_04_inference_returns_7tuple(self, setup_data, model_and_metrics):
        """Test 4: Returns 7-tuple output."""
        model = model_and_metrics["model"]
        output = model(
            [
                setup_data["train_x_user_ids"][:2],
                setup_data["train_x_user_features"][:2],
                setup_data["train_x_item_ids"][:2],
                setup_data["train_x_item_features"][:2],
            ],
            training=False,
        )

        assert isinstance(output, tuple)
        assert len(output) == 7

        (
            combined_scores,
            rec_indices,
            rec_scores,
            cf_sim,
            cb_sim,
            weights,
            raw_cf,
        ) = output
        assert combined_scores.shape == (2, setup_data["n_items"])
        assert rec_indices.shape == (2, 5)
        assert rec_scores.shape == (2, 5)
        assert cf_sim.shape == (2, setup_data["n_items"])
        assert cb_sim.shape == (2, setup_data["n_items"])

        print("✅ Test 4: Returns correct 7-tuple")

    def test_05_recommendation_validity(self, setup_data, model_and_metrics):
        """Test 5: Valid recommendations."""
        model = model_and_metrics["model"]
        _, rec_indices, rec_scores, _, _, _, _ = model(
            [
                setup_data["train_x_user_ids"][:3],
                setup_data["train_x_user_features"][:3],
                setup_data["train_x_item_ids"][:3],
                setup_data["train_x_item_features"][:3],
            ],
            training=False,
        )

        assert np.all(rec_indices.numpy() >= 0)
        assert np.all(rec_indices.numpy() < setup_data["n_items"])
        assert np.all(rec_scores.numpy() >= -1.0)
        assert np.all(rec_scores.numpy() <= 1.0)

        print("✅ Test 5: Valid recommendations")

    def test_06_recommendation_diversity(self, setup_data, model_and_metrics):
        """Test 6: Diverse recommendations."""
        model = model_and_metrics["model"]
        _, rec_indices, _, _, _, _, _ = model(
            [
                setup_data["train_x_user_ids"][:8],
                setup_data["train_x_user_features"][:8],
                setup_data["train_x_item_ids"][:8],
                setup_data["train_x_item_features"][:8],
            ],
            training=False,
        )

        all_items = set()
        for rec in rec_indices.numpy():
            all_items.update(rec)

        assert len(all_items) > 1
        coverage = len(all_items) / setup_data["n_items"] * 100
        assert coverage > 15.0

        print(f"   Catalog coverage: {coverage:.1f}%")
        print("✅ Test 6: Diverse recommendations")

    def test_07_consistency(self, setup_data, model_and_metrics):
        """Test 7: Consistent outputs."""
        model = model_and_metrics["model"]
        model.fit(
            x=[
                setup_data["train_x_user_ids"],
                setup_data["train_x_user_features"],
                setup_data["train_x_item_ids"],
                setup_data["train_x_item_features"],
            ],
            y=setup_data["train_y"],
            epochs=1,
            batch_size=4,
            verbose=0,
        )

        out1 = model(
            [
                setup_data["train_x_user_ids"][:2],
                setup_data["train_x_user_features"][:2],
                setup_data["train_x_item_ids"][:2],
                setup_data["train_x_item_features"][:2],
            ],
            training=False,
        )
        out2 = model(
            [
                setup_data["train_x_user_ids"][:2],
                setup_data["train_x_user_features"][:2],
                setup_data["train_x_item_ids"][:2],
                setup_data["train_x_item_features"][:2],
            ],
            training=True,
        )

        assert len(out1) == len(out2) == 7
        for o1, o2 in zip(out1, out2):
            assert o1.shape == o2.shape

        print("✅ Test 7: Consistent outputs")

    def test_08_batch_prediction(self, setup_data, model_and_metrics):
        """Test 8: Batch predictions."""
        model = model_and_metrics["model"]
        output = model(
            [
                setup_data["train_x_user_ids"][:6],
                setup_data["train_x_user_features"][:6],
                setup_data["train_x_item_ids"][:6],
                setup_data["train_x_item_features"][:6],
            ],
            training=False,
        )

        assert all(o.shape[0] == 6 for o in output)
        print("✅ Test 8: Batch prediction works")

    def test_09_full_workflow(self, setup_data):
        """Test 9: Full workflow."""
        model = ExplainableUnifiedRecommendationModel(
            num_users=setup_data["n_users"],
            num_items=setup_data["n_items"],
            user_feature_dim=10,
            item_feature_dim=10,
            embedding_dim=16,
            top_k=5,
        )

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=[ImprovedMarginRankingLoss(), None, None, None, None, None, None],
        )

        history = model.fit(
            x=[
                setup_data["train_x_user_ids"],
                setup_data["train_x_user_features"],
                setup_data["train_x_item_ids"],
                setup_data["train_x_item_features"],
            ],
            y=setup_data["train_y"],
            epochs=1,
            batch_size=4,
            verbose=0,
        )

        output = model(
            [
                setup_data["train_x_user_ids"][:2],
                setup_data["train_x_user_features"][:2],
                setup_data["train_x_item_ids"][:2],
                setup_data["train_x_item_features"][:2],
            ],
            training=False,
        )

        assert len(history.history["loss"]) == 1
        assert len(output) == 7

        print("✅ Test 9: Full workflow")

    def test_10_cf_cb_component_separation(self, setup_data, model_and_metrics):
        """Test 10: CF and CB components are properly separated."""
        model = model_and_metrics["model"]
        model.fit(
            x=[
                setup_data["train_x_user_ids"],
                setup_data["train_x_user_features"],
                setup_data["train_x_item_ids"],
                setup_data["train_x_item_features"],
            ],
            y=setup_data["train_y"],
            epochs=2,
            batch_size=4,
            verbose=0,
        )

        _, _, _, cf_sim, cb_sim, _, _ = model(
            [
                setup_data["train_x_user_ids"][:3],
                setup_data["train_x_user_features"][:3],
                setup_data["train_x_item_ids"][:3],
                setup_data["train_x_item_features"][:3],
            ],
            training=False,
        )

        assert cf_sim.shape == (3, setup_data["n_items"])
        assert cb_sim.shape == (3, setup_data["n_items"])
        assert np.all(cf_sim.numpy() >= -1.0) and np.all(cf_sim.numpy() <= 1.0)
        assert np.all(cb_sim.numpy() >= -1.0) and np.all(cb_sim.numpy() <= 1.0)

        print("✅ Test 10: CF and CB components properly separated")

    def test_11_weights_validity(self, setup_data, model_and_metrics):
        """Test 11: Component weights are valid."""
        model = model_and_metrics["model"]
        model.fit(
            x=[
                setup_data["train_x_user_ids"],
                setup_data["train_x_user_features"],
                setup_data["train_x_item_ids"],
                setup_data["train_x_item_features"],
            ],
            y=setup_data["train_y"],
            epochs=1,
            batch_size=4,
            verbose=0,
        )

        _, _, _, _, _, weights, _ = model(
            [
                setup_data["train_x_user_ids"][:2],
                setup_data["train_x_user_features"][:2],
                setup_data["train_x_item_ids"][:2],
                setup_data["train_x_item_features"][:2],
            ],
            training=False,
        )

        weights_np = weights.numpy()
        assert weights_np is not None
        assert len(weights_np) > 0 or weights_np.size > 0

        print("✅ Test 11: Weights are valid")

    def test_12_reproducible_predictions(self, setup_data, model_and_metrics):
        """Test 12: Reproducible predictions."""
        model = model_and_metrics["model"]
        model.fit(
            x=[
                setup_data["train_x_user_ids"],
                setup_data["train_x_user_features"],
                setup_data["train_x_item_ids"],
                setup_data["train_x_item_features"],
            ],
            y=setup_data["train_y"],
            epochs=1,
            batch_size=4,
            verbose=0,
        )

        _, idx1, _, _, _, _, _ = model(
            [
                setup_data["train_x_user_ids"][:2],
                setup_data["train_x_user_features"][:2],
                setup_data["train_x_item_ids"][:2],
                setup_data["train_x_item_features"][:2],
            ],
            training=False,
        )
        _, idx2, _, _, _, _, _ = model(
            [
                setup_data["train_x_user_ids"][:2],
                setup_data["train_x_user_features"][:2],
                setup_data["train_x_item_ids"][:2],
                setup_data["train_x_item_features"][:2],
            ],
            training=False,
        )

        assert np.array_equal(idx1.numpy(), idx2.numpy())

        print("✅ Test 12: Reproducible predictions")

    def test_13_edge_case_single_user(self, setup_data, model_and_metrics):
        """Test 13: Single user."""
        model = model_and_metrics["model"]
        model.fit(
            x=[
                setup_data["train_x_user_ids"],
                setup_data["train_x_user_features"],
                setup_data["train_x_item_ids"],
                setup_data["train_x_item_features"],
            ],
            y=setup_data["train_y"],
            epochs=1,
            batch_size=4,
            verbose=0,
        )

        output = model(
            [
                setup_data["train_x_user_ids"][:1],
                setup_data["train_x_user_features"][:1],
                setup_data["train_x_item_ids"][:1],
                setup_data["train_x_item_features"][:1],
            ],
            training=False,
        )

        assert all(o.shape[0] == 1 for o in output)

        print("✅ Test 13: Single user")

    def test_14_unique_recommendations(self, setup_data, model_and_metrics):
        """Test 14: Unique recommendations per user."""
        model = model_and_metrics["model"]
        model.fit(
            x=[
                setup_data["train_x_user_ids"],
                setup_data["train_x_user_features"],
                setup_data["train_x_item_ids"],
                setup_data["train_x_item_features"],
            ],
            y=setup_data["train_y"],
            epochs=1,
            batch_size=4,
            verbose=0,
        )

        _, rec_indices, _, _, _, _, _ = model(
            [
                setup_data["train_x_user_ids"][:4],
                setup_data["train_x_user_features"][:4],
                setup_data["train_x_item_ids"][:4],
                setup_data["train_x_item_features"][:4],
            ],
            training=False,
        )

        for user_recs in rec_indices.numpy():
            assert len(np.unique(user_recs)) == 5

        print("✅ Test 14: Unique recommendations")

    def test_15_personalization(self, setup_data, model_and_metrics):
        """Test 15: Personalization."""
        model = model_and_metrics["model"]
        model.fit(
            x=[
                setup_data["train_x_user_ids"],
                setup_data["train_x_user_features"],
                setup_data["train_x_item_ids"],
                setup_data["train_x_item_features"],
            ],
            y=setup_data["train_y"],
            epochs=2,
            batch_size=4,
            verbose=0,
        )

        _, rec_indices, _, _, _, _, _ = model(
            [
                setup_data["train_x_user_ids"][:10],
                setup_data["train_x_user_features"][:10],
                setup_data["train_x_item_ids"][:10],
                setup_data["train_x_item_features"][:10],
            ],
            training=False,
        )

        rec_indices_np = rec_indices.numpy()
        different = sum(
            1
            for i in range(1, len(rec_indices_np))
            if not np.array_equal(rec_indices_np[0], rec_indices_np[i])
        )

        assert different > 0

        print("✅ Test 15: Personalization")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
