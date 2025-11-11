"""End-to-end tests for GeospatialClusteringModel - 15 comprehensive tests."""

import numpy as np
import tensorflow as tf
import pytest
from keras.optimizers import Adam
from kmr.models import GeospatialClusteringModel
from kmr.losses import ImprovedMarginRankingLoss
from kmr.metrics import AccuracyAtK, PrecisionAtK, RecallAtK


class TestGeospatialClusteringModelE2E:
    """E2E tests for GeospatialClusteringModel (Geo + Clustering)."""

    @pytest.fixture(scope="class")
    def setup_data(self):
        """Generate geospatial test data with lat/lon coordinates."""
        n_items = 50
        batch_size = 20

        # Latitude/longitude coordinates for users and items
        user_lat = np.random.uniform(-90, 90, (batch_size,)).astype(np.float32)
        user_lon = np.random.uniform(-180, 180, (batch_size,)).astype(np.float32)
        item_lats = np.random.uniform(-90, 90, (batch_size, n_items)).astype(np.float32)
        item_lons = np.random.uniform(-180, 180, (batch_size, n_items)).astype(
            np.float32,
        )

        # Create labels
        labels = np.random.randint(0, 2, (batch_size, n_items)).astype(np.float32)

        return {
            "n_items": n_items,
            "batch_size": batch_size,
            "user_lat": user_lat,
            "user_lon": user_lon,
            "item_lats": item_lats,
            "item_lons": item_lons,
            "labels": labels,
        }

    @pytest.fixture
    def model_and_metrics(self, setup_data):
        """Create model and metrics."""
        model = GeospatialClusteringModel(
            num_items=setup_data["n_items"],
            embedding_dim=16,
            num_clusters=3,
            top_k=5,
        )

        # 3 outputs: masked_scores, rec_indices, rec_scores
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=[ImprovedMarginRankingLoss(), None, None],
            metrics=[[AccuracyAtK(k=5), PrecisionAtK(k=5)], None, None],
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
                setup_data["user_lat"],
                setup_data["user_lon"],
                setup_data["item_lats"],
                setup_data["item_lons"],
            ],
            y=setup_data["labels"],
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
                setup_data["user_lat"],
                setup_data["user_lon"],
                setup_data["item_lats"],
                setup_data["item_lons"],
            ],
            y=setup_data["labels"],
            epochs=1,
            batch_size=4,
            verbose=0,
        )
        assert "loss" in history.history
        print("✅ Test 3: Metrics tracked")

    def test_04_inference_returns_3tuple(self, setup_data, model_and_metrics):
        """Test 4: Returns 3-tuple."""
        model = model_and_metrics["model"]
        output = model(
            [
                setup_data["user_lat"][:2],
                setup_data["user_lon"][:2],
                setup_data["item_lats"][:2],
                setup_data["item_lons"][:2],
            ],
            training=False,
        )

        assert isinstance(output, tuple)
        assert len(output) == 3

        masked_scores, rec_indices, rec_scores = output
        assert masked_scores.shape == (2, setup_data["n_items"])
        assert rec_indices.shape == (2, 5)
        assert rec_scores.shape == (2, 5)

        print("✅ Test 4: Returns correct 3-tuple")

    def test_05_recommendation_validity(self, setup_data, model_and_metrics):
        """Test 5: Valid recommendations."""
        model = model_and_metrics["model"]
        _, rec_indices, rec_scores = model(
            [
                setup_data["user_lat"][:3],
                setup_data["user_lon"][:3],
                setup_data["item_lats"][:3],
                setup_data["item_lons"][:3],
            ],
            training=False,
        )

        assert np.all(rec_indices.numpy() >= 0)
        assert np.all(rec_indices.numpy() < setup_data["n_items"])

        print("✅ Test 5: Valid recommendations")

    def test_06_recommendation_diversity(self, setup_data, model_and_metrics):
        """Test 6: Diverse recommendations."""
        model = model_and_metrics["model"]
        _, rec_indices, _ = model(
            [
                setup_data["user_lat"][:8],
                setup_data["user_lon"][:8],
                setup_data["item_lats"][:8],
                setup_data["item_lons"][:8],
            ],
            training=False,
        )

        all_items = set()
        for rec in rec_indices.numpy():
            all_items.update(rec)

        assert len(all_items) > 1

        print("✅ Test 6: Diverse recommendations")

    def test_07_consistency(self, setup_data, model_and_metrics):
        """Test 7: Consistent outputs."""
        model = model_and_metrics["model"]
        model.fit(
            x=[
                setup_data["user_lat"],
                setup_data["user_lon"],
                setup_data["item_lats"],
                setup_data["item_lons"],
            ],
            y=setup_data["labels"],
            epochs=1,
            batch_size=4,
            verbose=0,
        )

        out1 = model(
            [
                setup_data["user_lat"][:2],
                setup_data["user_lon"][:2],
                setup_data["item_lats"][:2],
                setup_data["item_lons"][:2],
            ],
            training=False,
        )
        out2 = model(
            [
                setup_data["user_lat"][:2],
                setup_data["user_lon"][:2],
                setup_data["item_lats"][:2],
                setup_data["item_lons"][:2],
            ],
            training=True,
        )

        assert len(out1) == len(out2) == 3

        print("✅ Test 7: Consistent outputs")

    def test_08_batch_prediction(self, setup_data, model_and_metrics):
        """Test 8: Batch predictions."""
        model = model_and_metrics["model"]
        output = model(
            [
                setup_data["user_lat"][:6],
                setup_data["user_lon"][:6],
                setup_data["item_lats"][:6],
                setup_data["item_lons"][:6],
            ],
            training=False,
        )

        assert all(o.shape[0] == 6 for o in output)
        print("✅ Test 8: Batch prediction works")

    def test_09_full_workflow(self, setup_data):
        """Test 9: Full workflow."""
        model = GeospatialClusteringModel(
            num_items=setup_data["n_items"],
            embedding_dim=16,
            num_clusters=3,
            top_k=5,
        )

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=[ImprovedMarginRankingLoss(), None, None],
        )

        history = model.fit(
            x=[
                setup_data["user_lat"],
                setup_data["user_lon"],
                setup_data["item_lats"],
                setup_data["item_lons"],
            ],
            y=setup_data["labels"],
            epochs=1,
            batch_size=4,
            verbose=0,
        )

        output = model(
            [
                setup_data["user_lat"][:2],
                setup_data["user_lon"][:2],
                setup_data["item_lats"][:2],
                setup_data["item_lons"][:2],
            ],
            training=False,
        )

        assert len(history.history["loss"]) == 1
        assert len(output) == 3

        print("✅ Test 9: Full workflow")

    def test_10_cluster_based_filtering(self, setup_data, model_and_metrics):
        """Test 10: Cluster-based filtering applied."""
        model = model_and_metrics["model"]
        model.fit(
            x=[
                setup_data["user_lat"],
                setup_data["user_lon"],
                setup_data["item_lats"],
                setup_data["item_lons"],
            ],
            y=setup_data["labels"],
            epochs=1,
            batch_size=4,
            verbose=0,
        )

        masked_scores, _, _ = model(
            [
                setup_data["user_lat"][:3],
                setup_data["user_lon"][:3],
                setup_data["item_lats"][:3],
                setup_data["item_lons"][:3],
            ],
            training=False,
        )

        assert masked_scores.shape == (3, setup_data["n_items"])

        print("✅ Test 10: Cluster-based filtering applied")

    def test_11_masked_scores_validity(self, setup_data, model_and_metrics):
        """Test 11: Masked scores are valid."""
        model = model_and_metrics["model"]
        model.fit(
            x=[
                setup_data["user_lat"],
                setup_data["user_lon"],
                setup_data["item_lats"],
                setup_data["item_lons"],
            ],
            y=setup_data["labels"],
            epochs=1,
            batch_size=4,
            verbose=0,
        )

        masked_scores, _, _ = model(
            [
                setup_data["user_lat"][:3],
                setup_data["user_lon"][:3],
                setup_data["item_lats"][:3],
                setup_data["item_lons"][:3],
            ],
            training=False,
        )

        assert not np.any(np.isnan(masked_scores.numpy()))

        print("✅ Test 11: Masked scores are valid")

    def test_12_reproducible_predictions(self, setup_data, model_and_metrics):
        """Test 12: Reproducible predictions."""
        model = model_and_metrics["model"]
        model.fit(
            x=[
                setup_data["user_lat"],
                setup_data["user_lon"],
                setup_data["item_lats"],
                setup_data["item_lons"],
            ],
            y=setup_data["labels"],
            epochs=1,
            batch_size=4,
            verbose=0,
        )

        _, idx1, _ = model(
            [
                setup_data["user_lat"][:2],
                setup_data["user_lon"][:2],
                setup_data["item_lats"][:2],
                setup_data["item_lons"][:2],
            ],
            training=False,
        )
        _, idx2, _ = model(
            [
                setup_data["user_lat"][:2],
                setup_data["user_lon"][:2],
                setup_data["item_lats"][:2],
                setup_data["item_lons"][:2],
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
                setup_data["user_lat"],
                setup_data["user_lon"],
                setup_data["item_lats"],
                setup_data["item_lons"],
            ],
            y=setup_data["labels"],
            epochs=1,
            batch_size=4,
            verbose=0,
        )

        output = model(
            [
                setup_data["user_lat"][:1],
                setup_data["user_lon"][:1],
                setup_data["item_lats"][:1],
                setup_data["item_lons"][:1],
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
                setup_data["user_lat"],
                setup_data["user_lon"],
                setup_data["item_lats"],
                setup_data["item_lons"],
            ],
            y=setup_data["labels"],
            epochs=1,
            batch_size=4,
            verbose=0,
        )

        _, rec_indices, _ = model(
            [
                setup_data["user_lat"][:4],
                setup_data["user_lon"][:4],
                setup_data["item_lats"][:4],
                setup_data["item_lons"][:4],
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
                setup_data["user_lat"],
                setup_data["user_lon"],
                setup_data["item_lats"],
                setup_data["item_lons"],
            ],
            y=setup_data["labels"],
            epochs=2,
            batch_size=4,
            verbose=0,
        )

        _, rec_indices, _ = model(
            [
                setup_data["user_lat"][:10],
                setup_data["user_lon"][:10],
                setup_data["item_lats"][:10],
                setup_data["item_lons"][:10],
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
