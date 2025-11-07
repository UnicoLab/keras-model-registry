"""Unit tests for GeospatialClusteringModel with Keras compatibility."""

import numpy as np
import pytest
import keras
import tensorflow as tf

from kmr.models import GeospatialClusteringModel
from kmr.losses import ImprovedMarginRankingLoss


class TestGeospatialClusteringModelKerasCompatibility:
    """Test suite for GeospatialClusteringModel."""

    @pytest.fixture
    def model_config(self):
        """Provide common model configuration."""
        return {
            "num_items": 50,
            "embedding_dim": 16,
            "num_clusters": 4,
            "top_k": 5,
            "threshold": 0.1,
        }

    @pytest.fixture
    def model(self, model_config):
        """Create a fresh model instance."""
        return GeospatialClusteringModel(**model_config)

    @pytest.fixture
    def sample_data(self):
        """Generate sample data."""
        batch_size = 8
        num_items = 50
        user_lat = np.random.uniform(-90, 90, (batch_size,)).astype(np.float32)
        user_lon = np.random.uniform(-180, 180, (batch_size,)).astype(np.float32)
        item_lats = np.random.uniform(-90, 90, (batch_size, num_items)).astype(
            np.float32,
        )
        item_lons = np.random.uniform(-180, 180, (batch_size, num_items)).astype(
            np.float32,
        )
        labels = np.random.randint(0, 2, (batch_size, num_items)).astype(np.float32)
        return (user_lat, user_lon, item_lats, item_lons), labels

    # ===== Initialization Tests =====

    def test_initialization(self, model_config):
        """Test model initialization."""
        model = GeospatialClusteringModel(**model_config)
        assert model.num_items == 50
        assert model.embedding_dim == 16
        assert model.num_clusters == 4
        assert model.top_k == 5

    def test_invalid_num_items(self):
        """Test invalid num_items raises error."""
        with pytest.raises(ValueError):
            GeospatialClusteringModel(num_items=-1)

    def test_invalid_top_k(self):
        """Test invalid top_k raises error."""
        with pytest.raises(ValueError):
            GeospatialClusteringModel(num_items=100, top_k=150)

    # ===== Inference Tests =====

    def test_call_inference_mode(self, model, sample_data):
        """Test call() in inference mode returns tuple."""
        inputs, _ = sample_data
        output = model(inputs, training=False)
        assert isinstance(output, tuple)
        assert len(output) == 2
        indices, scores = output
        assert indices.shape == (8, 5)  # (batch_size, top_k)
        assert scores.shape == (8, 5)

    def test_predict(self, model, sample_data):
        """Test predict()."""
        inputs, _ = sample_data
        indices, scores = model.predict(inputs)
        assert indices.shape == (8, 5)
        assert scores.shape == (8, 5)

    # ===== Compilation Tests =====

    def test_compile_basic(self, model):
        """Test compilation."""
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=ImprovedMarginRankingLoss(),
        )
        assert model.optimizer is not None

    def test_get_config(self, model):
        """Test serialization."""
        config = model.get_config()
        assert config["num_items"] == 50
        assert config["embedding_dim"] == 16

    def test_from_config(self, model):
        """Test deserialization."""
        config = model.get_config()
        new_model = GeospatialClusteringModel.from_config(config)
        assert new_model.num_items == model.num_items

    # ===== Integration Tests =====

    def test_inference_produces_valid_output(self, model, sample_data):
        """Test that inference produces valid output."""
        inputs, _ = sample_data
        indices, scores = model(inputs, training=False)
        # Just verify shapes are reasonable
        assert indices.ndim == 2
        assert scores.ndim == 2
        assert indices.shape[0] == 8  # batch_size
        assert scores.shape[0] == 8

    def test_output_dtypes(self, model, sample_data):
        """Test output dtypes."""
        inputs, _ = sample_data
        indices, scores = model(inputs, training=False)
        assert scores.dtype == tf.float32
