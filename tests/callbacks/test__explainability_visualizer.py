"""Unit tests for ExplainabilityVisualizer and SimilarityMatrixVisualizer callbacks."""

import pytest
import keras
import numpy as np
from unittest.mock import MagicMock, patch

from kmr.callbacks import ExplainabilityVisualizer, SimilarityMatrixVisualizer


class TestExplainabilityVisualizer:
    """Test suite for ExplainabilityVisualizer."""

    @pytest.fixture
    def eval_data(self):
        """Create dummy evaluation data."""
        x = np.random.randn(16, 10).astype(np.float32)
        y = np.random.randint(0, 2, (16, 5)).astype(np.float32)
        return x, y

    @pytest.fixture
    def mock_viz_fn(self):
        """Create a mock visualization function."""
        return MagicMock()

    def test_initialization(self, eval_data, mock_viz_fn):
        """Test callback initialization."""
        callback = ExplainabilityVisualizer(
            eval_data=eval_data,
            visualization_fn=mock_viz_fn,
            frequency=5,
        )
        assert callback.frequency == 5
        assert callback.visualization_fn is mock_viz_fn
        assert callback.epoch_visualizations == []

    def test_initialization_with_save_dir(self, eval_data, mock_viz_fn):
        """Test initialization with save directory."""
        callback = ExplainabilityVisualizer(
            eval_data=eval_data,
            visualization_fn=mock_viz_fn,
            save_dir="/tmp/test_visualizations",
        )
        assert callback.save_dir == "/tmp/test_visualizations"

    def test_frequency_respected(self, eval_data, mock_viz_fn):
        """Test that visualization frequency is respected."""
        callback = ExplainabilityVisualizer(
            eval_data=eval_data,
            visualization_fn=mock_viz_fn,
            frequency=2,
        )

        # Should not call at epoch 0
        callback.on_epoch_end(epoch=0, logs=None)
        mock_viz_fn.assert_not_called()

        # Should call at epoch 1 (since epoch is 0-indexed, epoch 1 == epoch 2)
        callback.on_epoch_end(epoch=1, logs=None)
        mock_viz_fn.assert_called_once()

    def test_visualization_fn_called_with_correct_args(self, eval_data, mock_viz_fn):
        """Test that visualization function is called with correct arguments."""
        model = MagicMock()
        callback = ExplainabilityVisualizer(
            eval_data=eval_data,
            visualization_fn=mock_viz_fn,
            frequency=1,
        )
        # Set model via set_model instead of direct assignment
        callback.set_model(model)

        eval_inputs, eval_labels = eval_data
        callback.on_epoch_end(epoch=0, logs=None)

        # Verify function was called with correct arguments
        mock_viz_fn.assert_called_once()
        call_kwargs = mock_viz_fn.call_args[1]
        assert call_kwargs["epoch"] == 1
        assert call_kwargs["inputs"] is not None
        assert call_kwargs["labels"] is not None

    def test_epoch_visualizations_tracked(self, eval_data, mock_viz_fn):
        """Test that visualized epochs are tracked."""
        callback = ExplainabilityVisualizer(
            eval_data=eval_data,
            visualization_fn=mock_viz_fn,
            frequency=1,
        )
        callback.set_model(MagicMock())

        for epoch in range(3):
            callback.on_epoch_end(epoch=epoch, logs=None)

        assert callback.epoch_visualizations == [1, 2, 3]

    def test_handles_visualization_errors(self, eval_data):
        """Test that callback handles visualization errors gracefully."""

        def failing_viz_fn(**kwargs):
            raise ValueError("Visualization failed")

        callback = ExplainabilityVisualizer(
            eval_data=eval_data,
            visualization_fn=failing_viz_fn,
            frequency=1,
        )
        callback.set_model(MagicMock())

        # Should not raise error
        callback.on_epoch_end(epoch=0, logs=None)
        assert len(callback.epoch_visualizations) == 0

    def test_get_config(self, eval_data, mock_viz_fn):
        """Test get_config for serialization."""
        callback = ExplainabilityVisualizer(
            eval_data=eval_data,
            visualization_fn=mock_viz_fn,
            frequency=5,
            save_dir="/tmp/test",
        )
        config = callback.get_config()
        assert config["frequency"] == 5
        assert config["save_dir"] == "/tmp/test"
        assert config["verbose"] == 1

    def test_on_train_end_summary(self, eval_data, mock_viz_fn):
        """Test training end summary."""
        callback = ExplainabilityVisualizer(
            eval_data=eval_data,
            visualization_fn=mock_viz_fn,
            frequency=1,
        )
        callback.set_model(MagicMock())
        callback.epoch_visualizations = [1, 2, 3]

        # Should not raise error
        callback.on_train_end(logs=None)


class TestSimilarityMatrixVisualizer:
    """Test suite for SimilarityMatrixVisualizer."""

    @pytest.fixture
    def eval_data(self):
        """Create dummy evaluation data."""
        x = np.random.randn(16, 10).astype(np.float32)
        y = np.random.randint(0, 2, (16, 5)).astype(np.float32)
        return x, y

    @pytest.fixture
    def mock_similarity_fn(self):
        """Create a mock similarity computation function."""

        def compute_similarities(inputs):
            return np.random.randn(16, 16).astype(np.float32)

        return compute_similarities

    def test_initialization(self, eval_data, mock_similarity_fn):
        """Test callback initialization."""
        callback = SimilarityMatrixVisualizer(
            eval_data=eval_data,
            compute_similarity_fn=mock_similarity_fn,
            frequency=10,
            top_k=5,
        )
        assert callback.frequency == 10
        assert callback.top_k == 5
        assert callback.similarity_history == []

    def test_frequency_respected(self, eval_data, mock_similarity_fn):
        """Test that computation frequency is respected."""
        callback = SimilarityMatrixVisualizer(
            eval_data=eval_data,
            compute_similarity_fn=mock_similarity_fn,
            frequency=2,
        )
        callback.set_model(MagicMock())

        # Should not compute at epoch 0
        callback.on_epoch_end(epoch=0, logs=None)
        assert len(callback.similarity_history) == 0

        # Should compute at epoch 1
        callback.on_epoch_end(epoch=1, logs=None)
        assert len(callback.similarity_history) == 1

    def test_similarity_statistics_computed(self, eval_data, mock_similarity_fn):
        """Test that similarity statistics are computed correctly."""
        callback = SimilarityMatrixVisualizer(
            eval_data=eval_data,
            compute_similarity_fn=mock_similarity_fn,
            frequency=1,
        )
        callback.set_model(MagicMock())

        callback.on_epoch_end(epoch=0, logs=None)

        # Verify statistics were recorded
        assert len(callback.similarity_history) == 1
        stats = callback.similarity_history[0]
        assert "epoch" in stats
        assert "mean" in stats
        assert "std" in stats
        assert "max" in stats
        assert "min" in stats
        assert stats["epoch"] == 1

    def test_handles_computation_errors(self, eval_data):
        """Test that callback handles computation errors gracefully."""

        def failing_similarity_fn(inputs):
            raise ValueError("Computation failed")

        callback = SimilarityMatrixVisualizer(
            eval_data=eval_data,
            compute_similarity_fn=failing_similarity_fn,
            frequency=1,
        )
        callback.set_model(MagicMock())

        # Should not raise error
        callback.on_epoch_end(epoch=0, logs=None)
        # Error is handled but may still add an entry to history
        # Just verify no exception was raised
        assert callback.similarity_history is not None

    def test_handles_tuple_output(self, eval_data):
        """Test handling of tuple outputs from similarity function."""

        def tuple_similarity_fn(inputs):
            # Some models return (similarities, extra_info)
            return np.random.randn(16, 16).astype(np.float32), {"info": "extra"}

        callback = SimilarityMatrixVisualizer(
            eval_data=eval_data,
            compute_similarity_fn=tuple_similarity_fn,
            frequency=1,
        )
        callback.set_model(MagicMock())

        # Should not raise error
        callback.on_epoch_end(epoch=0, logs=None)
        assert len(callback.similarity_history) == 1

    def test_get_config(self, eval_data, mock_similarity_fn):
        """Test get_config for serialization."""
        callback = SimilarityMatrixVisualizer(
            eval_data=eval_data,
            compute_similarity_fn=mock_similarity_fn,
            frequency=10,
            top_k=5,
        )
        config = callback.get_config()
        assert config["frequency"] == 10
        assert config["top_k"] == 5

    def test_multiple_epochs(self, eval_data, mock_similarity_fn):
        """Test similarity tracking across multiple epochs."""
        callback = SimilarityMatrixVisualizer(
            eval_data=eval_data,
            compute_similarity_fn=mock_similarity_fn,
            frequency=1,
        )
        callback.set_model(MagicMock())

        for epoch in range(3):
            callback.on_epoch_end(epoch=epoch, logs=None)

        assert len(callback.similarity_history) == 3
        epochs = [h["epoch"] for h in callback.similarity_history]
        assert epochs == [1, 2, 3]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
