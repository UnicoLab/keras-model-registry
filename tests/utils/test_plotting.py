"""Tests for plotting utilities."""

import numpy as np
import pytest
import plotly.graph_objects as go
from kerasfactory.utils import KerasFactoryPlotter


class TestTimeSeriesPlotting:
    """Test time series plotting methods."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample time series data."""
        n_samples, seq_len, pred_len, n_features = 5, 96, 12, 7
        X = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
        y_true = np.random.randn(n_samples, pred_len, n_features).astype(np.float32)
        y_pred = y_true + 0.1 * np.random.randn(n_samples, pred_len, n_features).astype(
            np.float32,
        )
        return X, y_true, y_pred

    def test_plot_timeseries(self, sample_data):
        """Test basic time series plotting."""
        X, y_true, y_pred = sample_data
        fig = KerasFactoryPlotter.plot_timeseries(
            X,
            y_true,
            y_pred,
            n_samples_to_plot=3,
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert "Time Series" in fig.layout.title.text

    def test_plot_timeseries_only_input(self, sample_data):
        """Test time series plotting with only input."""
        X, _, _ = sample_data
        fig = KerasFactoryPlotter.plot_timeseries(X, y_true=None, y_pred=None)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_plot_timeseries_comparison(self, sample_data):
        """Test forecast comparison plotting."""
        _, y_true, y_pred = sample_data
        fig = KerasFactoryPlotter.plot_timeseries_comparison(
            y_true,
            y_pred,
            sample_idx=0,
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # True and predicted

    def test_plot_timeseries_comparison_2d(self):
        """Test forecast comparison with 2D arrays."""
        y_true = np.random.randn(24).astype(np.float32)
        y_pred = y_true + 0.1 * np.random.randn(24).astype(np.float32)
        fig = KerasFactoryPlotter.plot_timeseries_comparison(y_true, y_pred)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2

    def test_plot_decomposition(self):
        """Test decomposition plotting."""
        n_samples = 100
        original = np.random.randn(n_samples).astype(np.float32)
        trend = np.linspace(-1, 1, n_samples).astype(np.float32)
        seasonal = np.sin(np.linspace(0, 4 * np.pi, n_samples)).astype(np.float32)
        residual = original - trend - seasonal

        fig = KerasFactoryPlotter.plot_decomposition(
            original,
            trend,
            seasonal,
            residual,
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 4

    def test_plot_forecasting_metrics(self, sample_data):
        """Test forecasting metrics plotting."""
        _, y_true, y_pred = sample_data
        # Flatten for metrics calculation
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()

        fig = KerasFactoryPlotter.plot_forecasting_metrics(y_true_flat, y_pred_flat)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_plot_forecast_horizon_analysis(self, sample_data):
        """Test forecast horizon analysis."""
        _, y_true, y_pred = sample_data
        fig = KerasFactoryPlotter.plot_forecast_horizon_analysis(y_true, y_pred)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_plot_multiple_features_forecast(self, sample_data):
        """Test multi-feature forecast plotting."""
        X, y_true, y_pred = sample_data
        fig = KerasFactoryPlotter.plot_multiple_features_forecast(
            X,
            y_true,
            y_pred,
            sample_idx=0,
            n_features_to_plot=3,
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0


class TestTrainingVisualization:
    """Test training history visualization."""

    def test_plot_training_history(self):
        """Test training history plotting."""
        history = {
            "loss": [0.5, 0.4, 0.3, 0.2],
            "val_loss": [0.6, 0.5, 0.4, 0.3],
            "mae": [0.3, 0.25, 0.2, 0.15],
        }

        fig = KerasFactoryPlotter.plot_training_history(
            history,
            metrics=["loss", "mae"],
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert "Training" in fig.layout.title.text or "loss" in str(fig.data[0])


class TestClassificationMetrics:
    """Test classification metrics visualization."""

    def test_plot_confusion_matrix(self):
        """Test confusion matrix plotting."""
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 1, 0, 2, 2])

        fig = KerasFactoryPlotter.plot_confusion_matrix(y_true, y_pred)

        assert isinstance(fig, go.Figure)
        assert "Confusion Matrix" in fig.layout.title.text

    def test_plot_roc_curve(self):
        """Test ROC curve plotting."""
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        y_scores = np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.3, 0.85, 0.15])

        fig = KerasFactoryPlotter.plot_roc_curve(y_true, y_scores)

        assert isinstance(fig, go.Figure)
        assert "ROC" in fig.layout.title.text

    def test_plot_precision_recall_curve(self):
        """Test precision-recall curve plotting."""
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        y_scores = np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.3, 0.85, 0.15])

        fig = KerasFactoryPlotter.plot_precision_recall_curve(y_true, y_scores)

        assert isinstance(fig, go.Figure)
        assert "Precision" in fig.layout.title.text or "Recall" in fig.layout.title.text


class TestAnomalyDetection:
    """Test anomaly detection visualization."""

    def test_plot_anomaly_scores(self):
        """Test anomaly scores plotting."""
        scores = np.random.randn(100).astype(np.float32)
        labels = np.random.randint(0, 2, 100)
        threshold = 2.0

        fig = KerasFactoryPlotter.plot_anomaly_scores(scores, labels, threshold)

        assert isinstance(fig, go.Figure)
        assert "Anomaly" in fig.layout.title.text


class TestPerformanceMetrics:
    """Test performance metrics visualization."""

    def test_plot_performance_metrics(self):
        """Test performance metrics bar chart."""
        metrics = {
            "Accuracy": 0.95,
            "Precision": 0.92,
            "Recall": 0.88,
            "F1": 0.90,
        }

        fig = KerasFactoryPlotter.plot_performance_metrics(metrics)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_plot_timeseries_single_sample(self):
        """Test plotting with single sample."""
        X = np.random.randn(1, 96, 7).astype(np.float32)
        y_true = np.random.randn(1, 12, 7).astype(np.float32)
        y_pred = np.random.randn(1, 12, 7).astype(np.float32)

        fig = KerasFactoryPlotter.plot_timeseries(
            X,
            y_true,
            y_pred,
            n_samples_to_plot=1,
        )

        assert isinstance(fig, go.Figure)

    def test_plot_decomposition_with_none_values(self):
        """Test decomposition with various input shapes."""
        original = np.array([1.0, 2.0, 3.0, 4.0, 5.0]).astype(np.float32)
        trend = np.array([1.0, 1.5, 2.0, 2.5, 3.0]).astype(np.float32)
        seasonal = np.array([0.0, 0.5, 1.0, 1.5, 2.0]).astype(np.float32)
        residual = original - trend - seasonal

        fig = KerasFactoryPlotter.plot_decomposition(
            original,
            trend,
            seasonal,
            residual,
        )

        assert isinstance(fig, go.Figure)

    def test_plot_metrics_empty_data(self):
        """Test metrics with minimal data."""
        y_true = np.array([1.0, 2.0, 3.0]).astype(np.float32)
        y_pred = np.array([1.1, 2.1, 2.9]).astype(np.float32)

        fig = KerasFactoryPlotter.plot_forecasting_metrics(y_true, y_pred)

        assert isinstance(fig, go.Figure)
