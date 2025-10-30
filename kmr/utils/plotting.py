"""Plotting utilities for KMR models and metrics visualization."""

from typing import Any
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class KMRPlotter:
    """Utility class for creating consistent visualizations across KMR notebooks."""

    @staticmethod
    def plot_training_history(
        history: Any,
        metrics: list[str] = None,
        title: str = "Training Progress",
        height: int = 400,
    ) -> go.Figure:
        """Create training history plots.

        Args:
            history: Keras training history object
            metrics: List of metrics to plot (default: ['loss', 'accuracy'])
            title: Plot title
            height: Plot height

        Returns:
            Plotly figure
        """
        if metrics is None:
            metrics = ["loss", "accuracy"]

        # Determine subplot layout
        n_metrics = len(metrics)
        if n_metrics <= 2:
            rows, cols = 1, n_metrics
        elif n_metrics <= 4:
            rows, cols = 2, 2
        else:
            rows, cols = 3, 2

        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[
                f"Training and Validation {metric.title()}" for metric in metrics
            ],
        )

        colors = ["blue", "red", "green", "orange", "purple", "brown"]

        for i, metric in enumerate(metrics):
            if metric in history.history:
                row = (i // cols) + 1
                col = (i % cols) + 1

                # Training metric
                fig.add_trace(
                    go.Scatter(
                        x=list(range(1, len(history.history[metric]) + 1)),
                        y=history.history[metric],
                        mode="lines",
                        name=f"Training {metric.title()}",
                        line=dict(color=colors[0]),
                    ),
                    row=row,
                    col=col,
                )

                # Validation metric
                val_metric = f"val_{metric}"
                if val_metric in history.history:
                    fig.add_trace(
                        go.Scatter(
                            x=list(range(1, len(history.history[val_metric]) + 1)),
                            y=history.history[val_metric],
                            mode="lines",
                            name=f"Validation {metric.title()}",
                            line=dict(color=colors[1]),
                        ),
                        row=row,
                        col=col,
                    )

        fig.update_layout(height=height, title_text=title, showlegend=True)

        fig.update_xaxes(title_text="Epoch")
        fig.update_yaxes(title_text="Value")

        return fig

    @staticmethod
    def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Confusion Matrix",
        height: int = 400,
    ) -> go.Figure:
        """Create confusion matrix heatmap.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            title: Plot title
            height: Plot height

        Returns:
            Plotly figure
        """
        from collections import Counter

        # Create confusion matrix
        cm = Counter(zip(y_true, y_pred, strict=False))
        n_classes = len(np.unique(y_true))

        if n_classes == 2:
            cm_matrix = np.array(
                [
                    [cm.get((0, 0), 0), cm.get((0, 1), 0)],
                    [cm.get((1, 0), 0), cm.get((1, 1), 0)],
                ],
            )
            x_labels = ["Predicted 0", "Predicted 1"]
            y_labels = ["Actual 0", "Actual 1"]
        else:
            # Multi-class confusion matrix
            cm_matrix = np.zeros((n_classes, n_classes))
            for (true_label, pred_label), count in cm.items():
                cm_matrix[true_label, pred_label] = count
            x_labels = [f"Predicted {i}" for i in range(n_classes)]
            y_labels = [f"Actual {i}" for i in range(n_classes)]

        fig = go.Figure()

        fig.add_trace(
            go.Heatmap(
                z=cm_matrix,
                x=x_labels,
                y=y_labels,
                text=cm_matrix.astype(int),
                texttemplate="%{text}",
                textfont={"size": 16},
                colorscale="Blues",
            ),
        )

        fig.update_layout(title=title, height=height)

        return fig

    @staticmethod
    def plot_predictions_vs_actual(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Predictions vs Actual Values",
        height: int = 500,
    ) -> go.Figure:
        """Create predictions vs actual values scatter plot.

        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
            height: Plot height

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=y_true,
                y=y_pred,
                mode="markers",
                name="Predictions",
                marker=dict(color="blue", opacity=0.6),
            ),
        )

        # Add perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                name="Perfect Prediction",
                line=dict(color="red", dash="dash"),
            ),
        )

        fig.update_layout(
            title=title,
            xaxis_title="Actual Values",
            yaxis_title="Predicted Values",
            height=height,
        )

        return fig

    @staticmethod
    def plot_anomaly_scores(
        scores: np.ndarray,
        labels: np.ndarray,
        threshold: float = None,
        title: str = "Anomaly Score Distribution",
        height: int = 400,
    ) -> go.Figure:
        """Create anomaly score distribution plot.

        Args:
            scores: Anomaly scores
            labels: True labels (0=normal, 1=anomaly)
            threshold: Anomaly threshold
            title: Plot title
            height: Plot height

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        # Separate scores by label
        normal_scores = scores[labels == 0]
        anomaly_scores = scores[labels == 1]

        # Plot histograms
        fig.add_trace(
            go.Histogram(x=normal_scores, name="Normal", opacity=0.7, nbinsx=30),
        )

        fig.add_trace(
            go.Histogram(x=anomaly_scores, name="Anomaly", opacity=0.7, nbinsx=30),
        )

        # Add threshold line if provided
        if threshold is not None:
            fig.add_vline(
                x=threshold,
                line_dash="dash",
                line_color="green",
                annotation_text="Threshold",
            )

        fig.update_layout(
            title=title,
            xaxis_title="Anomaly Score",
            yaxis_title="Frequency",
            height=height,
        )

        return fig

    @staticmethod
    def plot_performance_metrics(
        metrics_dict: dict[str, float],
        title: str = "Performance Metrics",
        height: int = 400,
    ) -> go.Figure:
        """Create performance metrics bar chart.

        Args:
            metrics_dict: Dictionary of metric names and values
            title: Plot title
            height: Plot height

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        metric_names = list(metrics_dict.keys())
        metric_values = list(metrics_dict.values())

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

        fig.add_trace(
            go.Bar(
                x=metric_names,
                y=metric_values,
                marker_color=colors[: len(metric_names)],
            ),
        )

        fig.update_layout(
            title=title,
            xaxis_title="Metrics",
            yaxis_title="Score",
            height=height,
        )

        return fig

    @staticmethod
    def plot_precision_recall_curve(
        y_true: np.ndarray,
        y_scores: np.ndarray,
        title: str = "Precision-Recall Curve",
        height: int = 400,
    ) -> go.Figure:
        """Create precision-recall curve.

        Args:
            y_true: True labels
            y_scores: Prediction scores
            title: Plot title
            height: Plot height

        Returns:
            Plotly figure
        """
        # Calculate precision and recall for different thresholds
        thresholds = np.linspace(y_scores.min(), y_scores.max(), 100)
        precisions = []
        recalls = []

        for thresh in thresholds:
            y_pred = (y_scores > thresh).astype(int)
            if np.sum(y_pred) > 0:
                # Calculate precision and recall manually
                tp = np.sum((y_pred == 1) & (y_true == 1))
                fp = np.sum((y_pred == 1) & (y_true == 0))
                fn = np.sum((y_pred == 0) & (y_true == 1))

                prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0

                precisions.append(prec)
                recalls.append(rec)
            else:
                precisions.append(0)
                recalls.append(0)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=recalls,
                y=precisions,
                mode="lines",
                name="PR Curve",
                line=dict(width=3),
            ),
        )

        fig.update_layout(
            title=title,
            xaxis_title="Recall",
            yaxis_title="Precision",
            height=height,
        )

        return fig

    @staticmethod
    def plot_context_dependency(
        context_values: np.ndarray,
        accuracies: list[float],
        title: str = "Model Performance by Context",
        height: int = 400,
    ) -> go.Figure:
        """Create context dependency plot.

        Args:
            context_values: Context values or bin labels
            accuracies: Accuracies for each context bin
            title: Plot title
            height: Plot height

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        if isinstance(context_values[0], (int, float)):
            x_labels = [f"Bin {i+1}" for i in range(len(context_values))]
        else:
            x_labels = context_values

        fig.add_trace(go.Bar(x=x_labels, y=accuracies, marker_color="lightblue"))

        fig.update_layout(
            title=title,
            xaxis_title="Context Bins",
            yaxis_title="Accuracy",
            height=height,
        )

        return fig

    @staticmethod
    def create_comprehensive_plot(plot_type: str, **kwargs) -> go.Figure:
        """Create comprehensive plots with multiple subplots.

        Args:
            plot_type: Type of comprehensive plot ('anomaly_detection', 'classification', 'regression')
            **kwargs: Additional arguments for the specific plot type

        Returns:
            Plotly figure
        """
        if plot_type == "anomaly_detection":
            return KMRPlotter._create_anomaly_detection_plot(**kwargs)
        elif plot_type == "classification":
            return KMRPlotter._create_classification_plot(**kwargs)
        elif plot_type == "regression":
            return KMRPlotter._create_regression_plot(**kwargs)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")

    @staticmethod
    def _create_anomaly_detection_plot(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        scores: np.ndarray,
        threshold: float,
        title: str = "Anomaly Detection Results",
    ) -> go.Figure:
        """Create comprehensive anomaly detection plot."""
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Anomaly Score Distribution",
                "Confusion Matrix",
                "Precision-Recall Curve",
                "Performance Metrics",
            ),
            specs=[
                [{"type": "histogram"}, {"type": "heatmap"}],
                [{"type": "scatter"}, {"type": "bar"}],
            ],
        )

        # Plot 1: Anomaly scores distribution
        normal_scores = scores[y_true == 0]
        anomaly_scores = scores[y_true == 1]

        fig.add_trace(
            go.Histogram(x=normal_scores, name="Normal", opacity=0.7, nbinsx=30),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Histogram(x=anomaly_scores, name="Anomaly", opacity=0.7, nbinsx=30),
            row=1,
            col=1,
        )
        fig.add_vline(
            x=threshold,
            line_dash="dash",
            line_color="green",
            annotation_text="Threshold",
            row=1,
            col=1,
        )

        # Plot 2: Confusion Matrix
        from collections import Counter

        cm = Counter(zip(y_true, y_pred, strict=False))
        cm_matrix = np.array(
            [
                [cm.get((0, 0), 0), cm.get((0, 1), 0)],
                [cm.get((1, 0), 0), cm.get((1, 1), 0)],
            ],
        )

        fig.add_trace(
            go.Heatmap(
                z=cm_matrix,
                x=["Predicted Normal", "Predicted Anomaly"],
                y=["Actual Normal", "Actual Anomaly"],
                text=cm_matrix,
                texttemplate="%{text}",
                textfont={"size": 16},
                colorscale="Blues",
            ),
            row=1,
            col=2,
        )

        # Plot 3: Precision-Recall Curve
        pr_curve = KMRPlotter.plot_precision_recall_curve(y_true, scores)
        fig.add_trace(pr_curve.data[0], row=2, col=1)

        # Plot 4: Performance metrics
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
        )

        metrics_dict = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1-Score": f1_score(y_true, y_pred),
        }

        metrics_plot = KMRPlotter.plot_performance_metrics(metrics_dict)
        fig.add_trace(metrics_plot.data[0], row=2, col=2)

        fig.update_layout(height=800, title_text=title, showlegend=True)

        return fig

    @staticmethod
    def _create_classification_plot(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: np.ndarray = None,
        title: str = "Classification Results",
    ) -> go.Figure:
        """Create comprehensive classification plot."""
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Confusion Matrix",
                "Performance Metrics",
                "Score Distribution",
                "Precision-Recall Curve",
            ),
            specs=[
                [{"type": "heatmap"}, {"type": "bar"}],
                [{"type": "histogram"}, {"type": "scatter"}],
            ],
        )

        # Plot 1: Confusion Matrix
        cm_plot = KMRPlotter.plot_confusion_matrix(y_true, y_pred)
        fig.add_trace(cm_plot.data[0], row=1, col=1)

        # Plot 2: Performance Metrics
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
        )

        metrics_dict = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1-Score": f1_score(y_true, y_pred),
        }

        metrics_plot = KMRPlotter.plot_performance_metrics(metrics_dict)
        fig.add_trace(metrics_plot.data[0], row=1, col=2)

        # Plot 3: Score Distribution (if scores provided)
        if y_scores is not None:
            fig.add_trace(
                go.Histogram(x=y_scores, name="Scores", nbinsx=30),
                row=2,
                col=1,
            )

        # Plot 4: Precision-Recall Curve (if scores provided)
        if y_scores is not None:
            pr_curve = KMRPlotter.plot_precision_recall_curve(y_true, y_scores)
            fig.add_trace(pr_curve.data[0], row=2, col=2)

        fig.update_layout(height=800, title_text=title, showlegend=True)

        return fig

    @staticmethod
    def _create_regression_plot(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Regression Results",
    ) -> go.Figure:
        """Create comprehensive regression plot."""
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Predictions vs Actual",
                "Residuals",
                "Performance Metrics",
                "Error Distribution",
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "histogram"}],
            ],
        )

        # Plot 1: Predictions vs Actual
        pred_plot = KMRPlotter.plot_predictions_vs_actual(y_true, y_pred)
        fig.add_trace(pred_plot.data[0], row=1, col=1)
        fig.add_trace(pred_plot.data[1], row=1, col=1)

        # Plot 2: Residuals
        residuals = y_true - y_pred
        fig.add_trace(
            go.Scatter(x=y_pred, y=residuals, mode="markers", name="Residuals"),
            row=1,
            col=2,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)

        # Plot 3: Performance Metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        metrics_dict = {
            "MAE": mean_absolute_error(y_true, y_pred),
            "MSE": mean_squared_error(y_true, y_pred),
            "R²": r2_score(y_true, y_pred),
        }

        metrics_plot = KMRPlotter.plot_performance_metrics(metrics_dict)
        fig.add_trace(metrics_plot.data[0], row=2, col=1)

        # Plot 4: Error Distribution
        fig.add_trace(
            go.Histogram(x=residuals, name="Residuals", nbinsx=30),
            row=2,
            col=2,
        )

        fig.update_layout(height=800, title_text=title, showlegend=True)

        return fig
