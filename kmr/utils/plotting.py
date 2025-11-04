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
            history: Keras training history object or dict with history data
            metrics: List of metrics to plot (default: ['loss', 'accuracy'])
            title: Plot title
            height: Plot height

        Returns:
            Plotly figure
        """
        if metrics is None:
            metrics = ["loss", "accuracy"]

        # Handle both History objects and dicts
        if isinstance(history, dict):
            hist_dict = history
        else:
            hist_dict = history.history

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
            if metric in hist_dict:
                row = (i // cols) + 1
                col = (i % cols) + 1

                # Training metric
                fig.add_trace(
                    go.Scatter(
                        x=list(range(1, len(hist_dict[metric]) + 1)),
                        y=hist_dict[metric],
                        mode="lines",
                        name=f"Training {metric.title()}",
                        line=dict(color=colors[0]),
                    ),
                    row=row,
                    col=col,
                )

                # Validation metric
                val_metric = f"val_{metric}"
                if val_metric in hist_dict:
                    fig.add_trace(
                        go.Scatter(
                            x=list(range(1, len(hist_dict[val_metric]) + 1)),
                            y=hist_dict[val_metric],
                            mode="lines",
                            name=f"Validation {metric.title()}",
                            line=dict(color=colors[1]),
                        ),
                        row=row,
                        col=col,
                    )

        fig.update_layout(title_text=title, height=height, showlegend=True)
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
    def plot_roc_curve(
        y_true: np.ndarray,
        y_scores: np.ndarray,
        title: str = "ROC Curve",
        height: int = 400,
    ) -> go.Figure:
        """Create ROC (Receiver Operating Characteristic) curve.

        Args:
            y_true: True labels (binary: 0 or 1)
            y_scores: Prediction scores or probabilities
            title: Plot title
            height: Plot height

        Returns:
            Plotly figure
        """
        # Calculate ROC curve for different thresholds
        thresholds = np.linspace(y_scores.max(), y_scores.min(), 100)
        tpr_list = []
        fpr_list = []

        for thresh in thresholds:
            y_pred = (y_scores > thresh).astype(int)

            # Calculate true positive rate and false positive rate
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            tn = np.sum((y_pred == 0) & (y_true == 0))

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

            tpr_list.append(tpr)
            fpr_list.append(fpr)

        # Calculate AUC (Area Under the Curve) using trapezoidal rule
        fpr_array = np.array(fpr_list)
        tpr_array = np.array(tpr_list)
        auc = np.trapz(tpr_array, fpr_array)

        fig = go.Figure()

        # Add ROC curve
        fig.add_trace(
            go.Scatter(
                x=fpr_list,
                y=tpr_list,
                mode="lines",
                name=f"ROC Curve (AUC = {auc:.3f})",
                line=dict(color="blue", width=3),
            ),
        )

        # Add diagonal reference line (random classifier)
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random Classifier",
                line=dict(color="red", dash="dash", width=2),
            ),
        )

        fig.update_layout(
            title=title,
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=height,
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
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
            x_labels = list(context_values)

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

    @staticmethod
    def plot_timeseries(
        X: np.ndarray,
        y_true: np.ndarray = None,
        y_pred: np.ndarray = None,
        n_samples_to_plot: int = 5,
        feature_idx: int = 0,
        title: str = "Time Series Forecast",
        height: int = 500,
    ) -> go.Figure:
        """Plot time series data with optional predictions.

        Args:
            X: Input sequences of shape (n_samples, seq_len, n_features).
            y_true: True target sequences of shape (n_samples, pred_len, n_features).
            y_pred: Predicted sequences of shape (n_samples, pred_len, n_features).
            n_samples_to_plot: Number of samples to visualize.
            feature_idx: Which feature to plot.
            title: Plot title.
            height: Plot height.

        Returns:
            Plotly figure.
        """
        fig = make_subplots(
            rows=n_samples_to_plot,
            cols=1,
            subplot_titles=[f"Sample {i+1}" for i in range(n_samples_to_plot)],
            vertical_spacing=0.05,
        )

        seq_len = X.shape[1]

        for sample_idx in range(min(n_samples_to_plot, len(X))):
            row = sample_idx + 1

            # Plot input sequence
            x_vals = list(range(seq_len))
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=X[sample_idx, :, feature_idx],
                    mode="lines",
                    name="Input",
                    line=dict(color="blue", width=2),
                ),
                row=row,
                col=1,
            )

            # Plot true target
            if y_true is not None:
                pred_len = y_true.shape[1]
                y_vals = list(range(seq_len, seq_len + pred_len))
                fig.add_trace(
                    go.Scatter(
                        x=y_vals,
                        y=y_true[sample_idx, :, feature_idx],
                        mode="lines",
                        name="True",
                        line=dict(color="green", width=2),
                    ),
                    row=row,
                    col=1,
                )

            # Plot predictions
            if y_pred is not None:
                pred_len = y_pred.shape[1]
                y_vals = list(range(seq_len, seq_len + pred_len))
                fig.add_trace(
                    go.Scatter(
                        x=y_vals,
                        y=y_pred[sample_idx, :, feature_idx],
                        mode="lines",
                        name="Predicted",
                        line=dict(color="red", width=2, dash="dash"),
                    ),
                    row=row,
                    col=1,
                )

        fig.update_layout(title=title, height=height, showlegend=True)
        fig.update_xaxes(title_text="Time Steps", row=n_samples_to_plot, col=1)
        fig.update_yaxes(
            title_text="Value",
            row=int((n_samples_to_plot + 1) / 2),
            col=1,
        )

        return fig

    @staticmethod
    def plot_timeseries_comparison(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_idx: int = 0,
        title: str = "Forecast Comparison",
        height: int = 400,
    ) -> go.Figure:
        """Plot single time series forecast comparison.

        Args:
            y_true: True sequences of shape (n_samples, pred_len, n_features) or (pred_len, n_features).
            y_pred: Predicted sequences of shape (n_samples, pred_len, n_features) or (pred_len, n_features).
            sample_idx: Index of sample to plot (if 3D arrays).
            title: Plot title.
            height: Plot height.

        Returns:
            Plotly figure.
        """
        if len(y_true.shape) == 3:
            y_true = y_true[sample_idx]
        if len(y_pred.shape) == 3:
            y_pred = y_pred[sample_idx]

        fig = go.Figure()

        x_vals = list(range(len(y_true)))

        # For multivariate, plot first feature
        if len(y_true.shape) > 1:
            y_true_vals = y_true[:, 0]
            y_pred_vals = y_pred[:, 0]
        else:
            y_true_vals = y_true
            y_pred_vals = y_pred

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_true_vals,
                mode="lines+markers",
                name="True",
                line=dict(color="green", width=2),
            ),
        )

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_pred_vals,
                mode="lines+markers",
                name="Predicted",
                line=dict(color="red", width=2, dash="dash"),
            ),
        )

        fig.update_layout(
            title=title,
            xaxis_title="Time Steps",
            yaxis_title="Value",
            height=height,
        )

        return fig

    @staticmethod
    def plot_decomposition(
        original: np.ndarray,
        trend: np.ndarray = None,
        seasonal: np.ndarray = None,
        residual: np.ndarray = None,
        title: str = "Time Series Decomposition",
        height: int = 600,
    ) -> go.Figure:
        """Plot time series decomposition into components.

        Args:
            original: Original time series.
            trend: Trend component.
            seasonal: Seasonal component.
            residual: Residual component.
            title: Plot title.
            height: Plot height.

        Returns:
            Plotly figure.
        """
        components = {"Original": original}
        if trend is not None:
            components["Trend"] = trend
        if seasonal is not None:
            components["Seasonal"] = seasonal
        if residual is not None:
            components["Residual"] = residual

        n_components = len(components)
        fig = make_subplots(
            rows=n_components,
            cols=1,
            subplot_titles=list(components.keys()),
            vertical_spacing=0.08,
        )

        x_vals = list(range(len(original)))

        for i, (name, component) in enumerate(components.items()):
            row = i + 1
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=component,
                    mode="lines",
                    name=name,
                    line=dict(color=["blue", "green", "orange", "red"][i]),
                ),
                row=row,
                col=1,
            )

        fig.update_layout(title=title, height=height, showlegend=False)
        fig.update_yaxes(title_text="Value", row=int((n_components + 1) / 2), col=1)
        fig.update_xaxes(title_text="Time Steps", row=n_components, col=1)

        return fig

    @staticmethod
    def plot_forecasting_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Forecasting Metrics",
        height: int = 400,
    ) -> go.Figure:
        """Calculate and plot forecasting error metrics.

        Args:
            y_true: True values.
            y_pred: Predicted values.
            title: Plot title.
            height: Plot height.

        Returns:
            Plotly figure with metrics.
        """
        # Calculate errors
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100

        metrics_dict = {"MAE": mae, "RMSE": rmse, "MAPE (%)": mape}

        return KMRPlotter.plot_performance_metrics(metrics_dict, title, height)

    @staticmethod
    def plot_forecast_horizon_analysis(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Forecast Error by Horizon",
        height: int = 400,
    ) -> go.Figure:
        """Analyze forecast error across different forecast horizons.

        Args:
            y_true: True sequences of shape (n_samples, pred_len) or (n_samples, pred_len, n_features).
            y_pred: Predicted sequences of same shape.
            title: Plot title.
            height: Plot height.

        Returns:
            Plotly figure.
        """
        # Handle multivariate by taking first feature
        if len(y_true.shape) > 2:
            y_true = y_true[:, :, 0]
        if len(y_pred.shape) > 2:
            y_pred = y_pred[:, :, 0]

        pred_len = y_true.shape[1]
        mae_by_horizon = []

        for t in range(pred_len):
            mae = np.mean(np.abs(y_true[:, t] - y_pred[:, t]))
            mae_by_horizon.append(mae)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=list(range(1, pred_len + 1)),
                y=mae_by_horizon,
                mode="lines+markers",
                name="MAE",
                line=dict(color="blue", width=2),
            ),
        )

        fig.update_layout(
            title=title,
            xaxis_title="Forecast Horizon (steps ahead)",
            yaxis_title="Mean Absolute Error",
            height=height,
        )

        return fig

    @staticmethod
    def plot_multiple_features_forecast(
        X: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_idx: int = 0,
        n_features_to_plot: int = None,
        title: str = "Multi-Feature Forecast",
        height: int = 500,
    ) -> go.Figure:
        """Plot forecasts for multiple features side-by-side.

        Args:
            X: Input sequences.
            y_true: True target sequences.
            y_pred: Predicted sequences.
            sample_idx: Which sample to plot.
            n_features_to_plot: Number of features to plot (default: all).
            title: Plot title.
            height: Plot height.

        Returns:
            Plotly figure.
        """
        n_features = X.shape[2]
        if n_features_to_plot is None:
            n_features_to_plot = min(n_features, 4)

        seq_len = X.shape[1]
        pred_len = y_true.shape[1]

        fig = make_subplots(
            rows=1,
            cols=n_features_to_plot,
            subplot_titles=[f"Feature {i}" for i in range(n_features_to_plot)],
        )

        for feat_idx in range(n_features_to_plot):
            col = feat_idx + 1

            # Input
            x_vals = list(range(seq_len))
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=X[sample_idx, :, feat_idx],
                    mode="lines",
                    name="Input",
                    line=dict(color="blue"),
                    showlegend=(feat_idx == 0),
                ),
                row=1,
                col=col,
            )

            # True target
            y_vals = list(range(seq_len, seq_len + pred_len))
            fig.add_trace(
                go.Scatter(
                    x=y_vals,
                    y=y_true[sample_idx, :, feat_idx],
                    mode="lines",
                    name="True",
                    line=dict(color="green"),
                    showlegend=(feat_idx == 0),
                ),
                row=1,
                col=col,
            )

            # Predicted
            fig.add_trace(
                go.Scatter(
                    x=y_vals,
                    y=y_pred[sample_idx, :, feat_idx],
                    mode="lines",
                    name="Predicted",
                    line=dict(color="red", dash="dash"),
                    showlegend=(feat_idx == 0),
                ),
                row=1,
                col=col,
            )

        fig.update_layout(title=title, height=height, showlegend=True)

        return fig
