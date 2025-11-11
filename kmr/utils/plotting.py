"""Plotting utilities for KMR models and metrics visualization."""

from typing import Any
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _kmeans_clustering_numpy(
    data: np.ndarray,
    n_clusters: int,
    max_iter: int = 100,
) -> np.ndarray:
    """K-means clustering using pure numpy.

    Args:
        data: Array of shape (n_samples, n_features)
        n_clusters: Number of clusters
        max_iter: Maximum iterations

    Returns:
        Cluster labels array of shape (n_samples,)
    """
    np.random.seed(42)
    n_samples, n_features = data.shape

    # Initialize centroids randomly
    centroids = data[np.random.choice(n_samples, n_clusters, replace=False)]
    cluster_labels = np.zeros(n_samples, dtype=int)

    for _ in range(max_iter):
        # Assign to nearest centroid
        # Compute distances: (n_clusters, n_samples)
        distances = np.array(
            [np.linalg.norm(data - centroid, axis=1) for centroid in centroids],
        )
        new_labels = np.argmin(distances, axis=0)

        # Check convergence
        if np.array_equal(cluster_labels, new_labels):
            break
        cluster_labels = new_labels

        # Update centroids
        for k in range(n_clusters):
            mask = cluster_labels == k
            if mask.sum() > 0:
                centroids[k] = data[mask].mean(axis=0)

    return cluster_labels


def _agglomerative_clustering_numpy(
    distance_matrix: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    """Simple agglomerative clustering using pure numpy.

    Uses Ward-like linkage (minimizes within-cluster variance).

    Args:
        distance_matrix: Pairwise distance matrix of shape (n_samples, n_samples)
        n_clusters: Number of clusters

    Returns:
        Cluster labels array of shape (n_samples,)
    """
    n_samples = distance_matrix.shape[0]
    if n_clusters >= n_samples:
        return np.arange(n_samples)

    # Initialize: each sample is its own cluster
    clusters = [[i] for i in range(n_samples)]

    # Iteratively merge clusters
    while len(clusters) > n_clusters:
        # Find two closest clusters
        min_dist = np.inf
        merge_i, merge_j = 0, 1

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                # Compute average distance between clusters (simple linkage)
                dists = []
                for idx_i in clusters[i]:
                    for idx_j in clusters[j]:
                        dists.append(distance_matrix[idx_i, idx_j])
                avg_dist = np.mean(dists) if dists else np.inf

                if avg_dist < min_dist:
                    min_dist = avg_dist
                    merge_i, merge_j = i, j

        # Merge clusters
        clusters[merge_i].extend(clusters[merge_j])
        del clusters[merge_j]

    # Assign labels
    labels = np.zeros(n_samples, dtype=int)
    for cluster_id, cluster_indices in enumerate(clusters):
        for idx in cluster_indices:
            labels[idx] = cluster_id

    return labels


def _plot_simple_dendrogram(
    fig: go.Figure,
    user_ids: np.ndarray,
    cluster_labels: np.ndarray,
    row: int,
    col: int,
) -> None:
    """Plot a simple dendrogram-like visualization using cluster assignments.

    Args:
        fig: Plotly figure
        user_ids: User IDs
        cluster_labels: Cluster assignments
        row: Subplot row
        col: Subplot column
    """
    # Group users by cluster
    cluster_groups = {}
    for user_id, cluster_id in zip(user_ids, cluster_labels, strict=False):
        if cluster_id not in cluster_groups:
            cluster_groups[cluster_id] = []
        cluster_groups[cluster_id].append(user_id)

    # Create simple tree visualization
    y_positions = []
    x_positions = []
    text_labels = []

    for cluster_id in sorted(cluster_groups.keys()):
        users = sorted(cluster_groups[cluster_id])
        cluster_y = cluster_id
        for i, user_id in enumerate(users):
            y_positions.append(cluster_y + i * 0.1)
            x_positions.append(i)
            text_labels.append(f"U{user_id}")

    fig.add_trace(
        go.Scatter(
            x=x_positions,
            y=y_positions,
            mode="markers+text",
            text=text_labels,
            textposition="top center",
            marker=dict(size=8, color="black"),
            showlegend=False,
        ),
        row=row,
        col=col,
    )
    fig.update_xaxes(title_text="Cluster Groups", row=row, col=col)
    fig.update_yaxes(title_text="Cluster ID", row=row, col=col)


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

    @staticmethod
    def plot_recommendation_scores(
        scores: np.ndarray,
        top_k: int = 10,
        title: str = "Recommendation Scores",
        height: int = 400,
    ) -> go.Figure:
        """Plot recommendation scores for top-K items.

        Args:
            scores: Recommendation scores array of shape (n_items,) or (n_samples, n_items)
            top_k: Number of top items to highlight
            title: Plot title
            height: Plot height

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        if len(scores.shape) == 1:
            scores = scores.reshape(1, -1)

        # Get top-K indices for first sample
        top_k_indices = np.argsort(scores[0])[-top_k:][::-1]
        top_k_scores = scores[0][top_k_indices]

        # Plot all scores
        fig.add_trace(
            go.Scatter(
                x=list(range(len(scores[0]))),
                y=scores[0],
                mode="markers",
                name="All Items",
                marker=dict(color="lightblue", opacity=0.5),
            ),
        )

        # Highlight top-K
        fig.add_trace(
            go.Scatter(
                x=top_k_indices,
                y=top_k_scores,
                mode="markers",
                name=f"Top-{top_k}",
                marker=dict(color="red", size=10),
            ),
        )

        fig.update_layout(
            title=title,
            xaxis_title="Item Index",
            yaxis_title="Recommendation Score",
            height=height,
        )

        return fig

    @staticmethod
    def plot_geospatial_recommendations(
        user_lat: np.ndarray,
        user_lon: np.ndarray,
        item_lats: np.ndarray,
        item_lons: np.ndarray,
        recommended_indices: np.ndarray = None,
        title: str = "Geospatial Recommendations",
        height: int = 600,
    ) -> go.Figure:
        """Plot geospatial recommendations on a map.

        Args:
            user_lat: User latitudes
            user_lon: User longitudes
            item_lats: Item latitudes
            item_lons: Item longitudes
            recommended_indices: Indices of recommended items (optional)
            title: Plot title
            height: Plot height

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        # Plot all items
        fig.add_trace(
            go.Scatter(
                x=item_lons,
                y=item_lats,
                mode="markers",
                name="All Items",
                marker=dict(color="lightblue", size=5, opacity=0.5),
            ),
        )

        # Plot recommended items
        if recommended_indices is not None:
            if len(recommended_indices.shape) > 1:
                recommended_indices = recommended_indices[0]  # Take first sample
            rec_lats = item_lats[recommended_indices]
            rec_lons = item_lons[recommended_indices]
            fig.add_trace(
                go.Scatter(
                    x=rec_lons,
                    y=rec_lats,
                    mode="markers",
                    name="Recommended",
                    marker=dict(color="red", size=10),
                ),
            )

        # Plot user location
        if len(user_lat.shape) == 0:
            fig.add_trace(
                go.Scatter(
                    x=[user_lon],
                    y=[user_lat],
                    mode="markers",
                    name="User",
                    marker=dict(color="green", size=15, symbol="star"),
                ),
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=user_lon[:1],
                    y=user_lat[:1],
                    mode="markers",
                    name="User",
                    marker=dict(color="green", size=15, symbol="star"),
                ),
            )

        fig.update_layout(
            title=title,
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            height=height,
        )

        return fig

    @staticmethod
    def plot_similarity_matrix(
        similarity_matrix: np.ndarray,
        title: str = "Similarity Matrix",
        height: int = 500,
    ) -> go.Figure:
        """Plot similarity matrix as a heatmap.

        Args:
            similarity_matrix: Similarity matrix of shape (n_users, n_items), (n_items, n_items),
                              or (n_items,) for single user
            title: Plot title
            height: Plot height

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        # Handle different input shapes
        if len(similarity_matrix.shape) == 1:
            # 1D array: (n_items,) -> reshape to (1, n_items) for visualization
            similarity_matrix = similarity_matrix.reshape(1, -1)
        elif len(similarity_matrix.shape) == 2 and similarity_matrix.shape[0] == 1:
            # Already (1, n_items) - use as is
            pass
        # else: (n_users, n_items) or (n_items, n_items) - use as is

        fig.add_trace(
            go.Heatmap(
                z=similarity_matrix,
                colorscale="Viridis",
                colorbar=dict(title="Similarity"),
            ),
        )

        # Determine y-axis label based on shape
        if similarity_matrix.shape[0] == 1:
            yaxis_title = "User (single)"
        elif similarity_matrix.shape[0] == similarity_matrix.shape[1]:
            yaxis_title = "Items"
        else:
            yaxis_title = "Users"

        fig.update_layout(
            title=title,
            xaxis_title="Items",
            yaxis_title=yaxis_title,
            height=height,
        )

        return fig

    @staticmethod
    def plot_recommendation_metrics(
        metrics_dict: dict[str, float],
        title: str = "Recommendation Metrics",
        height: int = 400,
    ) -> go.Figure:
        """Plot recommendation system metrics.

        Args:
            metrics_dict: Dictionary of metric names and values
            title: Plot title
            height: Plot height

        Returns:
            Plotly figure
        """
        return KMRPlotter.plot_performance_metrics(metrics_dict, title, height)

    @staticmethod
    def plot_recommendation_diversity(
        recommendations: np.ndarray,
        user_ids: np.ndarray | None = None,
        title: str = "Recommendation Diversity Across Users",
        height: int = 500,
    ) -> go.Figure:
        """Plot recommendation diversity across users.

        Visualizes which items are recommended to different users, helping to detect
        if the model is recommending the same items to all users (model collapse).

        Args:
            recommendations: Array of shape (n_users, top_k) with recommended item indices
            user_ids: Optional array of user IDs for labeling (default: [0, 1, 2, ...])
            title: Plot title
            height: Plot height

        Returns:
            Plotly figure
        """
        recommendations = np.asarray(recommendations)
        n_users, top_k = recommendations.shape

        if user_ids is None:
            user_ids = np.arange(n_users)

        fig = go.Figure()

        # Create heatmap showing which items are recommended to which users
        # Only include items that are actually recommended (more efficient)
        all_items = set()
        for rec in recommendations:
            all_items.update(rec)
        all_items = sorted(all_items)
        n_items = len(all_items)

        # Create binary matrix: (n_users, n_items)
        diversity_matrix = np.zeros((n_users, n_items), dtype=float)
        item_to_idx = {item: idx for idx, item in enumerate(all_items)}

        for u_idx, _ in enumerate(user_ids):
            for item_idx in recommendations[u_idx]:
                if item_idx in item_to_idx:
                    diversity_matrix[u_idx, item_to_idx[item_idx]] = 1.0

        fig.add_trace(
            go.Heatmap(
                z=diversity_matrix,
                x=[f"Item {i}" for i in all_items],
                y=[f"User {uid}" for uid in user_ids],
                colorscale="Viridis",
                colorbar=dict(title="Recommended"),
                showscale=True,
            ),
        )

        fig.update_layout(
            title=title,
            xaxis_title="Items",
            yaxis_title="Users",
            height=height,
        )

        # Calculate diversity metrics
        unique_items_per_user = [len(np.unique(rec)) for rec in recommendations]
        shared_items = len(
            set(recommendations[0]).intersection(
                *[set(rec) for rec in recommendations[1:]],
            ),
        )
        diversity_ratio = 1.0 - (shared_items / top_k)

        # Add annotation
        fig.add_annotation(
            text=f"Shared items across all users: {shared_items}/{top_k}<br>"
            f"Diversity ratio: {diversity_ratio:.2%}<br>"
            f"Avg unique items per user: {np.mean(unique_items_per_user):.1f}",
            xref="paper",
            yref="paper",
            x=1.02,
            y=0.5,
            showarrow=False,
            align="left",
        )

        return fig

    @staticmethod
    def plot_user_clusters(
        similarity_matrices: np.ndarray,
        user_ids: np.ndarray | None = None,
        n_clusters: int | None = None,
        method: str = "hierarchical",
        title: str = "User Clusters Based on Similarity Patterns",
        height: int = 600,
    ) -> tuple[go.Figure, np.ndarray]:
        """Cluster and visualize users based on their similarity patterns.

        Takes user-item similarity matrices and clusters users who have similar
        recommendation patterns. Useful for understanding user segments.

        Uses pure numpy for clustering (no sklearn dependency). For hierarchical
        clustering with dendrogram, sklearn/scipy are optional but recommended.

        Args:
            similarity_matrices: Array of shape (n_users, n_items) with user-item similarities
            user_ids: Optional array of user IDs for labeling (default: [0, 1, 2, ...])
            n_clusters: Number of clusters (auto-determined if None)
            method: Clustering method - 'hierarchical' or 'kmeans' (default: 'hierarchical')
            title: Plot title
            height: Plot height

        Returns:
            Tuple of (figure, cluster_labels) where cluster_labels is (n_users,) array
        """
        similarity_matrices = np.asarray(similarity_matrices)
        n_users, n_items = similarity_matrices.shape

        if user_ids is None:
            user_ids = np.arange(n_users)

        # Compute user-user similarity (cosine similarity between similarity vectors)
        # Normalize each user's similarity vector
        norms = np.linalg.norm(similarity_matrices, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        normalized = similarity_matrices / norms

        # Compute pairwise cosine similarity
        user_similarity = np.dot(normalized, normalized.T)  # (n_users, n_users)

        # Convert to distance matrix for clustering
        user_distance = 1 - user_similarity
        np.fill_diagonal(user_distance, 0)  # Ensure diagonal is 0

        # Determine number of clusters if not specified
        if n_clusters is None:
            # Simple heuristic: sqrt of number of users (works well in practice)
            n_clusters = max(2, int(np.sqrt(n_users)))

        # Perform clustering using pure numpy
        if method == "hierarchical":
            # Simple agglomerative clustering using numpy
            cluster_labels = _agglomerative_clustering_numpy(user_distance, n_clusters)
        else:
            # K-means using numpy
            cluster_labels = _kmeans_clustering_numpy(normalized, n_clusters)

        # Create subplots: 2D projection, similarity matrix, and cluster sizes
        rows, cols = 1, 3
        subplot_titles = (
            "User Clusters (2D Projection)",
            "User-User Similarity Matrix",
            "Cluster Sizes",
        )

        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=subplot_titles,
            specs=[[{"type": "scatter"}, {"type": "heatmap"}, {"type": "bar"}]],
            vertical_spacing=0.15,
            horizontal_spacing=0.08,
            column_widths=[
                0.4,
                0.4,
                0.2,
            ],  # Better proportions: 40% for 2D, 40% for heatmap, 20% for bar chart
        )

        # 2. 2D projection of users colored by cluster (using SVD/PCA via numpy)
        # Simple 2D projection using first two principal components via SVD
        centered = normalized - normalized.mean(axis=0)
        U, s, Vt = np.linalg.svd(centered, full_matrices=False)
        user_projection = U[:, :2] * s[:2]
        explained_var = (s[:2] ** 2) / (s**2).sum()

        # Color by cluster
        colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]

        for cluster_id in range(n_clusters):
            mask = cluster_labels == cluster_id
            fig.add_trace(
                go.Scatter(
                    x=user_projection[mask, 0],
                    y=user_projection[mask, 1],
                    mode="markers+text",
                    marker=dict(
                        size=12,
                        color=colors[cluster_id % len(colors)],
                        opacity=0.8,
                        line=dict(width=1.5, color="black"),
                    ),
                    text=[f"U{uid}" for uid in user_ids[mask]],
                    textposition="top center",
                    name=f"Cluster {cluster_id}",
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

        fig.update_xaxes(
            title_text=f"PC1 ({explained_var[0]:.1%} variance)",
            title_font=dict(size=12),
            row=1,
            col=1,
            showgrid=True,
            gridwidth=1,
            gridcolor="lightgray",
        )
        fig.update_yaxes(
            title_text=f"PC2 ({explained_var[1]:.1%} variance)",
            title_font=dict(size=12),
            row=1,
            col=1,
            showgrid=True,
            gridwidth=1,
            gridcolor="lightgray",
        )

        # 3. User-user similarity heatmap (ordered by cluster)
        cluster_order = np.argsort(cluster_labels)
        ordered_similarity = user_similarity[np.ix_(cluster_order, cluster_order)]

        fig.add_trace(
            go.Heatmap(
                z=ordered_similarity,
                x=[f"U{uid}" for uid in user_ids[cluster_order]],
                y=[f"U{uid}" for uid in user_ids[cluster_order]],
                colorscale="Viridis",
                colorbar=dict(
                    title=dict(text="Similarity", font=dict(size=11)),
                    x=1.02,
                    len=0.6,
                    thickness=15,
                ),
                showscale=True,
            ),
            row=1,
            col=2,
        )

        # Update heatmap axes for better readability
        fig.update_xaxes(
            title_text="Users",
            title_font=dict(size=12),
            row=1,
            col=2,
            tickangle=-45,
            tickfont=dict(size=8),
        )
        fig.update_yaxes(
            title_text="Users",
            title_font=dict(size=12),
            row=1,
            col=2,
            tickfont=dict(size=8),
        )

        # 3. Cluster sizes bar chart
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        fig.add_trace(
            go.Bar(
                x=[f"Cluster {i}" for i in unique_labels],
                y=counts,
                marker_color=[colors[i % len(colors)] for i in unique_labels],
                text=counts,
                textposition="auto",
                showlegend=False,
            ),
            row=1,
            col=3,
        )

        fig.update_xaxes(
            title_text="Cluster",
            title_font=dict(size=12),
            row=1,
            col=3,
            showgrid=True,
            gridwidth=1,
            gridcolor="lightgray",
        )
        fig.update_yaxes(
            title_text="Number of Users",
            title_font=dict(size=12),
            row=1,
            col=3,
            showgrid=True,
            gridwidth=1,
            gridcolor="lightgray",
        )

        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor="center",
                font=dict(size=16),
            ),
            height=height,
            width=1400,  # Wider for better proportions
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
            ),
            margin=dict(l=50, r=50, t=80, b=50),
        )

        return fig, cluster_labels

    @staticmethod
    def plot_recommendation_comparison(
        models: list[str],
        metrics: dict[str, list[float]],
        title: str = "Model Comparison",
        height: int = 500,
    ) -> go.Figure:
        """Compare multiple recommendation models across metrics.

        Args:
            models: List of model names
            metrics: Dictionary mapping metric names to lists of values (one per model)
            title: Plot title
            height: Plot height

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

        for i, (metric_name, values) in enumerate(metrics.items()):
            fig.add_trace(
                go.Bar(
                    name=metric_name,
                    x=models,
                    y=values,
                    marker_color=colors[i % len(colors)],
                ),
            )

        fig.update_layout(
            title=title,
            xaxis_title="Models",
            yaxis_title="Metric Value",
            height=height,
            barmode="group",
        )

        return fig

    @staticmethod
    def plot_training_history_comprehensive(
        history: Any,
        title: str = "Training Progress",
        height: int = 400,
        width: int = 1200,
    ) -> go.Figure:
        """Create comprehensive training history plot with loss and all metrics.

        Args:
            history: Keras training history object or dict with history data
            title: Plot title
            height: Plot height
            width: Plot width

        Returns:
            Plotly figure
        """
        # Handle both History objects and dicts
        if isinstance(history, dict):
            hist_dict = history
        else:
            hist_dict = history.history

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Training Loss", "Training Metrics"),
        )

        # Plot loss
        fig.add_trace(
            go.Scatter(
                y=hist_dict["loss"],
                name="Loss",
                line=dict(color="red", width=2),
            ),
            row=1,
            col=1,
        )
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_yaxes(title_text="Loss Value", row=1, col=1)

        # Plot metrics if available
        metrics_to_plot = [k for k in hist_dict.keys() if k != "loss"]
        colors = ["blue", "green", "purple", "orange", "brown"]
        for i, metric in enumerate(metrics_to_plot[:5]):  # Limit to 5 metrics
            fig.add_trace(
                go.Scatter(
                    y=hist_dict[metric],
                    name=metric,
                    line=dict(color=colors[i % len(colors)], width=2),
                ),
                row=1,
                col=2,
            )
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_yaxes(title_text="Metric Value", row=1, col=2)

        fig.update_layout(height=height, width=width, title_text=title, showlegend=True)

        return fig

    @staticmethod
    def plot_similarity_distribution(
        similarity_matrices: np.ndarray,
        train_y: np.ndarray,
        n_users: int = 5,
        title: str = "Similarity Score Distribution",
        height: int = 400,
        width: int = 1000,
    ) -> go.Figure:
        """Plot similarity score distribution for positive vs negative items.

        Args:
            similarity_matrices: Array of shape (n_users, n_items) with similarity scores
            train_y: Binary labels of shape (n_users, n_items) indicating positive items
            n_users: Number of users to analyze
            title: Plot title
            height: Plot height
            width: Plot width

        Returns:
            Plotly figure with statistics
        """
        similarity_matrices = np.asarray(similarity_matrices)
        train_y = np.asarray(train_y)

        # Collect similarity scores for positive and negative items
        positive_similarities = []
        negative_similarities = []

        for i in range(min(n_users, len(similarity_matrices))):
            similarities = similarity_matrices[i]
            user_labels = (
                train_y[i] if i < len(train_y) else np.zeros(len(similarities))
            )

            positive_mask = user_labels > 0.5
            negative_mask = user_labels < 0.5

            if positive_mask.any():
                positive_similarities.extend(similarities[positive_mask])
            if negative_mask.any():
                negative_similarities.extend(similarities[negative_mask])

        # Create distribution plot
        fig = go.Figure()

        if positive_similarities:
            fig.add_trace(
                go.Histogram(
                    x=positive_similarities,
                    name="Positive Items",
                    opacity=0.7,
                    nbinsx=30,
                    marker=dict(color="green"),
                ),
            )

        if negative_similarities:
            fig.add_trace(
                go.Histogram(
                    x=negative_similarities,
                    name="Negative Items",
                    opacity=0.7,
                    nbinsx=30,
                    marker=dict(color="red"),
                ),
            )

        fig.update_xaxes(title_text="Similarity Score")
        fig.update_yaxes(title_text="Frequency")
        fig.update_layout(
            height=height,
            width=width,
            title=title,
            barmode="overlay",
            showlegend=True,
        )

        # Print statistics
        stats = {
            "positive_mean": np.mean(positive_similarities)
            if positive_similarities
            else 0,
            "positive_std": np.std(positive_similarities)
            if positive_similarities
            else 0,
            "negative_mean": np.mean(negative_similarities)
            if negative_similarities
            else 0,
            "negative_std": np.std(negative_similarities)
            if negative_similarities
            else 0,
            "separation": (
                np.mean(positive_similarities) > np.mean(negative_similarities)
                if positive_similarities and negative_similarities
                else False
            ),
        }

        return fig, stats

    @staticmethod
    def plot_topk_scores(
        user_features: np.ndarray,
        item_features: np.ndarray,
        model: Any,
        user_idx: int = 0,
        title: str = "Top-K Recommendation Scores",
        height: int = 400,
        width: int = 900,
    ) -> go.Figure:
        """Plot top-K recommendation scores for a sample user.

        Args:
            user_features: User feature array
            item_features: Item feature array
            model: Trained recommendation model with compute_similarities method
            user_idx: User index to analyze
            title: Plot title
            height: Plot height
            width: Plot width

        Returns:
            Plotly figure
        """
        import tensorflow as tf
        from kmr.layers import TopKRecommendationSelector

        # Handle different item_features shapes
        if len(item_features.shape) == 3:
            # item_features is 3D: (n_users, n_items, feature_dim)
            sample_user_feat = tf.constant([user_features[user_idx]])
            sample_item_feats = tf.constant([item_features[user_idx]])
            n_items = item_features.shape[1]
        else:
            # item_features is 2D: (n_items, feature_dim)
            sample_user_feat = tf.constant([user_features[user_idx]])
            sample_item_feats = tf.constant([item_features])
            n_items = item_features.shape[0]

        # Call model and extract dictionary output
        try:
            # Try with 4 inputs first (Unified models: user_ids, user_features, item_ids, item_features)
            sample_user_ids = tf.constant([user_idx], dtype=tf.int32)
            sample_item_ids = tf.constant([np.arange(n_items, dtype=np.int32)])
            output = model(
                [sample_user_ids, sample_user_feat, sample_item_ids, sample_item_feats],
                training=False,
            )
        except (ValueError, TypeError, IndexError):
            # Fall back to 2 inputs for other models
            output = model([sample_user_feat, sample_item_feats], training=False)

        # Extract similarities from dictionary output
        if isinstance(output, dict):
            # Get similarity/score matrix - try different possible keys
            if "similarities" in output:
                similarities = output["similarities"]
            elif "scores" in output:
                similarities = output["scores"]
            elif "combined_scores" in output:
                similarities = output["combined_scores"]
            elif "masked_scores" in output:
                similarities = output["masked_scores"]
            else:
                # Fall back to first value if no known key found
                similarities = next(iter(output.values()))
            rec_indices = output.get("rec_indices", None)
            rec_scores = output.get("rec_scores", None)
        else:
            # Fallback for older tuple-based outputs
            similarities = output if not isinstance(output, tuple) else output[0]
            rec_indices = None
            rec_scores = None

        # If we don't have rec_indices/scores from the model, compute them
        if rec_indices is None or rec_scores is None:
            selector = TopKRecommendationSelector(k=model.top_k)
            rec_indices, rec_scores = selector(similarities)

        rec_scores_np = (
            rec_scores[0].numpy()
            if hasattr(rec_scores[0], "numpy")
            else np.array(rec_scores[0])
        )
        rec_indices_np = (
            rec_indices[0].numpy()
            if hasattr(rec_indices[0], "numpy")
            else np.array(rec_indices[0])
        )

        # Plot top-K scores
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=[f"Item {i}" for i in rec_indices_np],
                y=rec_scores_np,
                marker=dict(color=rec_scores_np, colorscale="Viridis", showscale=True),
            ),
        )
        fig.update_layout(
            title=f"{title} for User {user_idx}",
            xaxis_title="Recommended Items",
            yaxis_title="Similarity Score",
            height=height,
            width=width,
        )

        return fig

    @staticmethod
    def plot_prediction_confidence(
        similarity_matrices: np.ndarray,
        user_ids: np.ndarray | None = None,
        title: str = "Model Prediction Confidence",
        height: int = 400,
        width: int = 900,
    ) -> go.Figure:
        """Plot model prediction confidence (difference between top and 2nd scores).

        Args:
            similarity_matrices: Array of shape (n_users, n_items) with similarity scores
            user_ids: Optional array of user IDs for labeling
            title: Plot title
            height: Plot height
            width: Plot width

        Returns:
            Plotly figure with mean confidence
        """
        similarity_matrices = np.asarray(similarity_matrices)
        n_users = len(similarity_matrices)

        if user_ids is None:
            user_ids = np.arange(n_users)

        confidence_scores = []

        for i in range(n_users):
            similarities = similarity_matrices[i]
            sorted_scores = np.sort(similarities)[::-1]
            if len(sorted_scores) > 1:
                confidence = sorted_scores[0] - sorted_scores[1]
            else:
                confidence = sorted_scores[0] if len(sorted_scores) > 0 else 0
            confidence_scores.append(confidence)

        # Plot confidence
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=[f"User {uid}" for uid in user_ids],
                y=confidence_scores,
                marker=dict(color="steelblue", line=dict(color="darkblue", width=1)),
            ),
        )
        fig.update_layout(
            title=f"{title} (Top Score - 2nd Place)",
            xaxis_title="User",
            yaxis_title="Confidence Score",
            height=height,
            width=width,
        )

        mean_confidence = np.mean(confidence_scores)
        return fig, mean_confidence

    @staticmethod
    def plot_embedding_space(
        user_features: np.ndarray,
        model: Any,
        user_ids: np.ndarray | None = None,
        title: str = "User Embedding Space",
        height: int = 500,
        width: int = 900,
    ) -> go.Figure:
        """Plot user embedding space (first 2 dimensions).

        Args:
            user_features: User feature array of shape (n_users, n_features)
            model: Trained model with user_tower attribute
            user_ids: Optional array of user IDs for labeling
            title: Plot title
            height: Plot height
            width: Plot width

        Returns:
            Plotly figure
        """
        import tensorflow as tf

        user_features = np.asarray(user_features)
        n_users = len(user_features)

        if user_ids is None:
            user_ids = np.arange(n_users)

        # Get user embeddings
        sample_user_feats = tf.constant(user_features)
        user_embeddings = model.user_tower(sample_user_feats, training=False).numpy()

        # Plot first 2 dimensions
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=(
                    user_embeddings[:, 0]
                    if user_embeddings.shape[1] > 0
                    else np.arange(len(user_embeddings))
                ),
                y=(
                    user_embeddings[:, 1]
                    if user_embeddings.shape[1] > 1
                    else user_embeddings[:, 0]
                ),
                mode="markers+text",
                text=[f"User {uid}" for uid in user_ids],
                textposition="top center",
                marker=dict(
                    size=12,
                    color=np.arange(n_users),
                    colorscale="Viridis",
                    line=dict(color="darkblue", width=1),
                    showscale=True,
                ),
            ),
        )
        fig.update_layout(
            title=f"{title} (First 2 Dimensions)",
            xaxis_title="Embedding Dim 1",
            yaxis_title="Embedding Dim 2",
            height=height,
            width=width,
        )

        return fig

    @staticmethod
    def create_recommendation_diagnostic_report(
        model: Any,
        history: Any,
        user_features: np.ndarray,
        item_features: np.ndarray,
        train_y: np.ndarray,
        n_sample_users: int = 10,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """Create comprehensive diagnostic report for recommendation models.

        This is a one-stop function that generates all diagnostic plots and metrics
        for a trained recommendation model.

        Args:
            model: Trained recommendation model
            history: Training history from model.fit()
            user_features: User feature array
            item_features: Item feature array
            train_y: Binary labels for training data
            n_sample_users: Number of sample users for analysis
            top_k: Number of top recommendations (defaults to model.top_k)

        Returns:
            Dictionary with all figures and statistics
        """
        import tensorflow as tf
        from kmr.layers import TopKRecommendationSelector

        if top_k is None:
            top_k = model.top_k

        n_sample_users = min(n_sample_users, len(user_features))
        sample_user_indices = np.arange(n_sample_users)

        # Generate recommendations and similarity matrices
        all_rec_indices = []
        all_rec_scores = []
        all_similarity_matrices = []

        for i in range(n_sample_users):
            user_idx = sample_user_indices[i]

            # Prepare model inputs based on model type and available features
            # Check if model expects 4 inputs (Unified models) or 2 inputs (other models)
            try:
                # Try with 4 inputs first (Unified models: user_ids, user_features, item_ids, item_features)
                # For unified models, item_features is typically 3D: (batch_size, n_items, item_feature_dim)
                sample_user_ids = tf.constant([user_idx], dtype=tf.int32)
                sample_user_feat = tf.constant([user_features[user_idx]])

                # Handle different item_features shapes
                if len(item_features.shape) == 3:
                    # Already in correct format (n_users, n_items, feature_dim)
                    sample_item_feats = tf.constant([item_features[user_idx]])
                    n_items = item_features.shape[1]
                else:
                    # item_features is 2D (n_items, feature_dim) - expand batch dimension
                    sample_item_feats = tf.constant([item_features])
                    n_items = item_features.shape[0]

                sample_item_ids = tf.constant([np.arange(n_items, dtype=np.int32)])
                output = model(
                    [
                        sample_user_ids,
                        sample_user_feat,
                        sample_item_ids,
                        sample_item_feats,
                    ],
                    training=False,
                )
            except (ValueError, TypeError, IndexError):
                # Fall back to 2 inputs for other models
                sample_user_feat = tf.constant([user_features[user_idx]])

                # Handle different item_features shapes for 2-input models
                if len(item_features.shape) == 3:
                    sample_item_feats = tf.constant([item_features[user_idx]])
                else:
                    sample_item_feats = tf.constant([item_features])

                output = model([sample_user_feat, sample_item_feats], training=False)

            # Extract scores from dictionary (works for all models with different key names)
            # Try different keys depending on model type
            if isinstance(output, dict):
                # Get similarity/score matrix - try different possible keys
                if "similarities" in output:
                    similarities = output["similarities"]
                elif "scores" in output:
                    similarities = output["scores"]
                elif "combined_scores" in output:
                    similarities = output["combined_scores"]
                elif "masked_scores" in output:
                    similarities = output["masked_scores"]
                else:
                    # If none of the standard keys, try the first available value
                    similarities = next(iter(output.values()))

                # Get recommendation indices and scores
                rec_indices = output.get("rec_indices", None)
                rec_scores = output.get("rec_scores", None)
            else:
                # Fallback for older tuple-based outputs
                similarities = output if not isinstance(output, tuple) else output[0]
                rec_indices = None
                rec_scores = None

            # If we don't have rec_indices/scores from the model, compute them
            if rec_indices is None or rec_scores is None:
                selector = TopKRecommendationSelector(k=top_k)
                rec_indices, rec_scores = selector(similarities)

            rec_indices_np = (
                rec_indices[0].numpy()
                if hasattr(rec_indices[0], "numpy")
                else np.array(rec_indices[0])
            )
            rec_scores_np = (
                rec_scores[0].numpy()
                if hasattr(rec_scores[0], "numpy")
                else np.array(rec_scores[0])
            )
            similarity_np = (
                similarities[0].numpy()
                if hasattr(similarities[0], "numpy")
                else np.array(similarities[0])
            )

            all_rec_indices.append(rec_indices_np)
            all_rec_scores.append(rec_scores_np)
            all_similarity_matrices.append(similarity_np)

        all_rec_indices = np.array(all_rec_indices)
        all_similarity_matrices = np.array(all_similarity_matrices)

        # Calculate diversity metrics
        unique_items_per_user = [len(np.unique(rec)) for rec in all_rec_indices]
        shared_items = len(
            set(all_rec_indices[0]).intersection(
                *[set(rec) for rec in all_rec_indices[1:]],
            ),
        )
        diversity_ratio = 1.0 - (shared_items / top_k) if top_k > 0 else 0.0

        # Generate all plots
        report = {
            "figures": {},
            "metrics": {
                "diversity": {
                    "shared_items": shared_items,
                    "diversity_ratio": diversity_ratio,
                    "avg_unique_items_per_user": np.mean(unique_items_per_user),
                },
            },
        }

        # 1. Training history
        report["figures"][
            "training_history"
        ] = KMRPlotter.plot_training_history_comprehensive(history)

        # 2. Similarity distribution
        fig_sim, sim_stats = KMRPlotter.plot_similarity_distribution(
            all_similarity_matrices,
            train_y,
            n_users=n_sample_users,
        )
        report["figures"]["similarity_distribution"] = fig_sim
        report["metrics"]["similarity"] = sim_stats

        # 3. Top-K scores
        report["figures"]["topk_scores"] = KMRPlotter.plot_topk_scores(
            user_features,
            item_features,
            model,
            user_idx=0,
        )

        # 4. Prediction confidence
        fig_conf, mean_conf = KMRPlotter.plot_prediction_confidence(
            all_similarity_matrices,
            sample_user_indices,
        )
        report["figures"]["prediction_confidence"] = fig_conf
        report["metrics"]["mean_confidence"] = mean_conf

        # 5. Embedding space (skip if model doesn't have user_tower)
        try:
            report["figures"]["embedding_space"] = KMRPlotter.plot_embedding_space(
                user_features[sample_user_indices],
                model,
                sample_user_indices,
            )
        except (AttributeError, ValueError, TypeError):
            # Skip embedding space plot for models without user_tower
            report["figures"]["embedding_space"] = None

        # 6. Recommendation diversity
        report["figures"][
            "recommendation_diversity"
        ] = KMRPlotter.plot_recommendation_diversity(
            all_rec_indices,
            sample_user_indices,
        )

        # 7. User clusters
        fig_clusters, cluster_labels = KMRPlotter.plot_user_clusters(
            all_similarity_matrices,
            sample_user_indices,
            n_clusters=3,
        )
        report["figures"]["user_clusters"] = fig_clusters
        report["metrics"]["cluster_labels"] = cluster_labels

        return report

    @staticmethod
    def print_diagnostic_summary(report: dict[str, Any]) -> None:
        """Print diagnostic summary from report.

        Args:
            report: Report dictionary from create_recommendation_diagnostic_report
        """
        print("\n" + "=" * 70)
        print("✅ MODEL DIAGNOSIS COMPLETE")
        print("=" * 70)

        # Diversity metrics
        div_metrics = report["metrics"]["diversity"]
        print("\n📊 Diversity Metrics:")
        print(f"   Shared items across all users: {div_metrics['shared_items']} items")
        print(f"   Diversity ratio: {div_metrics['diversity_ratio']:.2%}")
        print(
            f"   Avg unique items per user: {div_metrics['avg_unique_items_per_user']:.1f}",
        )

        # Similarity metrics
        if "similarity" in report["metrics"]:
            sim_metrics = report["metrics"]["similarity"]
            print("\n📊 Similarity Score Analysis:")
            print(
                f"   Positive items - Mean: {sim_metrics['positive_mean']:.4f}, "
                f"Std: {sim_metrics['positive_std']:.4f}",
            )
            print(
                f"   Negative items - Mean: {sim_metrics['negative_mean']:.4f}, "
                f"Std: {sim_metrics['negative_std']:.4f}",
            )
            print(
                f"   Separation (Pos > Neg): {'✅ Yes' if sim_metrics['separation'] else '❌ No'}",
            )

        # Confidence
        if "mean_confidence" in report["metrics"]:
            print(f"\n📊 Mean Confidence: {report['metrics']['mean_confidence']:.4f}")
            print("   (Higher values indicate more confident predictions)")

        print("\n" + "=" * 70)
        print("Key verification criteria:")
        print("  ✓ Loss decreases over epochs → Model learning")
        print("  ✓ Metrics improve over epochs → Better recommendations")
        print("  ✓ Positive > Negative similarities → Correct ranking")
        print("  ✓ High confidence scores → Confident predictions")
        print("  ✓ Diverse recommendations → No model collapse")
        print("  ✓ User clustering → Meaningful patterns learned")
        print("\nIf all checks pass → Model is working correctly! 🎉")
        print("=" * 70)
