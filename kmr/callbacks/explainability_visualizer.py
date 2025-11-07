"""Explainability visualizer callback for recommendation models.

This callback generates and logs visualizations of model explanations
during training, helping understand model behavior and debugging.
"""

from typing import Any, Optional
from collections.abc import Callable

import keras
import numpy as np
from loguru import logger


class ExplainabilityVisualizer(keras.callbacks.Callback):
    """Visualizes model explanations during training.

    This callback generates visualizations of similarity matrices, embedding spaces,
    and recommendation explanations at specified intervals during training.

    Args:
        eval_data: Validation/evaluation data tuple (inputs, labels).
        visualization_fn: Callable function that generates visualizations.
        frequency: Generate visualizations every N epochs (default=5).
        save_dir: Directory to save visualizations (optional).
        verbose: Verbosity level (default=1).
        name: Optional name for the callback.

    Example:
        ```python
        from kmr.callbacks import ExplainabilityVisualizer
        from kmr.utils.plotting import KMRPlotter

        def plot_fn(model, inputs, labels, epoch):
            indices, scores = model.predict(inputs)
            KMRPlotter.plot_similarity_distribution(
                scores, title=f"Epoch {epoch}"
            )

        model = TwoTowerModel(num_items=100)
        model.compile(optimizer=keras.optimizers.Adam(), loss=loss_fn, metrics=metrics)

        callback = ExplainabilityVisualizer(
            eval_data=(val_inputs, val_labels),
            visualization_fn=plot_fn,
            frequency=5
        )
        model.fit(x=train_data, y=train_labels, callbacks=[callback])
        ```
    """

    def __init__(
        self,
        eval_data: tuple[Any, Any],
        visualization_fn: Optional[Callable] = None,
        frequency: int = 5,
        save_dir: Optional[str] = None,
        verbose: int = 1,
        name: str = "ExplainabilityVisualizer",
        **kwargs: Any,
    ) -> None:
        """Initialize the explainability visualizer callback."""
        super().__init__(**kwargs)
        self.eval_data = eval_data
        self.visualization_fn = visualization_fn
        self.frequency = frequency
        self.save_dir = save_dir
        self.verbose = verbose
        self.name = name
        self.epoch_visualizations = []

        if self.save_dir:
            import os

            os.makedirs(self.save_dir, exist_ok=True)
            logger.info(f"Visualizations will be saved to: {self.save_dir}")

    def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        """Generate and log visualizations at the end of epochs.

        Args:
            epoch: Current epoch number (0-indexed).
            logs: Dictionary containing metric values.
        """
        if (epoch + 1) % self.frequency == 0:
            try:
                if self.visualization_fn:
                    if self.verbose >= 1:
                        logger.info(f"Generating explanations for epoch {epoch + 1}...")

                    eval_inputs, eval_labels = self.eval_data

                    # Call the visualization function
                    self.visualization_fn(
                        model=self.model,
                        inputs=eval_inputs,
                        labels=eval_labels,
                        epoch=epoch + 1,
                        save_dir=self.save_dir,
                    )

                    self.epoch_visualizations.append(epoch + 1)

                    if self.verbose >= 1:
                        logger.info(
                            f"✓ Explanations generated successfully for epoch {epoch + 1}",
                        )

            except Exception as e:
                logger.warning(
                    f"Failed to generate explanations at epoch {epoch + 1}: {str(e)}",
                )

    def on_train_end(self, logs: dict[str, float] | None = None) -> None:
        """Log summary of visualizations generated during training.

        Args:
            logs: Final metric values.
        """
        if self.verbose >= 1:
            if self.epoch_visualizations:
                logger.info(
                    f"✅ Generated explanations at epochs: {self.epoch_visualizations}",
                )
            else:
                logger.info("⚠ No explanations were generated during training")

    def get_config(self) -> dict[str, Any]:
        """Get callback configuration for serialization.

        Returns:
            Dictionary with callback configuration.
        """
        return {
            "frequency": self.frequency,
            "save_dir": self.save_dir,
            "verbose": self.verbose,
            "name": self.name,
        }


class SimilarityMatrixVisualizer(keras.callbacks.Callback):
    """Specialized callback for visualizing user-item similarity matrices.

    This callback computes and logs similarity matrices to track how
    recommendations change during training.

    Args:
        eval_data: Validation/evaluation data.
        compute_similarity_fn: Function that computes similarity matrices.
        frequency: Visualize every N epochs (default=10).
        top_k: Show top-K similarities (default=5).
        verbose: Verbosity level (default=1).

    Example:
        ```python
        callback = SimilarityMatrixVisualizer(
            eval_data=(val_inputs, val_labels),
            compute_similarity_fn=model.compute_similarities,
            frequency=5,
            top_k=5
        )
        model.fit(x=train_data, y=train_labels, callbacks=[callback])
        ```
    """

    def __init__(
        self,
        eval_data: tuple[Any, Any],
        compute_similarity_fn: Callable,
        frequency: int = 10,
        top_k: int = 5,
        verbose: int = 1,
        name: str = "SimilarityMatrixVisualizer",
        **kwargs: Any,
    ) -> None:
        """Initialize the similarity matrix visualizer."""
        super().__init__(**kwargs)
        self.eval_data = eval_data
        self.compute_similarity_fn = compute_similarity_fn
        self.frequency = frequency
        self.top_k = top_k
        self.verbose = verbose
        self.name = name
        self.similarity_history = []

    def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        """Compute and log similarity metrics.

        Args:
            epoch: Current epoch number (0-indexed).
            logs: Dictionary containing metric values.
        """
        if (epoch + 1) % self.frequency == 0:
            try:
                eval_inputs, _ = self.eval_data

                # Call the model directly to get unified output
                output = self.model(eval_inputs, training=False)

                # Extract similarities from unified output
                if isinstance(output, tuple):
                    # New unified output format: (similarities, indices, scores, ...)
                    similarities = output[0]
                else:
                    # Backward compatibility with raw similarities
                    similarities = output

                # Compute statistics
                mean_sim = float(np.mean(similarities))
                std_sim = float(np.std(similarities))
                max_sim = float(np.max(similarities))
                min_sim = float(np.min(similarities))

                if self.verbose >= 1:
                    logger.info(
                        f"Epoch {epoch + 1} - Similarity Stats | "
                        f"Mean: {mean_sim:.4f}, Std: {std_sim:.4f}, "
                        f"Range: [{min_sim:.4f}, {max_sim:.4f}]",
                    )

                self.similarity_history.append(
                    {
                        "epoch": epoch + 1,
                        "mean": mean_sim,
                        "std": std_sim,
                        "max": max_sim,
                        "min": min_sim,
                    },
                )

            except Exception as e:
                logger.warning(
                    f"Failed to compute similarities at epoch {epoch + 1}: {str(e)}",
                )

    def get_config(self) -> dict[str, Any]:
        """Get callback configuration.

        Returns:
            Dictionary with callback configuration.
        """
        return {
            "frequency": self.frequency,
            "top_k": self.top_k,
            "verbose": self.verbose,
            "name": self.name,
        }
