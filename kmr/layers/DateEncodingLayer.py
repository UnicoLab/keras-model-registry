"""DateEncodingLayer for encoding date components into cyclical features.

This layer takes date components (year, month, day, day of week) and encodes them
into cyclical features using sine and cosine transformations.
"""

from typing import Any

import keras
import numpy as np
from keras import ops


class DateEncodingLayer(keras.layers.Layer):
    """Layer for encoding date components into cyclical features.

    This layer takes date components (year, month, day, day of week) and encodes them
    into cyclical features using sine and cosine transformations. The year is normalized
    to a range between 0 and 1 based on min_year and max_year.

    Args:
        min_year: Minimum year for normalization (default: 1900)
        max_year: Maximum year for normalization (default: 2100)
        **kwargs: Additional layer arguments

    Input shape:
        Tensor with shape: `(..., 4)` containing [year, month, day, day_of_week]

    Output shape:
        Tensor with shape: `(..., 8)` containing cyclical encodings:
        [year_sin, year_cos, month_sin, month_cos, day_sin, day_cos, dow_sin, dow_cos]
    """

    def __init__(self, min_year: int = 1900, max_year: int = 2100, **kwargs):
        """Initialize the layer."""
        super().__init__(**kwargs)
        self.min_year = min_year
        self.max_year = max_year

        # Validate inputs
        if min_year >= max_year:
            raise ValueError(
                f"min_year ({min_year}) must be less than max_year ({max_year})",
            )

    def call(self, inputs) -> Any:
        """Apply the layer to the inputs.

        Args:
            inputs: Tensor with shape (..., 4) containing [year, month, day, day_of_week]

        Returns:
            Tensor with shape (..., 8) containing cyclical encodings
        """
        # Extract date components
        year = ops.cast(inputs[..., 0], dtype="float32")
        month = ops.cast(inputs[..., 1], dtype="float32")
        day = ops.cast(inputs[..., 2], dtype="float32")
        day_of_week = ops.cast(inputs[..., 3], dtype="float32")

        # Normalize year to [0, 1]
        year_normalized = (year - self.min_year) / (self.max_year - self.min_year)

        # Encode year (normalized to [0, 1])
        year_sin = ops.sin(2 * np.pi * year_normalized)
        year_cos = ops.cos(2 * np.pi * year_normalized)

        # Encode month (1-12)
        month_sin = ops.sin(2 * np.pi * month / 12)
        month_cos = ops.cos(2 * np.pi * month / 12)

        # Encode day of month (1-31)
        day_sin = ops.sin(2 * np.pi * day / 31)
        day_cos = ops.cos(2 * np.pi * day / 31)

        # Encode day of week (0-6)
        dow_sin = ops.sin(2 * np.pi * day_of_week / 7)
        dow_cos = ops.cos(2 * np.pi * day_of_week / 7)

        # Combine all features
        encoded = ops.stack(
            [
                year_sin,
                year_cos,
                month_sin,
                month_cos,
                day_sin,
                day_cos,
                dow_sin,
                dow_cos,
            ],
            axis=-1,
        )

        return encoded

    def compute_output_shape(self, input_shape) -> tuple[int, ...]:
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input tensor

        Returns:
            Output shape
        """
        return input_shape[:-1] + (8,)

    def get_config(self) -> dict[str, Any]:
        """Return the configuration of the layer.

        Returns:
            Dictionary containing the layer configuration
        """
        config = super().get_config()
        config.update(
            {
                "min_year": self.min_year,
                "max_year": self.max_year,
            },
        )
        return config
