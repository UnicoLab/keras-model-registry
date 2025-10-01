"""SeasonLayer for adding seasonal information based on month.

This layer adds seasonal information based on the month, encoding it as a one-hot vector
for the four seasons: Winter, Spring, Summer, and Fall.
"""

import keras
from keras import ops


class SeasonLayer(keras.layers.Layer):
    """Layer for adding seasonal information based on month.

    This layer adds seasonal information based on the month, encoding it as a one-hot vector
    for the four seasons: Winter, Spring, Summer, and Fall.

    Args:
        **kwargs: Additional layer arguments

    Input shape:
        Tensor with shape: `(..., 4)` containing [year, month, day, day_of_week]

    Output shape:
        Tensor with shape: `(..., 8)` containing the original 4 components plus
        4 one-hot encoded season values
    """

    def __init__(self, **kwargs):
        """Initialize the layer."""
        super().__init__(**kwargs)

    def call(self, inputs):
        """Apply the layer to the inputs.

        Args:
            inputs: Tensor with shape (..., 4) containing [year, month, day, day_of_week]

        Returns:
            Tensor with shape (..., 8) containing the original 4 components plus
            4 one-hot encoded season values
        """
        # Extract month (second component)
        month = inputs[..., 1]
        
        # Create one-hot encoded seasons
        # Winter: December (12), January (1), February (2)
        # Spring: March (3), April (4), May (5)
        # Summer: June (6), July (7), August (8)
        # Fall: September (9), October (10), November (11)
        
        # Initialize season tensors with zeros
        winter = ops.zeros_like(month, dtype="float32")
        spring = ops.zeros_like(month, dtype="float32")
        summer = ops.zeros_like(month, dtype="float32")
        fall = ops.zeros_like(month, dtype="float32")
        
        # Set season values based on month
        # Winter
        winter = ops.where(
            ops.logical_or(
                ops.equal(month, 12),
                ops.logical_or(
                    ops.equal(month, 1),
                    ops.equal(month, 2)
                )
            ),
            ops.ones_like(month, dtype="float32"),
            winter
        )
        
        # Spring
        spring = ops.where(
            ops.logical_or(
                ops.equal(month, 3),
                ops.logical_or(
                    ops.equal(month, 4),
                    ops.equal(month, 5)
                )
            ),
            ops.ones_like(month, dtype="float32"),
            spring
        )
        
        # Summer
        summer = ops.where(
            ops.logical_or(
                ops.equal(month, 6),
                ops.logical_or(
                    ops.equal(month, 7),
                    ops.equal(month, 8)
                )
            ),
            ops.ones_like(month, dtype="float32"),
            summer
        )
        
        # Fall
        fall = ops.where(
            ops.logical_or(
                ops.equal(month, 9),
                ops.logical_or(
                    ops.equal(month, 10),
                    ops.equal(month, 11)
                )
            ),
            ops.ones_like(month, dtype="float32"),
            fall
        )
        
        # Stack season values
        seasons = ops.stack([winter, spring, summer, fall], axis=-1)
        
        # Concatenate original inputs with seasons
        return ops.concatenate([inputs, seasons], axis=-1)

    def compute_output_shape(self, input_shape):
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input tensor

        Returns:
            Output shape
        """
        return input_shape[:-1] + (input_shape[-1] + 4,)

    def get_config(self):
        """Return the configuration of the layer.

        Returns:
            Dictionary containing the layer configuration
        """
        return super().get_config() 