"""Recommendation feedback adjustment layer."""

from typing import Any
from keras import KerasTensor
from keras.saving import register_keras_serializable

from kmr.layers._base_layer import BaseLayer


@register_keras_serializable(package="kmr.layers")
class FeedbackAdjustmentLayer(BaseLayer):
    """Adjusts recommendation scores based on user feedback.

    Multiplies prediction scores by feedback signals to adjust recommendations
    based on user's historical feedback or explicit preferences.

    Input: Tuple of (predictions, feedback)
    Output: Adjusted predictions (same shape as input predictions)
    """

    def __init__(self, name: str | None = None, **kwargs: Any) -> None:
        """Initialize layer."""
        self._validate_params()
        super().__init__(name=name, **kwargs)

    def _validate_params(self) -> None:
        """Validate parameters (no-op)."""
        pass

    def call(
        self,
        inputs: tuple[KerasTensor, KerasTensor] | None = None,
    ) -> KerasTensor:
        """Apply feedback adjustment.

        Args:
            inputs: Tuple of (predictions, feedback) or None.

        Returns:
            Adjusted predictions.
        """
        if inputs is None:
            raise ValueError("inputs cannot be None")

        predictions, feedback = inputs

        # Apply feedback by multiplication
        adjusted = predictions * feedback
        return adjusted

    def get_config(self) -> dict[str, Any]:
        """Get configuration."""
        return super().get_config()
