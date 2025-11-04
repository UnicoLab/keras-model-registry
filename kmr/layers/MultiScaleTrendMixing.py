"""Multi-Scale Trend Mixing layer for hierarchical trend pattern mixing."""

import math
from typing import Any
from keras import layers, ops
from keras import KerasTensor
from keras.saving import register_keras_serializable
from kmr.layers._base_layer import BaseLayer


@register_keras_serializable(package="kmr.layers")
class MultiScaleTrendMixing(BaseLayer):
    """Mixes trend patterns across multiple scales top-down.

    Processes trend components at different temporal resolutions,
    mixing information from fine to coarse scales.

    Args:
        seq_len: Input sequence length.
        down_sampling_window: Window size for downsampling.
        down_sampling_layers: Number of downsampling layers.
        name: Optional name for the layer.

    Example:
        ```python
        import keras
        from kmr.layers import MultiScaleTrendMixing

        # Create trend mixing layer
        trend_mix = MultiScaleTrendMixing(seq_len=100, down_sampling_window=2,
                                         down_sampling_layers=2)

        # Process list of trend components at different scales
        trend_list = [keras.random.normal((32, 100, 8)),
                      keras.random.normal((32, 50, 8))]
        mixed_trends = trend_mix(trend_list)
        print("Number of outputs:", len(mixed_trends))
        ```
    """

    def __init__(
        self,
        seq_len: int,
        down_sampling_window: int,
        down_sampling_layers: int,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the MultiScaleTrendMixing layer.

        Args:
            seq_len: Sequence length.
            down_sampling_window: Downsampling window size.
            down_sampling_layers: Number of downsampling layers.
            name: Optional layer name.
            **kwargs: Additional keyword arguments.
        """
        # Set private attributes
        self._seq_len = seq_len
        self._down_sampling_window = down_sampling_window
        self._down_sampling_layers = down_sampling_layers

        # Validate parameters
        self._validate_params()

        # Set public attributes BEFORE calling parent's __init__
        self.seq_len = self._seq_len
        self.down_sampling_window = self._down_sampling_window
        self.down_sampling_layers = self._down_sampling_layers
        self.up_sampling_layers_list = None

        # Call parent's __init__ after setting public attributes
        super().__init__(name=name, **kwargs)

    def _validate_params(self) -> None:
        """Validate layer parameters."""
        if not isinstance(self._seq_len, int) or self._seq_len <= 0:
            raise ValueError(f"seq_len must be a positive integer, got {self._seq_len}")
        if (
            not isinstance(self._down_sampling_window, int)
            or self._down_sampling_window <= 0
        ):
            raise ValueError(
                f"down_sampling_window must be a positive integer, got {self._down_sampling_window}",
            )
        if (
            not isinstance(self._down_sampling_layers, int)
            or self._down_sampling_layers <= 0
        ):
            raise ValueError(
                f"down_sampling_layers must be a positive integer, got {self._down_sampling_layers}",
            )

    def build(self, input_shape: tuple[int, ...]) -> None:
        """Build the layer.

        Args:
            input_shape: Shape of input tensor.
        """
        self.up_sampling_layers_list = []
        for i in reversed(range(self.down_sampling_layers)):
            current_len = math.ceil(
                self.seq_len // (self.down_sampling_window ** (i + 1)),
            )
            next_len = math.ceil(self.seq_len // (self.down_sampling_window**i))

            # Create a dict of dense1 and dense2 instead of Sequential
            layer_dict = {
                "dense1": layers.Dense(current_len, activation="gelu"),
                "dense2": layers.Dense(next_len),
            }
            self.up_sampling_layers_list.append(layer_dict)

        super().build(input_shape)

    def _apply_up_sampling_layer(self, x: KerasTensor, layer_idx: int) -> KerasTensor:
        """Apply up-sampling layer (sequential Dense operations).

        Args:
            x: Input tensor.
            layer_idx: Index of the layer to apply.

        Returns:
            Processed tensor.
        """
        layer_dict = self.up_sampling_layers_list[layer_idx]
        x = layer_dict["dense1"](x)
        x = layer_dict["dense2"](x)
        return x

    def call(self, inputs: list[KerasTensor]) -> list[KerasTensor]:
        """Mix trend patterns across scales.

        Args:
            inputs: List of trend tensors at different scales.

        Returns:
            List of mixed trend patterns.
        """
        # Handle single input case - just return as-is
        if len(inputs) == 1:
            return [ops.transpose(inputs[0], (0, 2, 1))]

        # mixing low->high
        trend_list_reverse = list(reversed(inputs))
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [ops.transpose(out_low, (0, 2, 1))]

        for i in range(len(trend_list_reverse) - 1):
            # Upsample and mix
            out_high_res = self._apply_up_sampling_layer(out_low, i)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(ops.transpose(out_low, (0, 2, 1)))

        out_trend_list.reverse()
        return out_trend_list

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update(
            {
                "seq_len": self.seq_len,
                "down_sampling_window": self.down_sampling_window,
                "down_sampling_layers": self.down_sampling_layers,
            },
        )
        return config
