"""TimeMixer model for time series forecasting."""

import math
from typing import Any
from keras import layers, ops
from keras import KerasTensor
from keras.saving import register_keras_serializable
from kerasfactory.models._base import BaseModel
from kerasfactory.layers.DataEmbeddingWithoutPosition import (
    DataEmbeddingWithoutPosition,
)
from kerasfactory.layers.ReversibleInstanceNormMultivariate import (
    ReversibleInstanceNormMultivariate,
)
from kerasfactory.layers.PastDecomposableMixing import PastDecomposableMixing


@register_keras_serializable(package="kerasfactory.models")
class TimeMixer(BaseModel):
    """TimeMixer: Decomposable Multi-Scale Mixing for Time Series Forecasting.

    A state-of-the-art time series forecasting model that uses series decomposition
    and multi-scale mixing to capture both trend and seasonal patterns.

    Args:
        seq_len: Input sequence length.
        pred_len: Prediction horizon.
        n_features: Number of time series features.
        d_model: Model dimension (default: 32).
        d_ff: Feed-forward dimension (default: 32).
        e_layers: Number of encoder layers (default: 4).
        dropout: Dropout rate (default: 0.1).
        decomp_method: Decomposition method ('moving_avg' or 'dft_decomp').
        moving_avg: Moving average window size (default: 25).
        top_k: Top-k frequencies for DFT (default: 5).
        channel_independence: 0 for channel-dependent, 1 for independent (default: 0).
        down_sampling_layers: Number of downsampling layers (default: 1).
        down_sampling_window: Downsampling window size (default: 2).
        down_sampling_method: Downsampling method ('avg', 'max', 'conv').
        use_norm: Whether to use normalization (default: True).
        decoder_input_size_multiplier: Decoder input multiplier (default: 0.5).
        name: Optional model name.

    Example:
        ```python
        import keras
        from kerasfactory.models import TimeMixer

        # Create model
        model = TimeMixer(
            seq_len=96,
            pred_len=12,
            n_features=7,
            d_model=32,
            e_layers=2
        )

        # Compile and train
        model.compile(optimizer='adam', loss='mse')

        x = keras.random.normal((32, 96, 7))
        y = keras.random.normal((32, 12, 7))
        model.fit(x, y, epochs=10)

        # Make predictions
        predictions = model.predict(x)
        ```
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_features: int,
        d_model: int = 32,
        d_ff: int = 32,
        e_layers: int = 4,
        dropout: float = 0.1,
        decomp_method: str = "moving_avg",
        moving_avg: int = 25,
        top_k: int = 5,
        channel_independence: int = 0,
        down_sampling_layers: int = 1,
        down_sampling_window: int = 2,
        down_sampling_method: str = "avg",
        use_norm: bool = True,
        decoder_input_size_multiplier: float = 0.5,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize TimeMixer model."""
        # Store parameters
        self._seq_len = seq_len
        self._pred_len = pred_len
        self._n_features = n_features
        self._d_model = d_model
        self._d_ff = d_ff
        self._e_layers = e_layers
        self._dropout = dropout
        self._decomp_method = decomp_method
        self._moving_avg = moving_avg
        self._top_k = top_k
        self._channel_independence = channel_independence
        self._down_sampling_layers = down_sampling_layers
        self._down_sampling_window = down_sampling_window
        self._down_sampling_method = down_sampling_method
        self._use_norm = use_norm
        self._decoder_input_size_multiplier = decoder_input_size_multiplier

        # Validate parameters
        self._validate_params()

        # Create model
        super().__init__(name=name or "TimeMixer", **kwargs)

        # Store as public attributes
        self.seq_len = self._seq_len
        self.pred_len = self._pred_len
        self.n_features = self._n_features
        self.d_model = self._d_model
        self.d_ff = self._d_ff
        self.e_layers = self._e_layers
        self.dropout_rate = self._dropout
        self.decomp_method = self._decomp_method
        self.moving_avg_kernel = self._moving_avg
        self.top_k = self._top_k
        self.channel_independence = self._channel_independence
        self.down_sampling_layers_count = self._down_sampling_layers
        self.down_sampling_window_size = self._down_sampling_window
        self.down_sampling_method = self._down_sampling_method
        self.use_norm = self._use_norm

        # Build label_len
        self.label_len = int(math.ceil(seq_len * decoder_input_size_multiplier))
        if (self.label_len >= seq_len) or (self.label_len <= 0):
            raise ValueError(
                f"Check decoder_input_size_multiplier={decoder_input_size_multiplier}, range (0,1)",
            )

    def _validate_params(self) -> None:
        """Validate parameters."""
        if self._decomp_method not in ["moving_avg", "dft_decomp"]:
            raise ValueError(
                f"decomp_method must be 'moving_avg' or 'dft_decomp', got {self._decomp_method}",
            )
        if self._channel_independence not in [0, 1]:
            raise ValueError(
                f"channel_independence must be 0 or 1, got {self._channel_independence}",
            )

    def build(self, input_shape: tuple[int, ...]) -> None:
        """Build model layers."""
        # Embedding
        self.embedding = DataEmbeddingWithoutPosition(
            c_in=self.n_features,
            d_model=self.d_model,
            dropout=self.dropout_rate,
        )

        # Normalization
        self.normalizer = ReversibleInstanceNormMultivariate(
            num_features=self.n_features,
            affine=True,
        )

        # Encoder blocks
        self.pdm_blocks = []
        for _ in range(self.e_layers):
            block = PastDecomposableMixing(
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                down_sampling_window=self.down_sampling_window_size,
                down_sampling_layers=self.down_sampling_layers_count,
                d_model=self.d_model,
                dropout=self.dropout_rate,
                channel_independence=self.channel_independence,
                decomp_method=self.decomp_method,
                d_ff=self.d_ff,
                moving_avg=self.moving_avg_kernel,
                top_k=self.top_k,
            )
            self.pdm_blocks.append(block)

        # Projection layers
        self.project_layers = []
        for i in range(self.down_sampling_layers_count + 1):
            current_len = math.ceil(
                self.seq_len // (self.down_sampling_window_size**i),
            )
            layer = layers.Dense(self.pred_len)
            self.project_layers.append(layer)

        # Output projection
        self.output_projection = layers.Dense(self.n_features)

        super().build(input_shape)

    def call(
        self,
        inputs: KerasTensor | tuple[KerasTensor, KerasTensor],
        training: bool | None = None,
    ) -> KerasTensor:
        """Forward pass of TimeMixer.

        Args:
            inputs: Input tensor or tuple of (x, x_mark) where x is the time series
                   and x_mark is optional temporal features.
            training: Whether in training mode.

        Returns:
            Forecast tensor of shape (batch, pred_len, n_features).
        """
        if isinstance(inputs, (list, tuple)):
            if len(inputs) == 2:
                x_enc, x_mark_enc = inputs
            else:
                x_enc = inputs[0]
                x_mark_enc = None
        else:
            x_enc = inputs
            x_mark_enc = None

        # Normalize
        x_enc_norm = self.normalizer(x_enc, mode="norm")

        # Embedding
        enc_out = self.embedding(
            [x_enc_norm, x_mark_enc] if x_mark_enc is not None else x_enc_norm,
        )

        # Encoder blocks
        enc_out_list = [enc_out]
        for block in self.pdm_blocks:
            enc_out_list = block(enc_out_list)

        # Decode
        dec_out = enc_out_list[0]
        dec_out = self.project_layers[0](ops.transpose(dec_out, (0, 2, 1)))
        dec_out = ops.transpose(dec_out, (0, 2, 1))

        # Output projection and denormalization
        dec_out = self.output_projection(dec_out)
        dec_out = self.normalizer(dec_out, mode="denorm")

        # Take last pred_len steps
        dec_out = dec_out[:, -self.pred_len :, :]

        return dec_out

    def get_config(self) -> dict[str, Any]:
        """Get model configuration."""
        config = super().get_config()
        config.update(
            {
                "seq_len": self.seq_len,
                "pred_len": self.pred_len,
                "n_features": self.n_features,
                "d_model": self.d_model,
                "d_ff": self.d_ff,
                "e_layers": self.e_layers,
                "dropout": self.dropout_rate,
                "decomp_method": self.decomp_method,
                "moving_avg": self.moving_avg_kernel,
                "top_k": self.top_k,
                "channel_independence": self.channel_independence,
                "down_sampling_layers": self.down_sampling_layers_count,
                "down_sampling_window": self.down_sampling_window_size,
                "down_sampling_method": self.down_sampling_method,
                "use_norm": self.use_norm,
                "decoder_input_size_multiplier": self.label_len / self.seq_len,
            },
        )
        return config
