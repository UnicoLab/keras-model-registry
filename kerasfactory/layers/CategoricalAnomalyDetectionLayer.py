from typing import Any
import numpy as np
from keras import KerasTensor
from keras import layers, ops
from loguru import logger
from kerasfactory.layers._base_layer import BaseLayer
from keras.saving import register_keras_serializable

# Type aliases
NumericRule = tuple[str, float]  # e.g. (">", 0)
CategoricalRule = tuple[str, list[str]]  # e.g. ("==", ["red", "green"])
Rule = NumericRule | CategoricalRule
FeatureType = str  # "numerical", "cat_string", or "cat_int"
FeatureSpace = dict[str, FeatureType]
BusinessRules = dict[str, list[Rule]]


@register_keras_serializable(package="kerasfactory.layers")
class CategoricalAnomalyDetectionLayer(BaseLayer):
    """Backend-agnostic anomaly detection for categorical features.

    This layer detects anomalies in categorical features by checking if values belong to
    a predefined set of valid categories. Values not in this set are considered anomalous.

    The layer uses a Keras StringLookup or IntegerLookup layer internally to efficiently
    map input values to indices, which are then used to determine if a value is valid.

    Attributes:
        dtype: The data type of input values ('string' or 'int32').
        lookup: A Keras lookup layer for mapping values to indices.
        vocabulary: list of valid categorical values.

    Example:
        ```python
        layer = CategoricalAnomalyDetectionLayer(dtype='string')
        layer.initialize_from_stats(vocabulary=['red', 'green', 'blue'])
        outputs = layer(tf.constant([['red'], ['purple']]))
        print(outputs['anomaly'])  # [[False], [True]]
        ```
    """

    def __init__(self, dtype: str = "string", **kwargs) -> None:
        """Initializes the layer.

        Args:
            dtype: Data type of input values ('string' or 'int32'). Defaults to 'string'.
            **kwargs: Additional layer arguments.

        Raises:
            ValueError: If dtype is not 'string' or 'int32'.
        """
        self._dtype = None  # Initialize private attribute
        self.lookup: layers.StringLookup | layers.IntegerLookup | None = None
        self.built = False
        super().__init__(**kwargs)
        self.set_dtype(dtype.lower())  # Use setter method

    def build(self, _: tuple[int | None, int]) -> None:
        """Builds the layer.

        Args:
            input_shape: Shape of input tensor (batch_size, feature_dim).
        """
        if not self.built:
            # Ensure lookup layer is initialized
            if self.lookup is None:
                self.set_dtype(self._dtype)

            # Initialize lookup with empty vocabulary if not already set
            if not self.lookup.vocabulary_size():
                # For empty vocabulary, use adapt with empty array
                self.lookup.adapt(np.array([]).reshape(-1, 1))

            self.built = True

    @property
    def dtype(self) -> Any:
        """Get the dtype of the layer."""
        return self._dtype

    def set_dtype(self, value) -> None:
        """Set the dtype and initialize the appropriate lookup layer."""
        self._dtype = value
        if self._dtype == "string":
            self.lookup = layers.StringLookup(
                output_mode="int",
                num_oov_indices=1,
                name="string_lookup",
            )
        elif self._dtype == "int":
            self.lookup = layers.IntegerLookup(
                output_mode="int",
                num_oov_indices=1,
                name="int_lookup",
            )
        else:
            raise ValueError(f"Unsupported dtype: {value}")

    def initialize_from_stats(self, vocabulary: list[str | int]) -> None:
        """Initializes the layer with a vocabulary of valid values.

        Args:
            vocabulary: list of valid categorical values.
        """
        # Convert vocabulary to numpy array
        # For empty vocabulary, add a dummy value that will never match
        vocab_array = (
            np.array(["__EMPTY_VOCABULARY__"])
            if not vocabulary
            else np.array(vocabulary)
        )

        # Initialize the lookup layer with the vocabulary
        self.lookup.adapt(vocab_array.reshape(-1, 1))
        logger.info("Categorical layer initialized with vocabulary: {}", vocabulary)

    def compute_output_shape(
        self,
        input_shape: tuple[int | None, int],
    ) -> dict[str, tuple[int | None, int]]:
        """Compute the output shape of the layer.

        Args:
            input_shape: Input shape tuple.

        Returns:
            Dictionary mapping output names to their shapes.
        """
        batch_size = input_shape[0]
        return {
            "score": (batch_size, 1),
            "proba": (batch_size, 1),
            "threshold": (1, 1),
            "anomaly": (batch_size, 1),
            "reason": (batch_size, 1),
            "value": input_shape,
        }

    def call(
        self,
        inputs: KerasTensor,
        _: bool | None = None,
    ) -> dict[str, KerasTensor]:
        """Forward pass of the layer.

        Args:
            inputs: Input tensor.
            _: Training mode (unused).

        Returns:
            Dictionary containing anomaly detection results.
        """
        # Ensure layer is built
        if not self.built:
            self.build(inputs.shape)

        # Check if lookup layer is initialized
        if self.lookup is None:
            raise ValueError("Lookup layer not initialized. Call build() first.")

        # In this statistical branch we simply check membership using the lookup.
        mapped = self.lookup(inputs)
        anomaly = ops.equal(mapped, 0)
        score = ops.cast(anomaly, "float32") * 100.0
        proba = score  # 100 if anomaly, else 0
        threshold = ops.convert_to_tensor([0], dtype="float32")

        # Create the reason strings
        anomaly_reason = ops.convert_to_tensor(
            "Categorical anomaly: token not in allowed vocabulary",
            dtype="string",
        )
        normal_reason = ops.convert_to_tensor(
            "Token within allowed vocabulary",
            dtype="string",
        )

        reason = ops.where(anomaly, anomaly_reason, normal_reason)

        return {
            "score": score,
            "proba": proba,
            "threshold": threshold,
            "anomaly": anomaly,
            "reason": reason,
            "value": inputs,
        }

    def get_config(self) -> dict[str, Any]:
        """Get the configuration of the layer."""
        config = super().get_config()
        config.update(
            {
                "dtype": self.dtype,
                "vocabulary": self.lookup.get_vocabulary()
                if self.lookup is not None
                else [],
            },
        )
        return config

    @classmethod
    def from_config(cls, config) -> Any:
        """Create layer from configuration."""
        # Get vocabulary from config
        vocabulary = config.pop("vocabulary", [])
        # Create layer instance
        layer = cls(**config)
        # Initialize vocabulary
        if vocabulary:
            layer.initialize_from_stats(vocabulary)
        return layer
