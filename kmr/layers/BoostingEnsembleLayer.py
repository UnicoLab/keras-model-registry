"""This module implements a BoostingEnsembleLayer that aggregates multiple BoostingBlocks in parallel.
Their outputs are combined via learnable weights to form an ensemble prediction.
This is similar in spirit to boosting ensembles but implemented in a differentiable, end-to-end manner.
"""

from typing import Any
from loguru import logger
from keras import ops
from keras import KerasTensor
from keras.saving import register_keras_serializable
from kmr.layers._base_layer import BaseLayer
from kmr.layers.BoostingBlock import BoostingBlock


@register_keras_serializable(package="kmr.layers")
class BoostingEnsembleLayer(BaseLayer):
    """Ensemble layer of boosting blocks for tabular data.

    This layer aggregates multiple boosting blocks (weak learners) in parallel. Each
    learner produces a correction to the input. A gating mechanism (via learnable weights)
    then computes a weighted sum of the learners' outputs.

    Args:
        num_learners: Number of boosting blocks in the ensemble. Default is 3.
        learner_units: Number of hidden units in each boosting block. Can be an int for
            single hidden layer or a list of ints for multiple hidden layers. Default is 64.
        hidden_activation: Activation function for hidden layers in boosting blocks. Default is 'relu'.
        output_activation: Activation function for the output layer in boosting blocks. Default is None.
        gamma_trainable: Whether the scaling factor gamma in boosting blocks is trainable. Default is True.
        dropout_rate: Optional dropout rate to apply in boosting blocks. Default is None.
        name: Optional name for the layer.

    Input shape:
        N-D tensor with shape: (batch_size, ..., input_dim)

    Output shape:
        Same shape as input: (batch_size, ..., input_dim)

    Example:
        ```python
        import keras
        from kmr.layers import BoostingEnsembleLayer

        # Create sample input data
        x = keras.random.normal((32, 16))  # 32 samples, 16 features

        # Basic usage
        ensemble = BoostingEnsembleLayer(num_learners=3, learner_units=64)
        y = ensemble(x)
        print("Ensemble output shape:", y.shape)  # (32, 16)

        # Advanced configuration
        ensemble = BoostingEnsembleLayer(
            num_learners=5,
            learner_units=[32, 16],  # Two hidden layers in each learner
            hidden_activation='selu',
            dropout_rate=0.1
        )
        y = ensemble(x)
        ```
    """

    def __init__(
        self,
        num_learners: int = 3,
        learner_units: int | list[int] = 64,
        hidden_activation: str = "relu",
        output_activation: str | None = None,
        gamma_trainable: bool = True,
        dropout_rate: float | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        # Set private attributes before calling parent's __init__
        self._num_learners = num_learners
        self._learner_units = learner_units
        self._hidden_activation = hidden_activation
        self._output_activation = output_activation
        self._gamma_trainable = gamma_trainable
        self._dropout_rate = dropout_rate

        # Validate parameters
        if num_learners <= 0:
            raise ValueError(f"num_learners must be positive, got {num_learners}")
        if dropout_rate is not None and not 0 <= dropout_rate < 1:
            raise ValueError("dropout_rate must be between 0 and 1")

        # Set public attributes before calling parent's __init__
        self.num_learners = self._num_learners
        self.learner_units = self._learner_units
        self.hidden_activation = self._hidden_activation
        self.output_activation = self._output_activation
        self.gamma_trainable = self._gamma_trainable
        self.dropout_rate = self._dropout_rate
        self.learners = None
        self.alpha = None

        super().__init__(name=name, **kwargs)

    def _validate_params(self) -> None:
        """Validate layer parameters."""
        if self.num_learners <= 0:
            raise ValueError(f"num_learners must be positive, got {self.num_learners}")
        if self.dropout_rate is not None and not 0 <= self.dropout_rate < 1:
            raise ValueError("dropout_rate must be between 0 and 1")

    def build(self, input_shape: tuple[int, ...]) -> None:
        """Builds the layer with the given input shape.

        Args:
            input_shape: Tuple of integers defining the input shape.
        """
        # Create a list of boosting blocks
        self.learners = [
            BoostingBlock(
                hidden_units=self.learner_units,
                hidden_activation=self.hidden_activation,
                output_activation=self.output_activation,
                gamma_trainable=self.gamma_trainable,
                dropout_rate=self.dropout_rate,
                name=f"learner_{i}",
            )
            for i in range(self.num_learners)
        ]

        # Learnable weights for combining learner outputs
        self.alpha = self.add_weight(
            name="alpha",
            shape=(self.num_learners,),
            initializer="zeros",
            trainable=True,
        )

        logger.debug(
            f"BoostingEnsembleLayer built with num_learners={self.num_learners}, "
            f"learner_units={self.learner_units}, activation={self.hidden_activation}",
        )
        super().build(input_shape)

    def call(self, inputs: KerasTensor, training: bool | None = None) -> KerasTensor:
        """Forward pass of the layer.

        Args:
            inputs: Input tensor.
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode. Only relevant when using dropout.

        Returns:
            Output tensor of same shape as input.
        """
        # Get outputs from each learner
        learner_outputs = [
            learner(inputs, training=training) for learner in self.learners
        ]

        # Stack outputs: shape (batch, num_learners, features)
        stacked = ops.stack(learner_outputs, axis=1)

        # Compute normalized weights
        weights = ops.softmax(self.alpha)  # Shape: (num_learners,)
        weights = ops.reshape(weights, (1, self.num_learners, 1))

        # Aggregate outputs using the weights
        ensemble_output = ops.sum(stacked * weights, axis=1)

        return ensemble_output

    def get_config(self) -> dict[str, Any]:
        """Returns the config of the layer.

        Returns:
            Python dictionary containing the layer configuration.
        """
        config = super().get_config()

        # Use public attributes after initialization
        config.update(
            {
                "num_learners": self.num_learners,
                "learner_units": self.learner_units,
                "hidden_activation": self.hidden_activation,
                "output_activation": self.output_activation,
                "gamma_trainable": self.gamma_trainable,
                "dropout_rate": self.dropout_rate,
            },
        )
        return config
