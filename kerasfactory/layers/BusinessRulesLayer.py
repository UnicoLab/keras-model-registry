"""This module implements a BusinessRulesLayer that allows applying configurable business rules
to neural network outputs. This enables combining learned patterns with explicit domain knowledge.
"""

from typing import Any
from loguru import logger
from keras import ops, initializers
from keras.ops import convert_to_tensor
from keras import KerasTensor
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
class BusinessRulesLayer(BaseLayer):
    """Evaluates business-defined rules for anomaly detection.

    This layer applies user-defined business rules to detect anomalies. Rules can be
    defined for both numerical and categorical features.

    For numerical features:
        - Comparison operators: '>' and '<'
        - Example: [(">", 0), ("<", 100)] for range validation

    For categorical features:
        - Set operators: '==', 'in', '!=', 'not in'
        - Example: [("in", ["red", "green", "blue"])] for valid categories

    Attributes:
        rules: List of rule tuples (operator, value).
        feature_type: Type of feature ('numerical' or 'categorical').

    Example:
        ```python
        # Numerical rules
        layer = BusinessRulesLayer(rules=[(">", 0), ("<", 100)], feature_type="numerical")
        outputs = layer(tf.constant([[50.0], [-10.0]]))
        print(outputs['business_anomaly'])  # [[False], [True]]

        # Categorical rules
        layer = BusinessRulesLayer(
            rules=[("in", ["red", "green"])],
            feature_type="categorical"
        )
        outputs = layer(tf.constant([["red"], ["blue"]]))
        print(outputs['business_anomaly'])  # [[False], [True]]
        ```
    """

    def __init__(
        self,
        rules: list[Rule],
        feature_type: str,
        trainable_weights: bool = True,
        weight_initializer: str | initializers.Initializer = "ones",
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initializes the layer.

        Args:
            rules: List of rule tuples (operator, value).
            feature_type: Type of feature ('numerical' or 'categorical').
            trainable_weights: Whether to use trainable weights for soft rule enforcement.
                Default is True.
            weight_initializer: Initializer for rule weights. Default is 'ones'.
            name: Optional name for the layer.
            **kwargs: Additional layer arguments.

        Raises:
            ValueError: If feature_type is invalid or rules have invalid operators.
        """
        # Set attributes before calling parent's __init__
        self._rules = rules
        self._feature_type = feature_type
        self._weights_trainable = trainable_weights
        self._weight_initializer = initializers.get(weight_initializer)

        # Validate feature type
        if feature_type not in ["numerical", "categorical"]:
            raise ValueError(
                f"Invalid feature_type: {feature_type}. "
                "Must be 'numerical' or 'categorical'",
            )

        super().__init__(name=name, **kwargs)

        # Set public attributes
        self.rules = self._rules
        self.feature_type = self._feature_type
        self.weights_trainable = self._weights_trainable
        self.weight_initializer = self._weight_initializer

    def build(self, input_shape: tuple[int, ...]) -> None:
        """Build the layer.

        Args:
            input_shape: Shape of input tensor.
        """
        # Create trainable weights for each rule
        self.rule_weights = []
        for i, _ in enumerate(self.rules):
            weight = self.add_weight(
                name=f"rule_weight_{i}",
                shape=(1,),
                initializer=self.weight_initializer,
                trainable=self.weights_trainable,
            )
            self.rule_weights.append(weight)

        logger.debug(
            f"BusinessRulesLayer built with {len(self.rules)} rules, trainable={self.weights_trainable}",
        )
        super().build(input_shape)

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
            "business_score": (batch_size, 1),
            "business_proba": (batch_size, 1),
            "business_anomaly": (batch_size, 1),
            "business_reason": (batch_size, 1),
            "business_value": input_shape,
        }

    def call(
        self,
        inputs: KerasTensor,
        _: bool | None = None,
    ) -> dict[str, KerasTensor]:
        """Forward pass of the layer.

        Args:
            inputs: Input tensor.
            training: Whether in training mode. Not used.

        Returns:
            dictionary containing:
                - business_score: Violation score (0 = no violation)
                - business_proba: Probability of violation (0-100)
                - business_anomaly: Boolean mask of violations
                - business_reason: String tensor explaining violations
                - business_value: Original input values
        """
        x = inputs
        if ops.ndim(x) == 1:
            x = ops.expand_dims(x, -1)

        if self.feature_type == "numerical":
            return self._apply_numerical_rules(x)
        elif self.feature_type == "categorical":
            return self._apply_categorical_rules(x)
        else:
            raise ValueError(f"Unsupported feature type: {self.feature_type}")

    def _apply_numerical_rules(self, x: KerasTensor) -> dict[str, KerasTensor]:
        """Apply numerical rules to input tensor.

        Args:
            x: Input tensor.

        Returns:
            dictionary with rule evaluation results.
        """
        x = ops.cast(x, "float32")
        violations = []
        scores = []
        reasons = []

        for (op, value), weight in zip(self.rules, self.rule_weights, strict=False):
            # Handle both numeric and string values
            if isinstance(value, int | float):
                value_t = convert_to_tensor(float(value), dtype="float32")
            elif isinstance(value, list):
                # For list values, try to convert the first element to float
                try:
                    value_t = convert_to_tensor(float(value[0]), dtype="float32")
                except (ValueError, TypeError, IndexError):
                    # Skip this rule if value cannot be converted to float
                    continue
            else:
                # For string values, try to convert to float
                try:
                    value_t = convert_to_tensor(float(value), dtype="float32")
                except (ValueError, TypeError):
                    # Skip this rule if value cannot be converted to float
                    continue

            if op == ">":
                violation = ops.less(x, value_t)
                score = ops.maximum(value_t - x, ops.zeros_like(x))
                reason = convert_to_tensor(
                    f"Value below minimum ({value})",
                    dtype="string",
                )
            elif op == "<":
                violation = ops.greater_equal(x, value_t)
                score = ops.maximum(x - value_t, ops.zeros_like(x))
                reason = convert_to_tensor(
                    f"Value above maximum ({value})",
                    dtype="string",
                )
            else:
                raise ValueError(f"Unsupported operator: {op}")

            # Apply weight to violation and score
            violation = ops.cast(violation, "float32") * weight
            score = score * weight

            violations.append(violation)
            scores.append(score)
            reasons.append(reason)

        # Combine results
        final_violation = ops.cast(
            ops.any(ops.stack(violations, axis=0), axis=0),
            "bool",
        )
        scores_stacked = ops.stack(scores, axis=0)
        final_score = ops.max(scores_stacked, axis=0)
        final_proba = ops.mean(ops.stack(violations, axis=0), axis=0) * 100.0

        # Select reason for highest scoring violation
        max_score_idx = ops.argmax(ops.stack(scores, axis=0), axis=0)
        final_reason = ops.take(convert_to_tensor(reasons), max_score_idx)

        return {
            "business_score": final_score,
            "business_proba": final_proba,
            "business_anomaly": final_violation,
            "business_reason": final_reason,
            "business_value": x,
        }

    def _apply_categorical_rules(self, x: KerasTensor) -> dict[str, KerasTensor]:
        """Apply categorical rules to input tensor.

        Args:
            x: Input tensor.

        Returns:
            dictionary with rule evaluation results.
        """
        x = ops.cast(x, "string")
        violations = []
        reasons = []

        for (op, values), weight in zip(self.rules, self.rule_weights, strict=False):
            # Ensure values is a list of strings
            values = (
                [str(values)]
                if not isinstance(values, list)
                else [str(v) for v in values]
            )

            # Convert values to tensor
            values_t = convert_to_tensor(values, dtype="string")

            # Check membership
            eq = ops.equal(x, ops.expand_dims(values_t, 0))
            is_member = ops.any(eq, axis=1, keepdims=True)

            if op in ("==", "in"):
                violation = ops.logical_not(is_member)
                reason = convert_to_tensor(
                    f"Value not in allowed set: {values}",
                    dtype="string",
                )
            elif op in ("!=", "not in"):
                violation = is_member
                reason = convert_to_tensor(
                    f"Value in disallowed set: {values}",
                    dtype="string",
                )
            else:
                raise ValueError(f"Unsupported operator: {op}")
            # Apply weight to violation
            violation = ops.cast(violation, "float32") * weight

            violations.append(violation)
            reasons.append(reason)

        # Combine results
        final_violation = ops.cast(
            ops.any(ops.stack(violations, axis=0), axis=0),
            "bool",
        )
        final_proba = ops.mean(ops.stack(violations, axis=0), axis=0) * 100.0

        # Use violation probability as score for categorical features
        final_score = final_proba / 100.0

        # Select reason for first violation
        violations_stacked = ops.stack(violations, axis=0)
        violation_idx = ops.argmax(violations_stacked, axis=0)
        final_reason = ops.take(convert_to_tensor(reasons), violation_idx)

        return {
            "business_score": final_score,
            "business_proba": final_proba,
            "business_anomaly": final_violation,
            "business_reason": final_reason,
            "business_value": x,
        }

    def get_config(self) -> dict[str, Any]:
        """Return the config of the layer.

        Returns:
            Layer configuration dictionary.
        """
        config = super().get_config()

        # Use private attributes during initialization, public attributes after
        rules = getattr(self, "rules", getattr(self, "_rules", None))
        feature_type = getattr(
            self,
            "feature_type",
            getattr(self, "_feature_type", None),
        )
        weights_trainable = getattr(
            self,
            "weights_trainable",
            getattr(self, "_weights_trainable", None),
        )
        weight_initializer = getattr(
            self,
            "weight_initializer",
            getattr(self, "_weight_initializer", None),
        )

        config.update(
            {
                "rules": rules,
                "feature_type": feature_type,
                "trainable_weights": weights_trainable,
                "weight_initializer": initializers.serialize(weight_initializer)
                if weight_initializer
                else None,
            },
        )
        return config
