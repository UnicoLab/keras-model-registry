"""This module implements a DistributionTransformLayer that applies various transformations
to make data more normally distributed or to handle specific distribution types better.
It's particularly useful for preprocessing data before anomaly detection or other statistical analyses.
"""

from typing import Any
from loguru import logger
from keras import ops, KerasTensor
from keras.saving import register_keras_serializable
from keras import backend
from kerasfactory.layers._base_layer import BaseLayer


@register_keras_serializable(package="kerasfactory.layers")
class DistributionTransformLayer(BaseLayer):
    """Layer for transforming data distributions to improve anomaly detection.

    This layer applies various transformations to make data more normally distributed
    or to handle specific distribution types better. Supported transformations include
    log, square root, Box-Cox, Yeo-Johnson, arcsinh, cube-root, logit, quantile,
    robust-scale, and min-max.

    When transform_type is set to 'auto', the layer automatically selects the most
    appropriate transformation based on the data characteristics during training.

    Args:
        transform_type: Type of transformation to apply. Options are 'none', 'log', 'sqrt',
            'box-cox', 'yeo-johnson', 'arcsinh', 'cube-root', 'logit', 'quantile',
            'robust-scale', 'min-max', or 'auto'. Default is 'none'.
        lambda_param: Parameter for parameterized transformations like Box-Cox and Yeo-Johnson.
            Default is 0.0.
        epsilon: Small value added to prevent numerical issues like log(0). Default is 1e-10.
        min_value: Minimum value for min-max scaling. Default is 0.0.
        max_value: Maximum value for min-max scaling. Default is 1.0.
        clip_values: Whether to clip values to the specified range in min-max scaling. Default is True.
        auto_candidates: list of transformation types to consider when transform_type is 'auto'.
            If None, all available transformations will be considered. Default is None.
        name: Optional name for the layer.

    Input shape:
        N-D tensor with shape: (batch_size, ..., features)

    Output shape:
        Same shape as input: (batch_size, ..., features)

    Example:
        ```python
        import keras
        import numpy as np
        from kerasfactory.layers import DistributionTransformLayer

        # Create sample input data with skewed distribution
        x = keras.random.exponential((32, 10))  # 32 samples, 10 features

        # Apply log transformation
        log_transform = DistributionTransformLayer(transform_type="log")
        y = log_transform(x)
        print("Transformed output shape:", y.shape)  # (32, 10)

        # Apply Box-Cox transformation with lambda=0.5
        box_cox = DistributionTransformLayer(transform_type="box-cox", lambda_param=0.5)
        z = box_cox(x)

        # Apply arcsinh transformation (handles both positive and negative values)
        arcsinh_transform = DistributionTransformLayer(transform_type="arcsinh")
        a = arcsinh_transform(x)

        # Apply min-max scaling to range [0, 1]
        min_max = DistributionTransformLayer(transform_type="min-max", min_value=0.0, max_value=1.0)
        b = min_max(x)

        # Use automatic transformation selection
        auto_transform = DistributionTransformLayer(transform_type="auto")
        c = auto_transform(x)  # Will select the best transformation during training
        ```
    """

    def __init__(
        self,
        transform_type: str = "none",
        lambda_param: float = 0.0,
        epsilon: float = 1e-10,
        min_value: float = 0.0,
        max_value: float = 1.0,
        clip_values: bool = True,
        auto_candidates: list[str] | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the DistributionTransformLayer.

        Args:
            transform_type: Type of transformation to apply.
            lambda_param: Lambda parameter for Box-Cox transformation.
            epsilon: Small value to avoid division by zero.
            min_value: Minimum value for clipping.
            max_value: Maximum value for clipping.
            clip_values: Whether to clip values.
            auto_candidates: List of candidate transformations for auto mode.
            name: Name of the layer.
            **kwargs: Additional keyword arguments.
        """
        # Set private attributes first
        self._transform_type = transform_type
        self._lambda_param = lambda_param
        self._epsilon = epsilon
        self._min_value = min_value
        self._max_value = max_value
        self._clip_values = clip_values
        self._auto_candidates = auto_candidates

        # Set public attributes BEFORE calling parent's __init__
        self.transform_type = self._transform_type
        self.lambda_param = self._lambda_param
        self.epsilon = self._epsilon
        self.min_value = self._min_value
        self.max_value = self._max_value
        self.clip_values = self._clip_values
        self.auto_candidates = self._auto_candidates

        # Define valid transformations
        self._valid_transforms = [
            "none",
            "log",
            "sqrt",
            "box-cox",
            "yeo-johnson",
            "arcsinh",
            "cube-root",
            "logit",
            "quantile",
            "robust-scale",
            "min-max",
            "auto",
        ]

        # Set default auto candidates if not provided
        if self.auto_candidates is None and self.transform_type == "auto":
            # Exclude 'none' and 'auto' from candidates
            self.auto_candidates = [
                t for t in self._valid_transforms if t not in ["none", "auto"]
            ]

        # Validate parameters
        self._validate_params()

        # Initialize auto-mode variables
        self._selected_transform = None
        self._is_initialized = False

        # Call parent's __init__
        super().__init__(name=name, **kwargs)

    def _validate_params(self) -> None:
        """Validate layer parameters."""
        if self.transform_type not in self._valid_transforms:
            raise ValueError(
                f"transform_type must be one of {self._valid_transforms}, got {self.transform_type}",
            )

        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {self.epsilon}")

        if self.min_value >= self.max_value:
            raise ValueError(
                f"min_value must be less than max_value, got min_value={self.min_value}, max_value={self.max_value}",
            )

        if self.transform_type == "auto" and self.auto_candidates:
            for candidate in self.auto_candidates:
                if candidate not in self._valid_transforms or candidate in [
                    "auto",
                    "none",
                ]:
                    raise ValueError(
                        f"Invalid transformation candidate: {candidate}. "
                        f"Candidates must be valid transformations excluding 'auto' and 'none'.",
                    )

    def build(self, input_shape: tuple[int, ...]) -> None:
        """Builds the layer with the given input shape.

        Args:
            input_shape: tuple of integers defining the input shape.
        """
        # For auto mode, we need to store the selected transformation
        if self.transform_type == "auto":
            # Create a variable to store the selected transformation index
            self._selected_transform_idx = self.add_weight(
                name="selected_transform_idx",
                shape=(1,),
                dtype="int32",
                trainable=False,
                initializer="zeros",
            )

            # Create a variable to store the selected lambda parameter
            self._selected_lambda = self.add_weight(
                name="selected_lambda",
                shape=(1,),
                dtype="float32",
                trainable=False,
                initializer="zeros",
            )

        logger.debug(
            f"DistributionTransformLayer built with transform_type={self.transform_type}, "
            f"lambda_param={self.lambda_param}",
        )
        super().build(input_shape)

    def _compute_statistics(
        self,
        x: KerasTensor,
    ) -> tuple:
        """Compute statistics for the input tensor.

        Args:
            x: Input tensor

        Returns:
            Tuple containing (min, max, median, interquartile_range) tensors
        """
        # Compute min and max along each feature dimension
        x_min = ops.min(x, axis=0, keepdims=True)
        x_max = ops.max(x, axis=0, keepdims=True)

        # For median and IQR, we need to sort the values
        # This is an approximation since Keras doesn't have direct percentile functions
        x_sorted = ops.sort(x, axis=0)
        n = ops.shape(x)[0]

        # Compute median (50th percentile)
        median_idx = n // 2
        median = (
            (x_sorted[median_idx - 1] + x_sorted[median_idx]) / 2.0
            if n % 2 == 0
            else x_sorted[median_idx]
        )

        # Compute 25th and 75th percentiles for IQR
        q1_idx = n // 4
        q3_idx = (3 * n) // 4
        q1 = x_sorted[q1_idx]
        q3 = x_sorted[q3_idx]

        # Compute IQR
        iqr = q3 - q1

        # Add small epsilon to IQR to avoid division by zero
        iqr = ops.maximum(iqr, self.epsilon)

        return x_min, x_max, median, iqr

    def _calculate_skewness(self, x: KerasTensor) -> KerasTensor:
        """Calculate the skewness of the input tensor.

        Args:
            x: Input tensor

        Returns:
            Skewness value
        """
        # Calculate mean and standard deviation
        mean = ops.mean(x, axis=0, keepdims=True)
        std = ops.std(x, axis=0, keepdims=True)

        # Add epsilon to std to avoid division by zero
        std = ops.maximum(std, self.epsilon)

        # Calculate skewness
        skewness = ops.mean(ops.power((x - mean) / std, 3.0), axis=0)

        # Return absolute value of skewness
        return ops.abs(skewness)

    def _calculate_kurtosis(self, x: KerasTensor) -> KerasTensor:
        """Calculate the kurtosis of the input tensor.

        Args:
            x: Input tensor

        Returns:
            Kurtosis value
        """
        # Calculate mean and standard deviation
        mean = ops.mean(x, axis=0, keepdims=True)
        std = ops.std(x, axis=0, keepdims=True)

        # Add epsilon to std to avoid division by zero
        std = ops.maximum(std, self.epsilon)

        # Calculate kurtosis
        kurtosis = ops.mean(ops.power((x - mean) / std, 4.0), axis=0) - 3.0

        # Return absolute value of excess kurtosis
        return ops.abs(kurtosis)

    def _has_negative_values(self, x: KerasTensor) -> bool:
        """Check if the input tensor has negative values.

        Args:
            x: Input tensor

        Returns:
            Boolean indicating if the tensor has negative values
        """
        # For graph mode compatibility, return a tensor
        return ops.any(x < 0)

    def _has_zeros(self, x: KerasTensor) -> bool:
        """Check if the input tensor has zero values.

        Args:
            x: Input tensor

        Returns:
            Boolean indicating if the tensor has zero values
        """
        # For graph mode compatibility, return a tensor
        return ops.any(ops.abs(x) < self.epsilon)

    def _is_bounded_01(self, x: KerasTensor) -> bool:
        """Check if the input tensor is bounded in (0, 1).

        Args:
            x: Input tensor

        Returns:
            Boolean indicating if the tensor is bounded in (0, 1)
        """
        # In symbolic mode, we can't use ops.all directly as a boolean
        # Instead, we'll check if min > 0 and max < 1
        min_val = ops.min(x)
        max_val = ops.max(x)

        # For graph mode compatibility, return a tensor
        min_gt_zero = min_val > 0
        max_lt_one = max_val < 1

        return min_gt_zero & max_lt_one  # Using & operator for element-wise logical AND

    def _select_best_transformation(self, x: KerasTensor) -> tuple:
        """Select the best transformation based on data characteristics.

        Args:
            x: Input tensor

        Returns:
            tuple of (selected_transform_type, selected_lambda_param)
        """
        # If transform_type is not "auto", just return it
        if self.transform_type != "auto":
            return self.transform_type, self.lambda_param

        # Calculate data characteristics
        skewness = self._calculate_skewness(x)
        kurtosis = self._calculate_kurtosis(x)

        # Average skewness and kurtosis across features
        avg_skewness = ops.mean(skewness)
        avg_kurtosis = ops.mean(kurtosis)

        # Check for negative values and zeros
        has_negative = self._has_negative_values(x)
        has_zeros = self._has_zeros(x)
        is_bounded_01 = self._is_bounded_01(x)

        # Default transformation
        transform_type = "none"
        lambda_param = 0.0

        # If we have specific candidates, filter based on them
        if self.auto_candidates:
            candidates = self.auto_candidates
        else:
            candidates = []

            # For bounded data in (0, 1)
            if is_bounded_01:
                candidates.extend(["logit", "arcsinh", "min-max", "quantile"])

            # For positive data with zeros
            elif not has_negative and has_zeros:
                candidates.extend(["sqrt", "cube-root", "arcsinh"])

            # For strictly positive data (no zeros)
            elif not has_negative and not has_zeros:
                candidates.extend(["log", "sqrt", "box-cox", "arcsinh", "cube-root"])

            # For mixed positive and negative data
            else:
                candidates.extend(
                    ["yeo-johnson", "arcsinh", "cube-root", "robust-scale", "quantile"],
                )

            # Add general transformations that work for most data
            candidates.extend(["arcsinh", "yeo-johnson", "robust-scale", "quantile"])

            # Remove duplicates while preserving order
            candidates = list(dict.fromkeys(candidates))

        # Simple heuristic for transformation selection based on skewness and kurtosis
        abs_skewness = ops.abs(avg_skewness)
        abs_kurtosis = ops.abs(avg_kurtosis)

        # Select transformation based on data characteristics
        if is_bounded_01:
            # For bounded data in (0, 1)
            transform_type = "logit" if abs_skewness > 0.5 else "min-max"
        elif not has_negative:
            # For positive data
            if abs_skewness > 1.0:
                # Highly skewed positive data
                transform_type = "arcsinh" if has_zeros else "log"
            elif abs_skewness > 0.5:
                # Moderately skewed positive data
                transform_type = "sqrt"
            else:
                # Mildly skewed or symmetric positive data
                transform_type = "arcsinh"
        else:
            # For mixed positive and negative data
            if abs_skewness > 1.0 or abs_kurtosis > 3.0:
                # Highly skewed or heavy-tailed mixed data
                transform_type = "yeo-johnson"
            elif abs_skewness > 0.5:
                # Moderately skewed mixed data
                transform_type = "arcsinh"
            else:
                # Mildly skewed or symmetric mixed data
                transform_type = "cube-root"

        # Ensure the selected transformation is in the candidates list
        if candidates and transform_type not in candidates:
            # If not, select the first candidate
            transform_type = candidates[0]

        # Never select "none" for bounded data in (0, 1)
        if is_bounded_01 and transform_type == "none":
            # Choose from preferred transformations for bounded data
            bounded_transforms = ["logit", "arcsinh", "min-max"]
            # Filter by available candidates if specified
            if candidates:
                bounded_transforms = [t for t in bounded_transforms if t in candidates]
            if bounded_transforms:
                transform_type = bounded_transforms[0]

        # For positive skewed data, ensure we select from appropriate transformations
        if not has_negative and abs_skewness > 0.5 and transform_type == "none":
            # Choose from preferred transformations for positive skewed data
            pos_skewed_transforms = ["log", "sqrt", "box-cox", "arcsinh", "cube-root"]
            # Filter by available candidates if specified
            if candidates:
                pos_skewed_transforms = [
                    t for t in pos_skewed_transforms if t in candidates
                ]
            if pos_skewed_transforms:
                transform_type = pos_skewed_transforms[0]

        # For mixed data, ensure we select from appropriate transformations
        if has_negative and transform_type == "none":
            # Choose from preferred transformations for mixed data
            mixed_transforms = ["yeo-johnson", "arcsinh", "cube-root"]
            # Filter by available candidates if specified
            if candidates:
                mixed_transforms = [t for t in mixed_transforms if t in candidates]
            if mixed_transforms:
                transform_type = mixed_transforms[0]

        # Set lambda parameter based on transformation type and skewness direction
        if transform_type == "box-cox" or transform_type == "yeo-johnson":
            # For positive skew, lambda=0 (log-like)
            # For negative skew, lambda=2 (square-like)
            lambda_param = 0.0 if avg_skewness > 0 else 2.0

        return transform_type, lambda_param

    def _apply_transform(self, x: KerasTensor) -> KerasTensor:
        """Apply the selected transformation to the input tensor.

        Args:
            x: Input tensor

        Returns:
            Transformed tensor
        """
        try:
            if self.transform_type == "none":
                return x

            elif self.transform_type == "log":
                # Add epsilon to avoid log(0)
                return ops.log(x + self.epsilon)

            elif self.transform_type == "sqrt":
                # Ensure non-negative values
                return ops.sqrt(ops.maximum(x, 0.0) + self.epsilon)

            elif self.transform_type == "box-cox":
                # Box-Cox transformation: (x^lambda - 1)/lambda if lambda != 0, log(x) if lambda == 0
                # Ensure x is positive
                x_pos = ops.maximum(x, self.epsilon)

                if abs(self.lambda_param) < self.epsilon:
                    # For lambda ≈ 0, use log transform
                    return ops.log(x_pos)
                else:
                    # Standard Box-Cox formula using power
                    powered = ops.power(x_pos, self.lambda_param)
                    return (powered - 1.0) / self.lambda_param

            elif self.transform_type == "yeo-johnson":
                # Yeo-Johnson works for both positive and negative values
                lambda_p = self.lambda_param

                # Create masks for positive and negative values
                pos_mask = x >= 0
                neg_mask = x < 0

                # Initialize result tensor with zeros
                result = ops.zeros_like(x)

                # Handle positive values
                if abs(lambda_p) < self.epsilon:
                    # For lambda ≈ 0 and x ≥ 0
                    pos_values = ops.log(x + 1.0)
                else:
                    # For other lambda and x ≥ 0
                    pos_values = (ops.power(x + 1.0, lambda_p) - 1.0) / lambda_p

                # Apply positive values where mask is True
                result = ops.where(pos_mask, pos_values, result)

                # Handle negative values
                if abs(lambda_p - 2.0) < self.epsilon:
                    # For lambda ≈ 2 and x < 0
                    neg_values = -ops.log(-x + 1.0)
                else:
                    # For other lambda and x < 0
                    neg_values = -(
                        (ops.power(-x + 1.0, 2.0 - lambda_p) - 1.0) / (2.0 - lambda_p)
                    )

                # Apply negative values where mask is True
                result = ops.where(neg_mask, neg_values, result)

                return result

            elif self.transform_type == "arcsinh":
                # Inverse hyperbolic sine transformation
                # Works well for both positive and negative values with heavy tails
                # Using the formula: asinh(x) = log(x + sqrt(x^2 + 1))
                # This handles both positive and negative values
                return ops.log(x + ops.sqrt(ops.square(x) + 1.0))

            elif self.transform_type == "cube-root":
                # Cube root transformation
                # Works well for both positive and negative values
                # For negative values, we take the negative cube root
                pos_mask = x >= 0
                neg_mask = x < 0

                # Initialize result tensor with zeros
                result = ops.zeros_like(x)

                # Handle positive values
                pos_values = ops.power(x + self.epsilon, 1.0 / 3.0)
                result = ops.where(pos_mask, pos_values, result)

                # Handle negative values
                neg_values = -ops.power(-x + self.epsilon, 1.0 / 3.0)
                result = ops.where(neg_mask, neg_values, result)

                return result

            elif self.transform_type == "logit":
                # Logit transformation: log(x / (1 - x))
                # Useful for data in the range (0, 1)
                # Clip values to (epsilon, 1-epsilon) to avoid numerical issues
                # Use a larger epsilon for logit to prevent extreme values
                safe_epsilon = ops.maximum(self.epsilon, 1e-5)
                x_clipped = ops.clip(x, safe_epsilon, 1.0 - safe_epsilon)
                return ops.log(x_clipped / (1.0 - x_clipped))

            elif self.transform_type == "quantile":
                # Approximate quantile transformation to normal distribution
                # This is a simplified version that uses the empirical CDF
                # Sort the values along each feature dimension
                x_sorted = ops.sort(x, axis=0)
                n = ops.shape(x)[0]

                # Compute ranks for each value
                # This is an approximation since Keras doesn't have direct rank functions
                # We'll use a loop to find the rank of each value
                ranks = ops.zeros_like(x)

                # For each value in x, find its position in the sorted array
                for i in range(n):
                    # Create a mask where x equals the current value
                    mask = ops.equal(x, x_sorted[i : i + 1])
                    # Set the rank for those positions
                    ranks = ops.where(mask, ops.ones_like(x) * (i + 0.5), ranks)

                # Convert ranks to quantiles in (0, 1)
                quantiles = ranks / n

                # Apply inverse normal CDF (approximation)
                # We use a simple approximation: quantile * 6 - 3
                # This maps [0, 1] to approximately [-3, 3], covering most of the normal distribution
                return quantiles * 6.0 - 3.0

            elif self.transform_type == "robust-scale":
                # Robust scaling using median and IQR
                # (x - median) / IQR
                _, _, median, iqr = self._compute_statistics(x)
                return (x - median) / iqr

            elif self.transform_type == "min-max":
                # Min-max scaling to [min_value, max_value]
                x_min, x_max, _, _ = self._compute_statistics(x)

                # Avoid division by zero
                denominator = ops.maximum(x_max - x_min, self.epsilon)

                # Scale to [0, 1]
                scaled = (x - x_min) / denominator

                # Scale to [min_value, max_value]
                scaled = scaled * (self.max_value - self.min_value) + self.min_value

                # Optionally clip values to the specified range
                if self.clip_values:
                    scaled = ops.clip(scaled, self.min_value, self.max_value)

                return scaled

            else:
                # This should never happen due to validation in __init__
                raise ValueError(f"Unknown transformation type: {self.transform_type}")

        except Exception as e:
            # Add more context to the error
            raise type(e)(
                f"Error in {self.transform_type} transformation: {str(e)}",
            ) from e

    def call(self, inputs: KerasTensor, training: bool | None = None) -> KerasTensor:
        """Apply the selected transformation to the inputs.

        Args:
            inputs: Input tensor
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode

        Returns:
            Transformed tensor with the same shape as input
        """
        # Ensure inputs are cast to float32
        x = ops.cast(inputs, dtype="float32")

        # Handle auto transformation mode
        if self.transform_type == "auto":
            if training or not self._is_initialized:
                # During training or first call, select the best transformation
                best_transform, best_lambda = self._select_best_transformation(x)

                # Store the selected transformation
                transform_idx = self._valid_transforms.index(best_transform)

                # Use assign operations to update the variables
                if hasattr(self, "_selected_transform_idx"):
                    # Use the variable's assign method directly
                    self._selected_transform_idx.assign(ops.array([transform_idx]))
                    self._selected_lambda.assign(ops.array([best_lambda]))

                # Set the transformation type and lambda for this forward pass
                temp_transform_type = self.transform_type
                temp_lambda_param = self.lambda_param

                self.transform_type = best_transform
                self.lambda_param = best_lambda

                # Apply the transformation
                result = self._apply_transform(x)

                # Restore original values
                self.transform_type = temp_transform_type
                self.lambda_param = temp_lambda_param

                # Mark as initialized
                self._is_initialized = True

                return result
            else:
                # During inference, use the stored transformation
                # Get the transformation index in a way that works in both eager and graph mode
                if backend.backend() == "tensorflow":
                    # For TensorFlow backend, we can use numpy() safely
                    transform_idx = int(self._selected_transform_idx.numpy()[0])
                    lambda_param = float(self._selected_lambda.numpy()[0])
                else:
                    # For other backends, we need to handle symbolic tensors differently
                    # Use the first element of the variable directly
                    transform_idx = (
                        int(self._selected_transform_idx[0])
                        if not ops.is_tensor(self._selected_transform_idx)
                        else 0
                    )
                    lambda_param = (
                        float(self._selected_lambda[0])
                        if not ops.is_tensor(self._selected_lambda)
                        else 0.0
                    )

                # Set the transformation type and lambda for this forward pass
                temp_transform_type = self.transform_type
                temp_lambda_param = self.lambda_param

                self.transform_type = self._valid_transforms[transform_idx]
                self.lambda_param = lambda_param

                # Apply the transformation
                result = self._apply_transform(x)

                # Restore original values
                self.transform_type = temp_transform_type
                self.lambda_param = temp_lambda_param

                return result
        else:
            # For non-auto modes, just apply the transformation
            return self._apply_transform(x)

    def get_config(self) -> dict[str, Any]:
        """Returns the config of the layer.

        Returns:
            Python dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update(
            {
                "transform_type": self.transform_type,
                "lambda_param": self.lambda_param,
                "epsilon": self.epsilon,
                "min_value": self.min_value,
                "max_value": self.max_value,
                "clip_values": self.clip_values,
                "auto_candidates": self.auto_candidates,
            },
        )
        return config
