# """
# This module implements a TextPreprocessingLayer that performs basic text preprocessing
# operations such as lowercasing, punctuation removal, and stop word removal.
# """

# import re
# import string
# from typing import Any, Dict, List, Optional, Union

# from loguru import logger
# import tensorflow as tf  # Used for tensor operations
# from keras import ops
# from keras import KerasTensor
# from keras.saving import register_keras_serializable
# from kmr.layers._base_layer import BaseLayer

# @register_keras_serializable(package="kmr.layers")
# class TextPreprocessingLayer(BaseLayer):
#     """Layer for basic text preprocessing.

#     This layer performs basic text preprocessing operations:
#     - Converting text to lowercase
#     - Removing punctuation
#     - Removing stop words
#     - Normalizing whitespace

#     Args:
#         stop_words: List of stop words to remove. Default is None.
#         **kwargs: Additional keyword arguments passed to the parent class.

#     Input shape:
#         A string tensor of shape (batch_size,) or (batch_size, 1)

#     Output shape:
#         A string tensor of the same shape as the input.

#     Example:
#         ```python
#         # Create a text preprocessing layer
#         text_layer = TextPreprocessingLayer(stop_words=["a", "the", "is"])

#         # Process a batch of text
#         text_data = tf.constant(["Hello, World!", "This is a test."])
#         processed_text = text_layer(text_data)
#         ```
#     """

#     def __init__(
#         self, stop_words: Optional[List[str]] = None, **kwargs: Any
#     ) -> None:
#         """Initialize the TextPreprocessingLayer."""
#         super().__init__(**kwargs)
#         self.stop_words = stop_words or []
#         self._validate_params()

#         # Create a regex pattern for punctuation
#         self.punctuation_pattern = "[" + re.escape(string.punctuation) + "]"

#         # Create a regex pattern for stop words
#         if self.stop_words:
#             # Escape special regex characters and join with word boundaries
#             escaped_words = [re.escape(word) for word in self.stop_words]
#             self.stop_words_pattern = r"\b(" + "|".join(escaped_words) + r")\b"
#         else:
#             self.stop_words_pattern = None

#     def _validate_params(self) -> None:
#         """Validate the parameters."""
#         if not isinstance(self.stop_words, list):
#             raise ValueError(
#                 f"stop_words must be a list, got {type(self.stop_words)}"
#             )

#     def _preprocess_text(self, text_input: tf.Tensor) -> tf.Tensor:
#         """Preprocess a single text string.

#         Args:
#             text_input: A string tensor.

#         Returns:
#             A preprocessed string tensor.
#         """
#         # Convert to string if it's bytes
#         if (isinstance(text_input, bytes) or tf.executing_eagerly() and
#             hasattr(text_input, 'numpy') and isinstance(text_input.numpy(), bytes)):
#             text = text_input.decode('utf-8')
#         else:
#             text = text_input.decode('utf-8') if isinstance(text_input, bytes) else str(text_input)

#         # Convert to lowercase
#         text = text.lower()

#         # Remove punctuation
#         text = re.sub(self.punctuation_pattern, " ", text)

#         # Remove stop words if any
#         if self.stop_words_pattern:
#             text = re.sub(self.stop_words_pattern, "", text)

#         # Normalize whitespace (replace multiple spaces with a single space)
#         text = re.sub(r"\s+", " ", text)

#         # Trim leading and trailing spaces
#         text = text.strip()

#         return text

#     def call(self, inputs: tf.Tensor) -> tf.Tensor:
#         """Process the input tensor.

#         Args:
#             inputs: A string tensor of shape (batch_size,) or (batch_size, 1).

#         Returns:
#             A string tensor of the same shape as the input.
#         """
#         # Handle scalar inputs
#         input_shape = tf.shape(inputs)
#         is_scalar = tf.equal(tf.rank(inputs), 0)

#         # If scalar, expand to rank 1
#         if is_scalar:
#             inputs = tf.expand_dims(inputs, 0)

#         # Use tf.map_fn to apply the preprocessing function to each element
#         processed_texts = tf.map_fn(
#             lambda x: tf.py_function(
#                 self._preprocess_text,
#                 [x],
#                 tf.string
#             ),
#             inputs,
#             fn_output_signature=tf.string
#         )

#         # Restore original shape if input was scalar
#         if is_scalar:
#             processed_texts = tf.squeeze(processed_texts, 0)

#         return processed_texts

#     def compute_output_shape(self, input_shape: tuple) -> tuple:
#         """Return the output shape of the layer.

#         Args:
#             input_shape: Shape of the input.

#         Returns:
#             Shape of the output.
#         """
#         return input_shape

#     def get_config(self) -> Dict[str, Any]:
#         """Return the configuration of the layer.

#         Returns:
#             A dictionary containing the configuration of the layer.
#         """
#         config = super().get_config()
#         config.update({"stop_words": self.stop_words})
#         return config
