# """Unit tests for the TextPreprocessingLayer.

# Note: TensorFlow is used in tests for validation purposes only.
# The actual layer implementation uses only Keras 3 operations.
# """

# import unittest
# import tensorflow as tf  # Used for testing only
# from keras import Model, layers
# from kerasfactory.layers.TextPreprocessingLayer import TextPreprocessingLayer

# class TestTextPreprocessingLayer(unittest.TestCase):
#     """Test cases for the TextPreprocessingLayer."""

#     def setUp(self) -> None:
#         """Set up test fixtures."""
#         # Using TensorFlow for test data generation only
#         self.test_texts = tf.constant([
#             "Hello, world!",
#             "This is a test.",
#             "STOP WORDS should be removed.",
#             "Multiple   spaces   should be normalized."
#         ])
#         self.stop_words = ["is", "a", "be"]
#         tf.random.set_seed(42)  # For reproducibility

#     def test_initialization(self) -> None:
#         """Test layer initialization with various parameters."""
#         # Test default initialization
#         layer = TextPreprocessingLayer(stop_words=self.stop_words)
#         self.assertEqual(layer.stop_words, self.stop_words)

#         # Test with empty stop words list
#         empty_stop_words = []
#         layer = TextPreprocessingLayer(stop_words=empty_stop_words)
#         self.assertEqual(layer.stop_words, empty_stop_words)

#     def test_invalid_initialization(self) -> None:
#         """Test layer initialization with invalid parameters."""
#         # Test with non-list stop_words
#         with self.assertRaises(ValueError):
#             TextPreprocessingLayer(stop_words="not a list")

#     def test_lowercase_conversion(self) -> None:
#         """Test that text is properly converted to lowercase."""
#         layer = TextPreprocessingLayer(stop_words=[])
#         output = layer(self.test_texts)

#         # Check that all text is lowercase
#         for i in range(len(self.test_texts)):
#             self.assertEqual(
#                 output[i].numpy().decode('utf-8'),
#                 output[i].numpy().decode('utf-8').lower()
#             )

#     def test_punctuation_removal(self) -> None:
#         """Test that punctuation is properly removed."""
#         layer = TextPreprocessingLayer(stop_words=[])
#         output = layer(self.test_texts)

#         # Check that common punctuation is removed
#         for i in range(len(self.test_texts)):
#             text = output[i].numpy().decode('utf-8')
#             self.assertNotIn(',', text)
#             self.assertNotIn('.', text)
#             self.assertNotIn('!', text)

#     def test_stop_word_removal(self) -> None:
#         """Test that stop words are properly removed."""
#         layer = TextPreprocessingLayer(stop_words=self.stop_words)
#         output = layer(self.test_texts)

#         # Check that stop words are removed
#         for i in range(len(self.test_texts)):
#             text = output[i].numpy().decode('utf-8')
#             for stop_word in self.stop_words:
#                 # Check that stop word is not present as a whole word
#                 self.assertNotRegex(text, r'\b' + stop_word + r'\b')

#     def test_whitespace_normalization(self) -> None:
#         """Test that whitespace is properly normalized."""
#         layer = TextPreprocessingLayer(stop_words=[])
#         output = layer(self.test_texts)

#         # Check that multiple spaces are normalized
#         for i in range(len(self.test_texts)):
#             text = output[i].numpy().decode('utf-8')
#             self.assertNotRegex(text, r'\s{2,}')  # No consecutive spaces

#     def test_output_shape(self) -> None:
#         """Test that output shape matches input shape."""
#         layer = TextPreprocessingLayer(stop_words=self.stop_words)
#         output = layer(self.test_texts)

#         # Check that output shape matches input shape
#         self.assertEqual(output.shape, self.test_texts.shape)

#     def test_serialization(self) -> None:
#         """Test layer serialization and deserialization."""
#         original_layer = TextPreprocessingLayer(stop_words=self.stop_words)
#         config = original_layer.get_config()

#         # Create new layer from config
#         restored_layer = TextPreprocessingLayer.from_config(config)

#         # Check if configurations match
#         self.assertEqual(restored_layer.stop_words, original_layer.stop_words)

#         # Check that outputs match
#         original_output = original_layer(self.test_texts)
#         restored_output = restored_layer(self.test_texts)

#         for i in range(len(self.test_texts)):
#             self.assertEqual(
#                 original_output[i].numpy().decode('utf-8'),
#                 restored_output[i].numpy().decode('utf-8')
#             )

#     def test_integration(self) -> None:
#         """Test integration with a simple model."""
#         # Create a simple model with the preprocessing layer
#         inputs = layers.Input(shape=(), dtype="string")
#         x = TextPreprocessingLayer(stop_words=self.stop_words)(inputs)

#         # Add a text vectorization layer
#         vectorize = layers.TextVectorization(max_tokens=1000)(x)

#         # Add a dense layer for classification
#         outputs = layers.Dense(1, activation="sigmoid")(vectorize)

#         model = Model(inputs=inputs, outputs=outputs)

#         # Compile the model
#         model.compile(optimizer="adam", loss="binary_crossentropy")

#         # Generate some dummy data
#         x_data = tf.constant(["Sample text one", "Sample text two", "Another example"])
#         y_data = tf.constant([0, 1, 0])

#         # Fit the model for one step to ensure everything works
#         model.fit(x_data, y_data, epochs=1, verbose=0)

#         # Make a prediction to ensure the pipeline works
#         pred = model.predict(tf.constant(["Test prediction"]))
#         self.assertEqual(pred.shape, (1, 1))

# if __name__ == "__main__":
#     unittest.main()
