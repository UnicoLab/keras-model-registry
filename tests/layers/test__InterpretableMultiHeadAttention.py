"""Unit tests for InterpretableMultiHeadAttention layer."""
import unittest
import keras
import tensorflow as tf
from keras import layers, Model, ops

from kmr.layers.InterpretableMultiHeadAttention import InterpretableMultiHeadAttention


class TestInterpretableMultiHeadAttention(unittest.TestCase):
    """Test cases for InterpretableMultiHeadAttention layer."""

    def setUp(self) -> None:
        """Set up test cases."""
        self.d_model = 64
        self.n_head = 4
        self.seq_len = 10
        self.batch_size = 32
        self.dropout_rate = 0.1
        self.layer = InterpretableMultiHeadAttention(
            d_model=self.d_model,
            n_head=self.n_head,
            dropout_rate=self.dropout_rate
        )

    def test_initialization(self) -> None:
        """Test layer initialization."""
        self.assertEqual(self.layer.mha.num_heads, self.n_head)
        self.assertEqual(self.layer.mha.key_dim, self.d_model)
        self.assertEqual(self.layer.mha._dropout, self.dropout_rate)

    def test_output_shape(self) -> None:
        """Test output shape of the layer."""
        # Create input tensors
        query = tf.random.normal((self.batch_size, self.seq_len, self.d_model))
        key = tf.random.normal((self.batch_size, self.seq_len, self.d_model))
        value = tf.random.normal((self.batch_size, self.seq_len, self.d_model))

        # Get output
        output = self.layer(query, key, value)

        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))

    def test_attention_scores(self) -> None:
        """Test if attention scores are properly stored and have correct shape."""
        # Create input tensors
        query = tf.random.normal((self.batch_size, self.seq_len, self.d_model))
        key = tf.random.normal((self.batch_size, self.seq_len, self.d_model))
        value = tf.random.normal((self.batch_size, self.seq_len, self.d_model))

        # Get output (this should store attention scores)
        _ = self.layer(query, key, value)

        # Check attention scores shape
        expected_attn_shape = (self.batch_size, self.n_head, self.seq_len, self.seq_len)
        self.assertEqual(self.layer.attention_scores.shape, expected_attn_shape)

        # Check if attention scores sum to 1 for each head and query
        attn_sums = ops.sum(self.layer.attention_scores, axis=-1)
        tf.debugging.assert_near(attn_sums, tf.ones_like(attn_sums), rtol=1e-5)

    def test_training_mode(self) -> None:
        """Test layer behavior in training mode."""
        # Create input tensors
        query = tf.random.normal((self.batch_size, self.seq_len, self.d_model))
        key = tf.random.normal((self.batch_size, self.seq_len, self.d_model))
        value = tf.random.normal((self.batch_size, self.seq_len, self.d_model))

        # Get outputs in training and inference modes
        training_output = self.layer(query, key, value, training=True)
        inference_output = self.layer(query, key, value, training=False)

        # Outputs should have same shape but potentially different values due to dropout
        self.assertEqual(training_output.shape, inference_output.shape)

    def test_serialization(self) -> None:
        """Test layer serialization and deserialization."""
        # Create a simple model with our layer
        query_input = layers.Input(shape=(self.seq_len, self.d_model))
        key_input = layers.Input(shape=(self.seq_len, self.d_model))
        value_input = layers.Input(shape=(self.seq_len, self.d_model))
        
        outputs = self.layer(query_input, key_input, value_input)
        model = Model(inputs=[query_input, key_input, value_input], outputs=outputs)

        # Save and load the model
        model_config = model.get_config()
        reconstructed_model = Model.from_config(model_config)

        # Test forward pass with the reconstructed model
        query = tf.random.normal((self.batch_size, self.seq_len, self.d_model))
        key = tf.random.normal((self.batch_size, self.seq_len, self.d_model))
        value = tf.random.normal((self.batch_size, self.seq_len, self.d_model))
        
        original_output = model([query, key, value])
        reconstructed_output = reconstructed_model([query, key, value])
        
        # Check that outputs have the same shape
        self.assertEqual(original_output.shape, reconstructed_output.shape)


if __name__ == '__main__':
    unittest.main()
