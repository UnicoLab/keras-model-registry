"""Test universal input handling across all KMR models."""

import numpy as np
import pytest
import keras
from keras import layers
from collections import OrderedDict

from kmr.models import Autoencoder, BaseFeedForwardModel, SFNEBlock, TerminatorModel


class TestUniversalInputHandling:
    """Test universal input handling across all KMR models."""

    def test_autoencoder_universal_inputs(self):
        """Test Autoencoder with various input formats."""
        # Create model
        model = Autoencoder(
            input_dim=10, encoding_dim=5, intermediate_dim=8, name="test_autoencoder",
        )

        # Test data
        batch_size = 32
        input_dim = 10
        test_data = np.random.randn(batch_size, input_dim).astype(np.float32)

        # Test 1: Single tensor input
        output1 = model(test_data)
        assert output1.shape == (batch_size, input_dim)

        # Test 2: List input
        output2 = model([test_data])
        assert output2.shape == (batch_size, input_dim)

        # Test 3: Dictionary input
        output3 = model({"input": test_data})
        assert output3.shape == (batch_size, input_dim)

        # Test 4: OrderedDict input
        output4 = model(OrderedDict({"input": test_data}))
        assert output4.shape == (batch_size, input_dim)

        # Test 5: Multiple inputs (concatenated)
        input1 = test_data[:, :5]
        input2 = test_data[:, 5:]
        output5 = model([input1, input2])
        assert output5.shape == (batch_size, input_dim)

        # Test 6: Dictionary with multiple inputs
        output6 = model({"input_0": input1, "input_1": input2})
        assert output6.shape == (batch_size, input_dim)

    def test_feed_forward_universal_inputs(self):
        """Test BaseFeedForwardModel with various input formats."""
        # Create model
        model = BaseFeedForwardModel(
            feature_names=["feature_1", "feature_2"],
            hidden_units=[64, 32],
            output_units=1,
            name="test_feed_forward",
        )

        # Test data
        batch_size = 32
        test_data = {
            "feature_1": np.random.randn(batch_size, 1).astype(np.float32),
            "feature_2": np.random.randn(batch_size, 1).astype(np.float32),
        }

        # Test 1: Dictionary input
        output1 = model(test_data)
        assert output1.shape == (batch_size, 1)

        # Test 2: OrderedDict input
        output2 = model(OrderedDict(test_data))
        assert output2.shape == (batch_size, 1)

        # Test 3: List input
        output3 = model([test_data["feature_1"], test_data["feature_2"]])
        assert output3.shape == (batch_size, 1)

        # Test 4: Single tensor input (concatenated)
        single_input = np.concatenate(
            [test_data["feature_1"], test_data["feature_2"]], axis=-1,
        )
        output4 = model(single_input)
        assert output4.shape == (batch_size, 1)

    def test_sfne_block_universal_inputs(self):
        """Test SFNEBlock with various input formats."""
        # Create model
        model = SFNEBlock(
            input_dim=10,
            output_dim=5,
            hidden_dim=8,
            num_layers=2,
            slow_network_layers=1,
            slow_network_units=4,
            name="test_sfne_block",
        )

        # Test data
        batch_size = 32
        input_dim = 10
        test_data = np.random.randn(batch_size, input_dim).astype(np.float32)

        # Test 1: Single tensor input
        output1 = model(test_data)
        assert output1.shape == (batch_size, 5)

        # Test 2: List input
        output2 = model([test_data])
        assert output2.shape == (batch_size, 5)

        # Test 3: Dictionary input
        output3 = model({"input": test_data})
        assert output3.shape == (batch_size, 5)

        # Test 4: Multiple inputs (concatenated)
        input1 = test_data[:, :5]
        input2 = test_data[:, 5:]
        output4 = model([input1, input2])
        assert output4.shape == (batch_size, 5)

    def test_terminator_model_universal_inputs(self):
        """Test TerminatorModel with various input formats."""
        # Create model
        model = TerminatorModel(
            input_dim=10,
            context_dim=5,
            output_dim=5,
            hidden_dim=8,
            num_layers=2,
            slow_network_layers=1,
            slow_network_units=4,
            name="test_terminator",
        )

        # Test data
        batch_size = 32
        input_dim = 10
        context_dim = 5
        input_data = np.random.randn(batch_size, input_dim).astype(np.float32)
        context_data = np.random.randn(batch_size, context_dim).astype(np.float32)

        # Test 1: List input [input, context]
        output1 = model([input_data, context_data])
        assert output1.shape == (batch_size, 5)

        # Test 2: Dictionary input
        output2 = model({"input": input_data, "context": context_data})
        assert output2.shape == (batch_size, 5)

        # Test 3: OrderedDict input
        output3 = model(OrderedDict({"input": input_data, "context": context_data}))
        assert output3.shape == (batch_size, 5)

        # Test 4: Single input (context will be zeros)
        output4 = model(input_data)
        assert output4.shape == (batch_size, 5)

        # Test 5: Tuple input
        output5 = model((input_data, context_data))
        assert output5.shape == (batch_size, 5)

    def test_models_with_preprocessing(self):
        """Test models with preprocessing models."""
        # Create a simple preprocessing model
        preprocessing_input = layers.Input(shape=(5,), name="preprocessing_input")
        x = layers.Dense(10, activation="relu", name="preprocessing_dense")(
            preprocessing_input,
        )
        preprocessing_model = keras.Model(
            inputs=preprocessing_input, outputs=x, name="preprocessing_model",
        )

        # Test Autoencoder with preprocessing
        model = Autoencoder(
            input_dim=10,
            encoding_dim=5,
            intermediate_dim=8,
            preprocessing_model=preprocessing_model,
            name="test_autoencoder_preprocessing",
        )

        # Test data
        batch_size = 32
        test_data = np.random.randn(batch_size, 5).astype(np.float32)

        # Test with single tensor input
        output = model(test_data)
        # When preprocessing model is used, Autoencoder returns a dictionary with anomaly detection results
        assert isinstance(output, dict)
        assert "reconstruction" in output
        assert output["reconstruction"].shape == (batch_size, 10)

        # Test with dictionary input
        output_dict = model({"input": test_data})
        assert isinstance(output_dict, dict)
        assert "reconstruction" in output_dict
        assert output_dict["reconstruction"].shape == (batch_size, 10)

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Create model
        model = BaseFeedForwardModel(
            feature_names=["feature_1", "feature_2"],
            hidden_units=[64, 32],
            output_units=1,
            name="test_feed_forward",
        )

        # Test with missing feature
        test_data = {"feature_1": np.random.randn(32, 1).astype(np.float32)}

        with pytest.raises(ValueError, match="incompatible with the layer"):
            model(test_data)

    def test_training_mode(self):
        """Test that models work in both training and inference modes."""
        # Create model
        model = Autoencoder(
            input_dim=10, encoding_dim=5, intermediate_dim=8, name="test_autoencoder",
        )

        # Test data
        test_data = np.random.randn(32, 10).astype(np.float32)

        # Test in training mode
        output_training = model(test_data, training=True)
        assert output_training.shape == (32, 10)

        # Test in inference mode
        output_inference = model(test_data, training=False)
        assert output_inference.shape == (32, 10)


if __name__ == "__main__":
    pytest.main([__file__])
