"""Example demonstrating BaseFeedForwardModel with preprocessing."""

import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from loguru import logger

from keras import Model, layers
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.metrics import MeanAbsoluteError

from kmr.models.feed_forward import BaseFeedForwardModel


def create_sample_data(file_path: Path) -> pd.DataFrame:
    """Create sample tabular data for demonstration."""
    logger.info(f"Creating sample data at {file_path}")

    # Generate synthetic data with different feature types
    np.random.seed(42)
    n_samples = 1000

    data = {
        "numeric_feature_1": np.random.normal(10, 3, n_samples),
        "numeric_feature_2": np.random.exponential(2, n_samples),
        "categorical_feature": np.random.choice(
            [0, 1, 2, 3], n_samples,
        ),  # Encoded as integers
        "boolean_feature": np.random.choice([0, 1], n_samples),  # Encoded as integers
        "target": np.random.normal(5, 1, n_samples),
    }

    df = pd.DataFrame(data)

    # Add some missing values to test preprocessing
    df.loc[df.sample(50).index, "numeric_feature_1"] = np.nan
    df.loc[df.sample(30).index, "categorical_feature"] = np.nan

    df.to_csv(file_path, index=False)
    logger.info(
        f"Created dataset with {len(df)} samples and {len(df.columns)} features",
    )

    return df


def create_preprocessing_model(input_dim: int) -> Model:
    """Create a simple preprocessing model."""
    logger.info(f"Creating preprocessing model for input dimension {input_dim}")

    # Simple preprocessing pipeline with correct input shape
    preprocessing_input = layers.Input(shape=(input_dim,), name="preprocessing_input")
    x = layers.Dense(32, activation="relu", name="preprocessing_dense_1")(
        preprocessing_input,
    )
    x = layers.Dropout(0.1, name="preprocessing_dropout_1")(x)
    x = layers.Dense(16, activation="relu", name="preprocessing_dense_2")(x)
    x = layers.Dropout(0.1, name="preprocessing_dropout_2")(x)
    preprocessing_model = Model(
        inputs=preprocessing_input, outputs=x, name="preprocessing_model",
    )

    return preprocessing_model


def run_feed_forward_example():
    """Run the complete BaseFeedForwardModel example."""
    logger.info("Starting BaseFeedForwardModel example")

    # Create temporary directory
    temp_dir = Path("temp_ff_example")
    temp_dir.mkdir(exist_ok=True)

    try:
        # 1. Create sample data
        csv_path = temp_dir / "sample_data.csv"
        df = create_sample_data(csv_path)

        # Split data
        train_df = df.iloc[:800].copy()
        test_df = df.iloc[800:].copy()

        train_path = temp_dir / "train_data.csv"
        test_path = temp_dir / "test_data.csv"
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        # Define features and target
        target_feature = "target"
        feature_names = [col for col in df.columns if col != target_feature]

        logger.info(f"Features: {feature_names}")
        logger.info(f"Target: {target_feature}")

        # 2. Create preprocessing model
        preprocessing_model = create_preprocessing_model(len(feature_names))

        # 3. Build BaseFeedForwardModel
        logger.info("Building BaseFeedForwardModel with preprocessing")
        model = BaseFeedForwardModel(
            feature_names=feature_names,
            hidden_units=[64, 32, 16],
            output_units=1,
            dropout_rate=0.2,
            activation="relu",
            preprocessing_model=preprocessing_model,
            name="feed_forward_with_preprocessing",
        )

        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=MeanSquaredError(),
            metrics=[MeanAbsoluteError()],
        )

        # 4. Prepare training data
        X_train = {
            name: train_df[name].values.astype(np.float32) for name in feature_names
        }
        y_train = train_df[target_feature].values.astype(np.float32)

        logger.info(f"Training data shape: {len(X_train[feature_names[0]])} samples")

        # 5. Train the model
        logger.info("Training the model")
        history = model.fit(
            X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1,
        )

        # 6. Evaluate on test data
        X_test = {
            name: test_df[name].values.astype(np.float32) for name in feature_names
        }
        y_test = test_df[target_feature].values.astype(np.float32)

        test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
        logger.info(f"Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")

        # 7. Make predictions
        predictions = model.predict(X_test, verbose=0)
        logger.info(f"Predictions shape: {predictions.shape}")
        logger.info(f"Sample predictions: {predictions[:5].flatten()}")
        logger.info(f"Sample true values: {y_test[:5]}")

        # 8. Save the model
        model_path = temp_dir / "saved_ff_model"
        logger.info(f"Saving model to {model_path}")
        model.save(model_path)

        # 9. Test model loading and prediction
        logger.info("Testing model loading and prediction")
        loaded_model = tf.keras.models.load_model(model_path)

        # Test with new data
        new_predictions = loaded_model.predict(X_test, verbose=0)

        # Verify predictions are similar
        np.testing.assert_allclose(predictions, new_predictions, rtol=1e-5)
        logger.info("Model loading and prediction test passed!")

        # 10. Test with raw data (including missing values)
        logger.info("Testing prediction with raw data including missing values")
        raw_test_data = {
            "numeric_feature_1": np.array([np.nan, 12.5, 8.3, 15.0]),
            "numeric_feature_2": np.array([1.2, np.nan, 3.7, 2.1]),
            "categorical_feature": np.array([0, np.nan, 2, 1]),
            "boolean_feature": np.array([1, 0, 1, 0]),
        }

        raw_predictions = loaded_model.predict(raw_test_data, verbose=0)
        logger.info(f"Raw data predictions: {raw_predictions.flatten()}")
        logger.info("Raw data prediction test passed!")

        logger.info("BaseFeedForwardModel example completed successfully!")

    except Exception as e:
        logger.error(f"Error in example: {e}")
        raise
    finally:
        # Clean up
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.info("Cleaned up temporary files")


if __name__ == "__main__":
    run_feed_forward_example()
