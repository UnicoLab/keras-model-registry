#!/usr/bin/env python3
"""
Complete example of using BaseFeedForwardModel with KDP preprocessing.

This example demonstrates:
1. Creating dummy CSV data
2. Building a model with KDP preprocessing layer
3. Training the model with included preprocessing
4. Saving the model
5. Testing prediction with raw data

Run this example:
    python examples/kdp_feed_forward_example.py
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.metrics import MeanAbsoluteError

from kmr.models.feed_forward import BaseFeedForwardModel
from kdp import TabularDataProcessor


def create_dummy_data(output_path: Path, n_samples: int = 1000) -> pd.DataFrame:
    """Create dummy CSV data for demonstration.

    Args:
        output_path: Path to save the CSV file
        n_samples: Number of samples to generate

    Returns:
        Generated DataFrame
    """
    print(f"📊 Creating dummy data with {n_samples} samples...")

    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate synthetic tabular data with different feature types
    data = {
        # Numerical features with different distributions
        "age": np.random.normal(35, 10, n_samples).astype(int),
        "income": np.random.exponential(50000, n_samples),
        "credit_score": np.random.normal(650, 100, n_samples).astype(int),
        # Categorical features
        "education": np.random.choice(
            ["High School", "Bachelor", "Master", "PhD"], n_samples,
        ),
        "employment_status": np.random.choice(
            ["Employed", "Unemployed", "Self-employed"], n_samples,
        ),
        "city_tier": np.random.choice(["Tier 1", "Tier 2", "Tier 3"], n_samples),
        # Boolean features
        "has_loan": np.random.choice([True, False], n_samples),
        "owns_property": np.random.choice([True, False], n_samples),
        # Target variable (loan approval probability)
        "loan_approval_probability": np.random.uniform(0, 1, n_samples),
    }

    df = pd.DataFrame(data)

    # Add some missing values to test preprocessing capabilities
    missing_indices = np.random.choice(
        df.index, size=int(0.05 * n_samples), replace=False,
    )
    df.loc[missing_indices, "income"] = np.nan

    missing_indices = np.random.choice(
        df.index, size=int(0.03 * n_samples), replace=False,
    )
    df.loc[missing_indices, "education"] = None

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"✅ Data saved to {output_path}")
    print(f"📈 Data shape: {df.shape}")
    print(f"🔍 Missing values:\n{df.isnull().sum()}")

    return df


def create_kdp_preprocessor(csv_path: Path) -> TabularDataProcessor:
    """Create and fit KDP preprocessor.

    Args:
        csv_path: Path to the CSV data file

    Returns:
        Fitted TabularDataProcessor
    """
    print("🔧 Creating KDP preprocessor...")

    # Initialize KDP processor with comprehensive configuration
    processor = TabularDataProcessor(
        target_column="loan_approval_probability",
        categorical_columns=[
            "education",
            "employment_status",
            "city_tier",
            "has_loan",
            "owns_property",
        ],
        numerical_columns=["age", "income", "credit_score"],
        fill_missing_values=True,
        normalize_numerical=True,
        encode_categorical=True,
        handle_outliers=True,
        outlier_method="iqr",
    )

    # Fit the processor on the data
    print("📚 Fitting preprocessor on data...")
    processor.fit(csv_path)

    print("✅ KDP preprocessor created and fitted successfully")
    return processor


def create_feed_forward_model(
    feature_names: list[str], preprocessing_model: tf.keras.Model,
) -> BaseFeedForwardModel:
    """Create BaseFeedForwardModel with KDP preprocessing.

    Args:
        feature_names: List of feature names
        preprocessing_model: KDP preprocessing model

    Returns:
        Compiled BaseFeedForwardModel
    """
    print("🏗️ Creating BaseFeedForwardModel with KDP preprocessing...")

    # Create the model with comprehensive architecture
    model = BaseFeedForwardModel(
        feature_names=feature_names,
        hidden_units=[128, 64, 32, 16],  # Deep architecture
        output_units=1,
        dropout_rate=0.3,  # Regularization
        activation="relu",
        preprocessing_model=preprocessing_model,
        kernel_initializer="he_normal",  # Better for ReLU
        name="loan_approval_predictor",
    )

    # Compile with appropriate optimizer and loss
    model.compile(
        optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
        loss=MeanSquaredError(),
        metrics=[MeanAbsoluteError(), "mape"],  # Mean Absolute Percentage Error
    )

    print("✅ Model created and compiled successfully")
    print(f"📊 Model architecture: {model.hidden_units} -> {model.output_units}")
    print(f"🔄 Input features: {len(model.feature_names)}")

    return model


def train_model(
    model: BaseFeedForwardModel,
    X_train: dict[str, np.ndarray],
    y_train: np.ndarray,
    X_val: dict[str, np.ndarray] = None,
    y_val: np.ndarray = None,
) -> tf.keras.callbacks.History:
    """Train the model with early stopping and callbacks.

    Args:
        model: The model to train
        X_train: Training features
        y_train: Training targets
        X_val: Validation features (optional)
        y_val: Validation targets (optional)

    Returns:
        Training history
    """
    print("🚀 Starting model training...")

    # Set up callbacks for better training
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss" if X_val is not None else "loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss" if X_val is not None else "loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # Train the model
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val) if X_val is not None else None,
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )

    print("✅ Training completed successfully")
    return history


def evaluate_model(
    model: BaseFeedForwardModel, X_test: dict[str, np.ndarray], y_test: np.ndarray,
) -> dict[str, float]:
    """Evaluate the model on test data.

    Args:
        model: The trained model
        X_test: Test features
        y_test: Test targets

    Returns:
        Dictionary of evaluation metrics
    """
    print("📊 Evaluating model on test data...")

    # Get predictions
    predictions = model.predict(X_test, verbose=0)

    # Calculate metrics
    mse = tf.keras.metrics.mean_squared_error(y_test, predictions).numpy()
    mae = tf.keras.metrics.mean_absolute_error(y_test, predictions).numpy()
    mape = tf.keras.metrics.mean_absolute_percentage_error(y_test, predictions).numpy()

    metrics = {"mse": float(mse), "mae": float(mae), "mape": float(mape)}

    print("✅ Evaluation completed")
    print(f"📈 Test MSE: {metrics['mse']:.4f}")
    print(f"📈 Test MAE: {metrics['mae']:.4f}")
    print(f"📈 Test MAPE: {metrics['mape']:.2f}%")

    return metrics


def test_raw_data_prediction(
    model: BaseFeedForwardModel, feature_names: list[str],
) -> None:
    """Test prediction with completely raw data (including missing values).

    Args:
        model: The trained model
        feature_names: List of feature names
    """
    print("🧪 Testing prediction with raw data...")

    # Create raw test data with missing values and different data types
    raw_test_cases = [
        {
            "age": 25,
            "income": 45000.0,
            "credit_score": 720,
            "education": "Bachelor",
            "employment_status": "Employed",
            "city_tier": "Tier 1",
            "has_loan": True,
            "owns_property": False,
        },
        {
            "age": 35,
            "income": np.nan,  # Missing value
            "credit_score": 580,
            "education": None,  # Missing value
            "employment_status": "Self-employed",
            "city_tier": "Tier 2",
            "has_loan": False,
            "owns_property": True,
        },
        {
            "age": 45,
            "income": 80000.0,
            "credit_score": 750,
            "education": "Master",
            "employment_status": "Employed",
            "city_tier": "Tier 1",
            "has_loan": False,
            "owns_property": True,
        },
    ]

    for i, test_case in enumerate(raw_test_cases, 1):
        print(f"\n🔍 Test case {i}:")
        print(f"   Input: {test_case}")

        # Convert to model input format
        X_test = {name: np.array([test_case[name]]) for name in feature_names}

        # Make prediction
        prediction = model.predict(X_test, verbose=0)
        loan_probability = prediction[0][0]

        print(f"   📊 Predicted loan approval probability: {loan_probability:.4f}")
        print(
            f"   💡 Recommendation: {'APPROVE' if loan_probability > 0.5 else 'REJECT'}",
        )


def main() -> None:
    """Main function demonstrating the complete workflow."""
    print("🚀 KMR BaseFeedForwardModel with KDP Integration Example")
    print("=" * 60)

    # Create temporary directory for data
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Step 1: Create dummy data
        csv_path = temp_path / "loan_data.csv"
        df = create_dummy_data(csv_path, n_samples=2000)

        # Step 2: Create KDP preprocessor
        processor = create_kdp_preprocessor(csv_path)

        # Step 3: Create preprocessing model
        preprocessing_model = processor.create_preprocessing_model()

        # Step 4: Define feature names (excluding target)
        feature_names = [
            col for col in df.columns if col != "loan_approval_probability"
        ]

        # Step 5: Create BaseFeedForwardModel
        model = create_feed_forward_model(feature_names, preprocessing_model)

        # Step 6: Prepare training data
        print("\n📊 Preparing training data...")
        train_size = int(0.8 * len(df))
        val_size = int(0.1 * len(df))

        train_df = df.iloc[:train_size]
        val_df = df.iloc[train_size : train_size + val_size]
        test_df = df.iloc[train_size + val_size :]

        X_train = {name: train_df[name].values for name in feature_names}
        y_train = train_df["loan_approval_probability"].values

        X_val = {name: val_df[name].values for name in feature_names}
        y_val = val_df["loan_approval_probability"].values

        X_test = {name: test_df[name].values for name in feature_names}
        y_test = test_df["loan_approval_probability"].values

        print(
            f"✅ Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}",
        )

        # Step 7: Train the model
        history = train_model(model, X_train, y_train, X_val, y_val)

        # Step 8: Evaluate the model
        metrics = evaluate_model(model, X_test, y_test)

        # Step 9: Save the model
        model_save_path = temp_path / "saved_model"
        print(f"\n💾 Saving model to {model_save_path}...")
        model.save(model_save_path)
        print("✅ Model saved successfully")

        # Step 10: Load and test the saved model
        print(f"\n📂 Loading model from {model_save_path}...")
        loaded_model = tf.keras.models.load_model(model_save_path)
        print("✅ Model loaded successfully")

        # Verify loaded model works
        loaded_predictions = loaded_model.predict(X_test[:5], verbose=0)
        original_predictions = model.predict(X_test[:5], verbose=0)

        if np.allclose(loaded_predictions, original_predictions, rtol=1e-5):
            print("✅ Loaded model predictions match original model")
        else:
            print("❌ Loaded model predictions differ from original model")

        # Step 11: Test with raw data
        test_raw_data_prediction(loaded_model, feature_names)

        # Step 12: Display training history
        print("\n📈 Training Summary:")
        print(f"   Final training loss: {history.history['loss'][-1]:.4f}")
        if "val_loss" in history.history:
            print(f"   Final validation loss: {history.history['val_loss'][-1]:.4f}")
        print(f"   Training epochs: {len(history.history['loss'])}")

        print("\n🎉 Example completed successfully!")
        print(
            f"📊 Final test metrics: MSE={metrics['mse']:.4f}, MAE={metrics['mae']:.4f}",
        )


if __name__ == "__main__":
    main()
