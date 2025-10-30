#!/usr/bin/env python3
"""
Unit tests for KDP preprocessing integration with TerminatorModel.
This script helps identify and fix issues with KDP preprocessing.
"""

import numpy as np
import tensorflow as tf
import keras
import warnings

warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Import KMR models and utilities
from kmr.models import TerminatorModel
from kmr.utils import KMRDataGenerator


def test_data_generation():
    """Test 1: Verify data generation is balanced and working correctly."""
    print("🧪 Test 1: Data Generation")
    print("=" * 50)

    # Generate balanced data
    X_train, X_test, y_train, y_test = KMRDataGenerator.generate_classification_data(
        n_samples=1000,
        n_features=10,
        n_classes=2,
        noise_level=0.05,
        include_interactions=True,
        include_nonlinear=True,
    )

    # Check data shapes
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    # Check class balance
    train_balance = np.bincount(y_train)
    test_balance = np.bincount(y_test)
    print(f"Training class distribution: {train_balance}")
    print(f"Test class distribution: {test_balance}")

    # Verify balance
    balance_ratio_train = (
        train_balance[1] / train_balance[0] if train_balance[0] > 0 else 0
    )
    balance_ratio_test = test_balance[1] / test_balance[0] if test_balance[0] > 0 else 0

    print(f"Training balance ratio: {balance_ratio_train:.3f}")
    print(f"Test balance ratio: {balance_ratio_test:.3f}")

    # Test passes if data is reasonably balanced
    is_balanced = 0.7 <= balance_ratio_train <= 1.3 and 0.7 <= balance_ratio_test <= 1.3
    print(f"✅ Data is balanced: {is_balanced}")

    return X_train, X_test, y_train, y_test, is_balanced


def test_basic_terminator_model(X_train, X_test, y_train, y_test):
    """Test 2: Verify basic TerminatorModel works correctly."""
    print("\n🧪 Test 2: Basic TerminatorModel")
    print("=" * 50)

    # Generate context data
    context_train, context_test, _, _ = KMRDataGenerator.generate_classification_data(
        n_samples=len(X_train),
        n_features=5,
        n_classes=2,
        noise_level=0.05,
        include_interactions=False,
        include_nonlinear=False,
    )

    # Create basic model
    model = TerminatorModel(
        input_dim=X_train.shape[1],
        context_dim=context_train.shape[1],
        output_dim=1,
        hidden_dim=64,
        num_layers=2,
        num_blocks=2,
        slow_network_units=32,
        slow_network_layers=2,
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", "precision", "recall"],
    )

    print("✅ Basic model created and compiled")

    # Train model
    train_data = [X_train, context_train]
    test_data = [X_test, context_test]

    history = model.fit(
        train_data,
        y_train,
        validation_data=(test_data, y_test),
        epochs=10,
        batch_size=64,
        verbose=0,
    )

    # Evaluate model
    test_loss, test_accuracy, test_precision, test_recall = model.evaluate(
        test_data,
        y_test,
        verbose=0,
    )

    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")

    # Calculate F1 score
    f1_score = (
        2 * (test_precision * test_recall) / (test_precision + test_recall)
        if (test_precision + test_recall) > 0
        else 0.0
    )
    print(f"Test F1-Score: {f1_score:.4f}")

    # Check predictions
    y_pred_proba = model.predict(test_data, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    pred_dist = np.bincount(y_pred)
    print(f"Prediction distribution: {pred_dist}")

    # Test passes if model achieves reasonable performance
    is_working = test_precision > 0.0 and test_recall > 0.0 and f1_score > 0.5
    print(f"✅ Basic model is working: {is_working}")

    return model, is_working, f1_score


def test_simple_preprocessing_model(X_train, X_test, y_train, y_test):
    """Test 3: Test simple custom preprocessing model."""
    print("\n🧪 Test 3: Simple Preprocessing Model")
    print("=" * 50)

    from keras import layers, Model

    # Create simple preprocessing model
    input_layer = layers.Input(shape=(X_train.shape[1],), name="input_0")
    normalized = layers.LayerNormalization()(input_layer)
    dense1 = layers.Dense(64, activation="relu")(normalized)
    dense2 = layers.Dense(32, activation="relu")(dense1)
    output_layer = layers.Dense(16, activation="relu")(dense2)

    preprocessing_model = Model(
        inputs=input_layer,
        outputs=output_layer,
        name="simple_preprocessor",
    )

    print("✅ Simple preprocessing model created")
    print(f"Input shape: {preprocessing_model.input_shape}")
    print(f"Output shape: {preprocessing_model.output_shape}")

    # Test preprocessing model
    preprocessed_sample = preprocessing_model.predict(X_train[:5], verbose=0)
    print(f"✅ Preprocessing test passed! Output shape: {preprocessed_sample.shape}")

    # Generate context data
    context_train, context_test, _, _ = KMRDataGenerator.generate_classification_data(
        n_samples=len(X_train),
        n_features=5,
        n_classes=2,
        noise_level=0.05,
        include_interactions=False,
        include_nonlinear=False,
    )

    # Create TerminatorModel with preprocessing
    model_with_prep = TerminatorModel(
        input_dim=preprocessing_model.output_shape[-1],
        context_dim=context_train.shape[1],
        output_dim=1,
        hidden_dim=64,
        num_layers=2,
        num_blocks=2,
        slow_network_units=32,
        slow_network_layers=2,
        preprocessing_model=preprocessing_model,
    )

    model_with_prep.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", "precision", "recall"],
    )

    print("✅ Model with preprocessing created and compiled")

    # Train model
    train_data = [X_train, context_train]
    test_data = [X_test, context_test]

    history = model_with_prep.fit(
        train_data,
        y_train,
        validation_data=(test_data, y_test),
        epochs=10,
        batch_size=64,
        verbose=0,
    )

    # Evaluate model
    test_loss, test_accuracy, test_precision, test_recall = model_with_prep.evaluate(
        test_data,
        y_test,
        verbose=0,
    )

    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")

    # Calculate F1 score
    f1_score = (
        2 * (test_precision * test_recall) / (test_precision + test_recall)
        if (test_precision + test_recall) > 0
        else 0.0
    )
    print(f"Test F1-Score: {f1_score:.4f}")

    # Check predictions
    y_pred_proba = model_with_prep.predict(test_data, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    pred_dist = np.bincount(y_pred)
    print(f"Prediction distribution: {pred_dist}")

    # Test passes if model achieves reasonable performance
    is_working = test_precision > 0.0 and test_recall > 0.0 and f1_score > 0.5
    print(f"✅ Simple preprocessing model is working: {is_working}")

    return model_with_prep, is_working, f1_score


def test_kdp_preprocessing_model(X_train, X_test, y_train, y_test):
    """Test 4: Test KDP preprocessing model."""
    print("\n🧪 Test 4: KDP Preprocessing Model")
    print("=" * 50)

    try:
        # Try to import KDP
        from kdp import PreprocessingModel, FeatureType
        import pandas as pd

        print("✅ KDP imported successfully")

        # Create KDP dataset
        kdp_data = {}
        for i in range(X_train.shape[1]):
            kdp_data[f"feature_{i}"] = X_train[:, i]

        df_kdp = pd.DataFrame(kdp_data)

        # Define feature specifications (all numerical)
        features_specs = {}
        for i in range(X_train.shape[1]):
            features_specs[f"feature_{i}"] = FeatureType.FLOAT_NORMALIZED

        print("✅ KDP dataset and feature specs created")

        # Save dataset for KDP
        df_kdp.to_csv("temp_kdp_test_data.csv", index=False)

        # Create KDP preprocessing model with minimal settings
        preprocessor = PreprocessingModel(
            path_data="temp_kdp_test_data.csv",
            features_specs=features_specs,
            use_distribution_aware=False,
            tabular_attention=False,
            use_feature_moe=False,
            feature_selection_placement=None,
        )

        # Build the preprocessor
        result = preprocessor.build_preprocessor()
        kdp_preprocessing_model = result["model"]

        print("✅ KDP preprocessing model built successfully")
        print(f"KDP input shape: {kdp_preprocessing_model.input_shape}")
        print(f"KDP output shape: {kdp_preprocessing_model.output_shape}")

        # Test KDP preprocessing model
        kdp_sample = kdp_preprocessing_model.predict(X_train[:5], verbose=0)
        print(f"✅ KDP preprocessing test passed! Output shape: {kdp_sample.shape}")

        # Generate context data
        (
            context_train,
            context_test,
            _,
            _,
        ) = KMRDataGenerator.generate_classification_data(
            n_samples=len(X_train),
            n_features=5,
            n_classes=2,
            noise_level=0.05,
            include_interactions=False,
            include_nonlinear=False,
        )

        # Create TerminatorModel with KDP preprocessing
        model_with_kdp = TerminatorModel(
            input_dim=kdp_preprocessing_model.output_shape[-1],
            context_dim=context_train.shape[1],
            output_dim=1,
            hidden_dim=128,  # Larger to handle KDP output
            num_layers=3,  # More layers
            num_blocks=3,  # More blocks
            slow_network_units=64,
            slow_network_layers=3,
            preprocessing_model=kdp_preprocessing_model,
        )

        model_with_kdp.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=0.0005,
            ),  # Lower learning rate
            loss="binary_crossentropy",
            metrics=["accuracy", "precision", "recall"],
        )

        print("✅ Model with KDP preprocessing created and compiled")

        # Train model
        train_data = [X_train, context_train]
        test_data = [X_test, context_test]

        history = model_with_kdp.fit(
            train_data,
            y_train,
            validation_data=(test_data, y_test),
            epochs=15,  # More epochs
            batch_size=32,  # Smaller batch size
            verbose=0,
        )

        # Evaluate model
        test_loss, test_accuracy, test_precision, test_recall = model_with_kdp.evaluate(
            test_data,
            y_test,
            verbose=0,
        )

        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")

        # Calculate F1 score
        f1_score = (
            2 * (test_precision * test_recall) / (test_precision + test_recall)
            if (test_precision + test_recall) > 0
            else 0.0
        )
        print(f"Test F1-Score: {f1_score:.4f}")

        # Check predictions
        y_pred_proba = model_with_kdp.predict(test_data, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        pred_dist = np.bincount(y_pred)
        print(f"Prediction distribution: {pred_dist}")

        # Analyze the issue
        if test_precision == 0.0 and test_recall == 0.0:
            print("❌ CRITICAL ISSUE: KDP model predicting only one class!")
            print(f"   Prediction distribution: {pred_dist}")
            print(f"   True distribution: {np.bincount(y_test)}")
            prob_range = [y_pred_proba.min(), y_pred_proba.max()]
            print(f"   Probability range: [{prob_range[0]:.4f}, {prob_range[1]:.4f}]")

            # Check if KDP is creating a bottleneck
            kdp_output = kdp_preprocessing_model.predict(X_test, verbose=0)
            print(f"   KDP output shape: {kdp_output.shape}")
            print(
                f"   KDP output range: [{kdp_output.min():.4f}, {kdp_output.max():.4f}]",
            )
            print(f"   KDP output std: {kdp_output.std():.4f}")

            if kdp_output.std() < 0.01:
                print(
                    "   ⚠️ KDP output has very low variance - information bottleneck!",
                )
            if kdp_output.shape[1] < X_train.shape[1]:
                print(
                    "   ⚠️ KDP is reducing dimensionality - may be losing information!",
                )

        # Test passes if model achieves reasonable performance
        is_working = test_precision > 0.0 and test_recall > 0.0 and f1_score > 0.5
        print(f"✅ KDP preprocessing model is working: {is_working}")

        # Clean up
        import os

        if os.path.exists("temp_kdp_test_data.csv"):
            os.remove("temp_kdp_test_data.csv")

        return model_with_kdp, is_working, f1_score

    except Exception as e:
        print(f"❌ KDP preprocessing failed: {e}")
        print("KDP is not available or has compatibility issues")
        return None, False, 0.0


def test_improved_kdp_preprocessing(X_train, X_test, y_train, y_test):
    """Test 5: Test improved KDP preprocessing with better configuration."""
    print("\n🧪 Test 5: Improved KDP Preprocessing")
    print("=" * 50)

    try:
        from kdp import PreprocessingModel, FeatureType
        import pandas as pd

        # Create KDP dataset with better feature names
        kdp_data = {}
        for i in range(X_train.shape[1]):
            kdp_data[f"num_feature_{i}"] = X_train[:, i]

        df_kdp = pd.DataFrame(kdp_data)

        # Define feature specifications with better types
        features_specs = {}
        for i in range(X_train.shape[1]):
            features_specs[
                f"num_feature_{i}"
            ] = FeatureType.FLOAT_RESCALED  # Use rescaling instead of normalization

        print("✅ Improved KDP dataset created")

        # Save dataset for KDP
        df_kdp.to_csv("temp_improved_kdp_data.csv", index=False)

        # Create KDP preprocessing model with better settings
        preprocessor = PreprocessingModel(
            path_data="temp_improved_kdp_data.csv",
            features_specs=features_specs,
            use_distribution_aware=True,  # Enable distribution awareness
            tabular_attention=False,  # Keep disabled for simplicity
            use_feature_moe=False,  # Keep disabled for simplicity
            feature_selection_placement=None,
        )

        # Build the preprocessor
        result = preprocessor.build_preprocessor()
        improved_kdp_model = result["model"]

        print("✅ Improved KDP preprocessing model built")
        print(f"Improved KDP input shape: {improved_kdp_model.input_shape}")
        print(f"Improved KDP output shape: {improved_kdp_model.output_shape}")

        # Test improved KDP preprocessing model
        improved_sample = improved_kdp_model.predict(X_train[:5], verbose=0)
        print(
            f"✅ Improved KDP preprocessing test passed! Output shape: {improved_sample.shape}",
        )

        # Generate context data
        (
            context_train,
            context_test,
            _,
            _,
        ) = KMRDataGenerator.generate_classification_data(
            n_samples=len(X_train),
            n_features=5,
            n_classes=2,
            noise_level=0.05,
            include_interactions=False,
            include_nonlinear=False,
        )

        # Create TerminatorModel with improved KDP preprocessing
        model_with_improved_kdp = TerminatorModel(
            input_dim=improved_kdp_model.output_shape[-1],
            context_dim=context_train.shape[1],
            output_dim=1,
            hidden_dim=256,  # Even larger
            num_layers=4,  # More layers
            num_blocks=4,  # More blocks
            slow_network_units=128,
            slow_network_layers=4,
            preprocessing_model=improved_kdp_model,
        )

        model_with_improved_kdp.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=0.0001,
            ),  # Even lower learning rate
            loss="binary_crossentropy",
            metrics=["accuracy", "precision", "recall"],
        )

        print("✅ Model with improved KDP preprocessing created and compiled")

        # Train model with better strategy
        train_data = [X_train, context_train]
        test_data = [X_test, context_test]

        history = model_with_improved_kdp.fit(
            train_data,
            y_train,
            validation_data=(test_data, y_test),
            epochs=20,  # More epochs
            batch_size=16,  # Smaller batch size
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=8,
                    restore_best_weights=True,
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=4,
                    min_lr=1e-7,
                ),
            ],
        )

        # Evaluate model
        (
            test_loss,
            test_accuracy,
            test_precision,
            test_recall,
        ) = model_with_improved_kdp.evaluate(test_data, y_test, verbose=0)

        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")

        # Calculate F1 score
        f1_score = (
            2 * (test_precision * test_recall) / (test_precision + test_recall)
            if (test_precision + test_recall) > 0
            else 0.0
        )
        print(f"Test F1-Score: {f1_score:.4f}")

        # Check predictions
        y_pred_proba = model_with_improved_kdp.predict(test_data, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        pred_dist = np.bincount(y_pred)
        print(f"Prediction distribution: {pred_dist}")

        # Test passes if model achieves reasonable performance
        is_working = test_precision > 0.0 and test_recall > 0.0 and f1_score > 0.5
        print(f"✅ Improved KDP preprocessing model is working: {is_working}")

        # Clean up
        import os

        if os.path.exists("temp_improved_kdp_data.csv"):
            os.remove("temp_improved_kdp_data.csv")

        return model_with_improved_kdp, is_working, f1_score

    except Exception as e:
        print(f"❌ Improved KDP preprocessing failed: {e}")
        return None, False, 0.0


def run_all_tests():
    """Run all unit tests and provide summary."""
    print("🚀 Running KDP Preprocessing Unit Tests")
    print("=" * 60)

    # Test 1: Data Generation
    X_train, X_test, y_train, y_test, data_ok = test_data_generation()

    if not data_ok:
        print("❌ Data generation failed - stopping tests")
        return None

    # Test 2: Basic TerminatorModel
    basic_model, basic_ok, basic_f1 = test_basic_terminator_model(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    # Test 3: Simple Preprocessing Model
    simple_model, simple_ok, simple_f1 = test_simple_preprocessing_model(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    # Test 4: KDP Preprocessing Model
    kdp_model, kdp_ok, kdp_f1 = test_kdp_preprocessing_model(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    # Test 5: Improved KDP Preprocessing Model
    (
        improved_kdp_model,
        improved_kdp_ok,
        improved_kdp_f1,
    ) = test_improved_kdp_preprocessing(X_train, X_test, y_train, y_test)

    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 60)
    print(f"Data Generation: {'✅ PASS' if data_ok else '❌ FAIL'}")
    print(
        f"Basic TerminatorModel: {'✅ PASS' if basic_ok else '❌ FAIL'} (F1: {basic_f1:.4f})",
    )
    print(
        f"Simple Preprocessing: {'✅ PASS' if simple_ok else '❌ FAIL'} (F1: {simple_f1:.4f})",
    )
    print(f"KDP Preprocessing: {'✅ PASS' if kdp_ok else '❌ FAIL'} (F1: {kdp_f1:.4f})")
    print(
        f"Improved KDP: {'✅ PASS' if improved_kdp_ok else '❌ FAIL'} (F1: {improved_kdp_f1:.4f})",
    )

    # Recommendations
    print("\n🔧 Recommendations:")
    if not kdp_ok and not improved_kdp_ok:
        print("❌ KDP preprocessing is not working - use simple preprocessing instead")
    elif not kdp_ok and improved_kdp_ok:
        print("✅ Improved KDP configuration works - use improved settings")
    elif kdp_ok and improved_kdp_ok:
        print("✅ Both KDP configurations work - choose based on performance")
    else:
        print("⚠️ Mixed results - investigate further")

    return {
        "basic_f1": basic_f1,
        "simple_f1": simple_f1,
        "kdp_f1": kdp_f1,
        "improved_kdp_f1": improved_kdp_f1,
    }


if __name__ == "__main__":
    results = run_all_tests()
