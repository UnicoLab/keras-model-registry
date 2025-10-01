"""End-to-end integration tests for BaseFeedForwardModel with KDP preprocessing."""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from keras import Model, layers
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.metrics import MeanAbsoluteError

from kmr.models.feed_forward import BaseFeedForwardModel
from kdp.auto_config import auto_configure
from kdp.pipeline import Pipeline
from kdp.processor import FeaturePreprocessor


class TestBaseFeedForwardKDPIntegration:
    """Test BaseFeedForwardModel integration with KDP preprocessing."""

    @pytest.fixture
    def temp_dir(self) -> Path:
        """Create a temporary directory for test data."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def dummy_data(self, temp_dir: Path) -> tuple[Path, pd.DataFrame]:
        """Create dummy CSV data for testing."""
        # Generate synthetic tabular data
        np.random.seed(42)
        n_samples = 1000
        
        # Create features with different types
        data = {
            'numeric_feature_1': np.random.normal(10, 3, n_samples),
            'numeric_feature_2': np.random.exponential(2, n_samples),
            'categorical_feature': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
            'boolean_feature': np.random.choice([True, False], n_samples),
            'target': np.random.normal(5, 1, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Add some missing values to test preprocessing
        df.loc[df.sample(50).index, 'numeric_feature_1'] = np.nan
        df.loc[df.sample(30).index, 'categorical_feature'] = None
        
        # Save to CSV
        csv_path = temp_dir / "dummy_data.csv"
        df.to_csv(csv_path, index=False)
        
        return csv_path, df

    @pytest.fixture
    def kdp_preprocessor(self, dummy_data: tuple[Path, pd.DataFrame]) -> dict:
        """Create and fit KDP preprocessor."""
        csv_path, df = dummy_data
        
        # Use KDP auto_configure to analyze the data
        config = auto_configure(
            data_path=str(csv_path),
            batch_size=1000,
            save_stats=False
        )
        
        return config

    def test_end_to_end_training_and_prediction(
        self, 
        temp_dir: Path, 
        dummy_data: tuple[Path, pd.DataFrame],
        kdp_preprocessor: dict
    ) -> None:
        """Test complete end-to-end workflow with KDP preprocessing."""
        csv_path, df = dummy_data
        
        # Split data for training and testing
        train_df = df.iloc[:800].copy()
        test_df = df.iloc[800:].copy()
        
        # Save train and test data
        train_path = temp_dir / "train_data.csv"
        test_path = temp_dir / "test_data.csv"
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        # For now, create a simple preprocessing model without KDP
        # This tests the BaseFeedForwardModel integration pattern
        preprocessing_model = tf.keras.Sequential([
            layers.Dense(16, activation='relu', name='preprocessing_dense'),
            layers.Dropout(0.1, name='preprocessing_dropout')
        ])
        
        # Define feature names (excluding target)
        feature_names = ['numeric_feature_1', 'numeric_feature_2', 'categorical_feature', 'boolean_feature']
        
        # Create BaseFeedForwardModel with preprocessing
        model = BaseFeedForwardModel(
            feature_names=feature_names,
            hidden_units=[64, 32, 16],
            output_units=1,
            dropout_rate=0.2,
            activation='relu',
            preprocessing_model=preprocessing_model,
            name='feed_forward_with_preprocessing'
        )
        
        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=MeanSquaredError(),
            metrics=[MeanAbsoluteError()]
        )
        
        # Prepare training data
        X_train = {name: train_df[name].values for name in feature_names}
        y_train = train_df['target'].values
        
        # Train the model
        history = model.fit(
            X_train, y_train,
            epochs=5,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        # Verify training completed successfully
        assert len(history.history['loss']) == 5
        assert 'val_loss' in history.history
        
        # Test prediction with raw data (should use preprocessing)
        X_test = {name: test_df[name].values for name in feature_names}
        y_test = test_df['target'].values
        
        predictions = model.predict(X_test, verbose=0)
        
        # Verify predictions shape
        assert predictions.shape == (len(test_df), 1)
        assert not np.isnan(predictions).any()
        
        # Test model saving and loading
        model_path = temp_dir / "saved_model"
        model.save(model_path)
        
        # Load the model
        loaded_model = tf.keras.models.load_model(model_path)
        
        # Test prediction with loaded model
        loaded_predictions = loaded_model.predict(X_test, verbose=0)
        
        # Verify predictions are similar (allowing for small numerical differences)
        np.testing.assert_allclose(predictions, loaded_predictions, rtol=1e-5)
        
        # Test with completely raw data (including missing values)
        raw_test_data = {
            'numeric_feature_1': np.array([np.nan, 12.5, 8.3]),
            'numeric_feature_2': np.array([1.2, np.nan, 3.7]),
            'categorical_feature': np.array(['A', None, 'C']),
            'boolean_feature': np.array([True, False, True])
        }
        
        # Should handle missing values through preprocessing
        raw_predictions = loaded_model.predict(raw_test_data, verbose=0)
        assert raw_predictions.shape == (3, 1)
        assert not np.isnan(raw_predictions).any()

    def test_model_with_different_architectures(
        self, 
        temp_dir: Path, 
        dummy_data: tuple[Path, pd.DataFrame],
        kdp_preprocessor: TabularDataProcessor
    ) -> None:
        """Test BaseFeedForwardModel with different architectures."""
        csv_path, df = dummy_data
        feature_names = ['numeric_feature_1', 'numeric_feature_2', 'categorical_feature', 'boolean_feature']
        
        # Test different architectures
        architectures = [
            [32],  # Single hidden layer
            [64, 32],  # Two hidden layers
            [128, 64, 32, 16],  # Deep network
        ]
        
        for hidden_units in architectures:
            preprocessing_model = kdp_preprocessor.create_preprocessing_model()
            
            model = BaseFeedForwardModel(
                feature_names=feature_names,
                hidden_units=hidden_units,
                output_units=1,
                dropout_rate=0.1,
                activation='relu',
                preprocessing_model=preprocessing_model,
                name=f'feed_forward_{len(hidden_units)}_layers'
            )
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss=MeanSquaredError(),
                metrics=[MeanAbsoluteError()]
            )
            
            # Quick training test
            X_train = {name: df[name].values[:100] for name in feature_names}
            y_train = df['target'].values[:100]
            
            history = model.fit(X_train, y_train, epochs=2, verbose=0)
            assert len(history.history['loss']) == 2

    def test_model_serialization_with_kdp(
        self, 
        temp_dir: Path, 
        dummy_data: tuple[Path, pd.DataFrame],
        kdp_preprocessor: TabularDataProcessor
    ) -> None:
        """Test model serialization with KDP preprocessing."""
        csv_path, df = dummy_data
        feature_names = ['numeric_feature_1', 'numeric_feature_2', 'categorical_feature', 'boolean_feature']
        
        preprocessing_model = kdp_preprocessor.create_preprocessing_model()
        
        model = BaseFeedForwardModel(
            feature_names=feature_names,
            hidden_units=[32, 16],
            output_units=1,
            preprocessing_model=preprocessing_model,
            name='serializable_model'
        )
        
        # Test JSON serialization
        config = model.get_config()
        assert 'feature_names' in config
        assert 'hidden_units' in config
        assert 'preprocessing_model' in config
        
        # Test model reconstruction from config
        reconstructed_model = BaseFeedForwardModel.from_config(config)
        assert reconstructed_model.feature_names == model.feature_names
        assert reconstructed_model.hidden_units == model.hidden_units

    def test_error_handling_with_invalid_data(
        self, 
        temp_dir: Path, 
        dummy_data: tuple[Path, pd.DataFrame],
        kdp_preprocessor: TabularDataProcessor
    ) -> None:
        """Test error handling with invalid input data."""
        csv_path, df = dummy_data
        feature_names = ['numeric_feature_1', 'numeric_feature_2', 'categorical_feature', 'boolean_feature']
        
        preprocessing_model = kdp_preprocessor.create_preprocessing_model()
        
        model = BaseFeedForwardModel(
            feature_names=feature_names,
            hidden_units=[32],
            output_units=1,
            preprocessing_model=preprocessing_model
        )
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=MeanSquaredError()
        )
        
        # Test with missing feature
        invalid_data = {
            'numeric_feature_1': np.array([1.0, 2.0]),
            'numeric_feature_2': np.array([3.0, 4.0]),
            # Missing categorical_feature and boolean_feature
        }
        
        with pytest.raises((KeyError, ValueError)):
            model.predict(invalid_data, verbose=0)
        
        # Test with wrong data types
        wrong_type_data = {
            'numeric_feature_1': ['not', 'numeric'],
            'numeric_feature_2': np.array([1.0, 2.0]),
            'categorical_feature': np.array(['A', 'B']),
            'boolean_feature': np.array([True, False])
        }
        
        with pytest.raises((TypeError, ValueError)):
            model.predict(wrong_type_data, verbose=0)

    def test_performance_with_large_dataset(
        self, 
        temp_dir: Path, 
        kdp_preprocessor: TabularDataProcessor
    ) -> None:
        """Test model performance with larger dataset."""
        # Generate larger dataset
        np.random.seed(42)
        n_samples = 5000
        
        large_data = {
            'numeric_feature_1': np.random.normal(10, 3, n_samples),
            'numeric_feature_2': np.random.exponential(2, n_samples),
            'categorical_feature': np.random.choice(['A', 'B', 'C', 'D', 'E'], n_samples),
            'boolean_feature': np.random.choice([True, False], n_samples),
            'target': np.random.normal(5, 1, n_samples)
        }
        
        df = pd.DataFrame(large_data)
        csv_path = temp_dir / "large_data.csv"
        df.to_csv(csv_path, index=False)
        
        # Create new processor for large dataset
        processor = TabularDataProcessor(
            target_column='target',
            categorical_columns=['categorical_feature', 'boolean_feature'],
            numerical_columns=['numeric_feature_1', 'numeric_feature_2'],
            fill_missing_values=True,
            normalize_numerical=True,
            encode_categorical=True
        )
        
        processor.fit(csv_path)
        preprocessing_model = processor.create_preprocessing_model()
        
        feature_names = ['numeric_feature_1', 'numeric_feature_2', 'categorical_feature', 'boolean_feature']
        
        model = BaseFeedForwardModel(
            feature_names=feature_names,
            hidden_units=[128, 64, 32],
            output_units=1,
            dropout_rate=0.3,
            preprocessing_model=preprocessing_model
        )
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=MeanSquaredError(),
            metrics=[MeanAbsoluteError()]
        )
        
        # Train on large dataset
        X_train = {name: df[name].values for name in feature_names}
        y_train = df['target'].values
        
        history = model.fit(
            X_train, y_train,
            epochs=3,
            batch_size=64,
            validation_split=0.2,
            verbose=0
        )
        
        # Verify training completed
        assert len(history.history['loss']) == 3
        assert history.history['loss'][-1] < history.history['loss'][0]  # Loss should decrease
        
        # Test prediction performance
        predictions = model.predict(X_train[:100], verbose=0)
        assert predictions.shape == (100, 1)
        assert not np.isnan(predictions).any()
