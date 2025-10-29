"""Data generation utilities for KMR model testing and demonstrations."""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
import keras


class KMRDataGenerator:
    """Utility class for generating synthetic datasets for KMR model testing."""
    
    @staticmethod
    def generate_regression_data(
        n_samples: int = 1000,
        n_features: int = 10,
        noise_level: float = 0.1,
        random_state: int = 42,
        include_interactions: bool = True,
        include_nonlinear: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate synthetic regression data.
        
        Args:
            n_samples: Number of samples
            n_features: Number of features
            noise_level: Level of noise to add
            random_state: Random seed
            include_interactions: Whether to include feature interactions
            include_nonlinear: Whether to include nonlinear relationships
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        np.random.seed(random_state)
        
        # Generate features
        X = np.random.normal(0, 1, (n_samples, n_features))
        
        # Add nonlinear relationships
        if include_nonlinear:
            X[:, 0] = X[:, 0] ** 2  # Quadratic relationship
            X[:, 1] = np.sin(X[:, 1])  # Sinusoidal relationship
            if n_features > 2:
                X[:, 2] = np.exp(X[:, 2] * 0.5)  # Exponential relationship
        
        # Add interactions
        if include_interactions and n_features >= 4:
            X[:, 3] = X[:, 2] * X[:, 3]  # Interaction term
        
        # Generate target with noise
        true_weights = np.random.normal(0, 1, n_features)
        y = np.dot(X, true_weights) + noise_level * np.random.normal(0, 1, n_samples)
        
        # Normalize features
        X_mean = tf.reduce_mean(X, axis=0)
        X_std = tf.math.reduce_std(X, axis=0)
        X_normalized = (X - X_mean) / (X_std + 1e-8)
        
        # Split data
        train_size = int(0.8 * n_samples)
        X_train = X_normalized[:train_size]
        X_test = X_normalized[train_size:]
        y_train = y[:train_size]
        y_test = y[train_size:]
        
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def generate_classification_data(
        n_samples: int = 1000,
        n_features: int = 10,
        n_classes: int = 2,
        noise_level: float = 0.1,
        include_interactions: bool = True,
        include_nonlinear: bool = True,
        random_state: int = 42,
        sparse_features: bool = True,
        sparse_ratio: float = 0.3
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate synthetic classification data.
        
        Args:
            n_samples: Number of samples
            n_features: Number of features
            n_classes: Number of classes
            noise_level: Level of noise to add
            include_interactions: Whether to include feature interactions
            include_nonlinear: Whether to include nonlinear relationships
            random_state: Random seed
            sparse_features: Whether to create sparse features
            sparse_ratio: Ratio of features that are relevant
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        np.random.seed(random_state)
        
        # Generate features
        X = np.random.normal(0, 1, (n_samples, n_features))
        
        # Add nonlinear relationships
        if include_nonlinear:
            X[:, 0] = X[:, 0] ** 2  # Quadratic relationship
            X[:, 1] = np.sin(X[:, 1])  # Sinusoidal relationship
            if n_features > 2:
                X[:, 2] = np.exp(X[:, 2] * 0.5)  # Exponential relationship
        
        # Add interactions
        if include_interactions and n_features >= 4:
            X[:, 3] = X[:, 2] * X[:, 3]  # Interaction term
        
        # Create sparse features if requested
        if sparse_features:
            sparse_mask = np.random.random(n_features) < sparse_ratio
            X_sparse = X.copy()
            X_sparse[:, ~sparse_mask] = 0
            X = X_sparse
        else:
            sparse_mask = np.ones(n_features, dtype=bool)  # All features are relevant
        
        # Create decision boundary
        if n_classes == 2:
            # Binary classification
            relevant_features = X[:, sparse_mask] if sparse_features else X
            decision_boundary = np.sum(relevant_features, axis=1) + 0.5 * np.sum(relevant_features**2, axis=1)
            decision_boundary += noise_level * np.random.normal(0, 1, n_samples)
            y = (decision_boundary > np.median(decision_boundary)).astype(int)
        else:
            # Multi-class classification
            centers = np.random.normal(0, 2, (n_classes, n_features))
            y = np.zeros(n_samples)
            for i in range(n_samples):
                distances = [np.linalg.norm(X[i] - center) for center in centers]
                y[i] = np.argmin(distances)
        
        # Normalize features
        X_mean = tf.reduce_mean(X, axis=0)
        X_std = tf.math.reduce_std(X, axis=0)
        X_normalized = (X - X_mean) / (X_std + 1e-8)
        
        # Split data
        train_size = int(0.8 * n_samples)
        X_train = X_normalized[:train_size]
        X_test = X_normalized[train_size:]
        y_train = y[:train_size]
        y_test = y[train_size:]
        
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def generate_anomaly_detection_data(
        n_normal: int = 1000,
        n_anomalies: int = 50,
        n_features: int = 50,
        random_state: int = 42,
        anomaly_type: str = "outlier"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate synthetic anomaly detection data.
        
        Args:
            n_normal: Number of normal samples
            n_anomalies: Number of anomaly samples
            n_features: Number of features
            random_state: Random seed
            anomaly_type: Type of anomalies ("outlier", "cluster", "drift")
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        np.random.seed(random_state)
        
        # Generate normal data (clustered)
        centers = [np.random.normal(0, 2, n_features) for _ in range(3)]
        normal_data = []
        for center in centers:
            cluster_data = np.random.normal(center, 1.0, (n_normal // 3, n_features))
            normal_data.append(cluster_data)
        
        # Add remaining samples to the last center
        remaining = n_normal - len(normal_data) * (n_normal // 3)
        if remaining > 0:
            last_center = centers[-1]
            remaining_data = np.random.normal(last_center, 1.0, (remaining, n_features))
            normal_data.append(remaining_data)
        
        normal_data = np.vstack(normal_data)
        
        # Generate anomaly data
        if anomaly_type == "outlier":
            anomaly_data = np.random.uniform(-10, 10, (n_anomalies, n_features))
        elif anomaly_type == "cluster":
            anomaly_center = np.random.normal(0, 5, n_features)
            anomaly_data = np.random.normal(anomaly_center, 0.5, (n_anomalies, n_features))
        elif anomaly_type == "drift":
            # Drift: same distribution but shifted
            drift_center = np.random.normal(3, 1, n_features)
            anomaly_data = np.random.normal(drift_center, 1.0, (n_anomalies, n_features))
        else:
            raise ValueError(f"Unknown anomaly type: {anomaly_type}")
        
        # Combine data
        all_data = np.vstack([normal_data, anomaly_data])
        labels = np.hstack([np.zeros(n_normal), np.ones(n_anomalies)])
        
        # Normalize data
        mean = tf.reduce_mean(all_data, axis=0)
        std = tf.math.reduce_std(all_data, axis=0)
        scaled_data = (all_data - mean) / (std + 1e-8)
        
        # Split data
        train_size = int(0.8 * len(scaled_data))
        X_train = scaled_data[:train_size]
        X_test = scaled_data[train_size:]
        y_train = labels[:train_size]
        y_test = labels[train_size:]
        
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def generate_context_data(
        n_samples: int = 1500,
        n_features: int = 15,
        n_context: int = 8,
        random_state: int = 42,
        context_effect: float = 0.3
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate synthetic data with context information.
        
        Args:
            n_samples: Number of samples
            n_features: Number of main features
            n_context: Number of context features
            random_state: Random seed
            context_effect: Strength of context effect
            
        Returns:
            Tuple of (X_train, X_test, context_train, context_test, y_train, y_test)
        """
        np.random.seed(random_state)
        
        # Generate main features
        X = np.random.normal(0, 1, (n_samples, n_features))
        
        # Generate context features (different distribution)
        context = np.random.uniform(-2, 2, (n_samples, n_context))
        
        # Create complex target that depends on both features and context
        context_weights = np.random.normal(0, 1, n_context)
        feature_weights = np.random.normal(0, 1, n_features)
        
        # Create context-dependent decision boundary
        context_effect_val = np.dot(context, context_weights)
        feature_effect = np.dot(X, feature_weights)
        interaction_effect = context_effect * np.sum(X[:, :5] * context[:, :5], axis=1)
        
        # Combine effects
        decision_boundary = feature_effect + context_effect_val + interaction_effect
        y = (decision_boundary > np.median(decision_boundary)).astype(int)
        
        # Normalize features
        X_mean = tf.reduce_mean(X, axis=0)
        X_std = tf.math.reduce_std(X, axis=0)
        X_normalized = (X - X_mean) / (X_std + 1e-8)
        
        context_mean = tf.reduce_mean(context, axis=0)
        context_std = tf.math.reduce_std(context, axis=0)
        context_normalized = (context - context_mean) / (context_std + 1e-8)
        
        # Split data
        train_size = int(0.8 * n_samples)
        X_train = X_normalized[:train_size]
        X_test = X_normalized[train_size:]
        context_train = context_normalized[:train_size]
        context_test = context_normalized[train_size:]
        y_train = y[:train_size]
        y_test = y[train_size:]
        
        return X_train, X_test, context_train, context_test, y_train, y_test
    
    @staticmethod
    def generate_multi_input_data(
        n_samples: int = 1000,
        feature_shapes: Dict[str, Tuple[int, ...]] = None,
        random_state: int = 42,
        task_type: str = "regression"
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """Generate multi-input data for preprocessing model testing.
        
        Args:
            n_samples: Number of samples
            feature_shapes: Dictionary mapping feature names to shapes
            random_state: Random seed
            task_type: Type of task - "regression" or "classification"
            
        Returns:
            Tuple of (X_train_dict, X_test_dict, y_train, y_test)
        """
        if feature_shapes is None:
            feature_shapes = {
                'feature1': (20,),
                'feature2': (15,),
                'feature3': (10,)
            }
        
        np.random.seed(random_state)
        
        X_train_dict = {}
        X_test_dict = {}
        
        # Generate data for each feature
        for feature_name, shape in feature_shapes.items():
            # Generate random data with different distributions for each feature
            if 'feature1' in feature_name:
                data = np.random.normal(0, 1, (n_samples,) + shape)
            elif 'feature2' in feature_name:
                data = np.random.uniform(-2, 2, (n_samples,) + shape)
            else:
                data = np.random.exponential(1, (n_samples,) + shape)
            
            # Normalize
            data_mean = tf.reduce_mean(data, axis=0)
            data_std = tf.math.reduce_std(data, axis=0)
            data_normalized = (data - data_mean) / (data_std + 1e-8)
            
            # Split
            train_size = int(0.8 * n_samples)
            X_train_dict[feature_name] = data_normalized[:train_size]
            X_test_dict[feature_name] = data_normalized[train_size:]
        
        # Generate target based on combined features (use full dataset before splitting)
        combined_features = np.concatenate([np.vstack([X_train_dict[name], X_test_dict[name]]) for name in feature_shapes.keys()], axis=1)
        target_weights = np.random.normal(0, 1, combined_features.shape[1])
        y = np.dot(combined_features, target_weights) + 0.1 * np.random.normal(0, 1, combined_features.shape[0])
        
        # Convert to classification if requested
        if task_type == "classification":
            y = (y > np.median(y)).astype(int)
        
        # Split target
        train_size = int(0.8 * n_samples)
        y_train = y[:train_size]
        y_test = y[train_size:]
        
        return X_train_dict, X_test_dict, y_train, y_test
    
    @staticmethod
    def create_preprocessing_model(
        input_shapes: Dict[str, Tuple[int, ...]],
        output_dim: int = 32,
        name: str = "preprocessing_model"
    ) -> keras.Model:
        """Create a preprocessing model for multi-input data.
        
        Args:
            input_shapes: Dictionary mapping input names to shapes
            output_dim: Output dimension
            name: Model name
            
        Returns:
            Keras preprocessing model
        """
        # Create input layers
        inputs = {}
        processed_inputs = []
        
        for input_name, input_shape in input_shapes.items():
            inputs[input_name] = keras.layers.Input(shape=input_shape, name=input_name)
            
            # Process each input
            if len(input_shape) == 1:
                # 1D input - use dense layers
                x = keras.layers.Dense(16, activation='relu')(inputs[input_name])
                x = keras.layers.Dropout(0.1)(x)
                x = keras.layers.Dense(16, activation='relu')(x)
            else:
                # Multi-dimensional input - use flatten + dense
                x = keras.layers.Flatten()(inputs[input_name])
                x = keras.layers.Dense(32, activation='relu')(x)
                x = keras.layers.Dropout(0.1)(x)
                x = keras.layers.Dense(16, activation='relu')(x)
            
            processed_inputs.append(x)
        
        # Combine processed inputs
        if len(processed_inputs) > 1:
            combined = keras.layers.Concatenate()(processed_inputs)
        else:
            combined = processed_inputs[0]
        
        # Final processing
        output = keras.layers.Dense(output_dim, activation='relu')(combined)
        output = keras.layers.Dropout(0.1)(output)
        
        # Create model
        model = keras.Model(inputs=inputs, outputs=output, name=name)
        
        return model
    
    @staticmethod
    def create_dataset(
        X: Union[np.ndarray, Dict[str, np.ndarray]],
        y: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True
    ) -> tf.data.Dataset:
        """Create a TensorFlow dataset from data.
        
        Args:
            X: Input data (array or dict of arrays)
            y: Target data
            batch_size: Batch size
            shuffle: Whether to shuffle data
            
        Returns:
            TensorFlow dataset
        """
        if isinstance(X, dict):
            # Multi-input data
            dataset = tf.data.Dataset.from_tensor_slices((X, y))
        else:
            # Single input data
            dataset = tf.data.Dataset.from_tensor_slices((X, y))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(y))
        
        dataset = dataset.batch(batch_size)
        
        return dataset
