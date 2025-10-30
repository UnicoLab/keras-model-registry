# 🔗 KDP Integration Guide

Learn how to integrate KMR layers with Keras Data Processor (KDP) for comprehensive tabular data processing workflows.

## 📋 Table of Contents

1. [KDP Overview](#kdp-overview)
2. [Basic Integration](#basic-integration)
3. [Advanced Workflows](#advanced-workflows)
4. [Best Practices](#best-practices)

## 🎯 KDP Overview

Keras Data Processor (KDP) provides powerful data preprocessing capabilities that complement KMR layers perfectly. This integration allows for:

- **Seamless data preprocessing** before KMR layer processing
- **End-to-end pipelines** from raw data to predictions
- **Production-ready workflows** with proper data validation
- **Scalable processing** for large datasets

## 🔧 Basic Integration

### Simple KDP + KMR Pipeline

```python
import keras
import numpy as np
from kmr.layers import TabularAttention, VariableSelection
from keras_data_processor import DataProcessor

def create_kdp_kmr_pipeline(input_dim, num_classes):
    """Create a pipeline combining KDP preprocessing with KMR layers."""
    
    # KDP preprocessing
    processor = DataProcessor(
        numerical_features=['feature_1', 'feature_2', 'feature_3'],
        categorical_features=['category_1', 'category_2'],
        target_column='target'
    )
    
    # KMR model
    inputs = keras.Input(shape=(input_dim,))
    x = VariableSelection(hidden_dim=64)(inputs)
    x = TabularAttention(num_heads=8, key_dim=64)(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    return processor, model

# Usage
processor, model = create_kdp_kmr_pipeline(input_dim=20, num_classes=3)
```

### End-to-End Training Pipeline

```python
def train_kdp_kmr_pipeline(processor, model, X_train, y_train, X_val, y_val):
    """Train a complete KDP + KMR pipeline."""
    
    # Preprocess data with KDP
    X_train_processed = processor.fit_transform(X_train, y_train)
    X_val_processed = processor.transform(X_val)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    history = model.fit(
        X_train_processed, y_train,
        validation_data=(X_val_processed, y_val),
        epochs=100,
        batch_size=32,
        verbose=1
    )
    
    return history

# Usage
history = train_kdp_kmr_pipeline(processor, model, X_train, y_train, X_val, y_val)
```

## 🚀 Advanced Workflows

### Multi-Stage Processing

```python
from kmr.layers import (
    DifferentiableTabularPreprocessor,
    AdvancedNumericalEmbedding,
    GatedFeatureFusion
)

def create_advanced_kdp_kmr_pipeline(input_dim, num_classes):
    """Create an advanced multi-stage pipeline."""
    
    # Stage 1: KDP preprocessing
    processor = DataProcessor(
        numerical_features=['feature_1', 'feature_2', 'feature_3'],
        categorical_features=['category_1', 'category_2'],
        target_column='target',
        preprocessing_steps=[
            'impute_missing',
            'normalize_numerical',
            'encode_categorical'
        ]
    )
    
    # Stage 2: KMR feature engineering
    inputs = keras.Input(shape=(input_dim,))
    x = DifferentiableTabularPreprocessor()(inputs)
    x = AdvancedNumericalEmbedding(embedding_dim=64)(x)
    x = VariableSelection(hidden_dim=64)(x)
    x = TabularAttention(num_heads=8, key_dim=64)(x)
    x = GatedFeatureFusion(hidden_dim=128)(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    return processor, model
```

### Custom Preprocessing Integration

```python
def create_custom_preprocessing_pipeline(input_dim, num_classes):
    """Create a pipeline with custom preprocessing steps."""
    
    # Custom KDP configuration
    processor = DataProcessor(
        numerical_features=['feature_1', 'feature_2', 'feature_3'],
        categorical_features=['category_1', 'category_2'],
        target_column='target',
        custom_preprocessing={
            'feature_1': 'log_transform',
            'feature_2': 'box_cox_transform',
            'category_1': 'target_encoding'
        }
    )
    
    # KMR model with preprocessing
    inputs = keras.Input(shape=(input_dim,))
    x = DifferentiableTabularPreprocessor()(inputs)
    x = AdvancedNumericalEmbedding(embedding_dim=64)(x)
    x = VariableSelection(hidden_dim=64)(x)
    x = TabularAttention(num_heads=8, key_dim=64)(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    return processor, model
```

## 📊 Data Validation and Quality

### Data Quality Checks

```python
def validate_data_quality(processor, X, y):
    """Validate data quality before processing."""
    
    # Check for missing values
    missing_values = X.isnull().sum()
    print("Missing values per feature:")
    print(missing_values)
    
    # Check for outliers
    numerical_features = processor.numerical_features
    for feature in numerical_features:
        Q1 = X[feature].quantile(0.25)
        Q3 = X[feature].quantile(0.75)
        IQR = Q3 - Q1
        outliers = X[(X[feature] < Q1 - 1.5 * IQR) | (X[feature] > Q3 + 1.5 * IQR)]
        print(f"Outliers in {feature}: {len(outliers)}")
    
    # Check data types
    print("Data types:")
    print(X.dtypes)
    
    return True

# Usage
validate_data_quality(processor, X_train, y_train)
```

### Preprocessing Validation

```python
def validate_preprocessing(processor, X_train, y_train, X_test, y_test):
    """Validate preprocessing results."""
    
    # Fit and transform training data
    X_train_processed = processor.fit_transform(X_train, y_train)
    X_test_processed = processor.transform(X_test)
    
    # Check for NaN values
    print("NaN values in processed data:")
    print(f"Training: {X_train_processed.isnull().sum().sum()}")
    print(f"Test: {X_test_processed.isnull().sum().sum()}")
    
    # Check data ranges
    print("Data ranges:")
    print(f"Training min: {X_train_processed.min().min()}")
    print(f"Training max: {X_train_processed.max().max()}")
    print(f"Test min: {X_test_processed.min().min()}")
    print(f"Test max: {X_test_processed.max().max()}")
    
    return X_train_processed, X_test_processed
```

## 🔄 Production Workflows

### Batch Processing

```python
def batch_process_data(processor, model, data_batches):
    """Process data in batches for large datasets."""
    
    results = []
    
    for batch in data_batches:
        # Preprocess batch
        batch_processed = processor.transform(batch)
        
        # Make predictions
        predictions = model.predict(batch_processed)
        
        results.append(predictions)
    
    return np.concatenate(results, axis=0)

# Usage
predictions = batch_process_data(processor, model, data_batches)
```

### Real-Time Processing

```python
def real_time_prediction(processor, model, new_data):
    """Process new data in real-time."""
    
    # Preprocess new data
    processed_data = processor.transform(new_data)
    
    # Make prediction
    prediction = model.predict(processed_data)
    
    return prediction

# Usage
new_prediction = real_time_prediction(processor, model, new_data)
```

## 📈 Performance Monitoring

### Model Performance Tracking

```python
def track_model_performance(processor, model, X_test, y_test):
    """Track model performance over time."""
    
    # Preprocess test data
    X_test_processed = processor.transform(X_test)
    
    # Make predictions
    predictions = model.predict(X_test_processed)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    
    # Calculate metrics
    accuracy = np.mean(predicted_classes == true_classes)
    
    # Log performance
    print(f"Model accuracy: {accuracy:.4f}")
    
    return accuracy

# Usage
accuracy = track_model_performance(processor, model, X_test, y_test)
```

### Data Drift Detection

```python
def detect_data_drift(processor, X_train, X_new):
    """Detect data drift between training and new data."""
    
    # Preprocess both datasets
    X_train_processed = processor.transform(X_train)
    X_new_processed = processor.transform(X_new)
    
    # Calculate statistical differences
    train_mean = X_train_processed.mean()
    new_mean = X_new_processed.mean()
    
    train_std = X_train_processed.std()
    new_std = X_new_processed.std()
    
    # Calculate drift score
    drift_score = np.mean(np.abs(train_mean - new_mean) / train_std)
    
    print(f"Data drift score: {drift_score:.4f}")
    
    if drift_score > 0.1:
        print("Warning: Significant data drift detected!")
    
    return drift_score

# Usage
drift_score = detect_data_drift(processor, X_train, X_new)
```

## 🛠️ Best Practices

### 1. **Data Preprocessing Order**

```python
def recommended_preprocessing_order():
    """Recommended order for data preprocessing."""
    
    # 1. Data validation and quality checks
    # 2. Missing value imputation
    # 3. Outlier detection and handling
    # 4. Feature scaling and normalization
    # 5. Categorical encoding
    # 6. Feature engineering
    # 7. Feature selection
    # 8. Model training
    
    pass
```

### 2. **Error Handling**

```python
def robust_preprocessing(processor, X, y):
    """Robust preprocessing with error handling."""
    
    try:
        # Preprocess data
        X_processed = processor.fit_transform(X, y)
        
        # Validate results
        if X_processed.isnull().any().any():
            raise ValueError("NaN values found in processed data")
        
        return X_processed
        
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return None

# Usage
X_processed = robust_preprocessing(processor, X_train, y_train)
```

### 3. **Memory Management**

```python
def memory_efficient_processing(processor, model, data_generator):
    """Memory-efficient processing for large datasets."""
    
    results = []
    
    for batch in data_generator:
        # Process batch
        batch_processed = processor.transform(batch)
        
        # Make predictions
        predictions = model.predict(batch_processed)
        
        results.append(predictions)
        
        # Clear memory
        del batch_processed
        del predictions
    
    return np.concatenate(results, axis=0)

# Usage
predictions = memory_efficient_processing(processor, model, data_generator)
```

## 📚 Next Steps

1. **Data Analyzer Examples**: Explore data analysis workflows
2. **Rich Docstrings Showcase**: See comprehensive examples
3. **BaseFeedForwardModel Guide**: Learn about feed-forward architectures
4. **API Reference**: Deep dive into layer parameters

---

**Ready for more examples?** Check out the [Data Analyzer Examples](data_analyzer_examples.md) next!