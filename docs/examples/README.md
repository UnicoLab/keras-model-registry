# 📚 Examples

Real-world examples and use cases demonstrating KMR layers in action. These examples show how to build production-ready tabular models for various domains and applications.

## 🎯 Quick Navigation

- [Rich Docstrings Showcase](rich_docstrings_showcase.md) - Comprehensive examples with detailed documentation
- [BaseFeedForwardModel Guide](feed_forward_guide.md) - Building feed-forward models with KMR
- [KDP Integration Guide](kdp_integration_guide.md) - Integrating with Keras Data Processor
- [Data Analyzer Examples](data_analyzer_examples.md) - Data analysis and preprocessing workflows

## 🚀 Getting Started

### 1. **Basic Classification**

```python
import keras
from kmr.layers import TabularAttention, VariableSelection

# Simple classification model
def create_classifier(input_dim: int, num_classes: int) -> keras.Model:
    inputs = keras.Input(shape=(input_dim,))
    x = VariableSelection(hidden_dim=64)(inputs)
    x = TabularAttention(num_heads=8, key_dim=64)(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    return keras.Model(inputs, outputs)

# Usage
model = create_classifier(
    input_dim=20,
    num_classes=3,
)
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)
```

### 2. **Regression with Feature Engineering**

```python
from kmr.layers import (
    DifferentiableTabularPreprocessor,
    AdvancedNumericalEmbedding,
    GatedFeatureFusion
)

def create_regressor(input_dim: int) -> keras.Model:
    inputs = keras.Input(shape=(input_dim,))

    x = DifferentiableTabularPreprocessor()(inputs)
    x = AdvancedNumericalEmbedding(embedding_dim=64)(x)
    x = GatedFeatureFusion(hidden_dim=128)(x)

    outputs = keras.layers.Dense(1)(x)

    return keras.Model(inputs, outputs)

# Usage
model = create_regressor(
    input_dim=20,
)
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae'],
)
```

## 🏗️ Architecture Examples

### 1. **Attention-Based Architecture**

```python
from kmr.layers import (
    MultiResolutionTabularAttention,
    InterpretableMultiHeadAttention,
    GatedFeatureFusion
)

def create_attention_model(input_dim, num_classes):
    inputs = keras.Input(shape=(input_dim,))
    
    # Multi-resolution attention
    x = MultiResolutionTabularAttention(
        num_heads=8,
        numerical_heads=4,
        categorical_heads=4,
    )(inputs)
    
    # Interpretable attention
    x = InterpretableMultiHeadAttention(
        num_heads=8,
        key_dim=64,
    )(x)
    
    # Feature fusion
    x = GatedFeatureFusion(hidden_dim=128)(x)
    
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    return keras.Model(inputs, outputs)
```

### 2. **Residual Network Architecture**

```python
from kmr.layers import GatedResidualNetwork, GatedLinearUnit

def create_residual_model(input_dim: int, num_classes: int) -> keras.Model:
    inputs = keras.Input(shape=(input_dim,))
    
    # Residual blocks
    x = GatedResidualNetwork(units=64, dropout_rate=0.1)(inputs)
    x = GatedResidualNetwork(units=64, dropout_rate=0.1)(x)
    x = GatedResidualNetwork(units=64, dropout_rate=0.1)(x)
    
    # Gated linear unit
    x = GatedLinearUnit(units=64)(x)
    
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    return keras.Model(inputs, outputs)
```

### 3. **Ensemble Architecture**

```python
from kmr.layers import TabularMoELayer, BoostingEnsembleLayer

def create_ensemble_model(input_dim: int, num_classes: int) -> keras.Model:
    inputs = keras.Input(shape=(input_dim,))
    
    # Mixture of experts
    x = TabularMoELayer(num_experts=4, expert_units=16)(inputs)
    
    # Boosting ensemble
    x = BoostingEnsembleLayer(
        num_learners=3,
        learner_units=64
    )(x)
    
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    return keras.Model(inputs, outputs)
```

## 🔧 Feature Engineering Examples

### 1. **Complete Feature Pipeline**

```python
from kmr.layers import (
    DifferentiableTabularPreprocessor,
    AdvancedNumericalEmbedding,
    DistributionAwareEncoder,
    VariableSelection,
    SparseAttentionWeighting
)

def create_feature_pipeline(input_dim: int) -> keras.Model:
    inputs = keras.Input(shape=(input_dim,))
    
    # Preprocessing
    x = DifferentiableTabularPreprocessor()(inputs)
    
    # Numerical embedding
    x = AdvancedNumericalEmbedding(embedding_dim=64)(x)
    
    # Distribution-aware encoding
    x = DistributionAwareEncoder(encoding_dim=64)(x)
    
    # Variable selection
    x = VariableSelection(hidden_dim=64)(x)
    
    # Sparse attention weighting
    x = SparseAttentionWeighting(temperature=1.0)(x)
    
    return keras.Model(inputs, x)
```

### 2. **Temporal Feature Processing**

```python
from kmr.layers import (
    DateParsingLayer,
    DateEncodingLayer,
    SeasonLayer
)

def create_temporal_pipeline() -> keras.Model:
    # Date parsing
    date_parser = DateParsingLayer()
    
    # Date encoding
    date_encoder = DateEncodingLayer(min_year=1900, max_year=2100)
    
    # Season extraction
    season_layer = SeasonLayer()
    
    return date_parser, date_encoder, season_layer

# Usage
date_parser, date_encoder, season_layer = create_temporal_pipeline()
```

## 🎯 Domain-Specific Examples

### 1. **Financial Modeling**

```python
def create_financial_model(input_dim: int , num_classes: int) -> keras.Model:
    """Model for financial risk assessment."""
    
    inputs = keras.Input(shape=(input_dim,))
    
    # Preprocessing for financial data
    x = DifferentiableTabularPreprocessor()(inputs)
    
    # Feature selection for risk factors
    x = VariableSelection(hidden_dim=64)(x)
    
    # Attention for complex relationships
    x = TabularAttention(num_heads=8, key_dim=64)(x)
    
    # Business rules integration
    x = BusinessRulesLayer(
        rules=[
            {'feature': 'credit_score', 'operator': '>', 'value': 600, 'weight': 1.0},
            {'feature': 'debt_ratio', 'operator': '<', 'value': 0.4, 'weight': 0.8}
        ],
        feature_type='numerical',
    )(x)
    
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    return keras.Model(inputs, outputs)
```

### 2. **Healthcare Analytics**

```python
def create_healthcare_model(input_dim: int, num_classes: int) -> keras.Model:
    """Model for healthcare outcome prediction."""
    
    inputs = keras.Input(shape=(input_dim,))
    
    # Preprocessing
    x = DifferentiableTabularPreprocessor()(inputs)
    
    # Advanced numerical embedding for medical features
    x = AdvancedNumericalEmbedding(embedding_dim=64)(x)
    
    # Distribution-aware encoding for lab values
    x = DistributionAwareEncoder(encoding_dim=64)(x)
    
    # Attention for symptom relationships
    x = TabularAttention(num_heads=8, key_dim=64)(x)
    
    # Anomaly detection for outliers
    x, anomalies = NumericalAnomalyDetection()(x)
    
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    return keras.Model(inputs, [outputs, anomalies])
```

### 3. **E-commerce Recommendation**

```python
def create_recommendation_model(input_dim: int, num_classes: int) -> keras.Model:
    """Model for e-commerce product recommendation."""
    
    inputs = keras.Input(shape=(input_dim,))
    
    # Preprocessing
    x = DifferentiableTabularPreprocessor()(inputs)
    
    # Feature selection for user preferences
    x = VariableSelection(hidden_dim=64)(x)
    
    # Multi-resolution attention for different feature types
    x = MultiResolutionTabularAttention(
        num_heads=8,
        numerical_heads=4,
        categorical_heads=4
    )(x)
    
    # Feature fusion for recommendation
    x = GatedFeatureFusion(hidden_dim=128)(x)
    
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    return keras.Model(inputs, outputs)
```

## 🚀 Performance Examples

### 1. **Memory-Efficient Model**

```python
def create_memory_efficient_model(input_dim: int, num_classes: int) -> keras.Model:
    """Memory-efficient model for large datasets."""
    
    inputs = keras.Input(shape=(input_dim,))
    
    # Use smaller dimensions
    x = VariableSelection(hidden_dim=32)(inputs)
    x = TabularAttention(num_heads=4, key_dim=32)(x)
    x = GatedFeatureFusion(hidden_dim=64)(x)
    
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    return keras.Model(inputs, outputs)
```

### 2. **Speed-Optimized Model**

```python
def create_speed_optimized_model(input_dim: int, num_classes: int) -> keras.Model:
    """Speed-optimized model for real-time inference."""
    
    inputs = keras.Input(shape=(input_dim,))
    
    # Minimal layers for speed
    x = VariableSelection(hidden_dim=32)(inputs)
    x = TabularAttention(num_heads=4, key_dim=32)(x)
    
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    return keras.Model(inputs, outputs)
```

## 🔍 Analysis and Interpretation

### 1. **Model Interpretation**

```python
def interpret_model(model, X_test, layer_name='tabular_attention'):
    """Interpret model using attention weights."""
    
    # Get attention weights
    attention_model = keras.Model(
        inputs=model.input,
        outputs=model.get_layer(layer_name).output,
    )
    
    attention_weights = attention_model.predict(X_test)
    
    # Analyze attention patterns
    mean_attention = np.mean(attention_weights, axis=0)
    print("Mean attention weights:", mean_attention)
    
    return attention_weights
```

### 2. **Feature Importance Analysis**

```python
def analyze_feature_importance(model, X_test, feature_names):
    """Analyze feature importance using attention weights."""
    
    # Get attention weights
    attention_weights = interpret_model(model, X_test)
    
    # Calculate feature importance
    feature_importance = np.mean(attention_weights, axis=(0, 1))
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    return importance_df
```

## 📊 Evaluation Examples

### 1. **Comprehensive Evaluation**

```python
def evaluate_model_comprehensive(model, X_test, y_test):
    """Comprehensive model evaluation."""
    
    # Basic evaluation
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    # Predictions
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    
    # Additional metrics
    from sklearn.metrics import classification_report, confusion_matrix
    
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print("\nClassification Report:")
    print(classification_report(true_classes, predicted_classes))
    
    return test_accuracy, test_loss
```

### 2. **Cross-Validation**

```python
from sklearn.model_selection import cross_val_score

def cross_validate_model(model, X, y, cv=5):
    """Cross-validation for model evaluation."""
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Cross-validation
    scores = cross_val_score(
        model, X, y, 
        cv=cv, 
        scoring='accuracy',
        verbose=0
    )
    
    print(f"Cross-validation scores: {scores}")
    print(f"Mean accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    return scores
```

## 📚 Next Steps

1. **Rich Docstrings Showcase**: See comprehensive examples with detailed documentation
2. **BaseFeedForwardModel Guide**: Learn about feed-forward model architectures
3. **KDP Integration Guide**: Integrate with Keras Data Processor
4. **Data Analyzer Examples**: Explore data analysis workflows

---

**Ready to dive deeper?** Check out the [Rich Docstrings Showcase](rich_docstrings_showcase.md) for comprehensive examples!