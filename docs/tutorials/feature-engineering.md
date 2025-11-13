# 🔧 Feature Engineering Tutorial

Master the art of feature engineering with KerasFactory layers. Learn how to transform, select, and create powerful features for your tabular models.

## 📋 Table of Contents

1. [Understanding Feature Engineering](#understanding-feature-engineering)
2. [Numerical Feature Processing](#numerical-feature-processing)
3. [Categorical Feature Handling](#categorical-feature-handling)
4. [Feature Selection Techniques](#feature-selection-techniques)
5. [Advanced Feature Creation](#advanced-feature-creation)
6. [Best Practices](#best-practices)

## 🎯 Understanding Feature Engineering

Feature engineering is the process of creating, transforming, and selecting features to improve model performance. KerasFactory provides specialized layers for this purpose.

### Why Feature Engineering Matters

```python
import numpy as np
import pandas as pd
from kerasfactory.layers import AdvancedNumericalEmbedding, DistributionAwareEncoder

# Example: Raw features vs Engineered features
raw_features = np.random.normal(0, 1, (1000, 10))

# Raw features - limited representation
print("Raw features shape:", raw_features.shape)

# Engineered features - richer representation
embedding_layer = AdvancedNumericalEmbedding(embedding_dim=64)
engineered_features = embedding_layer(raw_features)
print("Engineered features shape:", engineered_features.shape)
```

## 🔢 Numerical Feature Processing

### 1. **Advanced Numerical Embedding**

Transform numerical features into rich embeddings:

```python
from kerasfactory.layers import AdvancedNumericalEmbedding

def create_numerical_embedding(input_dim, embedding_dim=64):
    """Create numerical feature embeddings."""
    
    inputs = keras.Input(shape=(input_dim,))
    
    # Advanced numerical embedding
    x = AdvancedNumericalEmbedding(
        embedding_dim=embedding_dim,
        num_bins=10,
        hidden_dim=128
    )(inputs)
    
    return keras.Model(inputs, x)

# Usage
embedding_model = create_numerical_embedding(input_dim=20, embedding_dim=64)
```

### 2. **Distribution-Aware Encoding**

Automatically detect and encode feature distributions:

```python
from kerasfactory.layers import DistributionAwareEncoder

def create_distribution_aware_encoding(input_dim):
    """Create distribution-aware feature encoding."""
    
    inputs = keras.Input(shape=(input_dim,))
    
    # Distribution-aware encoding
    x = DistributionAwareEncoder(
        encoding_dim=64,
        detection_method='auto'
    )(inputs)
    
    return keras.Model(inputs, x)

# Usage
distribution_model = create_distribution_aware_encoding(input_dim=20)
```

### 3. **Distribution Transformation**

Transform features to follow specific distributions:

```python
from kerasfactory.layers import DistributionTransformLayer

def create_distribution_transform(input_dim):
    """Transform features to normal distribution."""
    
    inputs = keras.Input(shape=(input_dim,))
    
    # Distribution transformation
    x = DistributionTransformLayer(
        transform_type='auto',
        method='box-cox'
    )(inputs)
    
    return keras.Model(inputs, x)

# Usage
transform_model = create_distribution_transform(input_dim=20)
```

## 🏷️ Categorical Feature Handling

### 1. **Date and Time Features**

Process temporal features effectively:

```python
from kerasfactory.layers import DateParsingLayer, DateEncodingLayer, SeasonLayer

def create_temporal_features():
    """Create comprehensive temporal feature processing."""
    
    # Date parsing
    date_parser = DateParsingLayer()
    
    # Date encoding
    date_encoder = DateEncodingLayer(min_year=1900, max_year=2100)
    
    # Season extraction
    season_layer = SeasonLayer()
    
    return date_parser, date_encoder, season_layer

# Usage
date_parser, date_encoder, season_layer = create_temporal_features()

# Process date strings
date_strings = ['2023-01-15', '2023-06-20', '2023-12-25']
parsed_dates = date_parser(date_strings)
encoded_dates = date_encoder(parsed_dates)
seasonal_features = season_layer(parsed_dates)
```

### 2. **Text Preprocessing**

Handle text features in tabular data:

```python
from kerasfactory.layers import TextPreprocessingLayer

def create_text_preprocessing():
    """Create text preprocessing pipeline."""
    
    text_preprocessor = TextPreprocessingLayer(
        max_length=100,
        vocab_size=10000,
        tokenizer='word'
    )
    
    return text_preprocessor

# Usage
text_preprocessor = create_text_preprocessing()
```

## 🎯 Feature Selection Techniques

### 1. **Variable Selection**

Intelligently select important features:

```python
from kerasfactory.layers import VariableSelection

def create_variable_selection(input_dim, hidden_dim=64):
    """Create intelligent variable selection."""
    
    inputs = keras.Input(shape=(input_dim,))
    
    # Variable selection
    x = VariableSelection(
        hidden_dim=hidden_dim,
        dropout=0.1,
        use_context=False
    )(inputs)
    
    return keras.Model(inputs, x)

# Usage
selection_model = create_variable_selection(input_dim=20, hidden_dim=64)
```

### 2. **Gated Feature Selection**

Learnable feature selection with gating:

```python
from kerasfactory.layers import GatedFeatureSelection

def create_gated_selection(input_dim, hidden_dim=64):
    """Create gated feature selection."""
    
    inputs = keras.Input(shape=(input_dim,))
    
    # Gated feature selection
    x = GatedFeatureSelection(
        hidden_dim=hidden_dim,
        dropout=0.1,
        activation='relu'
    )(inputs)
    
    return keras.Model(inputs, x)

# Usage
gated_model = create_gated_selection(input_dim=20, hidden_dim=64)
```

### 3. **Sparse Attention Weighting**

Sparse feature weighting for efficiency:

```python
from kerasfactory.layers import SparseAttentionWeighting

def create_sparse_weighting(input_dim):
    """Create sparse attention weighting."""
    
    inputs = keras.Input(shape=(input_dim,))
    
    # Sparse attention weighting
    x = SparseAttentionWeighting(
        temperature=1.0,
        dropout=0.1,
        sparsity_threshold=0.1
    )(inputs)
    
    return keras.Model(inputs, x)

# Usage
sparse_model = create_sparse_weighting(input_dim=20)
```

## 🚀 Advanced Feature Creation

### 1. **Feature Fusion**

Combine multiple feature representations:

```python
from kerasfactory.layers import GatedFeatureFusion

def create_feature_fusion(input_dim1, input_dim2, hidden_dim=128):
    """Create feature fusion mechanism."""
    
    inputs1 = keras.Input(shape=(input_dim1,))
    inputs2 = keras.Input(shape=(input_dim2,))
    
    # Feature fusion
    x = GatedFeatureFusion(
        hidden_dim=hidden_dim,
        dropout=0.1,
        activation='relu'
    )([inputs1, inputs2])
    
    return keras.Model([inputs1, inputs2], x)

# Usage
fusion_model = create_feature_fusion(input_dim1=10, input_dim2=10, hidden_dim=128)
```

### 2. **Feature Cutout**

Data augmentation for features:

```python
from kerasfactory.layers import FeatureCutout

def create_feature_augmentation(input_dim):
    """Create feature augmentation pipeline."""
    
    inputs = keras.Input(shape=(input_dim,))
    
    # Feature cutout for augmentation
    x = FeatureCutout(
        cutout_prob=0.1,
        noise_value=0.0,
        training_only=True
    )(inputs)
    
    return keras.Model(inputs, x)

# Usage
augmentation_model = create_feature_augmentation(input_dim=20)
```

### 3. **Graph-Based Features**

Process features as a graph:

```python
from kerasfactory.layers import AdvancedGraphFeature, GraphFeatureAggregation

def create_graph_features(input_dim, hidden_dim=64):
    """Create graph-based feature processing."""
    
    inputs = keras.Input(shape=(input_dim,))
    
    # Graph feature processing
    x = AdvancedGraphFeature(
        hidden_dim=hidden_dim,
        num_heads=4,
        dropout=0.1
    )(inputs)
    
    # Graph aggregation
    x = GraphFeatureAggregation(
        aggregation_method='mean',
        hidden_dim=hidden_dim
    )(x)
    
    return keras.Model(inputs, x)

# Usage
graph_model = create_graph_features(input_dim=20, hidden_dim=64)
```

## 🎛️ Complete Feature Engineering Pipeline

### End-to-End Pipeline

```python
def create_complete_feature_pipeline(input_dim, num_classes):
    """Create a complete feature engineering pipeline."""
    
    inputs = keras.Input(shape=(input_dim,))
    
    # 1. Preprocessing
    x = DifferentiableTabularPreprocessor()(inputs)
    
    # 2. Numerical embedding
    x = AdvancedNumericalEmbedding(embedding_dim=64)(x)
    
    # 3. Distribution-aware encoding
    x = DistributionAwareEncoder(encoding_dim=64)(x)
    
    # 4. Variable selection
    x = VariableSelection(hidden_dim=64)(x)
    
    # 5. Sparse attention weighting
    x = SparseAttentionWeighting(temperature=1.0)(x)
    
    # 6. Feature fusion
    x = GatedFeatureFusion(hidden_dim=128)(x)
    
    # 7. Output layer
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return keras.Model(inputs, outputs)

# Usage
pipeline_model = create_complete_feature_pipeline(input_dim=20, num_classes=3)
pipeline_model.summary()
```

### Multi-Branch Pipeline

```python
def create_multi_branch_pipeline(input_dim, num_classes):
    """Create a multi-branch feature engineering pipeline."""
    
    inputs = keras.Input(shape=(input_dim,))
    
    # Branch 1: Numerical processing
    branch1 = AdvancedNumericalEmbedding(embedding_dim=64)(inputs)
    branch1 = DistributionAwareEncoder(encoding_dim=64)(branch1)
    
    # Branch 2: Selection processing
    branch2 = VariableSelection(hidden_dim=64)(inputs)
    branch2 = GatedFeatureSelection(hidden_dim=64)(branch2)
    
    # Branch 3: Graph processing
    branch3 = AdvancedGraphFeature(hidden_dim=64)(inputs)
    branch3 = GraphFeatureAggregation(hidden_dim=64)(branch3)
    
    # Fusion
    x = GatedFeatureFusion(hidden_dim=128)([branch1, branch2, branch3])
    
    # Output
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return keras.Model(inputs, outputs)

# Usage
multi_branch_model = create_multi_branch_pipeline(input_dim=20, num_classes=3)
```

## 📊 Best Practices

### 1. **Start Simple, Add Complexity**

```python
# Start with basic preprocessing
def basic_pipeline(inputs):
    x = DifferentiableTabularPreprocessor()(inputs)
    x = VariableSelection(hidden_dim=32)(x)
    return x

# Gradually add complexity
def advanced_pipeline(inputs):
    x = DifferentiableTabularPreprocessor()(inputs)
    x = AdvancedNumericalEmbedding(embedding_dim=64)(x)
    x = VariableSelection(hidden_dim=64)(x)
    x = SparseAttentionWeighting()(x)
    return x
```

### 2. **Monitor Feature Importance**

```python
# Use attention weights to understand feature importance
def create_interpretable_model(input_dim):
    inputs = keras.Input(shape=(input_dim,))
    
    # Attention layer with weights
    x, attention_weights = TabularAttention(
        num_heads=8,
        key_dim=64,
        use_attention_weights=True
    )(inputs)
    
    return keras.Model(inputs, [x, attention_weights])

# Get attention weights
model = create_interpretable_model(input_dim=20)
outputs, attention_weights = model.predict(X_test)
print("Attention weights shape:", attention_weights.shape)
```

### 3. **Feature Engineering Validation**

```python
# Validate feature engineering impact
def compare_models(X_train, y_train, X_test, y_test):
    """Compare models with and without feature engineering."""
    
    # Model 1: Raw features
    inputs1 = keras.Input(shape=(X_train.shape[1],))
    x1 = keras.layers.Dense(64, activation='relu')(inputs1)
    x1 = keras.layers.Dense(32, activation='relu')(x1)
    outputs1 = keras.layers.Dense(3, activation='softmax')(x1)
    model1 = keras.Model(inputs1, outputs1)
    
    # Model 2: With feature engineering
    inputs2 = keras.Input(shape=(X_train.shape[1],))
    x2 = DifferentiableTabularPreprocessor()(inputs2)
    x2 = AdvancedNumericalEmbedding(embedding_dim=64)(x2)
    x2 = VariableSelection(hidden_dim=64)(x2)
    x2 = TabularAttention(num_heads=8, key_dim=64)(x2)
    outputs2 = keras.layers.Dense(3, activation='softmax')(x2)
    model2 = keras.Model(inputs2, outputs2)
    
    # Compile and train both models
    for model in [model1, model2]:
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        model.fit(X_train, y_train, epochs=10, verbose=0)
    
    # Compare performance
    score1 = model1.evaluate(X_test, y_test, verbose=0)
    score2 = model2.evaluate(X_test, y_test, verbose=0)
    
    print(f"Raw features accuracy: {score1[1]:.4f}")
    print(f"Engineered features accuracy: {score2[1]:.4f}")
    
    return model1, model2

# Usage
model1, model2 = compare_models(X_train, y_train, X_test, y_test)
```

## 📚 Next Steps

1. **Model Building**: Learn advanced model architectures
2. **Examples**: See real-world feature engineering applications
3. **API Reference**: Deep dive into layer parameters
4. **Performance**: Optimize your feature engineering pipeline

---

**Ready to build models?** Check out [Model Building](model-building.md) next!
