# 📊 Data Analyzer Examples

Comprehensive examples demonstrating data analysis workflows with KMR layers. Learn how to analyze, visualize, and understand your tabular data before building models.

## 📋 Table of Contents

1. [Data Exploration](#data-exploration)
2. [Feature Analysis](#feature-analysis)
3. [Model Interpretation](#model-interpretation)
4. [Performance Analysis](#performance-analysis)

## 🔍 Data Exploration

### Basic Data Analysis

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from kmr.layers import DifferentiableTabularPreprocessor

def analyze_dataset(X, y, feature_names=None):
    """Comprehensive dataset analysis."""
    
    # Basic statistics
    print("Dataset Shape:", X.shape)
    print("Target Distribution:", np.bincount(y))
    
    # Missing values
    missing_values = pd.DataFrame(X).isnull().sum()
    print("Missing Values:")
    print(missing_values)
    
    # Data types
    print("Data Types:")
    print(pd.DataFrame(X).dtypes)
    
    # Basic statistics
    print("Basic Statistics:")
    print(pd.DataFrame(X).describe())
    
    return True

# Usage
analyze_dataset(X_train, y_train, feature_names)
```

### Feature Distribution Analysis

```python
def analyze_feature_distributions(X, feature_names=None):
    """Analyze feature distributions."""
    
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    
    # Plot distributions
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, feature in enumerate(feature_names[:4]):
        df[feature].hist(ax=axes[i], bins=30)
        axes[i].set_title(f'Distribution of {feature}')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
    # Statistical analysis
    for feature in feature_names:
        print(f"\n{feature}:")
        print(f"  Mean: {df[feature].mean():.4f}")
        print(f"  Std: {df[feature].std():.4f}")
        print(f"  Skewness: {df[feature].skew():.4f}")
        print(f"  Kurtosis: {df[feature].kurtosis():.4f}")

# Usage
analyze_feature_distributions(X_train, feature_names)
```

## 🔧 Feature Analysis

### Feature Importance Analysis

```python
from kmr.layers import VariableSelection, TabularAttention

def analyze_feature_importance(model, X_test, feature_names=None):
    """Analyze feature importance using model weights."""
    
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(X_test.shape[1])]
    
    # Get variable selection layer
    try:
        selection_layer = model.get_layer('variable_selection')
        selection_weights = selection_layer.get_weights()
        
        # Calculate feature importance
        feature_importance = np.mean(selection_weights[0], axis=1)
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        print("Feature Importance:")
        print(importance_df)
        
        # Plot importance
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df, x='importance', y='feature')
        plt.title('Feature Importance')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.show()
        
        return importance_df
        
    except Exception as e:
        print(f"Could not analyze feature importance: {e}")
        return None

# Usage
importance_df = analyze_feature_importance(model, X_test, feature_names)
```

### Attention Weight Analysis

```python
def analyze_attention_weights(model, X_test, feature_names=None):
    """Analyze attention weights for model interpretation."""
    
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(X_test.shape[1])]
    
    # Get attention layer
    try:
        attention_layer = model.get_layer('tabular_attention')
        
        # Create model that outputs attention weights
        attention_model = keras.Model(
            inputs=model.input,
            outputs=attention_layer.output
        )
        
        # Get attention weights
        attention_weights = attention_model.predict(X_test)
        
        # Analyze attention patterns
        mean_attention = np.mean(attention_weights, axis=0)
        std_attention = np.std(attention_weights, axis=0)
        
        # Create attention DataFrame
        attention_df = pd.DataFrame({
            'feature': feature_names,
            'mean_attention': mean_attention,
            'std_attention': std_attention
        }).sort_values('mean_attention', ascending=False)
        
        print("Attention Weights Analysis:")
        print(attention_df)
        
        # Plot attention weights
        plt.figure(figsize=(12, 6))
        sns.barplot(data=attention_df, x='mean_attention', y='feature')
        plt.title('Mean Attention Weights')
        plt.xlabel('Attention Weight')
        plt.tight_layout()
        plt.show()
        
        return attention_df
        
    except Exception as e:
        print(f"Could not analyze attention weights: {e}")
        return None

# Usage
attention_df = analyze_attention_weights(model, X_test, feature_names)
```

## 🧠 Model Interpretation

### Layer Output Analysis

```python
def analyze_layer_outputs(model, X_test, layer_names):
    """Analyze outputs from different model layers."""
    
    layer_outputs = {}
    
    for layer_name in layer_names:
        try:
            # Get layer
            layer = model.get_layer(layer_name)
            
            # Create model that outputs layer activations
            layer_model = keras.Model(
                inputs=model.input,
                outputs=layer.output
            )
            
            # Get layer outputs
            layer_output = layer_model.predict(X_test)
            layer_outputs[layer_name] = layer_output
            
            print(f"{layer_name} output shape: {layer_output.shape}")
            
        except Exception as e:
            print(f"Could not analyze layer {layer_name}: {e}")
    
    return layer_outputs

# Usage
layer_names = ['variable_selection', 'tabular_attention', 'gated_feature_fusion']
layer_outputs = analyze_layer_outputs(model, X_test, layer_names)
```

### Model Decision Analysis

```python
def analyze_model_decisions(model, X_test, y_test, feature_names=None):
    """Analyze model decision-making process."""
    
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(X_test.shape[1])]
    
    # Get predictions
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    
    # Analyze prediction confidence
    prediction_confidence = np.max(predictions, axis=1)
    
    print("Prediction Confidence Analysis:")
    print(f"Mean confidence: {np.mean(prediction_confidence):.4f}")
    print(f"Std confidence: {np.std(prediction_confidence):.4f}")
    print(f"Min confidence: {np.min(prediction_confidence):.4f}")
    print(f"Max confidence: {np.max(prediction_confidence):.4f}")
    
    # Analyze misclassifications
    misclassified = predicted_classes != true_classes
    misclassified_indices = np.where(misclassified)[0]
    
    print(f"\nMisclassified samples: {len(misclassified_indices)}")
    print(f"Misclassification rate: {len(misclassified_indices) / len(y_test):.4f}")
    
    # Analyze confidence of misclassified samples
    if len(misclassified_indices) > 0:
        misclassified_confidence = prediction_confidence[misclassified_indices]
        print(f"Mean confidence of misclassified: {np.mean(misclassified_confidence):.4f}")
    
    return {
        'predictions': predictions,
        'predicted_classes': predicted_classes,
        'true_classes': true_classes,
        'confidence': prediction_confidence,
        'misclassified_indices': misclassified_indices
    }

# Usage
decision_analysis = analyze_model_decisions(model, X_test, y_test, feature_names)
```

## 📈 Performance Analysis

### Training Performance Analysis

```python
def analyze_training_performance(history):
    """Analyze training performance and convergence."""
    
    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curve
    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy curve
    axes[1].plot(history.history['accuracy'], label='Training Accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[1].set_title('Model Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Analyze convergence
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    
    print("Final Performance:")
    print(f"Training Loss: {final_train_loss:.4f}")
    print(f"Validation Loss: {final_val_loss:.4f}")
    print(f"Training Accuracy: {final_train_acc:.4f}")
    print(f"Validation Accuracy: {final_val_acc:.4f}")
    
    # Check for overfitting
    if final_val_loss > final_train_loss * 1.1:
        print("Warning: Possible overfitting detected!")
    
    return {
        'final_train_loss': final_train_loss,
        'final_val_loss': final_val_loss,
        'final_train_acc': final_train_acc,
        'final_val_acc': final_val_acc
    }

# Usage
performance_analysis = analyze_training_performance(history)
```

### Model Comparison Analysis

```python
def compare_models(models, X_test, y_test, model_names=None):
    """Compare performance of multiple models."""
    
    if model_names is None:
        model_names = [f'model_{i}' for i in range(len(models))]
    
    results = []
    
    for model, name in zip(models, model_names):
        # Evaluate model
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        
        # Get predictions
        predictions = model.predict(X_test)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y_test, axis=1)
        
        # Calculate additional metrics
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        precision = precision_score(true_classes, predicted_classes, average='weighted')
        recall = recall_score(true_classes, predicted_classes, average='weighted')
        f1 = f1_score(true_classes, predicted_classes, average='weighted')
        
        results.append({
            'model': name,
            'accuracy': test_accuracy,
            'loss': test_loss,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results)
    
    print("Model Comparison:")
    print(comparison_df)
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy comparison
    sns.barplot(data=comparison_df, x='model', y='accuracy', ax=axes[0, 0])
    axes[0, 0].set_title('Accuracy Comparison')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Loss comparison
    sns.barplot(data=comparison_df, x='model', y='loss', ax=axes[0, 1])
    axes[0, 1].set_title('Loss Comparison')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Precision comparison
    sns.barplot(data=comparison_df, x='model', y='precision', ax=axes[1, 0])
    axes[1, 0].set_title('Precision Comparison')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # F1 comparison
    sns.barplot(data=comparison_df, x='model', y='f1', ax=axes[1, 1])
    axes[1, 1].set_title('F1 Score Comparison')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return comparison_df

# Usage
models = [model1, model2, model3]
model_names = ['Attention Model', 'Residual Model', 'Ensemble Model']
comparison_df = compare_models(models, X_test, y_test, model_names)
```

## 🔍 Advanced Analysis

### Feature Interaction Analysis

```python
def analyze_feature_interactions(X, feature_names=None):
    """Analyze feature interactions and correlations."""
    
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    
    # Calculate correlation matrix
    correlation_matrix = df.corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
    # Find highly correlated features
    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_value = correlation_matrix.iloc[i, j]
            if abs(corr_value) > 0.7:  # High correlation threshold
                high_corr_pairs.append({
                    'feature1': correlation_matrix.columns[i],
                    'feature2': correlation_matrix.columns[j],
                    'correlation': corr_value
                })
    
    if high_corr_pairs:
        print("Highly Correlated Feature Pairs:")
        for pair in high_corr_pairs:
            print(f"{pair['feature1']} - {pair['feature2']}: {pair['correlation']:.4f}")
    
    return correlation_matrix, high_corr_pairs

# Usage
correlation_matrix, high_corr_pairs = analyze_feature_interactions(X_train, feature_names)
```

### Model Robustness Analysis

```python
def analyze_model_robustness(model, X_test, y_test, noise_levels=[0.01, 0.05, 0.1]):
    """Analyze model robustness to noise."""
    
    results = []
    
    for noise_level in noise_levels:
        # Add noise to test data
        X_noisy = X_test + np.random.normal(0, noise_level, X_test.shape)
        
        # Evaluate model on noisy data
        test_loss, test_accuracy = model.evaluate(X_noisy, y_test, verbose=0)
        
        results.append({
            'noise_level': noise_level,
            'accuracy': test_accuracy,
            'loss': test_loss
        })
    
    # Create results DataFrame
    robustness_df = pd.DataFrame(results)
    
    print("Model Robustness Analysis:")
    print(robustness_df)
    
    # Plot robustness
    plt.figure(figsize=(10, 6))
    plt.plot(robustness_df['noise_level'], robustness_df['accuracy'], 'o-', label='Accuracy')
    plt.plot(robustness_df['noise_level'], robustness_df['loss'], 's-', label='Loss')
    plt.xlabel('Noise Level')
    plt.ylabel('Performance')
    plt.title('Model Robustness to Noise')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return robustness_df

# Usage
robustness_df = analyze_model_robustness(model, X_test, y_test)
```

## 📚 Next Steps

1. **Rich Docstrings Showcase**: See comprehensive examples
2. **BaseFeedForwardModel Guide**: Learn about feed-forward architectures
3. **KDP Integration Guide**: Integrate with Keras Data Processor
4. **API Reference**: Deep dive into layer parameters

---

**Ready for more examples?** Check out the [Rich Docstrings Showcase](rich_docstrings_showcase.md) next!