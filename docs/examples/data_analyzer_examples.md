# 📊 Data Analyzer Examples

Comprehensive examples demonstrating data analysis workflows with KerasFactory layers. Learn how to analyze, visualize, and understand your tabular data before building models.

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
import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import plot
from loguru import logger
from typing import Optional, Dict, Tuple, List, Any
import keras
from kerasfactory.layers import DifferentiableTabularPreprocessor

def analyze_dataset(X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> bool:
    """Comprehensive dataset analysis.
    
    Analyzes a dataset by computing basic statistics, missing values, data types,
    and statistical summaries. Useful for initial data exploration before model building.
    
    Args:
        X: Input feature array of shape (n_samples, n_features).
        y: Target array of shape (n_samples,).
        feature_names: Optional list of feature names. If None, auto-generated names are used.
    
    Returns:
        bool: True if analysis completed successfully.
        
    Example:
        ```python
        import numpy as np
        X_train = np.random.rand(100, 10)
        y_train = np.random.randint(0, 2, 100)
        analyze_dataset(X_train, y_train)
        ```
    """
    
    # Basic statistics
    logger.info(f"Dataset Shape: {X.shape}")
    logger.info(f"Target Distribution: {np.bincount(y)}")
    
    # Missing values
    missing_values = pd.DataFrame(X).isnull().sum()
    logger.info(f"Missing Values:\n{missing_values}")
    
    # Data types
    logger.info(f"Data Types:\n{pd.DataFrame(X).dtypes}")
    
    # Basic statistics
    logger.info(f"Basic Statistics:\n{pd.DataFrame(X).describe()}")
    
    return True

# Usage
# analyze_dataset(X_train, y_train, feature_names)
```

### Feature Distribution Analysis

```python
def analyze_feature_distributions(X: np.ndarray, feature_names: Optional[List[str]] = None) -> None:
    """Analyze and visualize feature distributions.
    
    Computes statistical measures (mean, std, skewness, kurtosis) for each feature
    and creates interactive Plotly histograms for visualization.
    
    Args:
        X: Input feature array of shape (n_samples, n_features).
        feature_names: Optional list of feature names. If None, auto-generated names are used.
    
    Returns:
        None
        
    Example:
        ```python
        import numpy as np
        X_train = np.random.rand(100, 10)
        analyze_feature_distributions(X_train)
        ```
    """
    
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    
    # Plot distributions using Plotly
    fig = go.Figure()
    
    for i, feature in enumerate(feature_names[:4]):
        fig.add_trace(go.Histogram(
            x=df[feature],
            name=feature,
            nbinsx=30,
            opacity=0.7
        ))
    
    fig.update_layout(
        title='Feature Distributions',
        xaxis_title='Value',
        yaxis_title='Frequency',
        barmode='overlay',
        height=600,
        width=900
    )
    
    plot(fig, auto_open=False)
    
    # Statistical analysis
    for feature in feature_names:
        logger.info(f"{feature}:")
        logger.info(f"  Mean: {df[feature].mean():.4f}")
        logger.info(f"  Std: {df[feature].std():.4f}")
        logger.info(f"  Skewness: {df[feature].skew():.4f}")
        logger.info(f"  Kurtosis: {df[feature].kurtosis():.4f}")

# Usage
# analyze_feature_distributions(X_train, feature_names)
```

## 🔧 Feature Analysis

### Feature Importance Analysis

```python
from kerasfactory.layers import VariableSelection, TabularAttention

def analyze_feature_importance(
    model: keras.Model,
    X_test: np.ndarray,
    feature_names: Optional[List[str]] = None
) -> Optional[pd.DataFrame]:
    """Analyze feature importance using model weights.
    
    Extracts feature importance scores from a variable selection layer and
    creates an interactive bar chart visualization.
    
    Args:
        model: Compiled Keras model with a 'variable_selection' layer.
        X_test: Test feature array of shape (n_samples, n_features).
        feature_names: Optional list of feature names. If None, auto-generated names are used.
    
    Returns:
        pd.DataFrame: DataFrame with features and their importance scores, sorted in descending order.
                     Returns None if the variable_selection layer is not found.
        
    Example:
        ```python
        import numpy as np
        import keras
        X_test = np.random.rand(50, 10)
        model = keras.Sequential([...])  # Your model here
        importance_df = analyze_feature_importance(model, X_test)
        ```
    """
    
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
        
        logger.info(f"Feature Importance:\n{importance_df}")
        
        # Plot importance using Plotly
        fig = go.Figure(data=[
            go.Bar(y=importance_df['feature'], x=importance_df['importance'], orientation='h')
        ])
        
        fig.update_layout(
            title='Feature Importance',
            xaxis_title='Importance Score',
            yaxis_title='Feature',
            height=600,
            width=900
        )
        
        plot(fig, auto_open=False)
        
        return importance_df
        
    except Exception as e:
        logger.error(f"Could not analyze feature importance: {e}")
        return None

# Usage
# importance_df = analyze_feature_importance(model, X_test, feature_names)
```

### Attention Weight Analysis

```python
def analyze_attention_weights(
    model: keras.Model,
    X_test: np.ndarray,
    feature_names: Optional[List[str]] = None
) -> Optional[pd.DataFrame]:
    """Analyze attention weights for model interpretation.
    
    Extracts and analyzes attention weights from a tabular attention layer,
    computing mean and standard deviation for each feature.
    
    Args:
        model: Compiled Keras model with a 'tabular_attention' layer.
        X_test: Test feature array of shape (n_samples, n_features).
        feature_names: Optional list of feature names. If None, auto-generated names are used.
    
    Returns:
        pd.DataFrame: DataFrame with features, mean attention weights, and standard deviations.
                     Returns None if the tabular_attention layer is not found.
        
    Example:
        ```python
        import numpy as np
        import keras
        X_test = np.random.rand(50, 10)
        model = keras.Sequential([...])  # Your model here
        attention_df = analyze_attention_weights(model, X_test)
        ```
    """
    
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
        
        logger.info(f"Attention Weights Analysis:\n{attention_df}")
        
        # Plot attention weights using Plotly
        fig = go.Figure(data=[
            go.Bar(
                y=attention_df['feature'],
                x=attention_df['mean_attention'],
                error_x=dict(type='data', array=attention_df['std_attention']),
                orientation='h'
            )
        ])
        
        fig.update_layout(
            title='Mean Attention Weights',
            xaxis_title='Attention Weight',
            yaxis_title='Feature',
            height=600,
            width=1000
        )
        
        plot(fig, auto_open=False)
        
        return attention_df
        
    except Exception as e:
        logger.error(f"Could not analyze attention weights: {e}")
        return None

# Usage
# attention_df = analyze_attention_weights(model, X_test, feature_names)
```

## 🧠 Model Interpretation

### Layer Output Analysis

```python
def analyze_layer_outputs(model: keras.Model, X_test: np.ndarray, layer_names: List[str]) -> Dict[str, np.ndarray]:
    """Analyze outputs from different model layers.
    
    Extracts activations from specified layers and logs their output shapes.
    Useful for understanding information flow through the model.
    
    Args:
        model: Compiled Keras model.
        X_test: Test feature array of shape (n_samples, n_features).
        layer_names: List of layer names to analyze.
    
    Returns:
        Dict[str, np.ndarray]: Dictionary mapping layer names to their output arrays.
        
    Example:
        ```python
        import numpy as np
        import keras
        X_test = np.random.rand(50, 10)
        model = keras.Sequential([...])  # Your model here
        layer_names = ['dense_1', 'dense_2']
        layer_outputs = analyze_layer_outputs(model, X_test, layer_names)
        ```
    """
    
    layer_outputs: Dict[str, np.ndarray] = {}
    
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
            
            logger.info(f"{layer_name} output shape: {layer_output.shape}")
            
        except Exception as e:
            logger.error(f"Could not analyze layer {layer_name}: {e}")
    
    return layer_outputs

# Usage
# layer_names = ['variable_selection', 'tabular_attention', 'gated_feature_fusion']
# layer_outputs = analyze_layer_outputs(model, X_test, layer_names)
```

### Model Decision Analysis

```python
def analyze_model_decisions(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Analyze model decision-making process.
    
    Computes predictions, confidence scores, and identifies misclassified samples.
    Provides detailed analysis of model prediction behavior.
    
    Args:
        model: Compiled Keras classification model.
        X_test: Test feature array of shape (n_samples, n_features).
        y_test: One-hot encoded target array of shape (n_samples, n_classes).
        feature_names: Optional list of feature names.
    
    Returns:
        Dict[str, Any]: Dictionary containing:
            - 'predictions': Raw model predictions
            - 'predicted_classes': Predicted class indices
            - 'true_classes': True class indices
            - 'confidence': Maximum prediction probability for each sample
            - 'misclassified_indices': Indices of misclassified samples
        
    Example:
        ```python
        import numpy as np
        import keras
        X_test = np.random.rand(50, 10)
        y_test = np.zeros((50, 3))
        y_test[np.arange(50), np.random.randint(0, 3, 50)] = 1
        model = keras.Sequential([...])  # Your model here
        decision_analysis = analyze_model_decisions(model, X_test, y_test)
        ```
    """
    
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(X_test.shape[1])]
    
    # Get predictions
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    
    # Analyze prediction confidence
    prediction_confidence = np.max(predictions, axis=1)
    
    logger.info("Prediction Confidence Analysis:")
    logger.info(f"Mean confidence: {np.mean(prediction_confidence):.4f}")
    logger.info(f"Std confidence: {np.std(prediction_confidence):.4f}")
    logger.info(f"Min confidence: {np.min(prediction_confidence):.4f}")
    logger.info(f"Max confidence: {np.max(prediction_confidence):.4f}")
    
    # Analyze misclassifications
    misclassified = predicted_classes != true_classes
    misclassified_indices = np.where(misclassified)[0]
    
    logger.info(f"Misclassified samples: {len(misclassified_indices)}")
    logger.info(f"Misclassification rate: {len(misclassified_indices) / len(y_test):.4f}")
    
    # Analyze confidence of misclassified samples
    if len(misclassified_indices) > 0:
        misclassified_confidence = prediction_confidence[misclassified_indices]
        logger.info(f"Mean confidence of misclassified: {np.mean(misclassified_confidence):.4f}")
    
    return {
        'predictions': predictions,
        'predicted_classes': predicted_classes,
        'true_classes': true_classes,
        'confidence': prediction_confidence,
        'misclassified_indices': misclassified_indices
    }

# Usage
# decision_analysis = analyze_model_decisions(model, X_test, y_test, feature_names)
```

## 📈 Performance Analysis

### Training Performance Analysis

```python
def analyze_training_performance(history: keras.callbacks.History) -> Dict[str, float]:
    """Analyze training performance and convergence.
    
    Visualizes training and validation loss/accuracy curves and detects overfitting.
    Returns final performance metrics.
    
    Args:
        history: Keras training history object from model.fit().
    
    Returns:
        Dict[str, float]: Dictionary containing:
            - 'final_train_loss': Final training loss
            - 'final_val_loss': Final validation loss
            - 'final_train_acc': Final training accuracy
            - 'final_val_acc': Final validation accuracy
        
    Example:
        ```python
        import keras
        model = keras.Sequential([...])
        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        history = model.fit(X_train, y_train, validation_split=0.2, epochs=10)
        performance_analysis = analyze_training_performance(history)
        ```
    """
    
    # Create subplots using Plotly
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Model Loss', 'Model Accuracy')
    )
    
    # Loss curves
    fig.add_trace(
        go.Scatter(y=history.history['loss'], name='Training Loss', mode='lines'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(y=history.history['val_loss'], name='Validation Loss', mode='lines'),
        row=1, col=1
    )
    
    # Accuracy curves
    fig.add_trace(
        go.Scatter(y=history.history['accuracy'], name='Training Accuracy', mode='lines'),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(y=history.history['val_accuracy'], name='Validation Accuracy', mode='lines'),
        row=1, col=2
    )
    
    # Update layout
    fig.update_xaxes(title_text='Epoch', row=1, col=1)
    fig.update_yaxes(title_text='Loss', row=1, col=1)
    fig.update_xaxes(title_text='Epoch', row=1, col=2)
    fig.update_yaxes(title_text='Accuracy', row=1, col=2)
    
    fig.update_layout(height=500, width=1200, title_text='Training Performance')
    plot(fig, auto_open=False)
    
    # Analyze convergence
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    
    logger.info("Final Performance:")
    logger.info(f"Training Loss: {final_train_loss:.4f}")
    logger.info(f"Validation Loss: {final_val_loss:.4f}")
    logger.info(f"Training Accuracy: {final_train_acc:.4f}")
    logger.info(f"Validation Accuracy: {final_val_acc:.4f}")
    
    # Check for overfitting
    if final_val_loss > final_train_loss * 1.1:
        logger.warning("Possible overfitting detected!")
    
    return {
        'final_train_loss': final_train_loss,
        'final_val_loss': final_val_loss,
        'final_train_acc': final_train_acc,
        'final_val_acc': final_val_acc
    }

# Usage
# performance_analysis = analyze_training_performance(history)
```

### Model Comparison Analysis

```python
def compare_models(
    models: List[keras.Model],
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """Compare performance of multiple models.
    
    Evaluates multiple models on test data, computes various metrics (accuracy, loss,
    precision, recall, F1), and creates comparative visualizations.
    
    Args:
        models: List of compiled Keras models to compare.
        X_test: Test feature array of shape (n_samples, n_features).
        y_test: One-hot encoded target array of shape (n_samples, n_classes).
        model_names: Optional list of names for the models. If None, auto-generated names are used.
    
    Returns:
        pd.DataFrame: DataFrame with comparison metrics for each model.
        
    Example:
        ```python
        import numpy as np
        import keras
        X_test = np.random.rand(50, 10)
        y_test = np.zeros((50, 3))
        y_test[np.arange(50), np.random.randint(0, 3, 50)] = 1
        model1 = keras.Sequential([...])
        model2 = keras.Sequential([...])
        models = [model1, model2]
        comparison_df = compare_models(models, X_test, y_test)
        ```
    """
    
    if model_names is None:
        model_names = [f'model_{i}' for i in range(len(models))]
    
    results: List[Dict[str, Any]] = []
    
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
    
    logger.info(f"Model Comparison:\n{comparison_df}")
    
    # Plot comparison using Plotly subplots
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Accuracy Comparison', 'Loss Comparison', 'Precision Comparison', 'F1 Score Comparison')
    )
    
    # Accuracy
    fig.add_trace(
        go.Bar(x=comparison_df['model'], y=comparison_df['accuracy'], name='Accuracy'),
        row=1, col=1
    )
    
    # Loss
    fig.add_trace(
        go.Bar(x=comparison_df['model'], y=comparison_df['loss'], name='Loss'),
        row=1, col=2
    )
    
    # Precision
    fig.add_trace(
        go.Bar(x=comparison_df['model'], y=comparison_df['precision'], name='Precision'),
        row=2, col=1
    )
    
    # F1 Score
    fig.add_trace(
        go.Bar(x=comparison_df['model'], y=comparison_df['f1'], name='F1 Score'),
        row=2, col=2
    )
    
    fig.update_layout(height=800, width=1200, showlegend=False, title_text='Model Comparison')
    plot(fig, auto_open=False)
    
    return comparison_df

# Usage
# models = [model1, model2, model3]
# model_names = ['Attention Model', 'Residual Model', 'Ensemble Model']
# comparison_df = compare_models(models, X_test, y_test, model_names)
```

## 🔍 Advanced Analysis

### Feature Interaction Analysis

```python
def analyze_feature_interactions(X: np.ndarray, feature_names: Optional[List[str]] = None) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Analyze feature interactions and correlations.
    
    Computes correlation matrix and identifies highly correlated feature pairs.
    Creates interactive heatmap visualization.
    
    Args:
        X: Input feature array of shape (n_samples, n_features).
        feature_names: Optional list of feature names. If None, auto-generated names are used.
    
    Returns:
        Tuple[pd.DataFrame, List[Dict[str, Any]]]: 
            - Correlation matrix as DataFrame
            - List of highly correlated feature pairs with their correlation values
        
    Example:
        ```python
        import numpy as np
        X_train = np.random.rand(100, 10)
        correlation_matrix, high_corr_pairs = analyze_feature_interactions(X_train)
        ```
    """
    
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    
    # Calculate correlation matrix
    correlation_matrix = df.corr()
    
    # Plot correlation heatmap using Plotly
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(correlation_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title='Correlation')
    ))
    
    fig.update_layout(
        title='Feature Correlation Matrix',
        height=700,
        width=800
    )
    
    plot(fig, auto_open=False)
    
    # Find highly correlated features
    high_corr_pairs: List[Dict[str, Any]] = []
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
        logger.info("Highly Correlated Feature Pairs:")
        for pair in high_corr_pairs:
            logger.info(f"{pair['feature1']} - {pair['feature2']}: {pair['correlation']:.4f}")
    
    return correlation_matrix, high_corr_pairs

# Usage
# correlation_matrix, high_corr_pairs = analyze_feature_interactions(X_train, feature_names)
```

### Model Robustness Analysis

```python
def analyze_model_robustness(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    noise_levels: Optional[List[float]] = None
) -> pd.DataFrame:
    """Analyze model robustness to noise.
    
    Evaluates model performance on test data with added Gaussian noise at various levels.
    Useful for assessing model stability and generalization.
    
    Args:
        model: Compiled Keras model.
        X_test: Test feature array of shape (n_samples, n_features).
        y_test: Target array for evaluation.
        noise_levels: Optional list of noise standard deviations to test. Defaults to [0.01, 0.05, 0.1].
    
    Returns:
        pd.DataFrame: DataFrame with noise levels and corresponding accuracy/loss values.
        
    Example:
        ```python
        import numpy as np
        import keras
        X_test = np.random.rand(50, 10)
        y_test = np.zeros((50, 3))
        y_test[np.arange(50), np.random.randint(0, 3, 50)] = 1
        model = keras.Sequential([...])  # Your model here
        robustness_df = analyze_model_robustness(model, X_test, y_test, noise_levels=[0.01, 0.05, 0.1])
        ```
    """
    
    if noise_levels is None:
        noise_levels = [0.01, 0.05, 0.1]
    
    results: List[Dict[str, Any]] = []
    
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
    
    logger.info(f"Model Robustness Analysis:\n{robustness_df}")
    
    # Plot robustness using Plotly
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=robustness_df['noise_level'],
        y=robustness_df['accuracy'],
        mode='lines+markers',
        name='Accuracy',
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=robustness_df['noise_level'],
        y=robustness_df['loss'],
        mode='lines+markers',
        name='Loss',
        marker=dict(size=8, symbol='square')
    ))
    
    fig.update_layout(
        title='Model Robustness to Noise',
        xaxis_title='Noise Level',
        yaxis_title='Performance',
        height=600,
        width=900,
        hovermode='x unified'
    )
    
    plot(fig, auto_open=False)
    
    return robustness_df

# Usage
# robustness_df = analyze_model_robustness(model, X_test, y_test)
```

## 📚 Next Steps

1. **Rich Docstrings Showcase**: See comprehensive examples
2. **BaseFeedForwardModel Guide**: Learn about feed-forward architectures
3. **KDP Integration Guide**: Integrate with Keras Data Processor
4. **API Reference**: Deep dive into layer parameters

---

**Ready for more examples?** Check out the [Rich Docstrings Showcase](rich_docstrings_showcase.md) next!