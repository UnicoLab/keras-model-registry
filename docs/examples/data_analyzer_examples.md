# 🔍 Data Analyzer Examples

The KMR Data Analyzer is a powerful tool that automatically analyzes your tabular datasets and recommends the best KMR layers for your specific data characteristics. This page provides comprehensive examples showing how to use the Data Analyzer effectively.

## 🚀 Quick Start

### Basic Analysis

```python
from kmr.utils.data_analyzer import DataAnalyzer

# Initialize the analyzer
analyzer = DataAnalyzer()

# Analyze your CSV file
results = analyzer.analyze_file("data/my_dataset.csv")

# Get layer recommendations
recommendations = results.get_layer_recommendations()
print("Recommended layers:", recommendations)
```

## 📊 Advanced Usage Examples

### Custom Analysis Parameters

```python
from kmr.utils.data_analyzer import DataAnalyzer

# Initialize with custom parameters
analyzer = DataAnalyzer(
    sample_size=5000,           # Analyze first 5000 rows
    correlation_threshold=0.8,  # High correlation threshold
    categorical_threshold=0.05, # 5% unique values = categorical
    missing_threshold=0.3       # 30% missing values threshold
)

# Analyze with detailed output
results = analyzer.analyze_file(
    "data/large_dataset.csv",
    output_format="detailed",
    include_statistics=True,
    include_correlations=True
)

# Get comprehensive insights
insights = results.get_data_insights()
statistics = results.get_statistics()
correlations = results.get_correlations()

print("Data insights:", insights)
print("Statistics:", statistics)
print("Top correlations:", correlations)
```

### Batch Analysis

```python
from kmr.utils.data_analyzer import DataAnalyzer
import os
import pandas as pd

analyzer = DataAnalyzer()

# Analyze multiple CSV files
data_dir = "data/"
csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

all_results = {}
for file in csv_files:
    file_path = os.path.join(data_dir, file)
    print(f"Analyzing {file}...")
    
    results = analyzer.analyze_file(file_path)
    all_results[file] = {
        'recommendations': results.get_layer_recommendations(),
        'insights': results.get_data_insights(),
        'shape': results.get_data_shape()
    }

# Create a summary DataFrame
summary_data = []
for file, data in all_results.items():
    summary_data.append({
        'file': file,
        'rows': data['shape'][0],
        'columns': data['shape'][1],
        'recommendations': ', '.join(data['recommendations']),
        'insights': data['insights']['data_type']
    })

summary_df = pd.DataFrame(summary_data)
print(summary_df)
```

## 🖥️ Command Line Interface

### Basic CLI Usage

```bash
# Analyze a single CSV file
kmr-analyze data/tabular_data.csv

# Analyze with verbose output
kmr-analyze data/tabular_data.csv --verbose

# Save results to file
kmr-analyze data/tabular_data.csv --output results.json
```

### Advanced CLI Options

```bash
# Analyze with custom parameters
kmr-analyze data/large_dataset.csv \
    --sample-size 10000 \
    --correlation-threshold 0.7 \
    --categorical-threshold 0.1 \
    --output detailed_analysis.json \
    --format json

# Analyze multiple files
kmr-analyze data/*.csv --batch --output batch_results.json

# Get specific layer recommendations
kmr-analyze data/tabular_data.csv --layers attention,embedding,feature_engineering

# Include additional analysis
kmr-analyze data/tabular_data.csv \
    --include-statistics \
    --include-correlations \
    --include-missing-analysis
```

### CLI Integration with Scripts

```bash
#!/bin/bash
# analyze_all_datasets.sh

echo "Starting batch analysis of all datasets..."

# Create output directory
mkdir -p analysis_results

# Analyze all CSV files
for file in data/*.csv; do
    filename=$(basename "$file" .csv)
    echo "Analyzing $filename..."
    
    kmr-analyze "$file" \
        --output "analysis_results/${filename}_analysis.json" \
        --format json \
        --include-statistics \
        --include-correlations
done

echo "Analysis complete! Results saved in analysis_results/"
```

## 🔄 Complete Workflow Examples

### From Data Analysis to Model Building

```python
from kmr.utils.data_analyzer import DataAnalyzer
from kmr.layers import TabularAttention, AdvancedNumericalEmbedding, VariableSelection
from kmr.models import BaseFeedForwardModel
import keras
import pandas as pd

def build_smart_model(csv_file):
    """Build a model based on data analysis recommendations."""
    
    # Step 1: Analyze the data
    analyzer = DataAnalyzer()
    analysis = analyzer.analyze_file(csv_file)
    
    # Step 2: Get recommendations and insights
    recommendations = analysis.get_layer_recommendations()
    insights = analysis.get_data_insights()
    data_shape = analysis.get_data_shape()
    
    print(f"Dataset shape: {data_shape}")
    print(f"Data type: {insights['data_type']}")
    print(f"Recommended layers: {recommendations}")
    
    # Step 3: Build model based on recommendations
    num_features = data_shape[1]
    inputs = keras.Input(shape=(num_features,))
    
    x = inputs
    
    # Apply recommended layers
    if "AdvancedNumericalEmbedding" in recommendations:
        x = AdvancedNumericalEmbedding(
            embedding_dim=32,
            mlp_hidden_units=64,
            num_bins=20
        )(x)
    
    if "VariableSelection" in recommendations:
        x = VariableSelection(
            nr_features=num_features,
            units=32
        )(x)
    
    if "TabularAttention" in recommendations:
        x = TabularAttention(
            num_heads=8,
            d_model=64,
            dropout_rate=0.1
        )(x)
    
    # Add final layers
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)
    
    # Create model
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model, analysis

# Use the function
model, analysis = build_smart_model("data/my_dataset.csv")
print("Model built with recommended layers!")
model.summary()
```

### Automated Model Architecture Selection

```python
from kmr.utils.data_analyzer import DataAnalyzer
from kmr.layers import *
import keras

class SmartModelBuilder:
    """Automatically build models based on data analysis."""
    
    def __init__(self):
        self.analyzer = DataAnalyzer()
        self.layer_mappings = {
            "TabularAttention": lambda shape: TabularAttention(num_heads=4, d_model=32),
            "AdvancedNumericalEmbedding": lambda shape: AdvancedNumericalEmbedding(embedding_dim=16),
            "VariableSelection": lambda shape: VariableSelection(nr_features=shape[1], units=32),
            "GatedFeatureFusion": lambda shape: GatedFeatureFusion(units=32),
            "DistributionTransformLayer": lambda shape: DistributionTransformLayer(),
            "DateEncodingLayer": lambda shape: DateEncodingLayer(),
        }
    
    def build_model(self, csv_file, task_type="classification"):
        """Build a model based on data analysis."""
        
        # Analyze data
        analysis = self.analyzer.analyze_file(csv_file)
        recommendations = analysis.get_layer_recommendations()
        data_shape = analysis.get_data_shape()
        
        # Build model
        inputs = keras.Input(shape=(data_shape[1],))
        x = inputs
        
        # Apply recommended layers
        for layer_name in recommendations:
            if layer_name in self.layer_mappings:
                layer = self.layer_mappings[layer_name](data_shape)
                x = layer(x)
        
        # Add task-specific final layers
        if task_type == "classification":
            x = keras.layers.Dense(32, activation='relu')(x)
            outputs = keras.layers.Dense(1, activation='sigmoid')(x)
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        else:  # regression
            x = keras.layers.Dense(32, activation='relu')(x)
            outputs = keras.layers.Dense(1)(x)
            loss = 'mse'
            metrics = ['mae']
        
        # Create and compile model
        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer='adam',
            loss=loss,
            metrics=metrics
        )
        
        return model, analysis

# Use the smart builder
builder = SmartModelBuilder()
model, analysis = builder.build_model("data/my_dataset.csv", task_type="classification")

print("Automatically built model!")
print("Used layers:", analysis.get_layer_recommendations())
model.summary()
```

## 📈 Analysis Results Interpretation

### Understanding Recommendations

```python
from kmr.utils.data_analyzer import DataAnalyzer

analyzer = DataAnalyzer()
results = analyzer.analyze_file("data/my_dataset.csv")

# Get detailed analysis
recommendations = results.get_layer_recommendations()
insights = results.get_data_insights()
statistics = results.get_statistics()

print("=== ANALYSIS RESULTS ===")
print(f"Dataset: {insights['file_name']}")
print(f"Shape: {insights['shape']}")
print(f"Data type: {insights['data_type']}")
print(f"Missing values: {insights['missing_percentage']:.2f}%")

print("\n=== RECOMMENDED LAYERS ===")
for layer in recommendations:
    print(f"✓ {layer}")

print("\n=== DATA CHARACTERISTICS ===")
print(f"Numerical features: {statistics['numerical_features']}")
print(f"Categorical features: {statistics['categorical_features']}")
print(f"High correlation pairs: {statistics['high_correlations']}")

# Get specific recommendations by category
attention_layers = [l for l in recommendations if 'attention' in l.lower()]
embedding_layers = [l for l in recommendations if 'embedding' in l.lower()]
feature_layers = [l for l in recommendations if any(x in l.lower() for x in ['feature', 'selection', 'transform'])]

print(f"\nAttention layers: {attention_layers}")
print(f"Embedding layers: {embedding_layers}")
print(f"Feature engineering layers: {feature_layers}")
```

## 🎯 Best Practices

### 1. Data Preprocessing Before Analysis

```python
import pandas as pd
from kmr.utils.data_analyzer import DataAnalyzer

# Clean your data before analysis
def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    # Remove completely empty columns
    df = df.dropna(axis=1, how='all')
    
    # Handle obvious data type issues
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try to convert to numeric
            try:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except:
                pass
    
    return df

# Preprocess and analyze
df = preprocess_data("data/raw_dataset.csv")
df.to_csv("data/cleaned_dataset.csv", index=False)

analyzer = DataAnalyzer()
results = analyzer.analyze_file("data/cleaned_dataset.csv")
```

### 2. Iterative Analysis

```python
from kmr.utils.data_analyzer import DataAnalyzer

def iterative_analysis(csv_file, iterations=3):
    """Perform iterative analysis with different parameters."""
    
    analyzer = DataAnalyzer()
    all_results = []
    
    for i in range(iterations):
        # Vary analysis parameters
        sample_size = 1000 * (i + 1)
        correlation_threshold = 0.5 + (i * 0.1)
        
        results = analyzer.analyze_file(
            csv_file,
            sample_size=sample_size,
            correlation_threshold=correlation_threshold
        )
        
        all_results.append({
            'iteration': i + 1,
            'sample_size': sample_size,
            'correlation_threshold': correlation_threshold,
            'recommendations': results.get_layer_recommendations()
        })
    
    return all_results

# Perform iterative analysis
results = iterative_analysis("data/my_dataset.csv")
for result in results:
    print(f"Iteration {result['iteration']}: {result['recommendations']}")
```

### 3. Validation and Testing

```python
from kmr.utils.data_analyzer import DataAnalyzer
import numpy as np

def validate_recommendations(csv_file, test_size=0.2):
    """Validate recommendations by testing on held-out data."""
    
    analyzer = DataAnalyzer()
    
    # Analyze full dataset
    full_results = analyzer.analyze_file(csv_file)
    full_recommendations = full_results.get_layer_recommendations()
    
    # Analyze sample
    sample_results = analyzer.analyze_file(csv_file, sample_size=int(1000 * (1 - test_size)))
    sample_recommendations = sample_results.get_layer_recommendations()
    
    # Compare recommendations
    overlap = set(full_recommendations) & set(sample_recommendations)
    stability = len(overlap) / len(set(full_recommendations) | set(sample_recommendations))
    
    print(f"Recommendation stability: {stability:.2f}")
    print(f"Full recommendations: {full_recommendations}")
    print(f"Sample recommendations: {sample_recommendations}")
    print(f"Overlap: {overlap}")
    
    return stability > 0.7  # 70% stability threshold

# Validate recommendations
is_stable = validate_recommendations("data/my_dataset.csv")
print(f"Recommendations are stable: {is_stable}")
```

## 🔧 Troubleshooting

### Common Issues and Solutions

```python
# Issue 1: Large file analysis
def analyze_large_file(file_path, chunk_size=10000):
    """Analyze large files in chunks."""
    analyzer = DataAnalyzer()
    
    # Use sampling for large files
    results = analyzer.analyze_file(
        file_path,
        sample_size=chunk_size,
        random_sampling=True
    )
    
    return results

# Issue 2: Memory issues
def memory_efficient_analysis(file_path):
    """Memory-efficient analysis."""
    analyzer = DataAnalyzer(
        sample_size=5000,  # Limit sample size
        correlation_threshold=0.8,  # Higher threshold to reduce computation
    )
    
    results = analyzer.analyze_file(file_path)
    return results

# Issue 3: Mixed data types
def handle_mixed_data(file_path):
    """Handle files with mixed data types."""
    import pandas as pd
    
    # Read and clean data
    df = pd.read_csv(file_path)
    
    # Convert mixed types
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try numeric conversion
            numeric_series = pd.to_numeric(df[col], errors='coerce')
            if not numeric_series.isna().all():
                df[col] = numeric_series
    
    # Save cleaned data
    cleaned_path = file_path.replace('.csv', '_cleaned.csv')
    df.to_csv(cleaned_path, index=False)
    
    # Analyze cleaned data
    analyzer = DataAnalyzer()
    return analyzer.analyze_file(cleaned_path)
```

## 📚 Integration Examples

### Jupyter Notebook Integration

```python
# In a Jupyter notebook
from kmr.utils.data_analyzer import DataAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns

# Analyze data
analyzer = DataAnalyzer()
results = analyzer.analyze_file("data/my_dataset.csv")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot recommendations
recommendations = results.get_layer_recommendations()
axes[0, 0].bar(range(len(recommendations)), [1] * len(recommendations))
axes[0, 0].set_xticks(range(len(recommendations)))
axes[0, 0].set_xticklabels(recommendations, rotation=45)
axes[0, 0].set_title("Recommended Layers")

# Plot data insights
insights = results.get_data_insights()
axes[0, 1].pie([insights['numerical_features'], insights['categorical_features']], 
               labels=['Numerical', 'Categorical'], autopct='%1.1f%%')
axes[0, 1].set_title("Feature Types")

# Plot missing values
missing_data = insights['missing_percentage']
axes[1, 0].bar(['Missing %'], [missing_data])
axes[1, 0].set_title("Missing Values")

# Plot correlations
correlations = results.get_correlations()
if correlations:
    corr_matrix = correlations[:10]  # Top 10 correlations
    axes[1, 1].barh(range(len(corr_matrix)), [abs(c) for c in corr_matrix])
    axes[1, 1].set_yticks(range(len(corr_matrix)))
    axes[1, 1].set_title("Top Correlations")

plt.tight_layout()
plt.show()
```

This comprehensive guide shows you how to effectively use the KMR Data Analyzer for intelligent layer recommendations and automated model building!
