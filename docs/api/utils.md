# 🔧 Utils API Reference

Welcome to the KMR Utilities documentation! This page provides documentation for KMR utility functions and tools, including the powerful **Data Analyzer** that can recommend appropriate layers for your tabular data.

!!! tip "What You'll Find Here"
    Each utility includes detailed documentation with:
    - ✨ **Complete parameter descriptions** with types and defaults
    - 🎯 **Usage examples** showing real-world applications
    - ⚡ **Best practices** and performance considerations
    - 🎨 **When to use** guidance for each utility
    - 🔧 **Implementation notes** for developers

!!! success "Smart Data Analysis"
    The Data Analyzer can automatically analyze your CSV files and recommend the best KMR layers for your specific dataset.

!!! example "CLI Integration"
    Use the command-line interface for quick data analysis and layer recommendations.

## 🔍 Data Analyzer

### 🧠 DataAnalyzer
Intelligent data analyzer that examines CSV files and recommends appropriate KMR layers based on data characteristics.

::: kmr.utils.data_analyzer.DataAnalyzer

#### 📋 Usage Examples

!!! example "Basic Data Analysis"
    ```python
    from kmr.utils.data_analyzer import DataAnalyzer
    
    # Initialize the analyzer
    analyzer = DataAnalyzer()
    
    # Analyze a CSV file
    results = analyzer.analyze_file("data/tabular_data.csv")
    
    # Get layer recommendations
    recommendations = results.get_layer_recommendations()
    print("Recommended layers:", recommendations)
    
    # Get data insights
    insights = results.get_data_insights()
    print("Data insights:", insights)
    ```

!!! example "Advanced Analysis with Custom Parameters"
    ```python
    from kmr.utils.data_analyzer import DataAnalyzer
    
    # Initialize with custom parameters
    analyzer = DataAnalyzer(
        sample_size=1000,  # Analyze first 1000 rows
        correlation_threshold=0.7,  # High correlation threshold
        categorical_threshold=0.1   # 10% unique values = categorical
    )
    
    # Analyze with detailed output
    results = analyzer.analyze_file(
        "data/large_dataset.csv",
        output_format="detailed",
        include_statistics=True
    )
    
    # Get specific recommendations
    attention_layers = results.get_recommendations_by_type("attention")
    feature_engineering = results.get_recommendations_by_type("feature_engineering")
    
    print("Attention layers:", attention_layers)
    print("Feature engineering:", feature_engineering)
    ```

!!! example "Batch Analysis of Multiple Files"
    ```python
    from kmr.utils.data_analyzer import DataAnalyzer
    import os
    
    analyzer = DataAnalyzer()
    
    # Analyze multiple CSV files
    data_dir = "data/"
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    all_results = {}
    for file in csv_files:
        file_path = os.path.join(data_dir, file)
        results = analyzer.analyze_file(file_path)
        all_results[file] = results.get_layer_recommendations()
    
    # Compare recommendations across datasets
    for file, recommendations in all_results.items():
        print(f"{file}: {recommendations}")
    ```

### 💻 DataAnalyzerCLI
Command-line interface for the data analyzer, allowing easy analysis of datasets from the terminal.

::: kmr.utils.data_analyzer_cli

#### 🖥️ CLI Usage Examples

!!! example "Basic CLI Analysis"
    ```bash
    # Analyze a single CSV file
    kmr-analyze data/tabular_data.csv
    
    # Analyze with verbose output
    kmr-analyze data/tabular_data.csv --verbose
    
    # Save results to file
    kmr-analyze data/tabular_data.csv --output results.json
    ```

!!! example "Advanced CLI Options"
    ```bash
    # Analyze with custom parameters
    kmr-analyze data/large_dataset.csv \
        --sample-size 5000 \
        --correlation-threshold 0.8 \
        --output detailed_analysis.json \
        --format json
    
    # Analyze multiple files
    kmr-analyze data/*.csv --batch --output batch_results.json
    
    # Get specific layer recommendations
    kmr-analyze data/tabular_data.csv --layers attention,embedding
    ```

!!! example "Integration with Jupyter Notebooks"
    ```python
    # In a Jupyter notebook, you can use the CLI output
    import json
    import subprocess
    
    # Run CLI analysis
    result = subprocess.run([
        'kmr-analyze', 'data/tabular_data.csv', 
        '--output', 'analysis.json', '--format', 'json'
    ], capture_output=True, text=True)
    
    # Load results
    with open('analysis.json', 'r') as f:
        analysis = json.load(f)
    
    # Use results in your notebook
    print("Recommended layers:", analysis['recommendations'])
    print("Data statistics:", analysis['statistics'])
    ```

#### 🔄 Complete Workflow Example

!!! example "End-to-End Data Analysis to Model Building"
    ```python
    from kmr.utils.data_analyzer import DataAnalyzer
    from kmr.layers import TabularAttention, AdvancedNumericalEmbedding
    from kmr.models import BaseFeedForwardModel
    import keras
    
    # Step 1: Analyze your data
    analyzer = DataAnalyzer()
    analysis = analyzer.analyze_file("data/my_dataset.csv")
    
    # Step 2: Get recommendations
    recommendations = analysis.get_layer_recommendations()
    print("Recommended layers:", recommendations)
    
    # Step 3: Build model based on recommendations
    if "TabularAttention" in recommendations:
        # Use tabular attention for feature relationships
        attention_layer = TabularAttention(
            num_heads=8,
            d_model=64,
            dropout_rate=0.1
        )
    
    if "AdvancedNumericalEmbedding" in recommendations:
        # Use advanced embedding for numerical features
        embedding_layer = AdvancedNumericalEmbedding(
            embedding_dim=32,
            mlp_hidden_units=64,
            num_bins=20
        )
    
    # Step 4: Create your model architecture
    inputs = keras.Input(shape=(100, 20))  # Based on your data shape
    
    # Apply recommended layers
    if 'embedding_layer' in locals():
        x = embedding_layer(inputs)
    else:
        x = inputs
    
    if 'attention_layer' in locals():
        x = attention_layer(x)
    
    # Add final layers
    x = keras.layers.Dense(64, activation='relu')(x)
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)
    
    # Create and compile model
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model built with recommended KMR layers!")
    model.summary()
    ```

!!! example "Automated Model Architecture Selection"
    ```python
    from kmr.utils.data_analyzer import DataAnalyzer
    from kmr.layers import *
    from kmr.models import BaseFeedForwardModel
    import keras
    
    def build_recommended_model(csv_file):
        """Automatically build a model based on data analysis."""
        
        # Analyze data
        analyzer = DataAnalyzer()
        analysis = analyzer.analyze_file(csv_file)
        recommendations = analysis.get_layer_recommendations()
        
        # Get data shape from analysis
        data_shape = analysis.get_data_shape()
        num_features = data_shape[1]
        
        # Build model based on recommendations
        inputs = keras.Input(shape=(num_features,))
        
        # Apply recommended layers
        x = inputs
        for layer_name in recommendations:
            if layer_name == "TabularAttention":
                x = TabularAttention(num_heads=4, d_model=32)(x)
            elif layer_name == "AdvancedNumericalEmbedding":
                x = AdvancedNumericalEmbedding(embedding_dim=16)(x)
            elif layer_name == "VariableSelection":
                x = VariableSelection(nr_features=num_features, units=32)(x)
            # Add more layer mappings as needed
        
        # Add final layers
        x = keras.layers.Dense(32, activation='relu')(x)
        outputs = keras.layers.Dense(1, activation='sigmoid')(x)
        
        model = keras.Model(inputs, outputs)
        return model, analysis
    
    # Use the function
    model, analysis = build_recommended_model("data/my_dataset.csv")
    print("Automatically built model with layers:", analysis.get_layer_recommendations())
    ```

## 🎨 Decorators

### ✨ Decorators
Utility decorators for common functionality in KMR components and enhanced development experience.

::: kmr.utils.decorators

#### 🔧 Usage Examples

!!! example "Layer Validation Decorator"
    ```python
    from kmr.utils.decorators import validate_inputs
    
    @validate_inputs
    def custom_layer_call(self, inputs, training=None):
        """Custom layer with automatic input validation."""
        # Your layer logic here
        return processed_outputs
    ```

!!! example "Performance Monitoring Decorator"
    ```python
    from kmr.utils.decorators import monitor_performance
    
    @monitor_performance
    def expensive_computation(self, data):
        """Function with automatic performance monitoring."""
        # Your computation here
        return result
    ```

!!! example "Serialization Helper Decorator"
    ```python
    from kmr.utils.decorators import serializable
    
    @serializable
    class CustomLayer:
        """Layer with automatic serialization support."""
        def __init__(self, param1, param2):
            self.param1 = param1
            self.param2 = param2
    ```
