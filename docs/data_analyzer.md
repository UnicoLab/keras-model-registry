# 🔍 KerasFactory Data Analyzer

The KerasFactory Data Analyzer is an intelligent utility that analyzes your tabular data and automatically recommends the best KerasFactory layers for your specific dataset.

!!! tip "Smart Recommendations"
    Just provide your CSV file, and the analyzer will suggest the most appropriate layers based on your data characteristics!

## ✨ Features

- 📊 **Automatic Analysis**: Analyzes single CSV files or entire directories
- 🎯 **Feature Detection**: Identifies numerical, categorical, date, and text features
- 🔍 **Data Insights**: Detects high cardinality, missing values, correlations, and patterns
- 🧩 **Layer Recommendations**: Suggests the best KerasFactory layers for your data
- 🔧 **Extensible**: Add custom recommendation rules
- 💻 **CLI & API**: Command-line interface and Python API
- 📈 **Performance Tips**: Guidance on layer configuration and optimization

## 🚀 Installation

The Data Analyzer is included with the KerasFactory package.

```bash
# Install from PyPI (recommended)
pip install kerasfactory

# Or install from source using Poetry
git clone https://github.com/UnicoLab/KerasFactory
cd KerasFactory
poetry install
```

## 💻 Usage

### 🖥️ Command-line Interface

The Data Analyzer can be used from the command line:

```bash
# Analyze a single CSV file
python -m kerasfactory.utils.data_analyzer_cli path/to/data.csv

# Analyze a directory of CSV files
python -m kerasfactory.utils.data_analyzer_cli path/to/data_dir/

# Save results to a JSON file
python -m kerasfactory.utils.data_analyzer_cli path/to/data.csv --output results.json

# Get only layer recommendations without detailed statistics
python -m kerasfactory.utils.data_analyzer_cli path/to/data.csv --recommendations-only
```

### 🐍 Python API

You can also use the Data Analyzer in your Python code:

```python
from kerasfactory.utils import DataAnalyzer, analyze_data

# Quick usage
results = analyze_data("path/to/data.csv")
recommendations = results["recommendations"]

# Or using the class directly
analyzer = DataAnalyzer()
result = analyzer.analyze_and_recommend("path/to/data.csv")

# Add custom layer recommendations
analyzer.register_recommendation(
    characteristic="continuous_features",
    layer_name="MyCustomLayer",
    description="Custom layer for continuous features",
    use_case="Special continuous feature processing"
)

# Analyze multiple files in a directory
result = analyzer.analyze_and_recommend("path/to/directory", pattern="*.csv")
```

## Data Characteristics

The analyzer identifies the following data characteristics:

- `continuous_features`: Numerical features
- `categorical_features`: Categorical features
- `date_features`: Date and time features
- `text_features`: Text features
- `high_cardinality_categorical`: Categorical features with high cardinality
- `high_missing_value_features`: Features with many missing values
- `feature_interaction`: Highly correlated feature pairs
- `time_series`: Date features that may indicate time series data
- `general_tabular`: General tabular data characteristics

## Layer Recommendations

For each data characteristic, the analyzer recommends appropriate KerasFactory layers along with descriptions and use cases.

### Example

For continuous features, the following layers might be recommended:

- `AdvancedNumericalEmbedding`: Embeds continuous features using both MLP and discretization approaches
- `DifferentialPreprocessingLayer`: Applies various normalizations and transformations to numerical features

## Extending Layer Recommendations

You can extend the layer recommendations by registering new layers:

```python
from kerasfactory.utils import DataAnalyzer

analyzer = DataAnalyzer()
analyzer.register_recommendation(
    characteristic="continuous_features",
    layer_name="MyCustomLayer",
    description="Custom layer for continuous features",
    use_case="Special continuous feature processing"
)
```

## Example Script

Check out the example script at `examples/data_analyzer_example.py` for a complete demonstration.

## Output Format

The analyzer returns a dictionary with the following structure:

```python
{
  "analysis": {
    "file": "filename.csv",  # For single file analysis
    "stats": {
      "row_count": 1000,
      "column_count": 10,
      "column_types": { ... },
      "characteristics": {
        "continuous_features": ["feature1", "feature2", ...],
        "categorical_features": ["feature3", "feature4", ...],
        ...
      },
      "missing_values": { ... },
      "cardinality": { ... },
      "numeric_stats": { ... }
    }
  },
  "recommendations": {
    "continuous_features": [
      ["LayerName1", "Description1", "UseCase1"],
      ["LayerName2", "Description2", "UseCase2"],
      ...
    ],
    "categorical_features": [ ... ],
    ...
  }
}
```

## Caveats

- The analyzer relies on heuristics to identify feature types, which may not always be accurate.
- Recommendations are based on general patterns and may need adjustment for specific use cases.
- Performance may degrade with very large CSV files due to memory constraints. 