#!/usr/bin/env python
"""Example of using the KMR Data Analyzer to recommend layers for building models.

This script demonstrates how to use the DataAnalyzer class to analyze CSV data
and get recommendations for layer usage in Keras models.
"""

import os
import json
import pandas as pd
import numpy as np
from kmr.utils import DataAnalyzer

# Path to save the example CSV
EXAMPLE_DATA_PATH = "example_data.csv"


def create_example_dataset():
    """Create a sample dataset with various feature types for demonstration."""
    # Set a seed for reproducibility
    np.random.seed(42)

    # Create a dataframe with 1000 rows
    n_samples = 1000

    # Generate sample data
    data = {
        # Numerical features
        "numeric_1": np.random.normal(0, 1, n_samples),
        "numeric_2": np.random.normal(10, 5, n_samples),
        "numeric_3": np.random.exponential(1, n_samples),
        "numeric_4": np.random.uniform(0, 100, n_samples),
        # Correlated numeric features
        "correlated_1": np.random.normal(0, 1, n_samples),
    }

    # Add a correlated feature
    data["correlated_2"] = data["correlated_1"] * 0.8 + np.random.normal(
        0,
        0.2,
        n_samples,
    )

    # Categorical features with different cardinality
    data["category_low"] = np.random.choice(["A", "B", "C"], n_samples)
    data["category_medium"] = np.random.choice(
        [f"Val_{i}" for i in range(20)],
        n_samples,
    )
    data["category_high"] = np.random.choice([f"ID_{i}" for i in range(500)], n_samples)

    # Date features
    base_date = pd.Timestamp("2020-01-01")
    data["date"] = [
        base_date + pd.Timedelta(days=i)
        for i in np.random.randint(0, 365 * 2, n_samples)
    ]

    # Text feature
    text_options = [
        "This is a short text description for testing purposes.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models can benefit from preprocessing layers.",
        "Keras Model Registry provides reusable layers for deep learning models.",
        "Data analysis helps identify the appropriate layers to use.",
    ]
    data["text"] = np.random.choice(text_options, n_samples)

    # Features with missing values
    data["missing_values"] = np.random.normal(0, 1, n_samples)
    missing_indices = np.random.choice(
        n_samples,
        size=int(n_samples * 0.3),
        replace=False,
    )
    data["missing_values"][missing_indices] = np.nan

    # Create the dataframe
    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv(EXAMPLE_DATA_PATH, index=False)

    return EXAMPLE_DATA_PATH


def main() -> None:
    """Run the example of data analysis and layer recommendations."""
    data_path = create_example_dataset()

    # Create analyzer
    analyzer = DataAnalyzer()

    # Analyze the data
    result = analyzer.analyze_and_recommend(data_path)

    # Print results
    stats = result["analysis"]["stats"]

    for characteristic, values in stats["characteristics"].items():
        if characteristic == "feature_interaction":
            pass
        elif values and values != ["all"]:
            pass

    for characteristic, layers in result["recommendations"].items():
        for _layer, _description, _use_case in layers:
            pass

    # Save detailed results to JSON
    with open("analysis_results.json", "w") as f:
        json.dump(result, f, indent=2)

    # Demonstrate how to add custom layer recommendations
    analyzer.register_recommendation(
        "high_cardinality_categorical",
        "MyCustomEmbedding",
        "Custom embedding layer for high cardinality features",
        "Specialized embedding for ID-like features",
    )

    # Get updated recommendations
    updated_recommendations = analyzer.recommend_layers(stats)
    for _layer, _description, _use_case in updated_recommendations[
        "high_cardinality_categorical"
    ]:
        pass

    # Clean up
    os.remove(data_path)


if __name__ == "__main__":
    main()
