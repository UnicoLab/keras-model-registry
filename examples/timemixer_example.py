"""
TimeMixer Model Example.

This script demonstrates how to use the TimeMixer model for time series forecasting.
It includes examples of:
- Basic model creation and training
- Multivariate forecasting
- Different decomposition methods
- Channel independence modes
- Custom model architectures
"""

import numpy as np
import keras
from kmr.models import TimeMixer


def example_basic_usage():
    """Example 1: Basic TimeMixer model usage."""
    print("=" * 80)
    print("Example 1: Basic TimeMixer Model Usage")
    print("=" * 80)

    # Create model
    model = TimeMixer(
        seq_len=96,
        pred_len=12,
        n_features=7,
        d_model=32,
        e_layers=2,
        decomp_method="moving_avg",
    )

    # Compile
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    # Generate synthetic data
    x_train = np.random.randn(100, 96, 7).astype(np.float32)
    y_train = np.random.randn(100, 12, 7).astype(np.float32)

    # Train
    print("\nTraining model...")
    model.fit(x_train, y_train, epochs=2, batch_size=16, verbose=1)

    # Predict
    print("\nMaking predictions...")
    x_test = np.random.randn(10, 96, 7).astype(np.float32)
    predictions = model.predict(x_test)
    print(f"Predictions shape: {predictions.shape}")
    print("✓ Basic usage example completed\n")


def example_dft_decomposition():
    """Example 2: Using DFT decomposition method."""
    print("=" * 80)
    print("Example 2: DFT Decomposition Method")
    print("=" * 80)

    # Create model with DFT decomposition
    model = TimeMixer(
        seq_len=100,
        pred_len=24,
        n_features=4,
        d_model=64,
        e_layers=4,
        decomp_method="dft_decomp",
        top_k=5,
    )

    model.compile(optimizer="adam", loss="mse")

    # Generate data
    x_train = np.random.randn(50, 100, 4).astype(np.float32)
    y_train = np.random.randn(50, 24, 4).astype(np.float32)

    # Train
    print("\nTraining with DFT decomposition...")
    model.fit(x_train, y_train, epochs=1, batch_size=8, verbose=0)

    # Evaluate
    loss = model.evaluate(x_train, y_train, verbose=0)
    print(f"Loss: {loss:.6f}")
    print("✓ DFT decomposition example completed\n")


def example_channel_independence():
    """Example 3: Channel independent processing."""
    print("=" * 80)
    print("Example 3: Channel Independence Mode")
    print("=" * 80)

    # Create channel-independent model
    model = TimeMixer(
        seq_len=96,
        pred_len=12,
        n_features=5,
        d_model=32,
        e_layers=2,
        channel_independence=1,  # Independent processing for each channel
    )

    model.compile(optimizer="adam", loss="mse")

    # Generate data
    x_train = np.random.randn(30, 96, 5).astype(np.float32)
    y_train = np.random.randn(30, 12, 5).astype(np.float32)

    print("\nTraining channel-independent model...")
    model.fit(x_train, y_train, epochs=1, batch_size=8, verbose=0)

    # Predict
    x_test = np.random.randn(5, 96, 5).astype(np.float32)
    predictions = model.predict(x_test, verbose=0)
    print(f"Predictions shape: {predictions.shape}")
    print("✓ Channel independence example completed\n")


def example_custom_architecture():
    """Example 4: Custom model architecture."""
    print("=" * 80)
    print("Example 4: Custom Model Architecture")
    print("=" * 80)

    # Create custom model
    model = TimeMixer(
        seq_len=120,
        pred_len=24,
        n_features=10,
        d_model=128,
        d_ff=256,
        e_layers=6,
        dropout=0.2,
        down_sampling_layers=2,
        down_sampling_window=3,
        use_norm=True,
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae", "mape"],
    )

    # Generate data
    x_train = np.random.randn(100, 120, 10).astype(np.float32)
    y_train = np.random.randn(100, 24, 10).astype(np.float32)

    print("\nTraining custom architecture model...")
    history = model.fit(
        x_train,
        y_train,
        epochs=2,
        batch_size=16,
        validation_split=0.1,
        verbose=1,
    )

    print(f"\nFinal training loss: {history.history['loss'][-1]:.6f}")
    print("✓ Custom architecture example completed\n")


def example_with_temporal_features():
    """Example 5: Using optional temporal features."""
    print("=" * 80)
    print("Example 5: Model with Temporal Features")
    print("=" * 80)

    # Create model
    model = TimeMixer(seq_len=96, pred_len=12, n_features=3, d_model=32, e_layers=2)

    model.compile(optimizer="adam", loss="mse")

    # Generate data
    x_train = np.random.randn(20, 96, 3).astype(np.float32)
    x_mark = np.random.randint(0, 13, (20, 96, 5)).astype(np.int32)
    y_train = np.random.randn(20, 12, 3).astype(np.float32)

    print("\nTraining with temporal features...")
    # Train with temporal features
    for epoch in range(1):
        # Create model input with temporal marks
        model.fit([x_train, x_mark], y_train, epochs=1, batch_size=8, verbose=0)

    # Make predictions with temporal features
    x_test = np.random.randn(5, 96, 3).astype(np.float32)
    x_mark_test = np.random.randint(0, 13, (5, 96, 5)).astype(np.int32)
    predictions = model([x_test, x_mark_test])
    print(f"Predictions with temporal features shape: {predictions.shape}")
    print("✓ Temporal features example completed\n")


def example_model_evaluation():
    """Example 6: Model evaluation and metrics."""
    print("=" * 80)
    print("Example 6: Model Evaluation")
    print("=" * 80)

    # Create model
    model = TimeMixer(seq_len=96, pred_len=12, n_features=4, d_model=32, e_layers=2)

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    # Generate data
    np.random.seed(42)
    x_train = np.random.randn(100, 96, 4).astype(np.float32)
    y_train = np.random.randn(100, 12, 4).astype(np.float32)

    x_val = np.random.randn(20, 96, 4).astype(np.float32)
    y_val = np.random.randn(20, 12, 4).astype(np.float32)

    # Train
    print("\nTraining model...")
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=2,
        batch_size=16,
        verbose=1,
    )

    # Evaluate
    print("\nEvaluating on validation set...")
    val_loss, val_mae = model.evaluate(x_val, y_val, verbose=0)
    print(f"Validation Loss: {val_loss:.6f}")
    print(f"Validation MAE: {val_mae:.6f}")

    # Predictions
    predictions = model.predict(x_val[:5], verbose=0)
    print(f"\nPredictions shape: {predictions.shape}")
    print("✓ Evaluation example completed\n")


def example_serialization():
    """Example 7: Model serialization and deserialization."""
    print("=" * 80)
    print("Example 7: Model Serialization")
    print("=" * 80)

    # Create and train model
    model = TimeMixer(seq_len=96, pred_len=12, n_features=3, d_model=32, e_layers=2)

    model.compile(optimizer="adam", loss="mse")

    # Get config
    config = model.get_config()
    print("\nModel configuration:")
    print(f"  - seq_len: {config['seq_len']}")
    print(f"  - pred_len: {config['pred_len']}")
    print(f"  - n_features: {config['n_features']}")
    print(f"  - d_model: {config['d_model']}")
    print(f"  - e_layers: {config['e_layers']}")
    print(f"  - decomp_method: {config['decomp_method']}")

    # Recreate from config
    new_model = TimeMixer.from_config(config)
    print("\n✓ Model successfully recreated from config\n")


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  TimeMixer Model Examples".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")

    example_basic_usage()
    example_dft_decomposition()
    example_channel_independence()
    example_custom_architecture()
    example_with_temporal_features()
    example_model_evaluation()
    example_serialization()

    print("=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
