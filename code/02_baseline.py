"""
Phase 2 — Baseline ANN
Reproduce the original model from mentor exactly, then record metrics.
Architecture: Input(2) → Dense(4, relu) → Dense(1)
Input features: Electrode1_resistance, Electrode2_resistance (NO temperature)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from pathlib import Path

from utils.data_loader import load_dataset, DATASETS, INPUT_FEATURES, OUTPUT_FEATURE
from utils.metrics import evaluate, print_metrics

SEED = 0
RESULTS_DIR = Path("results/phase2_baseline")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

tf.random.set_seed(SEED)
np.random.seed(SEED)


def build_model():
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(2,)),
        tf.keras.layers.Dense(4, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="mse",
        metrics=["mae"],
    )
    return model


all_results = {}

print("=" * 60)
print("BASELINE — Train & Evaluate per dataset")
print("=" * 60)

for i in DATASETS:
    print(f"\n--- Dataset {i} ---")
    df = load_dataset(i)

    scaler = MinMaxScaler(feature_range=(0, 1))
    df[INPUT_FEATURES] = scaler.fit_transform(df[INPUT_FEATURES])

    X = df[INPUT_FEATURES].values
    y = df[OUTPUT_FEATURE].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=SEED, shuffle=False
    )

    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    X_test  = tf.convert_to_tensor(X_test,  dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    y_test  = tf.convert_to_tensor(y_test,  dtype=tf.float32)

    model = build_model()
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=0,
    )

    y_pred = model.predict(X_test, verbose=0).flatten()
    metrics = evaluate(y_test.numpy(), y_pred)
    print_metrics(metrics, label=f"dataset{i}")
    all_results[f"dataset{i}"] = metrics

    # Loss curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history.history["loss"],     label="Train Loss")
    ax.plot(history.history["val_loss"], label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"Baseline — Dataset {i} Loss Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"loss_curve_dataset{i}.png", dpi=150)
    plt.close(fig)

    # Prediction vs ground truth
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(range(len(y_test)), y_test.numpy(), s=2, alpha=0.5, label="Ground Truth")
    ax.scatter(range(len(y_pred)), y_pred,         s=2, alpha=0.5, label="Predicted")
    ax.set_title(f"Baseline — Dataset {i} | R²={metrics['r2']:.4f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"prediction_dataset{i}.png", dpi=150)
    plt.close(fig)

# Save all metrics
with open(RESULTS_DIR / "baseline_metrics.json", "w") as f:
    json.dump(all_results, f, indent=2)

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
for name, m in all_results.items():
    print_metrics(m, label=name)

print(f"\n[Phase 2 Baseline complete] Results saved to {RESULTS_DIR}")
