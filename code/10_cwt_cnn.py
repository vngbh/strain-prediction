"""
Phase 10 — CWT + CNN-2D
Mentor suggestion: convert 1D sensor signal to 2D scalogram via CWT,
then apply 2D CNN (treating it like image classification).

Why this helps vs CNN-1D (Phase 6):
  - CNN-1D sees amplitude over time only
  - CWT scalogram captures BOTH time AND frequency → richer representation
  - 2D CNN can detect patterns at specific frequencies and time locations

Pipeline:
  window of 64 steps → CWT (scales × time) → 2D image → CNN-2D → Dense → Strain

Comparison: CNN-1D P6 vs CWT+CNN-2D P10

Input:  data/dataset{1-4}.csv — E1, E2; window=64 → scalogram (32 scales, 64 time, 2 channels)
Output: results/phase10_cwt_cnn/
          example_scalogram.png — raw E1 signal vs its CWT scalogram (visual sanity check)
          cwt_vs_cnn1d.png      — R² comparison: CNN-1D P6 vs CWT+CNN P10
          cwt_results.json
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import pywt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from pathlib import Path

from utils.data_loader import load_dataset, DATASETS, INPUT_FEATURES, OUTPUT_FEATURE, TEMP_LABEL

SEEDS       = [0, 42, 123]
WIN         = 64          # longer window — CWT needs more context than LSTM
SCALES      = np.arange(1, 33)   # 32 scales → scalogram shape: (32, 64)
WAVELET     = "morl"      # Morlet — standard for sensor/vibration signals
RESULTS_DIR = Path("results/phase10_cwt_cnn")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# CNN-1D P6 baseline for comparison
P6_RESULTS = {1: 0.900, 2: 0.782, 3: 0.838, 4: 0.647}

CALLBACKS = [
    tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True, verbose=0),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10, verbose=0),
]


def compute_cwt(signal_1d):
    """CWT on a 1D signal → 2D scalogram (n_scales, n_time)."""
    coeffs, _ = pywt.cwt(signal_1d, SCALES, WAVELET)
    return np.abs(coeffs)   # magnitude only


def make_cwt_sequences(X_sc, y, window):
    """
    For each time step t, take window of X, compute CWT per channel,
    stack channels → shape (n_scales, window, n_channels).
    """
    n_scales  = len(SCALES)
    n_channels = X_sc.shape[1]
    Xs, ys = [], []
    for i in range(window, len(X_sc)):
        segment = X_sc[i - window:i]          # (window, n_channels)
        # CWT per channel → stack → (n_scales, window, n_channels)
        cwt_channels = []
        for c in range(n_channels):
            cwt_channels.append(compute_cwt(segment[:, c]))   # (n_scales, window)
        scalogram = np.stack(cwt_channels, axis=-1)            # (n_scales, window, n_ch)
        Xs.append(scalogram)
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


def build_cnn2d():
    """2D CNN on scalogram: (n_scales, window, n_channels) → scalar."""
    m = tf.keras.Sequential([
        tf.keras.Input(shape=(len(SCALES), WIN, len(INPUT_FEATURES))),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1),
    ], name="cwt_cnn2d")
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return m


print("=" * 65)
print("PHASE 10 — CWT + CNN-2D")
print(f"Wavelet: {WAVELET} | Scales: {len(SCALES)} | Window: {WIN}")
print("=" * 65)

# Show one scalogram example
print("\n[Generating example scalogram for visualization...]")
df_ex = load_dataset(1)
X_ex  = df_ex[INPUT_FEATURES].values[:WIN]
sc_ex = MinMaxScaler().fit_transform(X_ex)
cwt_ex = compute_cwt(sc_ex[:, 0])

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(sc_ex[:, 0], lw=1.2)
axes[0].set_title("Raw E1 signal (normalized, 64 steps)", fontsize=10)
axes[0].set_xlabel("Time step"); axes[0].set_ylabel("Amplitude")
im = axes[1].imshow(cwt_ex, aspect="auto", cmap="jet",
                    extent=[0, WIN, SCALES[-1], SCALES[0]])
axes[1].set_title("CWT Scalogram (Morlet) — E1 channel", fontsize=10)
axes[1].set_xlabel("Time step"); axes[1].set_ylabel("Scale")
plt.colorbar(im, ax=axes[1], label="|CWT coefficients|")
fig.suptitle("Phase 10 — Raw signal → CWT scalogram", fontsize=11)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "example_scalogram.png", dpi=150)
plt.close(fig)
print("  Saved: example_scalogram.png")

# ── Main training loop ───────────────────────────────────────────────────────

print(f"\n{'Dataset':<14} {'CWT+CNN R²':>11} {'±std':>7} {'CNN-1D P6':>11} {'delta':>7}")
print("-" * 55)

all_results = {}

for ds_id in DATASETS:
    df   = load_dataset(ds_id)
    X    = df[INPUT_FEATURES].values
    y    = df[OUTPUT_FEATURE].values
    split = int(len(X) * 0.7)

    X_tr_raw, X_te_raw = X[:split], X[split:]
    y_tr, y_te          = y[:split], y[split:]

    scaler   = MinMaxScaler().fit(X_tr_raw)
    X_tr_sc  = scaler.transform(X_tr_raw)
    X_te_sc  = scaler.transform(X_te_raw)

    # CWT is expensive — compute once per dataset and reuse across all 3 seeds
    X_tr_cwt, y_tr_cwt = make_cwt_sequences(X_tr_sc, y_tr, WIN)
    X_te_cwt, y_te_cwt = make_cwt_sequences(X_te_sc, y_te, WIN)

    r2s = []
    for seed in SEEDS:
        tf.random.set_seed(seed); np.random.seed(seed)
        model = build_cnn2d()
        model.fit(X_tr_cwt, y_tr_cwt, epochs=300, batch_size=64,
                  validation_split=0.1, callbacks=CALLBACKS, verbose=0)
        pred = model.predict(X_te_cwt, verbose=0).flatten()
        r2s.append(float(r2_score(y_te_cwt, pred)))

    m, s  = np.mean(r2s), np.std(r2s)
    delta = m - P6_RESULTS[ds_id]
    sign  = "+" if delta >= 0 else ""
    print(f"ds{ds_id} ({TEMP_LABEL[ds_id]})   {m:>11.4f} {s:>7.4f} {P6_RESULTS[ds_id]:>11.3f} {sign}{delta:>6.3f}")

    all_results[ds_id] = {
        "cwt_r2_mean": round(m, 4),
        "cwt_r2_std":  round(s, 4),
        "cnn1d_r2":    P6_RESULTS[ds_id],
        "delta":       round(delta, 4),
        "all_seeds":   r2s,
    }

# ── Summary ──────────────────────────────────────────────────────────────────

cwt_means = [all_results[i]["cwt_r2_mean"] for i in DATASETS]
p6_means  = [P6_RESULTS[i] for i in DATASETS]

print("-" * 55)
print(f"{'Mean':<14} {np.mean(cwt_means):>11.4f} {'':>7} {np.mean(p6_means):>11.3f} "
      f"{np.mean(cwt_means)-np.mean(p6_means):>+7.3f}")

# ── Bar chart comparison ──────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(DATASETS))
w = 0.35
cwt_vals = [all_results[i]["cwt_r2_mean"] for i in DATASETS]
cwt_stds = [all_results[i]["cwt_r2_std"]  for i in DATASETS]
p6_vals  = [P6_RESULTS[i] for i in DATASETS]

ax.bar(x - w/2, p6_vals,  w, label="CNN-1D P6 (raw signal)", color="C3", alpha=0.8)
ax.bar(x + w/2, cwt_vals, w, yerr=cwt_stds, capsize=5,
       label="CWT+CNN-2D P10 (scalogram)", color="C2", alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f"ds{i}\n({TEMP_LABEL[i]})" for i in DATASETS])
ax.set_ylabel("R²"); ax.set_ylim(0, 1.05)
ax.axhline(0.8, color="green", lw=0.8, linestyle="--", alpha=0.6, label="R²=0.8")
ax.set_title("Phase 10 — CWT+CNN-2D vs CNN-1D P6", fontsize=11)
ax.legend()
fig.tight_layout()
fig.savefig(RESULTS_DIR / "cwt_vs_cnn1d.png", dpi=150)
plt.close(fig)

with open(RESULTS_DIR / "cwt_results.json", "w") as f:
    json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)

print(f"\n[Phase 10 complete] → {RESULTS_DIR}")
