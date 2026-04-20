"""
Phase 1 — Exploratory Data Analysis
Goal: Understand dataset structure, distributions, and the effect of Temperature on Strain.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from utils.data_loader import load_dataset, load_all_datasets, DATASETS, INPUT_FEATURES, OUTPUT_FEATURE

RESULTS_DIR = Path("results/phase1_eda")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TEMP_MAP = {0: 21, 1: 33, 2: 43, 3: 52, 4: 62}


# ── 1. Per-dataset summary ────────────────────────────────────────────────────

print("=" * 60)
print("DATASET SUMMARY")
print("=" * 60)

summaries = []
for i in DATASETS:
    df = load_dataset(i)
    summaries.append({
        "dataset":     i,
        "temperature": round(df["Temperature"].mean(), 1),
        "n_samples":   len(df),
        "strain_min":  round(df[OUTPUT_FEATURE].min(), 4),
        "strain_max":  round(df[OUTPUT_FEATURE].max(), 4),
        "experiments": sorted(df["Experiment"].unique().tolist()),
    })
    print(f"dataset{i}: {len(df):>6} samples | "
          f"Temp={df['Temperature'].mean():.1f}°C | "
          f"Strain=[{df[OUTPUT_FEATURE].min():.4f}, {df[OUTPUT_FEATURE].max():.4f}] | "
          f"Experiments={sorted(df['Experiment'].unique())}")

summary_df = pd.DataFrame(summaries)
summary_df.to_csv(RESULTS_DIR / "dataset_summary.csv", index=False)
print(f"\nSaved → {RESULTS_DIR}/dataset_summary.csv")


# ── 2. Correlation heatmap (all data combined) ────────────────────────────────

df_all = load_all_datasets()
cols = INPUT_FEATURES + ["Temperature", OUTPUT_FEATURE]

fig, ax = plt.subplots(figsize=(6, 5))
corr = df_all[cols].corr()
sns.heatmap(corr, annot=True, fmt=".3f", cmap="coolwarm", ax=ax)
ax.set_title("Correlation Matrix (all datasets)")
fig.tight_layout()
fig.savefig(RESULTS_DIR / "correlation_heatmap.png", dpi=150)
plt.close(fig)
print("Saved → correlation_heatmap.png")

print("\nCorrelation with Strain:")
print(corr[OUTPUT_FEATURE].drop(OUTPUT_FEATURE).sort_values())


# ── 3. Strain distribution per temperature ───────────────────────────────────

fig, axes = plt.subplots(1, len(DATASETS), figsize=(18, 4), sharey=False)
for i, ax in zip(DATASETS, axes):
    df = load_dataset(i)
    ax.hist(df[OUTPUT_FEATURE], bins=50, color=f"C{i}", alpha=0.8)
    ax.set_title(f"dataset{i} ({TEMP_MAP[i]}°C)")
    ax.set_xlabel("Strain")
    ax.set_ylabel("Count")
fig.suptitle("Strain Distribution per Temperature", fontsize=13)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "strain_distribution.png", dpi=150)
plt.close(fig)
print("Saved → strain_distribution.png")


# ── 4. Electrode resistance vs Strain (scatter, per dataset) ─────────────────

fig, axes = plt.subplots(2, len(DATASETS), figsize=(20, 8))
for i in DATASETS:
    df = load_dataset(i)
    axes[0, i].scatter(df["Electrode1_resistance"], df[OUTPUT_FEATURE],
                       s=1, alpha=0.3, color=f"C{i}")
    axes[0, i].set_title(f"E1 vs Strain ({TEMP_MAP[i]}°C)")
    axes[0, i].set_xlabel("Electrode1_resistance")

    axes[1, i].scatter(df["Electrode2_resistance"], df[OUTPUT_FEATURE],
                       s=1, alpha=0.3, color=f"C{i}")
    axes[1, i].set_title(f"E2 vs Strain ({TEMP_MAP[i]}°C)")
    axes[1, i].set_xlabel("Electrode2_resistance")

for ax in axes[:, 0]:
    ax.set_ylabel("Strain")

fig.suptitle("Electrode Resistance vs Strain per Temperature", fontsize=13)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "electrode_vs_strain.png", dpi=150)
plt.close(fig)
print("Saved → electrode_vs_strain.png")


# ── 5. Time-series signal (first 500 samples per dataset) ────────────────────

fig, axes = plt.subplots(3, len(DATASETS), figsize=(22, 10))
for i in DATASETS:
    df = load_dataset(i).head(500)
    axes[0, i].plot(df["Idx"], df["Electrode1_resistance"], color=f"C{i}", lw=0.8)
    axes[0, i].set_title(f"E1 ({TEMP_MAP[i]}°C)")

    axes[1, i].plot(df["Idx"], df["Electrode2_resistance"], color=f"C{i}", lw=0.8)
    axes[1, i].set_title(f"E2 ({TEMP_MAP[i]}°C)")

    axes[2, i].plot(df["Idx"], df[OUTPUT_FEATURE], color="black", lw=0.8)
    axes[2, i].set_title(f"Strain ({TEMP_MAP[i]}°C)")

for ax_row, label in zip(axes, ["E1 Resistance", "E2 Resistance", "Strain"]):
    ax_row[0].set_ylabel(label)

fig.suptitle("Time-Series Signals (first 500 samples)", fontsize=13)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "time_series.png", dpi=150)
plt.close(fig)
print("Saved → time_series.png")


print("\n[Phase 1 EDA complete] All plots saved to", RESULTS_DIR)
