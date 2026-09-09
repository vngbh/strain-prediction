# Project Notes

Current direction: **physics-normalized tree boosting** for resistive fiber strain prediction.

The project should stay focused on one method:

```
same-session calibration
  -> physical normalization z_i = (R_i - B_i) / S_i
  -> tree-based ML correction
  -> strain prediction
```

## What To Keep

- Main script: `src/physics_normalized_tree_boosting.py`
- Main results: `results/tables/` and `results/figures/`
- Main explanation: `REPORT.md`
- Source/background/raw data archive: `to_know_how/`

## What Not To Revive

Do not return to model shopping across ANN, CNN, LSTM, CWT, Optuna, ROCKET, residual learning, or
old diagnostic scripts unless the user explicitly asks. Those were exploration/history.

## Data Rules

- `Strain` is in percent. Full scale is 0.5% strain = 5000 microstrain.
- Use `dataset1.csv` to `dataset4.csv` for the main cross-session evaluation.
- `dataset0.csv` is the 21 degC reference-temperature file, but it is not part of the current main
  evaluation.
- Do not fit preprocessing or model-selection choices on the test session.
- Preserve temporal order; only use causal rolling features.

## Reporting Rules

- Describe the method as machine learning / AI, not deep learning.
- The novelty is the method: physical calibration and normalization before small-data ML correction.
- Do not claim a new algorithm.
- Report mean improvement, worst fold, 43 degC improvement, and seed stability.
- State the limitation clearly: one sensor dataset and a small number of independent session pairs.
