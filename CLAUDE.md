# Project Notes

Current direction: **globally frozen, physics-normalized causal tree boosting**
for cross-temperature and cross-session resistive-fiber strain prediction.

The project should stay focused on one method:

```
train on non-target temperatures -> freeze HGB
  -> fixed 125 s target physical calibration
  -> physical normalization z_i = (R_i - B_i) / S_i
  -> dynamic features requiring no future samples
  -> direct strain prediction
```

## What To Keep

- Main script: `src/frozen_protocol_audit.py`
- Main results: `results/tables/` and `results/figures/`
- Main explanation: `REPORT.md`
- Simple explanation: `ENGLISH_EXPLANATION.md`
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
- Preserve temporal order. The final method uses trailing features and the
  causal derivative requiring no future samples. Centered SG derivatives belong only to the
  explicitly labelled offline development experiment.
- No additional measurements can be collected. Never make completion depend on
  future data; state unsupported generalization as a permanent scope boundary.

## Reporting Rules

- Describe the method as machine learning / AI, not deep learning.
- The novelty is the application method: target calibration and physical
  normalization plus a globally frozen causal HGB readout.
- Do not claim a new algorithm.
- Do not claim that the broad idea "ML compensates temperature drift in sensors" is new. Treat it as
  related work; the narrow claim is cross-session strain estimation for thermally drawn resistive
  fiber sensors embedded in fiber-reinforced composites.
- HGB predicts strain directly. Do not say that it learns only a residual. Under
  the final protocol, residual HGB is weaker (24.2% mean improvement and one
  failed direction versus 37.6% and all directions for direct HGB).
- Do not describe the old target-session-trained HGB as a short-calibration
  deployment result. It uses almost a complete labelled calibration session and
  is only a development experiment.
- The main result is frozen LOTO with fixed 125 s level-window calibration:
  mean RMSE 0.0484% strain, mean R2 0.859, mean improvement 37.6%, worst-direction
  improvement 0.5%, all 8 directions improved.
- The eight directions come from four temperature groups on one specimen. Do
  not call either the directions or groups independent replicates.
- If asked "why not pure physics?", cite the matched 125 s result: frozen HGB
  improves physics by 37.6% on average and in every direction. Smoothing alone
  worsens performance; Ridge and raw+low-strain-offset HGB both fail. Removing
  `T` from the raw comparators does not rescue them.
- All final ablations must cite `frozen_protocol_ablations_summary.csv`, not the
  historical 54% within-temperature development table.
- Say "requires no future samples" or "online-compatible", not "zero delay".
  The trailing 21-sample smoother has about 1 s group lag, and final rolling
  history reaches 201 samples (20.1 s).
- HGB is deterministic in this configuration; do not present repeated HGB seeds
  as independent stability evidence.
- State that method selection and validation use the same finite corpus. LOTO is
  rigorous internal evidence, not untouched external confirmation.
- Describe the executable audits as finding no leakage in the tested access
  paths and boundaries. Never turn that into the absolute claim "there is no
  leakage."
