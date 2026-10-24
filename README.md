# Physics-Normalized Causal Tree Learning

Strain prediction for a resistive fiber sensor when temperature and measurement
session change, using a physically normalized and globally frozen tree model.

```text
other-temperature sessions -> train one causal HGB model -> freeze the model

new temperature/session -> fixed 125 s, four-level physical calibration
  -> z_i = (R_i - B_i) / S_i -> causal dynamic features
  -> frozen HGB -> strain estimate
```

The method is machine learning/AI, not deep learning. No additional data are
assumed or required by the current study.

## Main Result

Leave one complete temperature out of HGB training, calibrate physically on the
fixed first 125 seconds of one target session, and test on the opposite target
session:

| Method | Mean RMSE | Mean R2 | Mean improvement | Worst fold |
|---|---:|---:|---:|---:|
| Frozen causal HGB + 125 s calibration | **0.0484% strain** | **0.859** | **37.6%** | **+0.5%** |
| Frozen raw-resistance HGB, no `T` | 0.1156% strain | 0.203 | -80.8% | -414.7% |
| Raw HGB + target low-strain offset, no `T` | 0.0821% strain | 0.631 | -16.1% | -82.0% |

The normalized method improves the matched physical baseline in all eight
directional evaluations from four temperature groups under the repository's
nominal-level calibration bookkeeping; with measured band-mean calibration
levels it improves 7 of 8, with a similar mean gain (see `REPORT.md`,
Calibration Sensitivity). All groups share one
sensor/specimen and are not independent sensor replicates. This is retrospective
internal leave-one-temperature-out evidence, not external validation.

For source training, every source session likewise estimates its own `B`, `S`,
and fusion weight from its own first 125 seconds before its full causal feature
sequence is added to the global training set. Initial trailing windows use the
history available so far (`min_periods=1`), the first derivative is zero, and no
rows are dropped or padded across sessions.

## Why the Method Is Defensible

- Every major ablation uses the same frozen LOTO and fixed 125 s calibration.
- A strictly static normalized HGB improves 5.1% on average but degrades its
  worst direction by 91.0%; causal dynamic features raise this to 37.6% with a
  +0.5% worst direction.
- Smoothing alone makes the result worse (-6.7% mean).
- Ridge fails its worst direction (-108.7%).
- Residual HGB, centered features, and HGB with `T` each fail one direction.
- The final causal, no-`T`, direct HGB is the only tested variant that improves
  every direction (under nominal-level calibration bookkeeping).
- Raw HGB works within a temperature but fails when the held-out temperature
  changes. Among the tested variants, offset-and-sensitivity normalization
  provides the best mean transfer performance and the fewest failed directions.
- HGB is frozen before target calibration and receives zero target-condition
  labels for model training. Target labels are used only during the known 125 s
  proof-load calibration event.
- Prefix-causality, absent-test-label, truncated-calibration, altered-future,
  and shared-frozen-model audits all pass with exactly zero numerical
  prediction difference. No leakage was detected in these tested data-access
  paths and boundaries; this is not a proof that every possible leakage path is
  absent.
- October 2026 robustness checks (`REPORT.md`): the advantage over physics
  persists across sampled calibration-window positions, and the temporal
  inputs help beyond elapsed time and an explicit loading-program state; what
  they capture is unresolved, so the gain is not described as learning
  hysteresis.
- The final configuration is fixed inside the reported outer folds, but its
  family, calibration duration, features, and history lengths were selected
  retrospectively after examining this corpus. The result is not presented as
  an untouched estimate of the complete model-selection process.

## Scope

The contribution is the integration and validation of physical calibration,
causal dynamic descriptors, and small-data tree learning for this thermally
drawn embedded fiber sensor. It is not a new boosting algorithm and does not
claim that ML temperature compensation in general is new.

Only the existing one-sensor dataset is available. Generalization to new
sensors, materials, loading rates, or long-term operation cannot be established
and is stated as a limitation rather than deferred to hypothetical future data.

## Files

```text
src/frozen_protocol_audit.py             # entry point: runs everything, writes results/
src/strain/                              # the method, one module per responsibility
tools/check_results.py                   # check-only regression gate
results/tables/, results/figures/        # reproducible numeric results and figures
results/manifest.json                    # versions, data hashes, output hashes
REPORT.md                                # academic interpretation and limitations
ENGLISH_EXPLANATION.md                   # simple-language explanation
AGENTS.md                                # rules for coding agents (CLAUDE.md imports it)
research/SYNTHESIS.md                    # what the October 2026 analyses found
research/INDEX.md                        # every direction ever tried, and what to avoid
research/directions/, research/probes/   # pre-registrations, analysis scripts, outputs
research/debates/                        # verbatim Claude-Codex review transcripts
research/decisions.md                    # dated decision log
```

Raw data (`to_know_how/`) and local progress-report documents (`docs/`) are
ignored by Git.

## Run

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 src/frozen_protocol_audit.py
python3 tools/check_results.py     # verify reproduction, audit sensitivity, doc numbers
```
