# Direction: Latent Network-State Model Of The Two-Electrode Fiber Sensor

Status: proposed, 2026-10-20. Branch `direction/latent-state-model` (from
`direction/calibration-transfer-limits`). No new measurements exist or will
exist (confirmed by the user); everything uses the current corpus.

This branch deliberately leaves the single-method rule of `AGENTS.md`
("keep one method: frozen HGB"), at the user's request to develop the research
further. `main` keeps the HGB thesis unchanged.

## Motivation

- Caglar et al. (2021, pp. 5-6) report that at higher temperature the carbon
  black network of the cPE electrodes reorganizes (flocculation on annealing,
  lowering resistivity, saturating after hours, citing Hilarius et al.), and
  that loading/unloading hysteresis grows with temperature. They fit R(T) and
  dR/d(epsilon)(T) with exponentials and decouple T and strain with a neural
  network. They do not model a time-varying network state.
- Findings on the parent branch: slow error dominates the 43 degC failure; the
  short calibration cannot estimate the best static two-electrode map; the
  first cycle shows a higher apparent slope; session state shifts B/S more than
  a temperature-only model can explain.
- Hypothesis: a slow latent network state `s(t)`, shared by both electrodes but
  loading on them differently, drives both baseline and sensitivity changes.
  Two electrodes may make `s(t)` observable online without labels.

## Sealed Confirmation Set

`dataset0.csv` (21 degC, one session) is SEALED from 2026-10-20: no model
development, tuning, plotting, or summary statistics on it. It will be opened
once, for a pre-registered confirmatory test of the final model.
Prior exposure, disclosed: early project EDA (April 2026) and one probe in this
debate (`probes/probe_calibration_transfer.py`, label-free temperature
interpolation, used dataset0's full-label B/S as a source point).

## Step 0: Feasibility (identifiability), exploratory

Before building any model: within each session, are the trajectories of
B1, B2, S1, S2 (four-level calibration on sliding 125 s windows) explained by
ONE common latent factor, with loadings that are stable across sessions?
If not, a single shared state is not identifiable and the direction stops.
