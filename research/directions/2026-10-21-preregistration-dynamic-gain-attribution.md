# Pre-registration: Does Re-aligning Labels Contract The Static-Dynamic Gap?

Status: frozen by the git commit that adds this version together with
`research/probes/probe_dynamic_gain_attribution.py`, before that script is run.
Design reviewed by Codex (round g1; all points accepted). Branch
`direction/dynamic-gain-attribution` (from `direction/reference-lag`).
dataset0 stays SEALED and is not used.

## Question

The frozen HGB with causal history features beats the static normalized HGB.
The reference-alignment diagnostic found resistance aligning with reference
strain 1.8-9.1 s EARLIER (v2). History windows up to 20 s could absorb such an
offset. Does the static-dynamic RMSE gap contract when labels are re-aligned
by the apparent offset?

Overlap check: `research/INDEX.md` lists this as open, not done.

Estimand note: this uses absolute opposite-session RMSE gaps (% strain), not
REPORT's mean per-direction improvement over physics (37.6 % vs 5.1 %).
Data-informed: the offsets come from the same corpus.

## Pipeline (all arms)

Re-implemented from the repository protocol with explicit row masks, so that
re-aligned labels never need extrapolation:

- Re-aligned label `eps_a(t) = eps(t - tau)` by linear interpolation; rows with
  `t - tau` outside the session's time range are **invalid** (no label).
- Calibration window: rows with `t <= 125 s` of the calibration session's own
  clock (unchanged), restricted to valid rows; repository four-level
  calibration, fusion weight, and normalization. The calibration clock is
  never restarted.
- Features: repository causal features on the FULL session (resistance
  history preserved). Training: valid rows of the six source sessions; one
  frozen HGB per fold and variant. Scoring: valid rows of the opposite target
  session. Physics = fused normalized signal on the same rows.
- Sanity gate: the original-label arm must reproduce the repository's
  per-direction static and dynamic RMSE (`results/tables`) within 1e-9;
  otherwise the run stops.
- Guards: per session exactly two offset rows with electrodes {1, 2} and finite `tau_s`, `tau_odd`, `tau_even`; strictly
  increasing finite timestamps; complete finite fold rows; identical physics
  RMSE for static and dynamic variants within a fold and label set. Variants
  are selected by name.

## Arms

| Arm | Labels | Calibration labels | Role |
|---|---|---|---|
| A0 original | `eps` | same | reference (must reproduce repository) |
| A1 oracle | each session shifted by its own v2 session offset (mean of 2 electrodes) | same as training labels | primary, diagnostic |
| A2 source-constant | per fold, all sessions shifted by the median of the 6 source sessions' offsets | same | primary for any transfer statement |
| A3 oracle, fixed normalization | as A1 | ORIGINAL labels (B, S, fusion unchanged) | sensitivity: separates normalization changes |
| A4 cross-fitted oracle | run 1: every session shifted by its odd-cycle offset (`tau_odd`, mean of 2 electrodes), target scored on its even cycles; run 2: `tau_even`, scored on odd cycles | same as training labels | sensitivity: removes same-cycle selection of tau |

Cycles: consecutive troughs of the original reference strain
(`find_peaks(-strain, prominence=0.25, distance=600)`), indexed from 0; "odd"
cycles are indices 0, 2, 4, ... and "even" cycles 1, 3, 5, ..., matching how
`tau_odd`/`tau_even` were fitted in `probe_reference_lag_v2.py`. Rows outside
complete cycles are not scored in A4 or its reference.

A4 aggregation (explicit): per direction,
`G_A4,d = (G_run1,d + G_run2,d) / 2` and the matched reference
`G_A0cf,d = (G_A0 scored on even cycles,d + G_A0 scored on odd cycles,d) / 2`,
each `G` being the static-dynamic gap of subset RMSEs (averaging subset RMSE
gaps, not pooling squared errors). `F_A4 = 1 - mean_d(G_A4,d) /
mean_d(G_A0cf,d)`, with the same labels and floor as below.

## Metric And Decision Rule

Per direction: `G_d = RMSE_static,d - RMSE_dynamic,d` on the arm's labels.
Mean gap `G`. **Relative contraction of the mean static-dynamic RMSE gap:**
`F = 1 - G_arm / G_A0` (ratio of means; never clipped).

- Denominator floor: if `G_A0 < 0.005 %` strain, "ratio uninterpretable".
- `F < 0`: "gap expansion".
- `F >= 0.5` and `G_arm,d < G_A0,d` in >= 6/8 directions: "substantial
  contraction". `F >= 0.5` with fewer than 6: "heterogeneous".
- `0 <= F <= 0.2`: "minor contraction". Otherwise: "partial contraction".
- `F > 1` means the static model beats the dynamic one after re-alignment.

Verdicts are reported separately: A1 for the diagnostic question, A2 for
transfer. If A1 is "substantial" but A2 is not, the label is
"oracle-specific contraction; transfer unsupported". Even A2 predicts an
earlier-strain target, not the original one.

Thresholds are conventions; six directions is a consistency description, not
replication. Reported always (printed and in `<out>_gaps.csv`): every `G_d` and `Delta G_d = G_arm,d - G_reference,d` (A4 against its matched subset reference); static, dynamic,
physics RMSE; leave-one-temperature-group-out `F` for A1 and A2; A3 and A4 `F`.

## Interpretation Limits

- Re-alignment changes the prediction target, calibration, normalization,
  and training targets; comparisons are within each label set only.
- The oracle offset was fitted to maximize a static linear fit on the same
  session, which can favour the static model; A4 reduces but does not remove
  this.
- No mechanism (timing, branch asymmetry, response dynamics) is identified.
