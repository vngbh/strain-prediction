1. **Frozen-run audit.** HEAD is `b82b7691bfc1770728e66aa59705eee4bbf9bac8`; working-tree and staged diffs against it are empty. Only the output CSV is untracked. This verifies source identity; the CSV lacks recorded execution provenance.

The [CSV](/Users/vngbh/Documents/Lab/strain-prediction/research/probes/out/reference_lag_per_electrode_session.csv) gives **13/16 valid**, below the frozen minimum of 14: **“insufficient coverage.”** All 13 satisfy the positive-offset condition individually.

Recomputing only the three unresolved cases on the original grid confirmed unique minima at **+5.0 s**, with RMS still decreasing:

| Experiment/electrode | RMS at +4.9 → +5.0 s, resistance units |
|---|---:|
| 9/1 | 2337.36 → 2335.42 |
| 10/1 | 3067.07 → 3030.42 |
| 10/2 | 4908.84 → 4823.42 |

These establish boundary minima, without locating an optimum beyond +5 s. Dataset0 was untouched; no files were written.

2. **Wider-grid follow-up.** Legitimate as a dated, prospectively frozen **data-informed sensitivity analysis**. Seeing the original outcomes prevents treating it as untouched confirmation; [COS guidance](https://www.cos.io/initiatives/prereg) explicitly distinguishes such analyses.

Required safeguards:

- Disclose `b82b769`, the original verdict, all observed edge evidence, and what was known before choosing the new range. Preserve and report both runs.
- Justify ±15 s independently where possible. The existing registration supplies no documented physical timing bound; otherwise describe this as an engineering search range.
- Freeze one range, unchanged resolution/bootstrap/decision rules, and a stopping rule: further boundary hits remain unresolved.
- Apply the new procedure uniformly to all 16. Preserve the original 13 estimates in their original artifact; allow follow-up estimates/CIs to change and report those changes. Splicing three recovered cases into the original 13 would create a different procedure.
- Extend the fixed-support policy consistently. A 15 s margin leaves support unchanged in the three checked cases; verify the remainder and disclose any changes.
- Freeze code, command, versions and input/output hashes before execution. Dataset0 remains sealed.

3. **Descriptive interpretation.** Negative harmonic delays do **not** establish evidence against physical delay here. Under an ideal positive-gain pure delay, phases imply the same delay modulo phase wrapping. But finite-cycle linear detrending does not commute with shifting.

A synthetic counterexample using the frozen detrending calculation: an exact **3 s delay**, 114 s triangular period, with loading occupying 41% of the period, produces apparent delays **0.85, −0.35, −5.05 s** at harmonics 1/3/5. Thus the observed sign pattern can occur under pure delay. Static nonlinearity can also mix/reverse harmonics. Triangular harmonics themselves do not cause leakage in an exact periodic DFT; imperfect cycle windows and weak harmonic amplitudes require separate checks.

“Shift-only RMS comparable to HGB RMSE” is defensible only as a units/scale observation. It compares within-session, label-fitted reference displacement on selected support against cross-session prediction error. The former excludes unresolved cases. It supplies neither an error floor nor an explained fraction of HGB error.

The temperature description also overstates: available-electrode means are **2.62, 3.86, 3.70, 4.61 s**; 62 °C contributes only one resolved electrode-session.

**REPORT wording:**

> The frozen alignment analysis returned insufficient coverage (13/16; minimum 14). Resolved offsets were positive (1.84–4.61 s); available cross-fitted collapse ratios were 0.10–0.33. These are descriptive findings and do not identify a delay mechanism or quantify explained prediction error.

**INDEX wording:**

> Frozen reference-lag test: insufficient coverage; three unresolved cases had +5 s boundary minima. Any wider-grid follow-up is data-informed sensitivity analysis.