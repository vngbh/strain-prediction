# Agent Rules

Single source of rules for every coding agent (Claude Code, Codex, others).
`CLAUDE.md` imports this file; do not duplicate rules elsewhere. This file holds
no result numbers: numbers live in `results/tables/` and are quoted in
`REPORT.md`, checked by `tools/check_results.py`.

## 1. Project In One Line

Globally frozen, physics-normalized causal tree boosting for cross-temperature
and cross-session strain prediction in one embedded resistive fiber sensor:

```
train on non-target temperatures -> freeze HGB
  -> fixed 125 s target physical calibration
  -> physical normalization z_i = (R_i - B_i) / S_i
  -> dynamic features requiring no future samples
  -> direct strain prediction
```

## 2. Where Things Are

| Path | What it is | Read it when |
|---|---|---|
| `src/frozen_protocol_audit.py` | Entry point and runner: computes everything, writes `results/` + manifest | running or changing outputs |
| `src/strain/spec.py` | Paths, constants, ordered variant registries, raw loader | changing any setting or variant |
| `src/strain/calibration.py` | **Label-access boundary**: 125 s prefix, B/S fit, fusion, normalization | anything touching `Strain` labels |
| `src/strain/features.py` | **Time-causality boundary**: derivatives, rolling features | anything touching temporal features |
| `src/strain/metrics.py` | Per-fold scores, per-variant summaries | metric definitions |
| `src/strain/protocol.py` | Frozen LOTO: source fit, target evaluation, variant runs | train/test split logic |
| `src/strain/audits.py` | Executable leakage audits + failure gate | leakage questions |
| `src/strain/reporting.py` | Regional/temperature tables, figures | figures, derived tables |
| `tools/check_results.py` | Check-only regression gate (reproduction, audit sensitivity, docs) | after any change |
| `results/tables/`, `results/figures/` | Generated outputs; never edit by hand | citing numbers |
| `results/manifest.json` | Versions, raw-data hashes, output hashes (regression reference) | reproducibility questions |
| `REPORT.md` | Academic claims, results, limitations (canonical prose) | writing about results |
| `ENGLISH_EXPLANATION.md` | Plain-language explanation | non-specialist audience |
| `research/INDEX.md` | Catalogue of every direction ever tried, where it lives, and what to avoid | before starting anything new |
| `research/SYNTHESIS.md` | What the October 2026 cycle found and the proposed thesis corrections | before writing the thesis |
| `research/debates/` | Verbatim Claude-Codex debate transcripts | tracing a decision or concession |
| `research/decisions.md` | Dated decision log | before reviving or changing a choice |
| `research/directions/` | Candidate research directions with kill criteria | planning new work |
| `to_know_how/` (git-ignored) | Raw data, source paper, advisor comments, archive | data provenance |
| `docs/` (git-ignored) | Generated progress-report documents | never as a source of truth |

Module dependency direction (no cycles): `spec <- {calibration, features, metrics}
<- protocol <- {audits, reporting} <- entry point`. `reporting` also uses
`features.centered_derivative` to label turning points of the truth signal
(offline analysis, not a model input).

### Branches

| Branch / tag | Content | Rule |
|---|---|---|
| `main`, tag `baseline-frozen-hgb-2026-10-20` | The current research (frozen physics-normalized HGB) | Backup of the thesis as it stands. Only refactors that keep `tools/check_results.py` passing, and fixes the user asks for. |
| `direction/<name>` | One new research direction each (files in `research/directions/`, `research/probes/`) | All new-direction work, including REPORT changes it motivates, stays on its branch until the user decides to merge. |

## 3. Glossary

| Term | Meaning |
|---|---|
| dataset / temperature group | One CSV `dataset1..4` = one temperature (33, 43, 52, 62 degC). `dataset0` (21 degC) is not in the main evaluation. |
| session | One `Experiment` id inside a dataset (two per group, ~20 min each, 10 Hz). Temperature is constant within a session. |
| A / B | The lower / higher sorted `Experiment` id of a group. Not loading/unloading. |
| direction | `A_to_B`: calibrate on A's first 125 s, test on all of B. `B_to_A` is the reverse. |
| fold | One (held-out group, direction) pair; 8 in total. Both directions of a group share one frozen model. |
| physics baseline | Linear `z = (R - B) / S` with the same 125 s calibration, electrode-fused. |
| mean improvement | `mean_pct_improvement`: mean of the 8 per-direction RMSE improvements (primary metric). |
| ratio-of-means | `ratio_of_mean_rmse_reduction`: `1 - mean RMSE / mean physics RMSE`. A different question; never swap with the above. |
| units | `Strain` is percent strain; full scale 0.5% = 5000 microstrain. |

## 4. Data Rules

- Use `dataset1.csv` to `dataset4.csv` for the main cross-session evaluation.
- Do not fit preprocessing or model-selection choices on the test session.
- Preserve temporal order. Final features are trailing and the derivative is
  causal. Centered SG derivatives belong only to the labelled offline
  development ablation.
- Build features per session; never concatenate sessions before rolling.
- `session_signals(df, ...)` without `calibration_source` self-calibrates from
  `df`'s own prefix. Only source-session preprocessing may do that; target
  evaluation must pass the opposite session explicitly.
- `dataset0.csv` was sealed on 2026-10-20 and CONSUMED on 2026-10-21 by the
  one-shot confirmation on `direction/reference-lag` (marker
  `research/probes/out/dataset0_CONSUMED.json`). No untouched data remain;
  never present any later analysis as untouched confirmation.
- No additional measurements can be collected. Never make completion depend on
  future data; state unsupported generalization as a permanent scope boundary.

## 5. Scope Rules

- Keep one method. Do not return to model shopping (ANN, CNN, LSTM, CWT, Optuna,
  ROCKET, residual learning, old diagnostic scripts) unless the user explicitly
  asks. Those were exploration/history; see `research/decisions.md`.
- Never delete a tried direction. Superseded or failed work keeps a branch or an
  `archive/*` tag and an entry in `research/INDEX.md`; check the index for
  overlaps before starting new work.
- New research directions go in `research/directions/` with a hypothesis, the
  analysis on existing data, and a kill criterion, before any code is written.

## 6. Reporting Rules

- Describe the method as machine learning / AI, not deep learning.
- The novelty is the application method: target calibration and physical
  normalization plus a globally frozen causal HGB readout. Do not claim a new
  algorithm.
- Do not claim that "ML compensates temperature drift in sensors" is new. The
  narrow claim is cross-session strain estimation for thermally drawn resistive
  fiber sensors embedded in fiber-reinforced composites.
- HGB predicts strain directly. Do not say it learns only a residual; the
  residual-target variant is weaker and fails one direction
  (`frozen_protocol_ablations_summary.csv`).
- Do not describe the old target-session-trained HGB as a short-calibration
  deployment result; it was a development experiment.
- The main result is frozen LOTO with fixed 125 s level-window calibration: the
  `final_physics_causal_no_T_direct_HGB` row of
  `frozen_protocol_ablations_summary.csv`. Quote numbers from that table or from
  `REPORT.md`, never from memory.
- The eight directions come from four temperature groups on one specimen. Never
  call directions or groups independent replicates.
- "Why not pure physics?": cite the matched 125 s comparison in the summary
  table (final variant vs physics, every direction improved); smoothing alone,
  Ridge, and raw/raw+offset HGB fail; removing `T` does not rescue raw HGB.
- All final ablations cite `frozen_protocol_ablations_summary.csv`, not the
  historical within-temperature development table.
- Say "requires no future samples" or "online-compatible", not "zero delay"
  (21-sample smoother ~1 s group lag; rolling history up to 201 samples, 20.1 s).
- HGB is deterministic here; repeated seeds are not stability evidence.
- Method selection and validation used the same finite corpus. LOTO is rigorous
  internal evidence, not untouched external confirmation.
- Audits found no leakage in the tested access paths and boundaries. Never write
  "there is no leakage."

## 7. How To Verify A Change

```bash
.venv/bin/python tools/check_results.py --docs   # after editing docs (seconds)
.venv/bin/python tools/check_results.py          # after editing code (~1 min)
```

- A refactor must pass all three gates unchanged.
- An intended result change: run `python3 src/frozen_protocol_audit.py` (rewrites
  `results/` and the manifest), update the affected numbers in `REPORT.md`,
  `README.md`, `ENGLISH_EXPLANATION.md`, rerun the checker, and add a dated
  entry to `research/decisions.md`.
- New headline numbers in docs must be registered in `CLAIMS` in
  `tools/check_results.py` with a locator that matches exactly once.
- Ordering is part of the output contract: `DATASETS` order, sorted experiment
  ids, `VARIANTS` order, feature-column order, `groupby(sort=False)` in
  summaries. Do not replace these loops with sets or default-sorted groupbys.
- PNG bytes depend on the matplotlib backend and platform recorded in the
  manifest; CSV bytes do not.
