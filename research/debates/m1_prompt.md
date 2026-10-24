You are in an adversarial two-model debate with Claude. Claude wrote the proposal below for modularizing this repo and organizing its docs. Your job: attack it.

RULES: No compliments, no politeness filler, no "overall good". Concrete objections only, each with evidence (file:line, a concrete failure scenario, or tool behaviour you can verify). Concede a point only if you cannot find a real problem, and say "no objection" in one line. Read-only: do not modify any files.

Read: src/frozen_protocol_audit.py, CLAUDE.md, README.md, REPORT.md, ENGLISH_EXPLANATION.md, .gitignore.

Check specifically:
1. Is the module split right for a 751-line script that AI agents must read? Too fine / too coarse / wrong seams? Is the dependency order actually consistent with the current call graph (e.g. session_signals, error_regions using centered_derivative, audits re-calling session_signals/build_features)? Where would a refactor silently change numerical output (import-time state, ordering of dict/groupby, pickle fingerprint of model objects which depends on module path?, matplotlib)?
2. Is the docs split right? Does Codex read AGENTS.md from repo root? Is `@AGENTS.md` import in CLAUDE.md a real Claude Code feature? Is duplicating headline numbers across 4 docs a real risk, and what is a cheap enforcement?
3. Anything missing that would make agents work better (e.g. a tests/ regression check, a results manifest, a glossary of terms like direction/fold/session/group).

Output in Markdown: a numbered list of objections (severity: blocker/major/minor), then your counter-proposal as a concrete tree + doc table. Keep it under 900 words.

--- CLAUDE'S PROPOSAL ---
# Claude's proposal: modularize the code and organize the docs

## Goal
Split `src/frozen_protocol_audit.py` (751 lines, one file) into small modules with clear boundaries, so that
an AI agent (Claude or Codex) can read one module and reason about it without loading the rest. Organize
the docs so both agents read the same rules. Hard requirement: every CSV in results/tables/ must stay
byte-identical after refactoring (pure refactor, no behaviour change). `python3 src/frozen_protocol_audit.py`
must keep working because CLAUDE.md names it as the main script.

## Code layout

```
src/
  frozen_protocol_audit.py   # thin entry point: main() only, same outputs and file names
  strain/                    # importable package (src/ is on sys.path when running the script)
    __init__.py
    config.py        # paths, DATASETS, TEMP_VAL, ELECTRODES, FULL_SCALE, ROLLING_WINDOWS,
                     # CALIBRATION_PREFIX_SECONDS, HGB hyperparameters
    data.py          # load_raw(); the ONLY module that reads CSV files
    calibration.py   # calibration_prefix, four_point_calibration, fusion_weight_from_points,
                     # normalize, low_strain_offset_calibration, session_signals
    features.py      # rolling, causal_derivative, centered_derivative, build_features
    models.py        # make_hgb, make_ridge, model_fingerprint
    variants.py      # VARIANTS, WINDOW_VARIANTS (declarative experiment registry)
    protocol.py      # fit_source_model, evaluate_fold, run_variants: the ONLY place where the
                     # train / calibration / test session boundary is decided
    metrics.py       # score, summarize
    audits.py        # causality_and_label_audit (+ the assert gate)
    analysis.py      # error_regions, window_sensitivity, temperature/position summaries
    plots.py         # plot_protocol_results
```

Dependency rule (one direction only, no cycles):
`config <- data <- calibration <- features <- models <- protocol <- {audits, analysis, plots} <- entry point`
`variants` and `metrics` depend only on config.
Each module starts with a docstring of 3–6 lines: responsibility, inputs/outputs, and which leakage
boundary it owns (if any).

## Docs layout

| File | Audience | Role | Rule |
|---|---|---|---|
| `AGENTS.md` (new) | any agent (Codex reads it natively) | single source of agent rules: data rules, claim/wording rules, module map, how to verify a change | the authoritative rule file |
| `CLAUDE.md` | Claude Code | `@AGENTS.md` import + Claude-only notes | no duplicated rules |
| `README.md` | humans | what, how to run, file map | short |
| `REPORT.md` | advisor / thesis | academic claims, results, limitations | every number must come from a named table in results/tables/ |
| `ENGLISH_EXPLANATION.md` | non-specialists | plain explanation | no number that is not in REPORT.md |
| `research/decisions.md` (new, tracked) | agents + student | dated decision log: decision, evidence (table/file), status (kept/rejected/superseded) | append-only |
| `research/directions/` (new) | agents + student | one file per candidate research direction with hypothesis, kill criterion, evidence | for the follow-up research question |

`docs/` stays git-ignored (generated progress reports) — so the new tracked research notes go in `research/`,
not `docs/`.

## Verification workflow (written into AGENTS.md)
1. Before a code change: `python3 src/frozen_protocol_audit.py`, keep a copy of results/tables/.
2. After: rerun, `diff -r` the tables. A refactor must give zero diff. A method change must update REPORT.md
   numbers from the tables and add a decisions.md entry.

## Known weak points of my own proposal (attack these)
- 11 modules might be too fine for a 751-line script.
- AGENTS.md + CLAUDE.md import: does `@AGENTS.md` actually work and does Codex read AGENTS.md from repo root?
- Numbers are duplicated across README/REPORT/ENGLISH_EXPLANATION/CLAUDE.md; my rule is only social, not enforced.
