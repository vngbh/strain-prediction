1. **[blocker] Moving configuration breaks repository paths unless explicitly corrected.** [src/frozen_protocol_audit.py:23](/Users/vngbh/Documents/Lab/strain-prediction/src/frozen_protocol_audit.py:23) computes `ROOT` using `parents[1]`. In `src/strain/config.py`, that resolves to `src/`, so loading fails at `src/to_know_how/files/` and outputs move to `src/results/`. Preserve repository-relative resolution and keep filesystem writes inside execution.

2. **[major] Eleven modules fragment the reasoning agents actually need.** `data.py` contains a three-line loader; `models.py` mostly wraps constructors. Meanwhile, understanding one prediction still requires calibration, features, models, protocol, variants, and metrics. [fit_source_model:306](/Users/vngbh/Documents/Lab/strain-prediction/src/frozen_protocol_audit.py:306) exposes that coupling. Combine related numerical transformations and keep model construction beside fitting.

3. **[major] The dependency order is feasible, but the advertised chain misrepresents the graph and boundary ownership.** Audits directly rebuild calibration/signals/features at [473](/Users/vngbh/Documents/Lab/strain-prediction/src/frozen_protocol_audit.py:473); analysis directly calls `centered_derivative` at [577](/Users/vngbh/Documents/Lab/strain-prediction/src/frozen_protocol_audit.py:577). More seriously, `session_signals` defaults its calibration source to the input session at [278](/Users/vngbh/Documents/Lab/strain-prediction/src/frozen_protocol_audit.py:278), while `calibration_prefix` decides the time boundary. Protocol cannot honestly be the “ONLY” boundary owner. Document caller obligations, explicit dependencies, and offline label access.

4. **[major] Byte identity needs an explicit ordering contract.** Feature columns follow nested signal/window/operation order at [239](/Users/vngbh/Documents/Lab/strain-prediction/src/frozen_protocol_audit.py:239); training follows dataset order and sorted sessions at [308](/Users/vngbh/Documents/Lab/strain-prediction/src/frozen_protocol_audit.py:308); summaries preserve variant order using `sort=False` at [435](/Users/vngbh/Documents/Lab/strain-prediction/src/frozen_protocol_audit.py:435). Replacing loops with sets, concatenating sessions before rolling, or accepting default group sorting can alter features, models, or CSV bytes. Preserve these deliberately.

5. **[major] The verification workflow can pass without regenerating an output.** `main()` overwrites existing files without cleaning the directory at [692](/Users/vngbh/Documents/Lab/strain-prediction/src/frozen_protocol_audit.py:692). Delete the regional-table writer during refactoring: the old CSV remains and `diff -r` passes. Generate into an empty directory, require the exact eight-file inventory, then compare bytes. Add regression tests for session resets, feature ordering, audit failures, and shared model identity. CSV comparison also misses broken figures: plotting uses positional labels at [628](/Users/vngbh/Documents/Lab/strain-prediction/src/frozen_protocol_audit.py:628); verify both PNGs render with a headless backend.

6. **[major] Unpinned dependencies undermine the fingerprint contract.** [requirements.txt:1](/Users/vngbh/Documents/Lab/strain-prediction/requirements.txt:1) specifies no versions. I verified that installed sklearn inserts `_sklearn_version` into estimator pickle state. Thus fingerprints written into CSVs can change after an environment update even without changing your code. Record Python, dependencies, raw-data hashes, configuration, source revision, and output hashes in a results manifest.

7. **[major] The documentation rules permit synchronized wrong numbers.** The same headlines appear in [README:25](/Users/vngbh/Documents/Lab/strain-prediction/README.md:25), [REPORT:116](/Users/vngbh/Documents/Lab/strain-prediction/REPORT.md:116), [ENGLISH_EXPLANATION:56](/Users/vngbh/Documents/Lab/strain-prediction/ENGLISH_EXPLANATION.md:56), and [CLAUDE:58](/Users/vngbh/Documents/Lab/strain-prediction/CLAUDE.md:58). “Exists in REPORT” cannot distinguish mean directional improvement, 37.6%, from ratio-of-means reduction, 45.1%. Generate marked result blocks from named rows/columns and run a check-only command in CI. Remove result numbers from agent instructions.

8. **[minor] The proposed research hierarchy lacks vocabulary and overconstrains provenance.** At [413](/Users/vngbh/Documents/Lab/strain-prediction/src/frozen_protocol_audit.py:413), A/B mean sorted experiment identifiers, not loading/unloading direction; one temperature has two evaluations sharing one fitted model. Define session, direction, fold, group, units, and metric aggregation. Restrict “every number comes from a table” to computed results: literature years and calibration constants require different sources. Create direction files only when candidates exist.

Codex reading root `AGENTS.md`: no objection. [Official documentation](https://learn.chatgpt.com/docs/agent-configuration/agents-md).

Literal, unfenced `@AGENTS.md` imports in `CLAUDE.md`: no objection. [Claude Code documentation](https://code.claude.com/docs/en/memory#import-additional-files).

Moving the HGB factory changing pickle through its module path: no objection; identical fitted objects from factories named `__main__` and `strain.models` produced identical pickle bytes.

Tracked `research/` beside ignored `docs/`: no objection; `.gitignore:5` ignores `docs/`.

Counter-proposal:

```text
src/
  frozen_protocol_audit.py       # compatibility entry
  strain/
    __init__.py                  # no side effects
    spec.py                      # paths, ordered registries, raw loader
    signals.py                   # calibration, normalization, features
    protocol.py                  # constructors, training, fold evaluation
    audits.py                    # perturbations and explicit failure gate
    reporting.py                 # metrics, summaries, regions, plots
    runner.py                    # orchestration and output writing
tests/
  test_protocol_regression.py
  test_signal_contracts.py
tools/
  check_results.py               # fresh outputs against immutable baseline
  render_doc_results.py          # generate/check marked blocks
requirements.lock
results/manifest.json
research/decisions.md
```

| Document | Contract |
|---|---|
| `AGENTS.md` | Rules, module map, verification commands; no headlines |
| `CLAUDE.md` | Literal import plus necessary Claude-specific notes |
| `README.md` | Setup, ignored-data prerequisite, run command, generated headlines |
| `REPORT.md` | Claims, limitations, glossary, table/column provenance |
| `ENGLISH_EXPLANATION.md` | Plain explanation; generated numerical claims |
| `research/decisions.md` | Dated evidence and supersession links |

No files modified.