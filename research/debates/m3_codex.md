- **Major — [tools/check_results.py:74](/Users/vngbh/Documents/Lab/strain-prediction/tools/check_results.py:74): the “unfrozen model” control only deep-copies a frozen model.** It tests shared object identity, without changing model state during evaluation. In memory, I disabled the before/after fingerprint comparison at `src/strain/audits.py:115`; gate 2 still passed. A regression disabling mutation detection therefore escapes this sensitivity gate.

- **Major — [tools/check_results.py:50](/Users/vngbh/Documents/Lab/strain-prediction/tools/check_results.py:50): reproduction never verifies the stored outputs.** It hashes regenerated temporary files against the manifest, without reading the corresponding `results/` files. Deleting or corrupting a stored PNG leaves every gate unaffected. A read-only mock supplying corrupt stored-output bytes returned no failures and recorded zero reads of those outputs.

- **Minor — [tools/check_results.py:186](/Users/vngbh/Documents/Lab/strain-prediction/tools/check_results.py:186): README prose claims lack locators.** Only its three table rows are registered. In-memory edits swapping the prose’s 5.1/91.0 values and changing its 37.6% improvement to 45.1% still produced an empty docs-failure list. These claims occur at `README.md:43–45`, despite `AGENTS.md:129` requiring registration of new headline numbers.

(a) **survives** — all eight current CSVs and both PNGs match both saved baseline output directories byte for byte; their manifest hashes also match.

(b) The first three controls reach the intended code paths and pass sensitivity testing. Imported `session_signals` retains calibration-module globals, so patching `calibration_prefix` affects normalization while the audit’s separately bound prefix selector remains unchanged. The fourth control has the defect above.

(c) **survives for the registered swap** — exchanging 37.6/45.1 triggered documentation failures; unchecked prose remains vulnerable as demonstrated above.

(d) **survives** — comparison with `git show HEAD:CLAUDE.md` and the saved working-tree version found the previous rules preserved, with empirical result numbers removed.

Function-body comparison against the supplied baseline found no unintended computational changes. Tests and mutations were performed in memory; no files were changed.