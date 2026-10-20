"""Frozen physics-normalized causal HGB for cross-temperature strain prediction.

Module map (dependencies point one way only; see AGENTS.md):

    spec         constants, paths, ordered variant registries, raw loader
    calibration  LABEL-ACCESS boundary: 125 s prefix, B/S fit, fusion, normalization
    features     TIME-CAUSALITY boundary: trailing/centered derivatives, rolling features
    metrics      per-fold scores and per-variant summaries
    protocol     frozen LOTO: source fit, target evaluation, variant runs
    audits       executable leakage checks over the final-variant artifacts
    reporting    regional/temperature tables and figures

Importing this package has no side effects.
"""
