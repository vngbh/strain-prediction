"""Decisive check for calibration-diversity training (prespecified with Codex, round 4).

Comparator: a SINGLE source calibration offset chosen per outer fold by inner leave-one-
temperature-out validation on the three source temperatures only (target never seen), trained
on that offset replicated len(OFFSETS) times (row-matched). Diversity: all OFFSETS stacked.
Both scored on every outer target calibration window (10 s step).

Primary metric: mean over the 8 directions of the per-direction median RMSE across windows.
Spread metric: mean over directions of the per-direction IQR across windows.
Kill rule: if diversity does not lower the primary metric versus the selected single offset,
drop the diversity-specific claim; a sensitivity claim additionally needs a lower spread metric.
Run under two training policies: repository HGB (early stopping 'auto') and fixed 150
iterations (early_stopping=False).

Usage: .venv/bin/python -I research/probes/probe_diversity_vs_selected_offset.py <repo root> <out csv>
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, sys.argv[1] + "/src")
import strain.calibration as cal  # noqa: E402
from strain.features import build_features  # noqa: E402
from strain.protocol import make_hgb  # noqa: E402
from strain.spec import CALIBRATION_PREFIX_SECONDS, DATASETS, ROLLING_WINDOWS, TEMP_VAL, load_raw  # noqa: E402

STEP, INNER_STEP = 10.0, 40.0
OFFSETS = (0.0, 240.0, 480.0, 720.0)
raw = {d: load_raw(d) for d in DATASETS}
rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
cache = {}


def sessions(dataset):
    frame = raw[dataset]
    return [frame[frame.Experiment == e].reset_index(drop=True) for e in sorted(frame.Experiment.unique())]


def window(session, offset):
    t0 = float(session["Idx"].iloc[0]) + offset
    w = session[(session["Idx"] >= t0) & (session["Idx"] <= t0 + CALIBRATION_PREFIX_SECONDS)]
    return w.reset_index(drop=True)


def features(dataset, index, session, calibration_window, key):
    cache_key = (dataset, index, key)
    if cache_key not in cache:
        signals, _ = cal.session_signals(session, "physics", calibration_window)
        cache[cache_key] = build_features(signals, session["Temperature"].to_numpy(), "causal", False, ROLLING_WINDOWS)
    return cache[cache_key]


def fit(train_sets, offsets, replicas, fixed_iterations):
    x, y = [], []
    for d in train_sets:
        for i, s in enumerate(sessions(d)):
            for offset in offsets:
                f = features(d, i, s, window(s, offset), ("source", offset))
                x += [f] * replicas
                y += [s["Strain"].to_numpy()] * replicas
    model = make_hgb()
    if fixed_iterations:
        model.set_params(early_stopping=False)
    model.fit(np.vstack(x), np.concatenate(y))
    return model


def evaluate(model, dataset, step):
    out = []
    a, b = sessions(dataset)
    for direction, (ci, calib), (ti, test) in (("A_to_B", (0, a), (1, b)), ("B_to_A", (1, b), (0, a))):
        span = float(calib["Idx"].iloc[-1] - calib["Idx"].iloc[0]) - CALIBRATION_PREFIX_SECONDS
        for start in np.arange(0.0, span + 1e-9, step):
            f = features(dataset, ti, test, window(calib, start), ("target", ci, start))
            out.append((direction, start, rmse(model.predict(f), test["Strain"].to_numpy())))
    return out


def primary(frame, column):
    medians = frame.groupby(["held_out_dataset", "direction"])[column].median()
    iqr = frame.groupby(["held_out_dataset", "direction"])[column].quantile(0.75) - \
        frame.groupby(["held_out_dataset", "direction"])[column].quantile(0.25)
    return medians.mean(), iqr.mean(), medians


rows = []
for fixed in (False, True):
    policy = "fixed_150" if fixed else "repository"
    for held in DATASETS:
        sources = [d for d in DATASETS if d != held]
        inner = {}
        for offset in OFFSETS:
            scores = []
            for inner_held in sources:
                model = fit([d for d in sources if d != inner_held], (offset,), 1, fixed)
                result = pd.DataFrame(evaluate(model, inner_held, INNER_STEP), columns=["direction", "start", "rmse"])
                scores += result.groupby("direction")["rmse"].median().tolist()
            inner[offset] = float(np.mean(scores))
        chosen = min(inner, key=inner.get)
        selected = fit(sources, (chosen,), len(OFFSETS), fixed)
        diverse = fit(sources, OFFSETS, 1, fixed)
        for (direction, start, r_sel), (_, _, r_div) in zip(evaluate(selected, held, STEP), evaluate(diverse, held, STEP)):
            rows.append({"policy": policy, "held_out_dataset": held, "temperature": TEMP_VAL[held],
                         "direction": direction, "window_start_s": start, "chosen_offset": chosen,
                         "inner_scores": str({k: round(v, 5) for k, v in inner.items()}),
                         "selected_rmse": r_sel, "diverse_rmse": r_div,
                         "selected_n_iter": selected.n_iter_, "diverse_n_iter": diverse.n_iter_})
        print(f"[{policy}] held {TEMP_VAL[held]} degC: inner scores {inner} -> chosen {chosen} s; "
              f"n_iter selected={selected.n_iter_} diverse={diverse.n_iter_}", flush=True)

frame = pd.DataFrame(rows)
frame.to_csv(sys.argv[2], index=False)
for policy, part in frame.groupby("policy"):
    p_sel, s_sel, m_sel = primary(part, "selected_rmse")
    p_div, s_div, m_div = primary(part, "diverse_rmse")
    print(f"\n[{policy}] primary (mean of medians): selected={p_sel:.5f} diverse={p_div:.5f} "
          f"({100 * (p_div - p_sel) / p_sel:+.1f}%)")
    print(f"[{policy}] spread (mean IQR):         selected={s_sel:.5f} diverse={s_div:.5f}")
    print(f"[{policy}] directions where diverse median lower: {int((m_div < m_sel).sum())}/8")
    print(f"[{policy}] VERDICT: diversity-specific claim {'SURVIVES' if p_div < p_sel else 'KILLED'}; "
          f"sensitivity claim {'supported' if (p_div < p_sel and s_div < s_sel) else 'NOT supported'}")
