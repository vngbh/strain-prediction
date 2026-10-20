"""POST-HOC deviation fix for the preconditioning pre-registration (Codex round 8).

The frozen script's leakage gate compared c1, calibration parameters and fusion weights, but the
pre-registration also required identical PREDICTIONS. This post-hoc check (written after the
frozen run) compares HGB predictions for P1, P2, P3 when every calibration-session label after
250 s is corrupted. It also writes explicit P0d rows (identical to P0 by construction).

Usage: .venv/bin/python -I research/probes/probe_preconditioning_posthoc_leak.py <repo root> <out prefix>
"""
import importlib.util
import sys

import numpy as np
import pandas as pd

root = sys.argv[1]
spec = importlib.util.spec_from_file_location("frozen", f"{root}/research/probes/probe_preconditioning.py")
# Load only the helper definitions of the frozen script, not its scoring section.
source = open(spec.origin).read().split("# ---------------------------------------------------------------- pre-scoring")[0]
helpers = {"__name__": "frozen_helpers"}
sys.argv = [sys.argv[0], root, sys.argv[2]]
exec(compile(source, spec.origin, "exec"), helpers)

sessions, rule, window, features, elapsed = (helpers[k] for k in ("sessions", "rule", "window", "features", "elapsed"))
DATASETS, EVENT = helpers["DATASETS"], helpers["EVENT_SECONDS"]
make_hgb = helpers["make_hgb"]

max_diff = 0.0
for held in DATASETS:
    for name in ("P1", "P2", "P3"):
        x, y = [], []
        for d in DATASETS:
            if d == held:
                continue
            for s in sessions(d):
                x.append(features(s, window(s, *rule(name, s)))[0])
                y.append(s["Strain"].to_numpy())
        model = make_hgb().fit(np.vstack(x), np.concatenate(y))
        a, b = sessions(held)
        for calib, test in ((a, b), (b, a)):
            corrupted = calib.copy()
            late = elapsed(corrupted) > EVENT
            corrupted.loc[late, "Strain"] = np.random.default_rng(1).uniform(-5, 5, late.sum())
            clean = model.predict(features(test, window(calib, *rule(name, calib)))[0])
            dirty = model.predict(features(test, window(corrupted, *rule(name, corrupted)))[0])
            max_diff = max(max_diff, float(np.max(np.abs(clean - dirty))))
print(f"post-hoc prediction-level leakage check: max |prediction difference| = {max_diff:.3e}",
      "PASS" if max_diff == 0.0 else "FAIL")

rmse = pd.read_csv(sys.argv[2] + "_rmse.csv")
p0d = rmse[rmse.protocol == "P0"].assign(protocol="P0d")
p0d.to_csv(sys.argv[2] + "_p0d_rows.csv", index=False)
print(f"wrote {len(p0d)} explicit P0d rows (identical to P0 by construction)")
