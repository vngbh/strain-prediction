"""Probe: is the electrode difference z1 - z2 (label-free) an observer of slow drift?"""
import sys
import numpy as np, pandas as pd
DATA = sys.argv[1]
for i in range(1, 5):
    d = pd.read_csv(f"{DATA}/dataset{i}.csv")
    for e, g in d.groupby("Experiment"):
        g = g.sort_values("Idx"); s = g.Strain.values
        R = g[["Electrode1_resistance", "Electrode2_resistance"]].values
        A = np.vstack([np.ones_like(s), s]).T
        # calibrate on first 125 s only (deployment-like)
        m = g.Idx.values <= g.Idx.values[0] + 125
        coef = np.linalg.lstsq(A[m], R[m], rcond=None)[0]
        z = (R - coef[0]) / coef[1]
        err = z - s[:, None]                       # per-electrode error vs truth
        slow = lambda x: pd.Series(x).rolling(1001, center=True, min_periods=1).mean().values
        e1, e2 = slow(err[:, 0]), slow(err[:, 1])
        diff = slow(z[:, 0] - z[:, 1])             # label-free
        # best label-free linear map from diff to each electrode's slow error (oracle gain: upper bound)
        c1 = np.corrcoef(diff, e1)[0, 1]; c2 = np.corrcoef(diff, e2)[0, 1]
        k = np.polyfit(diff, (e1 + e2) / 2, 1)
        fused = z.mean(1); corr = fused - np.polyval(k, diff)
        rm = lambda x: np.sqrt(np.mean(x ** 2))
        print(f"T={g.Temperature.mean():.0f} exp={e:2d}  slowErr RMS e1={rm(e1):.3f} e2={rm(e2):.3f}  "
              f"corr(diff,e1)={c1:+.2f} corr(diff,e2)={c2:+.2f}  fused RMSE {rm(fused-s):.4f} -> oracle-gain corrected {rm(corr-s):.4f}  gain k={k[0]:+.2f}")
