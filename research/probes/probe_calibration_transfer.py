"""Quick probes for Claude's independent position (not part of the project pipeline).

1. Oracle per-session linear fit: floor for any static linear calibration.
2. Residual decomposition: slow drift (rolling mean over ~one cycle) vs fast within-cycle part.
3. Label-free temperature-indexed calibration: B_i(T), S_i(T) interpolated from the other
   temperatures' sessions (LOTO), target uses only its measured T. No target strain labels.
"""
import sys
import numpy as np
import pandas as pd

DATA = sys.argv[1]
sessions = []
for i in range(5):
    d = pd.read_csv(f"{DATA}/dataset{i}.csv")
    for e, g in d.groupby("Experiment"):
        g = g.sort_values("Idx").reset_index(drop=True)
        sessions.append(dict(ds=i, exp=e, T=g.Temperature.mean(), s=g.Strain.values,
                             R=g[["Electrode1_resistance", "Electrode2_resistance"]].values))


def fit_bs(s, R):
    A = np.vstack([np.ones_like(s), s]).T
    coef = np.linalg.lstsq(A, R, rcond=None)[0]  # 2 x 2: rows B, S; cols electrodes
    return coef[0], coef[1]


for x in sessions:
    x["B"], x["S"] = fit_bs(x["s"], x["R"])

rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
rows = []
for x in sessions:
    if x["ds"] == 0:
        continue
    s, R = x["s"], x["R"]
    z_or = (R - x["B"]) / x["S"]
    zf_or = z_or.mean(1)
    # oracle residual decomposition (electrode-mean)
    res = zf_or - s
    slow = pd.Series(res).rolling(1001, center=True, min_periods=1).mean().values  # ~100 s ~ one cycle
    # same-T other session full calibration (repeatability)
    other = [y for y in sessions if y["ds"] == x["ds"] and y["exp"] != x["exp"]][0]
    z_same = ((R - other["B"]) / other["S"]).mean(1)
    # LOTO temperature-indexed interpolation (log-linear in T, using all other temperatures incl. 21 C)
    src = [y for y in sessions if y["ds"] != x["ds"]]
    Ts = np.array([y["T"] for y in src])
    zs = []
    for k in range(2):
        for name in ("B", "S"):
            pass
    Bk, Sk = np.zeros(2), np.zeros(2)
    for k in range(2):
        lb = np.polyfit(Ts, np.log([y["B"][k] for y in src]), 2)
        ls = np.polyfit(Ts, np.log([y["S"][k] for y in src]), 2)
        Bk[k], Sk[k] = np.exp(np.polyval(lb, x["T"])), np.exp(np.polyval(ls, x["T"]))
    z_T = ((R - Bk) / Sk).mean(1)
    # same, but piecewise-linear interpolation (extrapolates flat-slope at edges)
    rows.append(dict(T=round(x["T"]), exp=x["exp"],
                     oracle_linear=rmse(zf_or, s),
                     oracle_slow_drift=float(np.sqrt(np.mean(slow ** 2))),
                     oracle_fast=float(np.sqrt(np.mean((res - slow) ** 2))),
                     sameT_other_session=rmse(z_same, s),
                     labelfree_T_interp=rmse(z_T, s),
                     B_err_pct=float(np.mean(np.abs(Bk / x["B"] - 1)) * 100),
                     S_err_pct=float(np.mean(np.abs(Sk / x["S"] - 1)) * 100)))

out = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(out.round(4).to_string())
print(out.drop(columns=["exp"]).mean().round(4).to_string())
