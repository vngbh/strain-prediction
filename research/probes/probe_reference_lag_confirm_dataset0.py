"""ONE-SHOT confirmation of the apparent-alignment finding on the sealed dataset0 (21 degC).

Implements research/directions/2026-10-21-preregistration-confirm-dataset0.md exactly. This script
must be run once, after the pre-registration commit. It reuses the frozen v2 estimator verbatim
(definitions executed from probe_reference_lag_v2.py after a hash check), first proves on a
development session that it reproduces the v2 output exactly, and only then reads dataset0.

Usage: .venv/bin/python -I research/probes/probe_reference_lag_confirm_dataset0.py <repo root> <out prefix>
"""
import hashlib
import io
import json
import platform
import subprocess
import sys
from datetime import datetime
from importlib import metadata

import numpy as np
import pandas as pd

root, out_prefix = sys.argv[1], sys.argv[2]
V2 = f"{root}/research/probes/probe_reference_lag_v2.py"
V2_SHA256 = "4d33754856d547c4790a04d4cfc55d0ef7bbf2ab9247b4dfbc7ad1499f8e3c5f"
DATASET0 = f"{root}/to_know_how/files/dataset0.csv"
DATASET0_SHA256 = "7862bef32dbed29c7f249939aff2057d451b70d924d60606746a477132af0d7e"
CONSUMED = f"{root}/research/probes/out/dataset0_CONSUMED.json"  # persistent one-shot marker


def stop(reason):
    raise SystemExit(f"RUN STOPPED (dataset0 not opened unless stated): {reason}")


GATE_CSV = f"{root}/research/probes/out/reference_lag_v2_per_electrode_session.csv"
PINS = {GATE_CSV: "43533eb5533c77166763b2ee38c8373bbc97005b9b631ee0c7b1667cf097e17f",
        f"{root}/src/strain/spec.py": "befaf4268d835df19e5d2e487de0407e9fba3d98d36e17226a84f43ea2a2005d"}
ENVIRONMENT = {"python": "3.13.11", "numpy": "2.4.4", "pandas": "3.0.2", "scipy": "1.17.1"}
for path, digest in PINS.items():
    if hashlib.sha256(open(path, "rb").read()).hexdigest() != digest:
        stop(f"pinned input hash mismatch: {path}")
actual = {"python": platform.python_version(), **{n: metadata.version(n) for n in ("numpy", "pandas", "scipy")}}
if actual != ENVIRONMENT:
    stop(f"environment differs from the pinned one: {actual}")
source = open(V2, "rb").read()
if hashlib.sha256(source).hexdigest() != V2_SHA256:
    stop("v2 estimator hash mismatch")
definitions = source.decode().split("\nrows = []\n")[0]
v2 = {"__name__": "v2_definitions"}
sys_argv, sys.argv = sys.argv, [V2, root, "/dev/null"]
try:
    exec(compile(definitions, V2, "exec"), v2)
finally:
    sys.argv = sys_argv
np_, refine, rms_curve, support_mask, cycles_of, interval = (v2[k] for k in
                                                           ("np", "refine", "rms_curve", "support_mask", "cycles_of", "interval"))
SHIFTS, N_BOOT, ELECTRODES = v2["SHIFTS"], v2["N_BOOT"], v2["ELECTRODES"]


def analyze(s, rng):
    """Verbatim core of the v2 per-session loop: tau*, paired bootstrap intervals, cross-fitted taus."""
    eps = s.Strain.to_numpy()
    t = s.Idx.to_numpy() - float(s.Idx.iloc[0])
    cycles = cycles_of(eps)
    odd, even = set(range(0, len(cycles), 2)), set(range(1, len(cycles), 2))
    full = support_mask(t, cycles)
    picks = [rng.integers(0, len(cycles), len(cycles)) for _ in range(N_BOOT)]
    boot_by_electrode = {}
    for k, electrode in enumerate(ELECTRODES, start=1):
        r = s[electrode].to_numpy()
        boot = []
        for pick in picks:
            weights = np.zeros(len(t))
            for i in pick:
                a, b = cycles[i]
                weights[a:b] += 1
            keep = (weights > 0) & full
            rep = np.repeat(np.flatnonzero(keep), weights[keep].astype(int))
            curve = []
            for shift in SHIFTS:
                X = np.column_stack([np.ones(len(rep)), np.interp(t[rep] - shift, t, eps), t[rep]])
                coef, *_ = np.linalg.lstsq(X, r[rep], rcond=None)
                curve.append(float(np.sqrt(np.mean((r[rep] - X @ coef) ** 2))))
            boot.append(refine(np.array(curve)))
        boot_by_electrode[k] = np.array(boot)
    diff_ci = interval(boot_by_electrode[1] - boot_by_electrode[2])
    rows = []
    for k, electrode in enumerate(ELECTRODES, start=1):
        r = s[electrode].to_numpy()
        ci = interval(boot_by_electrode[k])
        rows.append({"electrode": k, "cycles": len(cycles), "tau_s": refine(rms_curve(t, eps, r, full)),
                     "ci_low": ci[0], "ci_high": ci[1],
                     "boot_unresolved_fraction": float(np.mean(~np.isfinite(boot_by_electrode[k]))),
                     "session_diff_ci_low": diff_ci[0], "session_diff_ci_high": diff_ci[1],
                     "tau_odd": refine(rms_curve(t, eps, r, support_mask(t, cycles, odd))),
                     "tau_even": refine(rms_curve(t, eps, r, support_mask(t, cycles, even)))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- gate: reproduce v2 on a development session
dev = v2["load_raw"](1)
dev = dev[dev.Experiment == 3].reset_index(drop=True)  # first session of the v2 loop (same rng start)
mine = analyze(dev, np.random.default_rng(0))
gate_table = pd.read_csv(GATE_CSV)
ref = gate_table
ref = ref[(ref.dataset == 1) & (ref.experiment == 3)].sort_values("electrode").reset_index(drop=True)
cols = ["tau_s", "ci_low", "ci_high", "tau_odd", "tau_even", "session_diff_ci_low", "session_diff_ci_high"]
if len(ref) != 2 or not np.allclose(mine[cols].to_numpy(), ref[cols].to_numpy(), rtol=0, atol=1e-12):
    stop("estimator does not reproduce the v2 development output")
print("gate passed: v2 estimator reproduced exactly on dataset1 experiment 3")

# ---------------------------------------------------------------- open dataset0 (once)
record = {"opened_at": datetime.now().astimezone().isoformat(),
          "git_head": subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True,
                                     text=True).stdout.strip(),
          "python": platform.python_version(),
          "packages": {n: metadata.version(n) for n in ("numpy", "pandas", "scipy")},
          "status": "opened; analysis not finished"}
try:
    with open(CONSUMED, "x") as marker:  # exclusive: a second attempt cannot start
        json.dump(record, marker, indent=2)
except FileExistsError:
    raise SystemExit("RUN REFUSED: dataset0 was already consumed (see dataset0_CONSUMED.json); no retry allowed")
print("OPENING SEALED dataset0 (one-shot confirmation); consumption recorded before reading")


def finish(update):
    with open(CONSUMED) as marker:
        current = json.load(marker)
    current.update(update)
    with open(CONSUMED, "w") as marker:
        json.dump(current, marker, indent=2)


try:
    raw_bytes = open(DATASET0, "rb").read()
    if hashlib.sha256(raw_bytes).hexdigest() != DATASET0_SHA256:
        raise RuntimeError("dataset0 hash mismatch")
    frame = pd.read_csv(io.BytesIO(raw_bytes)).sort_values(["Experiment", "Idx"]).reset_index(drop=True)
    results = []
    for experiment in sorted(frame.Experiment.unique()):
        s = frame[frame.Experiment == experiment].reset_index(drop=True)
        results.append(analyze(s, np.random.default_rng(0)).assign(experiment=experiment,
                                                                   temperature_mean=float(s.Temperature.mean())))
    out = pd.concat(results, ignore_index=True)
    out.to_csv(out_prefix + "_dataset0.csv", index=False)

    valid = np.isfinite(out.tau_s) & np.isfinite(out.ci_low) & np.isfinite(out.ci_high)
    p1 = bool(valid.all() and (out.tau_s >= 0.3).all() and (out.ci_low > 0).all())
    by_session = out.groupby("experiment").first()
    p2 = bool((np.isfinite(by_session.session_diff_ci_low) & (by_session.session_diff_ci_low > 0)).all())
    threshold = float(gate_table[gate_table.temperature == 33].tau_s.mean())
    pairs = out.groupby("experiment").tau_s
    complete = pairs.apply(lambda v: len(v) == 2 and bool(np.isfinite(v).all()))
    session_mean = pairs.mean()
    p3 = bool(complete.all() and (session_mean < threshold).all())
    pd.set_option("display.width", 220)
    report = "\n".join([
        out.round(4).to_string(index=False),
        "",
        "P1 (primary): every electrode-session has tau* >= 0.3 s with interval > 0 -> "
        + ("CONFIRMED" if p1 else "NOT CONFIRMED"),
        "P2 (secondary): tau1 - tau2 paired interval entirely above 0 in every session -> "
        + ("met" if p2 else "not met"),
        f"P3 (exploratory): every session's two-electrode mean tau* below {threshold:.4f} s "
        f"(33 degC development mean) -> {'met' if p3 else 'not met'} {session_mean.round(3).to_dict()}",
    ])
    finish({"status": "completed", "P1": p1, "P2": p2, "P3": p3, "output": out_prefix + "_dataset0.csv"})
except Exception as error:  # any post-access failure: NOT CONFIRMED, marker kept, no retry
    finish({"status": f"technical failure after access: {error!r}", "P1": "NOT CONFIRMED (technical failure)"})
    raise SystemExit(f"P1 (primary): NOT CONFIRMED (technical failure after access: {error!r}); no retry allowed")

print(report)
