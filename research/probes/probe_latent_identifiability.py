"""Step 0 feasibility for the latent-state model (exploratory, dataset1-4 only; dataset0 sealed).

Input: research/probes/out/first_window.csv (four-level B1, S1, B2, S2 per 10 s window start).
Per session, use NON-overlapping windows (starts every 130 s) to limit autocorrelation, convert to
relative changes vs the session median, and run PCA on the 4 standardized-by-pooled-scale series.

Questions:
  Q1 one factor: share of variance explained by the first component, per session.
  Q2 stable loadings: cosine similarity of each session's first-component loading vector with the
     pooled loading (sign-aligned); and with every other session.
  Q3 multiplicative part: do S1 and S2 change together with B1, B2 (same sign in the loading)?

Usage: .venv/bin/python -I research/probes/probe_latent_identifiability.py <repo root>
"""
import sys

import numpy as np
import pandas as pd

frame = pd.read_csv(f"{sys.argv[1]}/research/probes/out/first_window.csv")
assert set(frame.dataset) <= {1, 2, 3, 4}, "dataset0 is sealed"
cols = ["B1", "B2", "S1", "S2"]
frame = frame[np.isclose(frame.window_start_s % 130, 0)]

relative = []
for (_, _), g in frame.groupby(["temperature", "experiment"]):
    r = g[cols] / g[cols].median() - 1
    relative.append(r.assign(session=f"{g.temperature.iloc[0]}C-{g.experiment.iloc[0]}"))
relative = pd.concat(relative, ignore_index=True)
scale = relative[cols].std()


def first_component(x):
    x = (x - x.mean()) / scale
    _, s, vt = np.linalg.svd(x.to_numpy(), full_matrices=False)
    v = vt[0] * np.sign(vt[0][np.argmax(np.abs(vt[0]))])
    return v, s[0] ** 2 / (s ** 2).sum(), len(x)


pooled, pooled_share, _ = first_component(relative[cols])
print("pooled first-component loading (B1, B2, S1, S2):", np.round(pooled, 2), f"share {pooled_share:.2f}")
rows, vectors = [], {}
for session, g in relative.groupby("session", sort=False):
    v, share, n = first_component(g[cols])
    if v @ pooled < 0:
        v = -v
    vectors[session] = v
    rows.append({"session": session, "windows": n, "pc1_share": share, "cos_to_pooled": float(v @ pooled),
                 **{f"load_{c}": x for c, x in zip(cols, v)}})
table = pd.DataFrame(rows)
print(table.round(2).to_string(index=False))
names = list(vectors)
cos = np.array([[vectors[a] @ vectors[b] for b in names] for a in names])
off = cos[~np.eye(len(names), dtype=bool)]
print(f"\nQ1 median PC1 share: {table.pc1_share.median():.2f} (min {table.pc1_share.min():.2f})")
print(f"Q2 cos to pooled: min {table.cos_to_pooled.min():.2f}; pairwise cos: median {np.median(off):.2f}, "
      f"min {off.min():.2f}")
same_sign = (np.sign(table[["load_B1", "load_B2", "load_S1", "load_S2"]]).nunique(axis=1) == 1).sum()
print(f"Q3 sessions where B and S load with the same sign: {same_sign}/{len(table)}")
