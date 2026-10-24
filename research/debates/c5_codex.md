`git diff ea86e3b` is empty. The confirmation script, preregistration, v2 estimator, gate CSV, and `spec.py` match that commit; all three pinned input hashes match. Only the two run artifacts are untracked.

The [marker](/Users/vngbh/Documents/Lab/strain-prediction/research/probes/out/dataset0_CONSUMED.json) records `2026-10-21T11:06:06.614399+09:00`, HEAD `ea86e3bba12179d355fcc005fb756f2137214406`, the pinned environment, `completed`, and verdicts false/true/true. In the frozen source, completion requires passing the development gate before dataset0 access. This is consistent with the reported execution; the marker alone cannot independently prove execution history.

Recalculation from the [CSV](/Users/vngbh/Documents/Lab/strain-prediction/research/probes/out/reference_lag_confirm_dataset0.csv) gives:

- **P1 not confirmed:** both estimates are below +0.3 s; electrode 2’s interval is negative.
- **P2 met:** paired interval `[0.236214, 0.268340] s` lies above zero.
- **P3 met:** mean `−0.031964 s < 2.617523 s`.

No confirmation code was executed; dataset0 was not read.

The proposed interpretation overreaches at “a fixed acquisition offset would appear at every temperature.” Apparent alignment absorbs response differences as well as timing. A constant timing contribution could coexist with changing response effects, including cancellation. The observations challenge a constant-offset-**only** model under otherwise comparable acquisition and fitted-response conditions; those conditions are undocumented. They do not exclude a shared contribution.

“Compatible with viscoelastic transfer” is acceptable only as an explicitly untested possibility alongside acquisition/setup differences and branch asymmetry. These results provide no discrimination favoring it. “Near zero” describes magnitude, not statistical equivalence to zero: both intervals exclude zero. The development range is **1.836–9.121 s**.

For [REPORT.md](/Users/vngbh/Documents/Lab/strain-prediction/REPORT.md), append:

> The one-shot dataset0 confirmation, preregistered in ea86e3b, did not confirm primary prediction P1. At approximately 21 °C, experiment 2 yielded apparent offsets of +0.095 s (95% cycle-bootstrap interval [0.046, 0.153]) and −0.159 s ([−0.217, −0.090]). Secondary prediction P2 was met: the paired electrode-difference interval was [0.236, 0.268] s. Exploratory prediction P3 was met: the two-electrode mean was −0.032 s, below the 33 °C development mean of 2.6175 s. Thus, the prespecified positive apparent-alignment finding remains limited to the eight development sessions at 33–62 °C. This single additional session from the same specimen establishes neither temperature dependence nor mechanism and does not exclude a shared acquisition-offset contribution. Acquisition comparability is undocumented. Intervals remain conditional on the cycle-bootstrap assumptions and 100 resamples. No untouched confirmation data remain.

For [INDEX.md](/Users/vngbh/Documents/Lab/strain-prediction/research/INDEX.md), update the reference-alignment status and append:

> One-shot dataset0 confirmation (frozen ea86e3b): P1 not confirmed; P2 met; exploratory P3 met. The 21 °C session yielded +0.095/−0.159 s. Positive apparent alignment remains a 33–62 °C development finding; no temperature dependence or mechanism established. Preserve the confirmation CSV and consumption marker. No confirmatory retry.

Replace the current sealing statement with:

> dataset0 was consumed on 2026-10-21 and is no longer sealed; no untouched data remain. Earlier exposure is disclosed in the confirmation preregistration.