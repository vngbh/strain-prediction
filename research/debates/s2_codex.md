**Do not freeze yet.** Neither gate was executed; no gate results were computed; dataset0 was not loaded.

1. **Gate A tests predictive transfer, but does not isolate shared-state coupling.** ([implementation](/Users/vngbh/Documents/Lab/strain-prediction/research/probes/probe_latent_gates.py:65))

   L receives the target cycle’s oracle electrode-1 coefficients, whereas the controls receive time/index only. That matches the stated oracle question, but L can win by sensing session-specific progression through a common loading program. It need not reveal coupling beyond that progression. Add a prespecified residual test: subtract source-fitted time/program predictions from both electrodes, then test whether electrode-1 residuals predict electrode-2 residuals on held-out temperatures.

   T2’s last-index hold is a defensible frozen extrapolation convention, but it tests that convention alongside coupling. Use common source/target index support for the primary comparison; report extrapolation separately. Define minimum coverage before execution.

   Pooling relative dB and dS is dimensionally legal, but gives the coefficient with greater relative variation more influence. Use source-only per-coefficient scales, report both component errors, and average session losses equally. If standardized loadings later enter B, transform them back to relative coefficient units.

   “Every temperature” is stringent, **not mathematically impossible**. However, near-zero target variation or coefficient-estimation noise can make a 10% gain unattainable even with transferable coupling. Either retain this explicitly as a conservative engineering requirement, or preregister an uncertainty/coverage-based “inconclusive” category. Failure cannot generally disprove shared-state models.

2. **Gate B’s Fisher entries are correct for the stated model at \(s=0\), conditional on known calibration and loadings.** ([implementation](/Users/vngbh/Documents/Lab/strain-prediction/research/probes/probe_latent_gates.py:104))

   The derivatives are \(S_{i0}\) for each sample’s strain and \(B_{i0}b_i+\epsilon_kS_{i0}g_i\) for the shared state. Nevertheless:

   - The inverse uses the entire 100 s horizon to estimate every sample. This is a **retrospective smoothing bound**, not a causal online bound. Rename it, or compute each sample’s bound using observations available through that sample. Declare the assumed state timescale.
   - **0.01% strain equals 100 microstrain.** It is a selectable noise budget, not an established necessary condition for improving HGB. Failing that threshold does not establish inability to beat HGB’s [0.0484% RMSE](/Users/vngbh/Documents/Lab/strain-prediction/REPORT.md:120). Change the decision language accordingly.
   - MAD(diff(R))/√2 estimates observation noise under independent Gaussian errors and negligible deterministic increments. Changing strain violates the latter assumption; common-channel and temporal noise also invalidate the diagonal white-noise model. Prespecify detrending using reference strain and local branch response, residual diagnostics, and a covariance-aware sensitivity calculation. Otherwise call this a difference-based scale, not measured sensor noise.
   - A right-singular vector of **unscaled relative changes** supplies dimensionless relative loadings; multiplying by B/S gives the required absolute derivatives. That units conversion is correct. However, the draft’s projection of two electrode-1 coordinates onto a four-dimensional vector is undefined; the code never computes that state.
   - A’s unrestricted 2×2 map can represent two factors; B independently assumes one. Use the same source-fitted rank-one model in both gates. Align its reference with calibration: median-relative loadings cannot automatically be transplanted unchanged onto prefix-relative coefficients.

3. **Implementation corrections before freezing.**

   - [Line 49](/Users/vngbh/Documents/Lab/strain-prediction/research/probes/probe_latent_gates.py:49) prepends sample zero unconditionally. Use consecutive detected troughs, or explicitly verify that zero is a trough.
   - [Line 75](/Users/vngbh/Documents/Lab/strain-prediction/research/probes/probe_latent_gates.py:75) can raise `KeyError` when dropped cycles leave an interior source-index gap. Specify interpolation or exclusion.
   - `best_control == 0` divides by zero before testing the pass condition—correcting my interim description. Guard it explicitly.
   - Replace the dense inverse with the block Schur complement. For blocks \(D,c,q\), check \(q-c^\top D^{-1}c\); unidentified cases need an infinite bound or explicit failure, never a pseudoinverse-derived finite bound.
   - Reject empty, nonfinite, zero-noise, and insufficient-coverage inputs explicitly.