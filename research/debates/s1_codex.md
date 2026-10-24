**Verdict: redesign the feasibility test before building a Kalman filter.** The PCA establishes low-dimensional fitted-parameter trajectories; it does not establish label-free observability or transferable network-state dynamics. I reproduced the probe. Dataset0 remained untouched; no files were written.

Prior work already covers much of the proposed framing:

| Prior work | What exists |
|---|---|
| Wang et al., 2021, [10.1016/j.compscitech.2021.108645](https://doi.org/10.1016/j.compscitech.2021.108645) | Carbon-particle/polymer fabric sensors: electromechanical relaxation model with gauge factors and status parameters. The discussion explicitly proposes Kalman sensor fusion; the demonstrated result is resistance-relaxation prediction. |
| Yong and Aw, 2022, [10.1007/s11220-022-00392-4](https://doi.org/10.1007/s11220-022-00392-4) | Carbon-black/elastomer resistance dynamics modeled through viscoelasticity, with a stated validity boundary when conductive filler fractures. |
| Amindehghan et al., 2021, [10.1109/ICRoM54204.2021.9663470](https://doi.org/10.1109/ICRoM54204.2021.9663470) | MWCNT/PDMS dynamic system identification and inverse strain estimation addressing relaxation errors. |
| Olowo et al., 2023, [10.3390/s23198280](https://doi.org/10.3390/s23198280) | PEDOT:PSS tactile sensors: identified state-space dynamics and paired elements for temperature compensation. Adjacent polymer precedent, rather than carbon-black network-state tracking. |
| Wu et al., 2019, [10.3390/s19010186](https://doi.org/10.3390/s19010186) | Dual-sensor Kalman fusion with an explicit drift state. Its complementary sensor responses supply information absent from a single drifting channel. |
| Caglar et al., [10.1002/admt.202000957](https://doi.org/10.1002/admt.202000957) | The actual thermally drawn platform already uses distinct electrode responses for temperature/strain decoupling and discusses carbon-black reorganization. |

The search did **not locate an exact demonstration** of frozen, cross-session, label-free tracking of shared baseline **and sensitivity** state in this thermally drawn platform. That is a possible application contribution, conditional on demonstrating observability and transfer. “Kalman compensation,” “latent sensor dynamics,” and “two-channel decoupling” are already occupied ground. Search absence does not establish priority.

Claude’s regime-change argument misreads the source paper. Its 35–45 °C in-phase/out-of-phase threshold concerns **orthogonally placed fibers**, attributed to deformation geometry, Poisson effects, and constituent thermal expansion. It does not establish a network-state threshold for these aligned electrodes. The separate annealing discussion supplies a mechanism hypothesis, not identification from these recordings.

The overlap objection needs precision: starts separated by 130 s produce non-overlapping 125 s windows. Nevertheless, adjacent windows remain serially dependent. Seven to nine observations provide weak support for stable four-dimensional loadings.

Time is already a strong alternative explanation. My additional calculation on the same sampled coefficient table found that a separate affine-time trend for each coefficient explains **47–97%** of pooled-scaled variation within sessions. Exactly proportional linear trends would give rank one without any shared physical state. Arbitrary monotone trends need not, but high PC1 alone cannot discriminate these explanations. Shared strain bands, branch imbalance, and correlated intercept/slope estimation errors further complicate interpretation.

More fundamentally, the PCA uses **strain labels throughout each session** to estimate B/S. Online operation observes neither those coefficients nor strain. Low-dimensional oracle coefficients do not prove online identifiability.

A sharper test, confined to datasets1–4:

1. Re-estimate coefficients using measured strain, common strain coverage, and balanced branches. Quantify coefficient uncertainty and covariance using contiguous blocks. Repeat with alternative window phases and first-cycle exclusion.
2. Learn coefficient loadings and scaling from source temperatures only. On held-out sessions, infer an oracle state from one electrode’s fitted coefficients and predict the other electrode’s coefficients. Compare against independent time-trend and loading-program controls. This tests transferable coupling before attempting label-free inference.
3. Test actual observability. For known functions \(B_i(s),S_i(s)\), the instantaneous Jacobian for unknown strain and state is
   \[
   J=
   \begin{bmatrix}
   S_1 & B'_1+\varepsilon S'_1\\
   S_2 & B'_2+\varepsilon S'_2
   \end{bmatrix}.
   \]
   Examine noise-scaled conditioning and finite-horizon recovery. Nearly parallel columns make strain and state difficult to distinguish. A filter can conceal this through restrictive priors; it cannot manufacture information.

Unrestricted functions also permit \( \tilde\varepsilon=\varepsilon+a(s)\), \( \tilde B_i=B_i-S_i a(s)\), leaving resistance unchanged. Dynamics and calibration restrictions must eliminate such ambiguities. A prefix anchors the initial mapping, not every subsequent drift/strain tradeoff.

E6 rejects the tested additive difference correction; it does not logically reject multiplicative tracking. But the proposed model must demonstrate that its extra sensitivity state supplies identifiable information. Otherwise it merely adds flexibility after a failed simpler correction.

Frozen LOTO at 33 and 62 °C requires extrapolating source-learned relationships. Regime-dependent loadings add parameters precisely where support is weakest. Temperature and session remain nested, so changing loadings cannot presently be assigned uniquely to temperature.

Dataset0 can support only a **narrow, previously exposed confirmation exercise**. With one session, there is no opposite-session calibration. Prespecify same-session first-125-s calibration and score only the subsequent suffix. First develop that exact protocol on datasets1–4. At 21 °C it tests downward extrapolation and subsequent within-session tracking—not cross-session transfer, specimen generalization, or the proposed mechanism. Disclose its earlier exposure.

The next step is the source-only coupling and observability test above. Proposed kill criteria, fixed before execution: require at least **10% lower held-out coefficient-prediction error than the strongest time/program control in every temperature group**, plus distinguishable strain/state recovery at measured noise levels. Failure ends the shared-state method direction. Dataset0 stays sealed throughout this gate.