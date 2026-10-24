**A. Literature findings**

I found related reports, but **no verified report matching the complete claim: seconds-scale apparent alignment between an embedded piezoresistive fiber and a surface strain gauge in a fiber-reinforced composite**. That search outcome does not establish priority.

- **Embedded FBG versus surface gauge:** Ling et al. (2005) compare both in a glass-fiber composite beam and report agreement below 100 Hz. The accessible abstract does not quantify a temporal offset. DOI: [10.4028/www.scientific.net/KEM.295-296.21](https://doi.org/10.4028/www.scientific.net/KEM.295-296.21).

- **Embedded CNT yarns:** Bai et al. (2021) report resistance hysteresis and residual resistance during cyclic loading of braided composites; their apparatus includes an extensometer and resistance strain gauge. Their “resistance lag” description concerns accumulating baseline resistance, without a quantified sensor–reference delay in seconds. DOI: [10.1155/2021/2427954](https://doi.org/10.1155/2021/2427954).

- **Intrinsic time dependence:** Anike et al. (2016) demonstrate strain-rate, hysteresis and relaxation effects in CNT yarns. Leber et al. (2023) report viscoelastic hysteresis in thermally drawn CNT/elastomer fibers. These establish adjacent response phenomena, rather than the requested embedded-versus-surface alignment measurement. DOIs: [10.3390/c2010003](https://doi.org/10.3390/c2010003), [10.1002/advs.202207573](https://doi.org/10.1002/advs.202207573).

- **Viscoelastic strain transfer:** Liang et al. model time-dependent transfer through a viscoelastic adhesive; Zhai et al. analyze embedded and bonded FBG transfer through viscoelastic materials and validate a polymer sensor package. This mechanism class is established. Ordinary elastic “shear lag” describes spatial transfer and does not itself establish temporal delay. DOIs: [10.1364/AO.57.005837](https://doi.org/10.1364/AO.57.005837), [10.1364/AO.58.004306](https://doi.org/10.1364/AO.58.004306).

- **Seconds-scale acquisition mismatch exists in composites:** Bulletti et al. report an acquisition lag below 9 s between ultrasonic measurements and a surface gauge on carbon–epoxy, attributed to gauge-meter display settling. This is an acquisition precedent, with a different sensing modality. DOI: [10.1155/2015/259621](https://doi.org/10.1155/2015/259621).

- **Dataset synchronization:** Lüders et al. explicitly disclose unsynchronized pressure/gauge acquisition systems and align their composite-vessel dataset using burst failure. ML soft-sensor calibration work also explicitly synchronizes recordings. I found no verified matching composite sensor–reference ML dataset failure report. DOIs: [10.1016/j.dib.2025.111333](https://doi.org/10.1016/j.dib.2025.111333), [10.1002/aisy.201900178](https://doi.org/10.1002/aisy.201900178).

The defensible novelty candidate is the **specific quantified apparent-alignment finding in this sensor configuration**, conditional on confirmation and the unresolved mechanism.

**B. Do not freeze yet**

The gate’s RNG premise checks out: `DATASETS=(1,2,3,4)`; dataset1’s sorted experiments are `[3,4]`. An independent replay of v2’s first session reproduced all seven checked columns within \(5.6\times10^{-17}\), passing `atol=1e-12`. Inspection found the copied numerical core consistent with v2.

Executing the inspected v2 prefix imports modules, inserts `root/src` into `sys.path`, asserts the dataset registry, and defines constants/functions. It does not load raw data. The substituted `sys.argv` supplies the required root; restoration lacks `try/finally`, but an execution failure currently aborts before target access.

Numbered blockers:

1. **One-shot consumption is unenforced.** [Lines 91–103](/Users/vngbh/Documents/Lab/strain-prediction/research/probes/probe_reference_lag_confirm_dataset0.py:91) permit reruns and output overwriting. A crash after reading or analyzing dataset0 leaves no persistent consumed marker. Before first access, create an exclusive, persistent attempt record independent of `out_prefix`; retain it on failure. Prespecify post-access technical-failure handling and prohibit a fresh confirmatory attempt.

2. **The opening boundary is misreported.** Line 92 reads the entire sealed file for hashing, before the opening announcement; hash failure then incorrectly says dataset0 was not opened. The preregistration’s “computed without reading content” is untenable unless it means copying a previously recorded hash. Mark consumption before hashing and preferably hash and parse the same byte buffer. I found no dataset0-opening path before the gate in the inspected code.

3. **P3 can pass with an unresolved electrode.** [Line 116](/Users/vngbh/Documents/Lab/strain-prediction/research/probes/probe_reference_lag_confirm_dataset0.py:116) uses pandas’ default missing-value exclusion. Require both electrode estimates finite before calculating each session mean. Specify “every session’s two-electrode mean.” Also distinguish the fixed rounded threshold `2.62` from the actual development mean, `2.6175230064425734`.

4. **Disclosure needs tightening.** Replace “not used in any October development step” with a statement explicitly bounded by the seal and alignment development; identify the disclosed October full-label probe and its date. State that the confirmation omits v2’s collapse and harmonic analyses, uses fresh session RNGs, and remains conditional on cycle-bootstrap assumptions and only 100 resamples. Pin the gate CSV, imported specification and environment. P1’s positive threshold and P2’s paired sign rule are otherwise explicit; invalid P2 intervals already fail.

Dataset0 was not accessed. The confirmation script was not executed. No files were changed.