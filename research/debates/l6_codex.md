1. Verified: HEAD is `81a4d86`; the v2 script matches that commit byte-for-byte and its registered SHA-256 (`4d337548…f8e3c5f`). Relative to v1, only the docstring, grid, and margin changed.

The CSV yields **16/16 valid**, all with `tau ≥ 0.3 s` and intervals wholly above zero: **“consistent apparent alignment (no mechanism inferred).”** The three newly resolved estimates match your figures. All 13 original **point estimates** are identical; uncertainty and secondary fields are not all unchanged. Experiment 8 electrode 1 gains a previously unresolved bootstrap draw and odd-cycle fit, changing its interval and collapse results. Paired differences match your description. Independently checked: both margins produce identical support masks in every session.

These checks verify source identity and the CSV-derived verdict; I did not rerun the full bootstrap. dataset0 remained unopened.

2. Claude’s remark needs narrower assumptions. Shared timestamps alone do not imply equal **fitted apparent offsets**: channel response, electrode-specific branch asymmetry, and model mismatch can alter them. The differences challenge a restrictive shared-offset-only explanation; they neither exclude a shared acquisition contribution nor identify its magnitude. Seven positive differences also require disclosure of the negative eighth.

Use:

> Under a model where both channels differ from the reference solely through one shared acquisition offset and otherwise follow the fitted response model, equal offsets would be expected. The electrode differences challenge that restricted explanation. However, apparent alignment also absorbs electrode-specific branch asymmetry and other response differences; it cannot separate these from a shared timing contribution. The acquisition arrangement is undocumented.

3. Exact wording for [REPORT.md](/Users/vngbh/Documents/Lab/strain-prediction/REPORT.md):

> The original pre-registered apparent-alignment analysis (v1, ±5 s grid) returned insufficient coverage: 13/16 electrode-sessions were valid, below the required 14; three 62 °C cases had minima at the positive grid boundary. A subsequently frozen, data-informed sensitivity analysis (v2, ±15 s grid) resolved all 16 and met the specified “consistent apparent alignment” rule: every estimate exceeded 0.3 s with a 95% cycle-bootstrap interval excluding zero. All 13 original point estimates were unchanged. This sensitivity result does not replace v1’s verdict or constitute independent confirmation. Positive offsets mean resistance aligns with earlier reference strain. Under the available loading program, apparent alignment cannot distinguish timing offsets from branch asymmetry or other response effects; no mechanism is inferred. Electrode-sessions are not independent replicates.

Exact wording for [research/INDEX.md](/Users/vngbh/Documents/Lab/strain-prediction/research/INDEX.md):

> Completed diagnostic: v1 (±5 s) returned insufficient coverage (13/16); data-informed v2 (±15 s; frozen in 81a4d86) resolved 16/16 and met the apparent-alignment rule. Original point estimates unchanged; no mechanism identified. Preserve both verdicts and artifacts. dataset0 remained sealed.

**Experiment 10 needs this specific REPORT caveat:**

> Experiment 10’s offsets were 9.121 and 8.478 s. Neither electrode had an estimable cross-fitted loop-area collapse ratio: no evaluation cycle satisfied the required bin-coverage rule. Its alignment estimates therefore lack that secondary check; they must not be interpreted as measured acquisition delays.