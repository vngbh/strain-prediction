G5’s kill verdict holds, but its numerical summary and interpretation need correction. I reaggregated the CSV and inspected the scripts; I did not refit models or inspect timing-probe results.

1. **G5 rejects the prespecified diversity accuracy claim.**

   From [the CSV](/Users/vngbh/Documents/Lab/strain-prediction/research/probes/out/diversity_vs_selected_offset.csv):

   | Policy | Selected primary | Diverse primary |
   |---|---:|---:|
   | Repository | 0.043613 | 0.044847 |
   | Fixed 150 | 0.043532 | 0.044764 |

   Thus “0.0436 versus 0.0449 under both policies” is inaccurate. Both policies nevertheless meet the kill rule. Diversity has **lower mean IQR** under both policies: repository 0.012620 versus selected 0.013866. Drop the accuracy contribution; do not infer that diversity offers no spread benefit.

   The [script](/Users/vngbh/Documents/Lab/strain-prediction/research/probes/probe_diversity_vs_selected_offset.py) excludes the outer target from inner selection. However, inner models use two temperatures and one training copy; outer selected models use three temperatures and four copies. Inner rankings need not represent that final fitting regime. This limits selection optimality, but does not overturn the observed outer comparison.

   Selecting 480 s in four overlapping outer folds establishes agreement **within this four-point grid**, not a stable optimum. For held-out 43°C, its repository inner advantage over 240 s is only approximately 0.00031. Neither finer timing resolution nor offsets beyond 720 s were tested.

   `n_iter=150` means no early termination. Repository auto stopping still reserves validation rows, explaining why the policies differ.

   The “0.0496 → 0.0436” observation is retrospectively selection-biased: the candidate grid followed corpus-wide F1/F2 inspection. Source-only inner selection does not erase that earlier access. It also compares one-copy baseline training against four-copy selected training. Retain it as exploratory sensitivity evidence, not an unbiased improvement estimate. G6 confirms that across-position optimization does not establish superiority at the protocol’s default window.

2. **The report’s opening overstates experimental control; its numerical checks are incomplete.**

   [REPORT.md:197](/Users/vngbh/Documents/Lab/strain-prediction/REPORT.md:197) says only target calibration changes. False: bookkeeping changes source calibration and refits HGB; static/dynamic analyses change model inputs. Replace it with: “These analyses retain source-only fitting and frozen target evaluation while varying calibration conventions, window positions, or feature sets.”

   The reported bookkeeping values, window win counts, and median RMSE values agree with their CSVs. However, “strongly dependent within sessions” needs qualification: at 52°C B_to_A, position–coverage Spearman correlation is −0.074 and position–low-band mean is 0.184.

   The temporal-control paragraph has scripts but **no committed CSV backing**: both control scripts print results without saving them.

   [The registered checks](/Users/vngbh/Documents/Lab/strain-prediction/tools/check_results.py:236) currently match once, correctly. Their weaknesses are:

   - `_weak_fold_windows` checks only the weakest direction’s numerator/denominator; it does not verify that seven other directions win everywhere, or that eight directions exist.
   - The 813 evaluations, 10 s spacing, fold identities, trends, and temporal-control comparisons are unchecked.
   - Locators are document-wide and omit comparison identity. The generic “at every position…” locator could match an unrelated comparison after editing. Scope them to this section and name both comparators.

3. **Before timing results: distinguish a useful interaction from another retrospective winner.**

   For a thesis contribution, I would require the prespecified ≥3/4 matching-bin result **plus** consistent matching advantages in at least three temperature groups, with direction-level effects reported. For an actionable selection claim, require ≥5% lower primary RMSE than a source-only selected fixed offset, using a timing rule fixed without outer-target labels.

   Pooled bin winners alone are insufficient. One offset winning every bin supports convention sensitivity and rejects matching. Tiny advantages, group reversals, or gains disappearing under matched training conditions make timing a dead end as a contribution. Neither outcome identifies timing causally: window content remains confounded.

The normal documentation command failed on Matplotlib’s unwritable cache. Evaluating its unchanged claim functions without the plotting import returned no failures. No files were written.