# Research Synthesis, October 2026

What the October 2026 research cycle did, what survived, what failed, and what
it means for the thesis. Numbers are quoted from branch REPORT sections and
pre-registration results; follow the links rather than copying numbers from
here. Where each item lives: `research/INDEX.md`. Reviewed by Codex (round
sy1); its corrections are applied, except one point noted in Section 5.

## Tóm tắt tiếng Việt

- **Câu hỏi của thầy:** nghiên cứu có bị "đại trà hóa" vì ai cũng dùng AI
  không? Đánh giá: lo lắng có cơ sở. Dữ liệu là dữ liệu bài lab môn học, bài
  báo gốc đã tách nhiệt độ bằng mạng nơ-ron, nên "dùng AI cho cảm biến này"
  không phải đóng góp mới.
- **Đã làm:** bốn hướng mới trên bốn nhánh git, phần lớn đăng ký trước và
  được Codex phản biện đối kháng, cùng một lần kiểm chứng duy nhất trên
  dataset0 (21 °C).
- **Đứng vững (có điều kiện):** HGB hơn vật lý ở mọi vị trí calibration khảo
  sát trong bảy hướng và 98/109 vị trí trong hướng còn lại; đặc trưng thời gian
  cải thiện kết quả trong cấu hình đã thử; chu kỳ tải đầu có độ dốc biểu kiến
  cao hơn (16/16, chưa rõ cơ chế).
- **Thất bại hoặc không xác nhận:** mô hình trạng thái ẩn; bỏ chu kỳ đầu; độ
  chính xác của đa dạng hóa calibration; dự đoán chính về độ lệch căn chỉnh
  biểu kiến dương không được xác nhận ở 21 °C.
- **Phải sửa trong luận văn:** không nói "HGB học hysteresis"; kết quả "8/8"
  phụ thuộc cách ghi điểm calibration (7/8 với strain đo thực); "fusion hai
  điện cực" thực chất là điện cực 2, điện cực 1 vẫn là đầu vào riêng của HGB.
- **Nhận định (của Claude và Codex ở vòng 1, có điều kiện):** chấp nhận được
  cho luận văn nếu được trình bày như một nghiên cứu có giới hạn, tái lập được;
  chưa có đột phá mới; dữ liệu đã chạm trần.

## 1. Starting Point

`main` (tag `baseline-frozen-hgb-2026-10-20`): frozen leave-one-temperature-out
HGB with a fixed 125 s physical calibration, causal history features, direct
strain target. The advisor asked whether the work is becoming generic "AI
applied to a sensor".

Assessment (Claude and Codex agreed in round 1): the concern is justified. The
data are a course-lab dataset (MECE4120 Lab 3); the source paper (Caglar et al.
2021) already decoupled temperature and strain with a neural network and held
out 43 degC. Codex's round-1 verdict: "thesis-acceptable if framed as a
reproducible, bounded investigation"; not publishable as novel AI.

## 2. How The Work Was Done

- Each direction on its own branch; `main` keeps the thesis plus shared docs.
- Claude and Codex in adversarial review; designs reviewed before freezing and
  runs audited after where Codex was available. Audit coverage and deviations
  are documented per analysis (e.g. the latent-gates interpretation and
  calibration F5 were not reviewed by Codex because of a usage limit; the
  preconditioning test records two compliance gaps closed post hoc).
- Pre-registrations were frozen by git commit before running; a commit order
  cannot prove that no earlier execution occurred. Prespecification on this
  previously explored corpus does not create untouched confirmation.
- Registered REPORT claims have automated CSV checks; the checker validates
  registered locators, not every number in every note.
- `dataset0` (21 degC) was sealed on 2026-10-20 and consumed once on
  2026-10-21. It was unused in alignment development but had prior exposure
  (April EDA, July Phases E/F, one 2026-10-20 full-label calibration probe).

## 3. Results By Direction

| Branch | Question | Outcome |
|---|---|---|
| `direction/calibration-transfer-limits` | What limits short calibration, and what does the frozen model add? | Several descriptive findings; six sub-ideas rejected, one of them by pre-registration |
| `direction/latent-state-model` | Is a shared latent network state observable from two electrodes? | Stopped: pre-registered gates A1 and B failed |
| `direction/reference-lag` | Is the sensor consistently offset in time from the reference gauge? | v1 insufficient coverage; v2 consistent offsets at 33-62 degC (data-informed sensitivity analysis); one-shot 21 degC confirmation NOT confirmed |
| `direction/dynamic-gain-attribution` | Does re-aligning labels by that offset contract the static-dynamic gap? | Partial contraction in all four arms |

### What held up

- **Position robustness.** The frozen HGB beats matched physics across all
  sampled calibration-window positions in seven directions and 98/109
  positions in the eighth (813 dependent windows); dynamic beats static HGB at
  all positions in seven directions and 96/109 in the eighth. These do not
  establish better absolute accuracy at later positions.
  (`direction/calibration-transfer-limits`, REPORT "Calibration Sensitivity".)
- **Temporal inputs improved performance under the tested configuration.** On
  the calibration branch, adding elapsed session time to static inputs did not
  recover the dynamic performance, and an explicit causal loading-program
  state recovered only part of the gap. On the attribution branch, the four
  label re-alignment arms showed partial mean-gap contraction (F = 0.206-0.380);
  this diagnostic does not quantify alignment's contribution to the original
  dynamic advantage.
- **First-cycle apparent sensitivity.** The first loading cycle's apparent
  slope exceeds later cycles' in 16/16 electrode-sessions under
  branch-balanced fitting (12/16 with a time term); no mechanism identified.
  Candidate related literature is identified; the specific first-cycle
  comparison remains unverified against full texts.
- **Apparent alignment at 33-62 degC.** v1 (pre-registered, +/-5 s grid)
  returned insufficient coverage; v2 (data-informed, +/-15 s) found positive
  apparent offsets of about 1.8-9.1 s in all 16 development
  electrode-sessions. Mechanism not identifiable with one ramp rate. Harmonic
  phase signs do not reject a pure delay, and the shift-only RMS is neither an
  error floor nor explained prediction error.

### What failed or was not confirmed

- Temperature-only (label-free) calibration; a frozen drift corrector from the
  electrode difference; a general calibration-window selection rule;
  matching source and target calibration timing.
- Calibration-diversity training: its accuracy claim was rejected (a single
  source-selected offset was better); a spread benefit remains possible.
- Excluding the first cycle before calibration: pre-registered criterion
  unmet.
- Shared latent-state model: stopped because gates A1 and B failed; residual
  coupling beyond time/program controls remained (A2 passed in all four
  groups), and the high cross-electrode residual correlation helped rather
  than hurt separability (the opposite reading was withdrawn).
- **Apparent alignment at 21 degC:** primary prediction P1 not confirmed in the
  one-shot dataset0 test (offsets +0.095 s and -0.159 s; both intervals exclude
  zero); secondary P2 and exploratory P3 were met. Acquisition comparability is
  undocumented and a shared timing contribution remains possible. The positive
  alignment finding is limited to the 33-62 degC development sessions; no
  temperature trend is claimed.

### What turned out to repeat earlier work

Three October "findings" repeated July 2026 work that had been removed from the
working tree (electrode 2 dominates fusion; temperature-only calibration is not
deployable; 43 degC is drift-heavy). This led to `research/INDEX.md`, the
`archive/*` tags, and the rule that no tried direction is ever deleted.

## 4. Proposed Corrections To The Thesis (`main`)

Applied to `main` on 2026-10-24 after the user delegated the decision
("do what you think is reasonable"): the four branches were merged and
`REPORT.md`, `README.md`, `ENGLISH_EXPLANATION.md`, and the reporting rules in
`AGENTS.md` were updated. The pre-integration thesis is kept at tag
`baseline-frozen-hgb-2026-10-20`.

1. Do not describe the dynamic gain as "learning hysteresis". Temporal inputs
   improved performance under the tested configuration; what they capture is
   unresolved.
2. The average advantage survived the reported calibration-bookkeeping
   sensitivity, while improvement in every direction did not (7/8 with
   measured band-mean strain levels, worst -5.5 %). Also report the
   window-position results with their caveat.
3. The constrained fusion selector gives electrode 1 zero weight, so the fused
   physics estimate is electrode 2; electrode 1 remains a separate HGB input
   (also found in July 2026).
4. Report the first-cycle observation and the 33-62 degC apparent alignment as
   descriptive observations with unresolved sensor, specimen, reference, and
   acquisition contributions, together with the unconfirmed 21 degC test.
5. Keep the scope boundary explicit: one specimen, one loading program,
   near-isothermal sessions, no untouched data left.

## 5. Where The Research Stands

- The average advantage of the frozen method held up under the calibration
  sensitivity analyses reported here; the all-direction claim did not, and its
  interpretation must become more modest.
- No breakthrough contribution was found; the strongest candidate (apparent
  alignment) failed its only possible confirmation.
- The corpus is exhausted for confirmatory purposes: no new measurements exist
  and dataset0 is consumed. Further hypothesis testing on dataset1-4 mainly adds
  selection risk.
- Recommended next step: apply the corrections above to the thesis and write
  it as method, robustness, what the temporal features capture, and limits.

Disagreement recorded: Codex (sy1) asked to delete the statement that the work
is acceptable for the thesis, because the repository sources do not establish
degree sufficiency. Claude kept it, attributed and conditional, because it is
Codex's own round-1 verdict, not a sourced fact.
