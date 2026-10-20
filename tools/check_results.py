"""Check-only regression gate. Never writes into results/ and never regenerates its reference.

    python3 tools/check_results.py            # all three gates
    python3 tools/check_results.py --docs     # documentation gate only (fast)

Gate 1, reproduction: rerun the pipeline into an empty temporary directory, require
the exact output inventory, and compare every output hash with results/manifest.json.
Gate 2, audit sensitivity: inject known leaks (future-dependent feature, calibration
tail access, test-label access, unfrozen model) and require the audit to catch each
one. A byte-identical rerun cannot detect a disabled detector; this gate can.
Gate 3, documentation: every registered numeric claim must occur exactly once at its
locator and equal the named table value rounded to the decimals written in the text.
"""

import argparse
import copy
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from frozen_protocol_audit import MANIFEST, OUTPUTS  # noqa: E402

TABLES = ROOT / "results" / "tables"
FINAL = "final_physics_causal_no_T_direct_HGB"
NUM = r"([-+]?\d+(?:\.\d+)?)"


# ---------------------------------------------------------------- gate 1
def gate_reproduction():
    reference = json.loads((ROOT / "results" / MANIFEST).read_text())["outputs_sha256"]
    failures = []
    for name in OUTPUTS:  # the stored outputs must match the manifest too
        stored = ROOT / "results" / name
        if not stored.is_file() or hashlib.sha256(stored.read_bytes()).hexdigest() != reference.get(name):
            failures.append(f"stored output missing or differs from manifest: results/{name}")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out"
        subprocess.run(
            [sys.executable, str(ROOT / "src" / "frozen_protocol_audit.py"), "--out", str(out)],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        produced = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
        expected = sorted([*OUTPUTS, MANIFEST])
        if produced != expected:
            failures.append(f"inventory {produced} != {expected}")
        for name in OUTPUTS:
            digest = hashlib.sha256((out / name).read_bytes()).hexdigest()
            if digest != reference.get(name):
                failures.append(f"hash mismatch: {name}")
    return failures


# ---------------------------------------------------------------- gate 2
def gate_audit_sensitivity():
    import strain.calibration as calibration
    import strain.features as features
    from strain.audits import causality_and_label_audit
    from strain.protocol import evaluate_fold, fit_source_model
    from strain.spec import VARIANTS, load_raw

    raw = {d: load_raw(d) for d in (1, 2, 3, 4)}
    variant = next(v for v in VARIANTS if v["name"] == FINAL)
    held_out = 2

    def artifacts(mutate_model=False):
        model = fit_source_model(raw, held_out, variant)
        a, b = sorted(raw[held_out].Experiment.unique())
        result = {}
        for direction, cal, test in (("A_to_B", a, b), ("B_to_A", b, a)):
            fold_model = copy.deepcopy(model) if (mutate_model and direction == "B_to_A") else model
            result[(held_out, direction)] = evaluate_fold(raw, held_out, direction, cal, test, variant, fold_model)[1]
        return result

    def audit_flags(make_artifacts):
        try:
            audit = causality_and_label_audit(make_artifacts())
        except (KeyError, ValueError, IndexError):
            return None  # the audit crashed on the forbidden access: also a detection
        return audit

    def detected(column, make_artifacts, patch, restore):
        patch()
        try:
            audit = audit_flags(make_artifacts)
        finally:
            restore()
        return audit is None or not audit[column].all()

    failures = []
    clean = causality_and_label_audit(artifacts())
    if not clean.filter(like="_pass").all().all():
        failures.append("clean run does not pass its own audit")

    original_derivative = features.causal_derivative
    original_prefix = calibration.calibration_prefix
    original_normalize = calibration.normalize

    def leaky_normalize(df, fit, weight1):
        z1, z2, zf = original_normalize(df, fit, weight1)
        return z1, z2, zf + 1e-6 * df["Strain"].to_numpy()

    from sklearn.ensemble import HistGradientBoostingRegressor

    original_predict = HistGradientBoostingRegressor.predict

    calls = {"n": 0}

    def mutating_predict(self, X):
        # Mutate state only from the second fold's inference on, so both directions still share
        # one pre-inference fingerprint and only the before/after comparison can catch it.
        calls["n"] += 1
        if calls["n"] >= 2:
            self.leak_counter_ = getattr(self, "leak_counter_", 0) + 1
        return original_predict(self, X)

    controls = [
        ("causality_pass", lambda: setattr(features, "causal_derivative", features.centered_derivative),
         lambda: setattr(features, "causal_derivative", original_derivative), artifacts),
        ("calibration_scope_pass", lambda: setattr(calibration, "calibration_prefix", lambda df: df),
         lambda: setattr(calibration, "calibration_prefix", original_prefix), artifacts),
        ("test_label_access_pass", lambda: setattr(calibration, "normalize", leaky_normalize),
         lambda: setattr(calibration, "normalize", original_normalize), artifacts),
        # one object per direction (identity check) ...
        ("frozen_shared_source_model_pass", lambda: None, lambda: None, lambda: artifacts(mutate_model=True)),
        # ... and a model mutated during inference (fingerprint before/after check)
        ("frozen_shared_source_model_pass", lambda: setattr(HistGradientBoostingRegressor, "predict", mutating_predict),
         lambda: setattr(HistGradientBoostingRegressor, "predict", original_predict), artifacts),
    ]
    for column, patch, restore, make in controls:
        if not detected(column, make, patch, restore):
            failures.append(f"injected leak NOT detected by {column}")
    return failures


# ---------------------------------------------------------------- gate 3
def _summary(name):
    return pd.read_csv(TABLES / name).set_index("variant")


def _weakest_fold():
    rows = pd.read_csv(TABLES / "frozen_protocol_ablations_per_fold.csv")
    rows = rows[rows["variant"] == FINAL]
    return rows.loc[rows["r2"].idxmin()]


ABL = "frozen_protocol_ablations_summary.csv"
SENS = "REPORT.md#Calibration Sensitivity"  # locator scoped to one section
WIN = "frozen_window_sensitivity_summary.csv"


def _row(table, variant, *columns):
    return lambda: [_summary(table).loc[variant, c] for c in columns]


PROBES = ROOT / "research" / "probes" / "out"


def _probe(name):
    return pd.read_csv(PROBES / name)


def _bookkeeping():
    rows = _probe("calibration_bookkeeping.csv")
    measured = rows[rows["convention"] == "measured_band_means"]["pct_improvement"]
    return [measured.mean(), measured.min(), int((measured > 0).sum())]


def _weak_fold_windows(name, better):
    """[positions where `better` holds in the weakest direction, positions in it].

    Also enforces the surrounding prose: exactly 8 directions, and the other 7 win at every
    position. If not, returns impossible values so the claim fails.
    """
    rows = _probe(name)
    shares = rows.groupby(["temperature", "direction"]).apply(lambda d: better(d).mean())
    if len(shares) != 8 or (shares == 1.0).sum() != 7:
        return [-1, -1]
    weak = rows.set_index(["temperature", "direction"]).loc[shares.idxmin()]
    return [int(better(weak).sum()), len(weak)]


def _first_cycle():
    rows = _probe("first_cycle_estimators.csv")
    bins, timed = rows["cycle_bins_first_vs_later_pct"], rows["ols_time_first_vs_later_pct"]
    return [int((bins > 0).sum()), len(rows), bins.min(), bins.max(),
            int(rows["cycle_bins_faster_initial_decline"].astype(bool).sum()), len(rows),
            int((timed > 0).sum()), len(rows)]


def _preconditioning():
    rows = _probe("preconditioning_rmse.csv")
    hgb = rows.pivot_table(index=["temperature", "direction"], columns="protocol", values="hgb_rmse")
    ratio = hgb["P1"] / hgb["P0"]
    return [hgb["P1"].mean(), hgb["P0"].mean(), 100 * (1 - ratio).mean(), int((ratio < 1).sum()),
            100 * (ratio.max() - 1)]


def _alignment():
    v1 = _probe("reference_lag_per_electrode_session.csv")
    v2 = _probe("reference_lag_v2_per_electrode_session.csv")
    valid = lambda f: int((f.tau_s.notna() & f.ci_low.notna()).sum())
    e10 = v2[(v2.temperature == 62) & (v2.experiment == 10)].sort_values("electrode").tau_s.tolist()
    return [valid(v1), len(v1), valid(v2), *e10]


def _gap_contraction():
    gaps = _probe("dynamic_gain_attribution_gaps.csv")
    out = []
    for arm in ("A1_oracle", "A2_source_constant", "A3_oracle_fixed_normalization"):
        out.append(1 - gaps[f"G_{arm}"].mean() / gaps["G_A0"].mean())
    out.append(1 - gaps["G_A4_crossfit"].mean() / gaps["G_A0_crossfit_subsets"].mean())
    out += [int((gaps["G_A1_oracle"] < gaps["G_A0"]).sum()), int((gaps["G_A2_source_constant"] < gaps["G_A0"]).sum()),
            int((gaps["G_A3_oracle_fixed_normalization"] < gaps["G_A0"]).sum()),
            int((gaps["G_A4_crossfit"] < gaps["G_A0_crossfit_subsets"]).sum()),
            gaps["G_A0"].mean(), gaps["G_A1_oracle"].mean()]
    return out


def _median_rmse_across_positions():
    groups = _probe("calibration_shift.csv").groupby(["temperature", "direction"])
    return groups[["physics_rmse", "static_rmse", "dynamic_rmse"]].median().mean().tolist()


def _ablation_row(label, variant):
    """REPORT 'Critical Ablations' table: mean improvement, worst direction."""
    return ("REPORT.md", rf"\| {label} \| \**{NUM}%\** \| \**{NUM}%\** \|",
            _row(ABL, variant, "mean_pct_improvement", "worst_fold_pct_improvement"))


CLAIMS = [
    # REPORT.md main table and its prose
    ("REPORT.md", rf"\| Physics only, matched 125 s calibration \| {NUM}% \|", _row(ABL, FINAL, "mean_physics_rmse")),
    ("REPORT.md", rf"\| \*\*Frozen HGB: physics-normalized, causal, no `T`\*\* \| \*\*{NUM}%\*\* \| \*\*{NUM}\*\* \| "
                  rf"\*\*{NUM}%\*\* \| \*\*{NUM}%\*\* \|",
     _row(ABL, FINAL, "mean_rmse", "mean_r2", "mean_pct_improvement", "worst_fold_pct_improvement")),
    ("REPORT.md", rf"\| Frozen raw-resistance HGB, no `T` \| {NUM}% \| {NUM} \| {NUM}% \| {NUM}% \|",
     _row(ABL, "raw_no_T_direct_HGB", "mean_rmse", "mean_r2", "mean_pct_improvement", "worst_fold_pct_improvement")),
    ("REPORT.md", rf"\| Frozen raw HGB \+ target low-strain offset, no `T` \| {NUM}% \| {NUM} \| {NUM}% \| {NUM}% \|",
     _row(ABL, "raw_low_strain_offset_no_T_direct_HGB", "mean_rmse", "mean_r2", "mean_pct_improvement",
          "worst_fold_pct_improvement")),
    ("REPORT.md", rf"The reported {NUM}% is the arithmetic mean of the eight per-direction",
     _row(ABL, FINAL, "mean_pct_improvement")),
    ("REPORT.md", rf"`1 - {NUM} / {NUM} = {NUM}%`",
     _row(ABL, FINAL, "mean_rmse", "mean_physics_rmse", "ratio_of_mean_rmse_reduction")),
    ("REPORT.md", rf"its ratio-of-means reduction is `{NUM}%`",
     _row(ABL, "raw_low_strain_offset_no_T_direct_HGB", "ratio_of_mean_rmse_reduction")),
    ("REPORT.md", rf"weakest-prediction fold is 43 degC A-to-B, with\s+`R2 = {NUM}` and `RMSE = {NUM}%`",
     lambda: [_weakest_fold()["r2"], _weakest_fold()["rmse"]]),
    _ablation_row(r"\*\*Causal, no-`T`, direct HGB \(final\)\*\*", FINAL),
    _ablation_row(r"Static normalized HGB \(no dynamic features\)", "physics_static_no_T_direct_HGB"),
    _ablation_row("Residual-target HGB", "physics_causal_no_T_residual_HGB"),
    _ablation_row("Centered derivative HGB", "physics_centered_no_T_direct_HGB"),
    _ablation_row("Causal HGB with `T`", "physics_causal_with_T_direct_HGB"),
    _ablation_row("Ridge on the same features", "physics_causal_no_T_Ridge"),
    _ablation_row("Causal smoothing only", "physics_causal_smoothing_only"),
    _ablation_row("Raw HGB with `T`", "raw_with_T_direct_HGB"),
    _ablation_row("Raw HGB without `T`", "raw_no_T_direct_HGB"),
    _ablation_row(r"Raw HGB \+ low-strain offset with `T`", "raw_low_strain_offset_with_T_direct_HGB"),
    _ablation_row(r"Raw HGB \+ low-strain offset without `T`", "raw_low_strain_offset_no_T_direct_HGB"),
    ("REPORT.md", rf"\| Local features only \| {NUM}% \| {NUM}% \|",
     _row(WIN, "history_local_only", "mean_pct_improvement", "worst_fold_pct_improvement")),
    ("REPORT.md", rf"\| Up to 21 samples \| {NUM}% \| {NUM}% \|",
     _row(WIN, "history_to_21", "mean_pct_improvement", "worst_fold_pct_improvement")),
    ("REPORT.md", rf"\| Up to 51 samples \| {NUM}% \| {NUM}% \|",
     _row(WIN, "history_to_51", "mean_pct_improvement", "worst_fold_pct_improvement")),
    ("REPORT.md", rf"\| Up to 101 samples \| {NUM}% \| {NUM}% \|",
     _row(WIN, "history_to_101", "mean_pct_improvement", "worst_fold_pct_improvement")),
    ("REPORT.md", rf"\| \*\*Up to 201 samples\*\* \| \*\*{NUM}%\*\* \| \*\*{NUM}%\*\* \|",
     _row(WIN, "history_to_201", "mean_pct_improvement", "worst_fold_pct_improvement")),
    # REPORT.md calibration sensitivity (research/probes/out)
    (SENS, rf"measured\s+mean strain instead gives a mean per-direction improvement of {NUM}%, a worst\s+"
                  rf"direction of {NUM}%, and {NUM}/8 improved directions", _bookkeeping),
    (SENS, rf"the frozen dynamic HGB outperformed matched physics across all\s+sampled calibration positions in seven directions and {NUM}/{NUM} positions in the\s+eighth",
     lambda: _weak_fold_windows("window_distribution.csv", lambda d: d.pct_improvement > 0)),
    (SENS, rf"the\s+dynamic HGB had lower RMSE than the static normalized HGB at every position in\s+seven directions and {NUM}/{NUM} positions in the eighth",
     lambda: _weak_fold_windows("calibration_shift.csv", lambda d: d.dynamic_rmse < d.static_rmse)),
    (SENS, rf"the median RMSE across positions was {NUM}% for physics, {NUM}% for static\s+HGB, "
                  rf"and {NUM}% for dynamic HGB", _median_rmse_across_positions),
    (SENS, rf"the first cycle was higher in {NUM}/{NUM} electrode-session combinations, by {NUM}% to\s+"
           rf"{NUM}% relative to the median of cycles 3–7\. A faster initial decline held in\s+{NUM}/{NUM}\. "
           rf"Adding a linear time term to window regressions reduces the first-higher\s+agreement to {NUM}/{NUM}",
     _first_cycle),
    (SENS, rf"mean HGB RMSE was {NUM}% versus {NUM}% for the\s+repository convention, a {NUM}% mean per-direction "
           rf"improvement, with {NUM}/8\s+directions improved and the worst direction {NUM}% worse",
     _preconditioning),
    ("REPORT.md#Reference Alignment (direction branch)",
     rf"returned insufficient coverage: {NUM}/{NUM} electrode-sessions were valid.*?"
     rf"\(v2, ±15 s grid\)\s+resolved all {NUM} and.*?Experiment 10's offsets were {NUM} and {NUM} s",
     _alignment),
    ("REPORT.md#Static-Dynamic Gap Under Label Re-alignment (direction branch)",
     rf"gap was {NUM} for oracle re-alignment,\s+{NUM} for source-constant re-alignment, {NUM} with original "
     rf"normalization\s+retained, and {NUM} for cross-fitted offsets; gaps decreased in {NUM}/8, {NUM}/8, "
     rf"{NUM}/8,\s+and {NUM}/8 directions.*?The oracle mean gap decreased from {NUM} to {NUM}",
     _gap_contraction),
    # README.md
    ("README.md", rf"\| Frozen causal HGB \+ 125 s calibration \| \*\*{NUM}% strain\*\* \| \*\*{NUM}\*\* \| "
                  rf"\*\*{NUM}%\*\* \| \*\*{NUM}%\*\* \|",
     _row(ABL, FINAL, "mean_rmse", "mean_r2", "mean_pct_improvement", "worst_fold_pct_improvement")),
    ("README.md", rf"\| Frozen raw-resistance HGB, no `T` \| {NUM}% strain \| {NUM} \| {NUM}% \| {NUM}% \|",
     _row(ABL, "raw_no_T_direct_HGB", "mean_rmse", "mean_r2", "mean_pct_improvement", "worst_fold_pct_improvement")),
    ("README.md", rf"\| Raw HGB \+ target low-strain offset, no `T` \| {NUM}% strain \| {NUM} \| {NUM}% \| {NUM}% \|",
     _row(ABL, "raw_low_strain_offset_no_T_direct_HGB", "mean_rmse", "mean_r2", "mean_pct_improvement",
          "worst_fold_pct_improvement")),
    ("README.md", rf"A strictly static normalized HGB improves {NUM}% on average but degrades its\s+"
                  rf"worst direction by {NUM}%; causal dynamic features raise this to {NUM}% with a\s+{NUM}% worst",
     lambda: [*_row(ABL, "physics_static_no_T_direct_HGB", "mean_pct_improvement")(),
              -_summary(ABL).loc["physics_static_no_T_direct_HGB", "worst_fold_pct_improvement"],
              *_row(ABL, FINAL, "mean_pct_improvement", "worst_fold_pct_improvement")()]),
    ("README.md", rf"Smoothing alone makes the result worse \({NUM}% mean\)",
     _row(ABL, "physics_causal_smoothing_only", "mean_pct_improvement")),
    ("README.md", rf"Ridge fails its worst direction \({NUM}%\)",
     _row(ABL, "physics_causal_no_T_Ridge", "worst_fold_pct_improvement")),
    # REPORT.md prose
    ("REPORT.md", rf"adding `T`\s+reduces mean improvement and causes one direction to degrade by {NUM}%",
     lambda: [-_summary(ABL).loc["physics_causal_with_T_direct_HGB", "worst_fold_pct_improvement"]]),
    ("REPORT.md", rf"raises the mean per-direction improvement from\s+{NUM}% to {NUM}% and the worst direction "
                  rf"from {NUM}% to {NUM}%",
     lambda: [*_row(ABL, "physics_static_no_T_direct_HGB", "mean_pct_improvement")(),
              *_row(ABL, FINAL, "mean_pct_improvement")(),
              *_row(ABL, "physics_static_no_T_direct_HGB", "worst_fold_pct_improvement")(),
              *_row(ABL, FINAL, "worst_fold_pct_improvement")()]),
    # ENGLISH_EXPLANATION.md
    ("ENGLISH_EXPLANATION.md", rf"helps on average by only\s+{NUM}% and makes its worst test {NUM}% worse\. "
                               rf"Adding causal signal history raises\s+the mean improvement to {NUM}% and makes even "
                               rf"the weakest direction {NUM}%",
     lambda: [*_row(ABL, "physics_static_no_T_direct_HGB", "mean_pct_improvement")(),
              -_summary(ABL).loc["physics_static_no_T_direct_HGB", "worst_fold_pct_improvement"],
              *_row(ABL, FINAL, "mean_pct_improvement", "worst_fold_pct_improvement")()]),
    ("ENGLISH_EXPLANATION.md", rf"Linear Ridge regression fails badly in its worst test \({NUM}%\)",
     _row(ABL, "physics_causal_no_T_Ridge", "worst_fold_pct_improvement")),
    ("ENGLISH_EXPLANATION.md", rf"Adding temperature makes one unseen-temperature direction {NUM}% worse",
     lambda: [-_summary(ABL).loc["physics_causal_with_T_direct_HGB", "worst_fold_pct_improvement"]]),
    ("ENGLISH_EXPLANATION.md", rf"fails across temperatures: mean `R2 = {NUM}`",
     _row(ABL, "raw_with_T_direct_HGB", "mean_r2")),
    ("ENGLISH_EXPLANATION.md", rf"Removing `T` does not rescue raw HGB: mean `R2 = {NUM}`",
     _row(ABL, "raw_no_T_direct_HGB", "mean_r2")),
    ("ENGLISH_EXPLANATION.md", rf"`T`, is still insufficient: mean `R2 = {NUM}`",
     _row(ABL, "raw_low_strain_offset_no_T_direct_HGB", "mean_r2")),
    ("ENGLISH_EXPLANATION.md", rf"The physically normalized frozen model reaches mean `R2 = {NUM}`",
     _row(ABL, FINAL, "mean_r2")),
    ("ENGLISH_EXPLANATION.md", rf"Mean RMSE: +{NUM}% strain", _row(ABL, FINAL, "mean_rmse")),
    ("ENGLISH_EXPLANATION.md", rf"Mean R2: +{NUM}\n", _row(ABL, FINAL, "mean_r2")),
    ("ENGLISH_EXPLANATION.md", rf"Improvement over physics: +{NUM}% on average", _row(ABL, FINAL, "mean_pct_improvement")),
    ("ENGLISH_EXPLANATION.md", rf"Smallest improvement: +{NUM}%", _row(ABL, FINAL, "worst_fold_pct_improvement")),
    ("ENGLISH_EXPLANATION.md", rf"The weakest test is 43 degC A-to-B, with `R2 = {NUM}` and `RMSE = {NUM}%`",
     lambda: [_weakest_fold()["r2"], _weakest_fold()["rmse"]]),
]


def _matches(text, value, expected):
    decimals = len(value.split(".")[1]) if "." in value else 0
    return float(value) == round(float(expected), decimals)


def gate_docs():
    failures = []
    for doc, pattern, source in CLAIMS:
        path, _, section = doc.partition("#")
        text = (ROOT / path).read_text()
        if section:
            heading = f"## {section}\n"
            if text.count(heading) != 1:
                failures.append(f"{doc}: section heading found {text.count(heading)} times")
                continue
            text = text.split(heading, 1)[1].split("\n## ", 1)[0]
        hits = re.findall(pattern, text, flags=re.S)
        if len(hits) != 1:
            failures.append(f"{doc}: locator matched {len(hits)} times (need exactly 1): {pattern}")
            continue
        values = hits[0] if isinstance(hits[0], tuple) else (hits[0],)
        for value, expected in zip(values, source(), strict=True):
            if not _matches(doc, value, expected):
                failures.append(f"{doc}: wrote {value}, table says {expected:.6g}: {pattern}")
    return failures


def main():
    parser = argparse.ArgumentParser(description="Check-only regression gate.")
    parser.add_argument("--docs", action="store_true", help="run only the documentation gate")
    args = parser.parse_args()
    gates = [("documentation", gate_docs)]
    if not args.docs:
        gates = [("reproduction", gate_reproduction), ("audit sensitivity", gate_audit_sensitivity), *gates]
    failed = False
    for name, gate in gates:
        failures = gate()
        print(f"[{'FAIL' if failures else 'PASS'}] {name}")
        for failure in failures:
            print(f"    {failure}")
        failed |= bool(failures)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
