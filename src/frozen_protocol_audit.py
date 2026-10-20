"""Audit every major design choice under the final frozen LOTO protocol.

All analyses use the existing four-temperature corpus. They are retrospective:
the model family and fixed hyperparameters were developed after inspecting this
corpus, so these checks strengthen internal validity but are not an untouched
external confirmation.

Entry point and runner: computes everything, writes results/ (or --out DIR) and
a manifest. The method itself lives in src/strain/ (see AGENTS.md).
"""

import argparse
import hashlib
import json
import platform
import sys
from importlib import metadata
from pathlib import Path

import matplotlib

from strain.audits import assert_audit_passes, causality_and_label_audit
from strain.metrics import summarize
from strain.protocol import run_variants, window_sensitivity
from strain.reporting import (
    error_regions,
    plot_protocol_results,
    position_summary,
    temperature_summary,
)
from strain.spec import DATA_DIR, DATASETS, RESULTS_DIR, VARIANTS, load_raw

# Exact output inventory (paths relative to the output directory). The manifest
# is written last and never hashes itself.
TABLES = (
    "causality_label_access_audit.csv",
    "frozen_protocol_ablations_per_fold.csv",
    "frozen_protocol_ablations_summary.csv",
    "frozen_protocol_by_temperature.csv",
    "frozen_protocol_edge_vs_interior.csv",
    "frozen_protocol_error_regions.csv",
    "frozen_window_sensitivity_per_fold.csv",
    "frozen_window_sensitivity_summary.csv",
)
FIGURES = ("frozen_best_worst_predictions.png", "frozen_protocol_audit.png")
OUTPUTS = tuple(f"tables/{name}" for name in TABLES) + tuple(f"figures/{name}" for name in FIGURES)
MANIFEST = "manifest.json"
PACKAGES = ("matplotlib", "numpy", "pandas", "scikit-learn", "scipy")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_manifest(out_dir):
    manifest = {
        "note": "Output hashes are the regression reference for tools/check_results.py. "
        "PNG bytes additionally depend on platform and fonts.",
        "python": platform.python_version(),
        "platform": f"{platform.system()}-{platform.machine()}",
        "matplotlib_backend": matplotlib.get_backend(),
        "packages": {name: metadata.version(name) for name in PACKAGES},
        "raw_data_sha256": {f"dataset{d}.csv": sha256(DATA_DIR / f"dataset{d}.csv") for d in DATASETS},
        "outputs_sha256": {name: sha256(out_dir / name) for name in OUTPUTS},
    }
    (out_dir / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=RESULTS_DIR, help="output directory (default: results/)")
    args = parser.parse_args(argv)
    out_tab, out_fig = args.out / "tables", args.out / "figures"
    out_tab.mkdir(parents=True, exist_ok=True)
    out_fig.mkdir(parents=True, exist_ok=True)
    raw = {dataset: load_raw(dataset) for dataset in DATASETS}

    per_fold, final_artifacts = run_variants(raw, VARIANTS)
    summary = summarize(per_fold)
    per_fold.to_csv(out_tab / "frozen_protocol_ablations_per_fold.csv", index=False)
    summary.to_csv(out_tab / "frozen_protocol_ablations_summary.csv", index=False)
    temperature_summary(per_fold).to_csv(out_tab / "frozen_protocol_by_temperature.csv", index=False)
    positions = position_summary(per_fold)
    positions.to_csv(out_tab / "frozen_protocol_edge_vs_interior.csv", index=False)

    audit = causality_and_label_audit(final_artifacts)
    audit.to_csv(out_tab / "causality_label_access_audit.csv", index=False)
    assert_audit_passes(audit)

    regions = error_regions(final_artifacts)
    regions.to_csv(out_tab / "frozen_protocol_error_regions.csv", index=False)

    window_per_fold, window_summary = window_sensitivity(raw)
    window_per_fold.to_csv(out_tab / "frozen_window_sensitivity_per_fold.csv", index=False)
    window_summary.to_csv(out_tab / "frozen_window_sensitivity_summary.csv", index=False)

    plot_protocol_results(per_fold, final_artifacts, out_fig)
    write_manifest(args.out)

    print("\nFrozen-protocol ablations:")
    print(summary.round(4).to_string(index=False))
    print("\nCausality and label-access audit:")
    print(audit.to_string(index=False))
    print("\nWindow sensitivity:")
    print(window_summary.round(4).to_string(index=False))
    print("\nEdge versus interior held-out temperatures:")
    print(positions.round(4).to_string(index=False))
    print("\nError regions, averaged across directions:")
    print(regions.groupby("region")[["model_rmse", "physics_rmse", "pct_improvement"]].mean().round(4))


if __name__ == "__main__":
    sys.exit(main())
