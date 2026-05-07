"""Run dataset construction pipeline.

Usage:
    python scripts/run_dataset_pipeline.py <metadata_csv> <layout_csv> <raw_dir> [output_parquet]
    python scripts/run_dataset_pipeline.py  # uses defaults in this script
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset.loader import load_experiment
from src.dataset.qc import run_qc

DEFAULTS = dict(
    metadata=ROOT / "data" / "metadata" / "20260325_aerobic_metadata.csv",
    layout=ROOT / "data" / "metadata" / "20260325_aerobic_layout.csv",
    raw_dir=ROOT / "data" / "raw",
    output=ROOT / "data" / "processed" / "20260325_aerobic_tidy.parquet",
)


def summarize_time_axis(df):
    """Return per experiment time interval, elapsed time, and units."""
    intervals = {}
    elapsed = {}
    units = {}

    for exp_id, group in df.groupby("exp_id", sort=False):
        time_values = sorted(group["time_value"].dropna().unique())
        time_unit = group["time_unit"].dropna().iloc[0] if not group.empty else ""
        diffs = sorted(
            {round(float(b - a), 10) for a, b in zip(time_values, time_values[1:])}
        )

        intervals[exp_id] = diffs[0] if len(diffs) == 1 else diffs
        elapsed[exp_id] = (
            round(float(time_values[-1] - time_values[0]), 10) if time_values else 0.0
        )
        units[exp_id] = time_unit

    return intervals, elapsed, units


def format_time_summary(values, units):
    """Append the time unit to scalar or list summaries."""
    formatted = {}
    for exp_id, value in values.items():
        unit = units.get(exp_id, "")
        if isinstance(value, list):
            formatted[exp_id] = [f"{v:g} {unit}".strip() for v in value]
        else:
            formatted[exp_id] = f"{value:g} {unit}".strip()
    return formatted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULTS["metadata"])
    parser.add_argument("--layout", type=Path, default=DEFAULTS["layout"])
    parser.add_argument("--raw-dir", type=Path, default=DEFAULTS["raw_dir"])
    parser.add_argument("--output", type=Path, default=DEFAULTS["output"])
    args = parser.parse_args()

    print("--- Loading experiments ---")
    df = load_experiment(args.metadata, args.layout, args.raw_dir)
    print(f"df shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Experiments: {df['exp_id'].unique().tolist()}")
    print(f"Timepoints per exp: {df.groupby('exp_id')['timepoint'].nunique().to_dict()}")
    intervals, elapsed, units = summarize_time_axis(df)
    print(f"Time interval per experiment: {format_time_summary(intervals, units)}")
    print(f"Total time elapsed per experiment: {format_time_summary(elapsed, units)}")
    print()

    print("--- Running QC ---")
    qc_df = run_qc(df)
    print(f"New df shape: {qc_df.shape}")
    print(f"New columns after QC: {[c for c in qc_df.columns if c not in df.columns]}")
    print(f"Blanks count: {qc_df['is_blank'].sum()}")
    print()

    print("--- Sample rows ---")
    print(qc_df.head(3).to_string(index=False))
    print()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    qc_df.to_parquet(args.output, index=False)
    print(f"Saved: {args.output}  ({args.output.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
