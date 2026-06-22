"""Run dataset construction pipeline.

Usage, select the dataset to process:
    python scripts/run_dataset_pipeline.py aerobic
    python scripts/run_dataset_pipeline.py anaerobic
"""

import argparse
import sys
from datetime import date
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset.loader import load_experiment
from src.dataset.qc import run_qc, report_blank_cv
from src.dataset.curves import extract_growth_curves

# Aerobic experiments
DEFAULTS_AEROBIC = dict(
    metadata=ROOT / "data" / "metadata" / "aerobic_metadata.csv",
    layout=ROOT / "data" / "metadata" / "aerobic_layout.csv",
    raw_dir=ROOT / "data" / "raw",
    output=ROOT / "data" / "processed" / "aerobic_tidy.parquet",
)

# Anaerobic experiments
DEFAULTS_ANAEROBIC = dict(
    metadata=ROOT / "data" / "metadata" / "anaerobic_metadata.csv",
    layout=ROOT / "data" / "metadata" / "anaerobic_layout.csv",
    raw_dir=ROOT / "data" / "raw",
    output=ROOT / "data" / "processed" / "anaerobic_tidy.parquet",
)

DEFAULT_PRESETS = {
    "aerobic": DEFAULTS_AEROBIC,
    "anaerobic": DEFAULTS_ANAEROBIC,
}


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Configure loguru for CLI output and optional file sink."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | {message}",
        colorize=sys.stderr.isatty(),
    )
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <7} | {message}",
            colorize=False,
            mode="w",
        )


def log_section(step: int, total: int, title: str) -> None:
    logger.info("─" * 60)
    logger.info("[{}/{}] {}", step, total, title)


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


def log_dataframe_summary(df, *, label: str = "dataframe") -> None:
    """Log shape, columns, and per-experiment counts for a tidy dataframe."""
    logger.info("{} shape: {} rows × {} cols", label, *df.shape)
    logger.debug("{} columns: {}", label, list(df.columns))

    exp_ids = df["exp_id"].unique().tolist()
    logger.info("experiments (n={}): {}", len(exp_ids), exp_ids)

    timepoints = df.groupby("exp_id")["timepoint"].nunique().to_dict()
    logger.info("timepoints per experiment: {}", timepoints)

    intervals, elapsed, units = summarize_time_axis(df)
    logger.info("time interval per experiment: {}", format_time_summary(intervals, units))
    logger.info("total elapsed per experiment: {}", format_time_summary(elapsed, units))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "preset",
        nargs="?",
        choices=DEFAULT_PRESETS.keys(),
        default="aerobic",
        help="Default dataset configuration to use.",
    )
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--layout", type=Path)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    args = parser.parse_args()

    defaults = DEFAULT_PRESETS[args.preset]
    args.metadata = args.metadata or defaults["metadata"]
    args.layout = args.layout or defaults["layout"]
    args.raw_dir = args.raw_dir or defaults["raw_dir"]
    args.output = args.output or defaults["output"]

    log_output = args.output.with_name(args.output.stem + "_log.log")
    setup_logging(args.log_level, log_file=log_output)

    logger.info("date: {}", date.today().isoformat())
    logger.info("starting dataset pipeline (preset={})", args.preset)
    logger.info("log file: {}", log_output)
    logger.debug("metadata: {}", args.metadata)
    logger.debug("layout:   {}", args.layout)
    logger.debug("raw_dir:  {}", args.raw_dir)
    logger.debug("output:   {}", args.output)

    total_steps = 4

    log_section(1, total_steps, "Loading experiments")
    df = load_experiment(args.metadata, args.layout, args.raw_dir)
    log_dataframe_summary(df, label="loaded dataframe")

    log_section(2, total_steps, "Running QC")
    qc_df = run_qc(df)
    report_blank_cv(qc_df)
    logger.info("QC output shape: {} rows × {} cols", *qc_df.shape)
    new_cols = [c for c in qc_df.columns if c not in df.columns]
    logger.info("new QC columns (n={}): {}", len(new_cols), new_cols)
    blanks = qc_df.groupby("exp_id")["is_blank"].sum().to_dict()
    logger.info("blanks per experiment: {}", blanks)

    log_section(3, total_steps, "Extracting growth curves")
    curves_df, long_df = extract_growth_curves(qc_df)
    logger.info("curves shape: {} rows × {} cols", *curves_df.shape)
    logger.debug("curves columns: {}", list(curves_df.columns))
    logger.info("long dataframe shape: {} rows × {} cols", *long_df.shape)

    logger.info("sample curve rows (first 3):\n{}", curves_df.head(3).to_string(index=False))

    log_section(4, total_steps, "Writing outputs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_parquet(args.output, index=False)
    logger.success(
        "saved long dataframe → {} ({:.1f} KB)",
        args.output,
        args.output.stat().st_size / 1024,
    )

    curves_output = args.output.with_name(args.output.stem + "_curves.csv")
    curves_df.to_csv(curves_output, index=False)
    logger.success(
        "saved curves → {} ({:.1f} KB)",
        curves_output,
        curves_output.stat().st_size / 1024,
    )

    logger.info("Pipeline complete")
    logger.success("saved log → {}", log_output)


if __name__ == "__main__":
    main()
