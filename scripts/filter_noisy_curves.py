"""
filter_noisy_curves.py
======================
Remove manually flagged noisy curves from a tidy_curves CSV.

Each flag in the config must specify all four fields:
    exp_id, date, strain, media

A row is dropped only if it matches ALL four fields exactly.
Fields in the CSV use different column names than the config keys —
the mapping is handled internally:

    config key  →  CSV column
    ----------     ----------
    exp_id      →  exp_id
    date        →  date        (stored as int, e.g. 20260417)
    strain      →  strain_name
    media       →  media_name

Usage:
    python scripts/filter_noisy_curves.py config/dataset/filter_aerobic.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml


# Maps the human-readable YAML key to the CSV column name.
_FIELD_MAP: dict[str, str] = {
    "exp_id": "exp_id",
    "date":   "date",
    "strain": "strain_name",
    "media":  "media_name",
}

_REQUIRED_KEYS = set(_FIELD_MAP.keys())


def load_config(config_path: Path) -> dict:
    """Load and validate the YAML config file."""
    if not config_path.exists():
        sys.exit(f"Error: config file not found: {config_path}")

    with config_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for key in ("input", "output", "flags"):
        if key not in cfg:
            sys.exit(f"Error: config is missing required key '{key}'")

    if not isinstance(cfg["flags"], list) or len(cfg["flags"]) == 0:
        sys.exit("Error: 'flags' must be a non-empty list of flag entries")

    for i, flag in enumerate(cfg["flags"]):
        missing = _REQUIRED_KEYS - set(flag.keys())
        if missing:
            sys.exit(
                f"Error: flag entry {i + 1} is missing required field(s): "
                f"{sorted(missing)}. All four fields (exp_id, date, strain, "
                f"media) must be specified."
            )

    return cfg


def build_drop_mask(df: pd.DataFrame, flags: list[dict]) -> pd.Series:
    """
    Build a boolean mask that is True for every row that matches
    at least one flag entry. A row matches a flag only when ALL
    four fields match exactly.

    The 'date' column is stored as int in the CSV, so the YAML
    value is cast to int before comparison.
    """
    # Start with nothing flagged.
    mask = pd.Series(False, index=df.index)

    for i, flag in enumerate(flags, start=1):
        flag_mask = pd.Series(True, index=df.index)

        for yaml_key, col in _FIELD_MAP.items():
            value = flag[yaml_key]

            # Cast date to int to match the CSV storage format.
            if col == "date":
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    sys.exit(
                        f"Error: flag entry {i} has a date value "
                        f"'{flag[yaml_key]}' that cannot be converted to "
                        f"an integer (expected format: YYYYMMDD)."
                    )

            flag_mask &= df[col] == value

        n_matched = flag_mask.sum()

        if n_matched == 0:
            print(
                f"  Warning: flag {i} matched no rows "
                f"(exp_id={flag['exp_id']!r}, date={flag['date']!r}, "
                f"strain={flag['strain']!r}, media={flag['media']!r})"
            )
        else:
            print(
                f"  Flag {i}: {n_matched} row(s) matched — "
                f"exp_id={flag['exp_id']!r}, date={flag['date']!r}, "
                f"strain={flag['strain']!r}, media={flag['media']!r}"
            )

        mask |= flag_mask

    return mask


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter noisy curves from a tidy_curves CSV."
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to the YAML config file "
             "(e.g. config/dataset/filter_aerobic.yaml)",
    )
    args = parser.parse_args()

    # ── 1. Load config ────────────────────────────────────────────────────
    cfg = load_config(args.config)

    input_path  = Path(cfg["input"])
    output_path = Path(cfg["output"])
    flags       = cfg["flags"]

    if not input_path.exists():
        sys.exit(f"Error: input file not found: {input_path}")

    # ── 2. Load data ──────────────────────────────────────────────────────
    df = pd.read_csv(input_path)

    initial_count = len(df)
    print(f"\nInput : {input_path}")
    print(f"Rows before filtering: {initial_count:,}")

    # ── 3. Build drop mask ────────────────────────────────────────────────
    print(f"\nApplying {len(flags)} flag(s):")
    drop_mask = build_drop_mask(df, flags)

    # ── 4. Filter ─────────────────────────────────────────────────────────
    df_filtered = df[~drop_mask].copy()

    final_count   = len(df_filtered)
    removed_count = initial_count - final_count

    print(f"\nRows after filtering : {final_count:,}")
    print(f"Rows removed         : {removed_count:,}")

    # ── 5. Write output ───────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_filtered.to_csv(output_path, index=False)

    print(f"\nOutput: {output_path}")
    print("Done.")


if __name__ == "__main__":
    main()
