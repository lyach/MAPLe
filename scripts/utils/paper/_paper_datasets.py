"""Shared loader for the dataset variants.

Normalises all five datasets onto one schema -- ``strain``, ``medium``
(canonicalised), ``medium_raw``, ``temperature``, ``label``.

Atmosphere is not carried: BacDive has no atmosphere field, so it exists for
MAPLE-Lab only and cannot be a row in a table comparing all five.

To keep in mind when choosing ``LAB_CSV``: in
``align_str_med_temp.csv`` atmosphere is nearly collinear with temperature
(25 C all-aerobic, 29 C all-anaerobic). So a temperature flip measured on that 
file also absorbs the atmosphere flip.
See the change in temp-driven flip by replacing the path:
   - ``align_str_med_temp.csv`` = .427
   - ``align_str_med_temp_atm.csv`` = 0.095
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT, Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.bacdive.media import canonical_media_name

BACDIVE_DIR = ROOT / "data" / "processed" / "bacdive"
ALIGN_DIR = ROOT / "data" / "processed" / "align"

LAB_CSV = ALIGN_DIR / "align_str_med_temp.csv"

# (key, display label, filename, is_lab)
CORPUS_SPECS: tuple[tuple[str, str, Path, bool], ...] = (
    ("bd_temp", "BD-temp (all)", BACDIVE_DIR / "bd_temp.parquet", False),
    ("bd_temp_w_genome", "BD-temp (genome)", BACDIVE_DIR / "bd_temp_w_genome.parquet", False),
    ("bd_med", "BD-med (all)", BACDIVE_DIR / "bd_med.parquet", False),
    ("bd_med_w_genome", "BD-med (genome)", BACDIVE_DIR / "bd_med_w_genome.parquet", False),
    ("lab", "MAPLE-Lab", LAB_CSV, True),
)

TAXON_COLS = ("genus", "family", "order", "phylum")


@dataclass(frozen=True)
class Corpus:
    """One normalised corpus plus the factor axes it actually varies."""

    key: str
    label: str
    path: Path
    df: pd.DataFrame

    @property
    def axes(self) -> tuple[str, ...]:
        """Factor axes reported."""
        found = ["strain", "medium"]
        if self.df["temperature"].nunique(dropna=True) > 1:
            found.append("temperature")
        return tuple(found)

    @property
    def has_genome(self) -> bool:
        return "genome_accession" in self.df.columns or (
            "ncbi_assembly_accession" in self.df.columns
        )


def _canon_map(names: pd.Series) -> dict[str, str]:
    """Canonicalise once per distinct name, not once per row."""
    return {n: canonical_media_name(n) for n in names.astype(str).unique()}


def load_corpus(key: str, label: str, path: Path, is_lab: bool) -> Corpus:
    raw = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)

    out = pd.DataFrame(index=raw.index)
    out["strain"] = raw["strain_name"].astype(str).str.strip()
    out["medium_raw"] = raw["media_name"].astype(str).str.strip()

    # Prefer the media_canon column the BacDive build already writes;
    # (Lab has no such column).
    if "media_canon" in raw.columns:
        out["medium"] = raw["media_canon"].astype(str).str.strip()
    else:
        out["medium"] = out["medium_raw"].map(_canon_map(out["medium_raw"]))

    out["temperature"] = pd.to_numeric(raw.get("temperature"), errors="coerce")
    out["label"] = pd.to_numeric(raw["growth_class"], errors="coerce").astype("Int64")

    for col in TAXON_COLS:
        if col in raw.columns:
            out[col] = raw[col].astype(str).str.strip().replace({"": pd.NA, "nan": pd.NA})
    for col in ("genome_accession", "ncbi_assembly_accession", "media_canon_status"):
        if col in raw.columns:
            out[col] = raw[col].astype(str).str.strip()

    # A row with no label cannot contribute to any table.
    n_before = len(out)
    out = out[out["label"].notna()].copy()
    if len(out) != n_before:
        print(f"  note: {key}: dropped {n_before - len(out)} row(s) with no label",
              file=sys.stderr)
    return Corpus(key=key, label=label, path=path, df=out)


def load_all(
    only: list[str] | None = None, lab_csv: Path | None = None
) -> list[Corpus]:
    """Load every dataset that exists on disk.

    ``lab_csv`` overrides which MAPLE-Lab slice is used.

    A missing file is reported and skipped rather than fatal, so the tables can
    still be built when one lineage has not been rebuilt yet.
    """
    corpora: list[Corpus] = []
    for key, label, path, is_lab in CORPUS_SPECS:
        if only and key not in only:
            continue
        if is_lab and lab_csv is not None:
            path = lab_csv
        if not path.exists():
            print(f"  warning: skipping {key}: no such file {path}", file=sys.stderr)
            continue
        corpora.append(load_corpus(key, label, path, is_lab))
    return corpora


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def to_markdown(df: pd.DataFrame) -> str:
    """Pipe table with the first column left-aligned, the rest right."""
    cols = list(df.columns)
    align = [":---"] + ["---:"] * (len(cols) - 1)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(align) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |")
    return "\n".join(lines)


def to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    cols = list(df.columns)
    spec = "l" + "r" * (len(cols) - 1)

    def esc(v: object) -> str:
        s = "" if pd.isna(v) else str(v)
        for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                     ("_", r"\_"), ("#", r"\#"), ("≥", r"$\geq$"), ("×", r"$\times$")):
            s = s.replace(a, b)
        return s

    out = [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}",
           rf"\label{{{label}}}", rf"\begin{{tabular}}{{{spec}}}", r"\toprule",
           " & ".join(esc(c) for c in cols) + r" \\", r"\midrule"]
    for _, row in df.iterrows():
        out.append(" & ".join(esc(v) for v in row) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


def emit(df: pd.DataFrame, fmt: str, out_path: Path, caption: str, tex_label: str) -> None:
    """Write the table in the requested format(s) and echo markdown to stdout."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wrote = []
    if fmt in {"csv", "all"}:
        df.to_csv(out_path.with_suffix(".csv"), index=False)
        wrote.append(out_path.with_suffix(".csv"))
    if fmt in {"md", "all"}:
        out_path.with_suffix(".md").write_text(to_markdown(df) + "\n")
        wrote.append(out_path.with_suffix(".md"))
    if fmt in {"latex", "all"}:
        out_path.with_suffix(".tex").write_text(to_latex(df, caption, tex_label) + "\n")
        wrote.append(out_path.with_suffix(".tex"))
    print()
    print(to_markdown(df))
    print()
    for path in wrote:
        print(f"wrote {path}")
