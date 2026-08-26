#!/usr/bin/env python3
"""Compare the internal plate-reader benchmark against the BacDive 52k table.

The two datasets are meant to play different roles in the paper: BacDive
supplies breadth (11.9k genomes, 2.4k media) with a label whose negatives come
from out-of-range *temperature*, and the internal set supplies a controlled
factorial (25 strains x 11 media x 2 atmospheres x 3 temperatures) whose
negatives come from *medium and atmosphere* at permissive temperatures.

This quantifies that claim and the transfer gap between them:

* coverage / overlap  -- taxonomy, media, genome accessions, condition space
* label structure     -- balance, and which factor actually carries the label
* identifiability     -- for each dataset, is a no-growth label separable by
                         temperature, by medium, or by atmosphere?
* transfer            -- in-distribution vs held-out-group predictive power
                         for each single factor, in both datasets

Read-only. Writes tables + figures under reports/compare_exp_bacdive/.

Usage: 
    python scripts/utils/compare_exp_bacdive.py
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, matthews_corrcoef
from sklearn.model_selection import GroupKFold, StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
EXP_CSV = ROOT / "data" / "processed" / "aero_anaero_tidy_curves_filtered.csv"
BD_PARQUET = ROOT / "data" / "external" / "bacdive_growth_52k_dedup_w_negs.parquet"
FIG_DIR = ROOT / "reports" / "compare_exp_bacdive"

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 80)
pd.set_option("display.max_colwidth", 60)

EXP_COLOR, BD_COLOR = "tab:green", "tab:purple"
LABEL_COLOR = {1: "tab:blue", 0: "tab:orange"}
SEED = 0

MEDIA_MAP: dict[str, str] = {
    "lb": "LB",
    "mb": "Marine Medium",
    "nb": "Nutrient Broth",
    "bhi": "BHI",
    "mrs": "MRS Broth",
    "thy": "THY",
    "tsb": "Tryptic Soy Broth",
    "gifu": "GAM",
    "mrc": "MRC",
    "cdm": "CDM",
    "m9": "M9 Minimal Medium",
}
# Long forms BacDive uses for the MEDIA_MAP acronyms
_MEDIA_EXTRA_ALIASES: dict[str, tuple[str, ...]] = {
    "lb": ("luria bertani", "lysogeny"),
    "bhi": ("brain heart infusion",),
    "thy": ("todd hewitt",),
    "gifu": ("gifu anaerobic",),
}

# Trailing / filler words that do not change medium identity
_GENERIC_MEDIA_TOKENS = frozenset({
    "broth", "medium", "media", "agar", "agars", "minimal", "modified",
    "plates", "plate", "ahar", "ahars", "ager",
})
_SKIP_LEADING_MEDIA_TOKENS = frozenset({
    "x", "modified", "modi", "half", "strength", "bacto", "diluted",
    "liquid", "solid", "pre", "reduced", "anaerobe", "anaerobic",
})
# Spelling variants BacDive uses for the same TSB identity.
_MEDIA_TOKEN_CANON = {
    "trypticase": "tryptic",
    "tryptone": "tryptic",
    "tryptose": "tryptic",
    "tripticase": "tryptic",
    "trypicase": "tryptic",
    "soya": "soy",
}
_SHORT_MEDIA_CODES = frozenset(MEDIA_MAP) | {
    v.lower() for v in MEDIA_MAP.values() if v.isalpha() and len(v) <= 4
}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _save(fig: plt.Figure, name: str, *, tight: bool = True) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")
    return path


def _write_table(df: pd.DataFrame, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    df.to_csv(path, index=False)
    print(f"  wrote {path}")


def blank(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip().replace({"None": "", "nan": ""})


def species_key(name: str) -> str:
    """Lowercased 'Genus species', dropping author/subsp. tails and strain tags."""
    toks = str(name).replace("_", " ").split()
    toks = [t for t in toks if t and not t.startswith("(")]
    return " ".join(toks[:2]).lower()


def accession_core(acc: str) -> str:
    body = str(acc).split(".", 1)[0]
    return body[4:] if body[:4] in ("GCA_", "GCF_") else body


def _media_tokens(name: str) -> list[str]:
    raw = [_MEDIA_TOKEN_CANON.get(t, t) for t in re.findall(r"[a-z0-9]+", str(name).lower())]
    out: list[str] = []
    for tok in raw:
        split = False
        for code in sorted(_SHORT_MEDIA_CODES, key=len, reverse=True):
            if tok.endswith(code) and len(tok) > len(code):
                prefix = tok[:-len(code)]
                if prefix and all(c.isdigit() or c == "x" for c in prefix):
                    out.extend([c for c in prefix if c != "x"] + [code])
                    split = True
                    break
        if not split:
            out.append(tok)
    return out


def _norm_media(name: str) -> str:
    return "".join(_media_tokens(name))


def exp_media_aliases(name: str) -> set[str]:
    """Short experimental code, MEDIA_MAP display name, and known long forms."""
    key = str(name).strip().lower()
    aliases = {key}
    mapped = MEDIA_MAP.get(key)
    if mapped:
        aliases.add(mapped.lower())
    aliases.update(_MEDIA_EXTRA_ALIASES.get(key, ()))
    return aliases


def _leading_media_core(name: str) -> list[str]:
    """Distinctive leading tokens of a BacDive medium, stripping dilutions and DSMZ tails."""
    toks = _media_tokens(name)
    i = 0
    while i < len(toks) and (
        toks[i].isdigit() or toks[i] in _SKIP_LEADING_MEDIA_TOKENS or len(toks[i]) == 1
    ):
        i += 1
    core: list[str] = []
    for t in toks[i:]:
        if t == "dsmz" or t in {"with", "supplemented", "containing"}:
            break
        if t in _GENERIC_MEDIA_TOKENS:
            continue
        core.append(t)
    return core


def media_alias_match(aliases: set[str], bd_name: str) -> bool:
    """True if a BacDive name is the same medium as any experimental alias.

    Short codes (lb, bhi, tsb, gam, ...) match as whole tokens, including
    parentheticals like ``marine broth (MB)``. Multi-word MEDIA_MAP names
    match when their distinctive tokens are the BacDive core
    (``Marine Medium`` ~ ``Marine Broth 2216``, not ``Marine Caulobacter``).
    Parenthetical expansions (``NA (Nutrient Agar)``) are tried as well;
    one-word parentheticals are only used as acronyms, not as ``(marine)``.
    """
    parens = re.findall(r"\(([^)]*)\)", str(bd_name))
    primary = re.sub(r"\([^)]*\)", " ", str(bd_name))
    if _media_alias_match_span(aliases, primary):
        return True
    short_alias_toks: set[str] = set()
    for a in aliases:
        at = _media_tokens(a)
        if len(at) == 1:
            short_alias_toks.add(at[0])
    for p in parens:
        pt = _media_tokens(p)
        if pt[:1] == ["dsmz"]:
            continue
        if len(pt) == 1:
            if pt[0] in short_alias_toks:
                return True
            continue
        if _media_alias_match_span(aliases, p):
            return True
    return False


def _media_alias_match_span(aliases: set[str], bd_name: str) -> bool:
    bd_toks = set(_media_tokens(bd_name))
    bd_lead = _leading_media_core(bd_name)
    bd_norm = _norm_media(bd_name)
    for alias in aliases:
        if _norm_media(alias) == bd_norm:
            return True
        a_toks = _media_tokens(alias)
        a_core = [t for t in a_toks if t not in _GENERIC_MEDIA_TOKENS]
        if not a_toks:
            continue
        if len(a_toks) == 1:
            if a_toks[0] in bd_toks:
                return True
            continue
        if not a_core:
            continue
        # "Marine Medium" ~ "Marine Broth 2216", but not "Marine Caulobacter medium".
        if len(a_core) == 1:
            rest = bd_lead[1:]
            if bd_lead[:1] == a_core and all(t.isdigit() for t in rest):
                return True
            continue
        if bd_lead[:len(a_core)] == a_core:
            return True
    return False


# --------------------------------------------------------------------------
# load
# --------------------------------------------------------------------------

def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    exp = pd.read_csv(EXP_CSV)
    exp["temp_num"] = pd.to_numeric(exp.temperature, errors="coerce")
    exp["species_key"] = exp.strain_name.map(species_key)
    exp["acc_core"] = exp.ncbi_assembly_accession.map(accession_core)

    bd = pd.read_parquet(BD_PARQUET)
    for c in ["strain_name_raw", "media_name", "genus", "family", "order",
              "phylum", "oxygen_tolerance", "complexity", "genome_accession"]:
        bd[c] = blank(bd[c])
    bd["temp_num"] = pd.to_numeric(bd.temperature, errors="coerce")
    bd["species_key"] = bd.strain_name_raw.map(species_key)
    bd["acc_core"] = bd.genome_accession.map(accession_core)
    return exp, bd


# --------------------------------------------------------------------------
# 1. overview
# --------------------------------------------------------------------------

def overview(exp: pd.DataFrame, bd: pd.DataFrame) -> pd.DataFrame:
    def row(name, d, strain_col, n_atm, n_temp):
        return {
            "dataset": name,
            "rows": len(d),
            "strains": d[strain_col].nunique(),
            "media": d.media_name.nunique(),
            "atmospheres": n_atm,
            "temperatures": n_temp,
            "positives": int((d.growth_class == 1).sum()),
            "negatives": int((d.growth_class == 0).sum()),
            "pos_rate": round(float(d.growth_class.mean()), 4),
            "rows_per_strain": round(len(d) / d[strain_col].nunique(), 1),
        }

    tab = pd.DataFrame([
        row("internal (plate reader)", exp, "strain_name",
            exp.atmosphere.nunique(), exp.temp_num.nunique()),
        row("BacDive 52k", bd, "bacdive_id", 0, int(bd.temp_num.nunique())),
    ])
    print("\n=== T1: dataset overview ===")
    print(tab.to_string(index=False))
    _write_table(tab, "t1_overview.csv")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    names = ["internal", "BacDive 52k"]
    ax = axes[0]
    pos = [float(exp.growth_class.mean()), float(bd.growth_class.mean())]
    ax.bar(names, pos, color=[EXP_COLOR, BD_COLOR])
    ax.axhline(0.5, ls="--", c="grey", lw=1)
    ax.set_ylim(0, 1); ax.set_ylabel("positive rate")
    ax.set_title("Label balance")
    for i, v in enumerate(pos):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center")

    ax = axes[1]
    negs = [int((exp.growth_class == 0).sum()), int((bd.growth_class == 0).sum())]
    ax.bar(names, negs, color=[EXP_COLOR, BD_COLOR])
    ax.set_yscale("log"); ax.set_ylabel("no-growth rows (log)")
    ax.set_title("Negative count")
    for i, v in enumerate(negs):
        ax.text(i, v * 1.1, f"{v:,}", ha="center")

    ax = axes[2]
    exp_cells = exp.groupby(["strain_name", "media_name", "atmosphere", "temp_num"]).ngroups
    exp_full = (exp.strain_name.nunique() * exp.media_name.nunique()
                * exp.atmosphere.nunique() * exp.temp_num.nunique())
    bd_cells = bd.groupby(["bacdive_id", "media_name"]).ngroups
    bd_full = bd.bacdive_id.nunique() * bd.media_name.nunique()
    dens = [100 * exp_cells / exp_full, 100 * bd_cells / bd_full]
    ax.bar(names, dens, color=[EXP_COLOR, BD_COLOR])
    ax.set_yscale("log"); ax.set_ylabel("% of full grid observed (log)")
    ax.set_title("Design density")
    for i, v in enumerate(dens):
        ax.text(i, v * 1.15, f"{v:.2g}%", ha="center")
    fig.suptitle("Internal benchmark vs BacDive 52k")
    _save(fig, "f1_overview.png")
    return tab


# --------------------------------------------------------------------------
# 2. temperature / condition space -- the headline contrast
# --------------------------------------------------------------------------

def condition_space(exp: pd.DataFrame, bd: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.5))
    for j, lab in enumerate((1, 0)):
        ax = axes[0, j]
        s = exp.loc[exp.growth_class == lab, "temp_num"].dropna()
        ax.hist(s, bins=np.arange(20, 45, 1), color=LABEL_COLOR[lab],
                edgecolor="white", linewidth=0.4)
        ax.set_title(f"internal — growth_class={lab}  n={len(s):,}")
        ax.set_xlabel("temperature (°C)"); ax.set_ylabel("rows")
        ax.set_xlim(0, 90)
    for j, lab in enumerate((1, 0)):
        ax = axes[1, j]
        s = bd.loc[bd.growth_class == lab, "temp_num"].dropna()
        ax.hist(s, bins=np.arange(-6, 92, 1), color=LABEL_COLOR[lab],
                edgecolor="white", linewidth=0.4)
        ax.set_title(f"BacDive 52k — growth_class={lab}  n={len(s):,}")
        ax.set_xlabel("temperature (°C)"); ax.set_ylabel("rows")
        ax.set_xlim(0, 90)
    fig.suptitle("Where the no-growth labels live in temperature space\n"
                 "internal: negatives at permissive T (medium/O2 driven) | "
                 "BacDive: negatives at temperature extremes")
    _save(fig, "f2_temp_by_label_both.png")

    print("\n=== T4: condition space ===")
    rows = []
    for name, d in [("internal", exp), ("BacDive 52k", bd)]:
        for lab in (1, 0):
            s = d.loc[d.growth_class == lab, "temp_num"].dropna()
            rows.append({
                "dataset": name, "growth_class": lab, "n": len(s),
                "temp_mean": round(s.mean(), 2), "temp_std": round(s.std(), 2),
                "temp_min": s.min(), "temp_p05": s.quantile(.05),
                "temp_p95": s.quantile(.95), "temp_max": s.max(),
                "n_distinct_temps": s.nunique(),
            })
    tab = pd.DataFrame(rows)
    print(tab.to_string(index=False))
    _write_table(tab, "t4_condition_space.csv")

    # internal: label rate across media x atmosphere (the structure BacDive lacks)
    piv = (exp.pivot_table(index="media_name", columns=["atmosphere", "temp_num"],
                           values="growth_class", aggfunc="mean"))
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(piv.values, cmap="RdYlBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f"{a}\n{t:g}°C" for a, t in piv.columns], fontsize=8)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index, fontsize=9)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="growth rate")
    ax.set_title("Internal benchmark: growth rate by medium x atmosphere x temperature")
    _save(fig, "f3_internal_media_atm_heatmap.png")


# --------------------------------------------------------------------------
# 3. identifiability: which factor separates a no-growth label?
# --------------------------------------------------------------------------

def identifiability(exp: pd.DataFrame, bd: pd.DataFrame) -> pd.DataFrame:
    rows = []

    # -- BacDive: hold (strain, medium) fixed, does temperature separate?
    g = bd.groupby(["bacdive_id", "media_name"]).growth_class.agg(
        n="size", pos="sum", uniq="nunique")
    both = g[g.uniq > 1]
    conf = bd.set_index(["bacdive_id", "media_name"])
    conf = bd[conf.index.isin(both.index)]
    sep = 0
    if len(conf):
        agg = conf.groupby(["bacdive_id", "media_name", "growth_class"]).temp_num.agg(
            ["min", "max"]).unstack("growth_class").dropna()
        agg.columns = [f"{a}_{b}" for a, b in agg.columns]
        if len(agg):
            sep = int(((agg["min_0"] < agg["min_1"]) | (agg["max_0"] > agg["max_1"])).sum())
            denom = len(agg)
        else:
            denom = 0
    else:
        denom = 0
    rows.append({"dataset": "BacDive 52k", "test": "dual-label (strain, medium) cells",
                 "n_cells": len(both),
                 "detail": f"{sep}/{denom} separated by temperature alone",
                 "frac": round(sep / denom, 4) if denom else np.nan})

    # -- BacDive: hold (strain, temperature) fixed, does the medium separate?
    g2 = bd.groupby(["bacdive_id", "temp_num"]).agg(
        n_media=("media_name", "nunique"), uniq=("growth_class", "nunique"))
    multi = g2[g2.n_media > 1]
    rows.append({"dataset": "BacDive 52k",
                 "test": "(strain, temperature) cells with >1 medium",
                 "n_cells": len(multi),
                 "detail": f"{int((multi.uniq > 1).sum())} show a medium-driven label flip",
                 "frac": round(float((multi.uniq > 1).mean()), 4) if len(multi) else np.nan})

    # -- internal: hold (strain, medium, atmosphere) fixed, does temperature separate?
    g3 = exp.groupby(["strain_name", "media_name", "atmosphere"]).agg(
        n_temp=("temp_num", "nunique"), uniq=("growth_class", "nunique"))
    m3 = g3[g3.n_temp > 1]
    rows.append({"dataset": "internal",
                 "test": "(strain, medium, atm) cells with >1 temperature",
                 "n_cells": len(m3),
                 "detail": f"{int((m3.uniq > 1).sum())} show a temperature-driven flip",
                 "frac": round(float((m3.uniq > 1).mean()), 4) if len(m3) else np.nan})

    # -- internal: hold (strain, atmosphere, temperature) fixed, does medium separate?
    g4 = exp.groupby(["strain_name", "atmosphere", "temp_num"]).agg(
        n_media=("media_name", "nunique"), uniq=("growth_class", "nunique"))
    m4 = g4[g4.n_media > 1]
    rows.append({"dataset": "internal",
                 "test": "(strain, atm, temperature) cells with >1 medium",
                 "n_cells": len(m4),
                 "detail": f"{int((m4.uniq > 1).sum())} show a medium-driven flip",
                 "frac": round(float((m4.uniq > 1).mean()), 4) if len(m4) else np.nan})

    # -- internal: hold (strain, medium, temperature) fixed, does atmosphere separate?
    g5 = exp.groupby(["strain_name", "media_name", "temp_num"]).agg(
        n_atm=("atmosphere", "nunique"), uniq=("growth_class", "nunique"))
    m5 = g5[g5.n_atm > 1]
    rows.append({"dataset": "internal",
                 "test": "(strain, medium, temperature) cells with both atmospheres",
                 "n_cells": len(m5),
                 "detail": f"{int((m5.uniq > 1).sum())} show an atmosphere-driven flip",
                 "frac": round(float((m5.uniq > 1).mean()), 4) if len(m5) else np.nan})

    tab = pd.DataFrame(rows)
    print("\n=== T5: which factor separates a no-growth label? ===")
    print(tab.to_string(index=False))
    _write_table(tab, "t5_identifiability.csv")
    return tab


# --------------------------------------------------------------------------
# 4. single-factor predictive power, in-distribution vs held-out group
# --------------------------------------------------------------------------

def _cv_score(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray | None,
              n_splits: int = 5) -> tuple[float, float]:
    if X.shape[1] == 0 or len(np.unique(y)) < 2:
        return np.nan, np.nan
    oof = np.full(len(y), np.nan)
    if groups is None:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        it = splitter.split(X, y)
    else:
        k = min(n_splits, len(np.unique(groups)))
        if k < 2:
            return np.nan, np.nan
        it = GroupKFold(n_splits=k).split(X, y, groups)
    for tr, te in it:
        if len(np.unique(y[tr])) < 2:
            continue
        m = HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.1, random_state=SEED,
            categorical_features="from_dtype")
        m.fit(X.iloc[tr], y[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    ok = ~np.isnan(oof)
    if ok.sum() == 0 or len(np.unique(y[ok])) < 2:
        return np.nan, np.nan
    pred = (oof[ok] >= 0.5).astype(int)
    # average precision on the no-growth class -- the minority class that matters
    ap = average_precision_score(1 - y[ok], 1 - oof[ok])
    return matthews_corrcoef(y[ok], pred), ap


def _cat(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for c in cols:
        if frame[c].nunique() > 250:
            out[c] = frame[c].astype("category").cat.codes.astype("int32")
        else:
            out[c] = frame[c].astype("category")
    return out


def factor_power(exp: pd.DataFrame, bd: pd.DataFrame) -> pd.DataFrame:
    rows = []

    med_flags = ["has_yeast_extract", "has_peptone", "has_blood_components"]
    # ---- internal
    y = exp.growth_class.astype(int).to_numpy()
    sets = {
        "majority class": pd.DataFrame(index=exp.index),
        "temperature only": exp[["temp_num"]],
        "atmosphere only": _cat(exp, ["atmosphere"]),
        "medium only (identity)": _cat(exp, ["media_name"]),
        "medium only (features)": pd.concat(
            [_cat(exp, ["complexity", "richness"]), exp[med_flags]], axis=1),
        "taxonomy only": _cat(exp, ["genus", "family", "phylum"]),
        "medium + T + atm": pd.concat(
            [exp[["temp_num"]], _cat(exp, ["atmosphere", "media_name"])], axis=1),
        "medium + T + atm + taxonomy": pd.concat(
            [exp[["temp_num"]], _cat(exp, ["atmosphere", "media_name", "genus",
                                           "family", "phylum"])], axis=1),
    }
    for label, X in sets.items():
        if X.shape[1] == 0:
            mcc_r = 0.0
            ap_r = float((y == 0).mean())
            mcc_g, ap_g = 0.0, float((y == 0).mean())
        else:
            mcc_r, ap_r = _cv_score(X, y, None)
            mcc_g, ap_g = _cv_score(X, y, exp.strain_name.to_numpy())
        rows.append({"dataset": "internal", "features": label,
                     "MCC_random": mcc_r, "AP_nogrowth_random": ap_r,
                     "MCC_strain_grouped": mcc_g, "AP_nogrowth_strain_grouped": ap_g})

    # ---- BacDive
    yb = bd.growth_class.astype(int).to_numpy()
    bd_flags = ["n_ingredients", "has_yeast_extract", "has_peptone", "has_agar",
                "has_blood_components", "has_nacl"]
    for c in bd_flags:
        bd[c] = pd.to_numeric(bd[c], errors="coerce")
    setsb = {
        "majority class": pd.DataFrame(index=bd.index),
        "temperature only": bd[["temp_num"]],
        "medium only (identity)": _cat(bd, ["media_name"]),
        "medium only (features)": pd.concat(
            [_cat(bd, ["complexity"]), bd[bd_flags]], axis=1),
        "taxonomy only": _cat(bd, ["genus", "family", "phylum"]),
        "medium + T": pd.concat([bd[["temp_num"]], _cat(bd, ["media_name"])], axis=1),
        "medium + T + taxonomy": pd.concat(
            [bd[["temp_num"]], _cat(bd, ["media_name", "genus", "family", "phylum"])],
            axis=1),
    }
    for label, X in setsb.items():
        if X.shape[1] == 0:
            mcc_r = 0.0
            ap_r = float((yb == 0).mean())
            mcc_g, ap_g = 0.0, float((yb == 0).mean())
        else:
            mcc_r, ap_r = _cv_score(X, yb, None)
            mcc_g, ap_g = _cv_score(X, yb, bd.family.to_numpy())
        rows.append({"dataset": "BacDive 52k", "features": label,
                     "MCC_random": mcc_r, "AP_nogrowth_random": ap_r,
                     "MCC_strain_grouped": mcc_g, "AP_nogrowth_strain_grouped": ap_g})

    tab = pd.DataFrame(rows).round(4)
    print("\n=== T6: single-factor predictive power ===")
    print("  (internal grouped by strain_name; BacDive grouped by family)")
    print(tab.to_string(index=False))
    _write_table(tab, "t6_factor_power.csv")

    for ds, fname in [("internal", "f4_factor_power_internal.png"),
                      ("BacDive 52k", "f5_factor_power_bacdive.png")]:
        sub = tab[tab.dataset == ds]
        fig, ax = plt.subplots(figsize=(9, 4))
        idx = np.arange(len(sub))
        ax.barh(idx - 0.2, sub.MCC_random, height=0.38,
                label="random CV (in-distribution)", color="tab:blue")
        ax.barh(idx + 0.2, sub.MCC_strain_grouped, height=0.38,
                label="grouped CV (held-out group)", color="tab:red")
        ax.set_yticks(idx); ax.set_yticklabels(sub.features, fontsize=9)
        ax.invert_yaxis(); ax.set_xlabel("MCC"); ax.set_xlim(0, 1)
        ax.legend(fontsize=8); ax.set_title(f"{ds}: what carries the label?")
        ax.grid(axis="x", alpha=0.3)
        _save(fig, fname)
    return tab


# --------------------------------------------------------------------------
# 5. overlap: taxonomy, media, genomes
# --------------------------------------------------------------------------

def overlap(exp: pd.DataFrame, bd: pd.DataFrame) -> None:
    bd_species = set(bd.species_key)
    bd_genus = set(bd.genus.str.lower())
    bd_family = set(bd.family.str.lower())
    bd_cores = set(bd.acc_core)
    n_by_genus = bd.groupby(bd.genus.str.lower()).bacdive_id.nunique()
    n_by_species = bd.groupby("species_key").bacdive_id.nunique()

    per = (exp.groupby("strain_name")
              .agg(n_rows=("growth_class", "size"),
                   pos_rate=("growth_class", "mean"),
                   species_key=("species_key", "first"),
                   genus=("genus", "first"),
                   family=("family", "first"),
                   acc=("ncbi_assembly_accession", "first"),
                   acc_core=("acc_core", "first"))
              .reset_index())
    per["species_in_bacdive"] = per.species_key.isin(bd_species)
    per["genus_in_bacdive"] = per.genus.str.lower().isin(bd_genus)
    per["family_in_bacdive"] = per.family.str.lower().isin(bd_family)
    per["genome_in_bacdive"] = per.acc_core.isin(bd_cores)
    per["n_bacdive_same_species"] = per.species_key.map(n_by_species).fillna(0).astype(int)
    per["n_bacdive_same_genus"] = per.genus.str.lower().map(n_by_genus).fillna(0).astype(int)
    per["pos_rate"] = per.pos_rate.round(3)

    print("\n=== T2: per-strain overlap with BacDive 52k ===")
    print(per.drop(columns=["acc_core", "species_key"]).to_string(index=False))
    _write_table(per, "t2_strain_overlap.csv")

    print("\n  summary:")
    for col, lab in [("species_in_bacdive", "same species present in BacDive"),
                     ("genus_in_bacdive", "same genus present"),
                     ("family_in_bacdive", "same family present"),
                     ("genome_in_bacdive", "SAME GENOME ASSEMBLY present (leakage risk)")]:
        print(f"    {lab:52s} {int(per[col].sum()):2d} / {len(per)}")

    # media overlap — expand each internal short code with MEDIA_MAP aliases
    exp_media = sorted(set(exp.media_name.str.lower()))
    bd_media = set(bd.media_name.str.lower())
    exp_alias_norms: set[str] = set()
    per_media_rows = []
    alias_hits: set[str] = set()
    for em in exp_media:
        aliases = exp_media_aliases(em)
        exp_alias_norms.update(_norm_media(a) for a in aliases)
        matched = sorted(m for m in bd_media if media_alias_match(aliases, m))
        alias_hits.update(matched)
        per_media_rows.append({
            "exp_media": em,
            "aliases": "; ".join(sorted(aliases)),
            "n_bacdive_matches": len(matched),
            "bacdive_names": "; ".join(matched),
        })
    exact = set(exp_media) & bd_media
    norm_hits = {m for m in bd_media if _norm_media(m) in exp_alias_norms}
    n_exp_hit = sum(r["n_bacdive_matches"] > 0 for r in per_media_rows)

    print("\n=== T3: media overlap ===")
    print(f"  internal media                     : {len(exp_media)}")
    print(f"  BacDive media                      : {len(bd_media)}")
    print(f"  exact name matches                 : {len(exact)}")
    print(f"  alias-normalised exact matches     : {len(norm_hits)}")
    print(f"  robust alias matches               : {len(alias_hits)}")
    print(f"  internal media with a BacDive hit  : {n_exp_hit} / {len(exp_media)}")
    print("  per internal medium:")
    for r in per_media_rows:
        preview = r["bacdive_names"].split("; ")[:6] if r["bacdive_names"] else []
        tail = " ..." if r["n_bacdive_matches"] > 6 else ""
        print(f"    {r['exp_media']:6s}  aliases={r['aliases']:28s}  "
              f"n={r['n_bacdive_matches']:4d}  e.g. {preview}{tail}")
    mt = pd.DataFrame({
        "metric": ["internal_media", "bacdive_media", "exact_matches",
                   "alias_normalised_exact", "robust_alias_matches",
                   "internal_media_with_hit"],
        "value": [len(exp_media), len(bd_media), len(exact),
                  len(norm_hits), len(alias_hits), n_exp_hit],
    })
    _write_table(mt, "t3_media_overlap.csv")
    _write_table(pd.DataFrame(per_media_rows), "t3_media_overlap_by_medium.csv")

    # figure: taxonomic overlap
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    p = per.sort_values("n_bacdive_same_genus", ascending=True)
    ax.barh(p.strain_name, p.n_bacdive_same_genus.clip(lower=0.5),
            color=[BD_COLOR if v else "lightgrey" for v in p.n_bacdive_same_genus])
    ax.set_xscale("log"); ax.set_xlabel("BacDive strains in the same genus (log)")
    ax.tick_params(axis="y", labelsize=7)
    ax.set_title("Taxonomic support in BacDive for each internal strain")
    ax = axes[1]
    counts = [int(per.species_in_bacdive.sum()), int(per.genus_in_bacdive.sum()),
              int(per.family_in_bacdive.sum()), int(per.genome_in_bacdive.sum())]
    ax.bar(["species", "genus", "family", "same\nassembly"], counts, color=BD_COLOR)
    ax.axhline(len(per), ls="--", c="grey", lw=1)
    ax.set_ylabel(f"internal strains (of {len(per)})")
    ax.set_title("Overlap level with BacDive 52k")
    for i, v in enumerate(counts):
        ax.text(i, v + 0.4, str(v), ha="center")
    _save(fig, "f6_taxonomic_overlap.png")

    # figure: media feature-space comparison
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    ax = axes[0]
    for i, (name, d, col) in enumerate([("internal", exp, EXP_COLOR),
                                        ("BacDive", bd, BD_COLOR)]):
        vc = d.complexity.value_counts(normalize=True)
        for j, k in enumerate(["defined", "semi_defined", "undefined"]):
            ax.bar(j + (i - 0.5) * 0.38, vc.get(k, 0), width=0.36,
                   color=col, label=name if j == 0 else None)
    ax.set_xticks(range(3)); ax.set_xticklabels(["defined", "semi_def", "undefined"])
    ax.set_ylabel("fraction of rows"); ax.set_title("Medium complexity"); ax.legend(fontsize=8)

    ax = axes[1]
    flags = ["has_yeast_extract", "has_peptone", "has_blood_components"]
    for i, (name, d, col) in enumerate([("internal", exp, EXP_COLOR),
                                        ("BacDive", bd, BD_COLOR)]):
        vals = [pd.to_numeric(d[f], errors="coerce").mean() for f in flags]
        ax.bar(np.arange(3) + (i - 0.5) * 0.38, vals, width=0.36, color=col,
               label=name)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["yeast ext.", "peptone", "blood"], fontsize=8)
    ax.set_ylabel("fraction of rows"); ax.set_title("Medium ingredient flags")
    ax.legend(fontsize=8)

    ax = axes[2]
    mps = exp.groupby("strain_name").media_name.nunique()
    mpsb = bd.groupby("bacdive_id").media_name.nunique()
    ax.hist(mpsb, bins=np.arange(0.5, 20.5, 1), density=True, alpha=0.7,
            color=BD_COLOR, label=f"BacDive (med {mpsb.median():.0f})")
    ax.hist(mps, bins=np.arange(0.5, 20.5, 1), density=True, alpha=0.7,
            color=EXP_COLOR, label=f"internal (med {mps.median():.0f})")
    ax.set_xlabel("media tested per strain"); ax.set_ylabel("density")
    ax.set_title("Media per strain"); ax.legend(fontsize=8)
    _save(fig, "f7_media_space.png")


# --------------------------------------------------------------------------

def main() -> None:
    print("loading...")
    exp, bd = load()
    print(f"  internal {exp.shape}  |  bacdive {bd.shape}")
    overview(exp, bd)
    condition_space(exp, bd)
    identifiability(exp, bd)
    overlap(exp, bd)
    factor_power(exp, bd)
    print(f"\nall artifacts under {FIG_DIR}")


if __name__ == "__main__":
    main()
