#!/usr/bin/env python3
"""Where do the negatives come from, and what is a medium keyed by?

The concordance test only means something if positives and negatives are
drawn from the same process. This checks whether growth_class==0 rows are
observed no-growth records or generated pairs, and re-runs the intra-species
concordance on media_name (mediadive_id is blank on most rows).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PARQUET = ROOT / "data" / "external" / "bacdive_growth_40k.parquet"
MAP = ROOT / "data" / "external" / "bacdive_genome_map.csv"

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 80)
pd.set_option("display.max_colwidth", 70)


def blank(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip().replace({"None": "", "nan": ""})


def species_key(name: str) -> str:
    p = str(name).split()
    return " ".join(p[:2]).lower() if len(p) >= 2 else str(name).lower()


def main() -> None:
    df = pd.read_parquet(PARQUET)
    for c in ["bacdive_id", "strain_name_raw", "mediadive_id", "media_name",
              "media_name_full", "cip_medium_number", "composition_raw",
              "medium_link", "refs", "temperature"]:
        df[c] = blank(df[c])
    df["species_key"] = df.strain_name_raw.map(species_key)

    print("=== growth_class x mediadive_id present ===")
    print(pd.crosstab(df.growth_class, df.mediadive_id.eq(""),
                      rownames=["growth_class"], colnames=["mediadive_id blank"]).to_string())

    print("\n=== growth_class x cip_medium_number present ===")
    print(pd.crosstab(df.growth_class, df.cip_medium_number.eq(""),
                      rownames=["growth_class"], colnames=["cip blank"]).to_string())

    print("\n=== growth_class x composition_raw present ===")
    print(pd.crosstab(df.growth_class, df.composition_raw.eq(""),
                      rownames=["growth_class"], colnames=["composition blank"]).to_string())

    print("\n=== media_name coverage ===")
    print(f"  distinct media_name       : {df.media_name.nunique()}")
    print(f"  distinct media_name_full  : {df.media_name_full.nunique()}")
    print(f"  rows w/ blank media_name  : {int(df.media_name.eq('').sum())}")

    print("\n=== top media_name for POSITIVES ===")
    print(df[df.growth_class == 1].media_name.value_counts().head(12).to_string())
    print("\n=== top media_name for NEGATIVES ===")
    print(df[df.growth_class == 0].media_name.value_counts().head(12).to_string())

    print("\n=== sample negative rows ===")
    cols = ["bacdive_id", "strain_name_raw", "media_name", "mediadive_id",
            "cip_medium_number", "growth_class", "temperature"]
    print(df[df.growth_class == 0][cols].head(12).to_string(index=False))

    print("\n=== do negatives and positives share media_name vocabulary? ===")
    pos_media = set(df.loc[df.growth_class == 1, "media_name"])
    neg_media = set(df.loc[df.growth_class == 0, "media_name"])
    print(f"  media only in positives : {len(pos_media - neg_media)}")
    print(f"  media only in negatives : {len(neg_media - pos_media)}")
    print(f"  media in both           : {len(pos_media & neg_media)}")

    print("\n=== per-strain label composition ===")
    per = df.groupby("bacdive_id").growth_class.agg(n="size", pos="sum")
    per["neg"] = per.n - per.pos
    print(f"  strains with only positives : {int((per.neg == 0).sum())}")
    print(f"  strains with only negatives : {int((per.pos == 0).sum())}")
    print(f"  strains with both           : {int(((per.pos > 0) & (per.neg > 0)).sum())}")

    # ---- concordance re-run on media_name, the key that is actually populated
    sub = df[(df.media_name != "") & df.growth_class.notna()].copy()
    sub["growth_class"] = sub.growth_class.astype(int)
    # collapse to one label per (strain, medium) first, so a strain tested
    # twice on one medium does not count as two votes
    cell = (sub.groupby(["species_key", "media_name", "bacdive_id"])
               .growth_class.max().reset_index())
    grp = cell.groupby(["species_key", "media_name"]).growth_class.agg(
        n_strains="size", n_pos="sum", n_uniq="nunique")
    multi = grp[grp.n_strains > 1]
    print("\n=== intra-species concordance keyed on media_name ===")
    print(f"  (species, medium) cells w/ >1 distinct strain : {len(multi)}")
    print(f"  cells where strains DISAGREE                  : "
          f"{int((multi.n_uniq > 1).sum())} ({100*(multi.n_uniq>1).mean():.2f}%)")
    print(f"  strain-level observations in those cells       : {int(multi.n_strains.sum())}")
    maj = multi.assign(maj=lambda d: d[["n_pos"]].assign(
        neg=d.n_strains - d.n_pos).max(axis=1))
    print(f"  ceiling for one-genome-per-species (majority)  : "
          f"{maj.maj.sum()/multi.n_strains.sum():.4f}")

    print("\n-- the same, split by whether the cell contains any negative --")
    withneg = multi[(multi.n_pos < multi.n_strains)]
    print(f"  cells containing >=1 no-growth label : {len(withneg)}")
    print(f"  of those, disagreeing                : {int((withneg.n_uniq>1).sum())}")

    print("\n-- largest disagreeing cells --")
    d = multi[multi.n_uniq > 1].sort_values("n_strains", ascending=False)
    print(d.head(15).to_string())

    # unresolved-strain view
    gmap = pd.read_csv(MAP, dtype=str).fillna("")
    mapped = set(gmap.bacdive_id)
    unres = df[~df.bacdive_id.isin(mapped)]
    print("\n=== unresolved rows: label balance vs resolved ===")
    print(f"  unresolved rows   n={len(unres)}  pos_rate={unres.growth_class.mean():.3f}")
    res = df[df.bacdive_id.isin(mapped)]
    print(f"  resolved   rows   n={len(res)}  pos_rate={res.growth_class.mean():.3f}")


if __name__ == "__main__":
    main()
