#!/usr/bin/env python3
"""What actually distinguishes a no-growth row from a growth row?

The sampled negatives showed the same (strain, medium) pair labelled 0 at two
temperatures, which would mean the negative signal is temperature rather than
medium composition. If so, the media/genome framing of the task is largely
label-free and no amount of genome imputation changes that. This tests it.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PARQUET = ROOT / "data" / "external" / "bacdive_growth_40k.parquet"
FIG_DIR = ROOT / "reports" / "diagnose_negative_provenance"

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 80)

LABEL_COLOR = {1: "tab:blue", 0: "tab:orange"}


def blank(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip().replace({"None": "", "nan": ""})


def _save(fig: plt.Figure, name: str, *, tight: bool = True) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")
    return path


def plot_temp_by_label(df: pd.DataFrame) -> None:
    """Histogram + top-15 count bars of temperature, split by growth_class."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    temps = df.temp_num.dropna()
    lo, hi = temps.min(), temps.max()
    bins = range(int(lo) - 1, int(hi) + 3) if temps.size else 40
    for ax, lab in zip(axes, (1, 0)):
        s = df.loc[df.growth_class == lab, "temp_num"].dropna()
        ax.hist(s, bins=bins, color=LABEL_COLOR[lab], edgecolor="white",
                linewidth=0.4)
        ax.set_title(f"growth_class == {lab}  n={len(s):,}")
        ax.set_xlabel("temperature (°C)")
        ax.set_ylabel("rows")
    fig.suptitle("Temperature histogram by label")
    _save(fig, "temp_by_label_hist.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=False)
    for ax, lab in zip(axes, (1, 0)):
        vc = df.loc[df.growth_class == lab, "temp_num"].value_counts().head(15)
        ax.barh(vc.index.astype(str)[::-1], vc.values[::-1],
                color=LABEL_COLOR[lab])
        ax.set_title(f"growth_class == {lab}")
        ax.set_xlabel("rows")
        ax.set_ylabel("temperature (°C)")
    fig.suptitle("Top 15 temperature values by label")
    _save(fig, "temp_top15_by_label.png")


def _short(s: str, n: int) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _pair_label(strain: str, medium: str) -> str:
    # species: first two tokens of the raw strain name
    toks = str(strain).split()
    strain_s = " ".join(toks[:2]) if toks else str(strain)
    return f"{_short(strain_s, 28)}  |  {_short(medium, 22)}"


def plot_neg_outside_growth_range(conf: pd.DataFrame, ok: pd.DataFrame) -> None:
    """Show that no-growth temps sit outside each conflicted pair's growth range."""
    colder = ok["min_0"] < ok["min_1"]
    hotter = ok["max_0"] > ok["max_1"]
    inside = ~(colder | hotter)
    n_out = int((colder | hotter).sum())
    n = len(ok)
    pct = 100 * n_out / n if n else 0.0

    order = ok.assign(mid=lambda d: (d.min_1 + d.max_1) / 2).sort_values(
        ["mid", "min_1", "max_1"])
    n_show = min(20, len(order))
    take = (pd.Series(range(n_show)) * (len(order) - 1) / max(n_show - 1, 1)
            ).round().astype(int).drop_duplicates()
    sample = order.iloc[take]
    rank = pd.Series(range(len(sample)), index=sample.index)
    rows = conf.dropna(subset=["temp_num"]).copy()
    rows["rank"] = rows.set_index(["bacdive_id", "media_name"]).index.map(rank)
    rows = rows.dropna(subset=["rank"])
    pos = rows[rows.growth_class == 1]
    neg = rows[rows.growth_class == 0]

    # one label per sampled (strain, medium) pair
    name_col = "strain_name_raw" if "strain_name_raw" in conf.columns else None
    if name_col is not None:
        strain_by_pair = (
            conf.groupby(["bacdive_id", "media_name"])[name_col]
            .first()
            .reindex(sample.index)
            .fillna("")
        )
    else:
        strain_by_pair = pd.Series(
            [str(i[0]) for i in sample.index], index=sample.index)
    yticklabels = [
        _pair_label(strain_by_pair.loc[idx], idx[1]) for idx in sample.index
    ]

    fig = plt.figure(figsize=(12.5, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.7, 1], hspace=0.42, wspace=0.28)
    ax_strip = fig.add_subplot(gs[0, :])
    ax_bar = fig.add_subplot(gs[1, 0])
    ax_hist = fig.add_subplot(gs[1, 1])

    y = rank.to_numpy()
    span = sample.max_1 - sample.min_1
    wide = span > 0
    ax_strip.barh(y[wide.to_numpy()], span[wide].to_numpy(),
                  left=sample.min_1[wide].to_numpy(), height=0.55,
                  color=LABEL_COLOR[1], alpha=0.55, zorder=1,
                  label="growth range")
    ax_strip.scatter(pos.temp_num, pos["rank"], s=42, c=LABEL_COLOR[1],
                     alpha=0.95, edgecolors="white", linewidths=0.4, zorder=2,
                     label="growth")
    ax_strip.scatter(neg.temp_num, neg["rank"], s=52, c=LABEL_COLOR[0],
                     alpha=0.95, edgecolors="0.2", linewidths=0.45, zorder=3,
                     label="no-growth")
    ax_strip.set_xlabel("temperature (°C)")
    ax_strip.set_title(
        f"showing {len(sample):,} of {n:,} pairs — orange (no-growth) never sits on the blue range")
    ax_strip.set_yticks(y)
    ax_strip.set_yticklabels(yticklabels, fontsize=8)
    ax_strip.set_ylim(-0.8, len(sample) - 0.2)
    ax_strip.legend(frameon=False, loc="upper right", markerscale=1.1)
    ax_strip.invert_yaxis()

    cats = ["colder than\ngrowth min", "hotter than\ngrowth max",
            "both sides", "inside\ngrowth range"]
    vals = [int((colder & ~hotter).sum()), int((~colder & hotter).sum()),
            int((colder & hotter).sum()), int(inside.sum())]
    colors = ["steelblue", "tab:red", "tab:purple", "0.75"]
    bars = ax_bar.bar(cats, vals, color=colors)
    ax_bar.set_ylabel("pairs")
    ax_bar.set_title("where the no-growth envelope sits")
    for b, v in zip(bars, vals):
        ax_bar.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{v:,}", ha="center", va="bottom", fontsize=9)
    ax_bar.set_ylim(0, max(vals + [1]) * 1.15)

    below = (ok.loc[colder, "min_0"] - ok.loc[colder, "min_1"])
    above = (ok.loc[hotter, "max_0"] - ok.loc[hotter, "max_1"])
    ax_hist.axvline(0, color="0.3", ls="--", lw=1)
    if len(below):
        ax_hist.hist(below, bins=40, color="steelblue", alpha=0.85,
                     label=f"below min  n={len(below):,}")
    if len(above):
        ax_hist.hist(above, bins=40, color="tab:red", alpha=0.75,
                     label=f"above max  n={len(above):,}")
    ax_hist.set_xlabel("signed °C from growth [min, max]\n(<0 colder, >0 hotter, 0 = inside)")
    ax_hist.set_ylabel("pairs")
    ax_hist.set_title("how far outside the growth range")
    ax_hist.legend(frameon=False)

    fig.suptitle(
        f"{n_out:,} / {n:,} conflicted pairs ({pct:.1f}%): "
        "no-growth temperatures fall outside the growth range",
        y=0.98,
    )
    _save(fig, "conflicted_temp_outside_range.png", tight=False)


def plot_pair_both_labels(pair: pd.DataFrame) -> None:
    """How often the same (strain, medium) pair carries both labels."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    uniq_counts = pair.uniq.value_counts().sort_index()
    axes[0].bar(uniq_counts.index.astype(int), uniq_counts.values,
                color=["tab:gray", "tab:red"][:len(uniq_counts)],
                width=0.6)
    axes[0].set_xticks(list(uniq_counts.index.astype(int)))
    axes[0].set_xlabel("n unique growth_class per (strain, medium)")
    axes[0].set_ylabel("pairs")
    axes[0].set_title("pairs by label uniqueness")

    single = pair.loc[pair.uniq == 1, "n"]
    both = pair.loc[pair.uniq > 1, "n"]
    xmax = int(pair.n.quantile(0.99)) if len(pair) else 1
    bins = range(1, max(xmax, 2) + 2)
    axes[1].hist(single, bins=bins, alpha=0.7, color="tab:gray",
                 label=f"single label  n={len(single):,}")
    if len(both):
        axes[1].hist(both, bins=bins, alpha=0.8, color="tab:red",
                     label=f"both labels  n={len(both):,}")
    axes[1].set_xlim(0.5, xmax + 1.5)
    axes[1].set_xlabel("rows per (strain, medium) pair")
    axes[1].set_ylabel("pairs")
    axes[1].set_title("pair size, split by whether both labels occur")
    axes[1].legend(frameon=False)

    fig.suptitle("(strain, medium) pairs labelled both 1 and 0")
    _save(fig, "pair_both_labels_hist.png")


def main() -> None:
    df = pd.read_parquet(PARQUET)
    for c in ["bacdive_id", "media_name", "temperature", "ph", "mediadive_id",
              "cip_medium_number", "composition_raw"]:
        df[c] = blank(df[c])
    df["temp_num"] = pd.to_numeric(df.temperature, errors="coerce")

    print("=== temperature by label ===")
    print(df.groupby("growth_class").temp_num.describe().to_string())

    print("\n=== temperature value counts by label (top 15 each) ===")
    for lab in (1, 0):
        print(f"\n-- growth_class == {lab} --")
        print(df[df.growth_class == lab].temp_num.value_counts().head(15).to_string())
    plot_temp_by_label(df)

    print("\n=== is the SAME (strain, medium) pair labelled both 1 and 0? ===")
    pair = df.groupby(["bacdive_id", "media_name"]).growth_class.agg(
        n="size", pos="sum", uniq="nunique")
    both = pair[pair.uniq > 1]
    print(f"  (strain, medium) pairs total          : {len(pair)}")
    print(f"  pairs carrying BOTH labels            : {len(both)}")
    print(f"  rows inside those pairs               : {int(both.n.sum())}")
    print(f"  -> for these rows, medium+genome are identical and only the")
    print(f"     condition columns can separate the labels")
    plot_pair_both_labels(pair)

    print("\n=== within those conflicted pairs, does temperature separate the labels? ===")
    key = df.set_index(["bacdive_id", "media_name"]).index
    conf = df[key.isin(both.index)].copy()
    if len(conf):
        print(conf.groupby("growth_class").temp_num.describe().to_string())
        # per-pair: is max temp of negatives outside [min,max] of positives?
        agg = conf.groupby(["bacdive_id", "media_name", "growth_class"]).temp_num.agg(
            ["min", "max"]).unstack("growth_class")
        agg.columns = [f"{a}_{b}" for a, b in agg.columns]
        ok = agg.dropna()
        if len(ok):
            outside = ((ok["min_0"] < ok["min_1"]) | (ok["max_0"] > ok["max_1"]))
            print(f"\n  conflicted pairs where the no-growth temps fall OUTSIDE the "
                  f"growth temp range: {int(outside.sum())} / {len(ok)} "
                  f"({100*outside.mean():.1f}%)")
            plot_neg_outside_growth_range(conf, ok)

    print("\n=== how many negatives are the ONLY label for their (strain, medium) pair? ===")
    solo_neg = pair[(pair.uniq == 1) & (pair.pos == 0)]
    print(f"  pairs that are negative-only          : {len(solo_neg)}")
    print(f"  rows in them                          : {int(solo_neg.n.sum())}")

    print("\n=== negatives: does the medium have a composition at all? ===")
    neg = df[df.growth_class == 0]
    print(f"  negatives with composition_raw        : {int((neg.composition_raw!='').sum())} / {len(neg)}")
    print(f"  negatives with mediadive_id           : {int((neg.mediadive_id!='').sum())} / {len(neg)}")

    print("\n=== can the label be predicted from the medium name ALONE? ===")
    m = df.groupby("media_name").growth_class.agg(n="size", pos="sum")
    m["rate"] = m.pos / m.n
    big = m[m.n >= 50].sort_values("rate")
    print(f"  media with >=50 rows: {len(big)}")
    print("\n-- most negative-skewed media --")
    print(big.head(12).to_string())
    print("\n-- baseline: overall positive rate --")
    print(f"  {df.growth_class.mean():.4f}   (always-predict-growth accuracy)")

    # A medium-only lookup baseline: predict the medium's majority label.
    pred = df.media_name.map((m.rate >= 0.5).to_dict())
    acc = (pred.astype(float) == df.growth_class).mean()
    print(f"\n  medium-name-only majority-label accuracy : {acc:.4f}")
    print("  (a model with NO genome input; the genome must beat this to matter)")

    # add temperature to the lookup
    df["tbin"] = pd.cut(df.temp_num, [-100, 10, 20, 25, 30, 35, 40, 50, 200])
    m2 = df.groupby(["media_name", "tbin"], observed=True).growth_class.agg(
        n="size", pos="sum")
    m2["maj"] = (m2.pos / m2.n >= 0.5)
    pred2 = df.set_index(["media_name", "tbin"]).index.map(m2["maj"].to_dict())
    acc2 = (pd.Series(pred2, index=df.index).astype(float) == df.growth_class).mean()
    print(f"  medium-name + temperature-bin accuracy   : {acc2:.4f}")


if __name__ == "__main__":
    main()
