"""Leakage-safe RUOD splits + duplicate-free DUO evaluation set.

Needs results/dedup/groups.csv from scripts/find_duplicates.py (run on ruod+duo).
Writes results/splits/ruod_splits.csv (image, split, group, n_objects, classes)
and results/splits/duo_clean.csv (DUO images sharing no duplicate group with
any RUOD image).

    python scripts/make_splits.py
"""
import argparse

import pandas as pd

from uwiqa import RESULTS_ROOT
from uwiqa.data import load_manifest
from uwiqa.data.splits import RUOD_FRACTIONS, group_split


def class_table(df):
    rows = {}
    for s, g in df.groupby("split"):
        c = g["classes"].fillna("").str.split().explode().value_counts()
        rows[s] = c
    return pd.DataFrame(rows).fillna(0).astype(int).sort_index(key=lambda i: i.astype(int))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    groups = pd.read_csv(RESULTS_ROOT / "dedup" / "groups.csv")
    out = RESULTS_ROOT / "splits"
    out.mkdir(parents=True, exist_ok=True)

    ruod = groups[groups.dataset == "ruod"].merge(
        load_manifest("ruod")[["image", "n_objects", "classes"]], on="image")
    ruod = ruod.rename(columns={"split": "orig_split"})
    ruod["split"] = group_split(ruod["group"], RUOD_FRACTIONS, args.seed)
    assert ruod.groupby("group")["split"].nunique().max() == 1
    ruod[["image", "split", "orig_split", "group", "n_objects", "classes"]].to_csv(
        out / "ruod_splits.csv", index=False)

    print("RUOD leakage-safe split (images / objects):")
    print(ruod.groupby("split").agg(images=("image", "size"), objects=("n_objects", "sum"))
          .reindex(list(RUOD_FRACTIONS)))
    print("\nimages containing each class id, per split:")
    print(class_table(ruod)[list(RUOD_FRACTIONS)])

    duo = groups[groups.dataset == "duo"]
    ruod_groups = set(ruod["group"])
    clean = duo[~duo["group"].isin(ruod_groups)].merge(
        load_manifest("duo")[["image", "n_objects", "classes"]], on="image")
    clean = clean[clean.n_objects > 0]
    clean[["image", "split", "group", "n_objects", "classes"]].to_csv(out / "duo_clean.csv", index=False)
    print(f"\nDUO: {len(duo)} images, {int(duo['group'].isin(ruod_groups).sum())} share a "
          f"duplicate group with RUOD -> {len(clean)} clean labelled images kept for O5")


if __name__ == "__main__":
    main()
