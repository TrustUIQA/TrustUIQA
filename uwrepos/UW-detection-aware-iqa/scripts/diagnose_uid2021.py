"""Week-4 gate debugging: why are our UID2021 correlations below the paper's?

Paper (Hou et al., ACM TOMM 2023, Table 9, entire UID2021, 52-observer MOS):
SROCC UCIQE 0.6030, UIQM 0.5404, NIQE 0.3304. Table 8 gives per-subset values.

Checks:
 1. every sheet/column of the MOS workbook (is there more than one MOS?)
 2. image <-> MOS alignment: our parsed scene/method vs the table's own columns
 3. MOS distribution, per-method means, raw images
 4. image modes/sizes (RGBA, grayscale, 16-bit?)
 5. per-subset SROCC of UIQM/UCIQE/NIQE next to the paper's Table 8, for every
    MOS-like column found

    python scripts/diagnose_uid2021.py
"""
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr

from uwiqa import RESULTS_ROOT
from uwiqa.data import dataset_root, load_config, load_manifest

PAPER_T8 = {  # subset: (UCIQE, UIQM, NIQE) SROCC from Table 8
    "B": (0.6182, 0.5393, 0.2828), "BG": (0.5434, 0.5042, 0.3224), "G": (0.6668, 0.5864, 0.2507),
    "H": (0.6237, 0.4291, 0.3442), "LL": (0.6650, 0.4928, 0.2618), "T": (0.5599, 0.7156, 0.5473),
}
PAPER_T9 = {"uciqe": 0.6030, "uiqm": 0.5404, "niqe": 0.3304}

root = dataset_root("uid2021")
cfg = load_config()["uid2021"]
book = pd.read_excel(root / cfg["table"], sheet_name=None)
print("== 1. workbook sheets")
for name, df in book.items():
    print(f"  sheet {name!r}: shape {df.shape}; columns {list(df.columns)}")
tab = list(book.values())[0]
print(tab.describe(include="all").T.to_string())

print("\n== 2. alignment")
man = load_manifest("uid2021")
tab["stem"] = tab[cfg["image_col"]].astype(str).str.replace(r"\.\w+$", "", regex=True)
man["stem"] = man["image"].str.split("/").str[-1].str.replace(r"\.\w+$", "", regex=True)
m = man.merge(tab, on="stem", how="outer", indicator=True)
print("  merge:", m["_merge"].value_counts().to_dict(), "| duplicate names in table:",
      int(tab["stem"].duplicated().sum()))
if "Method" in tab and "Scene" in tab:
    both = m[m["_merge"] == "both"]
    print("  method mismatches:", int((both["method"] != both["Method"].astype(str)).sum()),
          "| scene mismatches:", int((both["scene"] != both["Scene"].astype(str)).sum()))
    print("  example rows:\n", both[["image", "method", "Method", "scene", "Scene", "mos"]].head(3).to_string())

print("\n== 3. MOS")
print("  overall:", man["mos"].describe().round(3).to_dict())
print("  per-method mean MOS (sorted):")
print(man.groupby("method")["mos"].agg(["mean", "std", "count"]).sort_values("mean").round(3).to_string())

print("\n== 4. image formats (sample of 120)")
fmt = pd.Series([f"{Image.open(root / p).mode} {Image.open(root / p).size}"
                 for p in man["image"].sample(min(120, len(man)), random_state=0)]).value_counts()
print(fmt.head(10).to_string())

print("\n== 5. per-subset SROCC vs paper Table 8")
scores = pd.read_csv(RESULTS_ROOT / "scores" / "uid2021.csv")
wide = scores.pivot_table(index="image", columns="metric", values="score")
d = man.merge(wide, left_on="image", right_index=True)
d["subset"] = d["scene"].str.split("_").str[0]
mos_cols = [c for c in tab.columns if "mos" in str(c).lower() or "score" in str(c).lower()]
for mc in mos_cols:
    dd = d.merge(tab[["stem", mc]].rename(columns={mc: "_mos"}), on="stem")
    print(f"\n  using MOS column {mc!r}")
    rows = []
    for sub, g in dd.groupby("subset"):
        r = {"subset": sub, "n": len(g)}
        pu, pq, pn = PAPER_T8.get(sub, (np.nan,) * 3)
        r.update({"uciqe": spearmanr(g["uciqe"], g["_mos"]).statistic, "paper_uciqe": pu,
                  "uiqm": spearmanr(g["uiqm"], g["_mos"]).statistic, "paper_uiqm": pq,
                  "niqe": -spearmanr(g["niqe"], g["_mos"]).statistic, "paper_niqe": pn})
        rows.append(r)
    allr = {"subset": "ALL", "n": len(dd),
            "uciqe": spearmanr(dd["uciqe"], dd["_mos"]).statistic, "paper_uciqe": PAPER_T9["uciqe"],
            "uiqm": spearmanr(dd["uiqm"], dd["_mos"]).statistic, "paper_uiqm": PAPER_T9["uiqm"],
            "niqe": -spearmanr(dd["niqe"], dd["_mos"]).statistic, "paper_niqe": PAPER_T9["niqe"]}
    print(pd.DataFrame(rows + [allr]).round(3).to_string(index=False))
    print("  enhanced-only (no raw) ALL: uciqe %.3f  uiqm %.3f" % (
        spearmanr(dd.loc[dd.method != "raw", "uciqe"], dd.loc[dd.method != "raw", "_mos"]).statistic,
        spearmanr(dd.loc[dd.method != "raw", "uiqm"], dd.loc[dd.method != "raw", "_mos"]).statistic))
