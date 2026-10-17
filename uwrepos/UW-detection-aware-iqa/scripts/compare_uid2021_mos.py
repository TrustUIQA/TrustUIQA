"""Is the gap to the UID2021 paper caused by the MOS update (52 -> 77 observers)?

Downloads the release linked from the UID2021 README until Dec 2022 (likely the
original 52-observer MOS used in the paper), aligns its MOS with ours by image
name, and recomputes SROCC of every scored metric against both MOS versions.
If UIQM/UCIQE/NIQE approach the paper's Table 9 (0.540 / 0.603 / 0.330) with
the old MOS, our pipeline is right and the difference is the MOS revision.

    python scripts/compare_uid2021_mos.py
"""
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

from uwiqa import DATA_ROOT, RESULTS_ROOT
from uwiqa.data import load_manifest

sys.path.insert(0, str(Path(__file__).parent))
from get_uid2021 import IMG_EXTS, URL_2022, fetch  # noqa: E402

PAPER_T9 = {"uciqe": 0.6030, "uiqm": 0.5404, "niqe": 0.3304}
stem = lambda s: Path(str(s)).stem

old_root = DATA_ROOT / "UID2021_2022"
if not old_root.exists() or not any(old_root.iterdir()):
    try:
        n = fetch(URL_2022, old_root)
        print(f"old release unpacked to {old_root}: {n} images")
    except Exception as e:
        sys.exit(f"Could not download the 2022 release ({e}). The link may no longer be public; "
                 "report the MOS-version caveat instead.")

tables = sorted(p for p in old_root.rglob("*") if p.suffix.lower() in {".xlsx", ".xls", ".csv", ".txt", ".mat"})
print("tables in the old release:", [str(t.relative_to(old_root)) for t in tables])
old = None
for t in tables:
    if t.suffix.lower() not in {".xlsx", ".xls", ".csv"}:
        continue
    sheets = pd.read_excel(t, sheet_name=None) if t.suffix.lower() != ".csv" else {"csv": pd.read_csv(t)}
    for name, df in sheets.items():
        print(f"  {t.name}/{name}: columns {list(df.columns)}, shape {df.shape}")
        ncol = next((c for c in df.columns if "name" in str(c).lower() or "image" in str(c).lower()), None)
        mcol = next((c for c in df.columns if "mos" in str(c).lower()), None)
        if old is None and ncol is not None and mcol is not None:
            old = df[[ncol, mcol]].rename(columns={ncol: "name", mcol: "mos_2022"})
if old is None:
    sys.exit("No table with a name column and a MOS column found in the old release; inspect it manually.")
old["stem"] = old["name"].map(stem)

n_old_imgs = sum(1 for p in old_root.rglob("*") if p.suffix.lower() in IMG_EXTS)
print(f"old release: {len(old)} MOS rows, {n_old_imgs} images")

man = load_manifest("uid2021")
man["stem"] = man["image"].map(stem)
d = man.merge(old[["stem", "mos_2022"]], on="stem", how="inner")
print(f"matched {len(d)}/{len(man)} images by name")
print(f"MOS 2022 vs current: SRCC {spearmanr(d['mos'], d['mos_2022']).statistic:.3f}, "
      f"mean |diff| {abs(d['mos'] - d['mos_2022']).mean():.3f}")

scores = pd.read_csv(RESULTS_ROOT / "scores" / "uid2021.csv")
hb = scores.groupby("metric")["higher_better"].first().astype(bool)
wide = scores.pivot_table(index="image", columns="metric", values="score")
d = d.merge(wide, left_on="image", right_index=True)
rows = []
for m in wide.columns:
    sgn = 1 if hb[m] else -1
    rows.append({"metric": m,
                 "srcc_current_mos": sgn * spearmanr(d[m], d["mos"]).statistic,
                 "srcc_2022_mos": sgn * spearmanr(d[m], d["mos_2022"]).statistic,
                 "paper_table9": PAPER_T9.get(m)})
out = pd.DataFrame(rows).sort_values("srcc_2022_mos", ascending=False)
(RESULTS_ROOT / "tables").mkdir(parents=True, exist_ok=True)
out.to_csv(RESULTS_ROOT / "tables" / "uid2021_mos_versions.csv", index=False)
print(out.round(3).to_string(index=False))
