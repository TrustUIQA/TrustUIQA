"""Cross-set random-forest fusion (mos/cross_eval.py fusion()) refit without the removed scores.

Same protocol as mos/cross_eval.py: features are the metrics shared by sets A and B, minus metrics trained on A or B,
300-tree forest on A's z-scored MOS, tested within scene on B's scenes whose photo is not in A.
Only change: the scores in common.DROP and the second UCIQE implementation are never features.
-> paper/iteration5/tables/fusion.csv. Run from the repo root: python -I paper/iteration5/scripts/fusion.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "mos"))
from common import DROP, IT, UCIQE_M  # noqa: E402
import cross_eval as ce  # noqa: E402

load = ce.load


def load_clean(ds):
    man, sc = load(ds)
    return man, {m: s for m, s in sc.items() if m not in DROP and m != UCIQE_M}


if __name__ == "__main__":
    ce.load = load_clean
    ce.REPO = Path(ce.REPO)
    out_csv = IT / "tables/fusion.csv"
    # cross_eval.fusion() writes mos/results/fusion.csv: run its loop but keep the original file untouched
    orig = (ce.REPO / "mos/results/fusion.csv").read_bytes()
    try:
        out = ce.fusion()
    finally:
        new = (ce.REPO / "mos/results/fusion.csv").read_bytes()
        (ce.REPO / "mos/results/fusion.csv").write_bytes(orig)
    out_csv.write_bytes(new)
    assert not set(out.best_single) & (DROP | {UCIQE_M})
    print(out.round(3).to_string())
