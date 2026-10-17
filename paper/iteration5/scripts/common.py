"""Shared loaders for iteration 5. Every number in the paper comes from the raw result files read here.

Run every script from the repo root with the uwiqa env: python -I paper/iteration5/scripts/<name>.py
Authors' rule (Abhinav, 14 Oct): the two scores in DROP are never shown, counted or cited.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
IT = REPO / "paper/iteration5"
R = REPO / "mos/results"
DROP = {"cp", "cpqi"}
UCIQE_M = "uciqe_matlab_fix"  # patched MATLAB UCIQE: used only in the implementation comparison
SETS = ["uid2021", "saud", "saud2", "sota"]
REFS = SETS + ["study"]
REF_NAME = {"uid2021": "UID2021", "saud": "SAUD", "saud2": "SAUD2.0", "sota": "SOTA", "study": "Our study"}
CHAINS = {"replica": "Pro replica", "nb2_replica": "NB2 replica", "enhance": "Pro enhance", "nb2_enhance": "NB2 enhance"}

HAND_UW = ["uiqm", "uicm", "uism", "uiconm", "uciqe", "ccf", "fdum"]
LEARNED_UW = ["uranker", "nuiq", "uiqavsi", "nr_uiqa", "twicemix", "quiq", "nmfc"]
VLM = ["qalign_8bit", "visualquality_r1"]
TRAINED = {"uiqavsi": "saud", "nr_uiqa": "saud", "nuiq": "saud", "nmfc": "saud", "quiq": "uid2021"}  # mos/cross_eval.py
GROUP_OF = {"saud": "saud", "saud2": "saud", "uid2021": "uid2021", "sota": "sota", "study": "study"}  # same lab = same group

NAME = {
    "afine_nr": "A-FINE", "brisque": "BRISQUE", "brisque_matlab": "BRISQUE (MATLAB)", "ccf": "CCF",
    "clipiqa": "CLIP-IQA", "clipiqa+": "CLIP-IQA+", "clipiqa+_rn50_512": "CLIP-IQA+ RN50", "clipiqa+_vitL14_512": "CLIP-IQA+ ViT-L",
    "cnniqa": "CNNIQA", "dbcnn": "DBCNN", "entropy": "Entropy", "fdum": "FDUM", "hyperiqa": "HyperIQA", "ilniqe": "IL-NIQE",
    "laion_aes": "LAION-Aes", "liqe": "LIQE", "liqe_mix": "LIQE-mix", "maclip": "MA-CLIP", "metaiqa": "MetaIQA",
    "niqe": "NIQE", "niqe_matlab": "NIQE (MATLAB)", "nmfc": "NMFC", "nr_uiqa": "NR-UIQA", "nrqm": "NRQM", "nuiq": "NUIQ",
    "paq2piq": "PaQ-2-PiQ", "pi": "PI", "piqe": "PIQE", "qalign_8bit": "Q-Align", "quiq": "QUIQ", "rali": "RALI",
    "topiq_iaa": "TOPIQ-IAA", "topiq_iaa_res50": "TOPIQ-IAA R50", "twicemix": "Twice-Mixing", "uciqe": "UCIQE",
    UCIQE_M: "UCIQE (MATLAB code)", "uicm": "UICM", "uiconm": "UIConM", "uiqavsi": "UIQAVSI", "uiqm": "UIQM", "uism": "UISM",
    "unique": "UNIQUE", "uranker": "URanker", "visualquality_r1": "VisualQuality-R1", "wadiqam_nr": "WaDIQaM-NR",
}
BASE = {"arniqa": "ARNIQA", "maniqa": "MANIQA", "musiq": "MUSIQ", "nima": "NIMA", "qualiclip+": "QualiCLIP+",
        "qualiclip": "QualiCLIP", "topiq_nr": "TOPIQ-NR", "tres": "TReS"}


def name(m):
    """Display name: checkpoint variants keep their training set, e.g. arniqa-tid -> ARNIQA-TID."""
    if m in NAME:
        return NAME[m]
    b, _, v = m.partition("-")
    if b in BASE:
        return BASE[b] + ("-" + v.upper().replace("PAQ2PIQ", "PaQ2PiQ").replace("VGG16", "VGG16") if v else "")
    raise KeyError(m)


# bib keys (scholar.bib) for table labels, by metric base name or reference set
CITE = {
    "uiqm": "panetta2015human", "uicm": "panetta2015human", "uism": "panetta2015human", "uiconm": "panetta2015human",
    "uciqe": "yang2015underwater", UCIQE_M: "wong2020uciqecode", "ccf": "wang2018imaging", "fdum": "yang2021reference",
    "uranker": "guo2023underwater", "nuiq": "jiang2022underwater", "uiqavsi": "chen2025underwater", "nr_uiqa": "yi2024no",
    "twicemix": "fu2022twice", "quiq": "karimi2025quiq", "nmfc": "chen2023underwater", "qalign_8bit": "wu2023q",
    "visualquality_r1": "wu2026visualquality", "rali": "zhao2026reasoning", "afine_nr": "chen2025toward",
    "brisque": "mittal2012no", "brisque_matlab": "mittal2012no", "cnniqa": "kang2014convolutional", "dbcnn": "zhang2018blind",
    "hyperiqa": "su2020blindly", "ilniqe": "zhang2015feature", "laion_aes": "schuhmann2024christophschuhmann",
    "liqe": "zhang2023blind", "liqe_mix": "zhang2023blind", "maclip": "liao2026beyond", "metaiqa": "zhu2020metaiqa",
    "niqe": "mittal2012making", "niqe_matlab": "mittal2012making", "nrqm": "ma2017learning", "paq2piq": "ying2020patches",
    "pi": "blau20182018", "piqe": "venkatanath2015blind", "unique": "zhang2021uncertainty", "wadiqam_nr": "bosse2017deep",
    "clipiqa": "wang2023exploring", "topiq": "chen2024topiq", "arniqa": "agnolucci2024arniqa", "maniqa": "yang2022maniqa",
    "musiq": "ke2021musiq", "nima": "talebi2018nima", "qualiclip": "agnolucci2024quality", "tres": "golestaneh2022no",
    "uid2021": "hou2023uid2021", "saud": "jiang2022underwater", "saud2": "jiang2024toward", "sota": "wang2022generation",
}


def _cite(k):
    return f"~\\cite{{{k}}}" if k else ""


def mcite(m):
    """Table label: display name plus its citation (entropy has none)."""
    b = m.partition("-")[0]
    return name(m) + _cite(CITE.get(m) or CITE.get(b) or CITE.get(re.sub(r"(\+.*|_.*)$", "", b)))


def cref(r):
    """Reference set label plus its citation (our study has none)."""
    return REF_NAME[r] + _cite(CITE.get(r))


def group(m):
    return "hand-crafted underwater" if m in HAND_UW else "learned underwater" if m in LEARNED_UW else \
        "vision-language" if m in VLM else "general-purpose"


def clean(idx):
    """Drop the removed scores and the second UCIQE implementation from a metric index."""
    return [m for m in idx if m not in DROP and m != UCIQE_M and not m.endswith("_gt")]


def kappa_table():
    """metric x ref within-scene Cohen's kappa with 95% CI columns (kappa, lo, hi) as a dict of DataFrames."""
    out = {}
    for s in SETS:
        k = pd.concat([pd.read_csv(R / f"{s}.csv"), pd.read_csv(R / f"extra_{s}.csv")]).drop_duplicates("metric")
        out[s] = k.set_index("metric")[["kappa", "kappa_lo", "kappa_hi", "n"]]
    st = pd.read_csv(R / "study_kappa_all.csv").set_index("metric")
    out["study"] = st[["kappa", "kappa_lo", "kappa_hi", "n"]]
    return out


def kappa_wide(col="kappa", with_matlab=False):
    k = kappa_table()
    w = pd.DataFrame({r: k[r][col] for r in REFS})
    keep = [m for m in w.index if m not in DROP and not m.endswith("_gt") and (with_matlab or m != UCIQE_M)]
    return w.loc[keep]


def drift(chain):
    """drift_all.csv (Banana100 rule, banana_drift.py) and per-image pass table, removed scores dropped."""
    d = pd.read_csv(REPO / f"out_banana_{chain}/drift_all.csv", index_col=0).drop(list(DROP), errors="ignore")
    p = pd.read_csv(REPO / f"out_banana_{chain}/drift_per_image.csv")
    p = p[~p.metric.isin(DROP)]
    return d, p


def drift_scores(chain, metric):
    """Per-image, per-step raw scores of one metric on one chain (scores/ or scores_ext/)."""
    for sub in ("scores", "scores_ext"):
        f = REPO / f"out_banana_{chain}/{sub}/{metric}.csv"
        if f.exists():
            return pd.read_csv(f, dtype={"image": str})
    raise FileNotFoundError(metric)


def last_delta(d):
    return [c for c in d.columns if re.fullmatch(r"d\d+", c)][-1]


def pref_ci():
    """Per edit of our study: share of votes preferring the edit, 95% interval resampling raters and photos (study/pref_cluster_ci.py)."""
    return pd.read_csv(R / "study_pref_ci.csv").set_index("condition")


if __name__ == "__main__":
    assert name("arniqa-tid") == "ARNIQA-TID" and name("uciqe") == "UCIQE"
    w = kappa_wide()
    print(len(w), "metrics,", w.notna().all(axis=1).sum(), "on all five references")
    missing = [m for m in w.index if m not in NAME and m.partition("-")[0] not in BASE]
    assert not missing, missing
    print("ok")
