"""Score every image of a MOS manifest (mos/manifests/<ds>.csv) with NR metrics.

One CSV per metric in mos/scores/<ds>/<metric>.csv (id,score,lower_better); finished metrics are skipped,
so reruns resume. Reuses score.py's metric loading. Run from the repo root:
  python -I mos/score_mos.py uid2021 [--metrics ...]   # visualquality_r1 runs only when named (slow 7B VLM)
  python -I mos/score_mos.py out_ablate --pool [--metrics ...]   # paper pools -> out_ablate/scores_ext/
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyiqa
import torch
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "paper/iteration1/scripts")]  # -I drops cwd from sys.path
from cpqi import LAM, cp  # noqa: E402
from score import UW, crop32, patch_qalign, rss_guard  # noqa: E402
from uwmetrics import _uicm, _uiconm, _uism  # noqa: E402
from mos.uw_extra import EXTRA, _rgb, nuiq_features, nuiq_scores  # noqa: E402

# UIQM's three sub-scores (Abhinav's supplementary reports them separately) and CPQI's colour term
UW = UW | {"uicm": _uicm, "uism": _uism, "uiconm": _uiconm, "cp": cp}
LOWER = {"cp"}  # hand-crafted metrics where lower = better

# underwater metrics, then every pyiqa NR model that scores a single natural image (face / caption / distribution
# metrics left out). nuiq ranks the images of one scene only (scores are not comparable across scenes).
METRICS = """uiqm uciqe uicm uism uiconm cp fdum uiqavsi twicemix nr_uiqa quiq nuiq
entropy brisque brisque_matlab niqe niqe_matlab ilniqe piqe nrqm pi
cnniqa dbcnn hyperiqa metaiqa wadiqam_nr paq2piq unique
musiq musiq-ava musiq-paq2piq musiq-spaq nima nima-koniq nima-spaq nima-vgg16-ava
arniqa arniqa-clive arniqa-csiq arniqa-flive arniqa-kadid arniqa-live arniqa-spaq arniqa-tid
clipiqa clipiqa+ clipiqa+_rn50_512 clipiqa+_vitL14_512 maclip
qualiclip qualiclip+ qualiclip+-clive qualiclip+-flive qualiclip+-spaq
liqe liqe_mix maniqa maniqa-kadid maniqa-pipal tres tres-flive
topiq_nr topiq_nr-flive topiq_nr-spaq topiq_iaa topiq_iaa_res50 laion_aes afine_nr
uranker qalign_8bit""".split()

def min224(p):
    """LIQE asserts a short side >= 224 (some SAUD images are smaller): upsample only those, bicubic, [1,3,H,W]."""
    x = cv2.imread(str(p))
    s = 224 / min(x.shape[:2])
    if s <= 1:
        return p
    x = cv2.resize(x, (int(np.ceil(x.shape[1] * s)), int(np.ceil(x.shape[0] * s))), interpolation=cv2.INTER_CUBIC)
    return torch.from_numpy(cv2.cvtColor(x, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)[None].float() / 255


def save(dst, ids, scores, lower_better, pool):
    """MOS sets: id,score,lower_better. Pools: score.py's mode,image,step,metric,score,lower_better."""
    d = pd.DataFrame(dict(id=list(ids), score=scores, lower_better=lower_better))
    if pool:
        d[["mode", "image", "step"]] = d.id.str.split("/", expand=True)
        d = d.assign(step=d.step.astype(int), metric=dst.stem)[["mode", "image", "step", "metric", "score",
                                                                 "lower_better"]]
    d.to_csv(dst, index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ds", help="manifest name in mos/manifests, or with --pool a score.py image pool (e.g. out_ablate)")
    ap.add_argument("--pool", action="store_true",
                    help="score <ds>/<mode>/<image>/<step>.png into <ds>/scores_ext/ (the pool's scores/ is untouched)")
    ap.add_argument("--metrics", nargs="+", default=METRICS)
    a = ap.parse_args()
    rss_guard()
    if a.pool:  # nuiq's scene = the source photo, so it ranks every version of one image
        paths = sorted((REPO / a.ds).glob("*/*/*.png"))
        man = pd.DataFrame(dict(id=[f"{p.parts[-3]}/{p.parts[-2]}/{p.stem}" for p in paths],
                                path=[str(p) for p in paths], scene=[p.parts[-2] for p in paths]))
        out = REPO / a.ds / "scores_ext"
    else:
        man = pd.read_csv(REPO / "mos/manifests" / f"{a.ds}.csv")
        out = REPO / "mos/scores" / a.ds
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for m in a.metrics:
        dst = out / f"{m}.csv"
        if dst.exists():
            continue
        if m == "nuiq":
            feats = [nuiq_features(_rgb(p)) for p in tqdm(man.path, desc=m, mininterval=30)]
            save(dst, man.id, nuiq_scores(np.stack(feats), man.scene), False, a.pool)
            print(f"done {m} ({len(man)})", flush=True)
            continue
        if m == "visualquality_r1":  # 7B VLM (vqr1_score.py), batches of 8; one at a time if a batch OOMs
            from vqr1_score import load
            vq, scores = load(), []
            if "w" in man:
                px = man.w * man.h
            else:  # edit manifests carry no size: read it from the file header
                from PIL import Image
                px = pd.Series([np.prod(Image.open(p).size) for p in man.path])
            for i in tqdm(range(0, len(man), 8), desc=m, mininterval=30):
                b = list(man.path[i:i + 8])
                # big images (SAUD/SOTA, up to 2.4 MP): a batch of 8 spills into host RAM on WSL and crawls, no OOM
                if px[i:i + 8].max() > 600_000:
                    scores += [vq([p])[0][0] for p in b]
                    continue
                try:
                    scores += vq(b)[0]
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    scores += [vq([p])[0][0] for p in b]
            save(dst, man.id, scores, False, a.pool)
            print(f"done {m} ({len(scores)}, unparsed {int(np.isnan(scores).sum())})", flush=True)
            continue
        if m in EXTRA:
            f, lower_better = EXTRA[m], False
        elif m in UW:
            # float: the UIQM sub-scores do r - g and max + min, which wrap around on uint8
            f, lower_better = (lambda p, fn=UW[m]: fn(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB).astype(
                np.float64))), m in LOWER
        else:
            try:
                if m.startswith("qalign"):
                    patch_qalign()
                f = pyiqa.create_metric(m, device=dev)
            except Exception as e:  # unknown name / weights download failure: report, keep going
                print(f"skip {m}: {e}", flush=True)
                continue
            lower_better = f.lower_better
            if m.startswith("afine"):
                f = (lambda p, g=f: g(crop32(p)))
            elif m.startswith(("liqe", "maclip")):  # maclip unfolds 224 px patches: same short-side limit
                f = (lambda p, g=f: g(min224(p)))
        scores = []
        try:
            with torch.no_grad():
                for p in tqdm(man.path, desc=m, mininterval=30):
                    scores.append(float(f(p)))
        except Exception as e:  # e.g. CUDA OOM on one model: keep the rest of the run alive
            print(f"fail {m}: {type(e).__name__}: {str(e)[:200]}", flush=True)
            continue
        finally:
            del f
            torch.cuda.empty_cache()
        save(dst, man.id, scores, lower_better, a.pool)
        print(f"done {m} ({len(scores)})", flush=True)
    # CPQI = PI + LAM * CP (Rajput et al., CVPRW 2026), lower = better (pools already have scores_cpqi/)
    if not a.pool and (out / "pi.csv").exists() and (out / "cp.csv").exists() and not (out / "cpqi.csv").exists():
        pi, c = (pd.read_csv(out / f"{k}.csv").set_index("id").score for k in ("pi", "cp"))
        pd.DataFrame(dict(id=pi.index, score=pi + LAM * c.loc[pi.index], lower_better=True)).to_csv(
            out / "cpqi.csv", index=False)
        print("done cpqi", flush=True)
