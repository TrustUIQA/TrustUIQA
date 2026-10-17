"""Detection-aware NR predictor (O4): raw-image features -> per-method utility delta.

Targets, per image and enhancer m:
    delta_m = AP(enhanced_m) - AP(raw)                      (regression)
    help_m  = delta_m > tau                                  (classification)
tau is the 95th percentile of |delta| for the null JPEG re-encode on the
training split, i.e. "helps by more than label noise".
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from uwiqa import RESULTS_ROOT

NULL = "null_jpeg95"


# ------------------------------------------------------------------ data
def load_utility(set_name: str, metric: str = "ap", tag: str = "main"):
    sc = pd.read_csv(RESULTS_ROOT / "utility" / set_name / tag / "scores.csv")
    wide = sc.pivot_table(index="image", columns="method", values=metric)
    delta = wide.drop(columns="raw").sub(wide["raw"], axis=0)
    return wide, delta


def load_embeddings(set_name: str, backbone: str) -> pd.DataFrame:
    parts = []
    for f in sorted(glob.glob(str(RESULTS_ROOT / "features" / set_name / "emb_*.npz"))):
        z = np.load(f)
        names = [b for b in backbone.split("+")]
        X = np.concatenate([z[b] for b in names], axis=1)
        parts.append(pd.DataFrame(X, index=z["images"]))
    return pd.concat(parts)


def load_iqa(set_name: str) -> pd.DataFrame:
    """Wide table: index image, columns (metric, method)."""
    iqa = pd.read_csv(RESULTS_ROOT / "features" / set_name / "iqa.csv")
    return iqa.pivot_table(index="image", columns=["metric", "method"], values="score")


def noise_tau(delta_null: np.ndarray, q: float = 0.95) -> float:
    return float(np.quantile(np.abs(delta_null), q))


# ------------------------------------------------------------------ models
class Standardizer:
    def fit(self, X):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        return self

    def __call__(self, X):
        return (X - self.mu) / self.sd


def fit_ridge(Xtr, Dtr, Xva, Dva, alphas=(1, 10, 100, 1e3, 1e4)):
    best = None
    for a in alphas:
        m = Ridge(alpha=a).fit(Xtr, Dtr)
        s = mean_srcc(m.predict(Xva), Dva)
        if best is None or s > best[0]:
            best = (s, a, m)
    return best[2], best[1]


def fit_logistic(Xtr, Htr, Xva, Hva, Cs=(1e-3, 1e-2, 1e-1, 1)):
    models = []
    for j in range(Htr.shape[1]):
        best = None
        for C in Cs:
            if Htr[:, j].min() == Htr[:, j].max():
                break
            m = LogisticRegression(C=C, class_weight="balanced", max_iter=2000).fit(Xtr, Htr[:, j])
            s = safe_auc(Hva[:, j], m.predict_proba(Xva)[:, 1])
            if best is None or s > best[0]:
                best = (s, m)
        models.append(best[1] if best else None)
    return models


def predict_logistic(models, X):
    return np.stack([m.predict_proba(X)[:, 1] if m is not None else np.full(len(X), 0.5)
                     for m in models], 1)


class MLP(torch.nn.Module):
    def __init__(self, d, m, hidden=256, p=0.2):
        super().__init__()
        self.body = torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.GELU(), torch.nn.Dropout(p))
        self.reg = torch.nn.Linear(hidden, m)
        self.cls = torch.nn.Linear(hidden, m)

    def forward(self, x):
        h = self.body(x)
        return self.reg(h), self.cls(h)


def rank_loss(pred, target, margin):
    """Pairwise logistic loss on in-batch pairs whose true deltas differ by > margin."""
    dp = pred[:, None, :] - pred[None, :, :]
    dt = target[:, None, :] - target[None, :, :]
    mask = dt.abs() > margin
    if not mask.any():
        return pred.sum() * 0
    return torch.nn.functional.softplus(-torch.sign(dt[mask]) * dp[mask]).mean()


def fit_mlp(Xtr, Dtr, Htr, Xva, Dva, tau, rank_w=0.5, epochs=400, lr=1e-3, wd=1e-2,
            batch=256, patience=40, seed=0, device="cpu"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    sd = Dtr.std(0) + 1e-6
    t = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
    Xt, Dt, Ht, Xv = t(Xtr), t(Dtr / sd), t(Htr), t(Xva)
    pos = Ht.mean(0).clamp(1e-3, 1 - 1e-3)
    net = MLP(Xtr.shape[1], Dtr.shape[1]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, best_state, bad = -np.inf, None, 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xt), device=device)
        for i in range(0, len(Xt), batch):
            idx = perm[i:i + batch]
            r, c = net(Xt[idx])
            loss = torch.nn.functional.smooth_l1_loss(r, Dt[idx])
            loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(
                c, Ht[idx], pos_weight=(1 - pos) / pos)
            if rank_w:
                loss = loss + rank_w * rank_loss(r, Dt[idx], tau / torch.tensor(sd, device=device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            s = mean_srcc(net(Xv)[0].cpu().numpy(), Dva)
        if s > best:
            best, bad = s, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()

    def predict(X):
        with torch.no_grad():
            r, c = net(t(X))
        return r.cpu().numpy() * sd, torch.sigmoid(c).cpu().numpy()

    return predict


# ------------------------------------------------------------------ evaluation
def safe_auc(y, s):
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else np.nan


def mean_srcc(P, D):
    return float(np.nanmean([stats.spearmanr(P[:, j], D[:, j]).statistic for j in range(D.shape[1])]))


def best_threshold(y, s):
    """Threshold on score s maximising balanced accuracy (tuned on validation only)."""
    qs = np.unique(np.quantile(s, np.linspace(0.02, 0.98, 49)))
    accs = [balanced_accuracy_score(y, s > q) for q in qs]
    return float(qs[int(np.argmax(accs))])


def evaluate(D, H, dhat, score, thresholds, methods) -> pd.DataFrame:
    """Per-method SRCC/PLCC of dhat vs D (if dhat given) and AUROC/bal-acc/F1 of score vs H."""
    rows = []
    for j, m in enumerate(methods):
        r = {"method": m, "pos_rate": H[:, j].mean()}
        if dhat is not None:
            r["srcc"] = stats.spearmanr(dhat[:, j], D[:, j]).statistic
            r["plcc"] = stats.pearsonr(dhat[:, j], D[:, j]).statistic if dhat[:, j].std() > 0 else np.nan
        r["auroc"] = safe_auc(H[:, j], score[:, j])
        yhat = score[:, j] > thresholds[j]
        r["bal_acc"] = balanced_accuracy_score(H[:, j], yhat)
        r["f1"] = f1_score(H[:, j], yhat, zero_division=0)
        rows.append(r)
    return pd.DataFrame(rows)
