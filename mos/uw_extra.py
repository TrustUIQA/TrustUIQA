"""Underwater-specific NR metrics that pyiqa lacks, as Python callables taking an image path (higher = better).

  fdum     Yang et al., SPIC 2021 (rmcong/Code-for-FDUM-method): Python port of the authors' MATLAB, checked
           against the MATLAB outputs saved in their Value/value.mat (see selfcheck).
  uiqavsi  Chen et al. 2023 (dart-into/UIQAVSI): authors' network + SAUD-trained weights, their demo's preprocessing.
           Trained on SAUD, so its SAUD score is in-sample.
  twicemix Fu et al., SPIC 2022: authors' VGG16 ranker and weights.
  nr_uiqa  Yi, Jiang & Zhou, Displays 2024: port of the 42 MATLAB features (match the released feature files) + their
           SAUD-trained SVR; in-sample on SAUD.
  quiq     QUIQ repo: 32-feature port of their notebook + their UID2021-trained GPR (pickles via a whitelist
           unpickler); in-sample on UID2021.
  nuiq     Jiang et al., TCSVT 2022: SVM-rank on 59 features, ranks images within one scene only; 49/59 features
           reproduce the authors' training features (see nuiq_features). In-sample on SAUD.
  ccf      Wang et al. 2018: colourfulness + block contrast + FADE fog density (PyFADE), linear model.
  nmfc     dart-into/NMFC: port of the 84 MATLAB features (match the released SAUD features, r >= 0.95) + eps-SVR
           rebuilt from their released SAUD features with their SVM.m protocol (no model released); in-sample on SAUD.

Third-party code/weights stay in the gitignored uwrepos/ (untrusted): weights load with weights_only=True, network.py was read
before use and is loaded by file path.
"""
import importlib.util
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import ndimage
from scipy.fft import dctn
from skimage.color import rgb2lab
from skimage.feature import canny

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uwmetrics import _uism  # noqa: E402

REPOS = Path(__file__).resolve().parents[1] / "uwrepos"


def _rgb(p):
    return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)


def _rgb2gray_u8(x):
    """MATLAB rgb2gray on uint8: rounded luma."""
    return np.round(x.astype(np.float64) @ [0.298936021293775, 0.587043074451121, 0.114020904255103]).clip(0, 255)


# ---------------------------------------------------------------- FDUM
def _fdum_colour(x):
    y1 = np.round(dctn(_rgb2gray_u8(x), norm="ortho")).clip(0, 255)  # uint8(dct2(gray))
    d = np.round(y1 - y1.mean()).clip(0, 255)  # uint8 minus double saturates at 0 ...
    fd = np.sqrt(np.minimum(d ** 2, 255).sum() / d.size)  # ... and .^2 saturates at 255 in MATLAB
    # makecform('srgb2lab') on uint8 returns uint8-encoded Lab (a, b offset by 128), D50 adapted white.
    # ponytail: skimage has no Bradford adaptation, so D50 here is an approximation of makecform.
    lab = rgb2lab(x, illuminant="D50")
    a, b = (np.round(lab[..., i] + 128).clip(0, 255) / 255 for i in (1, 2))
    chr_ = np.hypot(a, b).ravel()
    td = np.sqrt(np.mean(np.abs(1 - (chr_.mean() / chr_) ** 2)))
    return fd * td


def _block_contrast(a, rb=64, T=0.002):
    """CCF's contrast: RMS contrast summed over the 64x64 blocks that have enough Canny edges, / number of blocks."""
    m, n = a.shape
    stds = []
    for i in range(m // rb):
        for j in range(n // rb):
            blk = a[i * rb:(i + 1) * rb, j * rb:(j + 1) * rb]
            # MATLAB edge(...,'canny'): sigma sqrt(2), high threshold at the 70th percentile, low = 0.4 * high
            e = canny(blk, sigma=np.sqrt(2), low_threshold=0.28, high_threshold=0.7, use_quantiles=True)
            if e.sum() > rb * rb * T:
                stds.append(blk.std())
    return sum(stds) / ((m // rb) * (n // rb))


def _fdum_contrast(x, w=15):
    i1 = x.astype(np.float64) / 255
    # Contrast_CCF indexes the 3-D image with 2 subscripts, i.e. on R|G|B laid side by side
    ccf = _block_contrast(np.concatenate([i1[..., 0], i1[..., 1], i1[..., 2]], axis=1))
    dark = ndimage.minimum_filter(i1.min(axis=2), size=w)[w // 2: w // 2 + i1.shape[0] - w, w // 2: w // 2 + i1.shape[1] - w]
    with np.errstate(invalid="ignore"):  # flat dark channel: 0/0, and MATLAB's uint8(NaN) is 0
        y1 = np.round(np.nan_to_num((dark - dark.min()) / (dark.max() - dark.min())) * 255)
    return ccf * np.exp(-y1.mean() / 100)


def fdum_parts(x):
    # ponytail: the authors' sharpness term calls UIQM.p (encrypted p-code); ours is UIQM's UISM from uwmetrics.py
    return _fdum_colour(x), _fdum_contrast(x), _uism(x.astype(np.float64))


def fdum(p):
    c, k, s = fdum_parts(_rgb(p))
    return 0.2982 * c + 0.4439 * k + 0.028 * s


# ---------------------------------------------------------------- UIQAVSI
@lru_cache(1)
def _uiqavsi_model():
    spec = importlib.util.spec_from_file_location("uiqavsi_network", REPOS / "UIQAVSI/network.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    net = mod.ThreeStreamIQA()
    net.load_state_dict(torch.load(REPOS / "UIQAVSI/data/UIQAVSI_model_SAUD.pth", map_location="cpu", weights_only=True))
    return net.eval().to("cuda" if torch.cuda.is_available() else "cpu")


def _patches(x, s=32):
    """CropPatches(image, 32, 32): non-overlapping 32x32 tiles, last row/column of tiles dropped as in the demo."""
    h, w = x.shape[:2]
    return np.stack([x[i:i + s, j:j + s] for i in range(0, h - s, s) for j in range(0, w - s, s)])


@torch.no_grad()
def uiqavsi(p):
    bgr = cv2.imread(str(p))
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32) / 255  # ToPILImage(uint8) -> to_tensor
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gx, gy = (cv2.convertScaleAbs(cv2.Sobel(rgb, cv2.CV_32F, *d)) for d in ((1, 0), (0, 1)))
    gra = cv2.addWeighted(gx, 0.5, gy, 0.5, 0).astype(np.float32) / 255
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    mu = cv2.GaussianBlur(g, (7, 7), 7 / 6, borderType=cv2.BORDER_CONSTANT)
    sd = np.sqrt(np.abs(cv2.GaussianBlur(g * g, (7, 7), 7 / 6, borderType=cv2.BORDER_CONSTANT) - mu * mu))
    mscn = ((g - mu) / (sd + 1)).astype(np.float32)[..., None]  # PIL mode 'F': to_tensor keeps raw values
    net = _uiqavsi_model()
    dev = next(net.parameters()).device
    t = [torch.from_numpy(_patches(a)).permute(0, 3, 1, 2).to(dev) for a in (lab, gra, mscn)]
    return net(t).mean().item()


# ---------------------------------------------------------------- NR-UIQA (Yi, Jiang & Zhou, Displays 2024)
def _conv_same(a, k, axis):
    """MATLAB conv2(a, k, 'same') with a 1-D kernel along `axis` (1 = row vector k, 0 = k')."""
    from scipy.signal import convolve2d
    f = convolve2d(a, k[None] if axis == 1 else k[:, None], mode="full")
    s = int(np.ceil((len(k) - 1) / 2))  # MATLAB 'same' keeps full[ceil((L-1)/2) + (0..n-1)]; matters for even L
    return np.take(f, np.arange(s, s + a.shape[axis]), axis=axis)


def _ce(x, sigma=3.25, semi=0.1, border=20):
    """CE.m: perceived contrast energy maps for gray, blue-yellow, red-green (Groen et al.)."""
    t = np.array([9.225496406318721e-4, 8.969246659629488e-4, 2.069284034165411e-4]) * 255
    xs = np.arange(-3 * sigma, 3 * sigma + 1e-9, 1.0)  # -9.75 : 1 : 9.25 (20 taps, as MATLAB's colon gives)
    g = np.exp(xs ** 2 / (-2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)
    g /= g.sum()
    gx = (xs ** 2 / sigma ** 4 - 1 / sigma ** 2) * g
    gx -= gx.sum() / len(xs)
    gx /= np.sum(0.5 * xs * xs * gx)
    r, gg, b = (x[..., i] for i in range(3))
    uc, dc = border // 2, border // 2 - 1
    out = []
    for ch, th in zip((0.299 * r + 0.587 * gg + 0.114 * b, 0.5 * r + 0.5 * gg - b, r - gg), t):
        p = np.vstack([ch[:uc], ch, ch[-dc - 1:]])  # border_in: copy (not mirror) edge rows/cols
        p = np.hstack([p[:, :uc], p, p[:, -dc - 1:]])
        c = np.hypot(_conv_same(p, gx, 1), _conv_same(p, gx, 0))[uc:-dc - 1, uc:-dc - 1]  # border_out
        rr = c * c.max() / (c + c.max() * semi) if c.max() > 0 else c  # flat channel (e.g. grey image): no energy
        out.append(np.where(rr - th > 1e-7, rr - th, 0.0))
    return out  # gray, by, rg


def _entropy_u8(a):
    """MATLAB entropy(): 256-bin histogram over all elements, log2."""
    p = np.bincount(a.ravel(), minlength=256) / a.size
    p = p[p > 0]
    return -(p * np.log2(p)).sum()


def _con_entropy(u8):
    e0 = _entropy_u8(u8)
    i = u8.astype(np.int16)
    return [_entropy_u8(np.clip(i + s, 0, 255).astype(np.uint8)) - e0 for s in (120, 90, 60, 30, -30, -60, -90, -120)]


def nr_uiqa_features(u8):
    """The 42 features of Feature_extraction.m for an RGB uint8 image."""
    x = u8.astype(np.float64)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    rg, by = r - g, 0.5 * (r + g) - b
    ce_gray, ce_by, ce_rg = _ce(x)
    mx = x.max(axis=2)
    c1 = np.mean(np.divide(mx - x.min(axis=2), mx, out=np.zeros_like(mx), where=mx > 0))  # HSV saturation
    c5 = 0.3 * (np.hypot(by.mean(), rg.mean()) + np.sqrt(by.var(ddof=1) + rg.var(ddof=1)))
    f = [c1, ce_rg.mean(), ce_by.mean(), ce_gray.mean(), c5]
    chans = [x[..., i].ravel() / 255 for i in range(3)]
    f += [np.sqrt((c ** 2).sum() / (2 * c.size)) for c in chans]  # fitdist(...,'rayleigh') MLE
    xs = np.arange(1, 101) / 100
    for c, s in zip(chans, (0.2343, 0.2874, 0.3081)):
        a = s ** 2
        y = xs / a * np.exp(-xs ** 2 / (2 * a))
        area = 0.5 * abs(np.dot(xs, np.roll(y, 1)) - np.dot(y, np.roll(xs, 1)))  # polyarea
        y = 0.01 * y / area
        # ponytail: MATLAB histogram(…,100) may pick "nice" bin edges; here 100 equal bins over [min, max]
        h = np.histogram(c, 100, range=(c.min(), c.max()))[0] / c.size
        v1, v2 = h / h.sum(), y / y.sum()
        eps = np.finfo(float).eps
        f.append(np.sum(v1 * np.log(eps + v1 / (v2 + eps))))
    f += _con_entropy(u8)
    m, n = u8.shape[:2]
    xb, yb = int(round(m / 8)) * 8, int(round(n / 8)) * 8
    rs = u8 if (xb, yb) == (m, n) else cv2.resize(u8, (yb, xb), interpolation=cv2.INTER_CUBIC)
    f += list(np.mean([_con_entropy(rs[i * xb // 8:(i + 1) * xb // 8, j * yb // 8:(j + 1) * yb // 8])
                       for i in range(8) for j in range(8)], axis=0))
    lab = rgb2lab(u8)  # MATLAB rgb2lab: sRGB, D65
    mom = []
    for c in (lab[..., k].ravel() for k in range(3)):
        mu = c.mean()
        # (m3)^(1/3) of a negative number is complex in MATLAB; abs() later keeps |m3|^(1/3)
        mom += [mu, np.sqrt(np.mean((c - mu) ** 2)), abs(np.mean((c - mu) ** 3)) ** (1 / 3),
                np.mean((c - mu) ** 4) ** 0.25 - 3]
    f += list(np.abs(mom))
    import pywt
    coeffs = pywt.wavedec2(_rgb2gray_u8(u8), "haar", mode="symmetric", level=3)  # [cA3, lvl3, lvl2, lvl1]
    for cH, cV, cD in (coeffs[3], coeffs[2], coeffs[1]):
        e = [np.log10(1 + np.mean(c ** 2)) for c in (cH, cV, cD)]
        f.append(0.2 * (e[0] + e[1]) / 2 + 0.8 * e[2])
    # the released *_feature_mat.mat files (and so the SVR) hold [colour, Lab moments, Edi, nEdi, visibility],
    # not Contrast_feat.m's [Edi nEdi Moment]; in that order all 42 features match UID_feature_mat (r >= 0.998)
    return np.array(f, dtype=np.float64)[np.r_[0:11, 27:39, 11:27, 39:42]]


@lru_cache(1)
def _nr_uiqa_svr():
    from scipy.io import loadmat
    m = loadmat(REPOS / "NR-UIQA/svrmodel.mat")["svrmodel"][0, 0]
    return m["SVs"].toarray() if hasattr(m["SVs"], "toarray") else m["SVs"], m["sv_coef"].ravel(), \
        float(m["rho"].ravel()[0]), float(m["Parameters"].ravel()[3])


def nr_uiqa_predict(feats_scaled):
    """libsvm epsilon-SVR, RBF kernel: sum_i coef_i * exp(-gamma |x - sv_i|^2) - rho."""
    sv, coef, rho, gamma = _nr_uiqa_svr()
    d2 = ((feats_scaled[:, None, :] - sv[None]) ** 2).sum(-1)
    return np.exp(-gamma * d2) @ coef - rho


def nr_uiqa(p):
    """The released SVR was trained on SAUD after mapminmax(-1, 1) per feature; the authors saved only the scaled
    SAUD matrix, so the min/max come from our port's SAUD features (mos/nr_uiqa_saud_minmax.npy). In-sample on SAUD."""
    lo, hi = np.load(Path(__file__).with_name("nr_uiqa_saud_minmax.npy"))
    return float(nr_uiqa_predict((2 * (nr_uiqa_features(_rgb(p)) - lo) / (hi - lo) - 1)[None])[0])


# ---------------------------------------------------------------- Twice-Mixing (Fu et al., SPIC 2022)
@lru_cache(1)
def _twicemix_model():
    from torch import nn
    from torchvision.models import vgg16
    m = vgg16(weights=None)  # their vgg16.py: torchvision VGG16 with a new 1-output head
    m.classifier = nn.Sequential(nn.Linear(25088, 4096), nn.ReLU(), nn.Dropout(0.5), nn.Linear(4096, 4096),
                                 nn.ReLU(), nn.Dropout(0.5), nn.Linear(4096, 1))
    m.load_state_dict(torch.load(Path(__file__).resolve().parents[1] / "uwdata/twicemix/Twice-Mixing-model.pth", map_location="cpu",
                                 weights_only=True))
    return m.eval().to("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def twicemix(p):
    """test.py: full-resolution ToTensor input, no normalisation; ranking-trained, higher = better."""
    m = _twicemix_model()
    x = torch.from_numpy(_rgb(p)).permute(2, 0, 1)[None].float().div(255).to(next(m.parameters()).device)
    return m(x).item()


# ---------------------------------------------------------------- NUIQ (Jiang et al., TCSVT 2022, with SAUD)
def _aggd(v):
    """estimateaggdparam.m + the mean parameter, NaN semantics as in MATLAB (NaN rhat -> alpha = gam(1))."""
    from scipy.special import gamma as G
    gam = np.arange(0.2, 10.0005, 0.001)
    r_gam = G(2 / gam) ** 2 / (G(1 / gam) * G(3 / gam))
    with np.errstate(all="ignore"):
        ls, rs = np.sqrt(np.mean(v[v < 0] ** 2)), np.sqrt(np.mean(v[v > 0] ** 2))
        gh = ls / rs
        rhat = np.mean(np.abs(v)) ** 2 / np.mean(v ** 2)
        rn = rhat * (gh ** 3 + 1) * (gh + 1) / (gh ** 2 + 1) ** 2
        d = (r_gam - rn) ** 2
        a = gam[0] if np.all(np.isnan(d)) else gam[np.nanargmin(d)]
        mean = (rs - ls) * (G(2 / a) / G(1 / a)) * np.sqrt(G(1 / a)) / np.sqrt(G(3 / a))
    return [a, mean, ls ** 2, rs ** 2]


def _shift_diff(m):
    """m - circshift(m, [0 1]), border trimmed, as a vector."""
    return (m - np.roll(m, 1, axis=1))[1:-1, 1:-1].ravel()


def _mscn(x, c):
    from scipy.ndimage import correlate
    k = cv2.getGaussianKernel(7, 7 / 6)
    w = k @ k.T  # fspecial('gaussian', 7, 7/6), normalised
    mu = correlate(x, w, mode="constant")  # filter2(..., 'same') zero-pads
    sd = np.sqrt(np.abs(correlate(x * x, w, mode="constant") - mu * mu))
    return (x - mu) / (sd + c)


_RIU2 = np.array([bin(c).count("1") if bin(c ^ (((c << 1) | (c >> 7)) & 255)).count("1") <= 2 else 9
                  for c in range(256)])  # getmapping(8, 'riu2')


def _lbp_riu2(x):
    """lbp.m (Heikkila & Ahonen v0.3.2) with R=1, P=8, riu2 mapping, 'nh': strict N > C (skimage uses >=),
    bilinear diagonals, border pixels dropped. Returns the 10-bin normalised histogram."""
    d = x.astype(np.float64)
    h, w = d.shape[0] - 2, d.shape[1] - 2
    c = d[1:-1, 1:-1]
    code = np.zeros(c.shape, np.int64)
    for i in range(8):
        y, xx = 1 - np.sin(i * np.pi / 4), 1 + np.cos(i * np.pi / 4)
        ry, rx = int(np.rint(y)), int(np.rint(xx))
        if abs(xx - rx) < 1e-6 and abs(y - ry) < 1e-6:
            n = d[ry:ry + h, rx:rx + w]
        else:
            fy, fx, cy, cx = int(np.floor(y)), int(np.floor(xx)), int(np.ceil(y)), int(np.ceil(xx))
            ty, tx = y - fy, xx - fx
            n = ((1 - tx) * (1 - ty) * d[fy:fy + h, fx:fx + w] + tx * (1 - ty) * d[fy:fy + h, cx:cx + w]
                 + (1 - tx) * ty * d[cy:cy + h, fx:fx + w] + tx * ty * d[cy:cy + h, cx:cx + w])
        code += (n > c).astype(np.int64) << i
    return list(np.bincount(_RIU2[code].ravel(), minlength=10) / code.size)


def nuiq_features(u8):
    """The 59 features of NUIQ_feature_extraction.m for an RGB uint8 image."""
    x = u8.astype(np.float64)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    o1, o2 = (r - g) / np.sqrt(2), (r + g - 2 * b) / np.sqrt(6)
    lab = rgb2lab(u8)
    mom = []
    for c in (lab[..., k].ravel() for k in range(3)):
        mu = c.mean()
        mom += [mu, np.sqrt(np.mean((c - mu) ** 2)), abs(np.mean((c - mu) ** 3)) ** (1 / 3)]
    c1 = (0.01 * 255) ** 2
    f = _aggd(_mscn(np.abs(o1 - o2), c1).ravel())  # mapminmax on a column vector is the identity (constant rows)
    with np.errstate(all="ignore"):
        f += _aggd(_shift_diff(np.hypot(o1, o2)))
        f += _aggd(_shift_diff(np.arctan(o1 / o2)))
        (o1y, o1x), (o2y, o2x) = np.gradient(o1), np.gradient(o2)
        f += _aggd(_shift_diff(np.arctan(o1x / o2x)))
        f += _aggd(_shift_diff(np.arctan(o1y / o2y)))
    f += list(np.abs(mom))
    gray = _rgb2gray_u8(u8)
    f += _lbp_riu2(_mscn(gray, c1))
    s = 0.5
    xx, yy = np.meshgrid([-1, 0, 1], [-1, 0, 1])
    gdx = -(xx / (2 * np.pi * s ** 4)) * np.exp(-(xx * xx + yy * yy) / (2 * s ** 2))
    gdy = -(yy / (2 * np.pi * s ** 4)) * np.exp(-(xx * xx + yy * yy) / (2 * s ** 2))
    dy = ndimage.convolve(gray, gdy, mode="constant")
    dx = ndimage.convolve(gray, gdx.T, mode="constant")  # sic: the authors filter with gdx', which equals gdy
    gm = np.hypot(dx, dy)
    gm = np.floor(np.nan_to_num((gm - gm.min()) / (gm.max() - gm.min()) * 255) + 0.5)  # uint8(): half away from 0
    f += _lbp_riu2(gm)
    # The released code takes the GO (orientation) LBP here, but the authors' training features (train_feats.mat,
    # train.dat) repeat the GM histogram in these 10 slots, so the model expects that: SAUD within-scene SRCC
    # 0.741 this way vs 0.600 with the GO histogram (0.760 with the authors' own features). Their MSCN-LBP
    # block (slots 29-38) could not be reproduced from the released code; the other 49 features match.
    f += f[-10:]
    return np.array(f, dtype=np.float64)


def _nuiq_model():
    """SVM-rank linear model file: last line is '1 1:w1 2:w2 ... #'."""
    w = np.zeros(59)
    sv = (Path(__file__).resolve().parents[1] / "uwdata/nuiq/model").read_text().strip().splitlines()[-1]
    for tok in sv.split("#")[0].split()[1:]:
        k, v = tok.split(":")
        w[int(k) - 1] = float(v)
    return w


def nuiq_scores(feats, scenes):
    """SVM-rank linear model on features min-max scaled to [0, 1] within each scene, as demo.m does for one
    raw image's enhancements (mapminmax leaves constant features unchanged). Only within-scene comparisons
    are meaningful ("Do not test on single image")."""
    w = _nuiq_model()
    out = np.empty(len(feats))
    scenes = np.asarray(scenes)
    for s in np.unique(scenes):
        i = scenes == s
        f = feats[i]
        lo, hi = np.nanmin(f, 0), np.nanmax(f, 0)
        rng = np.where(hi > lo, hi - lo, 1)
        sc = np.where(hi > lo, (f - lo) / rng, f)
        out[i] = np.nan_to_num(sc) @ w  # ponytail: NaN features (degenerate images) count as 0
    return out


# ---------------------------------------------------------------- QUIQ (GPR on 32 hand-crafted features)
QUIQ = REPOS / "QUIQ-Underwater-Image-Quality-Assessment"
_PKL_ALLOW = {
    ("joblib.numpy_pickle", "NumpyArrayWrapper"), ("numpy", "ndarray"), ("numpy", "dtype"), ("_codecs", "encode"),
    ("numpy.core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "scalar"),
    ("numpy.random._pickle", "__randomstate_ctor"), ("numpy.random._pickle", "__bit_generator_ctor"),
    ("sklearn.preprocessing._data", "StandardScaler"), ("sklearn.gaussian_process._gpr", "GaussianProcessRegressor"),
    *(("sklearn.gaussian_process.kernels", k) for k in ("ConstantKernel", "Product", "RBF", "Sum")),
}


def _safe_joblib(path):
    """joblib.load with an exact whitelist of the globals these pickles were inspected to contain."""
    from joblib import numpy_pickle as npk

    class Safe(npk.NumpyUnpickler):
        def find_class(self, module, name):
            if (module, name) not in _PKL_ALLOW:
                raise RuntimeError(f"refused pickle global {module}.{name}")
            return super().find_class(module, name)

    with open(path, "rb") as f:
        return Safe(str(path), f, ensure_native_byte_order=True, mmap_mode=None).load()


def _quiq_mscn(img, scale):
    from skimage.filters import gaussian
    s = scale * 7 / 6
    mu = gaussian(img, sigma=s)
    return (img - mu) / (np.sqrt(np.maximum(gaussian(img ** 2, sigma=s) - mu ** 2, 1e-10)) + 1 / 255)


def _quiq_aggd(v):
    """Their estimate_aggd_params + fit_aggd (not the standard AGGD moment-matching fit)."""
    from scipy.special import gamma as G
    ls, rs = np.sqrt(np.mean(v[v < 0] ** 2)), np.sqrt(np.mean(v[v > 0] ** 2))
    gh = ls / rs if rs != 0 else 0
    ma = np.mean(np.abs(v))
    k = (ma ** 2 / np.mean(v ** 2)) / (G(2 / ma) / G(1 / ma)) ** 2
    return [np.sqrt(k * gh), np.sqrt(k / gh), ls ** 2, rs ** 2]


def _quiq_ggd(v):
    from scipy.optimize import minimize
    from scipy.special import gamma as G

    def loss(p):
        a, s = p
        b = s * np.sqrt(G(1 / a) / G(3 / a))
        return -np.sum(np.log(a / (2 * b * G(1 / a)) * np.exp(-(np.abs(v) / b) ** a) + 1e-10))
    return list(minimize(loss, [2.0, np.std(v)], bounds=[(0.1, 10), (1e-3, 1e3)]).x)


def _quiq_hp_var(gray_f32, cutoff):
    from scipy.fft import fft2, fftshift, ifft2, ifftshift
    r, c = gray_f32.shape
    yy, xx = np.ogrid[:r, :c]
    hp = (np.sqrt((yy - r // 2) ** 2 + (xx - c // 2) ** 2) > cutoff).astype(np.float32)
    return np.var(np.abs(ifft2(ifftshift(fftshift(fft2(gray_f32)) * hp))))


def quiq_features(u8):
    """single_bunch_test.ipynb feature extraction, quirks kept (BGR2GRAY applied to RGB data in the block-contrast
    term, uint8 truncation of float images)."""
    from skimage import color, exposure, img_as_float
    rgb = img_as_float(u8)
    bgr = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
    gray = color.rgb2gray(rgb)
    adh = exposure.equalize_adapthist(gray, clip_limit=0.05)
    g8 = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    h, w = g8.shape
    dc = np.var([g8[y:y + 32, x:x + 32].std() for y in range(0, h - 31, 32) for x in range(0, w - 31, 32)])
    hist = cv2.calcHist([(gray * 255).astype(np.uint8)], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    ent = -np.sum(hist * np.log2(hist + 1e-10))
    gb = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    f = [dc, ent, _quiq_hp_var(np.float32(gb), 10), _quiq_hp_var(np.float32(gb), 40),
         np.var(cv2.Canny(cv2.GaussianBlur(gb, (0, 0), sigmaX=0.5), 150, 200))]
    for s in (1, 2, 3):
        f += _quiq_aggd(_quiq_mscn(gray, s))
    for s in (1, 2, 3):
        f += _quiq_ggd(_quiq_mscn(adh, s))
    for ch in cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)):
        m = np.mean(ch)
        f += [m, np.var(ch), np.mean((ch - m) ** 3) / np.std(ch) ** 3]
    return np.array(f, dtype=np.float64)


@lru_cache(1)
def _quiq_model():
    return _safe_joblib(QUIQ / "scaler.pkl"), _safe_joblib(QUIQ / "learned_gpr_model.pkl")


def quiq(p):
    """QUIQ: StandardScaler + GaussianProcessRegressor fitted on 768 UID2021 images (80%), so in-sample on UID2021."""
    sc, gpr = _quiq_model()
    return float(gpr.predict(sc.transform(quiq_features(_rgb(p))[None]))[0])


# ---------------------------------------------------------------- CCF (Wang et al., Comput. Electr. Eng. 2018)
@lru_cache(1)
def _fade():
    """FADE (Choi et al. 2015) via cnyvfang/PyFADE, a MATLAB-aligned port (read before use; same fog-model .mat files
    as zhenglab/CCF). turbojpeg only decodes JPEG paths and we pass arrays, so a stub stands in for it."""
    import types
    sys.modules.setdefault("turbojpeg", types.ModuleType("turbojpeg"))
    sys.path.append(str(REPOS / "PyFADE/src"))  # appended: never shadows installed packages
    from pyfade import fade_array
    return fade_array


def ccf_parts(u8):
    x = u8.astype(np.float64)
    lg = np.log(x + 1e-5)
    rr, gg, bb = (lg[..., i] - lg[..., i].mean() for i in range(3))
    al, be = rr - gg, 0.5 * (rr + gg) - bb
    # sic: var(var(alpha)) = variance of alpha's column variances, as released
    col = 1000 * (np.sqrt(al.var(0, ddof=1).var(ddof=1) + be.var(0, ddof=1).var(ddof=1))
                  + 0.3 * np.hypot(al.mean(), be.mean())) / 85.59
    return col, _block_contrast(_rgb2gray_u8(u8)), _fade()(u8)


def ccf(p):
    """zhenglab/CCF CCF.m. Its mapminmax(x, 1, 10) on one scalar returns x unchanged (constant row), so the score is
    the paper's linear model c . [colourfulness, contrast, 10 - FADE]. Approximate: skimage Canny for MATLAB's edge."""
    col, con, fog = ccf_parts(_rgb(p))
    return 0.17593 * col + 0.61759 * con + 0.33988 * (10 - fog)


# ---------------------------------------------------------------- NMFC (dart-into/NMFC, colour-space multi-feature)
def _imresize_half(x):
    """MATLAB imresize(x, 0.5): bicubic (a = -0.5), antialiased (kernel widened by 1/scale), symmetric borders."""
    def cubic(t):
        a = np.abs(t)
        return ((1.5 * a ** 3 - 2.5 * a ** 2 + 1) * (a <= 1)
                + (-0.5 * a ** 3 + 2.5 * a ** 2 - 4 * a + 2) * ((a > 1) & (a <= 2)))

    for ax in (0, 1):
        n = x.shape[ax]
        u = np.arange(1, int(np.ceil(n * 0.5)) + 1) / 0.5 + 0.5 * (1 - 1 / 0.5)
        left = np.floor(u - 4)  # kernel width 4 / scale = 8
        ind = left[:, None] + np.arange(10)
        w = 0.5 * cubic(0.5 * (u[:, None] - ind))
        w /= w.sum(1, keepdims=True)
        ind = np.r_[np.arange(n), np.arange(n - 1, -1, -1)][np.mod(ind - 1, 2 * n).astype(int)]
        x = np.moveaxis(np.einsum("ok,okj->oj", w, np.moveaxis(x, ax, 0)[ind].reshape(len(u), 10, -1))
                        .reshape((len(u),) + np.moveaxis(x, ax, 0).shape[1:]), 0, ax)
    return x


def _clbp_s(x, r=0.5):
    """clbp.m CLBP_S with P = 8, riu2 mapping, 'nh': N >= C (unlike lbp.m), bilinear neighbours, 10-bin histogram."""
    oy = -np.floor(min(-r * np.sin(i * np.pi / 4) for i in range(8)))  # origin offset (0-based)
    by = int(np.ceil(max(r * np.sin(i * np.pi / 4) for i in range(8))) + oy) + 1
    h, w = x.shape[0] - by + 1, x.shape[1] - by + 1
    o = int(oy)
    c = x[o:o + h, o:o + w]
    code = np.zeros(c.shape, np.int64)
    for i in range(8):
        y, xx = -r * np.sin(i * np.pi / 4) + oy, r * np.cos(i * np.pi / 4) + oy
        ry, rx = int(np.rint(y)), int(np.rint(xx))
        if abs(xx - rx) < 1e-6 and abs(y - ry) < 1e-6:
            n = x[ry:ry + h, rx:rx + w]
        else:
            fy, fx, cy, cx = int(np.floor(y)), int(np.floor(xx)), int(np.ceil(y)), int(np.ceil(xx))
            ty, tx = y - fy, xx - fx
            n = ((1 - tx) * (1 - ty) * x[fy:fy + h, fx:fx + w] + tx * (1 - ty) * x[fy:fy + h, cx:cx + w]
                 + (1 - tx) * ty * x[cy:cy + h, fx:fx + w] + tx * ty * x[cy:cy + h, cx:cx + w])
        code += (n >= c).astype(np.int64) << i
    return list(np.bincount(_RIU2[code].ravel(), minlength=10) / code.size)


def _aggd6(v):
    """[alpha, mean, l^2, r^2] + skewness, kurtosis (MATLAB defaults: biased, non-excess)."""
    from scipy.stats import kurtosis, skew
    return _aggd(v) + [skew(v), kurtosis(v, fisher=False)]


def nmfc_features(u8):
    """The 84 features of NMFC_FeatureCompute.m for an RGB uint8 image."""
    lab = rgb2lab(u8)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    f = _aggd6(_mscn(np.abs(a - b), (0.01 * 255) ** 2).ravel())  # mapminmax on a column vector: identity
    with np.errstate(all="ignore"):
        f += _aggd6(_shift_diff(np.hypot(a, b)))
        f += _aggd6(_shift_diff(np.arctan(a / (b + 1e-7))))
        (ay, ax_), (by, bx) = np.gradient(a), np.gradient(b)
        f += _aggd6(_shift_diff(np.arctan(ax_ / (by + 1e-7))))  # sic: ima_x over imb_y
        f += _aggd6(_shift_diff(np.arctan(ay / (by + 1e-7))))
    mom = []
    for c in (L, a, b):
        mu = c.mean()
        mom += [mu, np.sqrt(np.mean((c - mu) ** 2)), abs(np.mean((c - mu) ** 3)) ** (1 / 3),
                np.mean((c - mu) ** 4) ** 0.25]  # (m3)^(1/3) is complex in MATLAB for m3 < 0; abs() keeps |.|
    f += list(np.abs(mom))
    centres = np.linspace(0, 100, 10)  # hist(L, 0:100/9:100): edges at midpoints, outer bins open
    edges = np.r_[-np.inf, (centres[:-1] + centres[1:]) / 2, np.inf]
    f += list(np.histogram(L.ravel(), edges)[0] / L.size)
    x = L
    for _ in range(2):
        m = _mscn(x, 1.0)
        f += _clbp_s(m) + _aggd6(m.ravel())
        x = _imresize_half(x)
    return np.array(f, dtype=np.float64)


@lru_cache(1)
def _nmfc_model():
    """demo.m uses SVM.m's model for split 1; the trained model is not released, so it is rebuilt from the released
    SAUD_NMFC_Feature.mat / SAUD_MOS.mat / Train|TestLineCell.mat exactly as SVM.m does: mapminmax(0, 1) fitted on
    the train rows (constant features unchanged), libsvm eps-SVR RBF, (c, g) picked by SRCC on the split's test
    rows. Trained on SAUD, so in-sample there."""
    from scipy.io import loadmat
    from scipy.stats import spearmanr
    from sklearn.svm import SVR
    d = REPOS / "NMFC"
    X = loadmat(d / "SAUD_NMFC_Feature.mat")["SAUD_NMFC_Feature"]
    y = loadmat(d / "SAUD_MOS.mat")["SAUD_MOS"].ravel().astype(np.float64)
    tr, te = (loadmat(d / f"{k}LineCell.mat")[f"{k}LineCell"][0, 0].ravel().astype(int) - 1 for k in ("Train", "Test"))
    lo, hi = X[tr].min(0), X[tr].max(0)
    rng = np.where(hi > lo, hi - lo, 1)

    def scale(a):
        return np.where(hi > lo, (a - lo) / rng, a)

    best = (-np.inf, None)
    for g in np.arange(-5, 0.01, 0.5):  # meshgrid(5:0.5:10, -5:0.5:0): rows g, columns c; strict > keeps first
        for c in np.arange(5, 10.01, 0.5):
            m = SVR(C=2 ** c, gamma=2 ** g, epsilon=0.1, tol=1e-3).fit(scale(X[tr]), y[tr])
            r = spearmanr(m.predict(scale(X[te])), y[te])[0]
            if r > best[0]:
                best = (r, (c, g))
    c, g = best[1]
    return SVR(C=2 ** c, gamma=2 ** g, epsilon=0.1, tol=1e-3).fit(scale(X[tr]), y[tr]), scale, best


def nmfc(p):
    m, scale, _ = _nmfc_model()
    return float(m.predict(scale(nmfc_features(_rgb(p))[None]))[0])


EXTRA = {"fdum": fdum, "uiqavsi": uiqavsi, "twicemix": twicemix, "nr_uiqa": nr_uiqa, "quiq": quiq, "ccf": ccf,
         "nmfc": nmfc}


def selfcheck():
    """FDUM port vs the MATLAB values the authors saved for their demo image b.png."""
    d = REPOS / "_fdum/x/Code-for-FDUM-method/Code/Test"
    c, k, s = fdum_parts(_rgb(d / "b.png"))
    print(f"FDUM b.png  colour {c:.4f} (MATLAB 2.2292)  contrast {k:.4f} (0.0831)  sharp {s:.4f} (7.3302)")
    print("FDUM a, a_enhance, b:", [round(fdum(d / f), 4) for f in ("a.png", "a_enhance.png", "b.png")],
          "(MATLAB 0.1243 0.3031 0.9069)")
    print("UIQAVSI demo images:", [round(uiqavsi(REPOS / "UIQAVSI/data" / f), 3)
                                   for f in ("2_Water-net.png", "2_UDCP.png", "10_RGHS.png")])


if __name__ == "__main__":
    selfcheck()
