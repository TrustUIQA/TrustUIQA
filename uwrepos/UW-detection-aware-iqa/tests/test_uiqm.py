import math

import cv2
import numpy as np
import pytest

from uwiqa.metrics import uicm, uiconm, uiqm, uism
from uwiqa.metrics.uiqm import _eme


# --- loop-based ports of the reference implementation (Islam et al.) -------
def _eme_ref(x, window_size):
    k1 = x.shape[1] / window_size
    k2 = x.shape[0] / window_size
    w = 2.0 / (k1 * k2)
    x = x[: int(window_size * k2), : int(window_size * k1)]
    val = 0.0
    for l in range(int(k1)):
        for k in range(int(k2)):
            block = x[k * window_size: window_size * (k + 1), l * window_size: window_size * (l + 1)]
            mx, mn = np.max(block), np.min(block)
            if mn == 0.0 or mx == 0.0:
                continue
            val += math.log(mx / mn)
    return w * val


def _uiconm_ref(x, window_size):
    k1 = x.shape[1] / window_size
    k2 = x.shape[0] / window_size
    w = -1.0 / (k1 * k2)
    x = x[: int(window_size * k2), : int(window_size * k1)]
    val = 0.0
    for l in range(int(k1)):
        for k in range(int(k2)):
            block = x[k * window_size: window_size * (k + 1), l * window_size: window_size * (l + 1), :]
            mx, mn = np.max(block), np.min(block)
            top, bot = mx - mn, mx + mn
            if bot == 0.0 or top == 0.0:
                continue
            val += (top / bot) * math.log(top / bot)
    return w * val


@pytest.mark.parametrize("shape", [(100, 100), (123, 257), (512, 512)])
def test_eme_matches_reference_loop(shape):
    rng = np.random.default_rng(1)
    x = rng.integers(0, 256, shape).astype(np.float64)
    x[:15, :15] = 0  # exercise the zero-block branch
    assert _eme(x, 10) == pytest.approx(_eme_ref(x, 10), rel=1e-10)


def test_uiconm_matches_reference_loop(underwater):
    img = underwater[:203, :311].astype(np.float64)
    assert uiconm(img) == pytest.approx(_uiconm_ref(img, 10), rel=1e-10)


def test_rejects_unit_range_floats(underwater):
    with pytest.raises(ValueError):
        uiqm(underwater.astype(np.float32) / 255)


def test_rejects_grayscale():
    with pytest.raises(ValueError):
        uiqm(np.zeros((64, 64), np.uint8))


def test_published_range(clean, underwater):
    # Literature: UIQM of raw underwater images ~0.5-3.5, enhanced ~2.5-5.5.
    for img in (clean, underwater):
        assert 0.0 < uiqm(img) < 6.0


def test_degradation_lowers_uiqm(clean, underwater):
    assert uiqm(underwater) < uiqm(clean)


def test_gray_has_no_colourfulness_gain(clean):
    gray = np.repeat(cv2.cvtColor(clean, cv2.COLOR_RGB2GRAY)[..., None], 3, axis=2)
    assert uicm(gray) == pytest.approx(0.0, abs=1e-9)
    assert uicm(clean) > uicm(gray)


def test_blur_lowers_sharpness(underwater):
    blurred = cv2.GaussianBlur(underwater, (0, 0), 3)
    assert uism(blurred) < uism(underwater)


def test_components_combine(underwater):
    c = uiqm(underwater, return_components=True)
    assert c["uiqm"] == pytest.approx(0.0282 * c["uicm"] + 0.2953 * c["uism"] + 3.5753 * c["uiconm"])


def test_legacy_blue_weight_differs(underwater):
    assert uism(underwater, legacy_blue_weight=True) != uism(underwater)
