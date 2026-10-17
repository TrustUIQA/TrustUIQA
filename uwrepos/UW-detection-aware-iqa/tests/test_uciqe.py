import cv2
import numpy as np
import pytest

from uwiqa.enhance.sweeps import saturation
from uwiqa.metrics import uciqe


@pytest.mark.parametrize("variant", ["paper", "legacy_cv2"])
def test_range_and_components(underwater, variant):
    c = uciqe(underwater, variant, return_components=True)
    assert c["uciqe"] == pytest.approx(0.4680 * c["sigma_c"] + 0.2745 * c["con_l"] + 0.2576 * c["mu_s"])
    assert 0.0 < c["uciqe"] < 1.5


def test_paper_variant_bounded_terms(clean):
    c = uciqe(clean, "paper", return_components=True)
    assert 0 <= c["con_l"] <= 1 and 0 <= c["mu_s"] <= 1


def test_gray_has_zero_chroma(clean):
    gray = np.repeat(cv2.cvtColor(clean, cv2.COLOR_RGB2GRAY)[..., None], 3, axis=2)
    c = uciqe(gray, "paper", return_components=True)
    assert c["sigma_c"] < 1e-3 and c["mu_s"] < 1e-3


def test_saturation_inflates_uciqe(underwater):
    """The gameability the project is about: more saturation -> higher score."""
    scores = [uciqe(saturation(underwater, s)) for s in (0, 0.5, 1, 2)]
    assert scores == sorted(scores)


def test_rejects_unit_range_floats(underwater):
    with pytest.raises(ValueError):
        uciqe(underwater / 255.0)


def test_unknown_variant(underwater):
    with pytest.raises(ValueError):
        uciqe(underwater, "nope")
