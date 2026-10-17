import numpy as np
import pytest

from uwiqa.enhance import CLASSICAL, SWEEPS, monotonic_inflation
from uwiqa.metrics import uiqm


@pytest.mark.parametrize("name", list(SWEEPS))
def test_sweep_identity_at_zero(underwater, name):
    out = SWEEPS[name](underwater, 0.0)
    assert out.dtype == np.uint8 and out.shape == underwater.shape
    assert np.abs(out.astype(int) - underwater).max() <= 1  # float round-trip tolerance


@pytest.mark.parametrize("name", list(CLASSICAL))
def test_classical_shapes(underwater, name):
    out = CLASSICAL[name](underwater)
    assert out.dtype == np.uint8 and out.shape == underwater.shape
    assert not np.array_equal(out, underwater)


@pytest.mark.parametrize("name", ["fusion", "udcp", "gray_world"])
def test_colour_correction_reduces_cast(underwater, name):
    """Red-deficient input: red/green mean ratio should move toward 1."""
    def rg(x):
        m = x.reshape(-1, 3).mean(0)
        return m[0] / m[1]
    assert abs(1 - rg(CLASSICAL[name](underwater))) < abs(1 - rg(underwater))


def test_odd_sizes(underwater):
    img = underwater[:301, :417]
    for f in CLASSICAL.values():
        assert f(img).shape == img.shape


def test_monotonic_inflation_summary():
    r = monotonic_inflation([0, 1, 2, 3], [1.0, 2.0, 3.0, 4.0])
    assert r["srcc_vs_strength"] == pytest.approx(1.0) and r["frac_increasing"] == 1.0


def test_red_shift_inflates_uiqm_initially(underwater):
    """Sanity check of the O2 hypothesis on one synthetic image."""
    s0, s1 = uiqm(underwater), uiqm(SWEEPS["red_shift"](underwater, 0.5))
    assert s1 > s0
