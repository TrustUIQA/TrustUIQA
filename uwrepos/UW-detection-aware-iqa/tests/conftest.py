import numpy as np
import pytest
from skimage import data


def _underwaterize(img: np.ndarray, depth: float = 1.0) -> np.ndarray:
    """Simple image-formation model: J*t + B*(1-t), red attenuated most."""
    x = img.astype(np.float64) / 255
    beta = np.array([1.2, 0.35, 0.25]) * depth
    t = np.exp(-beta)
    B = np.array([0.05, 0.45, 0.55])
    return np.clip((x * t + B * (1 - t)) * 255, 0, 255).astype(np.uint8)


@pytest.fixture(scope="session")
def clean():
    return data.astronaut()  # 512x512 RGB uint8, bundled with skimage


@pytest.fixture(scope="session")
def underwater(clean):
    return _underwaterize(clean, 1.5)
