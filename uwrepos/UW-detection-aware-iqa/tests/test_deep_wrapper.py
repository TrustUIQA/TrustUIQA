import numpy as np
import torch

from uwiqa.enhance.deep import wrap_tanh_generator


class Identity(torch.nn.Module):
    def forward(self, x):
        assert x.shape[-1] % 32 == 0 and x.shape[-2] % 32 == 0
        return x


class Halve(torch.nn.Module):
    """Stride-2 down + up, like a U-Net would; fails if input is not padded."""
    def forward(self, x):
        y = torch.nn.functional.avg_pool2d(x, 32)
        return torch.nn.functional.interpolate(y, scale_factor=32, mode="nearest") * 0 + x


def test_identity_roundtrip_with_padding():
    img = np.random.default_rng(0).integers(0, 256, (123, 257, 3), dtype=np.uint8)
    out = wrap_tanh_generator(Identity(), "cpu")(img)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert np.abs(out.astype(int) - img).max() <= 1


def test_odd_size_through_unet_like_net():
    img = np.zeros((101, 77, 3), np.uint8)
    assert wrap_tanh_generator(Halve(), "cpu")(img).shape == img.shape
