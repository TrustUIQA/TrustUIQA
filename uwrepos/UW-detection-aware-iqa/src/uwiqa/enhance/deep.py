"""Deep UIE methods with public PyTorch weights (inference only, GPU).

Network code is imported from each method's official repository (cloned by
scripts/get_enhancers.py into THIRD_PARTY), so weights always match the
architecture they were trained with.

Protocol choice (state it in the paper): fully-convolutional generators are
run at the working resolution (long side 1280), reflect-padded to a multiple
of their stride, rather than at their 256x256 training size. Down/up-sizing
to 256 would blur away the small objects the detector needs.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

THIRD_PARTY = Path(os.environ.get("THIRD_PARTY", "/kaggle/working/third_party"))


def wrap_tanh_generator(net: torch.nn.Module, device, multiple: int = 32,
                        fp16: bool = True) -> Callable[[np.ndarray], np.ndarray]:
    """uint8 RGB -> [-1,1] -> net -> tanh output -> uint8 RGB, with pad/crop to ``multiple``."""
    net = net.to(device).eval()
    use_half = fp16 and str(device).startswith("cuda")
    if use_half:
        net = net.half()

    @torch.no_grad()
    def run(img: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.array(img, copy=True)).to(device)  # copy: input may be read-only
        x = x.permute(2, 0, 1)[None].float()
        x = x / 127.5 - 1
        H, W = x.shape[-2:]
        ph, pw = (-H) % multiple, (-W) % multiple
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        y = net(x.half() if use_half else x).float()[..., :H, :W]
        y = ((y.clamp(-1, 1) + 1) * 127.5).round().byte()
        return y[0].permute(1, 2, 0).cpu().numpy()

    return run


def _load_state(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:  # older checkpoints pickled with extra objects
        return torch.load(path, map_location=device, weights_only=False)


def funiegan(device="cuda") -> Callable[[np.ndarray], np.ndarray]:
    """FUnIE-GAN (Islam, Xia & Sattar, IEEE RA-L 2020), official PyTorch generator."""
    repo = THIRD_PARTY / "FUnIE-GAN" / "PyTorch"
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from nets.funiegan import GeneratorFunieGAN  # noqa: E402  (official code)
    net = GeneratorFunieGAN()
    state = _load_state(repo / "models" / "funie_generator.pth", device)
    if isinstance(state, torch.nn.Module):
        state = state.state_dict()
    net.load_state_dict(state)
    return wrap_tanh_generator(net, device)


DEEP = {"funiegan": funiegan}
