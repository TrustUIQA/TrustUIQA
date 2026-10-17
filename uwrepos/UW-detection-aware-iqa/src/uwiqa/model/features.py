"""Frozen backbones for the detection-aware predictor (O4).

Every backbone sees the RAW image only: the predictor must decide whether to
enhance before any enhancement is run. Images are squashed to 224x224
(no centre crop) so objects near the borders still count.

Set UWIQA_RANDOM_BACKBONE=1 to build backbones without downloading weights
(tests / offline smoke runs only).
"""
from __future__ import annotations

import os

import cv2
import numpy as np
import torch

BACKBONES = {
    "clip_b32": ("vit_base_patch32_clip_224.openai", {}),
    "dinov2_s14": ("vit_small_patch14_dinov2.lvd142m", {"img_size": 224}),
    "resnet18": ("resnet18.a1_in1k", {}),
}


class Backbone:
    def __init__(self, name: str, device="cuda", size: int = 224):
        import timm
        arch, kw = BACKBONES[name]
        pretrained = os.environ.get("UWIQA_RANDOM_BACKBONE") != "1"
        self.model = timm.create_model(arch, pretrained=pretrained, num_classes=0, **kw).eval().to(device)
        cfg = self.model.pretrained_cfg
        self.mean = torch.tensor(cfg["mean"], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(cfg["std"], device=device).view(1, 3, 1, 1)
        self.device, self.size = device, size
        self.dim = self.model.num_features

    @torch.no_grad()
    def __call__(self, imgs: list[np.ndarray]) -> np.ndarray:
        x = np.stack([cv2.resize(im, (self.size, self.size), interpolation=cv2.INTER_AREA) for im in imgs])
        t = torch.from_numpy(x).to(self.device).permute(0, 3, 1, 2).float().div(255)
        return self.model((t - self.mean) / self.std).float().cpu().numpy()
