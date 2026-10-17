"""Fetch official code + pretrained weights for the deep enhancers into THIRD_PARTY.

    python scripts/get_enhancers.py            # FUnIE-GAN (weights ship inside its repo)
"""
import subprocess

from uwiqa.enhance.deep import THIRD_PARTY

REPOS = {"FUnIE-GAN": "https://github.com/xahidbuffon/FUnIE-GAN.git"}

THIRD_PARTY.mkdir(parents=True, exist_ok=True)
for name, url in REPOS.items():
    dest = THIRD_PARTY / name
    if dest.exists():
        print(f"{name}: present")
        continue
    subprocess.run(["git", "clone", "-q", "--depth", "1", url, str(dest)], check=True)
    print(f"{name}: cloned to {dest}")

w = THIRD_PARTY / "FUnIE-GAN" / "PyTorch" / "models" / "funie_generator.pth"
print(f"FUnIE-GAN weights: {w} ({w.stat().st_size / 1e6:.1f} MB)")
