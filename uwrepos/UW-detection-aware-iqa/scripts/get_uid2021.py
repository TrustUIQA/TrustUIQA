"""Download and unpack UID2021 from the authors' Google Drive link.

Licence: non-commercial research use; cite Hou et al., ACM TOMM 2023.
Skips if UID2021_ROOT already exists (e.g. attached as a Kaggle Dataset).

    python scripts/get_uid2021.py
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

URL = "https://drive.google.com/file/d/1cD8a_IsQHkQ_q-mDV6PvRql_gTZPRnk5/view"   # current (77-observer MOS)
URL_2022 = "https://drive.google.com/file/d/1oiTPSa6rIrsfTISaL9aach-MD3Fet5Al/view"  # README link until Dec 2022
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def fetch(url: str, root: Path) -> int:
    """Download a Drive archive into root and unpack it; returns the number of images."""
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=True)
    root.mkdir(parents=True, exist_ok=True)
    archive = root.parent / f"{root.name}_archive"
    # gdown < 6 needs --fuzzy for /file/d/<id>/view links; newer versions removed the flag (fuzzy by default)
    if subprocess.run(["gdown", "-q", "--fuzzy", url, "-O", str(archive)]).returncode != 0:
        subprocess.run(["gdown", "-q", url, "-O", str(archive)], check=True)
    kind = subprocess.run(["file", "-b", str(archive)], capture_output=True, text=True).stdout
    print("archive type:", kind.strip())
    if "HTML" in kind:  # Drive returned a permission / quota page instead of the file
        os.remove(archive)
        raise RuntimeError("Google Drive returned a web page, not the archive (file not public or quota hit)")
    if "Zip" in kind:
        shutil.unpack_archive(str(archive), str(root), "zip")
    else:  # RAR / 7z
        if shutil.which("7z") is None:  # not always present on Kaggle images
            subprocess.run("apt-get -qq update && apt-get -qq install -y p7zip-full p7zip-rar > /dev/null",
                           shell=True, check=False)
        if shutil.which("7z") is None:
            sys.exit("Could not get 7-Zip to unpack the archive. Download UID2021 manually, upload it as a "
                     "Kaggle Dataset, attach it, and set UID2021_ROOT (see notebook 01, cell 2).")
        subprocess.run(["7z", "x", "-y", f"-o{root}", str(archive)], check=True, stdout=subprocess.DEVNULL)
    os.remove(archive)
    return sum(1 for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS)


if __name__ == "__main__":
    from uwiqa.data import dataset_root

    root = dataset_root("uid2021")
    if root.exists() and any(root.iterdir()):
        print(f"UID2021 already present at {root}")
        sys.exit(0)
    print(f"UID2021 unpacked to {root}: {fetch(URL, root)} images (expected 960)")
