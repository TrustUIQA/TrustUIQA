from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image

from uwiqa import PROJECT_ROOT
from uwiqa.data import from_table


@pytest.fixture
def fake_uid2021(tmp_path):
    """Mimic UID2021 naming: SI_<subset>_<k> raw, ERI_<subset>_<k>_<METHOD> enhanced."""
    names = ["SI_B_1", "ERI_B_1_HP", "ERI_B_1_GDCP", "SI_BG_2", "ERI_BG_2_HP"]
    (tmp_path / "imgs" / "sub").mkdir(parents=True)
    for n in names:
        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(tmp_path / "imgs" / "sub" / f"{n}.png")
    pd.DataFrame({"name": names, "MOS": [2.1, 3.5, 3.0, 1.9, 2.8]}).to_csv(tmp_path / "mos.csv", index=False)
    return tmp_path


def test_uid2021_config_regexes(fake_uid2021):
    cfg = yaml.safe_load(open(PROJECT_ROOT / "configs" / "datasets.yaml"))["uid2021"]
    df = from_table(fake_uid2021, "mos.csv", "name", "MOS", image_dir="auto",
                    scene_regex=cfg["scene_regex"], method_regex=cfg["method_regex"],
                    method_default=cfg["method_default"])
    assert df["scene"].tolist() == ["B_1", "B_1", "B_1", "BG_2", "BG_2"]
    assert df["method"].tolist() == ["raw", "HP", "GDCP", "raw", "HP"]
    assert df["image"].iloc[0] == "imgs/sub/SI_B_1.png"


@pytest.mark.parametrize("layout", ["split_first", "images_first"])
def test_build_yolo_layouts(tmp_path, layout):
    from uwiqa.data import build_yolo
    for split in ("train", "valid", "test"):
        rel = ("{}", "images") if layout == "split_first" else ("images", "{}")
        img_dir = tmp_path / rel[0].format(split) / rel[1].format(split)
        lab_dir = tmp_path / Path(*[s.replace("images", "labels") for s in img_dir.relative_to(tmp_path).parts])
        img_dir.mkdir(parents=True)
        lab_dir.mkdir(parents=True)
        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(img_dir / f"{split}_1.jpg")
        (lab_dir / f"{split}_1.txt").write_text("3 0.5 0.5 0.1 0.1\n0 0.2 0.2 0.1 0.1\n3 0.7 0.7 0.1 0.1\n")
    df = build_yolo(tmp_path).set_index("split")
    assert sorted(df.index) == ["test", "train", "val"]
    assert (df["n_objects"] == 3).all() and (df["classes"] == "0 3").all()


def test_missing_image_is_reported(fake_uid2021):
    pd.DataFrame({"name": ["SI_B_1", "NOPE"], "MOS": [1, 2]}).to_csv(fake_uid2021 / "bad.csv", index=False)
    with pytest.raises(FileNotFoundError, match="NOPE"):
        from_table(fake_uid2021, "bad.csv", "name", "MOS")
