"""Standard manifest format used by every downstream script.

A manifest is a CSV with columns:
    image   path relative to the dataset root
    mos     subjective score (float) or empty
    scene   id of the underlying raw image (groups enhanced variants;
            used for intra-scene SRCC and leakage-free splits)
    method  'raw', 'reference', or the enhancement algorithm name

Builders below turn each dataset's native layout into this format. For MOS
datasets whose release format we have not verified (UID2021, SAUD, UWIQA,
LUIQD) use ``from_table`` driven by configs/datasets.yaml — inspect the
download first and fill in the column names / regexes.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
COLUMNS = ["image", "mos", "scene", "method"]


def _images(d: Path):
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in IMG_EXTS)


def build_uieb(root: str | Path) -> pd.DataFrame:
    """UIEB: raw-890/, reference-890/, challenging-60/ (no per-image MOS)."""
    root = Path(root)
    rows = []
    for sub, method in [("raw-890", "raw"), ("reference-890", "reference"),
                        ("challenging-60", "raw_challenging")]:
        d = root / sub
        if not d.exists():
            continue
        for p in _images(d):
            rows.append({"image": p.relative_to(root).as_posix(), "mos": None,
                         "scene": p.stem, "method": method})
    if not rows:
        raise FileNotFoundError(f"no UIEB folders under {root}")
    return pd.DataFrame(rows, columns=COLUMNS)


def build_euvp_test(root: str | Path) -> pd.DataFrame:
    """EUVP paired test set: test_samples/Inp and test_samples/GTr."""
    root = Path(root)
    base = root / "test_samples"
    rows = []
    for sub, method in [("Inp", "raw"), ("GTr", "reference")]:
        for p in _images(base / sub):
            rows.append({"image": p.relative_to(root).as_posix(), "mos": None,
                         "scene": p.stem, "method": method})
    if not rows:
        raise FileNotFoundError(f"no EUVP test_samples under {root}")
    return pd.DataFrame(rows, columns=COLUMNS)


SPLIT_NAMES = {"train": "train", "valid": "val", "val": "val", "test": "test"}
DET_COLUMNS = ["image", "label", "split", "n_objects", "classes"]


def build_yolo(root: str | Path) -> pd.DataFrame:
    """Detection manifest for a YOLO-format dataset.

    Handles both layouts seen in the ODverse33 Kaggle copy:
    ``<split>/images/x.jpg`` (RUOD) and ``images/<split>/x.jpg`` (DUO).
    Labels are found by swapping the ``images`` path component for
    ``labels`` and the extension for ``.txt``. Columns: image, label, split,
    n_objects, classes (space-separated class ids present).
    """
    root = Path(root)
    rows = []
    for p in _images(root):
        parts = list(p.relative_to(root).parts)
        if "images" not in parts:
            continue
        split = next((SPLIT_NAMES[s] for s in parts[:-1] if s in SPLIT_NAMES), "unknown")
        lab_parts = ["labels" if s == "images" else s for s in parts[:-1]] + [p.stem + ".txt"]
        lab = Path(*lab_parts)
        cls = []
        if (root / lab).exists():
            cls = [ln.split()[0] for ln in (root / lab).read_text().splitlines() if ln.strip()]
        rows.append({"image": p.relative_to(root).as_posix(), "label": lab.as_posix(),
                     "split": split, "n_objects": len(cls),
                     "classes": " ".join(sorted(set(cls), key=int))})
    if not rows:
        raise FileNotFoundError(f"no images/ folders under {root}")
    return pd.DataFrame(rows, columns=DET_COLUMNS)


TABLE_EXTS = {".xlsx", ".xls", ".csv", ".txt", ".mat"}


def inspect_dataset(root: str | Path, max_rows: int = 5) -> None:
    """Print image counts per folder and the head of every score-like table.

    Run this once after downloading a MOS dataset to fill in datasets.yaml.
    """
    root = Path(root)
    imgs = _images(root)
    print(f"{len(imgs)} images under {root}")
    for d, n in pd.Series([p.parent.relative_to(root).as_posix() for p in imgs]).value_counts().items():
        print(f"  {n:6d}  {d}/   e.g. {next(p.name for p in imgs if p.parent.relative_to(root).as_posix() == d)}")
    for t in sorted(p for p in root.rglob("*") if p.suffix.lower() in TABLE_EXTS):
        print(f"\n--- table: {t.relative_to(root).as_posix()}")
        try:
            print(_read_table(t).head(max_rows).to_string())
        except Exception as e:
            print(f"  (could not parse: {e})")


def _read_table(tp: Path, sheet: str | int = 0) -> pd.DataFrame:
    if tp.suffix.lower() in {".xls", ".xlsx"}:
        return pd.read_excel(tp, sheet_name=sheet)
    if tp.suffix.lower() == ".txt":
        return pd.read_csv(tp, sep=None, engine="python")
    return pd.read_csv(tp)


def from_table(root: str | Path, table: str, image_col: str, mos_col: str,
               image_dir: str = "auto", scene_regex: str | None = None,
               method_regex: str | None = None, method_default: str = "unknown",
               sheet: str | int = 0, image_suffix: str = "") -> pd.DataFrame:
    """Generic MOS-table → manifest.

    ``image_dir="auto"`` locates each image anywhere under ``root`` by file
    stem (extension in the table optional). ``scene_regex`` /
    ``method_regex`` are applied to the file name and must contain one
    capture group. Missing images are reported, not silently dropped.
    """
    root = Path(root)
    df = _read_table(root / table, sheet)
    names = [f"{x}{image_suffix}" for x in df[image_col].astype(str).str.strip()]
    if image_dir == "auto":
        by_stem = {p.stem: p for p in _images(root)}
        paths = [by_stem.get(Path(n).stem) for n in names]
        missing = [n for n, p in zip(names, paths) if p is None]
        images = [p.relative_to(root).as_posix() if p else None for p in paths]
    else:
        images = [Path(image_dir, n).as_posix() for n in names]
        missing = [p for p in images if not (root / p).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images in {table} not found, e.g. {missing[:3]}")
    out = pd.DataFrame({"image": images, "mos": pd.to_numeric(df[mos_col], errors="coerce")})
    stems = out["image"].map(lambda s: Path(s).stem)
    out["scene"] = stems.map(lambda n: _extract(scene_regex, n, n))
    out["method"] = stems.map(lambda n: _extract(method_regex, n, method_default))
    return out[COLUMNS]


def _extract(regex, s, default):
    if not regex:
        return default
    m = re.search(regex, s)
    return m.group(1) if m else default


def load_config() -> dict:
    """configs/datasets.yaml, overlaid per dataset with configs/local.yaml.

    local.yaml is git-ignored: put machine/session-specific values there
    (e.g. UID2021's MOS column names) so `git pull` never conflicts.
    """
    import yaml
    from uwiqa import PROJECT_ROOT
    cfg = yaml.safe_load(open(PROJECT_ROOT / "configs" / "datasets.yaml"))
    local = PROJECT_ROOT / "configs" / "local.yaml"
    if local.exists():
        for name, over in (yaml.safe_load(open(local)) or {}).items():
            cfg.setdefault(name, {}).update(over or {})
    return cfg


def set_local_config(name: str, **values) -> None:
    """Persist overrides for one dataset into configs/local.yaml."""
    import yaml
    from uwiqa import PROJECT_ROOT
    local = PROJECT_ROOT / "configs" / "local.yaml"
    cur = (yaml.safe_load(open(local)) or {}) if local.exists() else {}
    cur.setdefault(name, {}).update(values)
    yaml.safe_dump(cur, open(local, "w"), sort_keys=False)


def dataset_root(name: str) -> Path:
    """$<NAME>_ROOT if set, else DATA_ROOT / <root from the config>."""
    import os

    from uwiqa import DATA_ROOT
    env = os.environ.get(f"{name.upper()}_ROOT")
    if env:
        return Path(env)
    return DATA_ROOT / load_config()[name]["root"]


def manifest_path(name: str) -> Path:
    from uwiqa import MANIFEST_ROOT
    return MANIFEST_ROOT / f"{name}.csv"


def load_manifest(path_or_name: str | Path) -> pd.DataFrame:
    """Load by dataset name ('uid2021') or by explicit CSV path."""
    path = Path(path_or_name)
    if path.suffix != ".csv":
        path = manifest_path(str(path_or_name))
    df = pd.read_csv(path, dtype={"classes": str})
    if "split" in df.columns:  # detection manifest
        return df
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"manifest {path} missing columns {missing}")
    df["method"] = df["method"].astype(str)
    df["scene"] = df["scene"].astype(str)
    return df
