"""Build $MANIFEST_ROOT/<name>.csv (default results/manifests/) for every dataset
in configs/datasets.yaml (+ configs/local.yaml overrides).

    python scripts/build_manifests.py            # all datasets that exist
    python scripts/build_manifests.py --only uid2021
"""
import argparse
import sys

from uwiqa import MANIFEST_ROOT
from uwiqa.data import build_euvp_test, build_uieb, build_yolo, dataset_root, from_table, load_config

BUILDERS = {"uieb": build_uieb, "euvp_test": build_euvp_test, "yolo": build_yolo}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()

    cfg = load_config()
    out_dir = MANIFEST_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = True
    for name, c in cfg.items():
        if args.only and name not in args.only:
            continue
        root = dataset_root(name)
        if not root.exists():
            print(f"[skip] {name}: {root} not found")
            continue
        try:
            if c["builder"] == "table":
                kw = {k: v for k, v in c.items() if k not in {"root", "builder"} and v is not None}
                if any(str(v).startswith("TODO") for v in kw.values()):
                    print(f"[skip] {name}: fill in the TODOs in datasets.yaml first "
                          f"(run: python -c \"from uwiqa.data import inspect_dataset; "
                          f"inspect_dataset(r'{root}')\")")
                    continue
                df = from_table(root, **kw)
            else:
                df = BUILDERS[c["builder"]](root)
        except Exception as e:  # report and continue with other datasets
            print(f"[fail] {name}: {e}")
            ok = False
            continue
        dest = out_dir / f"{name}.csv"
        df.to_csv(dest, index=False)
        if "split" in df.columns:
            print(f"[ok]   {name}: {len(df)} images, {int(df['n_objects'].sum())} objects, "
                  f"splits {df['split'].value_counts().to_dict()} -> {dest}")
        else:
            n_mos = df["mos"].notna().sum()
            print(f"[ok]   {name}: {len(df)} images ({n_mos} with MOS, "
                  f"{df['scene'].nunique()} scenes, {df['method'].nunique()} methods) -> {dest}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
