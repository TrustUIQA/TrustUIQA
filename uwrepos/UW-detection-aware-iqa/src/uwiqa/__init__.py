"""Detection-aware no-reference quality assessment for enhanced underwater images.

Paths are env-configurable so the same code runs locally and on Kaggle, where
inputs are read-only (/kaggle/input) and outputs go to /kaggle/working.
"""
import os
from pathlib import Path

__version__ = "0.1.0"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("DATA_ROOT", PROJECT_ROOT / "data"))
RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", PROJECT_ROOT / "results"))
MANIFEST_ROOT = Path(os.environ.get("MANIFEST_ROOT", RESULTS_ROOT / "manifests"))
