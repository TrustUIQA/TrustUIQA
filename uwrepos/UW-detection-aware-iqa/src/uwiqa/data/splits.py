"""Leakage-safe splits: whole duplicate groups go to exactly one split."""
from __future__ import annotations

import numpy as np
import pandas as pd

# Detector never sees pred_*; the predictor trains on pred_train, is tuned on
# pred_val and reported on pred_test. Utility labels are built on pred_*.
RUOD_FRACTIONS = {"det_train": 0.45, "det_val": 0.05,
                  "pred_train": 0.30, "pred_val": 0.10, "pred_test": 0.10}


def group_split(groups: pd.Series, fractions: dict[str, float], seed: int = 0) -> pd.Series:
    """Assign each item a split so that no group spans two splits.

    Largest groups are placed first (random order within equal sizes), each
    into the split furthest below its target, so sizes track the fractions
    closely even when some groups are big. Returns a Series aligned with
    ``groups``.
    """
    if not np.isclose(sum(fractions.values()), 1.0):
        raise ValueError("fractions must sum to 1")
    sizes = groups.value_counts().to_dict()
    order = np.array(list(sizes), dtype=object)  # a copy: shuffling an Index view corrupts it
    np.random.default_rng(seed).shuffle(order)
    order = sorted(order, key=lambda g: -sizes[g])  # stable: shuffle breaks ties
    names = list(fractions)
    target = np.array([fractions[n] * len(groups) for n in names])
    filled = np.zeros(len(names))
    assign = {}
    for g in order:
        k = int(np.argmax(target - filled))
        assign[g] = names[k]
        filled[k] += sizes[g]
    return groups.map(assign)
