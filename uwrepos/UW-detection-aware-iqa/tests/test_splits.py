import numpy as np
import pandas as pd
import pytest

from uwiqa.data.splits import RUOD_FRACTIONS, group_split


def test_groups_never_span_splits_and_sizes_match():
    rng = np.random.default_rng(0)
    sizes = rng.choice([1, 1, 1, 2, 5, 30], size=3000)
    groups = pd.Series(np.repeat(np.arange(len(sizes)), sizes))
    split = group_split(groups, RUOD_FRACTIONS)
    assert pd.DataFrame({"g": groups, "s": split}).groupby("g")["s"].nunique().max() == 1
    frac = split.value_counts(normalize=True)
    for name, f in RUOD_FRACTIONS.items():
        assert frac[name] == pytest.approx(f, abs=0.02)


def test_deterministic_and_seed_dependent():
    groups = pd.Series(np.arange(500) // 2)
    a, b = group_split(groups, RUOD_FRACTIONS, 1), group_split(groups, RUOD_FRACTIONS, 1)
    assert a.equals(b)
    assert not a.equals(group_split(groups, RUOD_FRACTIONS, 2))


def test_bad_fractions():
    with pytest.raises(ValueError):
        group_split(pd.Series([1, 2]), {"a": 0.5, "b": 0.6})
