import numpy as np
from PIL import Image
from skimage import data

from uwiqa.data.dedup import connected_groups, dhash, hamming_pairs


def test_resized_and_recompressed_copy_matches(tmp_path, clean):
    Image.fromarray(clean).save(tmp_path / "a.png")
    Image.fromarray(clean).resize((300, 300)).save(tmp_path / "b.jpg", quality=70)
    Image.fromarray(data.coffee()).save(tmp_path / "c.png")
    H = np.stack([dhash(tmp_path / f) for f in ("a.png", "b.jpg", "c.png")])
    pairs = hamming_pairs(H, max_dist=20, chunk=2)
    assert [tuple(p[:2]) for p in pairs] == [(0, 1)]


def test_groups_are_transitive():
    pairs = np.array([[0, 1, 3], [1, 2, 4], [4, 5, 0]])
    g = connected_groups(6, pairs)
    assert g[0] == g[1] == g[2] and g[4] == g[5] and len(set(g)) == 3
